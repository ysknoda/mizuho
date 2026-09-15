"""Orchestration drivers: wire together space-group selection (Phase 0),
Wyckoff-assignment search (Phase 1), and numerical optimisation (Phase 2)
for the three CLI modes (single composition, composition pool, fully
automatic composition discovery).
"""

import sys
from typing import Dict, List

from .composition import _all_charges_zero, discover_compositions, parse_composition
from .config import _lattice_bounds_for_system, augment_coupling_constraints
from .data import WYCKOFF_DB
from .geometry import precompute_letter_feasibility, rank_by_feasibility_staged
from .io.poscar import write_poscar
from .io.report import print_summary
from .lattice import CRYSTAL_SYSTEM_OF_SG
from .optimize.cma import multistart_cma_es
from .optimize.driver import optimize_one_assignment
from .optimize.parallel import optimize_assignments_parallel
from .search import auto_search_spacegroups
from .search.dfs import wyckoff_tree_search
from .structure import assignments_to_sites


def build_sg_config(base_config: dict, sg: int, composition: str = None) -> dict:
    """
    Build a per-space-group working config: fixes `spacegroup`, derives
    lattice_bounds appropriate for that SG's crystal system, optionally
    overrides `composition` (used by the auto-composition-discovery
    driver, which tries many compositions per SG), and auto-injects
    anion-coupling constraints (config.augment_coupling_constraints) so
    cation-centred coordination/angle constraints are tied together
    through their shared anions by default.

    Centralised here so this sequence -- previously duplicated across
    cli.py::main and this module's two composition-driving functions --
    is defined exactly once.
    """
    sg_config = dict(base_config)
    sg_config["spacegroup"] = sg
    if composition is not None:
        sg_config["composition"] = composition
    sg_config["lattice_bounds"] = _lattice_bounds_for_system(
        base_config.get("lattice_bounds", {}), CRYSTAL_SYSTEM_OF_SG.get(sg, "triclinic"))
    return augment_coupling_constraints(sg_config)


def search_and_rank(sg_config: dict, max_cands: int, verbose: bool = False):
    """
    Run Phase 1's Wyckoff-assignment search (wyckoff_tree_search) to
    completion, then -- only when sg_config actually has
    coordination_constraints and/or angle_constraints to check --
    re-rank the results by geometric feasibility
    (geometry.rank_by_feasibility_staged) BEFORE truncating to
    max_cands.

    Why this matters: Phase 1's best-first search orders candidates by
    a cheap charge-balance/packing-fraction surrogate that has no
    notion of bond angles or coordination shape at all, so a
    chemically/geometrically implausible combination of Wyckoff sites
    (e.g. one whose target coordination number provably cannot be
    reached at any acceptable bond angle -- see
    geometry.local_shell's module docstring for a concrete example)
    can easily occupy one of the limited max_cands slots handed to
    Phase 2's expensive CMA-ES optimisation ahead of a plausible one,
    purely by discovery order. Before the search even runs, a small
    geometry.precompute_letter_feasibility table (built once per
    (space group, composition) pair, at the scale of a handful of
    Wyckoff letters -- see that function's docstring for what's cheap
    enough to precompute vs. what still needs rank_by_feasibility_staged's
    exact per-candidate check) is handed to wyckoff_tree_search so
    search.scoring.surrogate_score can bias the search's OWN discovery
    order towards geometrically-plausible letters directly, rather than
    relying entirely on re-ranking after the fact. The SAME table is
    also handed to rank_by_feasibility_staged, whose STAGE 1 uses it as
    a cheap per-site lookup instead of solving narrow constraint groups
    from scratch -- see that function's own docstring for why this
    matters far more than it looks: a composition whose narrow groups
    still have thousands of distinct multi-site sub-assignments (e.g.
    adamite Zn2(AsO4)(OH)) can spend HOURS re-solving them one at a
    time, while this table costs only as many solves as the space
    group has Wyckoff letters (adamite: 15), already paid for above.

    Unlike an earlier version of this function, wyckoff_tree_search is
    NOT run with an artificially widened tree_search.max_candidates
    here: that widening assumed the search's own budget was a
    reasonable proxy for the final candidate COUNT, which stopped being
    true once wyckoff_tree_search started expanding every
    shape-representative leaf into its full concrete-letter combination
    set internally (see search.dfs's module docstring) -- a handful of
    representative leaves can expand into tens of thousands of concrete
    ones, so trying to pre-size the search budget from max_cands no
    longer means anything reliable. wyckoff_tree_search's own
    tree_search.max_candidates (if the caller sets one) is honoured
    as-is; the shortlisting that actually matters for keeping this
    affordable now happens inside rank_by_feasibility_staged itself
    (its stage1_shortlist, sourced from tree_search.feasibility_shortlist
    if set).

    No-ops beyond the existing truncation when there's nothing to check
    (e.g. nacl/cscl-style configs with no coordination_constraints or
    angle_constraints at all): geometry.rank_by_feasibility_staged
    itself also short-circuits to a same-order, zero-cost pass in that
    case, so this is a true no-op end to end.

    Returns (valid_asgns, n_found, shell_seeds) -- n_found is the count
    BEFORE truncation (but after Phase 1's own search, which is run
    exactly once here), for status reporting. shell_seeds is a list
    parallel to valid_asgns (same length, same order); each entry is
    the shell_seed_info geometry.rank_by_feasibility already computed
    for that assignment's coupled local shells (see that function's
    docstring), or None where no ranking pass ran (e.g. no shape
    constraints, or a single candidate) -- pass shell_seeds[k] straight
    through to optimize.optimize_one_assignment's own shell_seed_info
    parameter so Phase 2 reuses this solve instead of repeating it.
    """
    has_shape_constraints = bool(sg_config.get("coordination_constraints")
                                  or sg_config.get("angle_constraints"))

    letter_feasibility = None
    if has_shape_constraints:
        feasibility_workers = sg_config.get("tree_search", {}).get("feasibility_workers")
        letter_feasibility = precompute_letter_feasibility(
            sg_config, multistart_cma_es, max_workers=feasibility_workers)

    valid_asgns = wyckoff_tree_search(sg_config, letter_feasibility=letter_feasibility)
    n_found = len(valid_asgns)
    if not valid_asgns:
        return valid_asgns, n_found, []

    shell_seeds = [None] * len(valid_asgns)
    if has_shape_constraints and len(valid_asgns) > 1:
        ts_cfg = sg_config.get("tree_search", {})
        stage1_shortlist = ts_cfg.get("feasibility_shortlist")
        if verbose:
            print(f"  Ranking {len(valid_asgns)} candidate(s) by geometric "
                  f"feasibility (staged) ...")
        ranked = rank_by_feasibility_staged(
            valid_asgns, sg_config, multistart_cma_es,
            stage1_shortlist=stage1_shortlist,
            letter_feasibility=letter_feasibility)
        valid_asgns = [a for a, _, _ in ranked]
        shell_seeds = [seed_info for _, _, seed_info in ranked]

    if len(valid_asgns) > max_cands:
        valid_asgns = valid_asgns[:max_cands]
        shell_seeds = shell_seeds[:max_cands]
    return valid_asgns, n_found, shell_seeds


def _optimize_candidates_parallel(valid_asgns: list, sg_config: dict,
                                   opt_cfg: dict, shell_seeds: list,
                                   workers: int) -> list:
    """
    Phase 2 candidate dispatch via optimize.parallel's
    optimize_assignments_parallel, shared by cli.py's single-
    composition mode and this module's _run_one_composition /
    _run_auto_composition_mode -- all three otherwise-independent
    Phase 2 loops hand off to this one function rather than each
    re-implementing the same candidate_workers branch (see
    pipeline.build_sg_config's own docstring for the analogous
    reasoning behind centralising per-SG config assembly).

    Prints one line per candidate AS IT COMPLETES (completion order,
    not valid_asgns order) -- deliberately simpler than the sequential
    loops' own per-candidate printing (label, lattice params, etc.),
    since per-candidate verbose output cannot interleave live across
    process boundaries under a process pool. Callers who need detailed
    diagnostics for a specific candidate should re-run without
    --workers/parallel.candidate_workers (sequential).

    Returns [(best_p, best_e, structure), ...] in the SAME order as
    valid_asgns (not completion order), so callers can zip it with
    valid_asgns exactly like the sequential loop it replaces.
    """
    n = len(valid_asgns)
    labels = [" + ".join(f"{a.element}@{a.entry.letter}" for a in asgn)
              for asgn in valid_asgns]
    done = [0]

    def _on_result(i, label, best_e):
        done[0] += 1
        print(f"  [{done[0]}/{n}] done: {label}  error={best_e:.6f}")

    return optimize_assignments_parallel(
        valid_asgns, sg_config, opt_cfg, max_workers=workers,
        labels=labels, shell_seeds=shell_seeds, on_result=_on_result)


def _run_one_composition(comp_str: str, base_config: dict,
                          opt_cfg: dict, verbose: bool,
                          run_idx: int, total_runs: int) -> list:
    """
    Run Phase-0 + Phase-1 + Phase-2 for a single composition string.
    Returns list of (error, comp_str, sg, asgn, best_p, structure, sg_config).
    """
    config = dict(base_config)
    config["composition"] = comp_str

    ionic_q  = config.get("ionic_charges", {})
    elements = list(parse_composition(comp_str).keys())
    ts_cfg   = config.get("tree_search", {})
    max_cands = ts_cfg.get("max_candidates", 50)
    sg_given  = config.get("spacegroup")
    auto_mode = (sg_given is None)

    print(f"\n{'*'*65}")
    print(f"  [{run_idx}/{total_runs}] Composition: {comp_str}")
    print(f"  Space group : {'AUTO' if auto_mode else sg_given}")
    print(f"{'*'*65}")

    # Phase 0: auto SG search
    if auto_mode:
        print(f"\n[Phase 0] Auto space-group search for {comp_str} ...")
        top_sgs = auto_search_spacegroups(config, verbose=verbose)
        if not top_sgs:
            print(f"  No viable SGs found for {comp_str}. Skipping.")
            return []
        print(f"  Selected SGs: {top_sgs}")
    else:
        top_sgs = [int(sg_given)]

    results = []

    for sg in top_sgs:
        sg_config = build_sg_config(config, sg)

        print(f"\n[Phase 1] SG {sg} - Wyckoff tree search for {comp_str} ...")
        try:
            valid_asgns, n_found, shell_seeds = search_and_rank(
                sg_config, max_cands, verbose=verbose)
        except Exception as e:
            print(f"  Tree search error: {e}")
            n_found, valid_asgns, shell_seeds = 0, [], []

        mode_label = ("composition-matched"
                      if _all_charges_zero(ionic_q, elements)
                      else "charge-neutral")
        print(f"  Found {n_found} {mode_label} assignment(s).")

        if not valid_asgns:
            continue

        workers = sg_config.get("parallel", {}).get("candidate_workers")
        if workers is not None and workers > 1 and len(valid_asgns) > 1:
            print(f"\n[Phase 2] SG {sg} - Optimising {len(valid_asgns)} "
                  f"candidate(s) for {comp_str} in parallel "
                  f"({workers} workers) ...")
            parallel_results = _optimize_candidates_parallel(
                valid_asgns, sg_config, opt_cfg, shell_seeds, workers)
            for (best_p, best_e, structure), asgn in zip(parallel_results, valid_asgns):
                results.append((best_e, comp_str, sg, asgn, best_p,
                                 structure, sg_config))
        else:
            print(f"\n[Phase 2] SG {sg} - Optimising {len(valid_asgns)} "
                  f"candidate(s) for {comp_str} ...")

            for k, asgn in enumerate(valid_asgns):
                label = " + ".join(
                    f"{a.element}@{a.entry.letter}" for a in asgn)
                print(f"\n  --- [{k+1}/{len(valid_asgns)}] {label} ---")

                best_p, best_e, structure = optimize_one_assignment(
                    asgn, sg_config, opt_cfg,
                    verbose=verbose, label=label,
                    shell_seed_info=shell_seeds[k])

                print(f"  Error = {best_e:.6f}")
                if structure is not None:
                    lp = structure["lattice_params"]
                    print(f"  Lattice: a={lp['a']:.4f}  c={lp.get('c', lp['a']):.4f} A")

                results.append((best_e, comp_str, sg, asgn, best_p,
                                 structure, sg_config))

    return results


def _run_auto_composition_mode(base_config: dict, args) -> None:
    """
    Fully automatic mode:
      (1) Select space groups  (Phase 0 auto-SG, or user-specified SG list)
      (2) For each selected SG, discover charge-neutral compositions
          automatically from the Wyckoff multiplicities
      (3) Run Wyckoff tree search + CMA-ES optimisation for every
          (composition, SG) pair
      (4) Global ranking and POSCAR output

    Controlled by config['auto_composition_settings']:
        elements            list  element symbols to use  (required)
        max_sites_per_elem  int   Wyckoff sites per element (default 3)
        max_atoms           int   atom-count ceiling (default 32)
        max_compositions    int   cap on compositions per SG (default 30)
        require_all_elements bool all elements must appear (default true)
    """
    ionic_q    = base_config.get("ionic_charges", {})
    ac_cfg     = base_config.get("auto_composition_settings", {})
    opt_cfg    = base_config.get("optimization", {})

    # Element list: from settings, or derived from ionic_charges keys
    elements = ac_cfg.get("elements") or list(ionic_q.keys())
    if not elements:
        print("ERROR: 'elements' must be set in auto_composition_settings "
              "or ionic_charges."); sys.exit(1)

    max_spe   = ac_cfg.get("max_sites_per_elem", 3)
    max_atoms = ac_cfg.get("max_atoms",           32)
    max_comp  = ac_cfg.get("max_compositions",    30)
    req_all   = ac_cfg.get("require_all_elements", True)

    # --- Banner ---
    print(f"\n{'='*65}")
    print(f"  Mizuho  -  Fully Automatic Composition + Structure Search")
    print(f"  Elements     : {elements}")
    print(f"  Ionic charges: {ionic_q}")
    print(f"  Max atoms/cell: {max_atoms}")
    print(f"  Max sites/elem: {max_spe}")
    sg_given = base_config.get("spacegroup")
    sg_list_given = base_config.get("spacegroup_list")
    if sg_given:
        print(f"  Space group  : {sg_given} (fixed)")
    elif sg_list_given:
        print(f"  Space groups : {sg_list_given} (user list)")
    else:
        print(f"  Space group  : AUTO (Phase 0 screening)")
    print(f"{'='*65}")

    # ================================================================
    # Step 1: Determine space group(s) to search
    # ================================================================
    if sg_given:
        sg_list = [int(sg_given)]
    elif sg_list_given:
        sg_list = [int(s) for s in sg_list_given]
    else:
        # Phase 0: use a dummy composition for SG screening
        # (pick the simplest possible composition for fast screening)
        dummy_comp = "".join(elements)   # e.g. "NaZrPO"
        dummy_cfg  = dict(base_config)
        dummy_cfg["composition"] = dummy_comp
        # Temporarily set a large charge tolerance for screening
        dummy_cfg["charge_tolerance_min"] = -999
        dummy_cfg["charge_tolerance_max"] =  999

        # For SG screening we need at least one valid Wyckoff assignment.
        # Use the first reachable charge-neutral composition we can find.
        # Find that composition from the simplest SG in the DB.
        screen_comp = None
        for sg_try in sorted(WYCKOFF_DB.keys()):
            comps = discover_compositions(
                sg_try, elements, ionic_q,
                max_sites_per_elem=2, max_atoms=24,
                max_compositions=1, require_all_elements=True)
            if comps:
                screen_comp = comps[0]
                break
        if screen_comp is None:
            screen_comp = dummy_comp

        print(f"\n[Step 1] Phase 0 SG screening (proxy composition: {screen_comp}) ...")
        proxy_cfg = dict(base_config)
        proxy_cfg["composition"] = screen_comp
        sg_list = auto_search_spacegroups(proxy_cfg, verbose=args.verbose)

        if not sg_list:
            print("  No viable SGs found by auto-screening.")
            print("  Falling back to all supported SGs in Wyckoff DB.")
            sg_list = sorted(WYCKOFF_DB.keys())

    print(f"\n  Space groups selected: {sg_list}")

    # ================================================================
    # Step 2: For each SG, discover compositions
    # ================================================================
    print(f"\n[Step 2] Auto-discovering charge-neutral compositions ...")
    print(f"  {'SG':>5}  {'Crystal system':>14}  # compositions  Compositions")
    print(f"  {'-'*62}")

    sg_comp_map: Dict[int, List[str]] = {}
    for sg in sg_list:
        if sg not in WYCKOFF_DB:
            continue
        comps = discover_compositions(
            sg, elements, ionic_q,
            max_sites_per_elem=max_spe,
            max_atoms=max_atoms,
            max_compositions=max_comp,
            require_all_elements=req_all)
        sg_comp_map[sg] = comps
        cs = CRYSTAL_SYSTEM_OF_SG.get(sg, "?")
        comp_preview = ", ".join(comps[:4])
        if len(comps) > 4:
            comp_preview += f" ... (+{len(comps)-4} more)"
        print(f"  {sg:>5}  {cs:>14}  {len(comps):>13}  {comp_preview}")

    total_pairs = sum(len(v) for v in sg_comp_map.values())
    print(f"\n  Total (SG, composition) pairs to optimise: {total_pairs}")

    if total_pairs == 0:
        print("  No compositions found. "
              "Try increasing max_atoms or max_sites_per_elem.")
        return

    # ================================================================
    # Steps 3 & 4: Tree search + optimisation for every (SG, comp)
    # ================================================================
    all_results = []   # (error, comp, sg, asgn, best_p, structure, sg_cfg)
    ts_cfg      = base_config.get("tree_search", {})
    max_cands   = ts_cfg.get("max_candidates", 50)
    pair_idx    = 0

    for sg, comps in sg_comp_map.items():
        cs = CRYSTAL_SYSTEM_OF_SG.get(sg, "?")

        for comp_str in comps:
            pair_idx += 1
            print(f"\n{'='*65}")
            print(f"  [{pair_idx}/{total_pairs}]  SG {sg} ({cs})  |  {comp_str}")
            print(f"{'='*65}")

            sg_config = build_sg_config(base_config, sg, composition=comp_str)

            # Step 3a: Wyckoff tree search
            print(f"\n[Phase 1] Wyckoff tree search ...")
            try:
                valid_asgns, n_found, shell_seeds = search_and_rank(
                    sg_config, max_cands, verbose=args.verbose)
            except Exception as exc:
                print(f"  Tree search error: {exc}")
                n_found, valid_asgns, shell_seeds = 0, [], []

            mode = ("composition-matched"
                    if _all_charges_zero(ionic_q, elements)
                    else "charge-neutral")
            print(f"  Found {n_found} {mode} assignment(s).")

            if not valid_asgns:
                print(f"  No valid Wyckoff assignments -- skipping.")
                continue

            # Step 3b: CMA-ES optimisation
            workers = sg_config.get("parallel", {}).get("candidate_workers")
            if workers is not None and workers > 1 and len(valid_asgns) > 1:
                print(f"\n[Phase 2] Optimising {len(valid_asgns)} "
                      f"candidate(s) in parallel ({workers} workers) ...")
                parallel_results = _optimize_candidates_parallel(
                    valid_asgns, sg_config, opt_cfg, shell_seeds, workers)
                for (best_p, best_e, structure), asgn in zip(parallel_results, valid_asgns):
                    all_results.append(
                        (best_e, comp_str, sg, asgn, best_p, structure, sg_config))
            else:
                print(f"\n[Phase 2] Optimising {len(valid_asgns)} candidate(s) ...")
                for k, asgn in enumerate(valid_asgns):
                    label = " + ".join(
                        f"{a.element}@{a.entry.letter}" for a in asgn)
                    print(f"\n  --- [{k+1}/{len(valid_asgns)}] {label} ---")

                    best_p, best_e, structure = optimize_one_assignment(
                        asgn, sg_config, opt_cfg,
                        verbose=args.verbose, label=label,
                        shell_seed_info=shell_seeds[k])

                    print(f"  Error = {best_e:.6f}")
                    if structure is not None:
                        lp = structure["lattice_params"]
                        print(f"  Lattice: a={lp['a']:.4f}  "
                              f"c={lp.get('c', lp['a']):.4f} A")

                    all_results.append(
                        (best_e, comp_str, sg, asgn, best_p, structure, sg_config))

    if not all_results:
        print("\n  No valid structures found."); return

    # ================================================================
    # Step 4: Global ranking + POSCAR output
    # ================================================================
    all_results.sort(key=lambda x: x[0])

    print(f"\n{'='*65}")
    print("  GLOBAL RANKING  (auto-discovered compositions)")
    print(f"{'='*65}")
    print(f"  {'Rank':>4}  {'Error':>12}  {'SG':>5}  "
          f"{'Composition':<22}  Assignment")
    print(f"  {'-'*72}")
    for rank, (err, comp, sg, asgn, _, _, _) in enumerate(all_results, 1):
        label = " + ".join(
            f"{a.element}@{a.entry.letter}({a.entry.multiplicity})"
            for a in asgn)
        print(f"  {rank:>4}  {err:>12.5f}  {sg:>5}  {comp:<22}  {label}")

    top_n = min(args.top, len(all_results))
    for rank in range(top_n):
        err, comp, sg, asgn, best_p, structure, sg_config = all_results[rank]
        if structure is None:
            print(f"\n  [Rank {rank+1}] No valid structure."); continue

        win_cfg = dict(sg_config)
        win_cfg["wyckoff_sites"] = assignments_to_sites(asgn)
        win_cfg["composition"]   = comp

        print_summary(structure, asgn, win_cfg, err,
                      param_vector=best_p)

        safe = comp.replace(" ", "_")
        out_path = (args.output if args.output and top_n == 1
                    else f"{safe}_auto_rank{rank+1}.vasp")
        write_poscar(structure, win_cfg, out_path)

    print("\n  Done.\n")


def _run_composition_pool(base_config: dict, pool: list, args) -> None:
    """
    Iterate over all compositions in the pool, run full search for each,
    and report a global ranking.
    """
    opt_cfg = base_config.get("optimization", {})
    total   = len(pool)

    print(f"\n{'='*65}")
    print(f"  Mizuho  -  Multi-Composition Pool Search")
    print(f"  Pool size    : {total} compositions")
    print(f"  Ionic charges: {base_config.get('ionic_charges', {})}")
    pool_sg = base_config.get("spacegroup")
    print(f"  Space group  : {'AUTO' if pool_sg is None else pool_sg}")
    print(f"{'='*65}")
    print(f"\n  Composition pool:")
    for i, c in enumerate(pool, 1):
        print(f"    [{i:3d}] {c}")

    all_results = []   # (error, comp_str, sg, asgn, best_p, structure, sg_cfg)

    for idx, comp_str in enumerate(pool, 1):
        comp_results = _run_one_composition(
            comp_str, base_config, opt_cfg,
            verbose=args.verbose,
            run_idx=idx, total_runs=total)
        all_results.extend(comp_results)

    if not all_results:
        print("\n  No valid structures found across all compositions.")
        return

    # Global ranking
    all_results.sort(key=lambda x: x[0])

    print(f"\n{'='*65}")
    print("  GLOBAL RANKING  (all compositions)")
    print(f"{'='*65}")
    print(f"  {'Rank':>4}  {'Error':>12}  {'SG':>5}  "
          f"{'Composition':<22}  Assignment")
    print(f"  {'-'*70}")
    for rank, (err, comp, sg, asgn, _, _, _) in enumerate(all_results, 1):
        label = " + ".join(
            f"{a.element}@{a.entry.letter}({a.entry.multiplicity})"
            for a in asgn)
        print(f"  {rank:>4}  {err:>12.5f}  {sg:>5}  {comp:<22}  {label}")

    # Output top-N POSCARs
    top_n = min(args.top, len(all_results))
    for rank in range(top_n):
        err, comp, sg, asgn, best_p, structure, sg_config = all_results[rank]
        if structure is None:
            print(f"\n  [Rank {rank+1}] No valid structure.")
            continue

        win_cfg = dict(sg_config)
        win_cfg["wyckoff_sites"] = assignments_to_sites(asgn)
        win_cfg["composition"]   = comp

        print_summary(structure, asgn, win_cfg, err,
                      param_vector=best_p)

        safe_comp = comp.replace(" ", "_")
        out_path  = (args.output if args.output and top_n == 1
                     else f"{safe_comp}_rank{rank+1}.vasp")
        write_poscar(structure, win_cfg, out_path)

    print("\n  Done.\n")
