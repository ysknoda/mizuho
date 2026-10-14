"""Command-line entry point."""

import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml

from .composition import _all_charges_zero, parse_composition
from .data import WYCKOFF_DB
from .diagnostics import (
    bound_contacts, lattice_names, print_coordination_table, print_diagnostics,
    print_symmetry_ladder,
)
from .io.cif import cif_to_mizuho, write_cif
from .io.poscar import write_poscar
from .io.report import print_summary
from .objective.breakdown import error_breakdown, print_error_breakdown
from .objective.energy import compute_error
from .lattice import CRYSTAL_SYSTEM_OF_SG, get_lattice_system
from .optimize.driver import optimize_one_assignment
from .pipeline import (
    _run_auto_composition_mode, _run_composition_pool, build_sg_config,
    merge_same_topology, optimise_candidates, promote_to_detected_supergroup,
    search_and_rank,
)
from .search import auto_search_spacegroups
from .normalizer import (
    assignment_key, canonical_key, letter_permutations, polar_directions,
)
from .objective.penalties import reject_removed_keys
from .structure import assignments_to_sites, build_structure, resolve_fixed_assignment
from .symmetry import SUCCESS_TOLERANCE, hall_number_of_sg


def _check_fixed_assignment_matches_composition(asgn, comp_str: str) -> None:
    """
    A user-supplied wyckoff_assignment is a hand-picked list, so unlike
    the tree search's own output it hasn't already been filtered for
    consistency with the target composition. Verify every element's
    total multiplicity is the SAME positive-integer multiple of its
    target composition count, and that no element is missing or extra.
    """
    from collections import defaultdict
    target = parse_composition(comp_str)
    got: dict = defaultdict(int)
    for a in asgn:
        got[a.element] += a.entry.multiplicity

    if set(got) != set(target):
        print(f"ERROR: wyckoff_assignment elements {sorted(got)} don't "
              f"match composition elements {sorted(target)}."); sys.exit(1)

    ratios = {e: got[e] / target[e] for e in target}
    k0 = next(iter(ratios.values()))
    if any(abs(r - k0) > 1e-9 or abs(k0 - round(k0)) > 1e-9 for r in ratios.values()):
        print(f"ERROR: wyckoff_assignment atom counts {dict(got)} are not "
              f"a consistent integer multiple of composition {target} "
              f"(per-element ratios: {ratios})."); sys.exit(1)


AUTO_SPACEGROUP_WARNING = (
    "  WARNING: the automatic space-group search is experimental. Its "
    "known-answer tests\n  (NaCl -> 225 among them) do not pass; check the "
    "detected space group of\n  every result (docs/adr/0021).")


def _check_mode(config: dict) -> None:
    """Whether a config may run, and in which space-group mode.

    "mode" must agree with whether "spacegroup" is present, so a config
    can't silently ask for one mode while supplying inputs for the other.
    The automatic space-group search is opt-in: it has not reached its
    known answers, so a config that simply omits "spacegroup" is rejected
    rather than handed a plausible structure in the wrong group
    (docs/adr/0021). Applies to every composition mode, all of which search
    space groups when none is given."""
    sg_given = config.get("spacegroup")
    mode = config.get("mode")
    if mode is not None and mode not in ("single_spacegroup", "auto_spacegroup"):
        print(f"ERROR: mode must be 'single_spacegroup' or 'auto_spacegroup', "
              f"got {mode!r}."); sys.exit(1)
    if mode == "single_spacegroup" and sg_given is None:
        print("ERROR: mode: single_spacegroup requires 'spacegroup' to be set."); sys.exit(1)
    if mode == "auto_spacegroup" and sg_given is not None:
        print("ERROR: mode: auto_spacegroup requires 'spacegroup' to be "
              "omitted (it selects one automatically)."); sys.exit(1)
    if sg_given is None and mode != "auto_spacegroup":
        print("ERROR: no 'spacegroup' given. The automatic space-group search "
              "does not yet reach its known answers (docs/adr/0021); set "
              "'spacegroup', or opt in with 'mode: auto_spacegroup'."); sys.exit(1)
    if sg_given is None:
        print(AUTO_SPACEGROUP_WARNING)


def _cmd_run(args) -> None:
    """`mizuho run`: predict structures (docs/adr/0013-cli-subcommands.md)."""
    # ----------------------------------------------------------------
    # Load config
    # ----------------------------------------------------------------
    config = _load_config(args.input)

    # A single injection point here propagates --seed to every downstream
    # consumer of config["optimization"]["seed"] (Phase 0 Stage 0-C,
    # Phase 2 optimize_one_assignment, and the auto_composition /
    # composition_pool drivers), because they all read that same nested
    # dict object via dict.get(..., default) rather than copies of it.
    if args.seed is not None:
        config.setdefault("optimization", {})["seed"] = args.seed
    if args.workers is not None:
        config.setdefault("parallel", {})["candidate_workers"] = args.workers

    # ================================================================
    # Mode selection
    # ================================================================
    # Priority:
    #   1. "auto_composition: true"  -> fully automatic: SG auto-search
    #      + composition auto-discovery from Wyckoff multiplicities
    #   2. "composition_pool": [...]  -> user-supplied pool (legacy)
    #   3. "composition": "..."       -> single composition (original)
    # ================================================================
    _check_mode(config)

    auto_comp = config.get("auto_composition", False)
    pool      = config.get("composition_pool")

    if auto_comp:
        _run_auto_composition_mode(config, args)
        return

    if pool:
        _run_composition_pool(config, pool, args)
        return

    # ----------------------------------------------------------------
    # Single-composition mode (original behaviour)
    # ----------------------------------------------------------------
    opt_cfg   = config.get("optimization", {})
    ionic_q   = config.get("ionic_charges", {})
    elements  = list(parse_composition(config["composition"]).keys())
    ts_cfg    = config.get("tree_search", {})
    max_cands = ts_cfg.get("max_candidates", 50)

    # Determine whether spacegroup is specified. "mode" is an optional,
    # explicit alternative to leaving "spacegroup" out to request
    # auto-search -- if given, it must agree with whether "spacegroup"
    # is present, so a config can't silently ask for one mode while
    # supplying inputs for the other.
    sg_given  = config.get("spacegroup")   # None or int
    auto_mode = (sg_given is None)

    # `wyckoff_assignment` (optional): a user-supplied, already-known
    # site assignment ([{element, letter}, ...]) that skips Phase 1's
    # combinatorial search entirely and hands the fixed assignment
    # straight to Phase 2. Meant for cases where the crystallographic
    # answer is already known (e.g. reproducing a published structure)
    # and the discrete search either isn't needed or -- for
    # compositions where several elements each occupy multiple
    # symmetry-inequivalent sites -- is impractically slow to reach the
    # right combination on its own (Phase 1's best-first ordering is
    # driven by a cheap charge-balance/packing surrogate, which has no
    # way to know a deep, multi-site-per-element combination is the
    # physically correct one until it's actually reached).
    fixed_wy = config.get("wyckoff_assignment")
    if fixed_wy is not None and auto_mode:
        print("ERROR: 'wyckoff_assignment' requires a fixed 'spacegroup' "
              "(it names Wyckoff letters of one specific space group)."); sys.exit(1)

    # ----------------------------------------------------------------
    # Banner
    # ----------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"  Crystal Structure Predictor v6")
    print(f"  Composition  : {config['composition']}")
    if auto_mode:
        print(f"  Space group  : AUTO, experimental (Phase 0 screening active)")
    else:
        sg = int(sg_given)
        if sg not in WYCKOFF_DB:
            print(f"ERROR: SG {sg} not supported. "
                  f"Supported: {sorted(WYCKOFF_DB.keys())}"); sys.exit(1)
        print(f"  Space group  : {sg} ({config.get('spacegroup_symbol','')})")
        print(f"  Lattice sys  : {get_lattice_system(sg)}")

    if _all_charges_zero(ionic_q, elements) or \
            ts_cfg.get("force_multiplicity_mode", False):
        print(f"  Search mode  : COVALENT  (multiplicity-ratio filter)")
    else:
        print(f"  Search mode  : IONIC  (charge-neutrality filter)")
        if ionic_q:
            print(f"  Ionic charges: {ionic_q}")

    hyb_csts = config.get("hybridisation_constraints", [])
    if hyb_csts:
        print(f"  Hybridisation: "
              + ", ".join(f"{h['element']}({h['hybridisation']})"
                          for h in hyb_csts))
    print(f"{'='*65}")

    # ================================================================
    # Phase 0: automatic space-group screening (auto mode only)
    # ================================================================
    if auto_mode:
        print("\n[Phase 0] Automatic space-group search ...")
        sg_search_cfg = config.get("spacegroup_search", {})
        top_k         = sg_search_cfg.get("top_sg_count", 5)

        top_sgs = auto_search_spacegroups(config, verbose=args.verbose)

        if not top_sgs:
            print("  Phase 0 found no viable space groups. "
                  "Check constraints or broaden lattice_bounds."); sys.exit(0)

        print(f"\n  Phase 0 selected {len(top_sgs)} space group(s) for "
              f"full optimisation: {top_sgs}")
    else:
        top_sgs = [int(sg_given)]

    # ================================================================
    # Phase 1 + 2: full tree search and optimisation for each SG
    # ================================================================
    all_results = []   # (error, sg, asgn, best_p, structure)

    for sg_idx, sg in enumerate(top_sgs):
        # Build a per-SG config
        sg_config = build_sg_config(config, sg)

        if auto_mode:
            print(f"\n{'-'*65}")
            print(f"  [SG {sg_idx+1}/{len(top_sgs)}] SG {sg} "
                  f"({CRYSTAL_SYSTEM_OF_SG.get(sg,'?')}) - full search")
            print(f"{'-'*65}")

        if fixed_wy is not None:
            print(f"\n[Phase 1] SG {sg} - using fixed wyckoff_assignment "
                  f"(Phase 1 search skipped) ...")
            try:
                asgn = resolve_fixed_assignment(sg, fixed_wy)
            except ValueError as e:
                print(f"ERROR: {e}"); sys.exit(1)
            _check_fixed_assignment_matches_composition(asgn, config["composition"])
            valid_asgns = [asgn]
            # No ranking pass ran for a directly-fixed assignment, so
            # there is no already-solved shell to reuse --
            # optimize_one_assignment solves it fresh when needed.
            shell_seeds = [None]
            print(f"  Using 1 fixed assignment.")
        else:
            # Phase 1: tree search (+ geometric-feasibility re-ranking)
            print(f"\n[Phase 1] SG {sg} - Wyckoff assignment tree search ...")
            valid_asgns, n_found, shell_seeds = search_and_rank(
                sg_config, max_cands, verbose=args.verbose)

            mode_label = ("composition-matched"
                          if _all_charges_zero(ionic_q, elements)
                          else "charge-neutral")
            print(f"  Found {n_found} {mode_label} assignment(s).")

            if not valid_asgns:
                print(f"  Skipping SG {sg} (no valid assignments).")
                continue

        # Phase 2: optimise each Wyckoff assignment; one entry per
        # topology class it reached, up to --top (ADR-0019)
        for asgn, r in optimise_candidates(valid_asgns, sg_config, opt_cfg,
                                           shell_seeds, args.verbose, args.top,
                                           header=f"SG{sg} "):
            if r is None:
                all_results.append((1e10, sg, asgn, None, None, sg_config))
            else:
                all_results.append((r.error, sg, asgn, r.params, r.structure, sg_config))

    if not all_results:
        print("\nNo valid structures found across all space groups."); sys.exit(0)

    # A structure whose detected symmetry is higher than the group it was
    # optimised in is re-expressed and re-optimised there (ADR-0010),
    # repeatedly, since the re-optimised structure may climb further.
    promoted = []
    for entry in all_results:
        err, sg, asgn, best_p, structure, sg_config = entry
        for _ in range(3):
            if structure is None:
                break
            up = promote_to_detected_supergroup(sg, best_p, structure, sg_config, config,
                                                opt_cfg, SUCCESS_TOLERANCE, verbose=True)
            if up is None:
                break
            err, sg, asgn, best_p, structure, sg_config = up
        promoted.append((err, sg, asgn, best_p, structure, sg_config))
    all_results = promoted

    # ----------------------------------------------------------------
    # Global ranking
    # ----------------------------------------------------------------
    all_results.sort(key=lambda x: x[0])
    # One row per bond-graph topology class: entries that reached the same
    # class are one structure (ADR-0019); "Same" counts the merged ones.
    ranked = merge_same_topology(all_results, lambda e: e[3], _run_entry_config)

    print(f"\n{'='*65}")
    print("  GLOBAL RANKING")
    print(f"{'='*65}")
    print(f"  {'Rank':>4}  {'Error':>12}  {'SG':>5}  "
          f"{'Crystal system':>14}  {'Same':>4}  Assignment")
    print(f"  {'-'*69}")
    for rank, ((err, sg, asgn, _, _, _), n_same) in enumerate(ranked, 1):
        label = " + ".join(
            f"{a.element}@{a.entry.letter}({a.entry.multiplicity})"
            for a in asgn)
        cs = CRYSTAL_SYSTEM_OF_SG.get(sg, "?")
        print(f"  {rank:>4}  {err:>12.5f}  {sg:>5}  {cs:>14}  {n_same:>4}  {label}")

    # ----------------------------------------------------------------
    # Output top-N structures
    # ----------------------------------------------------------------
    top_n = min(args.top, len(ranked))
    comp  = config["composition"].replace(" ", "_")

    for rank in range(top_n):
        err, sg, asgn, best_p, structure, sg_config = ranked[rank][0]
        if structure is None:
            print(f"\n  [Rank {rank+1}] No valid structure obtained.")
            continue

        win_cfg = dict(sg_config)
        win_cfg["wyckoff_sites"] = assignments_to_sites(asgn)

        print_summary(structure, asgn, win_cfg, err,
                      param_vector=best_p)
        print_diagnostics(structure, win_cfg, best_p, error_breakdown(best_p, win_cfg))

        ext = "cif" if args.format == "cif" else "vasp"
        out_path = (args.output if args.output and top_n == 1
                    else f"{comp}_rank{rank+1}.{ext}")
        if args.format == "cif":
            write_cif(best_p, win_cfg, out_path, title=f"{comp}_rank{rank+1}")
            print(f"\n  CIF written: {out_path}")
        else:
            write_poscar(structure, win_cfg, out_path)

    print("\n  Done.\n")


def _run_entry_config(entry) -> dict:
    """The full config of a `run` ranking entry."""
    err, sg, asgn, best_p, structure, sg_config = entry
    cfg = dict(sg_config)
    cfg["wyckoff_sites"] = assignments_to_sites(asgn)
    return cfg


# ======================================================================
# Shared helpers for the subcommands
# ======================================================================

def _load_config(path: str) -> dict:
    input_path = Path(path)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found."); sys.exit(1)
    with open(input_path) as fh:
        config = yaml.safe_load(fh)
    try:
        reject_removed_keys(config)
    except ValueError as e:
        print(f"ERROR: {e}"); sys.exit(1)
    return config


def _reference_config(config: dict, cif_path: str):
    """The config's constraints applied to a reference CIF's own topology,
    in mizuho's setting (mizuho.io.cif.cif_to_mizuho)."""
    try:
        ref = cif_to_mizuho(cif_path)
    except (OSError, ValueError, KeyError) as e:
        print(f"ERROR: cannot read {cif_path}: {e}"); sys.exit(1)
    if config.get("spacegroup") not in (None, ref.spacegroup):
        print(f"  NOTE: config names SG {config['spacegroup']}, the reference is "
              f"SG {ref.spacegroup}; the reference's group is used.")
    sg_config = build_sg_config(config, ref.spacegroup)
    sg_config["spacegroup"] = ref.spacegroup
    sg_config["wyckoff_sites"] = ref.config()["wyckoff_sites"]
    return ref, sg_config


def _lattice_dict(sg: int, param_vector) -> dict:
    return dict(zip(lattice_names(sg), (float(v) for v in param_vector)))


# ======================================================================
# check
# ======================================================================

def _cmd_check(args) -> None:
    """Evaluate the objective at a reference structure. No optimisation."""
    config = _load_config(args.config)
    ref, cfg = _reference_config(config, args.reference)
    st = build_structure(ref.param_vector, cfg)
    bd = error_breakdown(ref.param_vector, cfg)

    print(f"\n{'='*65}")
    print(f"  CHECK  {args.reference}")
    print(f"{'='*65}")
    print(f"  Space group : {ref.spacegroup}   (read at symprec {ref.symprec:g} A)")
    print(f"  Assignment  : {ref.label()}")
    lat = _lattice_dict(ref.spacegroup, ref.param_vector)
    print("  Lattice     : " + "  ".join(f"{k}={v:.4f}" for k, v in lat.items()))
    print(f"  Objective   : {bd['total']:.8f}")
    print_error_breakdown(bd)
    print_coordination_table(bd)
    print_symmetry_ladder(st)
    if "lattice_bounds" in cfg:
        from .structure import build_bounds
        for name, (lo, hi), v in zip(lattice_names(ref.spacegroup), build_bounds(cfg),
                                     ref.param_vector):
            if not lo <= v <= hi:
                print(f"  WARNING: reference {name} = {v:.4f} lies outside "
                      f"lattice_bounds [{lo}, {hi}]")
        for name, side, v, _ in bound_contacts(ref.param_vector, cfg):
            print(f"  NOTE: reference {name} = {v:.4f} is on the {side} bound")


# ======================================================================
# fit
# ======================================================================

# ADR-0005's global-minimum gate: converged means error <= reference + 1e-3,
# accurate means every lattice length within 3% of the reference.
_CONVERGED_MARGIN = 1e-3
_LATTICE_TARGET = 0.03


def _cmd_fit(args) -> None:
    """Fix a reference's topology, optimise, and compare with the reference."""
    config = _load_config(args.config)
    if args.seed is not None:
        config.setdefault("optimization", {})["seed"] = args.seed
    ref, cfg = _reference_config(config, args.reference)
    err_ref = compute_error(ref.param_vector, cfg)

    opt_cfg = dict(cfg.get("optimization", {}))
    if args.budget != 1:
        opt_cfg["n_restarts"] = int(opt_cfg.get("n_restarts", 8) * args.budget)
        opt_cfg["cma_maxiter"] = int(opt_cfg.get("cma_maxiter", 300) * args.budget)
    asgn = resolve_fixed_assignment(ref.spacegroup, ref.specs)
    print(f"\n  Fitting {ref.label()} in SG {ref.spacegroup} "
          f"(n_restarts={opt_cfg.get('n_restarts')}, cma_maxiter={opt_cfg.get('cma_maxiter')}) ...")
    best_p, best_e, structure = optimize_one_assignment(
        asgn, cfg, opt_cfg, verbose=args.verbose, label=ref.label())
    if structure is None:
        print("ERROR: no structure obtained."); sys.exit(1)

    print(f"\n{'='*65}")
    print(f"  FIT  {args.reference}")
    print(f"{'='*65}")
    print(f"  Objective at reference : {err_ref:.8f}")
    print(f"  Objective at fit       : {best_e:.8f}")
    lat_ref = _lattice_dict(ref.spacegroup, ref.param_vector)
    lat_fit = _lattice_dict(ref.spacegroup, best_p)
    print(f"\n  {'param':>6}  {'reference':>10}  {'fit':>10}  {'deviation':>10}")
    worst = 0.0
    for k in lat_ref:
        if k in ("a", "b", "c"):
            dev = (lat_fit[k] - lat_ref[k]) / lat_ref[k]
            worst = max(worst, abs(dev))
            print(f"  {k:>6}  {lat_ref[k]:10.4f}  {lat_fit[k]:10.4f}  {100 * dev:+9.2f}%")
        else:
            print(f"  {k:>6}  {lat_ref[k]:10.3f}  {lat_fit[k]:10.3f}  "
                  f"{lat_fit[k] - lat_ref[k]:+8.3f} deg")
    converged = best_e <= err_ref + _CONVERGED_MARGIN
    print(f"\n  converged (error <= reference + {_CONVERGED_MARGIN:g}) : {'yes' if converged else 'NO'}")
    print(f"  lattice within {100 * _LATTICE_TARGET:g}% of reference       : "
          f"{'yes' if worst <= _LATTICE_TARGET else 'NO'} (worst {100 * worst:.2f}%)")
    if best_e < err_ref and worst > _LATTICE_TARGET:
        print("  WARNING: the fit scores below the reference while its lattice is "
              "outside the target -- an impostor "
              "(docs/adr/0007-the-real-structure-must-be-the-unique-minimum.md)")
    print_diagnostics(structure, cfg, best_p, error_breakdown(best_p, cfg))
    if args.output:
        write_cif(best_p, cfg, args.output, title="fit")
        print(f"\n  CIF written: {args.output}")


# ======================================================================
# candidates
# ======================================================================

def _parse_label(text: str):
    pairs = []
    for tok in text.replace("+", " ").split():
        el, le = tok.split("@")
        pairs.append((el, le))
    return pairs


def _cmd_candidates(args) -> None:
    """Phase 1 only: the ranking after merging equivalent assignments."""
    config = _load_config(args.config)
    sg = config.get("spacegroup")
    if sg is None:
        print("ERROR: candidates needs a fixed 'spacegroup'."); sys.exit(1)
    sg = int(sg)
    base_seed = args.seed if args.seed is not None else config.get("optimization", {}).get("seed", 0)
    seeds = [base_seed + k for k in range(args.seeds)]

    rankings = []
    for seed in seeds:
        sg_config = build_sg_config(config, sg)
        sg_config["optimization"] = dict(sg_config.get("optimization", {}), seed=seed)
        asgns, n_found, _ = search_and_rank(sg_config, max_cands=10**9, verbose=args.verbose)
        rankings.append([assignment_key(sg, a) for a in asgns])
        print(f"  seed {seed}: {n_found} enumerated, {len(asgns)} equivalence classes")
    if not rankings[0]:
        print("  No candidates."); return

    first = rankings[0]
    rank_of = [{k: i + 1 for i, k in enumerate(r)} for r in rankings]
    entries = {e.letter: e for e in WYCKOFF_DB[sg]}

    def n_free(key):
        return sum(len(entries[le].free_params) for _, le in key)

    strata = sorted({n_free(k) for k in first})
    print(f"  {len(strata)} free-parameter strata: {strata}")

    def row(key):
        ranks = [r.get(key) for r in rank_of]
        within = sum(1 for k in first[:first.index(key) + 1] if n_free(k) == n_free(key))
        text = " ".join(f"{el}@{le}" for el, le in key)
        spread = (f"  ranks {ranks}  worst {max(ranks)}  spread {max(ranks) - min(ranks)}"
                  if len(ranks) > 1 else "")
        return (f"  {first.index(key) + 1:>6}  free {n_free(key):>3}  "
                f"stratum-rank {within:>5}  {text}{spread}")

    if args.find:
        key = canonical_key(sg, _parse_label(args.find))
        if key not in first:
            print(f"  {args.find}: not among the {len(first)} classes"); sys.exit(1)
        print(row(key))
        return
    print(f"\n  {'rank':>6}  {'free':>8}  {'stratum':>17}  assignment")
    for key in first[:args.top]:
        print(row(key))


# ======================================================================
# wyckoff
# ======================================================================

def _format_slot(slot) -> str:
    from fractions import Fraction
    def lin(coeffs):
        return "".join(f"{'+' if k > 0 else '-'}{abs(k) if abs(k) != 1 else ''}{n}"
                       for n, k in coeffs.items()).lstrip("+")
    if isinstance(slot, tuple):
        coeffs, off = slot
        return lin(coeffs) + (f"+{Fraction(off).limit_denominator(24)}" if off else "")
    if isinstance(slot, dict):
        return lin(slot)
    return str(Fraction(float(slot)).limit_denominator(24))


def _cmd_wyckoff(args) -> None:
    """List a space group's Wyckoff positions in mizuho's setting."""
    import spglib
    import warnings
    sg = args.spacegroup
    if sg not in WYCKOFF_DB:
        print(f"ERROR: no space group {sg}."); sys.exit(1)
    hall = hall_number_of_sg(sg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        t = spglib.get_spacegroup_type(hall)
    print(f"  SG {sg}  {t.international_full}  Hall {hall} '{t.hall_symbol}'  "
          f"setting choice '{t.choice or '-'}'  ({get_lattice_system(sg)})")
    classes = {}
    for perm in letter_permutations(sg):
        for a, b in perm.items():
            classes.setdefault(a, set()).add(b)
    polar = polar_directions(sg)
    if len(polar):
        print(f"  Floating origin along: {[list(map(float, v.round(6))) for v in polar]}")
    print(f"\n  {'letter':>6}  {'mult':>4}  {'free':>7}  {'representative':<24}  equivalent")
    for e in sorted(WYCKOFF_DB[sg], key=lambda e: (e.multiplicity, e.letter)):
        rep = "(" + ", ".join(_format_slot(s) for s in e.coord_template) + ")"
        eq = sorted(classes.get(e.letter, {e.letter}) - {e.letter})
        print(f"  {e.letter:>6}  {e.multiplicity:>4}  {','.join(e.free_params) or '-':>7}  "
              f"{rep:<24}  {' '.join(eq) or '-'}")


# ======================================================================
# compare
# ======================================================================

def _candidate_structure(path: str, config: dict, ref, ref_cfg: dict):
    """(parameter vector, config, space group) of a candidate. A CIF carries
    its own assignment; a JSON record (scripts/record_impostor.py, a
    budget-ladder result) is a parameter vector in the reference's
    assignment, accepted only when its `published_vector`, if present, is
    the reference's own."""
    if path.endswith(".json"):
        import json
        import numpy as np

        record = json.loads(Path(path).read_text())
        pv = np.asarray(record["param_vector"], float)
        if len(pv) != len(ref.param_vector):
            print(f"ERROR: {path}: {len(pv)} parameters, the reference "
                  f"{ref.label()} has {len(ref.param_vector)}"); sys.exit(1)
        pub = record.get("published_vector")
        if pub is not None and not np.allclose(pub, ref.param_vector, atol=1e-3):
            print(f"ERROR: {path} was recorded against a different reference "
                  f"vector (or assignment order) than {ref.label()}"); sys.exit(1)
        return pv, ref_cfg, ref.spacegroup
    other, cfg = _reference_config(config, path)
    return other.param_vector, cfg, other.spacegroup


def _cmd_compare(args) -> None:
    """Bond-graph topology class and geometric differences against a reference."""
    import json
    from . import bond_graph as bg

    config = _load_config(args.config)
    ref, ref_cfg = _reference_config(config, args.reference)
    cations = {e for e, q in (ref_cfg.get("ionic_charges") or {}).items() if q > 0}

    def analyse(name, pv, cfg, sg):
        st = build_structure(pv, cfg)
        crystal = bg.Crystal(st["lattice_matrix"], st["species"], st["frac_coords"], name)
        cut, ranges = bg.pair_cutoffs(cfg, set(st["species"]))
        graph = bg.build_graph(crystal, cut, cations)
        topo = bg.topology(graph, args.depth)
        grey = bg.grey_contacts(crystal, cut, ranges)
        return {"name": name, "sg": sg, "crystal": crystal, "cutoffs": cut, "ranges": ranges,
                "grey": grey, "topo": topo,
                "geo": bg.geometry(graph, topo), "objective": error_breakdown(pv, cfg)["total"]}

    r = analyse(args.reference, ref.param_vector, ref_cfg, ref.spacegroup)
    cands = [analyse(p, *_candidate_structure(p, config, ref, ref_cfg)) for p in args.candidates]

    classes = {r["topo"].signature: "A"}
    for c in cands:
        classes.setdefault(c["topo"].signature, chr(ord("A") + len(classes)))

    print(f"\n{'='*65}")
    print(f"  COMPARE  {args.reference}")
    print(f"{'='*65}")
    print("  Bond cutoffs : " + "  ".join(
        f"{'-'.join(sorted(k))} {v:g} A" for k, v in sorted(r["cutoffs"].items(), key=lambda x: sorted(x[0]))))
    for k, (lo, hi) in sorted(r["ranges"].items(), key=lambda x: sorted(x[0])):
        if hi > r["cutoffs"][k]:
            print(f"  Grey band    : {'-'.join(sorted(k))} ({r['cutoffs'][k]:g}, {hi:g}] A -- another "
                  f"constraint's cutoff counts contacts the bond graph does not")
    print(f"  Reference    : SG {ref.spacegroup}  {ref.label()}  objective {r['objective']:.6g}")
    print(f"\n  Topological types, class A (coordination sequence to depth {args.depth}):")
    count = Counter(r["topo"].types)
    names = {1: "corner", 2: "edge"}
    for k in sorted(count, key=lambda k: r["topo"].labels[k]):
        link = r["topo"].linkage.get(k, {})
        ltxt = ("  shares " + ", ".join(f"{names.get(s, f'face({s})')} {v:g}" for s, v in link.items())
                if link else "")
        print(f"    {r['topo'].labels[k]:<18} x{count[k]:<3} CS {' '.join(map(str, k[1]))}{ltxt}")

    def opt(v, fmt):
        return format(v, fmt) if v is not None else "-".rjust(7)

    def ngrey(x):
        return sum(len(v) for v in x["grey"].values())

    print(f"  Reference grey-band contacts per cell: {ngrey(r)}")
    print(f"\n  {'candidate':<40} {'class':>5} {'objective':>10} {'vol %':>7} {'shape':>7} "
          f"{'bond %':>7} {'angle':>7} {'grey':>5}")
    report = []
    for c in cands:
        row = {"name": c["name"], "class": classes[c["topo"].signature],
               "objective": c["objective"]}
        vol = shape = bmax = amax = None
        if c["sg"] == r["sg"]:
            row["strain"] = bg.cell_strain(r["crystal"].lattice, c["crystal"].lattice)
            vol, shape = 100 * row["strain"]["volume"], row["strain"]["shape"]
        same_types = set(r["topo"].labels.values()) == set(c["topo"].labels.values())
        if same_types:
            row["tables"] = bg.compare_geometry(r["geo"], c["geo"])
            bmax = max((x[4] for x in row["tables"]["bonds"]), default=0.0)
            amax = max((x[4] for x in row["tables"]["angles"]), default=0.0)
        report.append((c, row))
        row["grey"] = {"-".join(sorted(k)): v for k, v in c["grey"].items()}
        print(f"  {Path(c['name']).name[:40]:<40} {row['class']:>5} {c['objective']:10.4g} {opt(vol, '+7.2f')} "
              f"{opt(shape, '7.4f')} {opt(bmax, '7.2f')} {opt(amax, '7.2f')} {ngrey(c):>5}")
    print("  vol %, shape: Lagrangian cell strain against the reference (same space group only)."
          "\n  bond %, angle: largest shift of a type's min/mean/max (% of the mean length; deg)."
          "\n  grey: contacts per cell inside a grey band.")

    def triple(t, fmt):
        return "/".join(format(x, fmt) for x in t[:3])

    for c, row in report:
        print(f"\n  --- {c['name']}  (class {row['class']}) ---")
        for pair, ds in sorted(row["grey"].items()):
            print(f"    grey-band {pair} contacts: {' '.join(f'{d:.3f}' for d in sorted(ds))} A")
        if row["class"] != "A":
            for line in bg.describe_difference(r["topo"], c["topo"]):
                print(f"    {line}")
        if "tables" not in row:
            continue
        if row["class"] != "A":
            print("    first-shell types match: geometry compared by type label")
        if "strain" in row:
            print("    cell strain, principal values: "
                  + " ".join(f"{100 * x:+.2f}%" for x in row["strain"]["principal"]))
        for cat, unit, fmt in (("bonds", "%", ".4f"), ("angles", "deg", ".2f")):
            rows = row["tables"][cat][:args.top]
            print(f"    {cat}, reference min/mean/max -> candidate (largest {len(rows)}):")
            for _, key, a, b, score in rows:
                print(f"      {key:<38} {triple(a, fmt)} -> {triple(b, fmt)}  {score:6.2f} {unit}")
        print("    polyhedra, reference min/mean/max -> candidate:")
        for q, key, a, b, _ in row["tables"]["polyhedra"]:
            fmt = ".2f" if q == "sigma2" else ".4f"
            print(f"      {key:<18} {q:<7} {triple(a, fmt)} -> {triple(b, fmt)}")

    if args.json:
        def plain(x):
            if isinstance(x, dict):
                return {k if isinstance(k, str) else "-".join(k): plain(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [plain(v) for v in x]
            return x
        out = {"reference": args.reference, "depth": args.depth,
               "types": {r["topo"].labels[k]: {"count": n, "cs": list(k[1])}
                         for k, n in count.items()},
               "candidates": [plain(row) for _, row in report]}
        Path(args.json).write_text(json.dumps(out, indent=1))
        print(f"\n  JSON written: {args.json}")


# ======================================================================
# suggest (reserved)
# ======================================================================

def _cmd_suggest(args) -> None:
    print("ERROR: 'suggest' is reserved. Generating constraints from a reference "
          "waits for docs/plan-objective-uniqueness.md (docs/adr/0013-cli-subcommands.md).")
    sys.exit(2)


# ======================================================================
# Entry point
# ======================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="mizuho",
        description="Crystal structure generation from composition, space group "
                    "and bonding constraints.")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser("run", help="predict structures from a YAML config")
    p.add_argument("input", help="YAML input file")
    p.add_argument("--output", default=None,
                   help="Output file (default: <comp>_rank1.vasp or .cif)")
    p.add_argument("--format", choices=("poscar", "cif"), default="poscar",
                   help="Output format for single-composition runs (default: poscar)")
    p.add_argument("--top", type=int, default=1,
                   help="Write the top-N best structures, one per bond-graph topology "
                        "class (default: 1)")
    p.add_argument("--verbose", action="store_true",
                   help="Print detailed error breakdown")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed, propagated to Phase 0, Phase 1 and Phase 2. "
                        "Without --seed the config's optimization.seed is used, "
                        "if any.")
    p.add_argument("--workers", "-j", type=int, default=None,
                   help="Parallel worker processes for Phase 2 candidate "
                        "optimisation (overrides parallel.candidate_workers).")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("check", help="evaluate the objective at a reference CIF (no optimisation)")
    p.add_argument("config", help="YAML config whose constraints are evaluated")
    p.add_argument("reference", help="reference structure (CIF)")
    p.set_defaults(func=_cmd_check)

    p = sub.add_parser("fit", help="optimise a reference CIF's topology and compare with it")
    p.add_argument("config", help="YAML config")
    p.add_argument("reference", help="reference structure (CIF)")
    p.add_argument("--budget", type=float, default=1.0,
                   help="Multiply n_restarts and cma_maxiter by this (the "
                        "global-minimum gate uses 4)")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--output", default=None, help="Write the fitted structure as CIF")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=_cmd_fit)

    p = sub.add_parser("candidates", help="run Phase 1 only and list its ranking")
    p.add_argument("config", help="YAML config with a fixed spacegroup")
    p.add_argument("--seeds", type=int, default=1,
                   help="Rank over this many consecutive seeds and report the spread")
    p.add_argument("--seed", type=int, default=None, help="First seed (default: config's)")
    p.add_argument("--top", type=int, default=50, help="Rows to list (default: 50)")
    p.add_argument("--find", default=None,
                   help="Report only this assignment's class, e.g. 'Mg@a Al@d O@e'")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=_cmd_candidates)

    p = sub.add_parser("wyckoff", help="list a space group's Wyckoff positions in mizuho's setting")
    p.add_argument("spacegroup", type=int)
    p.set_defaults(func=_cmd_wyckoff)

    p = sub.add_parser("compare", help="bond-graph topology class and geometry of candidates "
                                       "against a reference (diagnostic, ADR-0018)")
    p.add_argument("config", help="YAML config: its coordination cutoffs define the bonds")
    p.add_argument("reference", help="reference structure (CIF)")
    p.add_argument("candidates", nargs="+",
                   help="candidate structures: CIF, or a JSON record with a param_vector "
                        "in the reference's assignment (tests/fixtures/impostors/*.json)")
    p.add_argument("--depth", type=int, default=10,
                   help="Coordination-sequence depth in bonds (default: 10)")
    p.add_argument("--top", type=int, default=6, help="Rows per geometry table (default: 6)")
    p.add_argument("--json", default=None, help="Also write the comparison as JSON")
    p.set_defaults(func=_cmd_compare)

    p = sub.add_parser("suggest", help="(reserved) generate constraints from a reference")
    p.add_argument("reference")
    p.set_defaults(func=_cmd_suggest)

    args = parser.parse_args(argv)
    args.func(args)
