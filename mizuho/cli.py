"""Command-line entry point."""

import argparse
import sys
from pathlib import Path

import yaml

from .composition import _all_charges_zero, parse_composition
from .data import WYCKOFF_DB
from .io.poscar import write_poscar
from .io.report import print_summary
from .lattice import CRYSTAL_SYSTEM_OF_SG, get_lattice_system
from .optimize.driver import optimize_one_assignment
from .pipeline import (
    _optimize_candidates_parallel, _run_auto_composition_mode,
    _run_composition_pool, build_sg_config, search_and_rank,
)
from .search import auto_search_spacegroups
from .structure import assignments_to_sites, resolve_fixed_assignment


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


def main():
    parser = argparse.ArgumentParser(
        description="Crystal structure predictor v6 - automatic space-group search.")
    parser.add_argument("input", help="YAML input file")
    parser.add_argument("--output", default=None,
                        help="Output POSCAR file (default: <comp>_rank1.vasp)")
    parser.add_argument("--top", type=int, default=1,
                        help="Write POSCAR for top-N best structures (default: 1)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed error breakdown")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility. Propagated "
                             "consistently to Phase 0 (space-group "
                             "auto-search), Phase 1, and Phase 2 (CMA-ES). "
                             "Without --seed, Phase 2 remains non-"
                             "deterministic (OS entropy) by default, "
                             "matching prior behaviour.")
    parser.add_argument("--workers", "-j", type=int, default=None,
                        help="Number of parallel worker processes for "
                             "Phase 2 candidate optimisation (default: "
                             "sequential, i.e. today's behaviour). "
                             "Overrides parallel.candidate_workers in the "
                             "YAML config.")
    args = parser.parse_args()

    # ----------------------------------------------------------------
    # Load config
    # ----------------------------------------------------------------
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found."); sys.exit(1)
    with open(input_path) as fh:
        config = yaml.safe_load(fh)

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
    mode      = config.get("mode")
    if mode is not None and mode not in ("single_spacegroup", "auto_spacegroup"):
        print(f"ERROR: mode must be 'single_spacegroup' or 'auto_spacegroup', "
              f"got {mode!r}."); sys.exit(1)
    if mode == "single_spacegroup" and sg_given is None:
        print("ERROR: mode: single_spacegroup requires 'spacegroup' to be set."); sys.exit(1)
    if mode == "auto_spacegroup" and sg_given is not None:
        print("ERROR: mode: auto_spacegroup requires 'spacegroup' to be "
              "omitted (it selects one automatically)."); sys.exit(1)
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
        print(f"  Space group  : AUTO (Phase 0 screening active)")
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

        # Phase 2: optimise each Wyckoff assignment
        workers = sg_config.get("parallel", {}).get("candidate_workers")
        if workers is not None and workers > 1 and len(valid_asgns) > 1:
            print(f"\n[Phase 2] SG {sg} - Optimising {len(valid_asgns)} "
                  f"candidate(s) in parallel ({workers} workers) ...")
            parallel_results = _optimize_candidates_parallel(
                valid_asgns, sg_config, opt_cfg, shell_seeds, workers)
            for (best_p, best_e, structure), asgn in zip(parallel_results, valid_asgns):
                all_results.append((best_e, sg, asgn, best_p, structure, sg_config))
        else:
            print(f"\n[Phase 2] SG {sg} - Optimising "
                  f"{len(valid_asgns)} candidate(s) ...")

            for k, asgn in enumerate(valid_asgns):
                label = " + ".join(
                    f"{a.element}@{a.entry.letter}" for a in asgn)
                print(f"\n  --- SG{sg} [{k+1}/{len(valid_asgns)}] {label} ---")

                best_p, best_e, structure = optimize_one_assignment(
                    asgn, sg_config, opt_cfg,
                    verbose=args.verbose, label=label,
                    shell_seed_info=shell_seeds[k])

                print(f"  Error = {best_e:.6f}")
                if structure is not None:
                    lp = structure["lattice_params"]
                    print(f"  Lattice: a={lp['a']:.4f} b={lp['b']:.4f} "
                          f"c={lp['c']:.4f} A")

                all_results.append((best_e, sg, asgn, best_p, structure, sg_config))

    if not all_results:
        print("\nNo valid structures found across all space groups."); sys.exit(0)

    # ----------------------------------------------------------------
    # Global ranking
    # ----------------------------------------------------------------
    all_results.sort(key=lambda x: x[0])

    print(f"\n{'='*65}")
    print("  GLOBAL RANKING")
    print(f"{'='*65}")
    print(f"  {'Rank':>4}  {'Error':>12}  {'SG':>5}  "
          f"{'Crystal system':>14}  Assignment")
    print(f"  {'-'*63}")
    for rank, (err, sg, asgn, _, _, _) in enumerate(all_results, 1):
        label = " + ".join(
            f"{a.element}@{a.entry.letter}({a.entry.multiplicity})"
            for a in asgn)
        cs = CRYSTAL_SYSTEM_OF_SG.get(sg, "?")
        print(f"  {rank:>4}  {err:>12.5f}  {sg:>5}  {cs:>14}  {label}")

    # ----------------------------------------------------------------
    # Output top-N structures
    # ----------------------------------------------------------------
    top_n = min(args.top, len(all_results))
    comp  = config["composition"].replace(" ", "_")

    for rank in range(top_n):
        err, sg, asgn, best_p, structure, sg_config = all_results[rank]
        if structure is None:
            print(f"\n  [Rank {rank+1}] No valid structure obtained.")
            continue

        win_cfg = dict(sg_config)
        win_cfg["wyckoff_sites"] = assignments_to_sites(asgn)

        print_summary(structure, asgn, win_cfg, err,
                      param_vector=best_p)

        out_path = (args.output if args.output and top_n == 1
                    else f"{comp}_rank{rank+1}.vasp")
        write_poscar(structure, win_cfg, out_path)

    print("\n  Done.\n")
