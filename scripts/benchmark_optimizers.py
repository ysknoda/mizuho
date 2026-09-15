#!/usr/bin/env python3
"""Benchmark optimizer backends across mizuho's fixture suite -- plan
phase G-2 (see the plan file's "フェーズG" section).

For each fixture:
  1. Run Phase 1's search_and_rank ONCE, shared across all backends.
     Its own ranking always uses multistart_cma_es internally
     (pipeline.py hard-imports it for the geometric-feasibility probe
     regardless of optimization.backend), so the candidate SET and
     ORDER are identical no matter which backend Phase 2 will use --
     a controlled variable, letting this benchmark isolate the
     optimizer's own contribution rather than conflating it with
     search/ranking differences.
  2. Optimize the TOP-1 ranked candidate with each backend in turn,
     using that fixture's own `optimization:` YAML budget as-is (a
     realistic, as-actually-used comparison, not an artificially
     uniform one -- nacl's tiny budget stays tiny, adamite's larger
     one stays larger).
  3. Record wall-clock time and best_error per (fixture, backend).

Reports a summary table to stdout and optionally writes the raw rows
as JSON (--out) for later analysis.

BACKENDS always includes the four zero-dependency backends (cma,
differential_evolution, basin_hopping, dual_annealing); tpe/gp/mcmc are
appended only when their optional package is actually importable (same
*_AVAILABLE flags optimize.driver's own dispatch table uses), so this
script degrades gracefully rather than crashing partway through a
multi-hour run in an environment missing one of them.

Note: basin_hopping's own niter maps to the SAME `cma_maxiter` budget
key cma/differential_evolution use, but each basin-hopping "iteration"
runs a full L-BFGS-B local minimisation -- for large/high-dimensional
fixtures (adamite, topaz) this backend can take substantially longer
than the other backends at the same nominal budget: measured (task #75,
re-verified after this script was first interrupted) at a consistent
~13-28x slowdown vs cma across fixtures once environmental noise (this
script had earlier recorded a since-unreproduced 138x outlier on
spinel_mgal2o4, most likely caused by system load during that
particular multi-hour run rather than a genuine algorithmic issue) is
controlled for. Expect this run to take from minutes (small fixtures)
to potentially an hour or more overall (adamite/topaz, especially
basin_hopping/mcmc) -- see the printed per-row timing to track
progress.

Usage:
    python3 scripts/benchmark_optimizers.py
    python3 scripts/benchmark_optimizers.py --fixtures nacl,cscl,diamond_si
    python3 scripts/benchmark_optimizers.py --out results.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from mizuho.optimize.driver import optimize_one_assignment
from mizuho.optimize.mcmc import EMCEE_AVAILABLE
from mizuho.optimize.optuna_backend import GP_AVAILABLE, OPTUNA_AVAILABLE
from mizuho.pipeline import build_sg_config, search_and_rank

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
BACKENDS = ["cma", "differential_evolution", "basin_hopping", "dual_annealing"]
if OPTUNA_AVAILABLE:
    BACKENDS.append("tpe")
if GP_AVAILABLE:
    BACKENDS.append("gp")
if EMCEE_AVAILABLE:
    BACKENDS.append("mcmc")
ALL_FIXTURES = [
    "nacl", "cscl", "diamond_si", "rutile_tio2", "ternary_ionic",
    "spinel_mgal2o4", "topaz", "adamite",
]


def _load_fixture(name):
    with open(FIXTURES_DIR / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def _is_non_degenerate(asgn):
    """Conservative candidate filter for benchmark purposes: every
    Wyckoff letter used exactly once across the WHOLE assignment. Some
    fixtures (nacl, cscl) have no coordination_constraints/angle_constraints
    at all, so search_and_rank's geometric ranking (which only kicks in
    for coupled groups derived from those) has nothing to rank on --
    its raw top-1 can be a physically degenerate same-site placement
    (e.g. Na@a + Cl@a) that every backend correctly converges to the
    same uninformative bad plateau on, telling us nothing about
    backend performance. "All letters distinct" is stricter than the
    true physical criterion (see search/base.py's
    _shape_group_sizes/_fixed_representative_counts -- a shared
    free-parameter letter can be fine), but that is acceptable here:
    this is a benchmark candidate filter, not a correctness gate."""
    letters = [a.entry.letter for a in asgn]
    return len(letters) == len(set(letters))


def benchmark_fixture(name, verbose=True):
    t_fixture = time.time()
    base_config = _load_fixture(name)
    sg_config = build_sg_config(base_config, base_config["spacegroup"])

    t0 = time.time()
    # Ask for more than 1 candidate so there is a non-degenerate one to
    # pick from (see _is_non_degenerate) -- still cheap, since
    # search_and_rank's own cost is dominated by the search/ranking
    # itself, not by how many of the already-ranked results get kept.
    valid_asgns, n_found, shell_seeds = search_and_rank(
        sg_config, max_cands=20, verbose=False)
    search_time = time.time() - t0
    if verbose:
        print(f"[{name}] search_and_rank: n_found={n_found}  "
              f"time={search_time:.2f}s", flush=True)

    if not valid_asgns:
        return [{"fixture": name, "backend": b, "n_found": n_found,
                  "search_time": search_time, "opt_time": None,
                  "error": None, "note": "no valid assignment found"}
                for b in BACKENDS]

    pick = next((i for i, a in enumerate(valid_asgns) if _is_non_degenerate(a)), 0)
    asgn = valid_asgns[pick]
    seed_info = shell_seeds[pick]
    if verbose and pick > 0:
        print(f"[{name}] top-{pick} candidate(s) were degenerate "
              f"(repeated Wyckoff letters); using rank {pick}", flush=True)
    opt_cfg_base = dict(sg_config.get("optimization", {}))

    rows = []
    for backend in BACKENDS:
        opt_cfg = dict(opt_cfg_base)
        opt_cfg["backend"] = backend
        t0 = time.time()
        _, best_e, structure = optimize_one_assignment(
            asgn, sg_config, opt_cfg, verbose=False, label=backend,
            shell_seed_info=seed_info)
        opt_time = time.time() - t0
        note = "" if structure is not None else "optimisation failed"
        if verbose:
            print(f"  [{name}] {backend:25s} error={best_e:.6f}  "
                  f"time={opt_time:.2f}s", flush=True)
        rows.append({"fixture": name, "backend": backend, "n_found": n_found,
                      "search_time": search_time, "opt_time": opt_time,
                      "error": best_e, "note": note})

    if verbose:
        print(f"[{name}] total: {time.time() - t_fixture:.2f}s", flush=True)
    return rows


def print_summary(all_rows):
    print()
    print(f"{'fixture':16s} {'backend':25s} {'search_s':>10s} "
          f"{'opt_s':>10s} {'error':>16s}")
    print("-" * 82)
    for r in all_rows:
        opt_s = f"{r['opt_time']:.2f}" if r["opt_time"] is not None else "N/A"
        err = f"{r['error']:.6f}" if r["error"] is not None else "N/A"
        print(f"{r['fixture']:16s} {r['backend']:25s} "
              f"{r['search_time']:10.2f} {opt_s:>10s} {err:>16s}"
              f"{'  ' + r['note'] if r['note'] else ''}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", default=",".join(ALL_FIXTURES),
                         help="Comma-separated fixture names (no .yaml)")
    parser.add_argument("--out", default=None, help="Write raw rows as JSON")
    args = parser.parse_args()
    fixtures = [f.strip() for f in args.fixtures.split(",") if f.strip()]

    t_start = time.time()
    all_rows = []
    for name in fixtures:
        print(f"=== {name} ===", flush=True)
        all_rows.extend(benchmark_fixture(name))

    print_summary(all_rows)
    print(f"\nTotal wall time: {time.time() - t_start:.2f}s")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(all_rows, f, indent=2)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
