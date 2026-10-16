"""Phase 1 feasibility-solve budget ladder, ADR-0020 measurement 3.

ADR-0012's remedy for a large rank spread is a stronger feasibility solve:
more restarts per local-shell solve. Every Phase 1 solve goes through
geometry.local_shell.solve_local_shell, so `--factor F` multiplies the
n_restarts its callers pass by F (generations and the dimension scaling are
unchanged).

Two modes:

  asO4   The separation the two xfails in tests/unit/test_geometry.py
         measure, at their own budgets times F: the known-feasible AsO4
         shell must score below the known-infeasible one (solve_local_shell,
         seeds 42-44), and rank ahead of it in the staged ranking
         (rank_by_feasibility_staged, seeds 42-43). Only the RELATIVE
         assertion is used: the first test's absolute thresholds (2.5, 3.5)
         are raw-unit values that predate ADR-0009 and cannot hold in
         normalised units at any budget.
  rank   The rank gate of tests/integration/test_search_gates.py for one
         mineral at factor F over the given seeds: rank of the published
         equivalence class, and Phase 1 time.

    python scripts/phase1_ladder.py asO4 --factor 2
    python scripts/phase1_ladder.py rank diopside_camgsi2o6 --factor 2
"""

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests" / "integration"))
sys.path.insert(0, str(ROOT / "tests" / "unit"))

import mizuho.geometry.local_shell as _ls  # noqa: E402

_ORIGINAL_SOLVE = _ls.solve_local_shell


def set_factor(factor: int) -> None:
    """Multiply every solve_local_shell caller's n_restarts by `factor`."""

    def solve(mini_config, optimize_fn, n_restarts=12, *args, **kwargs):
        return _ORIGINAL_SOLVE(mini_config, optimize_fn, n_restarts * factor, *args, **kwargs)

    _ls.solve_local_shell = solve if factor != 1 else _ORIGINAL_SOLVE


def as_o4(factor: int) -> dict:
    import test_geometry as tg
    from mizuho.optimize.cma import multistart_cma_es

    set_factor(1)  # budgets below are passed explicitly, times factor
    feas = tg._mini_config_for(tg._AS_O4_FEASIBLE_ASSIGNMENT)
    infeas = tg._mini_config_for(tg._AS_O4_INFEASIBLE_ASSIGNMENT)
    shell = []
    for seed in (42, 43, 44):
        _, fe = _ORIGINAL_SOLVE(feas, multistart_cma_es, n_restarts=20 * factor,
                                maxiter=400, seed=seed)
        _, ie = _ORIGINAL_SOLVE(infeas, multistart_cma_es, n_restarts=20 * factor,
                                maxiter=400, seed=seed)
        shell.append({"seed": seed, "feasible": fe, "infeasible": ie, "separated": bool(fe < ie)})

    feasible = tg._aszno_assignment("g", ["g", "g", "h"], "e", "a")
    infeasible = tg._aszno_assignment("g", ["h", "h"], "e", "a")
    noise = [tg._aszno_assignment("g", ["g", "g", "h"], "e", l) for l in ("b", "c", "d")] * 5
    staged = []
    for seed in (42, 43):
        ranked = _ls.rank_by_feasibility_staged(
            [feasible, infeasible] + noise, tg._SG58_ASZNO_COUPLED, multistart_cma_es,
            stage1_shortlist=4, stage1_n_restarts=6 * factor, stage1_maxiter=150,
            stage2_n_restarts=15 * factor, stage2_maxiter=350, seed=seed)
        order = [a for a, _, _ in ranked]
        staged.append({"seed": seed,
                       "feasible_rank": order.index(feasible) + 1,
                       "infeasible_rank": order.index(infeasible) + 1,
                       "separated": order.index(feasible) < order.index(infeasible)})
    return {"factor": factor, "solve_local_shell": shell, "staged": staged,
            "all_separated": all(r["separated"] for r in shell + staged)}


def rank(name: str, factor: int, seeds) -> dict:
    from test_reference_structure_gate import _REFS, FIXTURES
    from test_search_gates import _located_key
    from mizuho.pipeline import build_sg_config, search_and_rank

    set_factor(factor)
    ref = _REFS[name]
    base = yaml.safe_load(open(ROOT / FIXTURES / f"{ref.fixture}.yaml"))
    _, _, asg = ref.build()
    located = {a.element for a in asg}
    target = _located_key(ref.sg, asg, located)
    rows = []
    for seed in seeds:
        cfg = build_sg_config(base, ref.sg)
        cfg["optimization"] = dict(cfg.get("optimization", {}), seed=seed)
        t0 = time.time()
        asgns, n_found, _ = search_and_rank(cfg, max_cands=10**9)
        keys = [_located_key(ref.sg, a, located) for a in asgns]
        rows.append({"seed": seed, "seconds": time.time() - t0, "classes": len(keys),
                     "enumerated": n_found,
                     "rank": keys.index(target) + 1 if target in keys else None})
    ranks = [r["rank"] for r in rows]
    return {"mineral": name, "factor": factor, "seeds": rows,
            "worst": None if None in ranks else max(ranks),
            "max_candidates": base.get("tree_search", {}).get("max_candidates", 50)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("asO4", "rank"))
    ap.add_argument("mineral", nargs="?")
    ap.add_argument("--factor", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--out", default="gate-results/phase1_ladder")
    args = ap.parse_args()
    if args.mode == "asO4":
        rec, stem = as_o4(args.factor), f"asO4_x{args.factor}"
    else:
        if not args.mineral:
            ap.error("rank needs a mineral")
        rec, stem = rank(args.mineral, args.factor, args.seeds), f"rank_{args.mineral}_x{args.factor}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}.json").write_text(json.dumps(rec, indent=2, default=float))
    print(json.dumps(rec, indent=2, default=float))


if __name__ == "__main__":
    main()
