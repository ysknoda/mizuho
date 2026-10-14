"""Annealing and Monte Carlo against CMA-ES at equal evaluations.

Design review 2026-10-14, after ADR-0020's CONT measurement: do the
backends mizuho already has that accept uphill moves -- basin hopping
(Metropolis acceptance at temperature T) and dual annealing (generalised
simulated annealing) -- escape the discrete minima Stage 1 stops in
(adamite's Zn at CN 4, diopside's Ca at CN 6)?

For one mineral and seed, with the published assignment fixed: CMA-ES at the
x4 budget (early stopping off) sets the evaluation budget E; each other
backend gets E evaluations and keeps the best point it evaluated. Its
iteration count is unbounded, so its first restart runs as one long chain
until E is spent: the setting annealing and Metropolis sampling are meant
for. Every end point goes
through the same Stage 2 and is judged as ADR-0020 says: Stage-1 convergence
(error <= published + 1e-3) and tier 1 (published topology class).

Settings fixed before measuring: basin hopping T = 0.01, the depth of the
minima it has to leave (adamite 0.015, diopside 0.02), and stepsize 0.1 (the
default 0.5 would relocate fractional coordinates at random); dual annealing
with scipy's defaults.

    python scripts/compare_backends.py adamite --seeds 42 43
"""

import argparse
import functools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ablate_terms  # noqa: E402
from measure_cont import CONVERGED_MARGIN, _multistart, _setup  # noqa: E402

from mizuho.objective.energy import compute_error  # noqa: E402
from mizuho.optimize.basin_hopping import multistart_basin_hopping  # noqa: E402
from mizuho.optimize.driver import _stage2  # noqa: E402
from mizuho.optimize.dual_annealing import multistart_dual_annealing  # noqa: E402
from mizuho.optimize.endpoints import topology_key  # noqa: E402

BACKENDS = ("cma", "basin_hopping", "dual_annealing")
BH_T = 0.01
BH_STEPSIZE = 0.1


class Budget(Exception):
    pass


class Tracked:
    """The objective, counted, with a budget, remembering its best point (a
    backend stopped by the budget returns nothing of its own)."""

    def __init__(self, fn, limit=None, target=None):
        self.fn, self.limit, self.target = fn, limit, target
        self.n, self.first_hit = 0, None
        self.best_x, self.best_f = None, np.inf

    def __call__(self, x):
        if self.limit is not None and self.n >= self.limit:
            raise Budget
        self.n += 1
        f = self.fn(x)
        if f < self.best_f:
            self.best_x, self.best_f = np.array(x, dtype=float), f
        if self.first_hit is None and self.target is not None and f <= self.target:
            self.first_hit = self.n
        return f


def run(name, variant, seed, budget=4):
    cfg, pv_ref, opt, bounds, periodic, seeded_x0 = _setup(name, variant, seed)
    obj = functools.partial(compute_error, config=cfg)
    n1, it1 = int(opt.get("n_restarts", 8)), int(opt.get("cma_maxiter", 300))
    err_ref = compute_error(pv_ref, cfg)
    target = err_ref + CONVERGED_MARGIN
    common = dict(bounds=bounds, seed=seed, verbose=False, config=cfg,
                  periodic_mask=periodic, seeded_x0=seeded_x0, restart_workers=1,
                  endpoint_ledger=None)
    out = {}
    for backend in BACKENDS:
        tracked = Tracked(obj, limit=None if backend == "cma" else out["cma"]["evaluations"],
                          target=target)
        t0 = time.time()
        try:
            if backend == "cma":
                _multistart(tracked, bounds, periodic, cfg, opt, seed, seeded_x0,
                            n1 * budget, it1 * budget)
            elif backend == "basin_hopping":
                multistart_basin_hopping(tracked, n_restarts=n1, maxiter=10**6,
                                         stepsize=BH_STEPSIZE, T=BH_T, **common)
            else:
                multistart_dual_annealing(tracked, n_restarts=n1, maxiter=10**6, **common)
        except Budget:
            pass
        x1, f1 = tracked.best_x, tracked.best_f
        p, e = _stage2(obj, cfg, x1, f1, bounds, opt.get("refine_methods", ["L-BFGS-B"]),
                       opt.get("refine_maxiter", 1000), opt.get("refine_ftol", 1e-10),
                       False, False, name)
        out[backend] = {
            "evaluations": tracked.n,
            "seconds": time.time() - t0,
            "evaluations_to_converge": tracked.first_hit,
            "stage1_error": float(f1),
            "error_found": float(e),
            "stage1_converged": bool(e <= target),
            "tier1_same_class": topology_key(np.asarray(p), cfg) == topology_key(pv_ref, cfg),
            "param_vector": [float(v) for v in p],
        }
    return {"mineral": name, "variant": variant, "seed": seed,
            "error_published": float(err_ref), "bh_T": BH_T, "bh_stepsize": BH_STEPSIZE,
            "backends": out}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mineral")
    ap.add_argument("--variant", default="iii", choices=ablate_terms.VARIANTS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--out", default="gate-results/backends")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        rec = run(args.mineral, args.variant, seed, args.budget)
        (out / f"{args.mineral}_{args.variant}_seed{seed}.json").write_text(json.dumps(rec, indent=2))
        print(json.dumps({b: {k: v for k, v in r.items() if k != "param_vector"}
                          for b, r in rec["backends"].items()}, indent=2))


if __name__ == "__main__":
    main()
