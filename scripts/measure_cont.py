"""The CONT primitive alone, ADR-0020 measurement 2.

Asks whether resuming the best point beats independent restarts at the same
number of objective evaluations, on one mineral with its published Wyckoff
assignment fixed (as in the global-minimum gate). Three strategies:

  independent  the current Stage 1: multistart_cma_es at the x4 budget
               (n_restarts and cma_maxiter both x4), early stopping off so
               that its evaluation count is the full budget E.
  cont_cold    the x1 Stage 1 (the first n_restarts restarts, same starting
               points), then the rest of E spent on fresh CMA-ES runs
               started at the best point so far with step SIGMA_CONT.
  cont_warm    the same x1 Stage 1, then the rest of E spent continuing the
               best restart's own CMA-ES state (mean, step size, covariance,
               paths). A run that stops on its own criteria is followed by a
               cont_cold run from the best point.

E is the evaluation count the `independent` run used for that seed. Every
strategy's end point goes through the same Stage 2 (driver._stage2, not
counted) and is judged as ADR-0020 says: Stage-1 convergence
(error <= published + 1e-3) and tier 1 (published topology class).
SIGMA_CONT is fixed here, before the measurement, at a fifth of the
default sigma0.

    python scripts/measure_cont.py diopside_camgsi2o6 --variant iii --seeds 42 43 44 45 46
"""

import argparse
import functools
import json
import sys
import time
from pathlib import Path

import cma as _cma
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ablate_terms  # noqa: E402  (also puts tests/integration on sys.path)

import mizuho.optimize.cma as _cma_mod  # noqa: E402
from mizuho.objective.energy import compute_error  # noqa: E402
from mizuho.optimize.driver import _resolve_backend, _stage2  # noqa: E402
from mizuho.optimize.endpoints import topology_key  # noqa: E402
from mizuho.optimize.init_guess import _coordination_shell_seed_x0  # noqa: E402
from mizuho.structure import build_bounds  # noqa: E402

STRATEGIES = ("independent", "cont_cold", "cont_warm")
SIGMA_CONT = 0.05
CONVERGED_MARGIN = 1e-3


class Budget(Exception):
    """Raised inside the objective once the evaluation budget is spent."""


class Counted:
    """The objective with an evaluation counter and an optional budget."""

    def __init__(self, fn, limit=None, target=None):
        self.fn, self.limit, self.n = fn, limit, 0
        self.target, self.first_hit = target, None

    def __call__(self, x):
        if self.limit is not None and self.n >= self.limit:
            raise Budget
        self.n += 1
        f = self.fn(x)
        if self.first_hit is None and self.target is not None and f <= self.target:
            self.first_hit = self.n
        return f


class _Frame:
    """Normalised coordinates of _cma_es: lattice clipped, coordinates wrapped."""

    def __init__(self, bounds, periodic):
        self.lo = np.array([b[0] for b in bounds])
        self.hi = np.array([b[1] for b in bounds])
        self.span = self.hi - self.lo + 1e-12
        self.periodic = periodic

    def to_norm(self, x):
        return (np.asarray(x) - self.lo) / self.span

    def to_orig(self, x_norm):
        x_norm = np.asarray(x_norm)
        w = np.where(self.periodic, np.mod(x_norm, 1.0), np.clip(x_norm, 0.0, 1.0))
        return w * self.span + self.lo


def _opts(maxiter, tol, seed):
    opts = {"maxiter": maxiter, "tolfun": tol, "tolx": tol, "verbose": -9, "verb_log": 0}
    if seed is not None:
        opts["seed"] = int(seed) if int(seed) != 0 else 2**32 - 1
    return opts


def _iterate(es, objective, frame, best_x, best_f, maxgen):
    """_cma_es's generation loop. Returns (best_x, best_f, stopped_by_es)."""
    gen = 0
    try:
        while not es.stop() and gen < maxgen:
            sols = es.ask()
            xs = [frame.to_orig(s) for s in sols]
            fs = [objective(x) for x in xs]
            es.tell(sols, fs)
            gen += 1
            i = int(np.argmin(fs))
            if fs[i] < best_f:
                best_x, best_f = xs[i], fs[i]
    except Budget:
        return best_x, best_f, False
    return best_x, best_f, bool(es.stop())


def _setup(name, variant, seed):
    cfg, pv_ref, _ = ablate_terms._config(name, variant)
    opt = dict(cfg.get("optimization", {}))
    bounds = build_bounds(cfg)
    n_wy = sum(len(s["free_params"]) for s in cfg["wyckoff_sites"])
    periodic = np.array([False] * (len(bounds) - n_wy) + [True] * n_wy)
    seeded_x0 = None
    if cfg.get("coordination_constraints") or cfg.get("angle_constraints"):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        seeded_x0 = _coordination_shell_seed_x0(
            (lo + hi) / 2.0, bounds, cfg, _resolve_backend(opt.get("backend", "cma")),
            seed=seed)
    return cfg, np.asarray(pv_ref), opt, bounds, periodic, seeded_x0


def _multistart(objective, bounds, periodic, cfg, opt, seed, seeded_x0, n_restarts, maxiter,
                record=None):
    """multistart_cma_es exactly as the driver calls it, without early
    stopping. With `record`, every restart's CMA-ES object is kept."""
    original = _cma_mod._cma_es

    def recording(objective, x0, bounds, sigma0, maxiter, tol, seed, verbose,
                  periodic_mask, plateau_patience, plateau_rel_tol):
        frame = _Frame(bounds, periodic_mask)
        es = _cma.CMAEvolutionStrategy(frame.to_norm(x0), sigma0, _opts(maxiter, tol, seed))
        bx = frame.to_orig(frame.to_norm(x0))
        bf = objective(bx)
        bx, bf, _ = _iterate(es, objective, frame, bx, bf, maxiter)
        record.append((es, bx, bf))
        return bx, bf

    if record is not None:
        _cma_mod._cma_es = recording
    try:
        return _cma_mod.multistart_cma_es(
            objective=objective, bounds=bounds, n_restarts=n_restarts,
            sigma0=opt.get("sigma0", 0.25), maxiter=maxiter, tol=opt.get("cma_tol", 1e-8),
            seed=seed, verbose=False, config=cfg, periodic_mask=periodic,
            seeded_x0=seeded_x0, plateau_patience=opt.get("plateau_patience"),
            plateau_rel_tol=opt.get("plateau_rel_tol", 1e-3), restart_workers=1,
            endpoint_ledger=None)
    finally:
        _cma_mod._cma_es = original


def _continue(strategy, objective, frame, record, tol, maxiter, seed):
    """Spend the rest of the budget from the best restart of `record`."""
    es, bx, bf = min(record, key=lambda r: r[2])
    k = 0
    try:
        if strategy == "cont_warm":
            es.opts.set({"maxiter": es.countiter + 10**9})
            bx, bf, _ = _iterate(es, objective, frame, bx, bf, 10**9)
        while True:
            k += 1
            run_seed = _cma_mod.restart_seed(seed, 10_000 + k)
            es = _cma.CMAEvolutionStrategy(frame.to_norm(bx), SIGMA_CONT,
                                           _opts(maxiter, tol, run_seed))
            bx, bf, stopped = _iterate(es, objective, frame, bx, bf, maxiter)
            if not stopped:
                break
    except Budget:
        pass
    return bx, bf, k


def run(name, variant, seed, budget=4):
    cfg, pv_ref, opt, bounds, periodic, seeded_x0 = _setup(name, variant, seed)
    obj = functools.partial(compute_error, config=cfg)
    n1, it1 = int(opt.get("n_restarts", 8)), int(opt.get("cma_maxiter", 300))
    tol = opt.get("cma_tol", 1e-8)
    frame = _Frame(bounds, periodic)
    err_ref = compute_error(pv_ref, cfg)

    out = {}
    target = err_ref + CONVERGED_MARGIN
    counted = Counted(obj, target=target)
    t0 = time.time()
    x_ind, _ = _multistart(counted, bounds, periodic, cfg, opt, seed, seeded_x0,
                           n1 * budget, it1 * budget)
    E = counted.n
    out["independent"] = (x_ind, counted.n, time.time() - t0, 0, counted.first_hit)

    for strategy in ("cont_cold", "cont_warm"):
        counted = Counted(obj, limit=E, target=target)
        record = []
        t0 = time.time()
        _multistart(counted, bounds, periodic, cfg, opt, seed, seeded_x0, n1, it1,
                    record=record)
        bx, _, k = _continue(strategy, counted, frame, record, tol, it1 * budget, seed)
        out[strategy] = (bx, counted.n, time.time() - t0, k, counted.first_hit)

    rows = {}
    key_ref = topology_key(pv_ref, cfg)
    for strategy, (x1, n_eval, secs, k, first_hit) in out.items():
        f1 = obj(x1)
        p, e = _stage2(obj, cfg, x1, f1, bounds, opt.get("refine_methods", ["L-BFGS-B"]),
                       opt.get("refine_maxiter", 1000), opt.get("refine_ftol", 1e-10),
                       False, False, name)
        rows[strategy] = {
            "evaluations": n_eval,
            "seconds": secs,
            "continuation_runs": k,
            # Evaluations until Stage 1 first reached published + 1e-3.
            "evaluations_to_converge": first_hit,
            "stage1_error": float(f1),
            "error_found": float(e),
            "stage1_converged": bool(e <= err_ref + CONVERGED_MARGIN),
            "tier1_same_class": topology_key(np.asarray(p), cfg) == key_ref,
            "param_vector": [float(v) for v in p],
        }
    return {"mineral": name, "variant": variant, "seed": seed, "budget_evaluations": E,
            "error_published": float(err_ref), "sigma_cont": SIGMA_CONT, "strategies": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mineral")
    ap.add_argument("--variant", default="iii", choices=ablate_terms.VARIANTS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--out", default="gate-results/cont")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        rec = run(args.mineral, args.variant, seed, args.budget)
        (out / f"{args.mineral}_{args.variant}_seed{seed}.json").write_text(
            json.dumps(rec, indent=2))
        summary = {s: {k: v for k, v in r.items() if k != "param_vector"}
                   for s, r in rec["strategies"].items()}
        print(json.dumps({"seed": seed, "E": rec["budget_evaluations"],
                          "published": rec["error_published"], **summary}, indent=2))


if __name__ == "__main__":
    main()
