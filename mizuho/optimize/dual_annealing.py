"""Wrapper around scipy.optimize.dual_annealing (Generalized Simulated
Annealing, Xiang et al. 1997) with the same multi-start restart
management and starting-point heuristics as optimize.cma.
multistart_cma_es, so it satisfies the identical optimize_fn interface
contract (see geometry.local_shell.solve_local_shell's docstring) and is
a drop-in alternative backend anywhere multistart_cma_es is used today.

Confirmed by reading scipy's own source (scipy.optimize._dual_annealing)
before writing this wrapper, rather than assumed:

- Unlike differential_evolution, dual_annealing does NOT strictly
  validate x0 against bounds (EnergyState.reset only checks the
  resulting objective value is finite, retrying from a random point
  otherwise) -- so this backend does not need differential_evolution's
  epsilon-margin bounds-clipping workaround. A plain np.clip is applied
  anyway, defensively and at zero cost.
- VisitingDistribution.visiting already wraps every proposed step back
  into [lo, hi] via modular arithmetic (np.fmod), for EVERY dimension,
  unconditionally -- dual_annealing's own search mechanics are natively
  periodic/toroidal. This means periodic_mask needs no extra handling
  here (unlike differential_evolution, which has no native concept of a
  periodic dimension and needs an external wrap): dual_annealing already
  treats every dimension this way by construction, whether or not it is
  one of ours.
- Like basin hopping, each accepted step is followed by a bounded local
  minimiser (`minimizer_kwargs`, defaulting to the same "L-BFGS-B" +
  `bounds` this project's own Stage-2 refinement already uses) unless
  `no_local_search=True` -- left at its default (False) here, so a
  dual_annealing result is, by construction, already locally polished
  the same way basin hopping's is.
"""

import numpy as np
from scipy.optimize import dual_annealing

from .._parallel_utils import run_parallel_tasks
from .init_guess import (
    _apply_lattice_gradient_hint, _estimate_lattice_x0, _smart_init_x0,
)


def _run_one_restart_da(task):
    """Module-level (picklable) worker for multistart_dual_annealing's
    Pass-2 restart dispatch (phase H-4)."""
    (objective, bounds, maxiter, initial_temp, visit, accept,
     minimizer_kwargs, x0, run_seed) = task
    result = dual_annealing(
        objective,
        bounds=list(bounds),
        maxiter=maxiter,
        initial_temp=initial_temp,
        visit=visit,
        accept=accept,
        minimizer_kwargs=minimizer_kwargs,
        x0=x0,
        seed=run_seed,
    )
    return np.asarray(result.x, dtype=float), float(result.fun)


def multistart_dual_annealing(
        objective,
        bounds,
        n_restarts=8,
        maxiter=400,
        seed=None,
        verbose=False,
        config=None,
        periodic_mask=None,
        seeded_x0=None,
        initial_temp=5230.0,
        visit=2.62,
        accept=-5.0,
        plateau_patience=None,
        plateau_rel_tol=1e-3,
        restart_workers=1,
        **_ignored,
):
    """
    Run scipy's dual_annealing from n_restarts independent starting
    points and return the globally best result.

    Restart-slot conventions mirror optimize.cma.multistart_cma_es
    exactly (see that function's docstring): restart 0 uses a
    volume-aware lattice estimate when `config` is given, restart 1 the
    box centre, restart 2 uses `seeded_x0` as-is when provided (falling
    through to a random restart otherwise), and restarts beyond that
    draw uniformly at random -- with `smart_init`/lattice-gradient-hint
    nudges applied to every non-seeded restart, reusing the SAME
    heuristics multistart_cma_es/multistart_basin_hopping use so every
    backend's starting conditions are genuinely comparable.

    Parameters
    ----------
    maxiter : forwarded to dual_annealing's own `maxiter` (global-search
        iteration budget per restart).
    initial_temp, visit, accept : forwarded to dual_annealing as-is
        (initial annealing temperature, visiting-distribution shape, and
        acceptance-distribution shape); see its own documentation.
        Defaults match scipy's own.
    plateau_patience, plateau_rel_tol : accepted for interface
        compatibility with the shared optimize_fn contract but IGNORED
        here -- dual_annealing has its own internal
        temperature-based re-annealing schedule and this backend does
        not implement a second, separate early-stopping mechanism on
        top of it.
    periodic_mask : NOT specially honoured here (accepted for interface
        compatibility only) -- see module docstring: dual_annealing's
        own visiting distribution already wraps every dimension
        modularly by construction, so there is nothing extra to do.
    restart_workers : dispatch n_restarts' dual_annealing calls across
        a process pool instead of one at a time (phase H-4). 1 (the
        default) is fully sequential; see optimize.cma.multistart_cma_es's
        own restart_workers docstring for the shared design.

    Returns
    -------
    best_x : np.ndarray
    best_f : float
    """
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)

    global_best_x = (lo + hi) / 2.0
    global_best_f = 1e10

    has_smart = bool(config and config.get("smart_init"))
    minimizer_kwargs = {"method": "L-BFGS-B", "bounds": list(bounds)}

    # ---- Pass 1: x0 construction (always sequential) ----
    tasks = []
    for restart in range(n_restarts):
        is_seeded_restart = (restart == 2 and seeded_x0 is not None)

        if restart == 0 and config is not None:
            x0 = _estimate_lattice_x0(config, bounds)
            tag = "volume-aware"
        elif restart == 1 or (restart == 0 and config is None):
            x0 = (lo + hi) / 2.0
            tag = "box centre"
        elif is_seeded_restart:
            x0 = np.array(seeded_x0, dtype=float)
            tag = "coordination-shell seed"
        else:
            x0 = rng.uniform(lo, hi)
            tag = "random"

        if has_smart and not is_seeded_restart:
            x0 = _smart_init_x0(x0, bounds, config, rng)
            tag += " + smart_init"

        if restart >= 2 and config is not None and not is_seeded_restart:
            x0 = _apply_lattice_gradient_hint(x0, bounds, config)
            tag += " + grad_hint"

        x0 = np.clip(x0, lo, hi)

        if verbose:
            print(f"      [DA restart {restart+1}/{n_restarts}] {tag}")

        run_seed = None if seed is None else int(seed) + restart
        tasks.append((objective, bounds, maxiter, initial_temp, visit, accept,
                      minimizer_kwargs, x0, run_seed))

    # ---- Pass 2: run each restart's dual_annealing, sequentially or
    # across a process pool (restart_workers) ----
    results = run_parallel_tasks(_run_one_restart_da, tasks, max_workers=restart_workers)

    for restart, (x_r_raw, f_r) in enumerate(results):
        x_r = np.clip(x_r_raw, lo, hi)
        is_best = f_r < global_best_f
        if verbose:
            print(f"      [DA restart {restart+1}] error={f_r:.6f}"
                  + ("  *best*" if is_best else ""))

        if is_best:
            global_best_f = f_r
            global_best_x = x_r

    return global_best_x, global_best_f
