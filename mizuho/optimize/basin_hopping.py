"""Wrapper around scipy.optimize.basinhopping with the same multi-start
restart management and starting-point heuristics as
optimize.cma.multistart_cma_es, so it satisfies the identical
optimize_fn interface contract (see geometry.local_shell.solve_local_shell's
docstring) and is a drop-in alternative backend anywhere multistart_cma_es
is used today.

Basin hopping interleaves random "hops" with a LOCAL minimiser at each
accepted step -- this project already uses exactly that local minimiser
(L-BFGS-B) as its own Stage-2 refinement in optimize.driver, so this
backend effectively repeats that same local-refinement machinery many
times per run instead of once, rather than introducing an unfamiliar
new local-search method.
"""

import numpy as np
from scipy.optimize import basinhopping

from .._parallel_utils import run_parallel_tasks
from .init_guess import (
    _apply_lattice_gradient_hint, _estimate_lattice_x0, _smart_init_x0,
)


def _run_one_restart_bh(task):
    """Module-level (picklable) worker for multistart_basin_hopping's
    Pass-2 restart dispatch (phase H-4)."""
    objective, x0, maxiter, stepsize, T, minimizer_kwargs, run_seed = task
    result = basinhopping(
        objective,
        x0,
        niter=maxiter,
        stepsize=stepsize,
        T=T,
        minimizer_kwargs=minimizer_kwargs,
        seed=run_seed,
        disp=False,
    )
    return np.asarray(result.x, dtype=float), float(result.fun)


def multistart_basin_hopping(
        objective,
        bounds,
        n_restarts=8,
        maxiter=400,
        seed=None,
        verbose=False,
        config=None,
        periodic_mask=None,
        seeded_x0=None,
        stepsize=0.5,
        T=1.0,
        plateau_patience=None,
        plateau_rel_tol=1e-3,
        restart_workers=1,
        **_ignored,
):
    """
    Run scipy's basinhopping from n_restarts independent starting
    points and return the globally best result.

    Restart-slot conventions mirror optimize.cma.multistart_cma_es
    exactly (see that function's docstring): restart 0 uses a
    volume-aware lattice estimate when `config` is given, restart 1 the
    box centre, restart 2 uses `seeded_x0` as-is when provided
    (falling through to a random restart otherwise), and restarts
    beyond that draw uniformly at random -- with `smart_init`/lattice-
    gradient-hint nudges applied to every non-seeded restart, reusing
    the SAME heuristics multistart_cma_es uses so the two backends'
    starting conditions are genuinely comparable.

    Each restart's own internal local minimiser is L-BFGS-B, bounded by
    `bounds` (`minimizer_kwargs={"method": "L-BFGS-B", "bounds": bounds}`)
    -- identical to optimize.driver.optimize_one_assignment's own
    Stage-2 refinement step, so basin hopping's result is, by
    construction, always already locally polished; a caller that still
    runs a separate Stage-2 L-BFGS-B pass afterwards (the current
    default for every backend, kept for simplicity/uniformity) is
    performing a safe, improvement-gated no-op in the common case.

    Parameters
    ----------
    maxiter : forwarded to basinhopping's own `niter` (number of hops
        per restart).
    stepsize, T : forwarded to basinhopping as-is (random-displacement
        step size and Metropolis acceptance temperature); see its own
        documentation. Defaults match scipy's own.
    plateau_patience, plateau_rel_tol : accepted for interface
        compatibility with the shared optimize_fn contract but IGNORED
        here -- basinhopping has its own internal accept/reject dynamics
        and this backend does not implement a second, separate
        early-stopping mechanism on top of them.
    periodic_mask : NOT specially honoured here (accepted for interface
        compatibility only) -- basinhopping's random-displacement hop
        step has no native concept of a periodic dimension, and its
        follow-up L-BFGS-B local minimisation already keeps every
        accepted point within `bounds` regardless, so a hard box (the
        same one CMA-ES falls back to for non-periodic dimensions) is
        the correct behaviour here rather than an approximation of one.
    restart_workers : dispatch n_restarts' basinhopping calls across a
        process pool instead of one at a time (phase H-4). 1 (the
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

        if verbose:
            print(f"      [BH restart {restart+1}/{n_restarts}] {tag}")

        run_seed = None if seed is None else int(seed) + restart
        tasks.append((objective, x0, maxiter, stepsize, T, minimizer_kwargs, run_seed))

    # ---- Pass 2: run each restart's basinhopping, sequentially or
    # across a process pool (restart_workers) ----
    results = run_parallel_tasks(_run_one_restart_bh, tasks, max_workers=restart_workers)

    for restart, (x_r_raw, f_r) in enumerate(results):
        x_r = np.clip(x_r_raw, lo, hi)
        is_best = f_r < global_best_f
        if verbose:
            print(f"      [BH restart {restart+1}] error={f_r:.6f}"
                  + ("  *best*" if is_best else ""))

        if is_best:
            global_best_f = f_r
            global_best_x = x_r

    return global_best_x, global_best_f
