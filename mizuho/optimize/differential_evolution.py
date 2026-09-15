"""Wrapper around scipy.optimize.differential_evolution with the same
multi-start restart management and starting-point heuristics as
optimize.cma.multistart_cma_es, so it satisfies the identical
optimize_fn interface contract (see geometry.local_shell.solve_local_shell's
docstring) and is a drop-in alternative backend everywhere multistart_cma_es
is used today -- both the geometry/ package's dependency-injected
optimize_fn seam and, via optimize.driver's optimization.backend config
key, Phase 2's full-structure optimisation.

Unlike CMA-ES, scipy's differential_evolution already operates directly
in the bounds' own physical units (no normalised-space wrapping needed),
and each restart is a complete, self-contained population run rather
than a manually-driven ask/tell loop -- so there is no analogue to
cma.py's separate single-run `_cma_es` helper here.
"""

import functools

import numpy as np
from scipy.optimize import differential_evolution

from .._parallel_utils import run_parallel_tasks
from .init_guess import (
    _apply_lattice_gradient_hint, _estimate_lattice_x0, _smart_init_x0,
)


def _periodic_wrapped_objective(x, objective, lo, span, periodic_mask):
    x = np.asarray(x, dtype=float)
    x_use = np.where(periodic_mask, lo + np.mod((x - lo) / span, 1.0) * span, x)
    return objective(x_use)


def _wrap_periodic(objective, lo, hi, periodic_mask):
    """Wrap `objective` so periodic dimensions are folded back into
    [lo, hi] via modular arithmetic before evaluation, mirroring
    cma.py's wrap_or_clip -- best-effort only: scipy's own bounds
    handling already keeps trial vectors inside [lo, hi] in the common
    case, so this wrap is frequently a no-op, but it costs nothing and
    protects against any transient out-of-bounds evaluation reaching
    the objective with a hard clip's "sticky wall" bias instead of a
    wrap. None/all-False periodic_mask returns `objective` unchanged.

    Returns a functools.partial, NOT a local closure (phase H-4): this
    wrapped objective is embedded in every restart's task tuple, and
    when restart_workers>1 those tuples cross a process-pool boundary,
    which a local closure cannot survive pickling."""
    if periodic_mask is None or not np.any(periodic_mask):
        return objective
    span = hi - lo + 1e-12
    return functools.partial(_periodic_wrapped_objective, objective=objective,
                              lo=lo, span=span, periodic_mask=periodic_mask)


def _run_one_restart_de(task):
    """Module-level (picklable) worker for multistart_differential_evolution's
    Pass-2 restart dispatch (phase H-4)."""
    (obj, bounds, maxiter, popsize, mutation, recombination,
     run_seed, x0_clipped) = task
    result = differential_evolution(
        obj,
        bounds=list(bounds),
        maxiter=maxiter,
        popsize=popsize,
        mutation=mutation,
        recombination=recombination,
        seed=run_seed,
        x0=x0_clipped,
        polish=False,
        disp=False,
    )
    return np.asarray(result.x, dtype=float), float(result.fun)


def multistart_differential_evolution(
        objective,
        bounds,
        n_restarts=8,
        maxiter=400,
        seed=None,
        verbose=False,
        config=None,
        periodic_mask=None,
        seeded_x0=None,
        popsize=15,
        mutation=(0.5, 1.0),
        recombination=0.7,
        plateau_patience=None,
        plateau_rel_tol=1e-3,
        restart_workers=1,
        **_ignored,
):
    """
    Run scipy's differential_evolution from n_restarts independent
    populations and return the globally best result.

    Restart-slot conventions mirror optimize.cma.multistart_cma_es
    exactly, so this function is interchangeable with it as an
    optimize_fn: restart 0 uses a volume-aware lattice estimate
    (_estimate_lattice_x0) when `config` is given, restart 1 the box
    centre, restart 2 uses `seeded_x0` as-is when provided (falling
    through to a random restart otherwise, identical to
    multistart_cma_es's own behaviour), and restarts beyond that draw
    uniformly at random. `smart_init`/lattice-gradient-hint nudges are
    applied to every non-seeded restart exactly as in multistart_cma_es
    -- these are x0-CONSTRUCTION heuristics, agnostic to which optimizer
    consumes the resulting starting point, so reusing them here (rather
    than duplicating a second copy of this logic) keeps the two
    backends' starting conditions genuinely comparable.

    seeded_x0 is passed to scipy's own `x0` parameter (added in SciPy
    1.9, biases the initial population without abandoning its
    diversity) rather than replacing the whole population -- DE, unlike
    CMA-ES's single evolving mean, is inherently population-based, so
    a single seed point should nudge, not replace, that population.

    Parameters
    ----------
    plateau_patience, plateau_rel_tol : accepted for interface
        compatibility with the shared optimize_fn contract (geometry.
        local_shell.solve_local_shell always passes these) but IGNORED
        here -- scipy's differential_evolution has its own internal
        convergence tolerance (`tol`, left at scipy's default) and this
        backend does not implement a second, separate early-stopping
        mechanism on top of it.
    popsize, mutation, recombination : forwarded to
        scipy.optimize.differential_evolution as-is; see its own
        documentation. Defaults match scipy's own.
    periodic_mask : see _wrap_periodic -- honoured best-effort via an
        objective-input wrap, since scipy's DE has no native concept of
        a periodic (wrap-around) dimension.
    restart_workers : dispatch n_restarts' differential_evolution calls
        across a process pool instead of one at a time (phase H-4). 1
        (the default) is fully sequential; x0 construction always runs
        sequentially regardless of this value, so every restart's x0
        is identical either way -- see optimize.cma.multistart_cma_es's
        own restart_workers docstring for the shared design and the
        cosmetic (results-only, not verbose-console-order) effect this
        has when verbose=True.

    Returns
    -------
    best_x : np.ndarray
    best_f : float
    """
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    obj = _wrap_periodic(objective, lo, hi, periodic_mask)

    global_best_x = (lo + hi) / 2.0
    global_best_f = 1e10

    has_smart = bool(config and config.get("smart_init"))

    # ---- Pass 1: x0 construction (always sequential, see
    # restart_workers' own docstring above) ----
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
            print(f"      [DE restart {restart+1}/{n_restarts}] {tag}")

        run_seed = None if seed is None else int(seed) + restart

        # scipy's own x0 validation rejects ANY component outside
        # bounds, even by a tiny floating-point margin -- unlike
        # multistart_cma_es's normalised-space to_norm/wrap_or_clip,
        # which always projects x0 into the box instead of erroring.
        # The x0-construction heuristics reused above (_smart_init_x0,
        # _apply_lattice_gradient_hint) were written for that more
        # forgiving contract and can legitimately place a value exactly
        # AT a bound (e.g. a gradient-hint step clipped to lo[i]).
        # scipy rescales via (x - midpoint) * (1/span) + 0.5 internally
        # (DifferentialEvolutionSolver._unscale_parameters), a
        # DIFFERENT formula from a plain (x-lo)/(hi-lo) -- its extra
        # rounding step can push a value exactly at lo/hi a hair past
        # 0.0/1.0 (confirmed empirically: x0[i] == lo[i] bit-for-bit
        # still raised "x0 lay outside the specified bounds"). A plain
        # np.clip(x0, lo, hi) is therefore NOT sufficient; pull in by a
        # small margin so the rescaled value has room before it would
        # need to cross 0.0/1.0.
        margin = (hi - lo) * 1e-9
        x0_clipped = np.clip(x0, lo + margin, hi - margin)

        # polish=False: Stage 2 (L-BFGS-B) in optimize.driver already
        # polishes the winner across ALL restarts; polishing every
        # individual restart's own winner here would be redundant work.
        tasks.append((obj, bounds, maxiter, popsize, mutation, recombination,
                      run_seed, x0_clipped))

    # ---- Pass 2: run each restart's differential_evolution,
    # sequentially or across a process pool (restart_workers) ----
    results = run_parallel_tasks(_run_one_restart_de, tasks, max_workers=restart_workers)

    for restart, (x_r, f_r) in enumerate(results):
        is_best = f_r < global_best_f
        if verbose:
            print(f"      [DE restart {restart+1}] error={f_r:.6f}"
                  + ("  *best*" if is_best else ""))

        if is_best:
            global_best_f = f_r
            global_best_x = x_r

    return global_best_x, global_best_f
