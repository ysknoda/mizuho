"""Thin wrapper around the `cma` package (Hansen's reference CMA-ES
implementation) with periodic/box boundary handling and multi-start
restart management.

Replaces the original hand-rolled CMA-ES: the reference implementation
avoids the boundary-clipping distortion of naive box constraints and
the silent identity-covariance fallback the hand-rolled version used
on a Cholesky failure, while keeping this project's domain-specific
pieces (periodic wrap-around for free Wyckoff coordinates, the
volume-aware / smart_init / gradient-hint restart strategy) unchanged.
"""

import numpy as np
import cma as _cma

from .endpoints import run_restarts
from .init_guess import (
    _apply_lattice_gradient_hint, _estimate_lattice_x0, _smart_init_x0,
)


def restart_seed(seed, restart: int):
    """Seed for one restart of a multistart run, or None when unseeded.

    Derived through numpy's SeedSequence so restarts, and runs at
    consecutive seeds, draw independent streams. The former `seed +
    restart` made seed s's restart 1 replay seed s+1's restart 0, so the
    five consecutive seeds of ADR-0012's rank spread were not independent."""
    if seed is None:
        return None
    return int(np.random.SeedSequence([int(seed), int(restart)]).generate_state(1)[0])


def _cma_es(
        objective,
        x0,
        bounds,
        sigma0=0.3,
        maxiter=500,
        tol=1e-8,
        seed=None,
        verbose=False,
        periodic_mask=None,
        plateau_patience=None,
        plateau_rel_tol=1e-3,
):
    """
    Single-run CMA-ES minimiser with box/periodic bound handling.

    Parameters
    ----------
    objective : callable (n,) -> float
    x0        : initial mean, shape (n,)
    bounds    : list of (lo, hi) per dimension
    sigma0    : initial step-size in normalised [0,1] space (default 0.3)
    maxiter   : maximum number of generations
    tol       : cma's tolfun/tolx convergence tolerance
    seed      : random seed for reproducibility
    verbose   : print progress every 10% of maxiter
    periodic_mask : optional bool array, shape (n,). True marks a dimension
                    as periodic (e.g. a free Wyckoff fractional coordinate):
                    out-of-range proposals wrap around (mod 1) instead of
                    being clipped, so the search can reach true special
                    values at/near 0 or 1 without a "sticky wall" bias.
                    Non-periodic dimensions (e.g. lattice constants) are
                    still hard-clipped to their bounds. Defaults to all
                    dimensions non-periodic (pure clipping, prior behaviour).
    plateau_patience : optional int. When set, this restart stops early
                    once best_f's relative improvement over the last
                    plateau_patience generations falls below
                    plateau_rel_tol -- a PRACTICAL convergence check,
                    distinct from cma's own tolfun/tolx (`tol` above,
                    Hansen's reference stopping criteria): those are
                    tight enough (typically 1e-8) that they essentially
                    never fire for a physically-scaled objective whose
                    values range over 0.01-100+, so without this a
                    restart always burns its full maxiter budget even
                    long after it has, in practical terms, stopped
                    improving. None (the default) disables this and
                    preserves prior behaviour exactly.
    plateau_rel_tol : relative-improvement threshold used by
                    plateau_patience above, e.g. 1e-3 means "less than
                    0.1% improvement over the last plateau_patience
                    generations counts as a plateau". Ignored when
                    plateau_patience is None.

    Returns
    -------
    best_x : np.ndarray
    best_f : float
    """
    n = len(x0)
    if periodic_mask is None:
        periodic_mask = np.zeros(n, dtype=bool)

    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    span = hi - lo + 1e-12

    def to_norm(x):
        return (x - lo) / span

    def wrap_or_clip(x_norm):
        return np.where(periodic_mask, np.mod(x_norm, 1.0), np.clip(x_norm, 0.0, 1.0))

    def to_orig(x_norm):
        return wrap_or_clip(x_norm) * span + lo

    opts = {
        "maxiter": maxiter,
        "tolfun": tol,
        "tolx": tol,
        "verbose": -9,   # silence cma's own stdout logging
        "verb_log": 0,   # and its on-disk logging
    }
    if seed is not None:
        # pycma reads seed 0 (like None) as "seed from the clock", so a
        # caller's seed 0 -- e.g. multistart's `seed + restart` at restart 0
        # with seed 0 -- silently ran unseeded. Mapped to a fixed nonzero
        # value instead; any collision with a real caller seed of 2**32 - 1
        # is immaterial.
        opts["seed"] = int(seed) if int(seed) != 0 else 2**32 - 1

    es = _cma.CMAEvolutionStrategy(to_norm(x0), sigma0, opts)

    best_x = to_orig(to_norm(x0))
    best_f = objective(best_x)
    gen = 0
    plateau_ref_f = best_f
    plateau_ref_gen = 0

    while not es.stop() and gen < maxiter:
        solutions = es.ask()
        xs_orig = [to_orig(np.asarray(s)) for s in solutions]
        fs = [objective(x) for x in xs_orig]
        es.tell(solutions, fs)
        gen += 1

        idx_best = int(np.argmin(fs))
        if fs[idx_best] < best_f:
            best_f = fs[idx_best]
            best_x = xs_orig[idx_best]

        if plateau_patience and gen - plateau_ref_gen >= plateau_patience:
            denom = max(abs(plateau_ref_f), 1e-12)
            if (plateau_ref_f - best_f) / denom < plateau_rel_tol:
                if verbose:
                    print(f"      CMA-ES plateaued at gen {gen} "
                          f"(<{plateau_rel_tol:.1%} improvement over "
                          f"{plateau_patience} generations)")
                break
            plateau_ref_f, plateau_ref_gen = best_f, gen

        if verbose and gen % max(1, maxiter // 10) == 0:
            print(f"      CMA-ES gen {gen:4d}/{maxiter}  "
                  f"best={best_f:.6f}  sigma={es.sigma:.5f}")

    if verbose and es.stop():
        print(f"      CMA-ES converged at gen {gen} ({es.stop()})")

    return best_x, best_f


def _run_one_restart_cma(task):
    """Module-level (picklable) worker for multistart_cma_es's Pass-2
    restart dispatch (phase H-4). objective must itself be picklable
    when restart_workers>1 -- a functools.partial around a module-level
    function works, a local closure does not (see optimize.driver's
    own objective-construction comment)."""
    (objective, x0, bounds, sigma0, maxiter, tol, run_seed, periodic_mask,
     plateau_patience, plateau_rel_tol, verbose) = task
    return _cma_es(
        objective=objective, x0=x0, bounds=bounds, sigma0=sigma0,
        maxiter=maxiter, tol=tol, seed=run_seed, verbose=verbose,
        periodic_mask=periodic_mask, plateau_patience=plateau_patience,
        plateau_rel_tol=plateau_rel_tol)


def multistart_cma_es(
        objective,
        bounds,
        n_restarts=8,
        sigma0=0.25,
        maxiter=400,
        tol=1e-8,
        seed=None,
        verbose=False,
        config=None,
        periodic_mask=None,
        seeded_x0=None,
        plateau_patience=None,
        plateau_rel_tol=1e-3,
        restart_workers=1,
        endpoint_ledger=None,
        **_ignored,
):
    """
    Run CMA-ES from n_restarts independent random starting points and
    return the globally best result.

    Accepts and ignores unrecognised keyword arguments (**_ignored) so
    that a caller driving several interchangeable optimize_fn backends
    through one uniform call site (e.g. optimize.driver's
    optimization.backend dispatch) can pass every backend's own tuning
    keywords (e.g. differential_evolution's popsize) to all of them
    without each backend needing to special-case which kwargs apply to
    it -- mirrors the same defensive pattern optimize.differential_evolution
    and optimize.basin_hopping already use.

    Starting point strategy
    -----------------------
    Restart 0 : volume-aware estimate (_estimate_lattice_x0) + smart_init.
                Lattice constants are set to a physically motivated volume,
                and satellite atoms (e.g. H) are placed near their bonding
                partners (e.g. O) according to smart_init rules.

    Restart 1 : box centre + smart_init.

    Restart 2 : `seeded_x0` as-is, when given (see below) -- SKIPPED
                (falls through to a random restart, identical to prior
                behaviour) when `seeded_x0` is None, which is every
                caller that does not pass it and every config with no
                coordination_constraints/angle_constraints to seed from.

    Restarts 2+ (3+ when seeded_x0 was used): uniformly random +
                smart_init (applied when config has smart_init entries).

    smart_init
    ----------
    Defined in config["smart_init"]:
        - center:     O       # reference element (fixed side)
          satellite:  H       # element to be placed near center
          offset_min: 0.85    # A (optional, default from bond_constraint)
          offset_max: 1.10    # A (optional)

    Each satellite Wyckoff site is independently placed within
    [offset_min, offset_max] A of a distinct center atom.
    The remaining parameters (lattice, heavy atoms) are unchanged.
    NOT applied to the seeded restart -- see seeded_x0 below.

    Parameters
    ----------
    n_restarts : number of independent CMA-ES runs (default 8)
    sigma0     : initial step-size in normalised space (default 0.25)
    maxiter    : max generations per restart (default 400)
    config     : full config dict (optional; enables volume-aware init
                 and smart_init)
    seeded_x0  : optional pre-built starting point (see
                 optimize.init_guess._coordination_shell_seed_x0),
                 used as-is for restart 2 instead of a random draw.
                 Deliberately exempted from smart_init and the lattice
                 gradient hint below: both would overwrite or perturb
                 values a real local-shell solve already placed
                 deliberately, undoing the whole point of seeding.
    plateau_patience, plateau_rel_tol : forwarded to every restart's
                 _cma_es call as-is -- see that function's docstring.
                 None (the default) disables early stopping entirely,
                 identical to prior behaviour. geometry.local_shell.
                 solve_local_shell always sets these (its own mini-shell
                 feasibility solves); optimize.driver.optimize_one_assignment
                 (Phase 2's full-structure optimisation) also supports
                 them since phase G-8, via the opt_cfg keys of the same
                 name, but leaves them at None by default -- measured
                 (this project's own dev environment) at a real 2.0-2.3x
                 Stage-1 speedup with no change to the final error on
                 three fixtures, but not yet verified on larger/more
                 multimodal ones (topaz, adamite) where a short-term
                 plateau could plausibly be a weaker signal of true
                 convergence than on a smaller/simpler landscape, so it
                 stays opt-in there rather than becoming a new default.
    restart_workers : int, dispatch n_restarts' Pass-2 _cma_es calls
                 across a process pool instead of one at a time (phase
                 H-4). 1 (the default) is fully sequential -- x0
                 construction (Pass 1, below) always runs sequentially
                 regardless of this value, so the shared `rng`'s
                 consumption order (and therefore every restart's x0)
                 is identical either way; only whether the _cma_es
                 calls themselves run in parallel changes. NOTE: when
                 verbose=True, every restart's "[Restart N/M] tag" line
                 still prints during Pass 1 (restart order), and every
                 "error=..." result line prints during the Pass-2
                 reduce step (also restart order, since
                 run_parallel_tasks returns results in TASK order, not
                 completion order) -- but now as two separate batches
                 (all tags, then all results) rather than interleaved
                 tag/result/tag/result. Cosmetic console-output
                 reordering only; the RETURNED best_x/best_f are
                 unaffected by restart_workers.

    endpoint_ledger : optional optimize.endpoints.EndpointLedger. Every
                 restart's end point is handed to it in restart order,
                 and the remaining restarts are skipped once it says
                 stop (ADR-0019). None (the default) runs every
                 restart exactly as before.

    Returns
    -------
    best_x : np.ndarray
    best_f : float
    """
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    global_best_x = (lo + hi) / 2.0
    global_best_f = 1e10

    has_smart = bool(config and config.get("smart_init"))

    # ---- Pass 1: x0 construction (always sequential) ----
    # The shared `rng` is consumed in restart order regardless of
    # restart_workers, so every restart's x0 is bit-identical whether
    # Pass 2 below runs sequentially or in parallel.
    tasks = []
    for restart in range(n_restarts):
        is_seeded_restart = (restart == 2 and seeded_x0 is not None)

        # ---- Base initial point ----
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

        # ---- Apply smart_init to satellite atoms ----
        if has_smart and not is_seeded_restart:
            x0 = _smart_init_x0(x0, bounds, config, rng)
            tag += " + smart_init"

        # ---- Apply lattice gradient hint (restarts 2+, excluding the
        # seeded restart) ----
        # For restart 0 (volume-aware) and restart 1 (box centre) the
        # starting lattice is already set deliberately; gradient hint is
        # most useful for subsequent random restarts where the lattice
        # constants need to be steered away from bad regions. The seeded
        # restart's lattice values came from real local-shell solves,
        # not a heuristic guess, so this hint is skipped for it too.
        if restart >= 2 and config is not None and not is_seeded_restart:
            x0 = _apply_lattice_gradient_hint(x0, bounds, config)
            tag += " + grad_hint"

        if verbose:
            print(f"      [Restart {restart+1}/{n_restarts}] {tag}")

        run_seed = restart_seed(seed, restart)
        tasks.append((objective, x0, bounds, sigma0, maxiter, tol, run_seed,
                      periodic_mask, plateau_patience, plateau_rel_tol, verbose))

    # ---- Pass 2: run each restart's _cma_es, sequentially or across a
    # process pool (restart_workers) ----
    results = run_restarts(_run_one_restart_cma, tasks, restart_workers, endpoint_ledger)

    for restart, (x_r, f_r) in enumerate(results):
        is_best = f_r < global_best_f
        if verbose:
            print(f"      [Restart {restart+1}] error={f_r:.6f}"
                  + ("  *best*" if is_best else ""))

        if is_best:
            global_best_f = f_r
            global_best_x = x_r.copy()

    return global_best_x, global_best_f
