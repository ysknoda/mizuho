"""Two-stage numerical optimisation of lattice constants and free
Wyckoff coordinates for a single Wyckoff-site assignment.
"""

import functools
import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.optimize import minimize

from ..constraints import ideal_bond_config
from ..objective.energy import compute_error
from ..structure import assignments_to_sites, build_bounds, build_structure
from .basin_hopping import multistart_basin_hopping
from .cma import multistart_cma_es
from .differential_evolution import multistart_differential_evolution
from .dual_annealing import multistart_dual_annealing
from .endpoints import EndpointLedger, grouping_settings, topology_key
from .init_guess import _coordination_shell_seed_x0
from . import mcmc as _mcmc_module
from . import optuna_backend as _optuna_module
from ..objective import energy_jax as _energy_jax

logger = logging.getLogger(__name__)

# optimization.backend -> optimize_fn callable. Core backends have no
# extra dependencies and are always available. Optional backends
# (optuna's TPESampler/GPSampler, emcee) are registered ONLY when their
# package is actually importable -- the mizuho.optimize.mcmc/
# optuna_backend modules themselves are always safely importable
# regardless (each guards its own third-party import with try/except and
# exposes an *_AVAILABLE flag), so an environment without optuna/torch/
# emcee installed can still import mizuho.optimize normally as long as
# it never actually selects that backend. GPSampler needs more than
# plain optuna (also torch -- see optuna_backend's own module docstring
# for why), so it is gated on GP_AVAILABLE rather than OPTUNA_AVAILABLE;
# "bayesian" is an alias for "gp" (Gaussian-Process-based Bayesian
# optimisation is the literal/canonical meaning of the term -- TPE is
# also technically a Bayesian-optimisation method but is exposed under
# its own more specific name "tpe" to keep the two distinguishable).
# Centring (ADR-0028 (c)): the ideal-bond term's weight is a tie-break
# inside the zero-error region, and the error may rise by at most the
# tolerance (an order below Stage 1's 1e-3 convergence margin).
CENTRING_WEIGHT = 0.01
CENTRING_TOLERANCE = 1e-4

_CORE_BACKENDS = {
    "cma": multistart_cma_es,
    "differential_evolution": multistart_differential_evolution,
    "basin_hopping": multistart_basin_hopping,
    "dual_annealing": multistart_dual_annealing,
}
_OPTIONAL_BACKENDS = {}
if _optuna_module.OPTUNA_AVAILABLE:
    _OPTIONAL_BACKENDS["tpe"] = _optuna_module.multistart_optuna_tpe
if _optuna_module.GP_AVAILABLE:
    _OPTIONAL_BACKENDS["gp"] = _optuna_module.multistart_optuna_gp
    _OPTIONAL_BACKENDS["bayesian"] = _optuna_module.multistart_optuna_gp
if _mcmc_module.EMCEE_AVAILABLE:
    _OPTIONAL_BACKENDS["mcmc"] = _mcmc_module.multistart_mcmc


def _resolve_backend(name):
    """Resolve optimization.backend's string value to an optimize_fn
    callable. Raises a clear, actionable error for an unknown OR
    not-installed name rather than silently falling back to CMA-ES."""
    all_backends = {**_CORE_BACKENDS, **_OPTIONAL_BACKENDS}
    try:
        return all_backends[name]
    except KeyError:
        pass
    if name == "tpe" and not _optuna_module.OPTUNA_AVAILABLE:
        raise ImportError(
            "optimization.backend 'tpe' requires the optional 'optuna' "
            "package -- install it with: pip install mizuho[bayesian]")
    if name in ("gp", "bayesian") and not _optuna_module.GP_AVAILABLE:
        if not _optuna_module.OPTUNA_AVAILABLE:
            raise ImportError(
                f"optimization.backend {name!r} requires the optional "
                f"'optuna' (>=3.6) and 'torch' packages -- install them "
                f"with: pip install mizuho[gp]")
        raise ImportError(
            f"optimization.backend {name!r} requires the optional 'torch' "
            f"package (Optuna's GPSampler needs it for GP fitting and "
            f"acquisition-function optimisation) -- install it with: "
            f"pip install mizuho[gp]")
    if name == "mcmc" and not _mcmc_module.EMCEE_AVAILABLE:
        raise ImportError(
            f"optimization.backend {name!r} requires the optional 'emcee' "
            f"package -- install it with: pip install mizuho[mcmc]")
    raise ValueError(
        f"Unknown optimization.backend {name!r}; available backends: "
        f"{sorted(all_backends)}")


# scipy.optimize.minimize methods that accept a jac= argument (confirmed
# via scipy.optimize.show_options() for each -- same real-API-check
# discipline phase G-7 used for the ftol/fatol option-name split below).
# optimization.refine_jac's gradient is only ever passed to one of
# these; scipy accepts jac= for other methods too but merely warns and
# ignores it (RuntimeWarning, confirmed empirically), so restricting to
# this allowlist just avoids paying for a JAX gradient computation that
# would be silently discarded.
_JAC_CAPABLE_METHODS = {
    "L-BFGS-B", "TNC", "SLSQP", "CG", "BFGS", "Newton-CG",
    "dogleg", "trust-ncg", "trust-krylov", "trust-exact", "trust-constr",
}


def _refine_options(method, maxiter, ftol):
    """scipy.optimize.minimize's `options` dict keys are method-
    specific (confirmed via scipy.optimize.show_options() for each of
    the three methods refine_methods actually uses): L-BFGS-B and
    Powell both accept `ftol` directly with the same "relative error in
    fun(xopt)" meaning, but Nelder-Mead has no relative-tolerance option
    at all -- only the absolute `xatol`/`fatol` -- so `ftol` is mapped
    onto `fatol` there as the closest available analogue rather than
    silently dropped or passed through as an unrecognised option."""
    if method == "Nelder-Mead":
        return {"maxiter": maxiter, "fatol": ftol}
    return {"maxiter": maxiter, "ftol": ftol}


@dataclass
class AssignmentResult:
    """One topology class reached by one Wyckoff assignment (ADR-0019)."""
    params: np.ndarray
    error: float
    structure: dict
    topology: Optional[tuple]   # bond_graph signature; None without cutoffs
    n_endpoints: int            # Stage-1 endpoints that fell in this class
    n_zero: int                 # ...of which at or below the threshold
    converged: bool             # False: a sampling backend's fallback class
    n_stage1: int               # Stage-1 endpoints handed to the ledger
    stopped_early: bool


def optimize_assignment(
        assignments,
        config,
        opt_cfg,
        verbose=False,
        label="",
        shell_seed_info=None,
        max_classes=1,
):
    """
    Stage 1 and Stage 2 for one Wyckoff assignment, returning up to
    `max_classes` topology classes as AssignmentResult, lowest error first
    ([] if the optimisation failed). The best Stage-1 point is always the
    first class; further classes are the best Stage-1 endpoints of other
    topology classes (optimize.endpoints, ADR-0019), each refined by
    Stage 2. Stage 1 stops early once `early_stop_hits` endpoints at or
    below `zero_error_threshold` share one class (optimization.
    topology_grouping). With max_classes=1 the result is the same as
    optimize_one_assignment's.

    Optimise lattice constants and Wyckoff free coordinates for a single
    Wyckoff assignment using two stages:

    Stage 1 - Multi-start global search
        n_restarts independent runs of the selected `backend` (default
        "cma") from uniformly sampled starting points (first restart:
        box centre). CMA-ES adapts its covariance matrix to capture
        correlations between parameters, giving better convergence than
        axis-aligned samplers -- see optimize.cma's own module
        docstring, and optimize.differential_evolution/basin_hopping
        for the alternative backends' own tradeoffs. When the config
        has coordination_constraints/angle_constraints, one restart
        also seeds from _coordination_shell_seed_x0 -- see that
        function and multistart_cma_es's own seeded_x0 parameter (every
        backend shares this same restart-slot convention).

    Stage 2 - local refinement
        Polishes the best Stage-1 solution with one or more local
        minimisers (refine_methods, default just L-BFGS-B) for
        high-precision convergence. Applied uniformly regardless of
        `backend`: each method is gated on improvement (only accepted if
        it beats the current best), so for a backend that already ends
        each restart at a local optimum (basin hopping, dual_annealing)
        this is a safe, cheap no-op rather than a special case to code
        around.

    YAML optimisation block keys
    ----------------------------
    backend        : optimizer backend, one of "cma" (default),
                     "differential_evolution", "basin_hopping",
                     "dual_annealing" (scipy Generalized Simulated
                     Annealing, zero extra dependency),
                     "tpe" (Optuna TPESampler -- requires the optional
                     'optuna' package, pip install mizuho[bayesian]),
                     "gp"/"bayesian" (Optuna GPSampler, Gaussian-Process-
                     based Bayesian optimisation -- requires the optional
                     'optuna'(>=3.6) and 'torch' packages, pip install
                     mizuho[gp]),
                     "mcmc" (emcee ensemble sampler -- requires the
                     optional 'emcee' package, pip install mizuho[mcmc])
    n_restarts     : number of independent Stage-1 restarts for
                     cma/differential_evolution/basin_hopping/
                     dual_annealing; repurposed by tpe/gp/bayesian as its
                     n_startup_trials floor and by mcmc as an input to
                     its walker count -- neither of those two is a
                     restart-based algorithm, see their own modules'
                     docstrings                            (default 8)
    sigma0         : initial CMA-ES step-size -- CMA-ES only, ignored
                     by other backends           (default 0.25)
    cma_maxiter    : max Stage-1 iterations/restart, for ANY backend
                     (kept under its historical name for backward
                     compatibility with existing YAML configs -- it is
                     not CMA-ES-specific despite the name)
                                                  (default 400)
    cma_tol        : CMA-ES convergence tolerance -- CMA-ES only,
                     ignored by other backends    (default 1e-8)
    de_popsize, de_mutation, de_recombination : differential_evolution-only
                     tuning, forwarded as-is; see scipy.optimize.
                     differential_evolution's own docs for semantics
                     (defaults: 15, (0.5, 1.0), 0.7)
    bh_stepsize, bh_T : basin_hopping-only tuning, forwarded as-is; see
                     scipy.optimize.basinhopping's own docs
                     (defaults: 0.5, 1.0)
    da_initial_temp, da_visit, da_accept : dual_annealing-only tuning,
                     forwarded as-is; see scipy.optimize.dual_annealing's
                     own docs                (defaults: 5230.0, 2.62, -5.0)
    mcmc_temperature : mcmc-only; the T in p(x) ~ exp(-objective(x)/T).
                     None (default) auto-scales to the initial walkers'
                     own objective spread -- see optimize.mcmc's docstring
    gp_n_trials    : gp/bayesian-only; explicit Optuna trial-budget
                     override. None (default) uses min(n_restarts *
                     cma_maxiter, 200) -- GPSampler's per-trial cost (GP
                     refit + acquisition-function optimisation) makes
                     the much larger budgets tpe/other backends tolerate
                     impractically slow, see optimize.optuna_backend's
                     docstring                              (default None)
    seed           : random seed                     (default None)
    refine_methods : list of scipy.optimize.minimize method names run
                     in sequence for Stage 2, each accepted only if it
                     improves on the current best (same improvement-
                     gated pattern the original single-method step
                     always used -- see Stage 2's own code comment).
                     "Nelder-Mead"/"Powell" are derivative-free
                     alternatives to the default "L-BFGS-B" worth trying
                     because this project's objective has NO analytical
                     gradient anywhere (L-BFGS-B's is scipy's own
                     finite-difference approximation) and a few
                     documented non-smooth points (pairwise_distances_pbc's
                     27-image min() kinks, hard bond/angle-neighbour
                     cutoffs, angle calculation's own coarser PBC
                     handling) where that approximation can be
                     unreliable            (default ["L-BFGS-B"], i.e.
                     unchanged from every prior phase)
    refine_maxiter : max iterations, shared by every method in
                     refine_methods                    (default 1000)
    refine_ftol    : function-value convergence tolerance, shared by
                     every method in refine_methods (mapped onto
                     Nelder-Mead's `fatol` since it has no relative-
                     tolerance option of its own -- see _refine_options)
                                                        (default 1e-10)
    refine_jac     : supplies Stage 2 with an EXACT analytic gradient
                     for the coordination-number and volume penalty
                     terms (via JAX autodiff, see objective/energy_jax.py)
                     in place of scipy's default finite-difference
                     approximation, combined with a finite-difference
                     gradient for every other term (valid by linearity
                     of differentiation) -- only passed to methods in
                     refine_methods that accept jac= (_JAC_CAPABLE_METHODS);
                     ignored (jac=None, unchanged behaviour) for any
                     other method in that list, e.g. "Nelder-Mead"/
                     "Powell". Requires the optional 'jax' package
                     (raises ImportError otherwise) and is not
                     supported together with hybridisation_constraints
                     (raises ValueError -- see energy_jax's module
                     docstring for why that term is excluded).
                     MEASURED, NOT RECOMMENDED for general use: the
                     underlying gradient is exact (bit-identical final
                     error vs. the finite-difference default on every
                     real fixture tested), but
                     scripts/benchmark_jax_gradient.py's real Stage-2
                     comparison found it 2-8x SLOWER end to end on all
                     three fixtures tested (adamite/spinel_mgal2o4/
                     topaz), because Stage 1's CMA-ES already converges
                     close enough that Stage 2 typically needs only 0-1
                     L-BFGS-B iterations -- leaving no room to amortise
                     the JAX gradient's own one-time JIT-compilation
                     cost (~0.7-2s, measured) against per-iteration
                     savings. Kept available as an opt-in for problems
                     that may need many more Stage-2 iterations than
                     this project's own fixtures do, where that
                     calculus could differ -- see the
                     mizuho-py-wychoff-position plan's phase I-2 for
                     the full measured record; default False changes
                     nothing                            (default False)
    plateau_patience, plateau_rel_tol : cma-only (other backends accept
                     and ignore both via their own **_ignored); forwarded
                     to multistart_cma_es/_cma_es as-is -- stop a restart
                     early once its best_f's relative improvement over
                     the last plateau_patience generations falls below
                     plateau_rel_tol, instead of always running the full
                     cma_maxiter budget (see optimize.cma's own
                     docstring for the mechanism itself). None (the
                     default) disables this and preserves every prior
                     phase's behaviour exactly -- this is an opt-in, not
                     a new default, because although measured (this
                     project's own dev environment) at a real 2.0-2.3x
                     Stage-1 speedup with ZERO change to the final error
                     on three fixtures (rutile_tio2, spinel_mgal2o4,
                     ternary_ionic), that was not verified against the
                     larger/more multimodal fixtures (topaz, adamite)
                     where a short-term plateau could plausibly be a
                     weaker signal -- see solve_local_shell's own
                     _PLATEAU_PATIENCE/_PLATEAU_REL_TOL comment for the
                     analogous geometry-side reasoning that originally
                     motivated keeping this OFF here by default
                                                  (defaults: None, 1e-3)
    restart_workers : cma/differential_evolution/basin_hopping/
                     dual_annealing only (tpe/gp/mcmc accept and ignore
                     via their own **kwargs, since they do not use
                     n_restarts as an independent-restart loop -- see
                     G-3/G-4's own docstrings); forwarded to the
                     selected backend's own restart-dispatch. 1 (the
                     default) is fully sequential, unchanged from every
                     prior phase. >1 dispatches restarts across a
                     process pool (phase H-4) -- most useful when there
                     are FEW candidates but each is expensive (e.g.
                     wyckoff_assignment skips Phase 1 entirely, so
                     there is exactly one candidate), which is a
                     different scenario from parallel.candidate_workers
                     (phase H-1, useful when there are MANY candidates).
                     When this call is itself running inside a
                     candidate-level worker process (optimize.parallel's
                     optimize_assignments_parallel with max_workers>1),
                     restart_workers is forced back to 1 regardless of
                     what opt_cfg says, to avoid
                     candidate_workers x restart_workers processes
                     fighting over cores -- see optimize.parallel's own
                     module docstring for that safeguard.
                                                          (default: 1)

    topology_grouping : {zero_error_threshold, early_stop, early_stop_hits},
                     see optimize.endpoints (defaults: 1e-3, true, 2)

    `shell_seed_info`, if given, is the {group_index: best_param_vector}
    dict geometry.rank_by_feasibility already computed while ranking
    this assignment (see pipeline.search_and_rank, which threads it
    through) -- reused by _coordination_shell_seed_x0 instead of
    re-solving the same local shells. None (the default) makes
    _coordination_shell_seed_x0 solve fresh; harmless either way for a
    config with no coordination_constraints/angle_constraints, where
    that function is a no-op regardless.

    """
    cfg = dict(config)
    cfg["wyckoff_sites"] = assignments_to_sites(assignments)

    bounds = build_bounds(cfg)
    if not bounds:
        return []

    # Wyckoff free parameters (the trailing slice of the flat parameter
    # vector -- see build_bounds/build_structure) are periodic fractional
    # coordinates; lattice constants are not. Derived from the sites list
    # itself rather than a crystal-system lookup so it can never drift out
    # of sync with how build_bounds actually laid out the vector.
    n_wyckoff_params = sum(len(site["free_params"])
                           for site in cfg["wyckoff_sites"])
    n_lattice_params = len(bounds) - n_wyckoff_params
    periodic_mask = np.array(
        [False] * n_lattice_params + [True] * n_wyckoff_params)

    optimize_fn = _resolve_backend(opt_cfg.get("backend", "cma"))

    n_restarts  = opt_cfg.get("n_restarts",     8)
    sigma0      = opt_cfg.get("sigma0",         0.25)
    cma_maxiter = opt_cfg.get("cma_maxiter",    400)
    cma_tol     = opt_cfg.get("cma_tol",        1e-8)
    seed        = opt_cfg.get("seed",           None)
    refine_methods = opt_cfg.get("refine_methods", ["L-BFGS-B"])
    refine_max  = opt_cfg.get("refine_maxiter", 1000)
    refine_ftol = opt_cfg.get("refine_ftol",    1e-10)
    refine_jac  = opt_cfg.get("refine_jac",     False)
    plateau_patience = opt_cfg.get("plateau_patience", None)
    plateau_rel_tol  = opt_cfg.get("plateau_rel_tol", 1e-3)
    # Backend-specific tuning: each backend picks up only the keywords
    # it actually declares (multistart_cma_es takes sigma0/tol;
    # differential_evolution/basin_hopping ignore those two via their
    # own **_ignored and instead read de_*/bh_* below), so it is safe
    # to pass this whole union to every backend uniformly.
    de_popsize       = opt_cfg.get("de_popsize", 15)
    de_mutation      = opt_cfg.get("de_mutation", (0.5, 1.0))
    de_recombination = opt_cfg.get("de_recombination", 0.7)
    bh_stepsize       = opt_cfg.get("bh_stepsize", 0.5)
    bh_T              = opt_cfg.get("bh_T", 1.0)
    da_initial_temp   = opt_cfg.get("da_initial_temp", 5230.0)
    da_visit          = opt_cfg.get("da_visit", 2.62)
    da_accept         = opt_cfg.get("da_accept", -5.0)
    mcmc_temperature  = opt_cfg.get("mcmc_temperature", None)
    gp_n_trials       = opt_cfg.get("gp_n_trials", None)
    restart_workers   = opt_cfg.get("restart_workers", 1)

    seeded_x0 = None
    if cfg.get("coordination_constraints") or cfg.get("angle_constraints"):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        seeded_x0 = _coordination_shell_seed_x0(
            (lo + hi) / 2.0, bounds, cfg, optimize_fn,
            shell_seed_info=shell_seed_info, seed=seed)

    # functools.partial, NOT a local closure (`def obj(pv): return
    # compute_error(pv, cfg)`) -- phase H-4's restart-level parallelism
    # dispatches individual restarts, including this objective, across
    # a process pool, and a local closure can't be pickled across that
    # boundary (a module-level function bound via functools.partial
    # can). Candidate-level parallelism (H-1) never needed this: there,
    # the whole optimize_one_assignment call -- objective construction
    # included -- runs entirely inside one worker process, so `obj`
    # never itself has to cross a process boundary.
    obj = functools.partial(compute_error, config=cfg)

    # Validated here, BEFORE the broad try/except below, so a
    # misconfiguration (JAX not installed, or an unsupported
    # constraint type) is surfaced as a loud, actionable error rather
    # than silently swallowed into this function's usual "optimisation
    # failed" (None, 1e10, None) fallback -- that fallback is for
    # genuine runtime optimisation failures, not for "you asked for a
    # feature this config combination cannot support".
    if refine_jac and not _energy_jax.JAX_AVAILABLE:
        raise ImportError(
            "optimization.refine_jac requires the optional 'jax' "
            "package -- install it with: pip install mizuho[jax]")
    if refine_jac and cfg.get("hybridisation_constraints"):
        raise ValueError(
            "optimization.refine_jac is not supported together with "
            "hybridisation_constraints: this term's hard neighbour "
            "cutoff and 'ang > 150 deg -> skip' branch have not been "
            "validated for JAX differentiation (see phase I-2's design "
            "notes) -- remove hybridisation_constraints from this "
            "config or set refine_jac: false")

    # Centring inside the derived bond bands (ADR-0028 (c)): on by default,
    # a no-op for configs without derived bonds.
    centring_cfg = (ideal_bond_config(cfg)
                    if opt_cfg.get("centring", True) is not False else None)

    # Validated before the try below, like refine_jac: a bad
    # topology_grouping block is a configuration error, not a failed run.
    ledger = EndpointLedger(cfg, grouping_settings(opt_cfg))

    try:
        # Stage 1: multi-start global search (backend-selectable). Every
        # backend accepts (and ignores, via its own **kwargs) whichever
        # of these keywords it does not itself declare -- see
        # multistart_cma_es/multistart_differential_evolution/
        # multistart_basin_hopping's own docstrings.
        best_p, best_e = optimize_fn(
            objective     = obj,
            bounds        = bounds,
            n_restarts    = n_restarts,
            sigma0        = sigma0,
            maxiter       = cma_maxiter,
            tol           = cma_tol,
            seed          = seed,
            verbose       = verbose,
            config        = cfg,          # enables _estimate_lattice_x0
            periodic_mask = periodic_mask,
            seeded_x0     = seeded_x0,
            popsize       = de_popsize,
            mutation      = de_mutation,
            recombination = de_recombination,
            stepsize      = bh_stepsize,
            T             = bh_T,
            initial_temp  = da_initial_temp,
            visit         = da_visit,
            accept        = da_accept,
            mcmc_temperature = mcmc_temperature,
            gp_n_trials   = gp_n_trials,
            plateau_patience = plateau_patience,
            plateau_rel_tol  = plateau_rel_tol,
            restart_workers  = restart_workers,
            endpoint_ledger  = ledger,
        )
        if verbose:
            print(f"      Stage 1 ({opt_cfg.get('backend', 'cma')}) best error: {best_e:.6f}")
            if ledger.enabled:
                n_zero = sum(1 for c in ledger.classes.values() if c.n_zero)
                print(f"      Stage 1 endpoints: {ledger.n_added}, topology classes: "
                      f"{len(ledger.classes)} ({n_zero} at or below "
                      f"{ledger.threshold:g})"
                      + ("  -- stopped early" if ledger.stopped else ""))

        reps = [(best_p, best_e)]
        if max_classes > 1 and ledger.enabled:
            best_key = topology_key(best_p, cfg)
            for c in ledger.ranked():
                if len(reps) >= max_classes:
                    break
                if c.key != best_key:
                    reps.append((c.best_x, c.best_f))

        results = []
        for x0, f0 in reps:
            p, e = _stage2(obj, cfg, x0, f0, bounds, refine_methods, refine_max,
                           refine_ftol, refine_jac, verbose, label)
            if centring_cfg is not None:
                p, e = _centre(obj, cfg, centring_cfg, p, e, bounds, refine_max,
                               refine_ftol, verbose)
            key = topology_key(p, cfg) if ledger.enabled else None
            cls = ledger.classes.get(key) if key is not None else None
            results.append(AssignmentResult(
                params=p, error=e, structure=build_structure(p, cfg), topology=key,
                n_endpoints=cls.n_endpoints if cls else 0,
                n_zero=cls.n_zero if cls else 0,
                converged=cls.converged if cls else True,
                n_stage1=ledger.n_added, stopped_early=ledger.stopped))
        return _merge_classes(results)

    except Exception as exc:
        if verbose:
            import traceback
            traceback.print_exc()
            print(f"    [WARN] Optimisation failed for {label}: {exc}")
        return []


def _merge_classes(results: List[AssignmentResult]) -> List[AssignmentResult]:
    """Lowest error first, one entry per topology class: two refined
    points that ended in one class are one structure (ADR-0019). Entries
    without a class (no cutoffs) are never merged."""
    out, seen = [], set()
    for r in sorted(results, key=lambda r: r.error):
        if r.topology is not None and r.topology in seen:
            continue
        if r.topology is not None:
            seen.add(r.topology)
        out.append(r)
    return out


def _stage2(obj, cfg, best_p, best_e, bounds, refine_methods, refine_max,
            refine_ftol, refine_jac, verbose, label):
    """Stage 2: local refinement of one Stage-1 point."""
    # best_p (Stage 1's own result) is used as make_gradient_fn's
    # reference point -- it fixes which symmetry operations produce
    # each Wyckoff site's distinct orbit points (see that function's
    # own docstring for why this is safe: an ordinary optimiser
    # output is never exactly at a higher-symmetry special
    # position). Any unexpected internal failure here (as opposed
    # to the config-validation cases optimize_assignment checks first)
    # propagates to optimize_assignment's except-Exception
    # fallback, same as any other Stage-1/Stage-2 failure.
    jac_fn = _energy_jax.make_gradient_fn(cfg, best_p) if refine_jac else None

    # Stage 2: local refinement, one or more scipy.optimize.minimize
    # methods run in sequence (refine_methods, default just
    # "L-BFGS-B" -- unchanged behaviour unless a caller opts into
    # more). Each method's result is accepted only if it improves on
    # the CURRENT best (which may already be a previous method's
    # improvement within this same loop), so trying additional
    # methods can only help, never hurt, at the cost of one extra
    # cheap local-search call per additional method.
    for method in refine_methods:
        # jac is omitted entirely (not passed as jac=None) when
        # unused, rather than always included, so a refine_jac=False
        # call (the default) is a byte-for-byte identical minimize()
        # invocation to every prior phase -- including for callers
        # that monkeypatch minimize with a narrower signature not
        # expecting a jac keyword at all (see test_refine_methods.py,
        # predates phase I-2).
        kwargs = dict(method=method, bounds=bounds,
                      options=_refine_options(method, refine_max, refine_ftol))
        if jac_fn is not None and method in _JAC_CAPABLE_METHODS:
            kwargs["jac"] = jac_fn
        ref = minimize(obj, best_p, **kwargs)
        if not ref.success:
            logger.debug(
                "%s refinement for %s did not report convergence "
                "(status=%s: %s); its result is still accepted if "
                "its objective value improves on the current best, "
                "since the objective value -- not the local "
                "optimiser's own convergence flag -- is the ground "
                "truth here.",
                method, label, ref.status, ref.message)
        # Re-evaluate at ref.x instead of trusting ref.fun. On an
        # ABNORMAL termination (status 2) scipy's L-BFGS-B can
        # return a `fun` that does not correspond to the returned
        # `x` -- reproducible whenever the start point sits on a
        # bound, which is exactly what Stage 1 produces when the
        # true optimum is pinned at a lattice_bounds edge. Trusting
        # the stale value is unsafe in BOTH directions: an
        # understated `fun` gets accepted and propagates an error
        # this structure does not actually achieve (breaking the
        # invariant that compute_error(best_p) == best_e, which
        # candidate ranking and error_breakdown both rely on), and
        # an overstated one discards a refinement that was
        # genuinely an improvement. One extra objective call per
        # method per candidate is negligible against Stage 1's
        # thousands.
        ref_f = obj(ref.x)
        if ref_f < best_e:
            best_p, best_e = ref.x, ref_f
            if verbose:
                print(f"      {method} refined error -> {best_e:.6f}")

    return best_p, best_e


def _centre(obj, cfg, centring_cfg, best_p, best_e, bounds, refine_max,
            refine_ftol, verbose):
    """Centring (ADR-0028 (c)): from a refined point, minimise
    E + CENTRING_WEIGHT * E_ideal, E_ideal the objective with every derived
    bond band shrunk to its ideal length. Where E is zero over a region
    (wide derived bands), this picks the point of that region with bonds
    nearest their ideal lengths, which pulls a strained cell back (wurtzite
    c +11.3 % -> +1.3 %) and moves no published structure more than 2.1 %.
    Accepted only if E rises by at most CENTRING_TOLERANCE and the
    topology class is unchanged."""
    ideal = functools.partial(compute_error, config=centring_cfg)
    ref = minimize(lambda x: obj(x) + CENTRING_WEIGHT * ideal(x), best_p, method="L-BFGS-B",
                   bounds=bounds, options=_refine_options("L-BFGS-B", refine_max, refine_ftol))
    e = obj(ref.x)
    if e <= best_e + CENTRING_TOLERANCE and topology_key(ref.x, cfg) == topology_key(best_p, cfg):
        if verbose:
            print(f"      centred: error {best_e:.6f} -> {e:.6f}, "
                  f"ideal-bond error {ideal(best_p):.6f} -> {ideal(ref.x):.6f}")
        return ref.x, e
    return best_p, best_e


def optimize_one_assignment(
        assignments,
        config,
        opt_cfg,
        verbose=False,
        label="",
        shell_seed_info=None,
):
    """The best topology class of optimize_assignment, as
    (best_params, best_error, structure); (None, 1e10, None) on failure.
    See optimize_assignment for the stages and every YAML key."""
    results = optimize_assignment(assignments, config, opt_cfg, verbose=verbose,
                                  label=label, shell_seed_info=shell_seed_info)
    if not results:
        return None, 1e10, None
    best = results[0]
    return best.params, best.error, best.structure
