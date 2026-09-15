"""Optuna-based optimize_fn backends: TPESampler (Tree-structured Parzen
Estimator, a sequential density-ratio-estimation Bayesian-optimisation
method) and GPSampler (Gaussian-Process-based Bayesian optimisation --
the "classic" BO formulation: GP surrogate + acquisition function). Both
match the same interface contract as optimize.cma.multistart_cma_es (see
geometry.local_shell.solve_local_shell's docstring) so either is a
drop-in alternative everywhere multistart_cma_es is used today.

Optuna is an OPTIONAL dependency (pyproject.toml's `bayesian` extra) --
this module is safe to import even when optuna is not installed;
OPTUNA_AVAILABLE reflects whether it actually is, and multistart_optuna_tpe
raises a clear, actionable ImportError if called without it.

GPSampler needs MORE than plain optuna: it also requires torch (both GP
kernel-hyperparameter fitting and acquisition-function optimisation use
torch's autograd internally). Confirmed by reading optuna's own source:
optuna/samplers/_gp/sampler.py only imports torch lazily, inside
sample_relative, so `import optuna` itself never requires torch -- but
that also means GPSampler does NOT fail at construction time when torch
is missing, only on the first trial past its startup phase, and by then
n_startup_trials real objective evaluations have already been spent. So
GP_AVAILABLE is checked independently of OPTUNA_AVAILABLE at import time
(a plain `import torch` probe) and multistart_optuna_gp fails fast with
a clear, actionable message before running anything, rather than
surfacing torch's own confusing multi-frame ModuleNotFoundError mid-run.
`greenlet` is also recommended (pyproject.toml's `gp` extra installs it
too): GPSampler's acquisition-function optimisation runs several local
searches, and without greenlet it falls back to a sequential loop that
was measured (this project's own dev environment, 3-parameter problem)
at roughly 3x slower.

Unlike CMA-ES/differential_evolution/basin_hopping, neither Optuna
sampler is a restart-based algorithm: TPE's and GP's core strength is a
SEQUENTIAL surrogate model that gets better the more trials it has seen
in ONE continuous history, so splitting the budget into n_restarts
independent studies (as the other backends do) would throw away exactly
what makes them useful. Instead, n_restarts * maxiter is treated as ONE
combined trial budget for a single optuna.Study, and the restart-slot
starting points (volume-aware lattice estimate, box centre, seeded_x0)
are injected as the study's first few trials via study.enqueue_trial --
still explored and contributing to the surrogate model's early history,
just not as independent restarts.

GPSampler is far more expensive per trial than TPE: every trial past its
own startup phase refits the GP's kernel hyperparameters (gradient-based
marginal-log-likelihood maximisation) AND re-optimises the acquisition
function (QMC candidate search plus several torch/L-BFGS-B local
searches). Measured in this project's dev environment at ~15-30ms/trial
even for a small 3-parameter synthetic bowl (200 trials ~5s with
greenlet installed), growing with both trial count and dimensionality.
Reusing TPE's n_restarts*maxiter formula verbatim (often several
thousand under this project's default budgets) would make GPSampler
impractically slow, so multistart_optuna_gp instead caps the trial
count at _GP_DEFAULT_MAX_TRIALS unless the caller passes an explicit
gp_n_trials override.
"""

import warnings

import numpy as np

from .init_guess import _estimate_lattice_x0

try:
    import optuna
    OPTUNA_AVAILABLE = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    OPTUNA_AVAILABLE = False

if OPTUNA_AVAILABLE:
    try:
        import torch  # noqa: F401 -- presence check only; GPSampler
                       # itself imports torch lazily, see module docstring.
        GP_AVAILABLE = True
    except ImportError:
        GP_AVAILABLE = False
else:
    GP_AVAILABLE = False

# GPSampler's per-trial cost (GP refit + acquisition-function
# optimisation) grows with trial count; a few hundred is the practical
# ceiling for this project's per-evaluation cost budget -- see module
# docstring. Overridable via the gp_n_trials kwarg / YAML's
# optimization.gp_n_trials key.
_GP_DEFAULT_MAX_TRIALS = 200


def _run_optuna_study(sampler, objective, bounds, n_trials, config,
                       seeded_x0, verbose, label):
    """Shared study-construction/seeding/optimise/extract-best logic for
    every Optuna sampler this module supports -- the samplers differ
    only in how `sampler` itself was constructed by the caller."""
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    ndim = len(bounds)
    param_names = [f"x{i}" for i in range(ndim)]

    study = optuna.create_study(direction="minimize", sampler=sampler)

    # Restart-slot starting points, enqueued (not independently
    # restarted -- see module docstring) so they still seed the
    # surrogate model's early history, in the same priority order the
    # other backends use for restarts 0/1/2.
    seed_points = []
    if config is not None:
        seed_points.append(_estimate_lattice_x0(config, bounds))
    seed_points.append((lo + hi) / 2.0)
    if seeded_x0 is not None:
        seed_points.append(np.array(seeded_x0, dtype=float))
    for pt in seed_points:
        pt_clipped = np.clip(pt, lo, hi)
        study.enqueue_trial({name: float(v)
                              for name, v in zip(param_names, pt_clipped)})

    def trial_objective(trial):
        x = np.array([trial.suggest_float(name, float(lo[i]), float(hi[i]))
                       for i, name in enumerate(param_names)])
        return float(objective(x))

    if verbose:
        print(f"      [Optuna {label}] {n_trials} trials "
              f"({len(seed_points)} seeded)")

    study.optimize(trial_objective, n_trials=n_trials, show_progress_bar=False)

    best_x = np.array([study.best_params[name] for name in param_names])
    best_f = float(study.best_value)

    if verbose:
        print(f"      [Optuna {label}] best error: {best_f:.6f}")

    return best_x, best_f


def multistart_optuna_tpe(
        objective,
        bounds,
        n_restarts=8,
        maxiter=400,
        seed=None,
        verbose=False,
        config=None,
        periodic_mask=None,
        seeded_x0=None,
        **_ignored,
):
    """
    Run Optuna's TPESampler for n_restarts * maxiter total trials and
    return the best point found.

    Parameters
    ----------
    n_restarts, maxiter : their PRODUCT is the total trial budget (see
        module docstring for why TPE does not use discrete restarts).
        n_restarts also sets TPESampler's own n_startup_trials (its
        initial random-exploration phase before the Bayesian surrogate
        takes over), floored at 10.
    periodic_mask : accepted for interface compatibility but not
        needed -- trial.suggest_float(low, high) can never produce an
        out-of-bounds value in the first place, unlike the other
        backends' population/perturbation steps.
    plateau_patience, plateau_rel_tol : accepted (via **_ignored) for
        interface compatibility with the shared optimize_fn contract
        but IGNORED here, same as differential_evolution/basin_hopping
        -- this backend does not implement a separate early-stopping
        mechanism on top of its own trial budget.

    Returns
    -------
    best_x : np.ndarray
    best_f : float
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError(
            "optimization.backend 'tpe' requires the optional 'optuna' "
            "package -- install it with: pip install mizuho[bayesian]")

    n_trials = max(1, int(n_restarts) * int(maxiter))
    n_startup = max(10, int(n_restarts))

    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup)
    return _run_optuna_study(sampler, objective, bounds, n_trials, config,
                              seeded_x0, verbose, "TPE")


def multistart_optuna_gp(
        objective,
        bounds,
        n_restarts=8,
        maxiter=400,
        seed=None,
        verbose=False,
        config=None,
        periodic_mask=None,
        seeded_x0=None,
        gp_n_trials=None,
        **_ignored,
):
    """
    Run Optuna's GPSampler (Gaussian-Process-based Bayesian optimisation:
    a GP surrogate fit to all trials so far, plus an acquisition function
    optimised to pick the next point) and return the best point found.

    Parameters
    ----------
    gp_n_trials : explicit trial-budget override. None (default) uses
        min(n_restarts * maxiter, _GP_DEFAULT_MAX_TRIALS) -- see module
        docstring for why GPSampler needs a much smaller budget than TPE
        can comfortably handle.
    n_restarts : also sets GPSampler's own n_startup_trials (its initial
        random-exploration phase before the GP surrogate takes over),
        floored at 10 and capped at the actual trial budget so a small
        gp_n_trials override can never make every trial a startup trial.
    periodic_mask : accepted for interface compatibility but not
        needed, same reason as multistart_optuna_tpe.
    plateau_patience, plateau_rel_tol : accepted (via **_ignored) but
        ignored, same as every other backend in this module.

    Returns
    -------
    best_x : np.ndarray
    best_f : float
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError(
            "optimization.backend 'gp'/'bayesian' requires the optional "
            "'optuna' (>=3.6) and 'torch' packages -- install them with: "
            "pip install mizuho[gp]")
    if not GP_AVAILABLE:
        raise ImportError(
            "optimization.backend 'gp'/'bayesian' requires the optional "
            "'torch' package (Optuna's GPSampler uses it for GP fitting "
            "and acquisition-function optimisation) -- install it with: "
            "pip install mizuho[gp]")

    if gp_n_trials is not None:
        n_trials = max(1, int(gp_n_trials))
    else:
        n_trials = max(1, min(int(n_restarts) * int(maxiter), _GP_DEFAULT_MAX_TRIALS))
    n_startup = max(1, min(max(10, int(n_restarts)), n_trials))

    # GPSampler is marked experimental by optuna itself (interface may
    # still change upstream); mizuho treats it as a first-class
    # supported backend, so this per-construction warning would just be
    # noise for our users. Suppressed locally (not at module-import
    # time) because pytest -- and potentially other callers -- reset
    # the global warnings-filter list around each call, which would
    # silently defeat a filterwarnings() registered once at import time.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=optuna.exceptions.ExperimentalWarning)
        sampler = optuna.samplers.GPSampler(seed=seed, n_startup_trials=n_startup)
    return _run_optuna_study(sampler, objective, bounds, n_trials, config,
                              seeded_x0, verbose, "GP")
