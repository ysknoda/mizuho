"""Unit tests for mizuho.optimize.optuna_backend -- both
multistart_optuna_tpe (TPESampler) and multistart_optuna_gp (GPSampler).
Mirrors test_cma.py/test_differential_evolution.py's synthetic
quadratic-bowl-objective pattern (no crystal-structure domain code) to
verify each backend's own mechanics: it satisfies the same optimize_fn
interface contract multistart_cma_es does (see geometry.local_shell.
solve_local_shell's docstring), including its OWN (non-restart-based --
see the module's docstring) reading of the seeded_x0 convention, and
tolerating (accepting-and-ignoring) plateau_patience/plateau_rel_tol.

Tests that need optuna to actually run are individually skipped if it
is not installed (it is an optional dependency -- pyproject.toml's
`bayesian` extra); GP-specific tests additionally need torch (pyproject.
toml's `gp` extra) and are skipped separately via _needs_gp. The
unavailable-backend error tests always run, since they exercise the
guards rather than real optuna/torch behaviour.

GPSampler is far more expensive per trial than TPE (see optuna_backend's
own module docstring), so the GP tests below intentionally use much
smaller trial budgets than their TPE counterparts -- GP's much higher
sample efficiency on low-dimensional problems more than makes up for it
on these tiny synthetic bowls.
"""
import numpy as np
import pytest

from mizuho.optimize.optuna_backend import (
    GP_AVAILABLE,
    OPTUNA_AVAILABLE,
    multistart_optuna_gp,
    multistart_optuna_tpe,
)

_needs_optuna = pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="optuna not installed")
_needs_gp = pytest.mark.skipif(
    not GP_AVAILABLE, reason="optuna GPSampler needs torch (pip install mizuho[gp])")


@_needs_optuna
def test_converges_on_a_simple_quadratic_bowl():
    bounds = [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]
    target = np.array([7.3, 2.1, 5.5])

    def objective(x):
        return float(np.sum((x - target) ** 2))

    best_x, best_f = multistart_optuna_tpe(
        objective=objective, bounds=bounds, n_restarts=4, maxiter=50, seed=0)

    assert best_f == pytest.approx(0.0, abs=0.1)
    assert np.allclose(best_x, target, atol=0.5)


@_needs_optuna
def test_result_respects_bounds():
    bounds = [(2.0, 4.0), (-1.0, 1.0)]
    # Unique minimum well outside the box.
    target = np.array([10.0, 10.0])

    def objective(x):
        return float(np.sum((x - target) ** 2))

    best_x, _ = multistart_optuna_tpe(
        objective=objective, bounds=bounds, n_restarts=2, maxiter=20, seed=0)

    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    assert np.all(best_x >= lo - 1e-9) and np.all(best_x <= hi + 1e-9)


@_needs_optuna
def test_seeded_x0_is_enqueued_and_can_become_the_global_best():
    """Unlike the restart-based backends, seeded_x0 is enqueued as ONE
    of the study's first trials rather than occupying a literal
    'restart index 2' (see the module's own docstring for why TPE does
    not use discrete restarts). With n_restarts=2, maxiter=1
    (n_trials=2, config=None) the two enqueued trials are exactly
    [box centre, seeded_x0] in that order, so this exercises the
    seeded point deterministically."""
    bounds = [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]
    target = np.array([7.3, 2.1, 5.5])

    def objective(x):
        return float(np.sum((x - target) ** 2))

    best_x, best_f = multistart_optuna_tpe(
        objective=objective, bounds=bounds, n_restarts=2,
        maxiter=1, seed=0, config=None, seeded_x0=target)

    assert best_f == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(best_x, target, atol=1e-6)


@_needs_optuna
def test_plateau_kwargs_accepted_and_ignored():
    """plateau_patience/plateau_rel_tol are part of the shared
    optimize_fn contract (geometry.local_shell.solve_local_shell always
    passes them) but this backend has no separate early-stopping
    mechanism of its own -- passing them must not raise."""
    bounds = [(0.0, 5.0), (0.0, 5.0)]
    target = np.array([2.0, 3.0])

    def objective(x):
        return float(np.sum((x - target) ** 2))

    best_x, best_f = multistart_optuna_tpe(
        objective=objective, bounds=bounds, n_restarts=2, maxiter=20,
        seed=0, plateau_patience=15, plateau_rel_tol=1e-3)

    assert np.isfinite(best_f)
    assert len(best_x) == 2


def test_raises_clear_actionable_error_when_optuna_unavailable(monkeypatch):
    """Runs regardless of whether optuna is actually installed --
    exercises the guard itself, not real optuna behaviour."""
    import mizuho.optimize.optuna_backend as mod
    monkeypatch.setattr(mod, "OPTUNA_AVAILABLE", False)
    with pytest.raises(ImportError, match="optuna"):
        mod.multistart_optuna_tpe(objective=lambda x: 0.0, bounds=[(0.0, 1.0)])


# ---------------------------------------------------------------------
# GPSampler (Gaussian-Process-based Bayesian optimisation)
# ---------------------------------------------------------------------

@_needs_gp
def test_gp_converges_on_a_simple_quadratic_bowl():
    bounds = [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]
    target = np.array([7.3, 2.1, 5.5])

    def objective(x):
        return float(np.sum((x - target) ** 2))

    best_x, best_f = multistart_optuna_gp(
        objective=objective, bounds=bounds, n_restarts=2, maxiter=15, seed=0)

    assert best_f == pytest.approx(0.0, abs=0.1)
    assert np.allclose(best_x, target, atol=0.5)


@_needs_gp
def test_gp_result_respects_bounds():
    bounds = [(2.0, 4.0), (-1.0, 1.0)]
    # Unique minimum well outside the box.
    target = np.array([10.0, 10.0])

    def objective(x):
        return float(np.sum((x - target) ** 2))

    best_x, _ = multistart_optuna_gp(
        objective=objective, bounds=bounds, n_restarts=2, maxiter=5, seed=0)

    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    assert np.all(best_x >= lo - 1e-9) and np.all(best_x <= hi + 1e-9)


@_needs_gp
def test_gp_seeded_x0_is_enqueued_and_can_become_the_global_best():
    """Same enqueue-based determinism as TPE's own equivalent test above
    (see its docstring): n_restarts=2, maxiter=1 (config=None) means
    exactly 2 trials total, both enqueued (box centre then seeded_x0)
    and consumed before GPSampler's own sampling logic (random startup
    or GP-guided) ever runs, so this is deterministic regardless of
    GPSampler's own internals."""
    bounds = [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]
    target = np.array([7.3, 2.1, 5.5])

    def objective(x):
        return float(np.sum((x - target) ** 2))

    best_x, best_f = multistart_optuna_gp(
        objective=objective, bounds=bounds, n_restarts=2,
        maxiter=1, seed=0, config=None, seeded_x0=target)

    assert best_f == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(best_x, target, atol=1e-6)


@_needs_gp
def test_gp_plateau_kwargs_accepted_and_ignored():
    """plateau_patience/plateau_rel_tol are part of the shared
    optimize_fn contract but this backend has no separate early-stopping
    mechanism of its own -- passing them must not raise."""
    bounds = [(0.0, 5.0), (0.0, 5.0)]
    target = np.array([2.0, 3.0])

    def objective(x):
        return float(np.sum((x - target) ** 2))

    best_x, best_f = multistart_optuna_gp(
        objective=objective, bounds=bounds, n_restarts=2, maxiter=5,
        seed=0, plateau_patience=15, plateau_rel_tol=1e-3)

    assert np.isfinite(best_f)
    assert len(best_x) == 2


@_needs_gp
def test_gp_n_trials_override_is_respected():
    """An explicit gp_n_trials must be used verbatim as the actual
    number of objective evaluations, regardless of n_restarts/maxiter
    (which would otherwise imply a much larger nominal budget)."""
    bounds = [(0.0, 5.0), (0.0, 5.0)]
    calls = []

    def objective(x):
        calls.append(1)
        return float(np.sum(x ** 2))

    multistart_optuna_gp(
        objective=objective, bounds=bounds, n_restarts=8, maxiter=400,
        seed=0, gp_n_trials=7)

    assert len(calls) == 7


@_needs_gp
def test_gp_default_trial_budget_is_capped(monkeypatch):
    """Without an explicit override, the default trial budget must be
    capped (see module docstring: GPSampler's per-trial cost makes
    reusing TPE's n_restarts*maxiter formula verbatim impractically
    slow) rather than growing unboundedly with n_restarts*maxiter. The
    cap itself is monkeypatched down to a tiny value so this test stays
    fast regardless of the real default's magnitude."""
    import mizuho.optimize.optuna_backend as mod
    monkeypatch.setattr(mod, "_GP_DEFAULT_MAX_TRIALS", 6)

    bounds = [(0.0, 5.0), (0.0, 5.0)]
    calls = []

    def objective(x):
        calls.append(1)
        return float(np.sum(x ** 2))

    mod.multistart_optuna_gp(
        objective=objective, bounds=bounds, n_restarts=50, maxiter=50, seed=0)

    assert len(calls) == 6


def test_raises_clear_actionable_error_when_optuna_unavailable_for_gp(monkeypatch):
    """Runs regardless of whether optuna is actually installed --
    exercises the guard itself, not real optuna/torch behaviour."""
    import mizuho.optimize.optuna_backend as mod
    monkeypatch.setattr(mod, "OPTUNA_AVAILABLE", False)
    monkeypatch.setattr(mod, "GP_AVAILABLE", False)
    with pytest.raises(ImportError, match="optuna"):
        mod.multistart_optuna_gp(objective=lambda x: 0.0, bounds=[(0.0, 1.0)])


def test_raises_clear_actionable_error_when_torch_unavailable_but_optuna_present(monkeypatch):
    """Distinct guard from the optuna-missing case above: optuna itself
    is present but GPSampler's own torch dependency is not. Exercised
    unconditionally via monkeypatch (not tied to this environment's
    actual torch installation state), mirroring how the OPTUNA_AVAILABLE
    guard tests above are also environment-independent."""
    import mizuho.optimize.optuna_backend as mod
    monkeypatch.setattr(mod, "OPTUNA_AVAILABLE", True)
    monkeypatch.setattr(mod, "GP_AVAILABLE", False)
    with pytest.raises(ImportError, match="torch"):
        mod.multistart_optuna_gp(objective=lambda x: 0.0, bounds=[(0.0, 1.0)])
