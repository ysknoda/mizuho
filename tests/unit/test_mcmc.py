"""Unit tests for mizuho.optimize.mcmc.multistart_mcmc -- mirrors
test_cma.py/test_differential_evolution.py's synthetic quadratic-bowl-
objective pattern (no crystal-structure domain code) to verify this
backend's own mechanics: it satisfies the same optimize_fn interface
contract multistart_cma_es does (see geometry.local_shell.
solve_local_shell's docstring), including its OWN (ensemble-anchoring,
not restart-slot -- see the module's docstring) reading of the
seeded_x0 convention, and tolerating (accepting-and-ignoring)
plateau_patience/plateau_rel_tol.

MCMC is a SAMPLER, not a pure local optimiser -- its convergence
tolerances here are deliberately looser than the other backends' tests
to reflect that (see the module's own docstring on why its real value
in this project is diagnostic, not raw precision).

Tests that need emcee to actually run are individually skipped if it
is not installed (it is an optional dependency -- pyproject.toml's
`mcmc` extra); the unavailable-backend error test always runs, since
it exercises the guard rather than real emcee behaviour.
"""
import numpy as np
import pytest

from mizuho.optimize.mcmc import EMCEE_AVAILABLE, multistart_mcmc

_needs_emcee = pytest.mark.skipif(not EMCEE_AVAILABLE, reason="emcee not installed")


@_needs_emcee
def test_converges_reasonably_on_a_simple_quadratic_bowl():
    bounds = [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]
    target = np.array([7.3, 2.1, 5.5])

    def objective(x):
        return float(np.sum((x - target) ** 2))

    best_x, best_f = multistart_mcmc(
        objective=objective, bounds=bounds, n_restarts=8, maxiter=200, seed=0)

    assert best_f < 1.0
    assert np.allclose(best_x, target, atol=1.0)


@_needs_emcee
def test_result_respects_bounds():
    bounds = [(2.0, 4.0), (-1.0, 1.0)]
    # Unique minimum well outside the box.
    target = np.array([10.0, 10.0])

    def objective(x):
        return float(np.sum((x - target) ** 2))

    best_x, _ = multistart_mcmc(
        objective=objective, bounds=bounds, n_restarts=4, maxiter=60, seed=0)

    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    assert np.all(best_x >= lo - 1e-9) and np.all(best_x <= hi + 1e-9)


@_needs_emcee
def test_seeded_x0_meaningfully_improves_a_short_chain_result():
    """seeded_x0 anchors a handful of walkers NEAR it (with a small
    jitter -- MCMC's ensemble needs positional spread to function, so
    unlike the restart-based backends it cannot evaluate seeded_x0
    exactly as one deterministic trial). With very few steps, a chain
    seeded near the target must still land far closer to it than one
    with no seed at all, which has no informative starting point to
    work from in so few steps."""
    bounds = [(0.0, 20.0), (0.0, 20.0), (0.0, 20.0)]
    target = np.array([17.3, 1.4, 12.5])

    def objective(x):
        return float(np.sum((x - target) ** 2))

    _, seeded_f = multistart_mcmc(
        objective=objective, bounds=bounds, n_restarts=4, maxiter=3,
        seed=0, config=None, seeded_x0=target)
    _, unseeded_f = multistart_mcmc(
        objective=objective, bounds=bounds, n_restarts=4, maxiter=3,
        seed=0, config=None, seeded_x0=None)

    assert seeded_f < unseeded_f
    assert seeded_f < 5.0


@_needs_emcee
def test_explicit_temperature_override_accepted():
    bounds = [(0.0, 5.0), (0.0, 5.0)]
    target = np.array([2.0, 3.0])

    def objective(x):
        return float(np.sum((x - target) ** 2))

    best_x, best_f = multistart_mcmc(
        objective=objective, bounds=bounds, n_restarts=4, maxiter=30,
        seed=0, mcmc_temperature=0.5)

    assert np.isfinite(best_f)
    assert len(best_x) == 2


@_needs_emcee
def test_plateau_kwargs_accepted_and_ignored():
    """plateau_patience/plateau_rel_tol are part of the shared
    optimize_fn contract (geometry.local_shell.solve_local_shell always
    passes them) but an MCMC chain has no per-restart early-stopping
    of its own -- passing them must not raise."""
    bounds = [(0.0, 5.0), (0.0, 5.0)]
    target = np.array([2.0, 3.0])

    def objective(x):
        return float(np.sum((x - target) ** 2))

    best_x, best_f = multistart_mcmc(
        objective=objective, bounds=bounds, n_restarts=4, maxiter=30,
        seed=0, plateau_patience=15, plateau_rel_tol=1e-3)

    assert np.isfinite(best_f)
    assert len(best_x) == 2


def test_raises_clear_actionable_error_when_emcee_unavailable(monkeypatch):
    """Runs regardless of whether emcee is actually installed --
    exercises the guard itself, not real emcee behaviour."""
    import mizuho.optimize.mcmc as mod
    monkeypatch.setattr(mod, "EMCEE_AVAILABLE", False)
    with pytest.raises(ImportError, match="emcee"):
        mod.multistart_mcmc(objective=lambda x: 0.0, bounds=[(0.0, 1.0)])
