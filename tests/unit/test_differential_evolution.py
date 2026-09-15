"""Unit tests for mizuho.optimize.differential_evolution.
multistart_differential_evolution -- mirrors test_cma.py's synthetic
quadratic-bowl-objective pattern (no crystal-structure domain code) to
verify this backend's own mechanics: it satisfies the same optimize_fn
interface contract multistart_cma_es does (see geometry.local_shell.
solve_local_shell's docstring), including the seeded_x0 restart-slot
convention and tolerating (accepting-and-ignoring) plateau_patience/
plateau_rel_tol.
"""
import functools

import numpy as np
import pytest

from mizuho.optimize.differential_evolution import multistart_differential_evolution


def _module_level_bowl(x, target):
    """Module-level (picklable) objective, required for restart_workers>1
    tests -- a process pool cannot pickle a local closure."""
    return float(np.sum((x - target) ** 2))


def test_converges_on_a_simple_quadratic_bowl():
    bounds = [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]
    target = np.array([7.3, 2.1, 5.5])

    def objective(x):
        return float(np.sum((x - target) ** 2))

    best_x, best_f = multistart_differential_evolution(
        objective=objective, bounds=bounds, n_restarts=2, maxiter=60, seed=0)

    assert best_f == pytest.approx(0.0, abs=1e-4)
    assert np.allclose(best_x, target, atol=1e-2)


def test_result_respects_bounds():
    bounds = [(2.0, 4.0), (-1.0, 1.0)]
    # Unique minimum well outside the box -- the optimiser must not
    # report a point beyond [2,4]x[-1,1] even though the true minimum
    # (10, 10) is unreachable.
    target = np.array([10.0, 10.0])

    def objective(x):
        return float(np.sum((x - target) ** 2))

    best_x, _ = multistart_differential_evolution(
        objective=objective, bounds=bounds, n_restarts=1, maxiter=40, seed=0)

    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    assert np.all(best_x >= lo - 1e-9) and np.all(best_x <= hi + 1e-9)


def test_seeded_x0_restart_can_become_the_global_best():
    """Restart index 2 uses seeded_x0 as-is (mirrors multistart_cma_es's
    own restart-slot convention exactly). Placing the objective's unique
    minimum exactly at seeded_x0 with maxiter=1 leaves essentially no
    room for DE's own search to find it by chance in the other
    restarts, so recovering it exactly demonstrates the seed was
    genuinely used."""
    bounds = [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]
    target = np.array([7.3, 2.1, 5.5])

    def objective(x):
        return float(np.sum((x - target) ** 2))

    best_x, best_f = multistart_differential_evolution(
        objective=objective, bounds=bounds, n_restarts=3,
        maxiter=1, seed=0, config=None, seeded_x0=target)

    assert best_f == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(best_x, target, atol=1e-6)


def test_seeded_x0_ignored_when_fewer_than_three_restarts():
    """seeded_x0 is restart index 2 specifically -- with n_restarts=2 it
    never gets a slot, and must not silently substitute for restart
    0/1's own deliberate starting points (box centre, since config=None
    here). A loose threshold would be flaky (DE's own population-based
    search can occasionally stumble close to any fixed point with
    enough generations), so this uses maxiter=1 -- a single generation
    landing within sqrt(1e-4)=0.01 units of an exact point purely by
    chance is effectively impossible for a continuous random draw."""
    bounds = [(0.0, 10.0), (0.0, 10.0)]
    target = np.array([9.9, 9.9])

    def objective(x):
        return float(np.sum((x - target) ** 2))

    _, best_f = multistart_differential_evolution(
        objective=objective, bounds=bounds, n_restarts=2,
        maxiter=1, seed=0, config=None, seeded_x0=target)

    assert best_f > 1e-4


def test_periodic_mask_reaches_boundary_values():
    """A target sitting exactly at the periodic boundary (0.0/1.0) must
    still be reachable -- confirms periodic_mask's wrap is actually
    exercised, not merely accepted and ignored."""
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    target = np.array([0.02, 0.98])
    periodic_mask = np.array([True, True])

    def objective(x):
        return float(np.sum((x - target) ** 2))

    best_x, best_f = multistart_differential_evolution(
        objective=objective, bounds=bounds, n_restarts=2, maxiter=60,
        seed=1, periodic_mask=periodic_mask)

    assert best_f == pytest.approx(0.0, abs=1e-3)


def test_plateau_kwargs_accepted_and_ignored():
    """plateau_patience/plateau_rel_tol are part of the shared
    optimize_fn contract (geometry.local_shell.solve_local_shell always
    passes them) but this backend has no separate early-stopping
    mechanism of its own -- passing them must not raise and must not
    change the outcome versus omitting them."""
    bounds = [(0.0, 5.0), (0.0, 5.0)]
    target = np.array([2.0, 3.0])

    def objective(x):
        return float(np.sum((x - target) ** 2))

    best_x, best_f = multistart_differential_evolution(
        objective=objective, bounds=bounds, n_restarts=1, maxiter=40,
        seed=0, plateau_patience=15, plateau_rel_tol=1e-3)

    assert np.isfinite(best_f)
    assert len(best_x) == 2


# ---------------------------------------------------------------------------
# restart_workers (phase H-4): dispatching Pass-2's per-restart
# differential_evolution calls across a process pool instead of one at a
# time. Pass 1 (x0 construction) always runs sequentially regardless of
# this value, so every restart's x0 -- and therefore the final result --
# must be bit-identical whichever value is used.
# ---------------------------------------------------------------------------

def test_restart_workers_omitted_defaults_to_sequential():
    bounds = [(0.0, 10.0), (0.0, 10.0)]
    target = np.array([3.0, 7.0])

    def objective(x):
        return float(np.sum((x - target) ** 2))

    best_x, best_f = multistart_differential_evolution(
        objective=objective, bounds=bounds, n_restarts=2, maxiter=30, seed=0)

    assert np.isfinite(best_f)
    assert len(best_x) == 2


def test_restart_workers_parallel_matches_sequential():
    bounds = [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]
    target = np.array([7.3, 2.1, 5.5])
    objective = functools.partial(_module_level_bowl, target=target)

    x_seq, f_seq = multistart_differential_evolution(
        objective=objective, bounds=bounds, n_restarts=4,
        maxiter=30, seed=42, restart_workers=1)
    x_par, f_par = multistart_differential_evolution(
        objective=objective, bounds=bounds, n_restarts=4,
        maxiter=30, seed=42, restart_workers=3)

    assert f_par == f_seq
    assert np.array_equal(x_par, x_seq)


def test_restart_workers_parallel_matches_sequential_with_periodic_mask():
    """The periodic-dimension wrap (_wrap_periodic) must survive a
    process-pool round trip too -- it is now a functools.partial around
    a module-level function specifically so this works (phase H-4); a
    local closure would fail to pickle here."""
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    target = np.array([0.02, 0.98])
    periodic_mask = np.array([True, True])
    objective = functools.partial(_module_level_bowl, target=target)

    x_seq, f_seq = multistart_differential_evolution(
        objective=objective, bounds=bounds, n_restarts=3, maxiter=40,
        seed=1, periodic_mask=periodic_mask, restart_workers=1)
    x_par, f_par = multistart_differential_evolution(
        objective=objective, bounds=bounds, n_restarts=3, maxiter=40,
        seed=1, periodic_mask=periodic_mask, restart_workers=3)

    assert f_par == f_seq
    assert np.array_equal(x_par, x_seq)
