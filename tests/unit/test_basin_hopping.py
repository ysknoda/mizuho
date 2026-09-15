"""Unit tests for mizuho.optimize.basin_hopping.multistart_basin_hopping
-- mirrors test_cma.py's synthetic-objective pattern (no crystal-
structure domain code) to verify this backend's own mechanics: it
satisfies the same optimize_fn interface contract multistart_cma_es
does (see geometry.local_shell.solve_local_shell's docstring),
including the seeded_x0 restart-slot convention and tolerating
(accepting-and-ignoring) plateau_patience/plateau_rel_tol.

Basin hopping's own internal minimiser (L-BFGS-B) converges to A local
optimum from almost any starting point within very few iterations, so
-- unlike CMA-ES/DE, whose population needs many generations to
converge even on a smooth unimodal bowl -- a single-restart test on a
plain quadratic bowl reaches the true minimum regardless of seeding
and cannot distinguish "seeded" from "not seeded". The restart-slot
tests below instead use a two-basin (decoy) objective, so a restart
starting outside the true minimum's basin of attraction demonstrably
converges to the WRONG (decoy) optimum -- see
test_seeded_x0_ignored_when_fewer_than_three_restarts's docstring.
"""
import functools

import numpy as np
import pytest

from mizuho.optimize.basin_hopping import multistart_basin_hopping


def _module_level_bowl(x, target):
    """Module-level (picklable) objective, required for restart_workers>1
    tests -- a process pool cannot pickle a local closure."""
    return float(np.sum((x - target) ** 2))


def test_converges_on_a_simple_quadratic_bowl():
    bounds = [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]
    target = np.array([7.3, 2.1, 5.5])

    def objective(x):
        return float(np.sum((x - target) ** 2))

    best_x, best_f = multistart_basin_hopping(
        objective=objective, bounds=bounds, n_restarts=2, maxiter=20, seed=0)

    assert best_f == pytest.approx(0.0, abs=1e-4)
    assert np.allclose(best_x, target, atol=1e-2)


def test_result_respects_bounds():
    bounds = [(2.0, 4.0), (-1.0, 1.0)]
    target = np.array([10.0, 10.0])   # unreachable minimum, outside the box

    def objective(x):
        return float(np.sum((x - target) ** 2))

    best_x, _ = multistart_basin_hopping(
        objective=objective, bounds=bounds, n_restarts=1, maxiter=20, seed=0)

    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    assert np.all(best_x >= lo - 1e-9) and np.all(best_x <= hi + 1e-9)


def _two_basin_objective(target, decoy, decoy_penalty=0.3):
    """A minimum at `target` (value 0) and a second, easily-reached
    local minimum at `decoy` (value `decoy_penalty`) -- see module
    docstring for why this, not a plain unimodal bowl, is needed to
    tell a seeded restart apart from an unseeded one for this
    backend."""
    target = np.asarray(target, dtype=float)
    decoy = np.asarray(decoy, dtype=float)

    def objective(x):
        x = np.asarray(x, dtype=float)
        return float(min(np.sum((x - target) ** 2),
                          np.sum((x - decoy) ** 2) + decoy_penalty))
    return objective


def test_seeded_x0_restart_can_become_the_global_best():
    """Restart index 2 uses seeded_x0 as-is (mirrors multistart_cma_es's
    own restart-slot convention exactly). Seeding exactly at the true
    (non-decoy) minimum must reach it, even though the decoy minimum
    would otherwise be the one restarts 0/1 (both starting at the box
    centre, which coincides with the decoy here) converge to."""
    bounds = [(0.0, 10.0), (0.0, 10.0)]
    target = np.array([9.9, 9.9])
    decoy = np.array([5.0, 5.0])   # sits exactly at the box centre
    objective = _two_basin_objective(target, decoy)

    best_x, best_f = multistart_basin_hopping(
        objective=objective, bounds=bounds, n_restarts=3,
        maxiter=5, seed=0, config=None, seeded_x0=target)

    assert best_f == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(best_x, target, atol=1e-3)


def test_seeded_x0_ignored_when_fewer_than_three_restarts():
    """seeded_x0 is restart index 2 specifically -- with n_restarts=2 it
    never gets a slot. With config=None, BOTH restart 0 and restart 1
    fall back to the box centre (see multistart_cma_es's own restart-
    slot docstring, mirrored here), which coincides with the decoy
    minimum -- so if seeded_x0 were (incorrectly) still used, the
    result would be the TRUE minimum (0); if correctly ignored, both
    restarts converge only to the decoy (0.3)."""
    bounds = [(0.0, 10.0), (0.0, 10.0)]
    target = np.array([9.9, 9.9])
    decoy = np.array([5.0, 5.0])
    objective = _two_basin_objective(target, decoy)

    _, best_f = multistart_basin_hopping(
        objective=objective, bounds=bounds, n_restarts=2,
        maxiter=5, seed=0, config=None, seeded_x0=target)

    assert best_f == pytest.approx(0.3, abs=1e-6)


def test_plateau_kwargs_accepted_and_ignored():
    """plateau_patience/plateau_rel_tol are part of the shared
    optimize_fn contract (geometry.local_shell.solve_local_shell always
    passes them) but this backend has no separate early-stopping
    mechanism of its own -- passing them must not raise."""
    bounds = [(0.0, 5.0), (0.0, 5.0)]
    target = np.array([2.0, 3.0])

    def objective(x):
        return float(np.sum((x - target) ** 2))

    best_x, best_f = multistart_basin_hopping(
        objective=objective, bounds=bounds, n_restarts=1, maxiter=10,
        seed=0, plateau_patience=15, plateau_rel_tol=1e-3)

    assert np.isfinite(best_f)
    assert len(best_x) == 2


# ---------------------------------------------------------------------------
# restart_workers (phase H-4): dispatching Pass-2's per-restart
# basinhopping calls across a process pool instead of one at a time.
# Pass 1 (x0 construction) always runs sequentially regardless of this
# value, so every restart's x0 -- and therefore the final result -- must
# be bit-identical whichever value is used.
# ---------------------------------------------------------------------------

def test_restart_workers_omitted_defaults_to_sequential():
    bounds = [(0.0, 10.0), (0.0, 10.0)]
    target = np.array([3.0, 7.0])

    def objective(x):
        return float(np.sum((x - target) ** 2))

    best_x, best_f = multistart_basin_hopping(
        objective=objective, bounds=bounds, n_restarts=2, maxiter=10, seed=0)

    assert np.isfinite(best_f)
    assert len(best_x) == 2


def test_restart_workers_parallel_matches_sequential():
    bounds = [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]
    target = np.array([7.3, 2.1, 5.5])
    objective = functools.partial(_module_level_bowl, target=target)

    x_seq, f_seq = multistart_basin_hopping(
        objective=objective, bounds=bounds, n_restarts=4,
        maxiter=15, seed=42, restart_workers=1)
    x_par, f_par = multistart_basin_hopping(
        objective=objective, bounds=bounds, n_restarts=4,
        maxiter=15, seed=42, restart_workers=3)

    assert f_par == f_seq
    assert np.array_equal(x_par, x_seq)
