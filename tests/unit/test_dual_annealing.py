"""Unit tests for mizuho.optimize.dual_annealing.multistart_dual_annealing
-- mirrors test_cma.py/test_basin_hopping.py's synthetic-objective
pattern (no crystal-structure domain code) to verify this backend's own
mechanics: it satisfies the same optimize_fn interface contract
multistart_cma_es does (see geometry.local_shell.solve_local_shell's
docstring), including the seeded_x0 restart-slot convention and
tolerating (accepting-and-ignoring) plateau_patience/plateau_rel_tol.

Like basin hopping, dual_annealing's own internal minimiser (L-BFGS-B,
run after each accepted annealing step) converges tightly even from a
handful of iterations -- confirmed empirically before writing these
tests: a restart seeded exactly at a target with maxiter=5 lands on it
to floating-point precision, while an UNSEEDED restart starting at a
decoy minimum's basin does NOT escape it within a similarly small
maxiter. So, like test_basin_hopping.py, the seeded_x0-specific tests
below use a two-basin (decoy) objective rather than a plain bowl -- a
plain bowl cannot distinguish "seeded" from "not seeded" once the
embedded local search reaches full precision regardless of which basin
it started in.
"""
import functools

import numpy as np
import pytest

from mizuho.optimize.dual_annealing import multistart_dual_annealing


def _module_level_bowl(x, target):
    """Module-level (picklable) objective, required for restart_workers>1
    tests -- a process pool cannot pickle a local closure."""
    return float(np.sum((x - target) ** 2))


def test_converges_on_a_simple_quadratic_bowl():
    bounds = [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]
    target = np.array([7.3, 2.1, 5.5])

    def objective(x):
        return float(np.sum((x - target) ** 2))

    best_x, best_f = multistart_dual_annealing(
        objective=objective, bounds=bounds, n_restarts=2, maxiter=20, seed=0)

    assert best_f == pytest.approx(0.0, abs=1e-4)
    assert np.allclose(best_x, target, atol=1e-2)


def test_result_respects_bounds():
    bounds = [(2.0, 4.0), (-1.0, 1.0)]
    target = np.array([10.0, 10.0])   # unreachable minimum, outside the box

    def objective(x):
        return float(np.sum((x - target) ** 2))

    best_x, _ = multistart_dual_annealing(
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

    best_x, best_f = multistart_dual_annealing(
        objective=objective, bounds=bounds, n_restarts=3,
        maxiter=5, seed=0, config=None, seeded_x0=target)

    assert best_f == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(best_x, target, atol=1e-3)


def test_seeded_x0_ignored_when_fewer_than_three_restarts(monkeypatch):
    """seeded_x0 is restart index 2 specifically -- with n_restarts=2 it
    never gets a slot. Checked on the starting points handed to each
    restart rather than on where a stochastic global search happens to
    finish: whether dual annealing escapes a decoy basin in 5 iterations
    depends on its random stream, not on this rule."""
    import mizuho.optimize.dual_annealing as da
    captured = []

    def capture(fn, tasks, max_workers=None, on_result=None):
        captured.extend(tasks)
        return [(t[7], 1.0) for t in tasks]

    # Restart dispatch lives in optimize.endpoints.run_restarts (ADR-0019).
    import mizuho.optimize.endpoints as endpoints
    monkeypatch.setattr(endpoints, "run_parallel_tasks", capture)
    bounds = [(0.0, 10.0), (0.0, 10.0)]
    target = np.array([9.9, 9.9])
    multistart_dual_annealing(
        objective=lambda x: 0.0, bounds=bounds, n_restarts=2,
        maxiter=5, seed=0, config=None, seeded_x0=target)

    x0s = [t[7] for t in captured]
    assert len(x0s) == 2
    assert not any(np.allclose(x0, target) for x0 in x0s)


def test_plateau_kwargs_accepted_and_ignored():
    """plateau_patience/plateau_rel_tol are part of the shared
    optimize_fn contract (geometry.local_shell.solve_local_shell always
    passes them) but this backend has no separate early-stopping
    mechanism of its own -- passing them must not raise."""
    bounds = [(0.0, 5.0), (0.0, 5.0)]
    target = np.array([2.0, 3.0])

    def objective(x):
        return float(np.sum((x - target) ** 2))

    best_x, best_f = multistart_dual_annealing(
        objective=objective, bounds=bounds, n_restarts=1, maxiter=10,
        seed=0, plateau_patience=15, plateau_rel_tol=1e-3)

    assert np.isfinite(best_f)
    assert len(best_x) == 2


# ---------------------------------------------------------------------------
# restart_workers (phase H-4): dispatching Pass-2's per-restart
# dual_annealing calls across a process pool instead of one at a time.
# Pass 1 (x0 construction) always runs sequentially regardless of this
# value, so every restart's x0 -- and therefore the final result -- must
# be bit-identical whichever value is used.
# ---------------------------------------------------------------------------

def test_restart_workers_omitted_defaults_to_sequential():
    bounds = [(0.0, 10.0), (0.0, 10.0)]
    target = np.array([3.0, 7.0])

    def objective(x):
        return float(np.sum((x - target) ** 2))

    best_x, best_f = multistart_dual_annealing(
        objective=objective, bounds=bounds, n_restarts=2, maxiter=10, seed=0)

    assert np.isfinite(best_f)
    assert len(best_x) == 2


def test_restart_workers_parallel_matches_sequential():
    bounds = [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]
    target = np.array([7.3, 2.1, 5.5])
    objective = functools.partial(_module_level_bowl, target=target)

    x_seq, f_seq = multistart_dual_annealing(
        objective=objective, bounds=bounds, n_restarts=4,
        maxiter=15, seed=42, restart_workers=1)
    x_par, f_par = multistart_dual_annealing(
        objective=objective, bounds=bounds, n_restarts=4,
        maxiter=15, seed=42, restart_workers=3)

    assert f_par == f_seq
    assert np.array_equal(x_par, x_seq)
