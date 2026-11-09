"""scripts/replay_early_stop.py: replaying a stopping rule on a full
trajectory gives exactly what the optimiser does with the rule on."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import replay_early_stop as rp

from mizuho.optimize.cma import _cma_es


def _rastrigin(x):
    y = 6.0 * (np.asarray(x) - 0.3)
    return float(10 * len(y) + np.sum(y**2 - 10 * np.cos(2 * np.pi * y)))


def _run(patience, seed=7, maxiter=200):
    calls = []

    def obj(x):
        f = _rastrigin(x)
        calls.append(f)
        return f

    bounds = [(0.0, 1.0)] * 5
    best_x, best_f = _cma_es(
        obj,
        np.full(5, 0.8),
        bounds,
        sigma0=0.3,
        maxiter=maxiter,
        tol=1e-12,
        seed=seed,
        plateau_patience=patience,
        plateau_rel_tol=1e-3,
    )
    return np.array(calls), best_f


def test_a_stopped_restart_is_a_prefix_of_the_full_run():
    full, _ = _run(None)
    stopped, best_f = _run(10)
    assert len(stopped) < len(full)
    assert np.array_equal(stopped, full[: len(stopped)])
    assert best_f == stopped.min()


def test_plateau_replay_matches_the_optimiser():
    full, _ = _run(None)
    popsize = 4 + int(3 * np.log(5))
    for patience in (5, 10, 20):
        stopped, _ = _run(patience)
        g = rp.plateau_stop_generation(full, popsize, patience, 1e-3)
        assert g is not None
        assert 1 + g * popsize == len(stopped)


def test_deficit_rule_stops_k_generations_after_the_last_improvement():
    popsize = 2
    # generation:      0   1     2     3     4     5     6
    deficit = np.array([5, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3])
    f = np.linspace(10, 1, len(deficit))  # total keeps falling
    cn = np.full(len(deficit), 1.0)
    last, best_d = rp.last_improvement(f, deficit, cn, popsize, "deficit", 1e-3)
    assert list(last) == [0, 1, 1, 3, 3, 3, 3]
    assert rp.stop_generation(last, best_d, 3) == 6
    assert rp.stop_generation(last, best_d, 4) is None
    # the (deficit, total) rule sees the falling total and never stops
    last_f, best_f = rp.last_improvement(f, deficit, cn, popsize, "deficit_f", 1e-3)
    assert rp.stop_generation(last_f, best_f, 2) is None


def test_no_stop_once_the_deficit_is_zero():
    popsize = 1
    deficit = np.array([2, 0, 0, 0, 0, 0])
    f = np.full(len(deficit), 0.5)
    last, best_d = rp.last_improvement(f, deficit, f, popsize, "deficit", 1e-3)
    assert rp.stop_generation(last, best_d, 1) is None
