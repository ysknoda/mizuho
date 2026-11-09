"""optimization.cn_stop (ADR-0041): a Stage-1 restart whose best point
violates an integer coordination range stops once the best
(cn_deficit, soft CN term) has not improved for K generations."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import replay_early_stop as rp

from mizuho.errors import ConfigError
from mizuho.optimize.cma import _cma_es, multistart_cma_es
from mizuho.optimize.driver import cn_stop_patience, optimize_assignment

_BOUNDS = [(0.0, 1.0)] * 5


def _rastrigin(x):
    y = 6.0 * (np.asarray(x) - 0.3)
    return float(10 * len(y) + np.sum(y**2 - 10 * np.cos(2 * np.pi * y)))


class _Objective:
    """A synthetic objective with a coordination deficit and a soft CN term,
    logging every evaluation."""

    def __init__(self, deficit):
        self.deficit = deficit
        self.log = []

    def __call__(self, x, terms=None):
        f = _rastrigin(x)
        d = self.deficit(x)
        cn = round(f, 1)  # coarse, so the soft term plateaus
        self.log.append((f, d, cn))
        if terms is not None:
            terms.update(cn_deficit=d, cn=cn)
        return f


def _run(objective, patience, **kw):
    return _cma_es(
        objective,
        np.full(5, 0.8),
        _BOUNDS,
        sigma0=0.3,
        maxiter=200,
        tol=1e-12,
        seed=7,
        cn_stop_patience=patience,
        return_stop=True,
        **kw,
    )


def _always_violating(x):
    return 1 + int(x[0] > 0.9)


@pytest.mark.parametrize("k", [3, 5, 10])
def test_the_optimiser_stops_where_the_replay_says(k):
    full = _Objective(_always_violating)
    _, _, g_full = _run(full, None)
    assert g_full is None
    stopped = _Objective(_always_violating)
    x, f, g = _run(stopped, k)
    f_all, d_all, cn_all = (np.array(c) for c in zip(*full.log))
    popsize = 4 + int(3 * np.log(5))
    last, best_d = rp.last_improvement(
        f_all, d_all, cn_all, popsize, "deficit_cn", 1e-3
    )
    assert g is not None and g == rp.stop_generation(last, best_d, k)
    # a stopped restart is the prefix of the full one, and returns its best
    assert stopped.log == full.log[: len(stopped.log)]
    assert len(stopped.log) == 1 + g * popsize
    assert f == min(e[0] for e in stopped.log)


def test_never_stops_once_a_point_meets_every_range():
    full = _Objective(lambda x: 0)
    on = _Objective(lambda x: 0)
    x_full, f_full, _ = _run(full, None)
    x_on, f_on, g = _run(on, 1)
    assert g is None
    assert on.log == full.log and f_on == f_full


def test_off_calls_the_objective_with_the_vector_alone():
    calls = []

    def objective(x):  # no `terms` keyword: must not be passed one
        calls.append(_rastrigin(x))
        return calls[-1]

    x, f = _cma_es(objective, np.full(5, 0.8), _BOUNDS, sigma0=0.3, maxiter=20, seed=7)
    assert f == min(calls)


def test_multistart_reports_every_restart():
    stops = []
    multistart_cma_es(
        _Objective(_always_violating),
        _BOUNDS,
        n_restarts=3,
        maxiter=200,
        seed=1,
        cn_stop_patience=5,
        restart_stops=stops,
    )
    assert len(stops) == 3 and all(isinstance(g, int) for g in stops)


def test_patience_from_the_config():
    assert cn_stop_patience({}, 12) is None
    assert cn_stop_patience({"cn_stop": {}}, 12) == (108, 1e-3)
    assert cn_stop_patience({"cn_stop": {}}, 1) == (15, 1e-3)
    assert cn_stop_patience(
        {"cn_stop": {"patience_factor": 2, "min_patience": 5, "rel_tol": 0.01}}, 4
    ) == (8, 0.01)


@pytest.mark.parametrize(
    "block",
    [
        [],
        {"patience": 3},
        {"patience_factor": 0},
        {"patience_factor": True},
        {"min_patience": 2.5},
        {"rel_tol": -1},
    ],
)
def test_a_bad_block_is_a_config_error(block):
    with pytest.raises(ConfigError):
        cn_stop_patience({"cn_stop": block}, 4)


_BASE = {
    "elements": ["Al", "O", "N"],
    "ionic_charges": {"Al": 3, "O": -2, "N": -3},
    "charge_tolerance_min": 0,
    "charge_tolerance_max": 0,
    "coordination_numbers": {"Al": [4, 6]},
    "anion_coordination_numbers": {"O": [2, 4], "N": [3, 4]},
    "density_g_cm3": [3.26, 4.10],
}
_SPECS = [{"element": "Al", "letter": "b"}, {"element": "N", "letter": "b"}]


def _assignment_results(cn_stop):
    from mizuho.pipeline import build_sg_config
    from mizuho.structure import resolve_fixed_assignment

    cfg = build_sg_config(_BASE, 186, composition="AlN")
    opt = {"n_restarts": 3, "cma_maxiter": 30, "refine_maxiter": 5, "seed": 42}
    if cn_stop is not None:
        opt["cn_stop"] = cn_stop
    return optimize_assignment(resolve_fixed_assignment(186, _SPECS), cfg, opt)


def test_the_assignment_result_carries_the_stops():
    off = _assignment_results(None)
    assert off and all(r.cn_stopped is None for r in off)
    on = _assignment_results({"patience_factor": 0.1, "min_patience": 1})
    assert on and all(isinstance(r.cn_stopped, tuple) for r in on)
