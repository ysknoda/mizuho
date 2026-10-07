"""Stage-1 endpoints grouped by topology class, and early stopping
(optimize.endpoints, ADR-0019)."""

import json

import numpy as np
import pytest
import yaml

from mizuho.data import WYCKOFF_DB
from mizuho.io.cif import cif_to_mizuho
from mizuho.optimize import endpoints
from mizuho.optimize.driver import _resolve_backend, optimize_assignment
from mizuho.optimize.endpoints import (
    EndpointLedger, GroupingSettings, grouping_settings, run_restarts, topology_key,
)
from mizuho.pipeline import build_sg_config, merge_same_topology
from mizuho.structure import SiteAssignment, assignments_to_sites, build_bounds

FIX = "tests/fixtures"


def _quartz():
    """Published alpha-quartz and its round-4 zero-error impostor: the same
    topology class, both at error 0 (ADR-0017/0019)."""
    ref = cif_to_mizuho(f"{FIX}/reference/quartz_sg152.cif")
    cfg = build_sg_config(yaml.safe_load(open(f"{FIX}/quartz_sio2.yaml")), ref.spacegroup)
    cfg["wyckoff_sites"] = ref.config()["wyckoff_sites"]
    imp = json.load(open(f"{FIX}/impostors/quartz_sio2_round4_volume.json"))["param_vector"]
    return cfg, np.array(ref.param_vector), np.array(imp)


def _nacl_with_cutoff(nacl_config):
    """Rock salt with a coordination cutoff, so endpoints have a class:
    Na-Cl first shell a/2 = 2.82 A, second a*sqrt(3)/2 = 4.88 A at a = 5.64."""
    cfg = dict(nacl_config)
    cfg["coordination_constraints"] = [
        {"center_atom": "Na", "neighbor_atom": "Cl", "cn_min": 6, "cn_max": 6,
         "cutoff_distance": 3.5, "weight": 1.0}]
    na = next(e for e in WYCKOFF_DB[225] if e.letter == "a")
    cl = next(e for e in WYCKOFF_DB[225] if e.letter == "b")
    return cfg, [SiteAssignment("Na", na), SiteAssignment("Cl", cl)]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def test_settings_defaults():
    assert grouping_settings({}) == GroupingSettings(1e-3, True, 2)
    assert grouping_settings({"topology_grouping": None}) == GroupingSettings()


def test_settings_read_from_yaml_keys():
    s = grouping_settings({"topology_grouping": {
        "zero_error_threshold": 5e-3, "early_stop": False, "early_stop_hits": 3}})
    assert s == GroupingSettings(5e-3, False, 3)


@pytest.mark.parametrize("block, message", [
    ({"zero_error_threshold": 0}, "positive number"),
    ({"zero_error_threshold": -1e-3}, "positive number"),
    ({"zero_error_threshold": "1e-3"}, "positive number"),
    ({"zero_error_threshold": True}, "positive number"),
    ({"early_stop": "yes"}, "true or false"),
    ({"early_stop_hits": 0}, "positive integer"),
    ({"early_stop_hits": 1.5}, "positive integer"),
    ({"early_stop_hits": True}, "positive integer"),
    ({"zero_error_treshold": 1e-3}, "unknown key"),
])
def test_settings_reject_bad_values(block, message):
    with pytest.raises(ValueError, match=message):
        grouping_settings({"topology_grouping": block})


def test_settings_reject_a_non_mapping():
    with pytest.raises(ValueError, match="mapping"):
        grouping_settings({"topology_grouping": 1e-3})


def test_bad_settings_fail_loudly_in_the_driver(nacl_config):
    cfg, asgn = _nacl_with_cutoff(nacl_config)
    opt = dict(nacl_config["optimization"], topology_grouping={"early_stop_hits": 0})
    with pytest.raises(ValueError, match="early_stop_hits"):
        optimize_assignment(asgn, cfg, opt)


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

def test_published_quartz_and_its_impostor_share_a_class():
    cfg, pub, imp = _quartz()
    assert topology_key(pub, cfg) is not None
    assert topology_key(pub, cfg) == topology_key(imp, cfg)


def test_two_zero_error_endpoints_of_one_class_stop_the_search():
    cfg, pub, imp = _quartz()
    ledger = EndpointLedger(cfg)
    assert ledger.add(pub, 0.0) is False
    assert ledger.add(imp, 0.0) is True
    (cls,) = ledger.classes.values()
    assert (cls.n_endpoints, cls.n_zero) == (2, 2)


def test_endpoints_above_the_threshold_are_grouped_but_never_stop():
    cfg, pub, imp = _quartz()
    ledger = EndpointLedger(cfg)
    assert not ledger.add(pub, 0.5) and not ledger.add(imp, 0.5)
    (cls,) = ledger.classes.values()
    assert (cls.n_endpoints, cls.n_zero) == (2, 0)


def test_the_threshold_is_read_from_the_settings():
    cfg, pub, imp = _quartz()
    ledger = EndpointLedger(cfg, GroupingSettings(zero_error_threshold=1e-2))
    ledger.add(pub, 5e-3)
    assert ledger.add(imp, 5e-3) is True


def test_early_stop_off_never_stops():
    cfg, pub, imp = _quartz()
    ledger = EndpointLedger(cfg, GroupingSettings(early_stop=False))
    for _ in range(4):
        assert ledger.add(pub, 0.0) is False
    assert next(iter(ledger.classes.values())).n_zero == 4


def test_a_class_keeps_its_lowest_endpoint():
    cfg, pub, imp = _quartz()
    ledger = EndpointLedger(cfg, GroupingSettings(early_stop=False))
    ledger.add(imp, 0.2)
    ledger.add(pub, 0.1)
    (cls,) = ledger.classes.values()
    assert cls.best_f == 0.1 and np.allclose(cls.best_x, pub)


def test_a_config_without_cutoffs_has_no_classes(nacl_config):
    na = next(e for e in WYCKOFF_DB[225] if e.letter == "a")
    cl = next(e for e in WYCKOFF_DB[225] if e.letter == "b")
    cfg = dict(nacl_config, wyckoff_sites=assignments_to_sites(
        [SiteAssignment("Na", na), SiteAssignment("Cl", cl)]))
    ledger = EndpointLedger(cfg)
    assert not ledger.enabled
    assert ledger.add(np.array([5.64]), 0.0) is False
    assert ledger.add(np.array([5.64]), 0.0) is False and not ledger.classes


def test_fallback_is_used_only_without_a_converged_class():
    cfg, pub, imp = _quartz()
    ledger = EndpointLedger(cfg)
    endpoints.add_fallback(ledger, [(pub, 0.3), (imp, 0.4)])
    (cls,) = ledger.classes.values()
    assert cls.converged is False and cls.n_endpoints == 2
    ledger2 = EndpointLedger(cfg)
    ledger2.add(pub, 0.0)
    endpoints.add_fallback(ledger2, [(imp, 0.4)])
    assert next(iter(ledger2.classes.values())).n_endpoints == 1


# ---------------------------------------------------------------------------
# run_restarts: the stop point does not depend on restart_workers
# ---------------------------------------------------------------------------

class _CountingLedger:
    """Stops once `stop_at` endpoints have been added."""
    enabled = True

    def __init__(self, stop_at, early_stop=True):
        self.settings = GroupingSettings(early_stop=early_stop)
        self.stop_at, self.seen = stop_at, []

    def add(self, x, f, converged=True):
        self.seen.append(f)
        return len(self.seen) >= self.stop_at


def _sequential(fn, tasks, max_workers=None, on_result=None):
    return [fn(t) for t in tasks]


def _square(task):
    return np.array([task]), float(task)


@pytest.mark.parametrize("workers", [1, 2, 3, 8])
def test_stop_point_is_independent_of_the_wave_size(monkeypatch, workers):
    monkeypatch.setattr(endpoints, "run_parallel_tasks", _sequential)
    kept = run_restarts(_square, list(range(8)), workers, _CountingLedger(stop_at=3))
    assert [f for _, f in kept] == [0.0, 1.0, 2.0]


def test_without_a_stop_every_restart_runs(monkeypatch):
    monkeypatch.setattr(endpoints, "run_parallel_tasks", _sequential)
    for ledger in (None, _CountingLedger(stop_at=99), _CountingLedger(2, early_stop=False)):
        kept = run_restarts(_square, list(range(5)), 2, ledger)
        assert [f for _, f in kept] == [0.0, 1.0, 2.0, 3.0, 4.0]


# ---------------------------------------------------------------------------
# Every backend hands its endpoints to the ledger and stops on it
# ---------------------------------------------------------------------------

BACKENDS = ["cma", "differential_evolution", "basin_hopping", "dual_annealing",
            "mcmc", "tpe"]


def _backend_call(backend, cfg, asgn, ledger, n_restarts=6, maxiter=20):
    full = dict(cfg, wyckoff_sites=assignments_to_sites(asgn))
    from functools import partial
    from mizuho.objective.energy import compute_error
    return _resolve_backend(backend)(
        objective=partial(compute_error, config=full), bounds=build_bounds(full),
        n_restarts=n_restarts, maxiter=maxiter, seed=7, config=full,
        periodic_mask=np.array([False]), endpoint_ledger=ledger)


@pytest.mark.parametrize("backend", BACKENDS)
def test_every_backend_classifies_and_stops_early(nacl_config, backend):
    cfg, asgn = _nacl_with_cutoff(nacl_config)
    full = dict(cfg, wyckoff_sites=assignments_to_sites(asgn))
    ledger = EndpointLedger(full)
    x, f = _backend_call(backend, cfg, asgn, ledger)
    assert f <= 1e-3
    assert ledger.stopped
    assert len(ledger.classes) >= 1
    best = ledger.ranked()[0]
    assert best.n_zero >= 2 and best.converged
    assert topology_key(x, full) == best.key
    if backend in ("cma", "differential_evolution", "basin_hopping", "dual_annealing"):
        assert ledger.n_added < 6         # restarts were skipped


@pytest.mark.parametrize("backend", ["cma", "differential_evolution",
                                     "basin_hopping", "dual_annealing"])
def test_without_early_stop_a_restart_backend_returns_the_same_point(nacl_config, backend):
    """Handing endpoints to a ledger that never stops changes nothing."""
    cfg, asgn = _nacl_with_cutoff(nacl_config)
    full = dict(cfg, wyckoff_sites=assignments_to_sites(asgn))
    x0, f0 = _backend_call(backend, cfg, asgn, None, n_restarts=3)
    ledger = EndpointLedger(full, GroupingSettings(early_stop=False))
    x1, f1 = _backend_call(backend, cfg, asgn, ledger, n_restarts=3)
    assert f1 == f0 and np.array_equal(x1, x0)
    assert ledger.n_added == 3


def test_driver_reports_the_class_and_the_early_stop(nacl_config):
    cfg, asgn = _nacl_with_cutoff(nacl_config)
    opt = dict(nacl_config["optimization"], n_restarts=6)
    (best,) = optimize_assignment(asgn, cfg, opt)
    assert best.error <= 1e-3 and best.topology is not None
    assert best.stopped_early and best.n_stage1 == 2 and best.n_zero == 2
    off = dict(opt, topology_grouping={"early_stop": False})
    (full,) = optimize_assignment(asgn, cfg, off)
    assert not full.stopped_early and full.n_stage1 == 6


# ---------------------------------------------------------------------------
# Ranking: one row per class
# ---------------------------------------------------------------------------

def test_ranking_merges_entries_of_one_class():
    cfg, pub, imp = _quartz()
    entries = [(0.0, pub), (0.0, imp), (0.7, None)]
    kept = merge_same_topology(entries, lambda e: e[1], lambda e: cfg)
    assert [(e[0], n) for e, n in kept] == [(0.0, 1), (0.7, 0)]
