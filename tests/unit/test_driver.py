"""Unit tests for mizuho.optimize.driver.optimize_one_assignment's
Phase-D coordination-shell-seed wiring (see optimize.init_guess.
_coordination_shell_seed_x0 and optimize.cma.multistart_cma_es's
seeded_x0 parameter for the mechanism itself).
"""
import pickle

import pytest

from mizuho.objective.energy_jax import JAX_AVAILABLE
from mizuho.optimize.driver import optimize_one_assignment
from mizuho.structure import resolve_fixed_assignment

# jax is an optional extra (mizuho[jax]); the two tests below drive the
# REAL refine_jac path rather than monkeypatching JAX_AVAILABLE, so
# without it they hit driver.py's deliberate "install mizuho[jax]"
# ImportError instead of the behaviour they are asserting. Skip, the
# same way test_lattice.py does for numba and test_mcmc.py for emcee,
# so `pip install -e ".[test]" && pytest` -- the command README.md
# gives -- is green on a plain test install.
_needs_jax = pytest.mark.skipif(not JAX_AVAILABLE, reason="jax not installed")

_TINY_OPT_CFG = {"n_restarts": 1, "cma_maxiter": 5, "refine_maxiter": 5}


def _nacl_like_config():
    return {
        "spacegroup": 225,
        "composition": "NaCl",
        "ionic_charges": {"Na": 1, "Cl": -1},
        "charge_tolerance_min": 0,
        "charge_tolerance_max": 0,
        "lattice_bounds": {"a": [4.5, 6.5]},
        "bond_constraints": [
            {"atom1": "Na", "atom2": "Cl", "distance_min": 2.5, "distance_max": 3.2},
        ],
    }


def _aso4_like_config():
    return {
        "spacegroup": 58,
        "composition": "AsO3",
        "lattice_bounds": {"a": [7.8, 8.8], "b": [8.0, 9.0], "c": [5.6, 6.5]},
        "bond_constraints": [
            {"atom1": "As", "atom2": "O", "distance_min": 1.60, "distance_max": 1.75},
        ],
        "coordination_constraints": [
            {"center_atom": "As", "neighbor_atom": "O",
             "cn_min": 4, "cn_max": 4, "cutoff_distance": 1.9},
        ],
    }


def test_shell_seed_not_attempted_for_configs_without_shape_constraints(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "mizuho.optimize.driver._coordination_shell_seed_x0",
        lambda x0, *a, **k: calls.append(1) or x0)

    asgn = resolve_fixed_assignment(
        225, [{"element": "Na", "letter": "a"}, {"element": "Cl", "letter": "b"}])
    optimize_one_assignment(
        asgn, _nacl_like_config(), _TINY_OPT_CFG, verbose=False, label="test")

    assert calls == []   # nacl-style config: no coordination/angle constraints to seed from


def test_shell_seed_attempted_for_configs_with_shape_constraints(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "mizuho.optimize.driver._coordination_shell_seed_x0",
        lambda x0, *a, **k: calls.append(1) or x0)

    asgn = resolve_fixed_assignment(
        58, [{"element": "As", "letter": "g"}, {"element": "O", "letter": "g"},
             {"element": "O", "letter": "g"}, {"element": "O", "letter": "h"}])
    optimize_one_assignment(
        asgn, _aso4_like_config(), _TINY_OPT_CFG, verbose=False, label="test")

    assert calls == [1]   # exactly one seed attempt, regardless of n_restarts


def test_shell_seed_info_is_forwarded_to_seed_function(monkeypatch):
    received = {}
    monkeypatch.setattr(
        "mizuho.optimize.driver._coordination_shell_seed_x0",
        lambda x0, bounds, cfg, optimize_fn, shell_seed_info=None, **k:
            received.update(info=shell_seed_info) or x0)

    asgn = resolve_fixed_assignment(
        58, [{"element": "As", "letter": "g"}, {"element": "O", "letter": "g"},
             {"element": "O", "letter": "g"}, {"element": "O", "letter": "h"}])
    sentinel = {0: [1.0, 2.0, 3.0]}
    optimize_one_assignment(
        asgn, _aso4_like_config(), _TINY_OPT_CFG, verbose=False, label="test",
        shell_seed_info=sentinel)

    assert received["info"] is sentinel


# ---------------------------------------------------------------------------
# functools.partial objective construction + restart_workers forwarding
# (phase H-4). The objective handed to Stage 1's optimize_fn must itself
# be picklable (a functools.partial around the module-level compute_error,
# NOT a local `def obj(pv): ...` closure) so restart-level parallelism
# inside a multistart_* backend can dispatch it across a process pool --
# see optimize.driver's own comment at the objective's construction site.
# ---------------------------------------------------------------------------

def test_objective_passed_to_stage1_is_picklable(monkeypatch):
    captured = {}

    def _fake_stage1(objective, bounds, **kwargs):
        captured["objective"] = objective
        lo = [b[0] for b in bounds]
        return lo, 1e6

    monkeypatch.setattr("mizuho.optimize.driver._resolve_backend", lambda name: _fake_stage1)

    asgn = resolve_fixed_assignment(
        225, [{"element": "Na", "letter": "a"}, {"element": "Cl", "letter": "b"}])
    optimize_one_assignment(
        asgn, _nacl_like_config(), _TINY_OPT_CFG, verbose=False, label="test")

    # Must not raise -- a local closure would fail here with
    # "Can't pickle local object".
    reloaded = pickle.loads(pickle.dumps(captured["objective"]))
    assert reloaded is not None


def test_restart_workers_defaults_to_one(monkeypatch):
    captured = {}

    def _fake_stage1(objective, bounds, **kwargs):
        captured["restart_workers"] = kwargs["restart_workers"]
        lo = [b[0] for b in bounds]
        return lo, 1e6

    monkeypatch.setattr("mizuho.optimize.driver._resolve_backend", lambda name: _fake_stage1)

    asgn = resolve_fixed_assignment(
        225, [{"element": "Na", "letter": "a"}, {"element": "Cl", "letter": "b"}])
    optimize_one_assignment(
        asgn, _nacl_like_config(), _TINY_OPT_CFG, verbose=False, label="test")

    assert captured["restart_workers"] == 1


def test_restart_workers_forwarded_from_opt_cfg(monkeypatch):
    captured = {}

    def _fake_stage1(objective, bounds, **kwargs):
        captured["restart_workers"] = kwargs["restart_workers"]
        lo = [b[0] for b in bounds]
        return lo, 1e6

    monkeypatch.setattr("mizuho.optimize.driver._resolve_backend", lambda name: _fake_stage1)

    asgn = resolve_fixed_assignment(
        225, [{"element": "Na", "letter": "a"}, {"element": "Cl", "letter": "b"}])
    opt_cfg = dict(_TINY_OPT_CFG, restart_workers=4)
    optimize_one_assignment(
        asgn, _nacl_like_config(), opt_cfg, verbose=False, label="test")

    assert captured["restart_workers"] == 4


# ---------------------------------------------------------------------------
# optimization.refine_jac (phase I-2): JAX-based exact gradient for the
# CN/volume terms, combined with finite-difference for everything else
# (see objective/energy_jax.py). These tests isolate the driver-level
# WIRING (config validation, jac= dispatch) from the underlying JAX
# machinery itself (already covered by test_energy_jax.py) by mocking
# make_gradient_fn/minimize, following the same "isolate the Stage-2
# loop from real optimiser non-determinism" pattern phase G-7's own
# tests used.
# ---------------------------------------------------------------------------

def test_refine_jac_raises_import_error_when_jax_unavailable(monkeypatch):
    monkeypatch.setattr("mizuho.optimize.driver._energy_jax.JAX_AVAILABLE", False)

    asgn = resolve_fixed_assignment(
        58, [{"element": "As", "letter": "g"}, {"element": "O", "letter": "g"},
             {"element": "O", "letter": "g"}, {"element": "O", "letter": "h"}])
    opt_cfg = dict(_TINY_OPT_CFG, refine_jac=True)
    with pytest.raises(ImportError, match=r"mizuho\[jax\]"):
        optimize_one_assignment(asgn, _aso4_like_config(), opt_cfg, verbose=False, label="test")


@_needs_jax
def test_refine_jac_raises_value_error_with_hybridisation_constraints():
    asgn = resolve_fixed_assignment(
        58, [{"element": "As", "letter": "g"}, {"element": "O", "letter": "g"},
             {"element": "O", "letter": "g"}, {"element": "O", "letter": "h"}])
    cfg = dict(_aso4_like_config(),
               hybridisation_constraints=[{"element": "As", "hybridisation": "sp3"}])
    opt_cfg = dict(_TINY_OPT_CFG, refine_jac=True)
    with pytest.raises(ValueError, match="hybridisation_constraints"):
        optimize_one_assignment(asgn, cfg, opt_cfg, verbose=False, label="test")


def test_refine_jac_false_by_default_does_not_call_make_gradient_fn(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "mizuho.optimize.driver._energy_jax.make_gradient_fn",
        lambda cfg, ref: calls.append(1))

    asgn = resolve_fixed_assignment(
        225, [{"element": "Na", "letter": "a"}, {"element": "Cl", "letter": "b"}])
    optimize_one_assignment(
        asgn, _nacl_like_config(), _TINY_OPT_CFG, verbose=False, label="test")

    assert calls == []   # refine_jac defaults to False: unchanged behaviour


def test_refine_jac_passed_only_to_jac_capable_methods(monkeypatch):
    """L-BFGS-B accepts jac=; Powell does not (scipy accepts jac= for
    Powell too but silently ignores it with a warning -- driver.py
    avoids even computing it in that case, see _JAC_CAPABLE_METHODS)."""
    sentinel_jac_fn = object()
    monkeypatch.setattr("mizuho.optimize.driver._energy_jax.JAX_AVAILABLE", True)
    monkeypatch.setattr(
        "mizuho.optimize.driver._energy_jax.make_gradient_fn",
        lambda cfg, ref: sentinel_jac_fn)

    received = []

    class _FakeResult:
        success = True
        status = 0
        message = "ok"

    def _fake_minimize(fun, x0, method=None, bounds=None, jac=None, options=None):
        received.append((method, jac))
        r = _FakeResult()
        r.x = x0
        r.fun = 1e10   # never improves on Stage 1 -- irrelevant to this test
        return r

    monkeypatch.setattr("mizuho.optimize.driver.minimize", _fake_minimize)

    asgn = resolve_fixed_assignment(
        58, [{"element": "As", "letter": "g"}, {"element": "O", "letter": "g"},
             {"element": "O", "letter": "g"}, {"element": "O", "letter": "h"}])
    opt_cfg = dict(_TINY_OPT_CFG, refine_jac=True,
                   refine_methods=["L-BFGS-B", "Powell"])
    optimize_one_assignment(asgn, _aso4_like_config(), opt_cfg, verbose=False, label="test")

    assert received == [("L-BFGS-B", sentinel_jac_fn), ("Powell", None)]


@_needs_jax
def test_refine_jac_real_end_to_end_does_not_crash():
    """Real (non-mocked) sanity check: refine_jac=True must not crash
    and must produce a finite, non-degenerate error on a real
    coordination_constraints config. test_energy_jax.py already covers
    the gradient's own numerical correctness in isolation; this only
    confirms the driver-level wiring works end-to-end."""
    asgn = resolve_fixed_assignment(
        58, [{"element": "As", "letter": "g"}, {"element": "O", "letter": "g"},
             {"element": "O", "letter": "g"}, {"element": "O", "letter": "h"}])
    opt_cfg = dict(_TINY_OPT_CFG, refine_jac=True)
    _, err, _ = optimize_one_assignment(
        asgn, _aso4_like_config(), opt_cfg, verbose=False, label="test")
    assert err < 1e9   # not the "optimisation failed" sentinel (also excludes NaN)


@pytest.mark.slow
def test_restart_workers_real_speedup_with_equal_error():
    """End-to-end sanity check (not mocked): restart_workers is
    designed for a single EXPENSIVE candidate (e.g. wyckoff_assignment
    skips Phase 1 entirely, so there is exactly one) -- this uses
    spinel_mgal2o4's real top-1 candidate at its OWN (non-trivial,
    ~86s/candidate at the time G-2 measured it) optimization: budget,
    unlike a cheap fixture (rutile_tio2, measured separately) where
    each individual restart is too fast for process-pool dispatch
    overhead to pay off -- restart_workers' value proposition
    specifically needs restarts expensive enough to amortise that
    overhead (see optimize.driver's own restart_workers docstring)."""
    import time

    import yaml

    from mizuho.pipeline import build_sg_config, search_and_rank

    with open("tests/fixtures/spinel_mgal2o4.yaml") as fh:
        base_config = yaml.safe_load(fh)
    sg_config = build_sg_config(base_config, base_config["spacegroup"])
    valid_asgns, _, shell_seeds = search_and_rank(sg_config, max_cands=1)
    asgn = valid_asgns[0]
    opt_cfg = dict(sg_config["optimization"])

    t0 = time.time()
    _, e_seq, _ = optimize_one_assignment(
        asgn, sg_config, opt_cfg, verbose=False, shell_seed_info=shell_seeds[0])
    t_seq = time.time() - t0

    t0 = time.time()
    _, e_par, _ = optimize_one_assignment(
        asgn, sg_config, dict(opt_cfg, restart_workers=4), verbose=False,
        shell_seed_info=shell_seeds[0])
    t_par = time.time() - t0

    print(f"\n  sequential: {t_seq:.2f}s error={e_seq:.6f}   "
          f"restart_workers=4: {t_par:.2f}s error={e_par:.6f}")
    assert e_par == pytest.approx(e_seq, abs=1e-6)
