"""Unit tests for optimize.driver's plateau_patience/plateau_rel_tol
wiring into optimize_one_assignment's Stage 1 call (phase G-8).

Stage 1's own backend function is captured via a monkeypatched
_resolve_backend so the mocked tests below verify ONLY the wiring
(which opt_cfg keys reach which Stage-1 kwargs), not multistart_cma_es's
own plateau mechanism -- that mechanism itself is already covered
directly by test_cma.py's own
test_plateau_patience_reduces_evaluations_on_an_easy_objective, and
re-verifying it here would be duplicate coverage of the same underlying
code (mirrors test_backend_resolution.py's own reasoning for the same
kind of layering).
"""
import numpy as np
import pytest

from mizuho.optimize.driver import optimize_one_assignment
from mizuho.structure import resolve_fixed_assignment

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


def _nacl_assignment():
    return resolve_fixed_assignment(
        225, [{"element": "Na", "letter": "a"}, {"element": "Cl", "letter": "b"}])


def _install_capturing_backend(monkeypatch):
    """Stand in for _resolve_backend's result, capturing every keyword
    Stage 1 actually passes rather than running any real optimiser."""
    captured = {}

    def fake_backend(objective, bounds, **kwargs):
        captured.update(kwargs)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        return (lo + hi) / 2.0, 1.0

    monkeypatch.setattr(
        "mizuho.optimize.driver._resolve_backend", lambda name: fake_backend)
    return captured


def test_plateau_patience_defaults_to_none(monkeypatch):
    """Omitting plateau_patience entirely must preserve every prior
    phase's behaviour exactly: Stage 1 receives None, which
    multistart_cma_es's own docstring documents as disabling early
    stopping entirely."""
    captured = _install_capturing_backend(monkeypatch)

    optimize_one_assignment(
        _nacl_assignment(), _nacl_like_config(), _TINY_OPT_CFG,
        verbose=False, label="test")

    assert captured["plateau_patience"] is None
    assert captured["plateau_rel_tol"] == 1e-3


def test_explicit_plateau_patience_and_rel_tol_are_forwarded(monkeypatch):
    captured = _install_capturing_backend(monkeypatch)

    opt_cfg = dict(_TINY_OPT_CFG, plateau_patience=15, plateau_rel_tol=1e-2)
    optimize_one_assignment(
        _nacl_assignment(), _nacl_like_config(), opt_cfg, verbose=False, label="test")

    assert captured["plateau_patience"] == 15
    assert captured["plateau_rel_tol"] == 1e-2


@pytest.mark.slow
def test_plateau_patience_does_not_worsen_the_final_error_on_a_real_fixture(rutile_tio2_config):
    """Not-mocked, end-to-end sanity check (unlike the wiring tests
    above): enabling plateau_patience through a real fixture's opt_cfg
    must not raise, and must not make the final result WORSE than the
    default (disabled) behaviour -- mirrors test_refine_methods.py's own
    equivalent real-fixture check. (This project's own dev-environment
    measurement for this exact fixture: baseline 1.47s vs plateau 0.73s,
    error identical to 6 decimal places -- see phase G-8's plan-file
    writeup; this test checks correctness, not the timing itself, since
    asserting wall-clock speedups is inherently flaky.)"""
    from mizuho.pipeline import build_sg_config, search_and_rank

    sg_config = build_sg_config(rutile_tio2_config, rutile_tio2_config["spacegroup"])
    valid_asgns, n_found, shell_seeds = search_and_rank(sg_config, max_cands=1, verbose=False)
    asgn, seed_info = valid_asgns[0], shell_seeds[0]

    baseline_cfg = dict(sg_config["optimization"])
    _, baseline_e, _ = optimize_one_assignment(
        asgn, sg_config, baseline_cfg, verbose=False, label="baseline",
        shell_seed_info=seed_info)

    plateau_cfg = dict(sg_config["optimization"], plateau_patience=15, plateau_rel_tol=1e-3)
    _, plateau_e, structure = optimize_one_assignment(
        asgn, sg_config, plateau_cfg, verbose=False, label="plateau",
        shell_seed_info=seed_info)

    assert structure is not None
    assert plateau_e <= baseline_e + 1e-9
