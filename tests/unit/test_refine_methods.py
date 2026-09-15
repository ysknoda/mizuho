"""Unit tests for optimize.driver's Stage-2 refine_methods loop and its
_refine_options helper (phase G-7) -- see driver.py's own docstring
(YAML optimisation block keys / Stage 2 section) for the design.

Stage 1 (the backend search itself) is monkeypatched out via
_resolve_backend so these tests exercise ONLY the Stage-2 loop's own
improvement-gating logic in isolation, deterministically -- mirrors how
test_driver.py's existing tests isolate _coordination_shell_seed_x0
wiring the same way, rather than depending on real (non-deterministic-
budget) Stage-1 optimizer numerics.
"""
import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from mizuho.optimize.driver import _refine_options, optimize_one_assignment
from mizuho.structure import resolve_fixed_assignment

# Deliberately worse than every fake Stage-2 result used below, so the
# tests only ever exercise Stage 2's OWN comparisons against each other,
# never against Stage 1's.
_STAGE1_BAD_ERROR = 1e6
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


def _fake_stage1(objective, bounds, **kwargs):
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    return (lo + hi) / 2.0, _STAGE1_BAD_ERROR


class TestRefineOptions:
    def test_lbfgsb_gets_ftol_directly(self):
        assert _refine_options("L-BFGS-B", 500, 1e-9) == {"maxiter": 500, "ftol": 1e-9}

    def test_powell_gets_ftol_directly(self):
        assert _refine_options("Powell", 500, 1e-9) == {"maxiter": 500, "ftol": 1e-9}

    def test_nelder_mead_gets_ftol_mapped_to_fatol(self):
        """Nelder-Mead has no relative-tolerance option; the shared
        refine_ftol config key is mapped onto its closest analogue --
        the absolute fatol -- rather than silently dropped or passed
        through as an option Nelder-Mead does not recognise."""
        assert _refine_options("Nelder-Mead", 500, 1e-9) == {
            "maxiter": 500, "fatol": 1e-9}


def _install_fake_minimize(monkeypatch, fake_fun_by_method, x_value=5.5):
    """Stub Stage 2 with a SELF-CONSISTENT fake minimize: each method
    returns its own distinctive x, and the objective is stubbed to map
    that x back to exactly the fun value declared for that method, so
    `fun` really is the objective at `x` -- the contract a
    correctly-behaving scipy.optimize.minimize satisfies.

    This consistency matters: the Stage-2 loop re-evaluates the
    objective at ref.x rather than trusting ref.fun, because scipy's
    L-BFGS-B can return a `fun` belonging to a different point than the
    `x` it returns when it terminates abnormally (reproducible when the
    start point sits on a bound -- see driver.py's own comment). An
    intentionally inconsistent stub would therefore be testing the very
    pathology the loop now defends against, not its gating logic.
    """
    calls = []
    methods = list(fake_fun_by_method)
    x_by_method = {m: x_value + i for i, m in enumerate(methods)}
    fun_by_x = {x_by_method[m]: v for m, v in fake_fun_by_method.items()}

    def fake_minimize(fun, x0, method, bounds, options):
        calls.append(method)
        return OptimizeResult(
            x=np.full(len(x0), x_by_method[method]),
            fun=fake_fun_by_method[method],
            success=True, status=0, message="ok")

    def fake_compute_error(param_vector, config=None, verbose=False):
        return fun_by_x.get(float(param_vector[0]), _STAGE1_BAD_ERROR)

    monkeypatch.setattr("mizuho.optimize.driver.minimize", fake_minimize)
    monkeypatch.setattr(
        "mizuho.optimize.driver.compute_error", fake_compute_error)
    monkeypatch.setattr(
        "mizuho.optimize.driver._resolve_backend", lambda name: _fake_stage1)
    return calls


def test_stage2_result_is_re_evaluated_rather_than_taken_from_ref_fun(monkeypatch):
    """Regression test for the ref.fun/ref.x inconsistency documented in
    the mizuho plan's phase I-2 and left unfixed there.

    scipy's L-BFGS-B can terminate ABNORMAL (status 2) with a `fun` that
    does not belong to the `x` it returns -- reproducible whenever the
    start point sits exactly on a bound, which is precisely what Stage 1
    hands over when the optimum is pinned at a lattice_bounds edge. Here
    the stub reports a fun of 0.001 (better than anything) while its x
    is actually worth 42.0. Accepting the stale 0.001 would return a
    best_e that the returned structure does not achieve, breaking the
    invariant that compute_error(best_p) == best_e -- which candidate
    ranking and error_breakdown both depend on.
    """
    def fake_minimize(fun, x0, method, bounds, options):
        return OptimizeResult(x=np.full(len(x0), 7.25), fun=0.001,
                              success=False, status=2, message="ABNORMAL: ")

    def fake_compute_error(param_vector, config=None, verbose=False):
        return 42.0 if float(param_vector[0]) == 7.25 else _STAGE1_BAD_ERROR

    monkeypatch.setattr("mizuho.optimize.driver.minimize", fake_minimize)
    monkeypatch.setattr(
        "mizuho.optimize.driver.compute_error", fake_compute_error)
    monkeypatch.setattr(
        "mizuho.optimize.driver._resolve_backend", lambda name: _fake_stage1)

    best_p, best_e, _ = optimize_one_assignment(
        _nacl_assignment(), _nacl_like_config(), _TINY_OPT_CFG,
        verbose=False, label="test")

    # 42.0 still beats Stage 1's deliberately terrible error, so the
    # refinement IS accepted -- but at its true value, not the stale one.
    assert best_e == pytest.approx(42.0)
    assert best_p[0] == pytest.approx(7.25)
    assert best_e != pytest.approx(0.001)


def test_stage2_keeps_a_refinement_whose_ref_fun_is_overstated(monkeypatch):
    """The same staleness can also run the other way: a `fun` worse than
    the point's true value would silently discard a real improvement.
    Re-evaluating fixes both directions, so assert the accept case too."""
    def fake_minimize(fun, x0, method, bounds, options):
        return OptimizeResult(x=np.full(len(x0), 7.25),
                              fun=_STAGE1_BAD_ERROR * 10,
                              success=False, status=2, message="ABNORMAL: ")

    def fake_compute_error(param_vector, config=None, verbose=False):
        return 0.5 if float(param_vector[0]) == 7.25 else _STAGE1_BAD_ERROR

    monkeypatch.setattr("mizuho.optimize.driver.minimize", fake_minimize)
    monkeypatch.setattr(
        "mizuho.optimize.driver.compute_error", fake_compute_error)
    monkeypatch.setattr(
        "mizuho.optimize.driver._resolve_backend", lambda name: _fake_stage1)

    _, best_e, _ = optimize_one_assignment(
        _nacl_assignment(), _nacl_like_config(), _TINY_OPT_CFG,
        verbose=False, label="test")

    assert best_e == pytest.approx(0.5)


def test_default_refine_methods_runs_only_lbfgs_b(monkeypatch):
    """Omitting refine_methods entirely must behave exactly as every
    prior phase did: exactly one minimize() call, method='L-BFGS-B'."""
    calls = _install_fake_minimize(monkeypatch, {"L-BFGS-B": 1.0})

    optimize_one_assignment(
        _nacl_assignment(), _nacl_like_config(), _TINY_OPT_CFG,
        verbose=False, label="test")

    assert calls == ["L-BFGS-B"]


def test_refine_methods_are_tried_in_the_configured_order(monkeypatch):
    calls = _install_fake_minimize(monkeypatch, {
        "L-BFGS-B": 1.0, "Nelder-Mead": 1.0, "Powell": 1.0})

    opt_cfg = dict(_TINY_OPT_CFG, refine_methods=["L-BFGS-B", "Nelder-Mead", "Powell"])
    optimize_one_assignment(
        _nacl_assignment(), _nacl_like_config(), opt_cfg, verbose=False, label="test")

    assert calls == ["L-BFGS-B", "Nelder-Mead", "Powell"]


def test_a_later_method_can_improve_on_an_earlier_ones_result(monkeypatch):
    """A later refine_methods entry whose result has a LOWER objective
    value than an earlier one's must win, regardless of list order."""
    _install_fake_minimize(monkeypatch, {
        "L-BFGS-B": 5.0, "Nelder-Mead": 1.0, "Powell": 3.0})

    opt_cfg = dict(_TINY_OPT_CFG, refine_methods=["L-BFGS-B", "Nelder-Mead", "Powell"])
    _, best_e, structure = optimize_one_assignment(
        _nacl_assignment(), _nacl_like_config(), opt_cfg, verbose=False, label="test")

    assert best_e == pytest.approx(1.0)
    assert structure is not None


def test_a_non_improving_method_does_not_overwrite_a_better_result(monkeypatch):
    _install_fake_minimize(monkeypatch, {"L-BFGS-B": 1.0, "Nelder-Mead": 50.0})

    opt_cfg = dict(_TINY_OPT_CFG, refine_methods=["L-BFGS-B", "Nelder-Mead"])
    _, best_e, _ = optimize_one_assignment(
        _nacl_assignment(), _nacl_like_config(), opt_cfg, verbose=False, label="test")

    assert best_e == pytest.approx(1.0)


def test_empty_refine_methods_leaves_stage1_result_untouched(monkeypatch):
    """An empty list is a legitimate (if unusual) way to disable Stage 2
    entirely -- the loop simply never executes, and Stage 1's own
    result passes through unchanged."""
    calls = _install_fake_minimize(monkeypatch, {})

    opt_cfg = dict(_TINY_OPT_CFG, refine_methods=[])
    _, best_e, structure = optimize_one_assignment(
        _nacl_assignment(), _nacl_like_config(), opt_cfg, verbose=False, label="test")

    assert calls == []
    assert best_e == pytest.approx(_STAGE1_BAD_ERROR)
    assert structure is not None


@pytest.mark.slow
def test_nelder_mead_and_powell_run_for_real_on_the_nacl_fixture(nacl_config):
    """Not-mocked, end-to-end sanity check (unlike the deterministic
    tests above): both new refine_methods options must actually run to
    completion via scipy without raising, and must not make the final
    result WORSE than plain L-BFGS-B alone on this easy fixture."""
    from mizuho.pipeline import build_sg_config
    from mizuho.search.dfs import wyckoff_tree_search

    sg_config = build_sg_config(nacl_config, nacl_config["spacegroup"])
    valid = wyckoff_tree_search(sg_config)
    asgn = next(a for a in valid if len({x.entry.letter for x in a}) == 2)

    baseline_cfg = dict(sg_config["optimization"])
    _, baseline_e, _ = optimize_one_assignment(
        asgn, sg_config, baseline_cfg, verbose=False, label="baseline")

    combined_cfg = dict(sg_config["optimization"],
                         refine_methods=["L-BFGS-B", "Nelder-Mead", "Powell"])
    _, combined_e, structure = optimize_one_assignment(
        asgn, sg_config, combined_cfg, verbose=False, label="combined")

    assert structure is not None
    assert combined_e <= baseline_e + 1e-9
