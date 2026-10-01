"""Unit tests for mizuho.optimize.cma.multistart_cma_es's seeded_x0
restart slot (Phase D).
"""
import functools

import numpy as np
import pytest

from mizuho.optimize.cma import multistart_cma_es


def _module_level_bowl(x, target):
    """A module-level (picklable) objective -- required for
    restart_workers>1 tests below, since a process pool cannot pickle
    a local closure (unlike every other test in this file, which uses
    a local `def objective(x): ...` closure because restart_workers
    defaults to 1 -- fully sequential, no pickling involved)."""
    return float(np.sum((x - target) ** 2))


def test_seeded_x0_restart_can_become_the_global_best():
    """A deterministic check that restart 2 actually starts from
    seeded_x0: the objective's unique global minimum is placed exactly
    AT seeded_x0, so the very first (pre-generation) evaluation of that
    restart already achieves the best possible score -- no later
    generation, in this or any other restart, can beat it. With
    maxiter=1 there is essentially no room for CMA-ES's own search to
    "accidentally" find this exact point on its own, so recovering it
    demonstrates the seed was genuinely used as that restart's x0."""
    bounds = [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]
    target = np.array([7.3, 2.1, 5.5])   # deliberately off the box centre (5,5,5)

    def objective(x):
        return float(np.sum((x - target) ** 2))

    best_x, best_f = multistart_cma_es(
        objective=objective, bounds=bounds, n_restarts=3,
        maxiter=1, seed=0, config=None, seeded_x0=target)

    assert best_f == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(best_x, target, atol=1e-6)


def test_seeded_x0_none_behaves_like_a_normal_restart():
    """seeded_x0 omitted (the default) must NOT achieve the seeded
    restart's special result -- confirms the slot falls through to an
    ordinary random restart, identical in kind to the pre-Phase-D
    restart loop, instead of somehow still finding the seeded target.

    (multistart_cma_es is not reproducible call-to-call for a fixed
    seed -- see test_seeded_x0_ignored_when_fewer_than_three_restarts'
    docstring -- so this checks a single-call property rather than
    comparing two calls' numeric results.)
    """
    bounds = [(0.0, 10.0), (0.0, 10.0)]
    target = np.array([9.9, 9.9])

    def objective(x):
        return float(np.sum((x - target) ** 2))

    _, best_f = multistart_cma_es(
        objective=objective, bounds=bounds, n_restarts=3,
        maxiter=1, seed=0, config=None, seeded_x0=None)

    # A LOOSE threshold here would be flaky: with sigma0 large relative
    # to a small bounds box, a single generation's population can
    # occasionally land within "somewhat close" of any fixed point by
    # sheer luck (observed empirically with a threshold of 1.0). Landing
    # within sqrt(1e-4)=0.01 units of an exact point purely by chance is
    # a different matter entirely -- effectively impossible for a
    # continuous random draw, so this stays a reliable, non-flaky check.
    assert best_f > 1e-4


# ---------------------------------------------------------------------------
# plateau_patience / plateau_rel_tol: a practical, external "stop this
# restart once best_f has stopped meaningfully improving" check, distinct
# from cma's own tolfun/tolx (fixed at a very tight 1e-8 here -- see
# _cma_es's docstring -- which essentially never fires for this project's
# physically-scaled objectives, so without this a restart always burns its
# full maxiter). None (the default) must leave existing behaviour
# untouched -- see geometry.local_shell.solve_local_shell, the only
# caller that currently enables this.
# ---------------------------------------------------------------------------

def test_plateau_patience_none_does_not_change_the_call_signature():
    """Sanity check that omitting plateau_patience entirely (the default)
    still runs to completion without error -- i.e. this is a genuinely
    optional, backward-compatible addition."""
    bounds = [(0.0, 10.0), (0.0, 10.0)]
    target = np.array([3.0, 7.0])

    def objective(x):
        return float(np.sum((x - target) ** 2))

    best_x, best_f = multistart_cma_es(
        objective=objective, bounds=bounds, n_restarts=1,
        maxiter=20, seed=0, config=None)

    assert np.isfinite(best_f)
    assert len(best_x) == 2


def test_plateau_patience_reduces_evaluations_on_an_easy_objective():
    """An aggressive plateau setting (patience=3, rel_tol=0.9 -- "stop
    unless the last 3 generations improved best_f by at least 90%") must
    make a smooth, quickly-converging objective use MEASURABLY fewer
    function evaluations than the same run with plateau detection
    disabled, since cma's own tolfun/tolx (1e-8) essentially never fires
    this early on its own (see this module's module-level note above)."""
    bounds = [(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]

    def make_counted_objective(counter):
        def objective(x):
            counter[0] += 1
            return float(np.sum(x ** 2))
        return objective

    counter_with = [0]
    multistart_cma_es(
        objective=make_counted_objective(counter_with), bounds=bounds,
        n_restarts=1, maxiter=300, seed=1, config=None,
        plateau_patience=3, plateau_rel_tol=0.9)

    counter_without = [0]
    multistart_cma_es(
        objective=make_counted_objective(counter_without), bounds=bounds,
        n_restarts=1, maxiter=300, seed=1, config=None,
        plateau_patience=None)

    assert counter_with[0] < counter_without[0]


def test_seeded_x0_ignored_when_fewer_than_three_restarts():
    """seeded_x0 is restart index 2 specifically -- with n_restarts=2
    it never gets a slot at all, and must not silently substitute for
    restart 0/1's own deliberate starting points.

    Note: multistart_cma_es is NOT reproducible call-to-call even for
    the same seed (the underlying `cma` package has randomness beyond
    what its own `seed` option pins down -- confirmed empirically:
    two back-to-back calls with identical arguments returned different
    results), so this cannot compare two calls' results directly.
    Instead it checks a single-call property: seeded_x0's target is placed
    exactly at the objective's global minimum, so if seeded_x0 were
    (incorrectly) used as either restart's x0 despite n_restarts=2,
    that restart's very first (pre-generation) evaluation would already
    score 0 -- something a single CMA-ES generation from the box centre
    cannot stumble onto by chance.
    """
    bounds = [(0.0, 10.0), (0.0, 10.0)]
    target = np.array([9.9, 9.9])   # far from the box centre (5, 5)

    def objective(x):
        return float(np.sum((x - target) ** 2))

    _, best_f = multistart_cma_es(
        objective=objective, bounds=bounds, n_restarts=2,
        maxiter=1, seed=0, config=None, seeded_x0=target)

    # A LOOSE threshold here would be flaky: with sigma0 large relative
    # to a small bounds box, a single generation's population can
    # occasionally land within "somewhat close" of any fixed point by
    # sheer luck (observed empirically with a threshold of 1.0). Landing
    # within sqrt(1e-4)=0.01 units of an exact point purely by chance is
    # a different matter entirely -- effectively impossible for a
    # continuous random draw, so this stays a reliable, non-flaky check.
    assert best_f > 1e-4


# ---------------------------------------------------------------------------
# restart_workers (phase H-4): dispatching Pass-2's per-restart _cma_es
# calls across a process pool instead of one at a time. Pass 1 (x0
# construction) always runs sequentially regardless of this value, so
# every restart's x0 -- and therefore the final result -- must be
# bit-identical whichever value is used.
# ---------------------------------------------------------------------------

def test_restart_workers_omitted_defaults_to_sequential():
    bounds = [(0.0, 10.0), (0.0, 10.0)]
    target = np.array([3.0, 7.0])

    def objective(x):
        return float(np.sum((x - target) ** 2))

    best_x, best_f = multistart_cma_es(
        objective=objective, bounds=bounds, n_restarts=2,
        maxiter=15, seed=0, config=None)

    assert np.isfinite(best_f)
    assert len(best_x) == 2


def test_restart_workers_parallel_matches_sequential():
    bounds = [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]
    target = np.array([7.3, 2.1, 5.5])
    objective = functools.partial(_module_level_bowl, target=target)

    x_seq, f_seq = multistart_cma_es(
        objective=objective, bounds=bounds, n_restarts=5,
        maxiter=25, seed=42, config=None, restart_workers=1)
    x_par, f_par = multistart_cma_es(
        objective=objective, bounds=bounds, n_restarts=5,
        maxiter=25, seed=42, config=None, restart_workers=3)

    assert f_par == f_seq
    assert np.array_equal(x_par, x_seq)


def test_seed_zero_is_reproducible():
    """pycma treats seed 0 as 'seed from the clock'; mizuho must not."""
    from mizuho.optimize.cma import _cma_es

    def f(x):
        return float(np.sum((x - 0.3) ** 2))

    bounds = [(0.0, 1.0)] * 4
    runs = [_cma_es(f, np.full(4, 0.5), bounds, maxiter=15, seed=0) for _ in range(2)]
    np.testing.assert_array_equal(runs[0][0], runs[1][0])


def test_consecutive_seeds_do_not_share_restart_streams(monkeypatch):
    """Seed s restart 1 must not replay seed s+1 restart 0 (ADR-0012's
    five-seed spread needs independent draws)."""
    import mizuho.optimize.cma as cma_mod
    seen = {}

    def fake(objective, x0, bounds, sigma0, maxiter, tol, seed, verbose,
             periodic_mask, plateau_patience, plateau_rel_tol):
        seen.setdefault("seeds", []).append(seed)
        return np.asarray(x0), 1.0

    monkeypatch.setattr(cma_mod, "_cma_es", fake)
    bounds = [(0.0, 1.0)] * 2
    for s in (42, 43):
        multistart_cma_es(lambda x: 0.0, bounds, n_restarts=3, maxiter=1, seed=s)
    a, b = seen["seeds"][:3], seen["seeds"][3:]
    assert not set(a) & set(b)
    assert len(set(a)) == 3
