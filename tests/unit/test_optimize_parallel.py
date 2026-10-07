"""optimize_assignments_parallel must give the same results as calling
optimize_one_assignment sequentially, and must not crash on pickling
(the historical blocker -- a raw per-entry generator lambda embedded
in the site config -- no longer exists after the Wyckoff-database
rework, see optimize/parallel.py's module docstring)."""

import pickle
import time

import pytest

from mizuho.data import WYCKOFF_DB
from mizuho.optimize.driver import optimize_one_assignment
from mizuho.optimize.parallel import optimize_assignments_parallel
from mizuho.pipeline import build_sg_config, search_and_rank
from mizuho.structure import SiteAssignment


def test_site_assignment_is_picklable():
    entry = WYCKOFF_DB[225][0]
    sa = SiteAssignment("Na", entry)
    assert pickle.loads(pickle.dumps(sa)) == sa


def test_parallel_matches_sequential_results(nacl_config):
    na_entry = next(e for e in WYCKOFF_DB[225] if e.letter == "a")
    cl_entry = next(e for e in WYCKOFF_DB[225] if e.letter == "b")
    assignments_list = [
        [SiteAssignment("Na", na_entry), SiteAssignment("Cl", cl_entry)],
        [SiteAssignment("Na", cl_entry), SiteAssignment("Cl", na_entry)],
    ]
    opt_cfg = dict(nacl_config["optimization"])

    sequential = [optimize_one_assignment(a, nacl_config, opt_cfg, verbose=False)
                  for a in assignments_list]
    parallel = optimize_assignments_parallel(
        assignments_list, nacl_config, opt_cfg, max_workers=2)

    assert len(parallel) == len(sequential)
    for (sp, se, sstruct), (pp, pe, pstruct) in zip(sequential, parallel):
        assert pe == se  # same seed -> deterministic -> identical error


def test_parallel_with_single_worker_matches_sequential_path(nacl_config):
    na_entry = next(e for e in WYCKOFF_DB[225] if e.letter == "a")
    cl_entry = next(e for e in WYCKOFF_DB[225] if e.letter == "b")
    assignments_list = [[SiteAssignment("Na", na_entry), SiteAssignment("Cl", cl_entry)]]
    opt_cfg = dict(nacl_config["optimization"])

    result = optimize_assignments_parallel(
        assignments_list, nacl_config, opt_cfg, max_workers=1)
    assert len(result) == 1
    assert result[0][1] < 1.0


def test_parallel_with_shell_seeds_and_on_result_real_run(nacl_config):
    """Phase G-8/H-1: shell_seeds and on_result must survive an ACTUAL
    process-pool round trip (pickling), not just a mocked one."""
    na_entry = next(e for e in WYCKOFF_DB[225] if e.letter == "a")
    cl_entry = next(e for e in WYCKOFF_DB[225] if e.letter == "b")
    assignments_list = [
        [SiteAssignment("Na", na_entry), SiteAssignment("Cl", cl_entry)],
        [SiteAssignment("Na", cl_entry), SiteAssignment("Cl", na_entry)],
    ]
    opt_cfg = dict(nacl_config["optimization"])
    seen = []

    results = optimize_assignments_parallel(
        assignments_list, nacl_config, opt_cfg, max_workers=2,
        shell_seeds=[None, None],
        on_result=lambda i, label, best_e: seen.append(i))

    assert len(results) == 2
    assert sorted(seen) == [0, 1]


def _fake_optimize_one_assignment(assignments, config, opt_cfg,
                                   verbose=False, label="", shell_seed_info=None):
    return None, 42.0, None


def test_shell_seeds_forwarded_to_optimize_one_assignment(monkeypatch, nacl_config):
    captured = []

    def fake_ooa(assignments, config, opt_cfg, verbose=False, label="", shell_seed_info=None):
        captured.append(shell_seed_info)
        return None, 1.0, None

    monkeypatch.setattr("mizuho.optimize.parallel.optimize_one_assignment", fake_ooa)
    na_entry = next(e for e in WYCKOFF_DB[225] if e.letter == "a")
    cl_entry = next(e for e in WYCKOFF_DB[225] if e.letter == "b")
    assignments_list = [
        [SiteAssignment("Na", na_entry), SiteAssignment("Cl", cl_entry)],
        [SiteAssignment("Na", cl_entry), SiteAssignment("Cl", na_entry)],
    ]
    opt_cfg = dict(nacl_config["optimization"])
    seeds = [{"dummy": 1}, None]

    optimize_assignments_parallel(assignments_list, nacl_config, opt_cfg,
                                   max_workers=1, shell_seeds=seeds)

    assert captured == seeds


def test_shell_seeds_default_to_none(monkeypatch, nacl_config):
    captured = []

    def fake_ooa(assignments, config, opt_cfg, verbose=False, label="", shell_seed_info=None):
        captured.append(shell_seed_info)
        return None, 1.0, None

    monkeypatch.setattr("mizuho.optimize.parallel.optimize_one_assignment", fake_ooa)
    na_entry = next(e for e in WYCKOFF_DB[225] if e.letter == "a")
    cl_entry = next(e for e in WYCKOFF_DB[225] if e.letter == "b")
    assignments_list = [[SiteAssignment("Na", na_entry), SiteAssignment("Cl", cl_entry)]]
    opt_cfg = dict(nacl_config["optimization"])

    optimize_assignments_parallel(assignments_list, nacl_config, opt_cfg, max_workers=1)

    assert captured == [None]


def test_on_result_called_with_label_and_error(monkeypatch, nacl_config):
    monkeypatch.setattr("mizuho.optimize.parallel.optimize_one_assignment",
                         _fake_optimize_one_assignment)
    na_entry = next(e for e in WYCKOFF_DB[225] if e.letter == "a")
    cl_entry = next(e for e in WYCKOFF_DB[225] if e.letter == "b")
    assignments_list = [[SiteAssignment("Na", na_entry), SiteAssignment("Cl", cl_entry)]]
    opt_cfg = dict(nacl_config["optimization"])
    captured = []

    optimize_assignments_parallel(
        assignments_list, nacl_config, opt_cfg, max_workers=1,
        labels=["mylabel"], on_result=lambda i, lb, e: captured.append((i, lb, e)))

    assert captured == [(0, "mylabel", 42.0)]


def test_restart_workers_forced_to_one_when_actually_parallelizing(monkeypatch, nacl_config):
    """Phase H-4's nested-pool safeguard, wired in advance here (H-1):
    when a candidate-level process pool will actually run, the
    restart_workers a caller set in opt_cfg must be overridden to 1
    for every dispatched task, so a future H-4 restart-level pool can
    never nest inside an active candidate-level one. Intercepted at
    run_parallel_tasks (same-process boundary) rather than through a
    real process pool, since the override must be visible in the task
    tuples BEFORE they cross into worker processes."""
    captured = {}

    def fake_run_parallel_tasks(fn, tasks, max_workers=None, on_result=None):
        captured["tasks"] = tasks
        return [(None, 1.0, None) for _ in tasks]

    monkeypatch.setattr("mizuho.optimize.parallel.run_parallel_tasks",
                         fake_run_parallel_tasks)
    na_entry = next(e for e in WYCKOFF_DB[225] if e.letter == "a")
    cl_entry = next(e for e in WYCKOFF_DB[225] if e.letter == "b")
    assignments_list = [
        [SiteAssignment("Na", na_entry), SiteAssignment("Cl", cl_entry)],
        [SiteAssignment("Na", cl_entry), SiteAssignment("Cl", na_entry)],
    ]
    opt_cfg = dict(nacl_config["optimization"], restart_workers=4)

    optimize_assignments_parallel(assignments_list, nacl_config, opt_cfg, max_workers=2)

    for _, _, task_opt_cfg, _, _, _ in captured["tasks"]:
        assert task_opt_cfg["restart_workers"] == 1


@pytest.mark.slow
def test_parallel_matches_sequential_on_real_multi_candidate_fixture(rutile_tio2_config):
    """End-to-end sanity check (not mocked) on a real fixture with
    multiple genuine candidates (rutile_tio2 -> 16 assignments):
    parallel dispatch must reach the SAME final ranking as the
    sequential loop it replaces in cli.py/pipeline.py, using this
    fixture's own optimization: budget as-is (no artificial shrinking)."""
    sg_config = build_sg_config(rutile_tio2_config, rutile_tio2_config["spacegroup"])
    valid_asgns, n_found, shell_seeds = search_and_rank(sg_config, max_cands=50)
    assert len(valid_asgns) > 1

    opt_cfg = dict(sg_config["optimization"])

    t0 = time.time()
    sequential = [
        optimize_one_assignment(a, sg_config, opt_cfg, verbose=False,
                                 shell_seed_info=shell_seeds[k])
        for k, a in enumerate(valid_asgns)]
    sequential_time = time.time() - t0

    t0 = time.time()
    parallel = optimize_assignments_parallel(
        valid_asgns, sg_config, opt_cfg, max_workers=4, shell_seeds=shell_seeds)
    parallel_time = time.time() - t0

    print(f"\n  sequential: {sequential_time:.2f}s   "
          f"parallel(4 workers): {parallel_time:.2f}s   "
          f"n_candidates={len(valid_asgns)}")

    assert len(parallel) == len(sequential)
    seq_errors = sorted(e for _, e, _ in sequential)
    par_errors = sorted(e for _, e, _ in parallel)
    assert seq_errors == pytest.approx(par_errors)


def test_restart_workers_left_untouched_when_not_parallelizing(monkeypatch, nacl_config):
    """The mirror case: when THIS call falls back to the sequential
    path (max_workers<=1 or a single candidate), there is no pool to
    nest inside, so a caller-set restart_workers passes through
    unmodified -- see optimize.parallel's own module-level rationale."""
    captured = {}

    def fake_run_parallel_tasks(fn, tasks, max_workers=None, on_result=None):
        captured["tasks"] = tasks
        return [(None, 1.0, None) for _ in tasks]

    monkeypatch.setattr("mizuho.optimize.parallel.run_parallel_tasks",
                         fake_run_parallel_tasks)
    na_entry = next(e for e in WYCKOFF_DB[225] if e.letter == "a")
    cl_entry = next(e for e in WYCKOFF_DB[225] if e.letter == "b")
    assignments_list = [[SiteAssignment("Na", na_entry), SiteAssignment("Cl", cl_entry)]]
    opt_cfg = dict(nacl_config["optimization"], restart_workers=4)

    optimize_assignments_parallel(assignments_list, nacl_config, opt_cfg, max_workers=1)

    _, _, task_opt_cfg, _, _, _ = captured["tasks"][0]
    assert task_opt_cfg["restart_workers"] == 4


def test_mismatched_shell_seeds_length_raises_instead_of_dropping_candidates():
    """The internal zip() truncates to the shortest input, so a
    shell_seeds list shorter than assignments_list would silently drop
    candidates and return a short result list -- a wrong answer that
    looks complete. See .code-check/history.md.
    """
    import pytest
    from mizuho.optimize.parallel import optimize_assignments_parallel
    from mizuho.structure import resolve_fixed_assignment

    asgns = [resolve_fixed_assignment(225, [{"element": "Na", "letter": "a"},
                                             {"element": "Cl", "letter": "b"}])
             for _ in range(3)]

    with pytest.raises(ValueError, match="parallel to assignments_list"):
        optimize_assignments_parallel(asgns, {}, {}, shell_seeds=[None])

    with pytest.raises(ValueError, match="parallel to assignments_list"):
        optimize_assignments_parallel(asgns, {}, {}, labels=["only-one"])
