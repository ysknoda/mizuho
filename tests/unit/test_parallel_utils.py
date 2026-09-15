"""Unit tests for _parallel_utils.run_parallel_tasks -- the generic,
leaf-level process-pool dispatch helper introduced in phase H-1 (see
the plan file's "フェーズH" section) and shared by optimize.parallel
(candidate-level dispatch) and, in a later phase, geometry.local_shell
(precompute_letter_feasibility's map-reduce)."""
import time

import pytest

from mizuho._parallel_utils import run_parallel_tasks


def _square(x):
    return x * x


def _sleep_then_return_index(args):
    idx, delay = args
    time.sleep(delay)
    return idx


def _raise_if_three(x):
    if x == 3:
        raise ValueError("boom")
    return x


def test_sequential_fallback_with_max_workers_one():
    results = run_parallel_tasks(_square, [1, 2, 3, 4], max_workers=1)
    assert results == [1, 4, 9, 16]


def test_sequential_fallback_with_single_task_even_if_max_workers_high():
    results = run_parallel_tasks(_square, [5], max_workers=4)
    assert results == [25]


def test_parallel_path_preserves_task_order_not_completion_order():
    # Task 0 sleeps longest -> completes LAST, but must still land at
    # index 0 in the returned list (task order, not completion order).
    tasks = [(0, 0.3), (1, 0.02), (2, 0.02)]
    results = run_parallel_tasks(_sleep_then_return_index, tasks, max_workers=3)
    assert results == [0, 1, 2]


def test_on_result_called_once_per_task_sequential():
    seen = []
    run_parallel_tasks(_square, [1, 2, 3], max_workers=1,
                        on_result=lambda i, r: seen.append((i, r)))
    assert sorted(seen) == [(0, 1), (1, 4), (2, 9)]


def test_on_result_called_once_per_task_parallel():
    seen = []
    run_parallel_tasks(_square, [1, 2, 3, 4], max_workers=4,
                        on_result=lambda i, r: seen.append((i, r)))
    assert sorted(seen) == [(0, 1), (1, 4), (2, 9), (3, 16)]


def test_worker_exception_propagates_sequential():
    with pytest.raises(ValueError, match="boom"):
        run_parallel_tasks(_raise_if_three, [1, 2, 3, 4], max_workers=1)


def test_worker_exception_propagates_parallel():
    with pytest.raises(ValueError, match="boom"):
        run_parallel_tasks(_raise_if_three, [1, 2, 3, 4], max_workers=4)
