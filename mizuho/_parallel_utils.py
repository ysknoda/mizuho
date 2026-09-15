"""Generic process-pool task dispatch, shared by optimize/ and
geometry/ without either importing the other -- geometry/ must not
import optimize/ (see geometry/local_shell.py's own module docstring
for why), so this helper lives at the package root as a leaf module
that imports nothing else from mizuho itself.
"""
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, List, Optional


def run_parallel_tasks(fn: Callable, tasks: List,
                        max_workers: Optional[int] = None,
                        on_result: Optional[Callable[[int, object], None]] = None) -> List:
    """
    Apply fn to every item in tasks, returning results in the SAME
    order as tasks (not completion order). fn must be a module-level
    (picklable) callable taking one positional argument.

    Falls back to a plain sequential loop -- no subprocess, no
    pickling -- when max_workers resolves to 1 or there is only one
    task, so callers needn't special-case the single-worker/single-
    task case themselves.

    on_result(index, result), if given, is called once per task as
    soon as it completes: in task order for the sequential fallback,
    in COMPLETION order for the parallel path (useful for streaming
    progress under a process pool, where per-task printing inside fn
    itself can't interleave live across process boundaries). Any
    exception raised inside fn propagates to the caller either way
    (immediately in the sequential path, via Future.result() in the
    parallel path) -- a failing task is not silently swallowed.
    """
    n = len(tasks)
    if max_workers is None:
        max_workers = os.cpu_count() or 1

    if max_workers <= 1 or n <= 1:
        results = []
        for i, task in enumerate(tasks):
            result = fn(task)
            results.append(result)
            if on_result is not None:
                on_result(i, result)
        return results

    results = [None] * n
    with ProcessPoolExecutor(max_workers=min(max_workers, n)) as pool:
        future_to_index = {pool.submit(fn, task): i for i, task in enumerate(tasks)}
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            result = future.result()
            results[i] = result
            if on_result is not None:
                on_result(i, result)
    return results
