"""Parallel evaluation of independent Wyckoff-assignment candidates
using a process pool.

Each candidate's optimize_one_assignment() call is fully independent
(its own CMA-ES restarts, its own objective evaluations), so candidates
parallelise with no shared state. This was blocked in the original
design because assignments_to_sites() embedded a raw per-entry
generator lambda in the site config, which isn't picklable across
process boundaries -- eliminated by the Wyckoff-database rework (no
per-entry generator callables exist any more; coord_template is plain
data), so no worker-side "re-resolve the generator" workaround is
needed here.
"""

import os
from typing import Callable, List, Optional, Tuple

import numpy as np

from .._parallel_utils import run_parallel_tasks
from ..structure import SiteAssignment
from .driver import optimize_one_assignment


def _run_one(args):
    assignments, config, opt_cfg, label, shell_seed_info = args
    return optimize_one_assignment(assignments, config, opt_cfg,
                                    verbose=False, label=label,
                                    shell_seed_info=shell_seed_info)


def optimize_assignments_parallel(
        assignments_list: List[List[SiteAssignment]],
        config: dict,
        opt_cfg: dict,
        max_workers: Optional[int] = None,
        labels: Optional[List[str]] = None,
        shell_seeds: Optional[List] = None,
        on_result: Optional[Callable[[int, str, float], None]] = None,
) -> List[Tuple[Optional[np.ndarray], float, Optional[dict]]]:
    """
    Evaluate every candidate in assignments_list independently, in
    parallel. Returns [(best_p, best_e, structure), ...] in the SAME
    order as assignments_list (not completion order).

    Falls back to sequential evaluation (no subprocess overhead) when
    max_workers resolves to 1 or there is only one candidate -- verbose
    per-candidate progress printing only makes sense sequentially, so
    callers that need it should pass max_workers=1 explicitly.

    shell_seeds, if given, is a list parallel to assignments_list (see
    pipeline.search_and_rank's own shell_seeds return value): each
    entry is forwarded to optimize_one_assignment's shell_seed_info so
    Phase 2 reuses Phase 1's already-solved local shells instead of
    re-solving them from scratch inside the worker. None (the
    default) means every candidate solves fresh, same as passing a
    list of Nones explicitly.

    on_result(index, label, best_e), if given, is called once per
    candidate as soon as it completes (completion order under a
    process pool, task order in the sequential fallback) -- useful for
    progress reporting, since a worker's own per-candidate verbose
    output can't interleave live across process boundaries.
    """
    n = len(assignments_list)
    if labels is None:
        labels = [f"cand{i}" for i in range(n)]
    if shell_seeds is None:
        shell_seeds = [None] * n

    # The zip() below truncates to the SHORTEST input, so a mismatched
    # length here would silently drop candidates from the run and return
    # a short result list rather than failing -- a wrong answer that
    # looks like a complete one. pipeline.search_and_rank does keep these
    # in lockstep (it builds shell_seeds from valid_asgns and truncates
    # both together), so this is a guard against a future caller, not a
    # known bug.
    if len(labels) != n or len(shell_seeds) != n:
        raise ValueError(
            f"optimize_assignments_parallel: labels ({len(labels)}) and "
            f"shell_seeds ({len(shell_seeds)}) must each be parallel to "
            f"assignments_list ({n}).")

    # Mirrors run_parallel_tasks's own sequential-fallback threshold:
    # the restart_workers override below only makes sense when a
    # candidate-level process pool will ACTUALLY run concurrently with
    # this worker's own optimize_one_assignment call -- see
    # optimize.driver's restart_workers docstring for the nested-pool
    # hazard (candidate_workers x restart_workers processes fighting
    # over cores) this guards against. When THIS call itself falls
    # back to a plain sequential loop (max_workers<=1, or a single
    # candidate), there is no pool to nest inside, so restart_workers
    # (if the caller set one in opt_cfg) is left untouched.
    effective_workers = max_workers if max_workers is not None else (os.cpu_count() or 1)
    will_parallelize = effective_workers > 1 and n > 1
    task_opt_cfg = dict(opt_cfg, restart_workers=1) if will_parallelize else opt_cfg

    tasks = [(a, config, task_opt_cfg, lb, ss)
             for a, lb, ss in zip(assignments_list, labels, shell_seeds)]

    def _wrapped_on_result(i, result):
        _, best_e, _ = result
        on_result(i, labels[i], best_e)

    return run_parallel_tasks(
        _run_one, tasks, max_workers=max_workers,
        on_result=_wrapped_on_result if on_result is not None else None)
