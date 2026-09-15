"""Phase H-4's nested-process-pool safeguard: when candidate-level
parallelism (phase H-1, parallel.candidate_workers / optimize.parallel.
optimize_assignments_parallel) is active, restart-level parallelism
(phase H-4, optimization.restart_workers) inside each candidate worker
must be forced back to 1, so the two can never multiply out into
candidate_workers x restart_workers processes fighting over cores.

The forced-override MECHANISM itself (optimize.parallel.
optimize_assignments_parallel setting opt_cfg["restart_workers"]=1 in
the tasks it dispatches) is already covered, via mocking, by
test_optimize_parallel.py's test_restart_workers_forced_to_one_when_
actually_parallelizing / test_restart_workers_left_untouched_when_not_
parallelizing (added in phase H-1, in anticipation of H-4). This file
covers what those mocked tests deliberately don't: (1) whether nesting
a REAL ProcessPoolExecutor inside another is even technically safe on
this platform (an open question noted in the plan file when H-1 was
designed), and (2) a real, unmocked end-to-end run exercising H-1 and
H-4 together through the actual optimize_one_assignment/multistart_*
code path, not a stand-in.
"""
import functools

import numpy as np
import pytest
from concurrent.futures import ProcessPoolExecutor

from mizuho.data import WYCKOFF_DB
from mizuho.optimize.parallel import optimize_assignments_parallel
from mizuho.structure import SiteAssignment


def _inner_square(x):
    return x * x


def _outer_task(x):
    """Runs INSIDE a ProcessPoolExecutor worker and itself spawns a
    second, nested ProcessPoolExecutor -- exactly the shape H-1
    (candidate-level) + H-4 (restart-level) parallelism would produce
    if the restart_workers safeguard were ever bypassed."""
    with ProcessPoolExecutor(max_workers=2) as inner_pool:
        return sum(inner_pool.map(_inner_square, [x, x + 1]))


def test_nested_process_pools_do_not_crash():
    """If this ever starts failing (e.g. a future Python version
    forbidding a pool worker from spawning its own child pool), the
    restart_workers safeguard in optimize.parallel becomes load-bearing
    for CORRECTNESS, not just performance -- currently it is purely a
    performance safeguard (avoiding core oversubscription), since this
    test confirms nesting itself does not crash."""
    with ProcessPoolExecutor(max_workers=2) as outer_pool:
        results = list(outer_pool.map(_outer_task, [1, 2, 3]))
    assert results == [1 + 4, 4 + 9, 9 + 16]


def test_candidate_and_restart_parallelism_together_end_to_end(nacl_config):
    """Real (unmocked) run through optimize.parallel.
    optimize_assignments_parallel with BOTH candidate_workers-equivalent
    (max_workers=2) and a caller-set restart_workers>1 in opt_cfg. The
    safeguard must silently downgrade restart_workers to 1 inside each
    candidate worker -- if it didn't, and nesting were somehow unsafe,
    this call would hang or crash instead of returning normally."""
    na_entry = next(e for e in WYCKOFF_DB[225] if e.letter == "a")
    cl_entry = next(e for e in WYCKOFF_DB[225] if e.letter == "b")
    assignments_list = [
        [SiteAssignment("Na", na_entry), SiteAssignment("Cl", cl_entry)],
        [SiteAssignment("Na", cl_entry), SiteAssignment("Cl", na_entry)],
    ]
    opt_cfg = dict(nacl_config["optimization"], restart_workers=4)

    results = optimize_assignments_parallel(
        assignments_list, nacl_config, opt_cfg, max_workers=2)

    assert len(results) == 2
    for _, best_e, structure in results:
        assert np.isfinite(best_e)
        assert structure is not None
