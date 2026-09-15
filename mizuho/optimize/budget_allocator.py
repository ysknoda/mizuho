"""Successive-halving budget allocator: a bandit-style way to spend a
bounded number of expensive CMA-ES confirmation evaluations across many
candidates (Wyckoff assignments in a fixed space group, or space groups
in the hierarchical auto-search) without giving every candidate the
same full budget up front.

All candidates start with a small budget; only the better half (by
achieved error) survive to the next round, where the per-candidate
budget doubles. This replaces two previously hard-coded, unrelated
shortcuts that did the same job badly: Phase 1's "evaluate at most 3
assignments per SG" and Phase 0 Stage-C's dead TPE-trial-count knobs.
"""
from dataclasses import dataclass
from typing import Callable, List, Tuple, TypeVar

C = TypeVar("C")


@dataclass
class HalvingSchedule:
    initial_restarts:  int = 1
    initial_maxiter:   int = 30
    n_rounds:          int = 3
    keep_fraction:     float = 0.5
    budget_multiplier: float = 2.0


def successive_halving(
        candidates: List[C],
        evaluate: Callable[[C, int, int], float],
        schedule: HalvingSchedule = HalvingSchedule(),
) -> List[Tuple[C, float]]:
    """
    evaluate(candidate, n_restarts, cma_maxiter) -> error (lower is better)

    Returns [(candidate, final_error), ...] sorted ascending by error,
    covering every candidate that survived to the last round it was
    evaluated in (i.e. not just the round-1 winners -- the whole
    surviving set from the final round actually run).
    """
    if not candidates:
        return []

    survivors = list(candidates)
    n_restarts = schedule.initial_restarts
    maxiter = schedule.initial_maxiter
    last_scores: dict = {}

    for round_i in range(schedule.n_rounds):
        scores = [(c, evaluate(c, n_restarts, maxiter)) for c in survivors]
        scores.sort(key=lambda cs: cs[1])
        last_scores = {id(c): (c, s) for c, s in scores}

        if len(survivors) <= 1 or round_i == schedule.n_rounds - 1:
            break

        keep_n = max(1, int(round(len(scores) * schedule.keep_fraction)))
        survivors = [c for c, _ in scores[:keep_n]]
        n_restarts = max(1, int(round(n_restarts * schedule.budget_multiplier)))
        maxiter = max(10, int(round(maxiter * schedule.budget_multiplier)))

    ranked = sorted(last_scores.values(), key=lambda cs: cs[1])
    return ranked
