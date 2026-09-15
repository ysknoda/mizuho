"""Best-first branch-and-bound search over Wyckoff-site assignments,
replacing the exhaustive DFS as the default strategy.

Two changes from the exhaustive DFS (see its docstring for the shared
pruning/acceptance rules, which are unchanged):

1. CANONICAL ORDERING: children of a node are restricted to (element,
   Wyckoff entry) choices that are >= the last choice made on that
   node's own path, under a fixed total order. This generates every
   assignment MULTISET exactly once (as a combination) instead of
   generating all of its orderings and deduplicating after the fact --
   the transpositions the original DFS's canonical-label dedup was
   cleaning up after simply never get created here.
2. BEST-FIRST EXPANSION: a min-heap keyed on scoring.surrogate_score
   replaces the LIFO stack, so implausible partial assignments (bad
   charge balance, unrealistic packing) sink to the back of the queue
   instead of being explored in whatever order they happened to be
   pushed. Combined with budget.max_candidates actually being enforced
   (see base.SearchBudget -- the original max_candidates/
   fast_tree_budget.max_candidates was read from config but never
   consulted by the search loop), this lets a small budget find good
   assignments first instead of truncating an arbitrary DFS order.
"""

import heapq
import itertools
from collections import defaultdict
from typing import Dict, List, Tuple

from ..structure import SiteAssignment
from .base import (AssignmentProblem, SearchBudget, SearchResult,
                    SearchStrategy, TreeNode, _canonical_label,
                    _fixed_representative_counts, _shape_group_sizes,
                    _shape_key, canonical_representatives, _is_complete,
                    _ratio_consistent)
from .scoring import surrogate_score


def _choice_key(problem: AssignmentProblem, elem: str, letter: str) -> Tuple[int, str]:
    return (problem.elements.index(elem), letter)


class BestFirstBranchAndBoundStrategy(SearchStrategy):

    def search(self, problem: AssignmentProblem, budget: SearchBudget) -> SearchResult:
        valid_assignments: List[List[SiteAssignment]] = []
        seen_labels: set = set()
        nodes_expanded = 0
        budget_exhausted = False

        # All (element, entry) choices, canonically ordered. Branches
        # over canonical_representatives, not every wu_entries letter
        # -- see that function's docstring: letters sharing a shape
        # (multiplicity + free-parameter structure) are indistinguishable
        # to this search's own scoring, so trying each one as a separate
        # branch is pure combinatorial waste. Which specific letter to
        # use is decided later, by geometry.rank_by_feasibility via
        # base.expand_letter_choices, which actually can tell them apart.
        representatives = canonical_representatives(problem.wu_entries)
        shape_group_sizes = _shape_group_sizes(problem.wu_entries)
        all_choices = [
            (elem, entry)
            for elem in problem.elements
            for entry in representatives
        ]

        counter = itertools.count()  # heap tie-breaker (nodes aren't orderable)
        root = TreeNode([], 0.0, 0)
        heap = [(surrogate_score(root, problem), next(counter), root, (-1, ""))]

        while heap:
            if (budget.max_candidates is not None
                    and len(valid_assignments) >= budget.max_candidates):
                budget_exhausted = True
                break
            if (budget.max_nodes_expanded is not None
                    and nodes_expanded >= budget.max_nodes_expanded):
                budget_exhausted = True
                break
            if not heap:
                break

            _, _, node, last_key = heapq.heappop(heap)
            nodes_expanded += 1
            depth = len(node.assignments)

            if depth >= problem.max_sites:
                continue
            if node.total_atoms >= problem.max_atoms_per_fu:
                continue

            assigned_counts: Dict[str, int] = defaultdict(int)
            for a in node.assignments:
                assigned_counts[a.element] += a.entry.multiplicity

            fixed_rep_counts: Dict[str, int] = {}
            if problem.zero_charge_mode and not problem.allow_dup_wyckoff:
                fixed_rep_counts = _fixed_representative_counts(node.assignments)

            for elem, wentry in all_choices:
                key = _choice_key(problem, elem, wentry.letter)
                if key < last_key:
                    continue  # combinations, not permutations

                if (assigned_counts[elem]
                        >= problem.composition[elem] * problem.max_formula_units):
                    continue

                if (problem.zero_charge_mode and not problem.allow_dup_wyckoff
                        and not wentry.free_params
                        and fixed_rep_counts.get(wentry.letter, 0)
                            >= shape_group_sizes[_shape_key(wentry)]):
                    continue

                mult = wentry.multiplicity
                q_elem = problem.ionic_q.get(elem, 0)
                new_atoms = node.total_atoms + mult
                new_charge = node.total_charge + q_elem * mult

                if new_atoms > problem.max_atoms_per_fu:
                    continue

                new_counts = dict(assigned_counts)
                new_counts[elem] = new_counts.get(elem, 0) + mult

                too_many = any(
                    new_counts.get(e2, 0) >
                    problem.composition.get(e2, 1) * problem.max_formula_units + 0.01
                    for e2 in new_counts)
                if too_many:
                    continue

                if not _ratio_consistent(new_counts, problem.composition,
                                          problem.elements, elem):
                    continue

                new_assignments = node.assignments + [SiteAssignment(elem, wentry)]
                new_node = TreeNode(new_assignments, new_charge, new_atoms)

                complete = _is_complete(new_counts, problem.composition)
                if complete:
                    if problem.zero_charge_mode:
                        accepted = True
                    else:
                        accepted = problem.tol_min <= new_charge <= problem.tol_max

                    if accepted:
                        canon = _canonical_label(new_assignments)
                        if canon not in seen_labels:
                            seen_labels.add(canon)
                            valid_assignments.append(new_assignments)
                            if (budget.max_candidates is not None
                                    and len(valid_assignments) >= budget.max_candidates):
                                budget_exhausted = True
                                break
                    continue

                if not problem.zero_charge_mode:
                    remaining = problem.max_atoms_per_fu - new_atoms
                    max_corr = problem.max_q_per_atom * remaining
                    if new_charge > problem.tol_max + max_corr:
                        continue
                    if new_charge < problem.tol_min - max_corr:
                        continue

                score = surrogate_score(new_node, problem)
                heapq.heappush(heap, (score, next(counter), new_node, key))

        return SearchResult(accepted=valid_assignments, nodes_expanded=nodes_expanded,
                            budget_exhausted=budget_exhausted)
