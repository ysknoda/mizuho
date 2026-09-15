"""Exhaustive depth-first search over Wyckoff-site assignments for a
fixed space group and target composition.

Kept as ExhaustiveDFSStrategy after the best-first rewrite (best_first.py)
purely as a correctness oracle: it visits every node in the search tree
(no priority ordering, no early termination beyond the budget caps), so
on an unlimited budget it is guaranteed to find the same set of valid
assignments as the best-first strategy -- see
tests/unit/test_search_strategies.py.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from ..structure import SiteAssignment
from .base import (AssignmentProblem, SearchBudget, SearchResult,
                    SearchStrategy, TreeNode, _canonical_label,
                    _fixed_representative_counts, _shape_group_sizes,
                    _shape_key, canonical_representatives,
                    expand_letter_choices, _is_complete, _ratio_consistent)


class ExhaustiveDFSStrategy(SearchStrategy):
    """
    Two operating modes are selected automatically (see
    AssignmentProblem.zero_charge_mode):

    IONIC MODE  (default)
        Acceptance criterion: total charge in [tol_min, tol_max].
        Pruning: branches whose charge cannot be recovered within the
        remaining atom budget are discarded early.

    COVALENT / ZERO-CHARGE MODE
        Because the charge filter is vacuously satisfied for all-zero
        charges, the tree would otherwise enumerate every possible
        Wyckoff combination without discrimination, causing a combinatorial
        explosion. In covalent mode the search instead enforces:

          (a) STRICT RATIO CONSISTENCY at every node depth.
          (b) CANONICAL DEDUPLICATION (sorted, order-independent label).
          (c) SAME-LETTER EXCLUSION (optional, default on): reserving a
              ZERO-free-parameter shape group's representative more
              times than that group has distinct concrete letters is
              pruned, since expand_letter_choices could then never hand
              every reservation its own coordinate (a zero-free-param
              orbit is one exact, immovable point set -- see
              base._fixed_representative_counts). Counted across ALL
              elements combined, not per element: two different
              elements reserving the same shape group is just as
              degenerate as one element doing it twice, once the group
              is exhausted. Letters WITH free parameters are never
              pruned here -- each reservation can take its own
              coordinate value later, so reuse (by the same or a
              different element, any number of times) is always
              meaningful, e.g. real MOF structures repeatedly placing
              the same metal on a high-multiplicity, free-parameter
              letter such as 16f or 48g across several distinct sites.
              Set allow_duplicate_wyckoff=true to disable this pruning
              entirely.
    """

    def search(self, problem: AssignmentProblem, budget: SearchBudget) -> SearchResult:
        valid_assignments: List[List[SiteAssignment]] = []
        seen_labels: set = set()
        nodes_expanded = 0
        budget_exhausted = False

        # Branch over canonical_representatives, not every wu_entries
        # letter -- see that function's docstring in base.py: letters
        # sharing a shape are indistinguishable to this search, so this
        # avoids pure combinatorial waste. Computed once, not per node.
        representatives = canonical_representatives(problem.wu_entries)
        shape_group_sizes = _shape_group_sizes(problem.wu_entries)

        stack: List[TreeNode] = [TreeNode([], 0.0, 0)]

        while stack:
            if (budget.max_candidates is not None
                    and len(valid_assignments) >= budget.max_candidates):
                budget_exhausted = True
                break
            if (budget.max_nodes_expanded is not None
                    and nodes_expanded >= budget.max_nodes_expanded):
                budget_exhausted = True
                break

            node = stack.pop()
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

            for elem in problem.elements:
                if budget_exhausted:
                    break
                if (assigned_counts[elem]
                        >= problem.composition[elem] * problem.max_formula_units):
                    continue

                q_elem = problem.ionic_q.get(elem, 0)

                for wentry in representatives:
                    mult = wentry.multiplicity

                    if (problem.zero_charge_mode and not problem.allow_dup_wyckoff
                            and not wentry.free_params
                            and fixed_rep_counts.get(wentry.letter, 0)
                                >= shape_group_sizes[_shape_key(wentry)]):
                        continue

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

                    stack.append(new_node)

        return SearchResult(accepted=valid_assignments, nodes_expanded=nodes_expanded,
                            budget_exhausted=budget_exhausted)


def wyckoff_tree_search(
        config: dict,
        letter_feasibility: Optional[Dict[Tuple[str, str], float]] = None,
) -> List[List[SiteAssignment]]:
    """
    Enumerate valid Wyckoff site assignments for the given composition
    and space group. Dispatches to BestFirstBranchAndBoundStrategy by
    default (config['tree_search']['strategy'] = 'exhaustive' selects
    the DFS oracle instead); both return the same set of accepted
    leaves on an unlimited budget.

    `letter_feasibility`, if given, is an (element, letter) ->
    geometric-feasibility-score table from
    geometry.precompute_letter_feasibility, passed straight through to
    AssignmentProblem so surrogate_score can bias best-first's
    exploration order towards geometrically-plausible letters (see that
    function's docstring for what it approximates and why). Omitted by
    default -- callers that don't pass it (which is every existing call
    site except pipeline.search_and_rank) get identical behaviour to
    before this parameter existed.

    The two strategies branch over canonical_representatives (one
    letter per shape group -- see base.py) to keep the search itself
    cheap, so a raw SearchResult.accepted entry may use a
    representative letter standing in for its whole shape group on any
    given site. That is fine as long as every OTHER site in the same
    assignment is free to independently pick its own letter from ITS
    shape group -- but a raw representative-only assignment always
    reuses the SAME single letter for every site sharing a shape group,
    which is wrong whenever two sites in one assignment genuinely need
    to occupy two DIFFERENT orbits of that shape (e.g. NaCl/Fm-3m: 'a'
    and 'b' are both mult-4 fixed points sharing a shape key, so the
    only geometrically sensible assignment is Na@a + Cl@b -- but the
    raw search result only ever contains Na@a + Cl@a, since 'b' is
    never a branch choice at all once 'a' is picked as the group's
    representative). expand_letter_choices restores every concrete
    letter combination per assignment; results are re-deduplicated by
    canonical label since a shape group of size 1 (the common case)
    expands to exactly the same assignment it started from.
    """
    from .best_first import BestFirstBranchAndBoundStrategy

    ts_cfg = config.get("tree_search", {})
    strategy_name = ts_cfg.get("strategy", "best_first")
    strategy: SearchStrategy = (
        ExhaustiveDFSStrategy() if strategy_name == "exhaustive"
        else BestFirstBranchAndBoundStrategy())

    problem = AssignmentProblem.from_config(config)
    problem.letter_feasibility = letter_feasibility
    budget = SearchBudget.from_config(config)
    result = strategy.search(problem, budget)

    expanded: List[List[SiteAssignment]] = []
    seen_labels: set = set()
    for asgn in result.accepted:
        for concrete in expand_letter_choices(asgn, problem.wu_entries):
            canon = _canonical_label(concrete)
            if canon not in seen_labels:
                seen_labels.add(canon)
                expanded.append(concrete)
    return expanded
