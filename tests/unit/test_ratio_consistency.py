"""Regression tests for the Wyckoff-assignment ratio-consistency prune
(mizuho.search.base._ratio_consistent).

A previous version of this check compared every touched element's
CURRENT ratio-to-target and required them all within 1.01 of each
other. That is unsound whenever two or more elements each need
multiple sequential Wyckoff-site picks to reach their target count:
whichever element is fully resolved first (in canonical processing
order) permanently locks in its final ratio, and the next element's
in-progress partial ratio sits far below it until its last pick lands
-- so the assignment was pruned long before it could complete. That
silently made it impossible to find assignments where more than one
element occupies multiple symmetry-inequivalent Wyckoff sites, e.g. a
mineral with two distinct cation sites (adamite Zn2(AsO4)(OH), SG 58,
is the case that surfaced this: Zn must split across two different
4-fold sites and O across three different sites to reach the real
Z=4 cell).
"""

from mizuho.search.base import _ratio_consistent


def test_active_element_may_be_partway_to_frozen_ratio():
    # O is frozen at ratio 4.0 (20/5); Zn (active) has only reached
    # half its k=4 target so far (4/2 = 2.0 <= 4.0) -- must NOT be
    # pruned, since more Zn picks can still follow.
    new_counts = {"O": 20, "Zn": 4}
    composition = {"O": 5, "Zn": 2, "As": 1, "H": 1}
    elements = ["O", "Zn", "As", "H"]
    assert _ratio_consistent(new_counts, composition, elements, "Zn")


def test_active_element_reaching_frozen_ratio_exactly_is_fine():
    new_counts = {"O": 20, "Zn": 8}
    composition = {"O": 5, "Zn": 2, "As": 1, "H": 1}
    elements = ["O", "Zn", "As", "H"]
    assert _ratio_consistent(new_counts, composition, elements, "Zn")


def test_active_element_overshooting_frozen_ratio_is_pruned():
    # Zn already exceeds O's locked ratio (4.0) -- can never come back
    # down, so this branch can never complete.
    new_counts = {"O": 20, "Zn": 10}
    composition = {"O": 5, "Zn": 2, "As": 1, "H": 1}
    elements = ["O", "Zn", "As", "H"]
    assert not _ratio_consistent(new_counts, composition, elements, "Zn")


def test_frozen_elements_must_match_exactly():
    # O locked at ratio 4.0, Zn locked (index < As) at ratio 3.0 --
    # inconsistent, dead branch regardless of As.
    new_counts = {"O": 20, "Zn": 6, "As": 1}
    composition = {"O": 5, "Zn": 2, "As": 1, "H": 1}
    elements = ["O", "Zn", "As", "H"]
    assert not _ratio_consistent(new_counts, composition, elements, "As")


def test_first_element_alone_is_never_pruned():
    new_counts = {"O": 2}
    composition = {"O": 5, "Zn": 2, "As": 1, "H": 1}
    elements = ["O", "Zn", "As", "H"]
    assert _ratio_consistent(new_counts, composition, elements, "O")


def test_multi_site_per_element_assignment_now_reachable():
    """End-to-end guard: the discrete search must be able to find at
    least one assignment where Zn occupies two distinct Wyckoff
    letters (impossible before the _ratio_consistent fix -- see
    module docstring)."""
    from collections import defaultdict

    from mizuho.search.base import AssignmentProblem, SearchBudget
    from mizuho.search.best_first import BestFirstBranchAndBoundStrategy

    config = {
        "spacegroup": 58,
        "composition": "O5Zn2AsH",
        "ionic_charges": {"Zn": 2, "As": 5, "O": -2, "H": 1},
        "charge_tolerance_min": 0,
        "charge_tolerance_max": 0,
        "tree_search": {"max_sites": 8, "max_formula_units": 4},
    }
    problem = AssignmentProblem.from_config(config)
    budget = SearchBudget.from_config(config)
    result = BestFirstBranchAndBoundStrategy().search(problem, budget)

    found_multi_site_zn = False
    for asgn in result.accepted:
        letters = defaultdict(set)
        counts = defaultdict(int)
        for a in asgn:
            letters[a.element].add(a.entry.letter)
            counts[a.element] += a.entry.multiplicity
        if counts.get("Zn", 0) == 8 and len(letters["Zn"]) >= 2:
            found_multi_site_zn = True
            break

    assert found_multi_site_zn
