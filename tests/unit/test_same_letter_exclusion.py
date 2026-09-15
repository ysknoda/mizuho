"""Regression tests for the zero_charge_mode SAME-LETTER EXCLUSION
(mizuho.search.dfs / best_first, sharing mizuho.search.base's
_shape_group_sizes / _fixed_representative_counts helpers).

The exclusion used to blanket-block ANY reuse of a representative
Wyckoff letter by the same element -- regardless of whether that
letter had free parameters. Two problems followed from that:

1. Letters WITH free parameters (e.g. a real MOF's 16f or 48g) can be
   used any number of times, by the same or different elements, since
   each use takes its own independently-optimised coordinate value.
   Blocking reuse of these was pure over-restriction with no physical
   basis.
2. Letters WITHOUT free parameters that share a shape group with a
   sibling letter (e.g. space group 227's 'a'/'b', both mult-8 fixed
   points collapsed to representative 'a' by canonical_representatives)
   were blocked from being reserved a second time at all -- even though
   the group has a second, geometrically distinct concrete letter ('b')
   that expand_letter_choices could hand the second reservation. This
   silently made split-site topologies like Si@c + Si@d unreachable for
   any zero_charge_mode composition needing 2 sites from that group.

Both are fixed by keying the exclusion on free-parameter-ness and
shape-group size instead of blanket letter identity -- see
search/base.py's _shape_group_sizes/_fixed_representative_counts
docstrings for the full reasoning.

Test compositions below all pair Si with a Ge "anchor" atom: a
single-element target is always trivially "complete" after its very
first site (any atom count is an integer multiple of a target of 1),
so the search never even attempts a second site down that path. Adding
Ge forces the search to keep going until BOTH elements are present at
a matching ratio, which is what actually exercises multi-site reuse of
the same shape group.
"""

from mizuho.data import WYCKOFF_DB
from mizuho.search import (AssignmentProblem, BestFirstBranchAndBoundStrategy,
                            ExhaustiveDFSStrategy, SearchBudget, _canonical_label)
from mizuho.search.base import _fixed_representative_counts, _shape_group_sizes
from mizuho.search.dfs import wyckoff_tree_search
from mizuho.structure import SiteAssignment

SG227 = WYCKOFF_DB[227]


def _entry(sg_entries, letter):
    return next(e for e in sg_entries if e.letter == letter)


# ---------------------------------------------------------------------------
# _shape_group_sizes / _fixed_representative_counts: the helpers directly.
# ---------------------------------------------------------------------------

def test_shape_group_sizes_counts_distinct_letters_sharing_a_shape():
    sizes = _shape_group_sizes(SG227)
    a, b = _entry(SG227, "a"), _entry(SG227, "b")
    assert a.multiplicity == b.multiplicity == 8
    assert not a.free_params and not b.free_params
    # 'a' and 'b' share one shape key (8, ()) -- exactly 2 members.
    from mizuho.search.base import _shape_key
    assert sizes[_shape_key(a)] == 2


def test_fixed_representative_counts_ignores_free_parameter_letters():
    e = _entry(SG227, "e")
    assert e.free_params  # sanity: 'e' has a free parameter
    assignments = [SiteAssignment("Si", e), SiteAssignment("Si", e)]
    assert _fixed_representative_counts(assignments) == {}


def test_fixed_representative_counts_is_cross_element():
    a = _entry(SG227, "a")
    assignments = [SiteAssignment("Si", a), SiteAssignment("Ge", a)]
    # Two DIFFERENT elements each reserving 'a' still counts as 2
    # reservations against the same shape group -- see module docstring.
    assert _fixed_representative_counts(assignments) == {"a": 2}


def test_shape_group_exactly_exhausted_by_two_cross_element_reservations():
    """The arithmetic a caller must combine to decide reuse: once 2
    reservations (by any mix of elements) are on the table for the
    {a, b} group, its size (2) is exactly met -- a 3rd must be pruned."""
    a = _entry(SG227, "a")
    assignments = [SiteAssignment("Si", a), SiteAssignment("Ge", a)]
    counts = _fixed_representative_counts(assignments)
    sizes = _shape_group_sizes(SG227)
    from mizuho.search.base import _shape_key
    assert counts["a"] >= sizes[_shape_key(a)]


# ---------------------------------------------------------------------------
# Search-strategy level: both DFS and best-first must agree, and the
# previously-unreachable topologies must now be reachable.
#
# Si:Ge = 4:1 forces Si@c + Si@d (16 + 16 = 32 atoms) alongside a
# separate Ge@a (8 atoms): 32/4 == 8/1 == 8, an integer ratio match.
# ---------------------------------------------------------------------------

def _accepted_labels(result):
    return sorted(_canonical_label(a) for a in result.accepted)


def _si_ge_problem(composition, max_sites, max_formula_units, max_atoms_per_fu):
    return AssignmentProblem(
        sg=227, composition=composition, elements=["Si", "Ge"],
        wu_entries=SG227, ionic_q={},
        zero_charge_mode=True, allow_dup_wyckoff=False,
        tol_min=0.0, tol_max=0.0, max_sites=max_sites,
        max_formula_units=max_formula_units, max_atoms_per_fu=max_atoms_per_fu,
        max_q_per_atom=0,
    )


def test_zero_free_param_shape_group_reuse_now_reachable_up_to_its_size():
    """Si@c + Si@d (shape group {c, d}, size 2) alongside Ge@a must now
    be explorable -- previously the second Si reservation of
    representative 'c' was pruned outright, so this raw (pre-expansion)
    result never existed."""
    problem = _si_ge_problem({"Si": 4, "Ge": 1}, max_sites=3,
                              max_formula_units=8, max_atoms_per_fu=40)
    budget = SearchBudget()
    dfs = ExhaustiveDFSStrategy().search(problem, budget)
    best_first = BestFirstBranchAndBoundStrategy().search(problem, budget)

    assert _accepted_labels(dfs) == _accepted_labels(best_first)
    assert any(
        sum(1 for a in asgn if a.element == "Si" and a.entry.letter == "c") == 2
        for asgn in dfs.accepted)


def test_zero_free_param_shape_group_reuse_still_bounded_by_its_size():
    """No accepted assignment may reserve representative 'c' (shape
    group {c, d}, size 2) 3 or more times -- the group only has 2
    distinct concrete letters, so a 3rd reservation could never be
    handed a non-colliding coordinate, regardless of which element(s)
    hold the reservations."""
    problem = _si_ge_problem({"Si": 4, "Ge": 1}, max_sites=3,
                              max_formula_units=8, max_atoms_per_fu=40)
    budget = SearchBudget()
    dfs = ExhaustiveDFSStrategy().search(problem, budget)
    best_first = BestFirstBranchAndBoundStrategy().search(problem, budget)

    assert _accepted_labels(dfs) == _accepted_labels(best_first)
    assert not any(
        sum(1 for a in asgn if a.entry.letter == "c") >= 3
        for asgn in dfs.accepted)


def test_free_param_letter_reuse_now_unrestricted():
    """'e' (mult=32, one free parameter, its own shape group of size 1)
    reused twice for Si (alongside Ge@a) must now be reachable -- each
    reservation takes its own x value later, so there is no physical
    limit to reuse. Si:Ge = 8:1 -> Si@e + Si@e (64) vs Ge@a (8)."""
    problem = _si_ge_problem({"Si": 8, "Ge": 1}, max_sites=3,
                              max_formula_units=8, max_atoms_per_fu=72)
    budget = SearchBudget()
    dfs = ExhaustiveDFSStrategy().search(problem, budget)
    best_first = BestFirstBranchAndBoundStrategy().search(problem, budget)

    assert _accepted_labels(dfs) == _accepted_labels(best_first)
    assert any(
        sum(1 for a in asgn if a.element == "Si" and a.entry.letter == "e") == 2
        for asgn in dfs.accepted)


# ---------------------------------------------------------------------------
# Full pipeline (wyckoff_tree_search, including expand_letter_choices):
# the CONCRETE split-letter combination must actually surface.
# ---------------------------------------------------------------------------

def test_wyckoff_tree_search_finds_the_concrete_split_site_combination():
    config = {
        "composition": "Si4Ge",
        "spacegroup": 227,
        "charge_tolerance_min": 0,
        "charge_tolerance_max": 0,
        "tree_search": {"max_sites": 3, "max_formula_units": 8,
                         "max_atoms_per_fu": 40},
    }
    assignments = wyckoff_tree_search(config)
    si_letter_pairs = [
        sorted(a.entry.letter for a in asgn if a.element == "Si")
        for asgn in assignments]
    assert ["c", "d"] in si_letter_pairs
