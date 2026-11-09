"""The composition-free Wyckoff search (ADR-0033) against two oracles.

1. A brute-force enumeration of every multiset of (element, Wyckoff letter)
   sites: at most `max_sites` sites, at most `max_atoms` atoms in the
   primitive cell, every element present, charge-neutral, and no fixed
   (zero-free-parameter) letter used twice.
2. The union, over the compositions it finds, of the fixed-composition
   search with the same cell bound (ADR-0033 Q6 (i)).
"""

from itertools import combinations_with_replacement

import pytest

from mizuho.data import WYCKOFF_DB
from mizuho.search import AssignmentProblem
from mizuho.search.base import _canonical_label, has_fixed_overlap, reduced_formula
from mizuho.search.dfs import wyckoff_tree_search
from mizuho.structure import SiteAssignment
from mizuho.symmetry import centring_factor

ELEMENTS = ["Al", "O", "N"]
CHARGES = {"Al": 3, "O": -2, "N": -3}

# P, I, F, R and C lattices; cubic to triclinic.
CASES = [
    (221, 14, 4),
    (229, 14, 4),
    (225, 14, 4),
    (167, 14, 4),
    (186, 14, 4),
    (12, 14, 5),
    (15, 14, 5),
    (2, 14, 5),
]


def test_every_case_compares_something():
    """Every oracle case enumerates at least one assignment."""
    assert all(_brute_force(*case) for case in CASES)


def _config(sg, max_atoms, max_sites, strategy="best_first"):
    return {
        "spacegroup": sg,
        "elements": ELEMENTS,
        "ionic_charges": CHARGES,
        "tree_search": {
            "max_atoms": max_atoms,
            "max_sites": max_sites,
            "strategy": strategy,
        },
    }


def _brute_force(sg, max_atoms, max_sites):
    cap = max_atoms * centring_factor(sg)
    choices = [(e, w) for e in ELEMENTS for w in WYCKOFF_DB[sg]]
    found = set()
    for n in range(1, max_sites + 1):
        for combo in combinations_with_replacement(choices, n):
            if sum(w.multiplicity for _, w in combo) > cap:
                continue
            if {e for e, _ in combo} != set(ELEMENTS):
                continue
            if sum(CHARGES[e] * w.multiplicity for e, w in combo) != 0:
                continue
            asgn = [SiteAssignment(e, w) for e, w in combo]
            if has_fixed_overlap(asgn):
                continue
            found.add(_canonical_label(asgn))
    return found


@pytest.mark.parametrize("sg,max_atoms,max_sites", CASES)
@pytest.mark.parametrize("strategy", ["best_first", "exhaustive"])
def test_matches_brute_force(sg, max_atoms, max_sites, strategy):
    got = {
        _canonical_label(a)
        for a in wyckoff_tree_search(_config(sg, max_atoms, max_sites, strategy))
    }
    assert got == _brute_force(sg, max_atoms, max_sites)


@pytest.mark.parametrize("sg,max_atoms,max_sites", CASES)
def test_equals_union_of_fixed_composition_searches(sg, max_atoms, max_sites):
    free = wyckoff_tree_search(_config(sg, max_atoms, max_sites))
    by_comp = {}
    for a in free:
        by_comp.setdefault(reduced_formula(a, ELEMENTS), set()).add(_canonical_label(a))
    for comp, labels in by_comp.items():
        fixed_cfg = {
            "spacegroup": sg,
            "composition": comp,
            "ionic_charges": CHARGES,
            "tree_search": {
                "max_sites": max_sites,
                "max_formula_units": 1000,
                "max_atoms_per_fu": max_atoms * centring_factor(sg),
            },
        }
        fixed = {
            _canonical_label(a)
            for a in wyckoff_tree_search(fixed_cfg)
            if not has_fixed_overlap(a)
        }
        assert fixed == labels, comp


def test_some_case_is_nontrivial():
    """The oracles above must compare something: at least one case finds
    several compositions."""
    comps = {
        reduced_formula(a, ELEMENTS) for a in wyckoff_tree_search(_config(221, 14, 4))
    }
    assert len(comps) >= 2


def test_centring_factor():
    assert [centring_factor(g) for g in (1, 12, 148, 166, 225, 227, 229, 186)] == [
        1,
        2,
        3,
        3,
        4,
        4,
        2,
        1,
    ]


def test_reduced_formula_order_and_ones():
    from mizuho.data import WYCKOFF_DB as DB

    w = {e.letter: e for e in DB[221]}
    # Pm-3m: 1a, 3c, 3d -> Al 3 (c), O 3 (d), N 1 (a)
    asgn = [
        SiteAssignment("N", w["a"]),
        SiteAssignment("O", w["d"]),
        SiteAssignment("Al", w["c"]),
    ]
    assert reduced_formula(asgn, ELEMENTS) == "Al3O3N"


@pytest.mark.parametrize(
    "change,match",
    [
        ({"tree_search": {"max_sites": 3}}, "max_atoms"),
        (
            {"tree_search": {"max_atoms": 7, "max_formula_units": 2}},
            "max_formula_units",
        ),
        ({"ionic_charges": {"Al": 3, "O": -2, "N": 0}}, "non-zero ionic charge"),
        ({"ionic_charges": {"Al": 3, "O": 2, "N": 3}}, "cation and an anion"),
    ],
)
def test_rejects_incomplete_input(change, match):
    cfg = dict(_config(221, 7, 3), **change)
    with pytest.raises(ValueError, match=match):
        AssignmentProblem.from_config(cfg)


@pytest.mark.parametrize("sg", [225, 227, 166, 167, 229, 12])
def test_every_used_letter_is_in_the_letter_table(sg):
    """Each composition's config states the largest Z its assignments use,
    so geometry's letter-feasibility table covers every letter they use: a
    letter missing from it scores as perfect in the staged ranking. With
    the fixed-composition default of 4, F and R groups at 21 atoms lost
    5-15 used letters."""
    from mizuho.composition import parse_composition
    from mizuho.geometry.local_shell import _reachable_letters_for
    from mizuho.pipeline import max_formula_units

    groups = {}
    for a in wyckoff_tree_search(_config(sg, 21, 5)):
        groups.setdefault(reduced_formula(a, ELEMENTS), []).append(a)
    assert groups
    for comp, asgns in groups.items():
        formula, z = parse_composition(comp), max_formula_units(comp, asgns)
        for e in formula:
            used = {s.entry.letter for a in asgns for s in a if s.element == e}
            assert used <= set(_reachable_letters_for(WYCKOFF_DB[sg], e, formula, z)), (comp, e)
