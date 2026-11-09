"""Unit tests for mizuho.composition."""

import pytest

from mizuho.composition import (
    _all_charges_zero,
    parse_composition,
)


def test_parse_composition_simple():
    assert parse_composition("NaCl") == {"Na": 1, "Cl": 1}


def test_parse_composition_with_counts():
    assert parse_composition("TiO2") == {"Ti": 1, "O": 2}
    assert parse_composition("La2CuO4") == {"La": 2, "Cu": 1, "O": 4}
    assert parse_composition("BaTiO3") == {"Ba": 1, "Ti": 1, "O": 3}


@pytest.mark.parametrize("comp", [
    "CsCl", "O5Zn2AsH", "Si", "CaMgSi2O6", "Al2SiO4F2", "CaTiO3",
    "CaSiO3", "MgAl2O4",
])
def test_parse_composition_accepts_every_fixture_composition(comp):
    """Every composition string used by tests/fixtures/*.yaml must keep
    parsing -- the validation added below must not narrow the accepted
    grammar for real input."""
    assert parse_composition(comp)


def test_parse_composition_tolerates_surrounding_whitespace():
    assert parse_composition("  NaCl  ") == {"Na": 1, "Cl": 1}


@pytest.mark.parametrize("bad, reason", [
    ("tio2",   "lower-case element symbol parses to nothing"),
    ("",       "empty string"),
    ("TiO2.5", "fractional count silently truncated to TiO2"),
    ("H2O!!",  "trailing punctuation silently discarded"),
    ("Na Cl",  "embedded whitespace"),
    ("2TiO",   "leading digit"),
    ("Ti0O2",  "zero count is meaningless in a formula"),
])
def test_parse_composition_rejects_unparseable_input(bad, reason):
    """Regression test: re.findall silently DISCARDED whatever did not
    match, so each of these produced a wrong-but-plausible composition
    that then defined the entire search problem. A typo here must fail
    loudly at the entry point. See .code-check/history.md.
    """
    with pytest.raises(ValueError, match="Invalid composition"):
        parse_composition(bad)


def test_all_charges_zero():
    assert _all_charges_zero({}, ["Si", "O"])
    assert _all_charges_zero({"Si": 0, "O": 0}, ["Si", "O"])
    assert not _all_charges_zero({"Na": 1, "Cl": -1}, ["Na", "Cl"])


def test_reduced_formula_round_trips_through_the_parser():
    """The composition-free search builds composition strings itself
    (search.base.reduced_formula, ADR-0033); the strict parser must accept
    every one it emits."""
    from mizuho.search.base import reduced_formula
    from mizuho.search.dfs import wyckoff_tree_search

    asgns = wyckoff_tree_search({
        "spacegroup": 221, "elements": ["Al", "O", "N"],
        "ionic_charges": {"Al": 3, "O": -2, "N": -3},
        "tree_search": {"max_atoms": 14, "max_sites": 4}})
    assert asgns
    for a in asgns:
        comp = reduced_formula(a, ["Al", "O", "N"])
        counts = {}
        for s in a:
            counts[s.element] = counts.get(s.element, 0) + s.entry.multiplicity
        parsed = parse_composition(comp)
        k = counts["Al"] // parsed["Al"]
        assert {e: n * k for e, n in parsed.items()} == counts
