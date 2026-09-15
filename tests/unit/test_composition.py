"""Unit tests for mizuho.composition."""

import pytest

from mizuho.composition import (
    _all_charges_zero,
    discover_compositions,
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


def test_discover_compositions_output_round_trips_through_the_parser():
    """discover_compositions builds composition strings itself; the
    stricter parser must accept everything it emits."""
    comps = discover_compositions(
        225, ["Na", "Cl"], {"Na": 1, "Cl": -1}, max_compositions=5)
    assert comps
    for c in comps:
        assert parse_composition(c)


def test_all_charges_zero():
    assert _all_charges_zero({}, ["Si", "O"])
    assert _all_charges_zero({"Si": 0, "O": 0}, ["Si", "O"])
    assert not _all_charges_zero({"Na": 1, "Cl": -1}, ["Na", "Cl"])


def test_discover_compositions_zero_charge_returns_nonempty():
    """
    Regression test for a Phase-1 bug fix: discover_compositions used to
    always return [] for all-zero-charge (covalent) element sets, because
    it unconditionally tried to solve a charge-neutrality equation whose
    pivot coefficient is always 0 when every element has charge 0.
    """
    result = discover_compositions(
        sg=227, elements=["Si"], ionic_charges={},
        max_sites_per_elem=2, max_atoms=24)
    assert result, "expected at least one reachable covalent composition"
    assert all(c.startswith("Si") for c in result)


def test_discover_compositions_zero_charge_binary():
    result = discover_compositions(
        sg=227, elements=["Si", "C"], ionic_charges={},
        max_sites_per_elem=2, max_atoms=24)
    assert result


def test_discover_compositions_ionic_still_works():
    """Regression guard: the zero-charge-mode fix must not break the
    pre-existing ionic-mode code path."""
    result = discover_compositions(
        sg=225, elements=["Na", "Cl"],
        ionic_charges={"Na": 1, "Cl": -1},
        max_sites_per_elem=2, max_atoms=24)
    assert result
    assert "Na4Cl4" in result or any(c.startswith("Na") for c in result)
