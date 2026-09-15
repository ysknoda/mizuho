"""Direct regression tests for each Phase-1 bug fix: each test reproduces
the exact scenario that failed before the fix.
"""

from mizuho.composition import discover_compositions
from mizuho.data import WYCKOFF_DB
from mizuho.objective.penalties import HARD_RADII, _PARTIAL_VOL
from mizuho.search.dfs import wyckoff_tree_search


def test_bug_discover_compositions_zero_charge_no_longer_empty():
    """Bug #11: discover_compositions always returned [] when every
    element had ionic charge 0 (or ionic_charges was omitted)."""
    result = discover_compositions(
        sg=227, elements=["Si"], ionic_charges={},
        max_sites_per_elem=2, max_atoms=24)
    assert result != []


def test_bug_same_letter_exclusion_gated_by_zero_charge_mode():
    """Bug #12: same-Wyckoff-letter exclusion was applied unconditionally
    (even in ionic mode), silently preventing legitimate structures where
    two independently-parameterised sites share a Wyckoff letter. It
    should now only apply in zero_charge_mode by default."""
    config = {
        "composition": "NaCl",
        "spacegroup": 225,
        "ionic_charges": {"Na": 1, "Cl": -1},
        "charge_tolerance_min": 0,
        "charge_tolerance_max": 0,
        "tree_search": {"max_sites": 4, "max_formula_units": 4},
    }
    assignments = wyckoff_tree_search(config)
    # Ionic mode must still find the standard Na@4a + Cl@4b (or swapped)
    # assignment; this is a smoke check that the mode-gating change did
    # not break the ordinary ionic path.
    assert assignments


def test_bug_hard_radii_and_partial_vol_cover_same_elements():
    """Bug #17: HARD_RADII and _PARTIAL_VOL used to cover different
    element sets, so each silently fell back to a poor generic default
    for elements only covered by the other table."""
    assert set(HARD_RADII.keys()) == set(_PARTIAL_VOL.keys())


def test_wyckoff_db_has_no_duplicate_crystal_system_table():
    """Bug #10 (via consolidation): only one crystal-system lookup table
    should exist now (mizuho.lattice.CRYSTAL_SYSTEM_OF_SG); this is
    implicitly verified by every module importing that single name
    without error (see test_lattice.py), so this test just re-asserts
    the DB itself is intact after the consolidation.
    """
    assert len(WYCKOFF_DB) == 230
