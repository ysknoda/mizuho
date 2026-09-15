"""Sanity checks for the Wyckoff position database (mizuho.data)."""

import numpy as np

from mizuho.data import WYCKOFF_DB, build_point, expand_orbit, get_ops

GENERIC_VALUES = {"x": 0.311288, "y": 0.472891, "z": 0.195329}


def test_all_230_space_groups_present():
    assert len(WYCKOFF_DB) == 230
    assert set(WYCKOFF_DB.keys()) == set(range(1, 231))


def test_total_entry_count():
    total = sum(len(v) for v in WYCKOFF_DB.values())
    assert total == 1731


def test_multiplicity_matches_true_orbit_size():
    """Every WyckoffEntry's declared multiplicity must equal the size of
    the orbit the generic expand_orbit() mechanism actually produces
    from the space group's true symmetry operations -- i.e. the data is
    internally self-consistent under the same mechanism the runtime
    structure builder uses (no per-entry hand-written generators exist
    to get out of sync with the declared multiplicity any more)."""
    mismatches = []
    for sg, entries in WYCKOFF_DB.items():
        rotations, translations = get_ops(sg)
        for entry in entries:
            point = build_point(entry.coord_template, GENERIC_VALUES)
            orbit = expand_orbit(rotations, translations, point)
            if len(orbit) != entry.multiplicity:
                mismatches.append((sg, entry.letter, entry.multiplicity, len(orbit)))
    assert not mismatches, f"multiplicity mismatches: {mismatches[:10]}"


def _slot_coeffs(slot):
    if isinstance(slot, tuple):
        return slot[0]
    if isinstance(slot, dict):
        return slot
    return {}


def test_free_params_are_consistent_with_coord_template():
    """Every free parameter name referenced in coord_template must appear
    in the entry's free_params list, and vice versa (no declared free
    parameter that coord_template never actually uses)."""
    for sg, entries in WYCKOFF_DB.items():
        for entry in entries:
            used = {name for slot in entry.coord_template
                    for name in _slot_coeffs(slot)}
            assert used == set(entry.free_params), (
                f"SG {sg} {entry.letter}: coord_template references "
                f"{sorted(used)} but free_params={entry.free_params}")


def test_general_position_present_for_every_space_group():
    """Every space group must have an entry whose multiplicity equals
    the order of its symmetry-operation list (the general position) --
    9 space groups were missing this entirely in the original data."""
    for sg, entries in WYCKOFF_DB.items():
        rotations, _ = get_ops(sg)
        n_ops = len(rotations)
        assert any(e.multiplicity == n_ops for e in entries), (
            f"SG {sg}: no general position (n_ops={n_ops}) found")


def test_no_duplicate_letters_within_a_space_group():
    for sg, entries in WYCKOFF_DB.items():
        letters = [e.letter for e in entries]
        assert len(letters) == len(set(letters)), f"SG {sg}: duplicate letters {letters}"
