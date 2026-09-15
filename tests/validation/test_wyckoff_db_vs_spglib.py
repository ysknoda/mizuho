"""Cross-validate the Wyckoff position database against spglib's
authoritative space-group symmetry data.

Requires the optional `spglib` test dependency (`pip install ".[test]"`).
spglib is dev/test-only: mizuho.data.symmetry_ops.SYMMETRY_OPS_OF_SG is
static data generated once by scripts/generate_symmetry_ops_table.py,
and the runtime orbit-expansion mechanism (mizuho.data._orbit) is plain
numpy with no spglib dependency.

Three independent checks per space group:
  1. mizuho's own static SYMMETRY_OPS_OF_SG matches a FRESH pull from
     spglib.get_symmetry_from_database (i.e. the committed static data
     hasn't drifted from what spglib would generate today).
  2. Every WyckoffEntry's declared multiplicity equals the true orbit
     size under those operations (this is also asserted in
     tests/unit/test_wyckoff_db.py without spglib; repeated here against
     a live spglib pull rather than mizuho's own static copy).
  3. An anchored trial structure built from each entry's true orbit
     re-analyzes (via spglib.get_symmetry_dataset) back to the exact
     same space group number with a single, unambiguous Wyckoff letter
     matching the one stored in the database.
"""
import numpy as np
import pytest

spglib = pytest.importorskip("spglib")

from mizuho.data import WYCKOFF_DB, build_point, expand_orbit
from mizuho.lattice import get_lattice_system, build_lattice_matrix

GENERIC_VALUES = {"x": 0.311288, "y": 0.472891, "z": 0.195329}
ANCHOR_POINT = np.array([0.112837, 0.246893, 0.378512])

LATTICE_PARAMS = {
    "cubic":        {"a": 5.43, "b": 5.43, "c": 5.43},
    "tetragonal":   {"a": 4.7,  "b": 4.7,  "c": 3.1},
    "orthorhombic": {"a": 4.1,  "b": 5.3,  "c": 6.7},
    "hexagonal":    {"a": 4.9,  "b": 4.9,  "c": 7.2, "gamma": 120.0},
    "trigonal":     {"a": 4.9,  "b": 4.9,  "c": 7.2, "gamma": 120.0},
    "monoclinic":   {"a": 5.1,  "b": 6.2,  "c": 7.3, "beta": 99.5},
    "triclinic":    {"a": 4.3,  "b": 5.1,  "c": 6.6,
                      "alpha": 88.0, "beta": 95.0, "gamma": 102.0},
}


def _points_equal(p, q, tol=1e-4):
    d = np.asarray(p) - np.asarray(q)
    d -= np.round(d)
    return np.all(np.abs(d) < tol)


def _hall_number_for(sg):
    for hall in range(1, 531):
        if spglib.get_spacegroup_type(hall).number == sg:
            return hall
    raise AssertionError(f"no Hall number found for SG {sg}")


@pytest.fixture(scope="module")
def sg_to_hall():
    table = {}
    for hall in range(1, 531):
        info = spglib.get_spacegroup_type(hall)
        table.setdefault(info.number, hall)
    return table


@pytest.mark.parametrize("sg", sorted(WYCKOFF_DB.keys()))
def test_static_symmetry_ops_match_live_spglib(sg, sg_to_hall):
    from mizuho.data.symmetry_ops import SYMMETRY_OPS_OF_SG

    hall = sg_to_hall[sg]
    live = spglib.get_symmetry_from_database(hall)
    live_rot = [r.tolist() for r in live["rotations"]]
    live_trans = [[round(float(x), 12) for x in t] for t in live["translations"]]

    stored_rot, stored_trans = SYMMETRY_OPS_OF_SG[sg]
    assert stored_rot == live_rot, f"SG {sg}: rotation list drifted from spglib"
    assert len(stored_trans) == len(live_trans)
    for a, b in zip(stored_trans, live_trans):
        assert all(abs(x - y) < 1e-9 for x, y in zip(a, b)), (
            f"SG {sg}: translation drifted from spglib")


@pytest.mark.parametrize("sg", sorted(WYCKOFF_DB.keys()))
def test_multiplicity_and_letter_confirmed_by_spglib(sg, sg_to_hall):
    hall = sg_to_hall[sg]
    ops = spglib.get_symmetry_from_database(hall)
    rotations, translations = ops["rotations"], ops["translations"]
    n_ops = len(rotations)

    lattice_sys = get_lattice_system(sg)
    lattice = build_lattice_matrix(LATTICE_PARAMS[lattice_sys])

    anchor_orbit = expand_orbit(rotations, translations, ANCHOR_POINT)

    for entry in WYCKOFF_DB[sg]:
        point = build_point(entry.coord_template, GENERIC_VALUES)
        orbit = expand_orbit(rotations, translations, point)
        assert len(orbit) == entry.multiplicity, (
            f"SG {sg} {entry.letter}: declared multiplicity "
            f"{entry.multiplicity} != true orbit size {len(orbit)}")

        this_anchor = [p for p in anchor_orbit
                       if not any(_points_equal(p, q) for q in orbit)]
        positions = list(orbit) + this_anchor
        numbers = [14] * len(orbit) + [8] * len(this_anchor)
        dataset = spglib.get_symmetry_dataset((lattice, positions, numbers),
                                               symprec=1e-4)
        assert dataset is not None and dataset.number == sg, (
            f"SG {sg} {entry.letter}: re-analysis gave "
            f"{dataset.number if dataset else None}, expected {sg}")
        letters = set(dataset.wyckoffs[:len(orbit)])
        equiv = set(dataset.equivalent_atoms[:len(orbit)])
        assert len(letters) == 1 and len(equiv) == 1, (
            f"SG {sg} {entry.letter}: orbit split into multiple site classes")
        assert letters == {entry.letter}, (
            f"SG {sg}: stored letter {entry.letter!r} but spglib confirms "
            f"{letters!r}")
