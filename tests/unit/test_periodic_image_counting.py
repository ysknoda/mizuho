"""Physics gate for the periodic-image counting of ADR-0001.

Every case here has a coordination number that is a textbook fact, so the
objective either reproduces it or it is wrong. Before ADR-0001 the
coordination term reported NaCl as 3 (true 6), CsCl as 1 (true 8) and fcc
Cu as 3 (true 12), because it reduced each neighbour atom to its single
nearest periodic image and so collapsed several images of one site into
one countable neighbour.
"""

import numpy as np
import pytest

from mizuho.lattice import (
    build_lattice_matrix, self_image_distances, shortest_lattice_translation,
)
from mizuho.objective.breakdown import error_breakdown
from mizuho.objective.energy import compute_error
from mizuho.structure import assignments_to_sites, resolve_fixed_assignment

# (name, lattice constant, {element: fractional coords}, centre element,
#  neighbour element, cutoff, true CN)
_CASES = [
    ("rock salt NaCl", 5.64,
     {"Na": [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]],
      "Cl": [[.5, .5, .5], [0, 0, .5], [0, .5, 0], [.5, 0, 0]]},
     "Na", "Cl", 3.40, 6),
    ("caesium chloride CsCl", 4.11,
     {"Cs": [[0, 0, 0]], "Cl": [[.5, .5, .5]]},
     "Cs", "Cl", 4.00, 8),
    ("fluorite CaF2", 5.46,
     {"Ca": [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]],
      "F": [[x, y, z] for x in (.25, .75) for y in (.25, .75) for z in (.25, .75)]},
     "Ca", "F", 3.00, 8),
    ("diamond Si", 5.43,
     {"Si": [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0],
             [.25, .25, .25], [.25, .75, .75], [.75, .25, .75], [.75, .75, .25]]},
     "Si", "Si", 2.70, 4),
    ("fcc Cu", 3.61,
     {"Cu": [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]]},
     "Cu", "Cu", 3.00, 12),
]


def _p1_config(a, atoms, extra=None):
    """SG1 (P1, no symmetry): every listed coordinate is one atom."""
    specs, coords = [], []
    for elem, pts in atoms.items():
        for pt in pts:
            specs.append({"element": elem, "letter": "a"})
            coords.extend(pt)
    sites = assignments_to_sites(resolve_fixed_assignment(1, specs))
    cfg = {"spacegroup": 1, "wyckoff_sites": sites}
    if extra:
        cfg.update(extra)
    pv = np.array([a, a, a, 90.0, 90.0, 90.0] + [float(c) for c in coords])
    return cfg, pv


@pytest.mark.parametrize("name,a,atoms,centre,neigh,cutoff,true_cn",
                         _CASES, ids=[c[0] for c in _CASES])
def test_objective_reproduces_textbook_coordination_number(
        name, a, atoms, centre, neigh, cutoff, true_cn):
    """A constraint asking for exactly the true CN must be satisfied
    (penalty 0); one asking for anything else must not be."""
    def cn_cfg(lo, hi):
        return {"coordination_constraints": [{
            "center_atom": centre, "neighbor_atom": neigh,
            "cn_min": lo, "cn_max": hi, "cutoff_distance": cutoff,
            "smoothing_width": 0.01}]}

    cfg, pv = _p1_config(a, atoms, cn_cfg(true_cn, true_cn))
    entry = error_breakdown(pv, cfg)["cn"][0]
    assert entry["cn_per_atom"] == [true_cn] * len(entry["cn_per_atom"])
    assert entry["penalty"] == pytest.approx(0.0, abs=1e-6)

    # The old nearest-image count for these cases; must now be rejected.
    wrong = {6: 3, 8: 1, 12: 3}.get(true_cn)
    if wrong is not None and wrong != true_cn:
        cfg_w, pv_w = _p1_config(a, atoms, cn_cfg(wrong, wrong))
        assert error_breakdown(pv_w, cfg_w)["cn"][0]["penalty"] > 0.0


def test_hard_sphere_overlap_sees_an_atoms_own_periodic_image():
    """A lone Na (hard radius 0.95 A, so contact at 1.90 A) in a cubic
    cell has nothing but its own images for company. Before ADR-0001 the
    overlap term scored exactly 0.0 down to a = 0.2 A, leaving no
    physical floor under a collapsing lattice."""
    cfg, _ = _p1_config(6.0, {"Na": [[.5, .5, .5]]}, {"void": False})  # overlap alone (ADR-0031)
    err = {}
    for a in (6.0, 3.0, 2.0, 1.5, 1.0, 0.5):
        _, pv = _p1_config(a, {"Na": [[.5, .5, .5]]})
        err[a] = compute_error(pv, cfg)

    assert err[6.0] == pytest.approx(0.0, abs=1e-12)   # comfortably apart
    assert err[3.0] == pytest.approx(0.0, abs=1e-12)   # still clear of 1.90
    assert err[1.5] > 0.0                              # inside contact
    assert err[1.0] > err[1.5]                         # monotone: worse as it shrinks
    assert err[0.5] > err[1.0]


def test_self_image_contact_distance_is_a_property_of_the_cell():
    lat = build_lattice_matrix(
        {"a": 4.0, "b": 5.0, "c": 6.0, "alpha": 90, "beta": 90, "gamma": 90})
    d = self_image_distances(lat)
    assert d.shape == (13,)
    assert shortest_lattice_translation(lat) == pytest.approx(4.0)
    # The three lattice vectors themselves must be among the contacts.
    for expected in (4.0, 5.0, 6.0):
        assert np.any(np.isclose(d, expected))


def test_breakdown_still_agrees_with_compute_error():
    cfg, pv = _p1_config(5.64, _CASES[0][2], {
        "coordination_constraints": [{
            "center_atom": "Na", "neighbor_atom": "Cl",
            "cn_min": 6, "cn_max": 6, "cutoff_distance": 3.40}]})
    assert error_breakdown(pv, cfg)["total"] == pytest.approx(
        compute_error(pv, cfg), abs=1e-9)


# (name, lattice constant, atoms, centre, neighbour, cutoff, counted angles,
#  min angle, max angle). Angles above ignore_trans_above (150 deg) are not
# counted. Simple cubic: 6 neighbours, C(6,2) = 15 pairs, 3 of them at 180,
# 12 at 90. CsCl: 8 Cl at the cube corners, C(8,2) = 28 pairs: 12 at
# arccos(1/3) = 70.53, 12 at arccos(-1/3) = 109.47, 4 at 180.
_LONE_NEIGHBOUR_CASES = [
    ("simple cubic Si", 2.35, {"Si": [[0, 0, 0]]}, "Si", "Si", 2.60,
     12, 90.0, 90.0),
    ("caesium chloride CsCl", 4.11, {"Cs": [[0, 0, 0]], "Cl": [[.5, .5, .5]]},
     "Cs", "Cl", 4.00, 24, 70.5288, 109.4712),
]


@pytest.mark.parametrize("name,a,atoms,centre,neigh,cutoff,n_angles,lo,hi",
                         _LONE_NEIGHBOUR_CASES, ids=[c[0] for c in _LONE_NEIGHBOUR_CASES])
def test_angle_term_counts_images_of_a_lone_neighbour_atom(
        name, a, atoms, centre, neigh, cutoff, n_angles, lo, hi):
    """A neighbour species with one atom in the cell still has many
    neighbour images, and they subtend real angles at the centre
    (ADR-0001). The angle term used to skip such a constraint outright, so
    a one-atom square net scored zero against a tetrahedral band."""
    cfg, pv = _p1_config(a, atoms, {"angle_constraints": [{
        "atom_center": centre, "atom_neighbor": neigh,
        "angle_min": 105.0, "angle_max": 115.0, "angle_cutoff": cutoff}]})
    entry = error_breakdown(pv, cfg)["angle"][0]
    assert entry["n_triplets"] == n_angles
    assert entry["angle_min_observed"] == pytest.approx(lo, abs=1e-3)
    assert entry["angle_max_observed"] == pytest.approx(hi, abs=1e-3)
    assert entry["penalty"] > 0.0
    assert compute_error(pv, cfg) == pytest.approx(error_breakdown(pv, cfg)["total"], abs=1e-9)
    assert compute_error(pv, cfg) > 0.0


def test_angle_term_still_accepts_diamond():
    """Diamond Si's 109.47 deg angles sit inside [105, 115]: penalty 0."""
    cfg, pv = _p1_config(5.43, _CASES[3][2], {"angle_constraints": [{
        "atom_center": "Si", "atom_neighbor": "Si",
        "angle_min": 105.0, "angle_max": 115.0, "angle_cutoff": 2.70}]})
    entry = error_breakdown(pv, cfg)["angle"][0]
    assert entry["n_triplets"] == 8 * 6
    assert entry["angle_min_observed"] == pytest.approx(109.4712, abs=1e-3)
    assert compute_error(pv, cfg) == pytest.approx(0.0, abs=1e-12)
