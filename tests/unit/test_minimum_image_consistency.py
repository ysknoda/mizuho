"""Regression tests for two defects found by the code-check review of
the objective function (see .code-check/history.md).

1. The angle and hybridisation terms of compute_error/error_breakdown
   used a *component-wise* `diff -= np.round(diff)` minimum-image
   convention, while every distance-based term (overlap, bond, CN,
   repulsion) used lattice.pairwise_distances_pbc's explicit search
   over all 27 neighbouring images. Those two agree only for
   ORTHOGONAL cells. For a skewed (monoclinic/triclinic) cell they can
   disagree badly -- lattice.py's own _PAIRWISE_DOC explains exactly
   why -- so the angle term could look at a completely different
   periodic image of a neighbour than the CN term that counted it,
   silently dropping genuine bond angles from the penalty sum.

2. `np.concatenate([])` raises ValueError, so a coordination_constraint
   whose neighbour specification expands to NO species at all (a
   wildcard "X" in a structure with no opposite-sign species, or an
   explicit element list naming only absent elements) crashed the
   objective instead of contributing nothing.
"""

import numpy as np
import pytest

from mizuho.lattice import (
    build_lattice_matrix, minimum_image_vectors, pairwise_distances_pbc,
)
from mizuho.objective.breakdown import error_breakdown
from mizuho.objective.energy import compute_error
from mizuho.optimize.init_guess import _gradient_lattice_hint
from mizuho.structure import assignments_to_sites, resolve_fixed_assignment

# A triclinic cell well inside tests/fixtures/wollastonite_casio3.yaml's
# own lattice_bounds (alpha 70-110, beta 75-115, gamma 85-125 deg), i.e.
# a shape the optimiser is actively allowed to visit for that fixture.
_SKEW_LP = {"a": 5.0, "b": 5.0, "c": 5.0,
            "alpha": 70.0, "beta": 115.0, "gamma": 125.0}

# Found by random search in that cell (see .code-check/history.md): the
# centre-to-N2 separation is 2.1041 A by the true minimum image but
# 3.6550 A under component-wise rounding -- a 74% overstatement that
# pushes a real bond right outside any sane angle cutoff.
_CENTRE = [0.874077, 0.662215, 0.131616]
_NEIGH1 = [0.845074, 0.944948, 0.903917]   # 1.5200 A (both conventions)
_NEIGH2 = [0.569719, 0.145460, 0.192463]   # 2.1041 A true / 3.6550 A rounded


def _si_o2_triclinic_config(**extra):
    """SG1 (P1, triclinic, no symmetry): one Si + two O at exactly the
    fractional coordinates given, so atom indices map 1:1 to them."""
    sites = assignments_to_sites(resolve_fixed_assignment(
        1, [{"element": "Si", "letter": "a"},
            {"element": "O", "letter": "a"},
            {"element": "O", "letter": "a"}]))
    cfg = {"spacegroup": 1, "wyckoff_sites": sites}
    cfg.update(extra)
    return cfg


def _si_o2_triclinic_pv():
    lp = _SKEW_LP
    return np.array([lp["a"], lp["b"], lp["c"],
                     lp["alpha"], lp["beta"], lp["gamma"]]
                    + _CENTRE + _NEIGH1 + _NEIGH2)


# ---------------------------------------------------------------------
# 1. minimum_image_vectors itself
# ---------------------------------------------------------------------

@pytest.mark.parametrize("lp", [
    {"a": 4.0, "b": 4.0, "c": 4.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
    {"a": 4.0, "b": 5.5, "c": 7.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
    {"a": 5.0, "b": 5.0, "c": 5.0, "alpha": 90.0, "beta": 105.8, "gamma": 90.0},
    _SKEW_LP,
])
def test_minimum_image_vector_lengths_match_pairwise_distances_pbc(lp):
    """The whole point of the helper: its distances must be the SAME
    minimum image the rest of the objective already uses, for every
    crystal system -- that is the invariant the angle term was
    violating."""
    lat = build_lattice_matrix(lp)
    rng = np.random.default_rng(1234)
    fa = rng.random((6, 3))
    fb = rng.random((5, 3))

    _, dists = minimum_image_vectors(fa, fb, lat)
    assert dists == pytest.approx(pairwise_distances_pbc(fa, fb, lat), abs=1e-9)


def test_minimum_image_vectors_are_consistent_with_their_own_lengths():
    lat = build_lattice_matrix(_SKEW_LP)
    rng = np.random.default_rng(99)
    fa, fb = rng.random((4, 3)), rng.random((3, 3))
    vecs, dists = minimum_image_vectors(fa, fb, lat)
    assert np.linalg.norm(vecs, axis=2) == pytest.approx(dists, abs=1e-12)


def test_minimum_image_beats_componentwise_rounding_in_a_skewed_cell():
    """Pins the specific numbers quoted in this module's docstring, so a
    future refactor that quietly reintroduces component-wise rounding
    fails here with a readable diff rather than only showing up as a
    slightly different fixture error."""
    lat = build_lattice_matrix(_SKEW_LP)
    centre = np.array(_CENTRE)[None, :]
    neigh = np.array([_NEIGH1, _NEIGH2])

    _, dists = minimum_image_vectors(centre, neigh, lat)

    rounded = centre[:, None, :] - neigh[None, :, :]
    rounded -= np.round(rounded)
    rounded_d = np.linalg.norm(rounded @ lat, axis=2)

    assert dists[0, 0] == pytest.approx(1.5200, abs=5e-4)
    assert dists[0, 1] == pytest.approx(2.1041, abs=5e-4)
    # Neighbour 1 happens to agree; neighbour 2 is the one that exposes
    # the bug (component-wise rounding overstates it by ~1.55 A).
    assert rounded_d[0, 0] == pytest.approx(dists[0, 0], abs=5e-4)
    assert rounded_d[0, 1] == pytest.approx(3.6550, abs=5e-4)


# ---------------------------------------------------------------------
# 2. The angle term must see the same neighbours the CN term counts
# ---------------------------------------------------------------------

def test_angle_term_finds_bond_angle_missed_by_componentwise_rounding():
    """Both Si-O separations (1.52 A, 2.10 A) are genuine bonds inside
    the 2.6 A angle cutoff, so exactly one O-Si-O angle (136.5 deg)
    exists and violates the [100, 120] band. Before the fix the angle
    term measured neighbour 2 at 3.66 A, dropped it, and reported ZERO
    triplets and ZERO penalty for a structure that plainly has one."""
    cfg = _si_o2_triclinic_config(angle_constraints=[{
        "atom_center": "Si", "atom_neighbor": "O",
        "angle_min": 100.0, "angle_max": 120.0,
        "weight": 1.0, "angle_cutoff": 2.6}])
    pv = _si_o2_triclinic_pv()

    entry = error_breakdown(pv, cfg)["angle"][0]
    assert entry["n_triplets"] == 1
    assert entry["n_violated"] == 1
    assert entry["angle_min_observed"] == pytest.approx(136.49, abs=0.05)
    assert entry["penalty"] > 0.0
    # breakdown must still agree with the value actually optimised.
    assert error_breakdown(pv, cfg)["total"] == pytest.approx(
        compute_error(pv, cfg), abs=1e-9)


def test_angle_and_cn_terms_agree_on_the_neighbour_set():
    """The structural invariant behind the bug: an atom the CN term
    counts as a neighbour (soft count ~2 within 2.6 A) must be an atom
    the angle term can also form an angle with."""
    common = dict(
        coordination_constraints=[{
            "center_atom": "Si", "neighbor_atom": "O",
            "cn_min": 2, "cn_max": 2, "cutoff_distance": 2.6}],
        angle_constraints=[{
            "atom_center": "Si", "atom_neighbor": "O",
            "angle_min": 100.0, "angle_max": 120.0,
            "weight": 1.0, "angle_cutoff": 2.6}])
    pv = _si_o2_triclinic_pv()
    bd = error_breakdown(pv, _si_o2_triclinic_config(**common))

    assert bd["cn"][0]["cn_per_atom"] == [2]      # CN term sees both O
    assert bd["angle"][0]["n_triplets"] == 1      # so must the angle term


def test_orthogonal_cells_are_unaffected_by_the_minimum_image_fix():
    """Component-wise rounding is provably correct for an orthogonal
    lattice (the squared distance is separable per axis), so this fix
    must be an exact no-op for the cubic/tetragonal/orthorhombic
    fixtures -- guards against it being mistaken for a behaviour
    change there."""
    lat = build_lattice_matrix(
        {"a": 4.0, "b": 6.5, "c": 9.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0})
    rng = np.random.default_rng(2024)
    fa, fb = rng.random((7, 3)), rng.random((7, 3))

    vecs, _ = minimum_image_vectors(fa, fb, lat)
    rounded = fa[:, None, :] - fb[None, :, :]
    rounded -= np.round(rounded)
    assert vecs == pytest.approx(rounded @ lat, abs=1e-12)


# ---------------------------------------------------------------------
# 3. Empty wildcard expansion must be a no-op, not a crash
# ---------------------------------------------------------------------

def _single_si_config(**extra):
    sites = assignments_to_sites(resolve_fixed_assignment(
        1, [{"element": "Si", "letter": "a"}]))
    cfg = {"spacegroup": 1, "wyckoff_sites": sites}
    cfg.update(extra)
    return cfg


_EMPTY_EXPANSION_CASES = [
    # "X" on a covalent config (no ionic_charges -> centre charge 0)
    # expands to "every OTHER species", and Si is the only species.
    pytest.param({"neighbor_atom": "X"}, {}, id="wildcard-single-element"),
    # "X" on a cation centre in a structure containing no anion at all.
    pytest.param({"neighbor_atom": "X"}, {"ionic_charges": {"Si": 4}},
                 id="wildcard-no-opposite-sign"),
    # Explicit element list naming only species absent from the cell.
    pytest.param({"neighbor_atom": ["O", "F"]}, {}, id="list-all-absent"),
]


@pytest.mark.parametrize("cc_extra, cfg_extra", _EMPTY_EXPANSION_CASES)
def test_compute_error_survives_empty_neighbour_expansion(cc_extra, cfg_extra):
    cc = {"center_atom": "Si", "cn_min": 4, "cn_max": 4,
          "cutoff_distance": 2.6}
    cc.update(cc_extra)
    cfg = _single_si_config(coordination_constraints=[cc], **cfg_extra)
    baseline = _single_si_config(**cfg_extra)   # same config, no CN term
    pv = np.array([5.0, 5.0, 5.0, 90.0, 90.0, 90.0, 0.5, 0.5, 0.5])

    # A constraint with no neighbours to count must contribute exactly
    # nothing -- not raise, and not be swallowed into compute_error's
    # 1e10 "invalid parameter vector" sentinel either. Compared against
    # the otherwise-identical constraint-free config rather than against
    # 0.0, since unrelated terms (e.g. charge balance on a deliberately
    # non-neutral single-cation cell) may legitimately be non-zero.
    assert compute_error(pv, cfg) == pytest.approx(
        compute_error(pv, baseline), abs=1e-12)


@pytest.mark.parametrize("cc_extra, cfg_extra", _EMPTY_EXPANSION_CASES)
def test_error_breakdown_survives_empty_neighbour_expansion(cc_extra, cfg_extra):
    cc = {"center_atom": "Si", "cn_min": 4, "cn_max": 4,
          "cutoff_distance": 2.6}
    cc.update(cc_extra)
    cfg = _single_si_config(coordination_constraints=[cc], **cfg_extra)
    pv = np.array([5.0, 5.0, 5.0, 90.0, 90.0, 90.0, 0.5, 0.5, 0.5])

    bd = error_breakdown(pv, cfg)
    assert bd["cn"][0]["cn_per_atom"] == []
    assert bd["cn"][0]["penalty"] == 0.0
    assert bd["total"] == pytest.approx(compute_error(pv, cfg), abs=1e-9)


@pytest.mark.parametrize("cc_extra, cfg_extra", _EMPTY_EXPANSION_CASES)
def test_gradient_lattice_hint_survives_empty_neighbour_expansion(
        cc_extra, cfg_extra):
    """init_guess._gradient_lattice_hint shares the same expansion
    pattern and crashed identically."""
    cc = {"center_atom": "Si", "cn_min": 4, "cn_max": 4,
          "cutoff_distance": 2.6}
    cc.update(cc_extra)
    cfg = _single_si_config(coordination_constraints=[cc], **cfg_extra)
    pv = np.array([5.0, 5.0, 5.0, 90.0, 90.0, 90.0, 0.5, 0.5, 0.5])

    grad = _gradient_lattice_hint(pv, cfg)
    assert np.all(np.isfinite(grad))
