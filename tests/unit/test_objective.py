"""Unit tests for mizuho.objective."""

import math

import numpy as np
import pytest

from mizuho.objective.penalties import (
    _expand_wildcard, _midpoint_scale, _penalty, _SCALE_EPS,
    _soft_coordination_number,
)


def test_penalty_zero_inside_range():
    assert _penalty(5.0, 2.0, 8.0) == 0.0


def test_penalty_below_range():
    # v < vmin -> w * (vmin - v)^2
    assert _penalty(1.0, 2.0, 8.0, w=2.0) == pytest.approx(2.0 * (2.0 - 1.0) ** 2)


def test_penalty_above_range():
    # v > vmax -> w * (v - vmax)^2
    assert _penalty(10.0, 2.0, 8.0, w=2.0) == pytest.approx(2.0 * (10.0 - 8.0) ** 2)


def test_penalty_continuous_at_boundaries():
    # No jump discontinuity exactly at vmin/vmax.
    eps = 1e-9
    assert _penalty(2.0 - eps, 2.0, 8.0) == pytest.approx(0.0, abs=1e-12)
    assert _penalty(8.0 + eps, 2.0, 8.0) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# _soft_coordination_number: replaces a hard sum(d < cutoff) neighbour count
# with a logistic-weighted one so the coordination-number penalty (and its
# gradient, as seen by CMA-ES/L-BFGS-B) varies continuously with atomic
# positions instead of jumping only at the exact instant a neighbour crosses
# the cutoff. See mizuho.objective.penalties._soft_coordination_number.
# ---------------------------------------------------------------------------

def test_soft_cn_matches_hard_count_far_from_cutoff():
    # Neighbours well inside/outside the cutoff should sum to ~the same
    # integer count a hard threshold would give.
    distances = np.array([1.0, 1.2, 1.4, 5.0, 6.0])
    cn = _soft_coordination_number(distances, cutoff=2.0, width=0.10)
    assert cn == pytest.approx(3.0, abs=0.01)


def test_soft_cn_is_one_half_exactly_at_cutoff():
    distances = np.array([2.0])
    cn = _soft_coordination_number(distances, cutoff=2.0, width=0.10)
    assert cn == pytest.approx(0.5)


def test_soft_cn_varies_continuously_across_the_cutoff():
    """The whole point of the soft count: unlike a hard threshold, moving
    a neighbour by a tiny amount right at the cutoff must change the
    count (and hence the penalty gradient) by a correspondingly tiny,
    non-zero amount -- not by a full discrete step of 1, and not by
    exactly 0."""
    cutoff, width = 2.0, 0.10
    d_at        = _soft_coordination_number(np.array([2.000]), cutoff, width)
    d_just_in   = _soft_coordination_number(np.array([1.999]), cutoff, width)
    d_just_out  = _soft_coordination_number(np.array([2.001]), cutoff, width)

    assert 0.0 < d_just_out < d_at < d_just_in < 1.0
    assert abs(d_just_in - d_at) < 0.01
    assert abs(d_at - d_just_out) < 0.01


def test_soft_cn_ignores_non_finite_distances():
    distances = np.array([1.0, np.inf])
    cn = _soft_coordination_number(distances, cutoff=2.0, width=0.10)
    assert cn == pytest.approx(1.0, abs=1e-3)


def test_soft_cn_all_non_finite_is_zero():
    assert _soft_coordination_number(np.array([np.inf, np.inf]), cutoff=2.0) == 0.0


# ---------------------------------------------------------------------------
# _expand_wildcard: charge-sign-based "X" resolution
# ---------------------------------------------------------------------------

_ZAOH_SPECIES = {"Zn", "As", "O", "H"}
_ZAOH_CHARGES = {"Zn": 2, "As": 5, "O": -2, "H": 1}


def test_expand_wildcard_anion_center_gives_cations_only():
    # O (charge -2) is the only anion present -- "X" must resolve to
    # every CATION present (Zn, As, H), and never O itself, even though
    # other O atoms obviously coexist in the real structure.
    result = _expand_wildcard("X", _ZAOH_SPECIES, "O", _ZAOH_CHARGES)
    assert result == ["As", "H", "Zn"]


def test_expand_wildcard_cation_center_gives_anions_only():
    result = _expand_wildcard("X", _ZAOH_SPECIES, "Zn", _ZAOH_CHARGES)
    assert result == ["O"]


def test_expand_wildcard_zero_charge_center_falls_back_to_all_but_self():
    species = {"Si", "C"}
    charges = {"Si": 0, "C": 0}
    assert _expand_wildcard("X", species, "Si", charges) == ["C"]


def test_expand_wildcard_missing_ionic_charges_falls_back_to_all_but_self():
    species = {"Si", "C", "N"}
    assert sorted(_expand_wildcard("X", species, "Si", {})) == ["C", "N"]


def test_expand_wildcard_list_input_filtered_to_present_species():
    result = _expand_wildcard(["Zn", "As", "Mg"], _ZAOH_SPECIES, "O", _ZAOH_CHARGES)
    assert result == ["Zn", "As"]   # "Mg" dropped: not present; order preserved


def test_expand_wildcard_plain_string_unchanged():
    assert _expand_wildcard("O", _ZAOH_SPECIES, "Zn", _ZAOH_CHARGES) == ["O"]


# ---------------------------------------------------------------------------
# angle_constraints: exclude_neighbor_pairs
# ---------------------------------------------------------------------------

def _o_zn_h_h_param_vector():
    """SG1 (P1): every site is the fully-general 1a position (x,y,z all
    free, multiplicity 1, identity-only symmetry) -- lets a structure be
    placed by hand with exact, known bond angles, with no PBC image
    interference thanks to the oversized cell.

    O is the center; Zn/H1/H2 are its 3 neighbours, positioned so that
    Zn-O-H1 = Zn-O-H2 = 90 deg (inside [80,100]) while H1-O-H2 = 120 deg
    (outside [80,100], and well under the 150 deg ignore_trans_above
    default, so it is not filtered out by that unrelated mechanism).
    """
    from mizuho.structure import assignments_to_sites, resolve_fixed_assignment

    specs = [{"element": "O", "letter": "a"}, {"element": "Zn", "letter": "a"},
              {"element": "H", "letter": "a"}, {"element": "H", "letter": "a"}]
    sites = assignments_to_sites(resolve_fixed_assignment(1, specs))

    L = 20.0  # cell edge -- neighbours sit ~1-1.5 A from the centre, so
              # no periodic image ever competes with the intended one.
    c = 0.5
    o  = [c, c, c]
    zn = [c, c, c + 1.5 / L]                              # +z, dist 1.5
    h1 = [c + 1.0 / L, c, c]                               # +x, dist 1.0
    h2 = [c - 0.5 / L, c + 0.8660254 / L, c]               # 120 deg from h1, dist 1.0
    lattice = [L, L, L, 90.0, 90.0, 90.0]
    param_vector = np.array(lattice + o + zn + h1 + h2)
    return sites, param_vector


def test_angle_exclude_neighbor_pairs_drops_only_the_excluded_pair():
    from mizuho.objective.energy import compute_error

    sites, param_vector = _o_zn_h_h_param_vector()
    base_ac = {"atom_center": "O", "atom_neighbor": ["Zn", "H"],
               "angle_min": 80.0, "angle_max": 100.0,
               "angle_cutoff": 2.0, "weight": 1.0}

    config_no_exclude = {"spacegroup": 1, "wyckoff_sites": sites,
                          "angle_constraints": [dict(base_ac)]}
    config_excluded = {"spacegroup": 1, "wyckoff_sites": sites,
                        "angle_constraints": [
                            dict(base_ac, exclude_neighbor_pairs=[["H", "H"]])]}

    err_no_exclude = compute_error(param_vector, config_no_exclude)
    err_excluded   = compute_error(param_vector, config_excluded)

    # Zn-O-H1 = Zn-O-H2 = 90 deg satisfy [80,100] (0 penalty each);
    # H1-O-H2 = 120 deg violates it by 20 deg -> penalty =
    # weight*(20/90)^2 (band-midpoint scale, ADR-0009), divided by
    # N(center)=1 -> exactly that when not excluded, exactly 0 once the
    # H-H pair is dropped.
    assert err_no_exclude == pytest.approx((20.0 / 90.0) ** 2, abs=1e-6)
    assert err_excluded == pytest.approx(0.0, abs=1e-9)


def test_angle_exclude_neighbor_pairs_is_noop_when_absent():
    from mizuho.objective.energy import compute_error

    sites, param_vector = _o_zn_h_h_param_vector()
    ac = {"atom_center": "O", "atom_neighbor": ["Zn", "H"],
          "angle_min": 80.0, "angle_max": 100.0,
          "angle_cutoff": 2.0, "weight": 1.0}
    config = {"spacegroup": 1, "wyckoff_sites": sites, "angle_constraints": [ac]}

    assert compute_error(param_vector, config) == pytest.approx((20.0 / 90.0) ** 2, abs=1e-6)


# ---------------------------------------------------------------------------
# _penalty: scale parameter (phase I-1, relative-error normalisation)
# ---------------------------------------------------------------------------

def test_penalty_scale_default_reproduces_raw_behaviour():
    assert _penalty(1.0, 2.0, 8.0, w=2.0) == _penalty(1.0, 2.0, 8.0, w=2.0, scale=1.0)


def test_penalty_scale_shrinks_penalty_quadratically():
    # scale=2 halves the deviation before squaring -> quarters the penalty.
    raw    = _penalty(1.0, 2.0, 8.0, w=1.0)
    scaled = _penalty(1.0, 2.0, 8.0, w=1.0, scale=2.0)
    assert raw == pytest.approx(1.0)
    assert scaled == pytest.approx(0.25)


def test_penalty_scale_preserves_c1_continuity_at_boundary():
    eps = 1e-9
    assert _penalty(2.0 - eps, 2.0, 8.0, scale=3.0) == pytest.approx(0.0, abs=1e-12)
    assert _penalty(8.0 + eps, 2.0, 8.0, scale=3.0) == pytest.approx(0.0, abs=1e-12)


def test_penalty_matches_old_repulsion_inline_formula():
    """energy.py's repulsion term used to compute w*(dmin-d)**2 (d<dmin
    else 0.0) inline rather than via _penalty. The phase I-1 refactor to
    _penalty(d, dmin, np.inf, w) must be exactly equivalent for every d,
    including the boundary, before the behavioural change is trusted."""
    dmin, w = 2.0, 1.5
    for d in [0.5, 1.0, 1.9999, 2.0, 2.0001, 3.0, 100.0]:
        old = w * (dmin - d) ** 2 if d < dmin else 0.0
        new = _penalty(d, dmin, float("inf"), w)
        assert new == pytest.approx(old, abs=1e-12)


# ---------------------------------------------------------------------------
# _midpoint_scale
# ---------------------------------------------------------------------------

def test_midpoint_scale_typical_band():
    assert _midpoint_scale(1.6, 1.75) == pytest.approx(1.675)


def test_midpoint_scale_zero_width_band_does_not_divide_by_zero():
    # coordination_constraints are frequently cn_min == cn_max (a
    # zero-width band) in every real fixture -- must reduce to the
    # shared value itself, not blow up.
    assert _midpoint_scale(4.0, 4.0) == pytest.approx(4.0)


def test_midpoint_scale_eps_floor_when_midpoint_is_zero():
    assert _midpoint_scale(0.0, 0.0) == pytest.approx(_SCALE_EPS)
    assert _midpoint_scale(-1.0, 1.0) == pytest.approx(_SCALE_EPS)


# ---------------------------------------------------------------------------
# Relative-error normalisation (always on, ADR-0009), end-to-end through
# compute_error, verified against hand-computed expected values. Each
# helper builds a minimal SG1 (P1) synthetic structure so the expected
# penalty is exactly known, mirroring _o_zn_h_h_param_vector's pattern
# above.
# ---------------------------------------------------------------------------

def _two_atom_case(elem1, elem2, d):
    """SG1: two atoms elem1, elem2 exactly d Angstrom apart along +x, in
    an oversized cell (no periodic image competes with the intended
    pair)."""
    from mizuho.structure import assignments_to_sites, resolve_fixed_assignment

    specs = [{"element": elem1, "letter": "a"}, {"element": elem2, "letter": "a"}]
    sites = assignments_to_sites(resolve_fixed_assignment(1, specs))
    L = 20.0
    c = 0.5
    p1 = [c, c, c]
    p2 = [c + d / L, c, c]
    lattice = [L, L, L, 90.0, 90.0, 90.0]
    return sites, np.array(lattice + p1 + p2)


def test_bond_normalised_relative_error_matches_hand_calc():
    from mizuho.objective.energy import compute_error

    sites, pv = _two_atom_case("Zn", "O", d=2.0)
    base_bc = {"atom1": "Zn", "atom2": "O", "distance_min": 1.0,
               "distance_max": 1.5, "cutoff": 5.0, "weight": 2.0}
    cfg = {"spacegroup": 1, "wyckoff_sites": sites, "bond_constraints": [dict(base_bc)]}

    scale = (1.0 + 1.5) / 2.0   # 1.25
    expected = 2.0 * ((2.0 - 1.5) / scale) ** 2
    assert compute_error(pv, cfg) == pytest.approx(expected, abs=1e-9)


def test_angle_normalised_relative_error_matches_hand_calc():
    from mizuho.objective.energy import compute_error

    sites, param_vector = _o_zn_h_h_param_vector()
    ac = {"atom_center": "O", "atom_neighbor": ["Zn", "H"],
          "angle_min": 80.0, "angle_max": 100.0,
          "angle_cutoff": 2.0, "weight": 1.0}
    cfg = {"spacegroup": 1, "wyckoff_sites": sites, "angle_constraints": [dict(ac)]}

    scale = (80.0 + 100.0) / 2.0   # 90.0
    expected = ((120.0 - 100.0) / scale) ** 2
    assert compute_error(param_vector, cfg) == pytest.approx(expected, abs=1e-3)


def _center_with_neighbors_case(n_neighbors, d):
    """SG1: one centre atom C plus n_neighbors 'Nb' neighbours, each
    exactly d Angstrom from the centre along a distinct axis direction
    (only distances matter for a coordination-number test, not angles)."""
    from mizuho.structure import assignments_to_sites, resolve_fixed_assignment

    specs = ([{"element": "C", "letter": "a"}]
             + [{"element": "Nb", "letter": "a"} for _ in range(n_neighbors)])
    sites = assignments_to_sites(resolve_fixed_assignment(1, specs))
    L = 20.0
    c = 0.5
    coords = [c, c, c]
    directions = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1)]
    for dx, dy, dz in directions[:n_neighbors]:
        coords += [c + dx * d / L, c + dy * d / L, c + dz * d / L]
    lattice = [L, L, L, 90.0, 90.0, 90.0]
    return sites, np.array(lattice + coords)


def test_cn_normalised_relative_error_matches_hand_calc():
    from mizuho.objective.energy import compute_error

    sites, pv = _center_with_neighbors_case(3, d=1.5)
    cc = {"center_atom": "C", "neighbor_atom": "Nb", "cn_min": 2, "cn_max": 2,
          "cutoff_distance": 2.0, "weight": 1.5}
    cfg = {"spacegroup": 1, "wyckoff_sites": sites, "coordination_constraints": [dict(cc)]}

    cn_i = _soft_coordination_number(np.array([1.5, 1.5, 1.5]), cutoff=2.0, width=0.10)

    scale = 2.0   # _midpoint_scale(2, 2) == 2.0 (zero-width band)
    expected = 1.5 * ((cn_i - 2.0) / scale) ** 2
    assert compute_error(pv, cfg) == pytest.approx(expected, abs=1e-6)


def test_repulsion_normalised_relative_error_matches_hand_calc():
    from mizuho.objective.energy import compute_error

    sites, pv = _two_atom_case("H", "Li", d=1.2)
    rc = {"atom1": "H", "atom2": "Li", "distance_min": 1.5, "weight": 2.0}
    cfg = {"spacegroup": 1, "wyckoff_sites": sites, "repulsion_constraints": [dict(rc)]}

    scale = max(1.5, _SCALE_EPS)   # one-sided: dmin itself
    expected = 2.0 * ((1.5 - 1.2) / scale) ** 2
    assert compute_error(pv, cfg) == pytest.approx(expected, abs=1e-9)


def _single_atom_case(elem, a=10.0):
    from mizuho.structure import assignments_to_sites, resolve_fixed_assignment

    sites = assignments_to_sites(resolve_fixed_assignment(1, [{"element": elem, "letter": "a"}]))
    return sites, np.array([a, a, a, 90.0, 90.0, 90.0, 0.5, 0.5, 0.5])


def test_charge_normalised_relative_error_matches_hand_calc():
    from mizuho.objective.energy import compute_error

    sites, pv = _single_atom_case("Na")
    cfg = {"spacegroup": 1, "wyckoff_sites": sites,
           "ionic_charges": {"Na": 3.0}, "charge_balance_weight": 2.0}

    charge_scale = 3.0   # sum(abs(charges)) over the single Na atom
    expected = 2.0 * (3.0 / charge_scale) ** 2
    assert compute_error(pv, cfg) == pytest.approx(expected, abs=1e-9)


def test_volume_normalised_relative_error_matches_hand_calc():
    from mizuho.objective.energy import compute_error

    sites, pv = _single_atom_case("Na", a=10.0)   # V_cell = 10^3 = 1000 A^3
    cfg = {"spacegroup": 1, "wyckoff_sites": sites,
           "volume_constraint": {"volume_min": 200.0, "volume_max": 800.0, "weight": 3.0}}

    scale = (200.0 + 800.0) / 2.0   # 500.0
    expected = 3.0 * ((1000.0 - 800.0) / scale) ** 2
    assert compute_error(pv, cfg) == pytest.approx(expected, abs=1e-3)


# ---------------------------------------------------------------------------
# aspect_ratio_constraint (phase I-3): complements volume_constraint, which
# cannot detect a degenerate flat/needle cell whose TOTAL volume happens to
# be reasonable -- penalises max(a,b,c)/min(a,b,c) directly.
# ---------------------------------------------------------------------------

def test_aspect_ratio_zero_penalty_when_isotropic():
    from mizuho.objective.energy import compute_error

    sites, _ = _single_atom_case("Na", a=10.0)
    pv = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0, 0.5, 0.5, 0.5])
    cfg = {"spacegroup": 1, "wyckoff_sites": sites,
           "aspect_ratio_constraint": {"ratio_max": 3.0, "weight": 5.0}}
    assert compute_error(pv, cfg) == pytest.approx(0.0, abs=1e-9)


def test_aspect_ratio_zero_penalty_within_band():
    from mizuho.objective.energy import compute_error

    sites, _ = _single_atom_case("Na", a=10.0)
    # ratio = 20/10 = 2.0, within ratio_max=3.0 -> no penalty.
    pv = np.array([10.0, 10.0, 20.0, 90.0, 90.0, 90.0, 0.5, 0.5, 0.5])
    cfg = {"spacegroup": 1, "wyckoff_sites": sites,
           "aspect_ratio_constraint": {"ratio_max": 3.0, "weight": 5.0}}
    assert compute_error(pv, cfg) == pytest.approx(0.0, abs=1e-9)


def test_aspect_ratio_penalises_excessive_anisotropy_matches_hand_calc():
    from mizuho.objective.energy import compute_error

    sites, _ = _single_atom_case("Na", a=10.0)
    # ratio = 40/10 = 4.0, exceeds ratio_max=3.0 by 1.0
    pv = np.array([10.0, 10.0, 40.0, 90.0, 90.0, 90.0, 0.5, 0.5, 0.5])
    cfg = {"spacegroup": 1, "wyckoff_sites": sites,
           "aspect_ratio_constraint": {"ratio_max": 3.0, "weight": 5.0}}

    scale = (1.0 + 3.0) / 2.0   # _midpoint_scale(1.0, 3.0) == 2.0
    expected = 5.0 * ((4.0 - 3.0) / scale) ** 2
    assert compute_error(pv, cfg) == pytest.approx(expected, abs=1e-9)


def test_aspect_ratio_absent_by_default():
    from mizuho.objective.energy import compute_error

    sites, _ = _single_atom_case("Na", a=10.0)
    # Extreme anisotropy (ratio 25), but no aspect_ratio_constraint
    # configured -> no effect. a and b are kept well above Na's own
    # hard-sphere contact distance (2 x 0.95 = 1.90 A): since ADR-0001 the
    # overlap term sees an atom's own periodic images, so the former
    # a = b = 1.0 A cell is a genuine self-overlap and no longer isolates
    # the aspect-ratio term.
    pv = np.array([4.0, 4.0, 100.0, 90.0, 90.0, 90.0, 0.5, 0.5, 0.5])
    cfg = {"spacegroup": 1, "wyckoff_sites": sites}
    assert compute_error(pv, cfg) == pytest.approx(0.0, abs=1e-9)


def test_aspect_ratio_trivially_satisfied_for_cubic_system():
    """Cubic systems have a==b==c by construction (build_structure only
    exposes a single free lattice parameter) -- ratio is always exactly
    1.0, so the term is a permanent no-op there. No crystal-system
    dispatch is needed in the implementation; this just confirms it."""
    from mizuho.objective.energy import compute_error
    from mizuho.structure import assignments_to_sites, resolve_fixed_assignment

    sites = assignments_to_sites(resolve_fixed_assignment(
        225, [{"element": "Na", "letter": "a"}, {"element": "Cl", "letter": "b"}]))
    pv = np.array([5.5])
    cfg = {"spacegroup": 225, "wyckoff_sites": sites,
           "aspect_ratio_constraint": {"ratio_max": 1.01, "weight": 1000.0}}
    assert compute_error(pv, cfg) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# cell_validity_constraint (phase I-3): only meaningful for monoclinic/
# triclinic systems, where alpha/beta/gamma are free parameters. Penalises
# the standard crystallographic non-degeneracy factor
# G = 1 - cos(a)^2 - cos(b)^2 - cos(g)^2 + 2 cos(a)cos(b)cos(g), which is
# mathematically bounded to (0, 1], with G==1 exactly at orthogonal and
# G -> 0 at the degenerate (zero-volume) boundary.
# ---------------------------------------------------------------------------

def _triclinic_atom_case(a, b, c, alpha, beta, gamma):
    from mizuho.structure import assignments_to_sites, resolve_fixed_assignment

    sites = assignments_to_sites(resolve_fixed_assignment(1, [{"element": "Na", "letter": "a"}]))
    return sites, np.array([a, b, c, alpha, beta, gamma, 0.5, 0.5, 0.5])


def test_cell_validity_zero_penalty_when_orthogonal():
    from mizuho.objective.energy import compute_error

    sites, pv = _triclinic_atom_case(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
    cfg = {"spacegroup": 1, "wyckoff_sites": sites,
           "cell_validity_constraint": {"g_min": 0.3, "weight": 5.0}}
    assert compute_error(pv, cfg) == pytest.approx(0.0, abs=1e-9)


def test_cell_validity_zero_penalty_for_moderate_monoclinic_angle():
    from mizuho.objective.energy import compute_error

    # beta=100 deg alone (a common monoclinic-like distortion) keeps G
    # well above any reasonable g_min -- must not be penalised.
    sites, pv = _triclinic_atom_case(10.0, 10.0, 10.0, 90.0, 100.0, 90.0)
    cfg = {"spacegroup": 1, "wyckoff_sites": sites,
           "cell_validity_constraint": {"g_min": 0.3, "weight": 5.0}}
    assert compute_error(pv, cfg) == pytest.approx(0.0, abs=1e-9)


def test_cell_validity_penalises_near_degenerate_angle_matches_hand_calc():
    from mizuho.objective.energy import compute_error

    # alpha=30 deg (lattice vectors b, c strongly non-orthogonal) drives
    # G to 0.25, below g_min=0.3. Chosen rather than a yet more
    # degenerate angle because since ADR-0001 the overlap term sees an
    # atom's own periodic images: squashing the cell further (the former
    # alpha=5 deg) brings Na inside its own 1.90 A contact distance and
    # the total error would no longer isolate the cell-validity term.
    sites, pv = _triclinic_atom_case(10.0, 10.0, 10.0, 30.0, 90.0, 90.0)
    cfg = {"spacegroup": 1, "wyckoff_sites": sites,
           "cell_validity_constraint": {"g_min": 0.3, "weight": 5.0}}

    ca, cb, cg = math.cos(math.radians(30.0)), math.cos(math.radians(90.0)), math.cos(math.radians(90.0))
    G = 1.0 - ca**2 - cb**2 - cg**2 + 2.0 * ca * cb * cg

    scale = (0.3 + 1.0) / 2.0   # _midpoint_scale(0.3, 1.0)
    expected = 5.0 * ((0.3 - G) / scale) ** 2
    assert compute_error(pv, cfg) == pytest.approx(expected, abs=1e-9)


def test_cell_validity_absent_by_default():
    from mizuho.objective.energy import compute_error

    # Extremely degenerate angles, but no cell_validity_constraint
    # configured -> no effect from this term (build_lattice_matrix's own
    # silent floor-to-degenerate still applies; just not penalised here).
    sites, pv = _triclinic_atom_case(10.0, 10.0, 10.0, 30.0, 90.0, 90.0)
    cfg = {"spacegroup": 1, "wyckoff_sites": sites}
    assert compute_error(pv, cfg) == pytest.approx(0.0, abs=1e-9)


def test_cell_validity_trivially_satisfied_for_cubic_system():
    """Cubic (and orthorhombic/tetragonal/hexagonal/trigonal) systems
    have alpha=beta=gamma=90 (or 90/90/120) by construction -- G is
    always exactly 1.0, so this term is a permanent no-op there. No
    crystal-system dispatch is needed in the implementation; this just
    confirms it."""
    from mizuho.objective.energy import compute_error
    from mizuho.structure import assignments_to_sites, resolve_fixed_assignment

    sites = assignments_to_sites(resolve_fixed_assignment(
        225, [{"element": "Na", "letter": "a"}, {"element": "Cl", "letter": "b"}]))
    pv = np.array([5.5])
    cfg = {"spacegroup": 225, "wyckoff_sites": sites,
           "cell_validity_constraint": {"g_min": 0.999, "weight": 1000.0}}
    assert compute_error(pv, cfg) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("entry", ["compute_error", "error_breakdown"])
def test_removed_normalize_penalties_key_is_rejected(entry):
    """ADR-0009: the key is refused, not silently ignored, so nobody can
    believe `false` still selects the raw-unit objective."""
    from mizuho.objective.breakdown import error_breakdown
    from mizuho.objective.energy import compute_error

    fn = {"compute_error": compute_error, "error_breakdown": error_breakdown}[entry]
    sites, pv = _single_atom_case("Na")
    for value in (True, False):
        cfg = {"spacegroup": 1, "wyckoff_sites": sites, "normalize_penalties": value}
        with pytest.raises(ValueError, match="0009"):
            fn(pv, cfg)


# ---------------------------------------------------------------------------
# Integrality (ADR-0007, uniqueness plan phase D)
# ---------------------------------------------------------------------------

def _cn_config(center_charge, d, n=3, cn=(2, 3)):
    sites, pv = _center_with_neighbors_case(n, d=d)
    cfg = {"spacegroup": 1, "wyckoff_sites": sites,
           "coordination_constraints": [
               {"center_atom": "C", "neighbor_atom": "Nb", "cn_min": cn[0],
                "cn_max": cn[1], "cutoff_distance": 2.0, "weight": 1.5}]}
    if center_charge is not None:
        # Only the sign matters here; the charge term is switched off.
        cfg["ionic_charges"] = {"C": center_charge, "Nb": -center_charge}
        cfg["charge_balance_weight"] = 0.0
    return cfg, pv


def test_integrality_penalises_a_half_integer_soft_count_inside_the_band():
    """Three neighbours at 1.95 A (cutoff 2.0, width 0.1): each counts
    expit(0.5) = 0.6225, soft CN 1.8674. Inside [1, 3] so the band term is
    zero; 0.1326 from an integer, so integrality charges
    w * ((0.1326 - 0.05) / scale)^2 with the parent's scale."""
    from scipy.special import expit
    from mizuho.objective.energy import compute_error
    cfg, pv = _cn_config(+2, d=1.95, cn=(1, 3))
    cn = 3 * expit(0.5)
    dev = abs(cn - round(cn))
    expected = 1.5 * ((dev - 0.05) / 2.0) ** 2        # scale = midpoint(1, 3) = 2
    assert compute_error(pv, cfg) == pytest.approx(expected, rel=1e-9)


def test_integrality_exempts_anion_centres():
    from mizuho.objective.energy import compute_error
    cfg, pv = _cn_config(-2, d=1.95, cn=(1, 3))
    assert compute_error(pv, cfg) == pytest.approx(0.0, abs=1e-12)


def test_integrality_is_silent_within_the_dead_band():
    from mizuho.objective.energy import compute_error
    cfg, pv = _cn_config(+2, d=1.5, cn=(1, 3))        # soft CN 2.98, dev 0.02
    assert compute_error(pv, cfg) == pytest.approx(0.0, abs=1e-12)


def test_breakdown_reports_the_integrality_share():
    from mizuho.objective.breakdown import error_breakdown
    from mizuho.objective.energy import compute_error
    cfg, pv = _cn_config(+2, d=1.95, cn=(1, 3))
    entry = error_breakdown(pv, cfg)["cn"][0]
    assert entry["integrality_penalty"] == pytest.approx(compute_error(pv, cfg), rel=1e-9)
    assert entry["integrality_max_dev"] == pytest.approx(0.1326, abs=1e-4)
