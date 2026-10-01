"""Bond-valence-sum term and the Brese & O'Keeffe (1991) table."""

import math

import numpy as np
import pytest

from mizuho.data.bond_valence import B_BOND_VALENCE, TABLE2, TABLE3, r0
from mizuho.objective.bond_valence import bond_valence_term
from mizuho.objective.breakdown import error_breakdown
from mizuho.objective.energy import compute_error
from mizuho.structure import (
    assignments_to_sites,
    build_structure,
    resolve_fixed_assignment,
)


def test_table_is_transcribed_in_full():
    assert len(TABLE2) * 3 == 327 and len(TABLE3) * 9 == 639
    assert sum(1 for e in TABLE2.values() for v in e.values() if v[1]) == 48  # italics
    # Spot checks against the paper, including the two OCR disagreements.
    assert r0("Pb", 2, "O") == 2.112 and r0("Lu", 3, "N") == 2.11
    assert r0("Al", 3, "O") == 1.651 and r0("Al", 3, "F") == 1.545
    assert r0("Zn", 2, "S") == 2.09 and r0("Ti", 4, "O") == 1.815
    assert r0("Ti", 3, "O") == 1.791  # oxidation state matters
    assert r0("Xx", 1, "O") is None and r0("Al", 2, "O") is None
    assert B_BOND_VALENCE == 0.37


def _nacl(a, charges=None, bv=None):
    asg = resolve_fixed_assignment(
        225, [{"element": "Na", "letter": "a"}, {"element": "Cl", "letter": "b"}]
    )
    cfg = {
        "spacegroup": 225,
        "wyckoff_sites": assignments_to_sites(asg),
        "ionic_charges": charges or {"Na": 1, "Cl": -1},
        "bond_constraints": [
            {
                "atom1": "Na",
                "atom2": "Cl",
                "distance_min": 2.0,
                "distance_max": 3.5,
                "cutoff": 3.85,
            }
        ],
    }
    if bv is not None:
        cfg["bond_valence"] = bv
    return cfg, np.array([a])


def test_bvs_matches_hand_calc_and_anions_are_report_only():
    """Rock salt, a = 5.0 A: six Cl at 2.5 A inside the 3.85 A cutoff (next
    shell 4.33 A, beyond the 3.85 +- 0.3 A band). Na(I)-Cl R0 = 2.15 A, so
    BVS = 6 exp((2.15 - 2.5)/0.37) = 2.330, 1.330 over V = 1; penalty
    w * ((1.330 - 0.1) / 1)^2. Cl's BVS is the same and is NOT added."""
    cfg, pv = _nacl(5.0)
    bvs = 6 * math.exp((2.15 - 2.5) / 0.37)
    expected = ((bvs - 1.0 - 0.1) / 1.0) ** 2
    st = build_structure(pv, cfg)
    term = bond_valence_term(
        st["frac_coords"], st["lattice_matrix"], st["species"], cfg
    )
    assert term["cations"]["Na"]["bvs"][0] == pytest.approx(bvs, rel=1e-9)
    assert term["anions"]["Cl"]["bvs"][0] == pytest.approx(bvs, rel=1e-9)
    assert term["penalty"] == pytest.approx(expected, rel=1e-9)
    bd = error_breakdown(pv, cfg)
    assert bd["total"] == pytest.approx(compute_error(pv, cfg), rel=1e-12)
    assert bd["bond_valence"]["penalty"] == pytest.approx(expected, rel=1e-9)


def test_tolerance_override_and_weight():
    cfg, pv = _nacl(5.0, bv={"tolerance": {"Na": 2.0}})
    assert compute_error(pv, cfg) == pytest.approx(0.0, abs=1e-12)
    cfg, pv = _nacl(5.0, bv={"weight": 3.0})
    cfg0, _ = _nacl(5.0)
    assert compute_error(pv, cfg) == pytest.approx(
        3 * compute_error(pv, cfg0), rel=1e-12
    )


def test_cation_without_r0_is_skipped_and_reported():
    cfg, pv = _nacl(5.0, charges={"Na": 3, "Cl": -3})  # no Na(III)-Cl in the paper
    st = build_structure(pv, cfg)
    term = bond_valence_term(
        st["frac_coords"], st["lattice_matrix"], st["species"], cfg
    )
    assert term["skipped"] == ["Na"] and term["penalty"] == 0.0


def test_sum_is_continuous_across_the_cutoff():
    """A neighbour crossing the bond cutoff enters through the smootherstep
    band, not as a jump."""
    vals = []
    for cut in np.linspace(3.30, 3.40, 41):  # Cl second shell at 0.866 a
        cfg, pv = _nacl(3.88)  # second shell 3.36 A
        cfg["bond_constraints"][0]["cutoff"] = cut
        st = build_structure(pv, cfg)
        vals.append(
            bond_valence_term(
                st["frac_coords"], st["lattice_matrix"], st["species"], cfg
            )["cations"]["Na"]["bvs"][0]
        )
    assert np.max(np.abs(np.diff(vals))) < 0.05


def test_gradient_matches_finite_differences_with_the_term_active():
    pytest.importorskip("jax")
    from scipy.optimize import approx_fprime

    from mizuho.objective.energy_jax import make_gradient_fn

    cfg, pv = _nacl(5.0)
    cfg["coordination_constraints"] = [
        {
            "center_atom": "Na",
            "neighbor_atom": "Cl",
            "cn_min": 6,
            "cn_max": 6,
            "cutoff_distance": 3.85,
        }
    ]
    g = make_gradient_fn(cfg, pv)(pv)
    g_fd = approx_fprime(pv, lambda x: compute_error(x, cfg), 1e-6)
    assert g == pytest.approx(g_fd, rel=1e-4, abs=1e-6)


def test_term_can_be_switched_off():
    cfg, pv = _nacl(5.0, bv=False)
    assert compute_error(pv, cfg) == pytest.approx(0.0, abs=1e-12)
