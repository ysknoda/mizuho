"""scripts/measure_extra_terms.py: the candidate terms against compute_error,
the end members, and the bond-valence bound behind the anion-CN conflict."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import measure_extra_terms as mt

from mizuho.constraints import BAND_VU, bond_band
from mizuho.data.bond_valence import B_BOND_VALENCE, r0
from mizuho.io.cif import cif_to_mizuho
from mizuho.objective.energy import compute_error
from mizuho.pipeline import build_sg_config
from mizuho.structure import build_structure

REPO = Path(__file__).resolve().parents[2]

# The Al-O-N run's constraints (runs/alon/alon.yaml, ADR-0033).
ALON = {
    "mode": "auto_spacegroup",
    "elements": ["Al", "O", "N"],
    "ionic_charges": {"Al": 3, "O": -2, "N": -3},
    "charge_tolerance_min": 0,
    "charge_tolerance_max": 0,
    "coordination_numbers": {"Al": [4, 6]},
    "anion_coordination_numbers": {"O": [2, 4], "N": [3, 4]},
    "density_g_cm3": [3.26, 4.10],
}


@pytest.fixture(scope="module")
def references(tmp_path_factory):
    cfg_path = tmp_path_factory.mktemp("alon") / "alon.yaml"
    import yaml

    cfg_path.write_text(yaml.safe_dump(ALON))
    return mt.reference_scores(str(cfg_path))


def test_end_members_score_zero_under_every_term(references):
    """Corundum and wurtzite AlN keep their ADR-0033 objective values and
    pay nothing under any candidate term at any threshold."""
    err_cor, d_cor = references["corundum"]
    err_aln, d_aln = references["wurtzite AlN"]
    assert err_cor == pytest.approx(5.2e-4, abs=5e-6)
    assert err_aln == pytest.approx(4.5e-6, abs=5e-7)
    for d in (d_cor, d_aln):
        assert set(d["terms"]) == set(mt.TERM_NAMES)
        assert all(v == 0.0 for v in d["terms"].values()), d["terms"]
    # Corundum's shortest O-O (shared face) is the margin of aa_2.5.
    assert d_cor["anion_contacts"]["O-O"] == pytest.approx(2.526, abs=1e-3)
    assert {x for x, cn, _ in d_cor["anions"]} == {"O"}
    assert all(round(cn) == 4 for _, cn, _ in d_cor["anions"])
    assert all(round(cn) == 4 for _, cn, _ in d_aln["anions"])


def _corundum(scale):
    """Corundum's config under the Al-O-N constraints, with its lattice
    scaled by `scale` (so anion contacts and angles leave their bands)."""
    ref = cif_to_mizuho(str(REPO / "tests/fixtures/reference/corundum_H.cif"))
    cfg = build_sg_config(ALON, ref.spacegroup, composition="Al2O3")
    cfg["wyckoff_sites"] = ref.config()["wyckoff_sites"]
    p = np.array(ref.param_vector, float)
    p[:2] *= scale  # a and c lead the hexagonal parameter vector
    return p, cfg


@pytest.mark.parametrize("scale", [0.85, 0.9, 1.0])
def test_config_expressible_terms_equal_compute_error(scale):
    """aa (repulsion_constraints), axa (ignore_trans_above 180) and xalx
    (one CN for every Al) equal compute_error's own increase."""
    p, cfg = _corundum(scale)
    st = build_structure(p, cfg)
    d = mt.describe(st, cfg)
    e0 = compute_error(p, cfg)
    variants = mt.config_variants(cfg, d)
    assert {"aa_2.3", "axa_160", "xalx", "xalx_t4"} <= set(variants)
    for term, cfg_t in variants.items():
        assert compute_error(p, cfg_t) - e0 == pytest.approx(
            d["terms"][term], abs=1e-12
        )
    if scale == 0.85:  # O-O down to 2.15 A
        assert d["terms"]["aa_2.3"] > 0


def test_bvs_penalty_is_the_cation_form():
    """The anion term uses the cation term's formula (bond_valence.py)."""
    p, cfg = _corundum(0.95)
    st = build_structure(p, cfg)
    from mizuho.objective.bond_valence import bond_valence_term

    bv = bond_valence_term(
        np.asarray(st["frac_coords"]),
        np.asarray(st["lattice_matrix"]),
        st["species"],
        cfg,
    )
    info = bv["cations"]["Al"]
    assert info["penalty"] > 0
    assert mt.bvs_penalty(
        info["bvs"], info["valence"], info["tolerance"]
    ) == pytest.approx(info["penalty"], rel=1e-12)


def test_two_coordinated_oxygen_cannot_reach_its_valence():
    """Inside the derived Al-O band (s0 = 3/5 at the CN [4, 6] range's
    midpoint, +/- BAND_VU) a bond carries at most s0 + BAND_VU = 0.76 v.u.,
    so O at CN 2 sums to at most 1.52 v.u., 24 % below 2; N at CN 3 likewise
    2.28 against 3."""
    band = bond_band("Al", 3, "O", 4, 6)
    s_max = math.exp((r0("Al", 3, "O") - band["distance_min"]) / B_BOND_VALENCE)
    assert s_max == pytest.approx(3 / 5 + BAND_VU, abs=2e-3)
    assert 1 - 2 * s_max / 2 == pytest.approx(0.24, abs=2e-3)
    band_n = bond_band("Al", 3, "N", 4, 6)
    s_max_n = math.exp((r0("Al", 3, "N") - band_n["distance_min"]) / B_BOND_VALENCE)
    assert 1 - 3 * s_max_n / 3 == pytest.approx(0.24, abs=2e-3)
