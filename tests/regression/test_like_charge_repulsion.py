"""ADR-0037: derived like-charge minimum distances.

The Al-O-N composition-free run (ADR-0033) left zero-error classes with an
NO3-like N-O contact of 1.370 A (Al5O3N3 class 7, SG 157) and Al-Al pairs at
the hard-sphere limit of 1.00 A (Al6O3N4 class 241, SG 81): no derived term
held like-charge ions apart. Both CIFs are copied from runs/alon/classes.
The config is runs/alon/alon.yaml with the composition fixed."""

import pytest

from mizuho.cli import _reference_config
from mizuho.objective.breakdown import error_breakdown

REG = "tests/fixtures/regression"
REF = "tests/fixtures/reference"

ALON = {
    "mode": "auto_spacegroup",
    "elements": ["Al", "O", "N"],
    "ionic_charges": {"Al": 3, "O": -2, "N": -3},
    "charge_tolerance_min": 0, "charge_tolerance_max": 0,
    "coordination_numbers": {"Al": [4, 6]},
    "anion_coordination_numbers": {"O": [2, 4], "N": [3, 4]},
    "density_g_cm3": [3.26, 4.10],
}


def _score(cif, composition, **extra):
    ref, full = _reference_config(dict(ALON, composition=composition, **extra), cif)
    return error_breakdown(ref.param_vector, full)


def _rep(bd, pair):
    return sum(r["penalty"] for r in bd["repulsion"] if r.get("label", "").startswith(pair))


def test_nitrate_like_contact_is_rejected():
    cif = f"{REG}/alon_class_07_sg157.cif"
    assert _score(cif, "Al5O3N3", auto_repulsion=False)["total"] < 1e-6
    bd = _score(cif, "Al5O3N3")
    assert _rep(bd, "N-O") > 0.1
    assert bd["total"] > 0.1


def test_al_al_at_the_hard_sphere_limit_is_rejected():
    cif = f"{REG}/alon_class_241_sg81.cif"
    assert _score(cif, "Al6O3N4", auto_repulsion=False)["total"] < 1e-3
    bd = _score(cif, "Al6O3N4")
    assert _rep(bd, "Al-Al") > 0.05
    assert bd["total"] > 0.1


@pytest.mark.parametrize("cif, composition, expected", [
    (f"{REF}/corundum_H.cif", "Al2O3", 5.185e-4),   # ADR-0033: 5.2e-4
    (f"{REG}/aln_wurtzite.cif", "AlN", 4.497e-6),    # ADR-0033: 4.5e-6
])
def test_end_member_scores_are_unchanged(cif, composition, expected):
    """Corundum's face-sharing Al-Al (2.655 A) and shared-face O-O (2.526 A)
    are the tight cases; neither term acts on them."""
    bd = _score(cif, composition)
    assert all(r["penalty"] == 0.0 for r in bd["repulsion"])
    assert bd["total"] == pytest.approx(expected, rel=1e-3)
    assert bd["total"] == pytest.approx(_score(cif, composition, auto_repulsion=False)["total"],
                                        abs=1e-12)
