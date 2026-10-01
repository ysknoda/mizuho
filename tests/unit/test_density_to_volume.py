"""scripts/density_to_volume.py: V = Z M / (rho N_A)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from density_to_volume import molar_mass, volume_band  # noqa: E402


def test_molar_mass():
    assert molar_mass("Al2O3") == pytest.approx(2 * 26.982 + 3 * 15.999)
    assert molar_mass("Zn2AsO5H") == pytest.approx(2 * 65.38 + 74.922 + 5 * 15.999 + 1.008)


@pytest.mark.parametrize("formula,z,d_calc,v_reference", [
    # Handbook of Mineralogy D(calc.) is computed from a unit cell, so it must
    # map back onto that cell: an independent check of the formula and the
    # atomic weights (reference volumes from tests/fixtures/reference/).
    ("Al2O3", 6, 3.997, 255.03),       # corundum
    ("FeS2", 4, 5.013, 159.03),        # pyrite
    ("CaF2", 4, 3.180, 162.95),        # fluorite
    ("ZnO", 2, 5.6730, 47.68),         # zincite
])
def test_calculated_density_reproduces_the_reference_cell(formula, z, d_calc, v_reference):
    lo, hi = volume_band(formula, z, d_calc, d_calc)
    assert lo == pytest.approx(v_reference, rel=0.005)


def test_single_value_gets_two_percent_and_range_inverts():
    lo, hi = volume_band("SiO2", 3, 2.65)
    mid = volume_band("SiO2", 3, 2.65, 2.65)[0]
    assert lo == pytest.approx(mid / 1.02) and hi == pytest.approx(mid / 0.98)
    lo, hi = volume_band("TiO2", 4, 3.79, 3.97)
    assert lo < hi          # the denser end gives the smaller volume
