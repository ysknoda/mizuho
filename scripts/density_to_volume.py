#!/usr/bin/env python3
"""Convert a measured mineral density into a unit-cell volume band.

    V = Z * M / (rho * N_A)

Z is the formula-unit count of the fixture's cell, M the molar mass of its
composition, rho in g/cm^3 and V in A^3 (1 cm^3 = 1e24 A^3). A density range
(rho_min, rho_max) maps to (V_min, V_max) = (Z M/(rho_max N_A), Z M/(rho_min N_A));
a single value without a stated range gets +-2% (design review 2026-10-01, Q6).

Densities are D(meas.) of the Handbook of Mineralogy (Anthony, Bideaux, Bladh &
Nichols, eds., Mineralogical Society of America; PDFs in references/). D(calc.)
is never used: it is computed from a cell and would bring the reference back in.

    python scripts/density_to_volume.py Al2O3 6 3.98 4.10
"""

import argparse
import re

# IUPAC CIAAW abridged standard atomic weights (2021), conventional values.
ATOMIC_WEIGHT = {
    "H": 1.008,
    "O": 15.999,
    "F": 18.998,
    "Mg": 24.305,
    "Al": 26.982,
    "Si": 28.085,
    "S": 32.06,
    "Ca": 40.078,
    "Ti": 47.867,
    "Fe": 55.845,
    "Zn": 65.38,
    "As": 74.922,
}
AVOGADRO = 6.02214076e23  # mol^-1, exact (SI 2019)
SINGLE_VALUE_MARGIN = 0.02


def molar_mass(formula: str) -> float:
    total = 0.0
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        total += ATOMIC_WEIGHT[el] * (int(n) if n else 1)
    return total


def volume_band(formula: str, z: int, rho_min: float, rho_max: float = None):
    """(V_min, V_max) in A^3; a single density gets +-2%."""
    if rho_max is None:
        rho_min, rho_max = (
            rho_min * (1 - SINGLE_VALUE_MARGIN),
            rho_min * (1 + SINGLE_VALUE_MARGIN),
        )
    v = lambda rho: z * molar_mass(formula) / (rho * AVOGADRO) * 1e24
    return v(rho_max), v(rho_min)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("formula")
    ap.add_argument("z", type=int)
    ap.add_argument(
        "rho", type=float, nargs="+", help="density, or min and max (g/cm^3)"
    )
    a = ap.parse_args()
    lo, hi = volume_band(a.formula, a.z, *a.rho)
    print(
        f"{a.formula} Z={a.z}: M={molar_mass(a.formula):.3f} g/mol  V in [{lo:.2f}, {hi:.2f}] A^3"
    )


if __name__ == "__main__":
    main()
