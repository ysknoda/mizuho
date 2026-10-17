"""Standard atomic weights for density <-> volume conversions (ADR-0017, ADR-0030).

IUPAC CIAAW abridged standard atomic weights (2021), conventional values, in
unified atomic mass units (u = Da = g/mol). Extend the table when a fixture
with another element states a density; a missing element is an error, never
a guess.
"""

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

# 1 g/cm^3 in u/A^3: N_A u per gram over 1e24 A^3 per cm^3 (exact).
G_CM3_TO_U_A3 = AVOGADRO / 1e24
