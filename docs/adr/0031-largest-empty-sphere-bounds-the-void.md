# The largest empty sphere bounds a structure's voids

TiO2's exhaustive run left layered structures with open gaps (ADR-0026, ADR-0030). Design review
of 2026-10-17: a void term, which can also be asked for open structures on purpose.

## What to measure

- **Total void volume is not new information.** Cell volume minus the atoms' spheres is, for a
  given composition, a function of the volume per formula unit, which a density already bounds
  (ADR-0030).
- **R_hole, the largest empty sphere**, is: the largest distance from a point of the cell to the
  nearest atom centre, no radii (Zeo++'s Di without radii). A layer gap or channel raises it.
  A probe-accessible volume was rejected: it needs a probe radius nobody can cite.
- **Absolute A, not relative to bond length.** Eleven references (one refinement per mineral,
  adamite with H): R_hole 1.85-2.61 A, max/min 1.41; divided by the longest shortest bond,
  0.82-1.38, max/min 1.69.

| reference | R_hole (A) | reference | R_hole (A) |
| --- | --- | --- | --- |
| pyrite | 1.85 | wurtzite | 2.12 |
| corundum | 1.92 | quartz | 2.23 |
| anatase | 1.96 | wollastonite | 2.30 |
| topaz | 1.96 | sphalerite | 2.34 |
| forsterite | 2.11 | fluorite | 2.36 |
| adamite (with H) | 2.61 | rutile (Handbook cell) | 1.82 |

`adamite.cif` without H leaves H's site empty: 2.94.

## Decision

- **`void: {r_hole_min, r_hole_max, weight}`**, A, a band like the other constraints: a small
  `r_hole_max` asks for a dense structure, a large `r_hole_min` for a layered or porous one. No
  named presets.
- **Default r_hole_max = 2.65 A, only when nothing else bounds the volume**: no density and no
  `volume_constraint`. With either, the volume term already does this and the default would
  charge twice. `void: false` switches it off. 2.65 is the largest reference plus 1.5 %, chosen
  from data, not cited; validated on oxides, fluorides and two sulfides only, unknown for larger
  anions (Cl, Br, I).
- **Computation** (`objective/cell_terms.py`): a 0.5 A grid's largest nearest-atom distance is a
  lower bound, and that plus half the grid cell's longest diagonal an upper bound. When the
  bounds put R_hole inside the band, the penalty is exactly zero and nothing else is computed.
  Otherwise R_hole is exact: the largest empty sphere is centred on a Voronoi vertex, so it is the
  farthest vertex in the cell from its nearest atom, over every image within the upper bound of
  the cell. Penalty w ((R - bound) / scale)^2, scale r_hole_max one-sided, the band midpoint
  two-sided. R_hole is continuous and piecewise smooth in the parameters; gradients are scipy's
  finite differences, as for every term outside energy_jax.

## Measured

- Exact against analytic values (simple cubic a sqrt3/2, diamond and rock-salt tetrahedral sites
  a sqrt3/4, an open square-net layer), against a brute-force grid with +-3 images on 400 random
  triclinic cells, and bracketed by the grid bounds in every case (`tests/unit/test_cell_terms.py`).
  The penalty's finite-difference slope matches the analytic one on the open layer to 1e-5.
- On TiO2's survivors (ADR-0030) R_hole follows the volume: the layered class's members with an
  open gap (R_hole >= 2.87) all have V_fu >= 41 A^3, which a density already rejects, and the
  closed-gap member at V_fu 32.7 has R_hole 2.17, inside the references' range. With a density,
  R_hole adds nothing for TiO2; it matters without one, and as a request for open structures.
- Cost: about 5-8 ms an evaluation at random parameters against an objective of 1.5-6 ms; nearly
  nothing for dense structures, which the grid bound settles.
