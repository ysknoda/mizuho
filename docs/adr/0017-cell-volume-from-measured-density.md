# Cell volume bounded by the measured density

ADR-0016 left zero-error impostors on anatase, corundum and quartz. A cited density is a
chemical fact independent of the reference refinement, and two of the three impostors changed
the cell volume (anatase +5.7 to +9.4%, quartz +7.3%).

## Decision

- `volume_constraint` bounds come from the Handbook of Mineralogy D(meas.) (Anthony, Bideaux,
  Bladh & Nichols, eds., Mineralogical Society of America; one PDF per mineral in
  `references/`): V = Z M / (rho N_A), M from IUPAC 2021 atomic weights
  (`scripts/density_to_volume.py`). The stated range is used; a single value gets ±2%; zincite's
  5.66(2) uses its uncertainty. D(calc.) is never used (it is computed from a cell); it serves
  only to check the conversion, reproducing four reference cells within 0.5%.
- Eleven fixtures carry the bound. **Spinel is excluded**: its D(meas.) 3.6–4.1 is for natural,
  Fe- and Zn-bearing spinel; pure MgAl2O4 is lighter and its own reference cell (528.1 A^3) lies
  outside the band. Wollastonite is excluded as a Z = 1 model (CONTEXT.md).

## Measured (x4 global-minimum gate, seed 42, ADR-0015 + 0016 + this)

| mineral | result |
| --- | --- |
| fluorite, pyrite, sphalerite, spinel, wurtzite, **anatase** | pass (anatase +2.5 / −2.0 %, was c +4.0 %) |
| quartz | zero-error impostor, a −1.2 / c +3.7 %, volume +1.1 % |
| corundum | zero-error impostor, c +11.6 % on a bound, volume −0.2 % |
| forsterite | zero-error impostor, a +9.9 / b −5.4 / c −3.8 %, volume +0.0 % |
| diopside, topaz, adamite | stage 1: reached 8.3e-3, 4.1e-2, 1.0e-2 against published 0 — the optimiser's failure |

Every remaining impostor keeps the published volume and changes only the cell's shape. No term
so far — first shell, bond valence, density — constrains a volume-preserving change of shape.
This is recorded as the limit of the information these constraints carry (design review Q2 d),
pinned as `*_round4_volume` xfail fixtures.

## Budget ladder on diopside (design review item 1)

| budget | error reached (published 0) | lattice | bounds | time |
| --- | --- | --- | --- | --- |
| x4 | 8.3e-3 | +5.7 / −5.6 / +1.7 % | a, b, β | 63 min (with topaz) |
| x8 | 2.7e-5 | −4.2 / +3.2 / +2.7 %, β +6.8 % | β | 83 min |
| x16 | **0** | −3.4 / +5.6 / −3.0 %, β −7.4 % | b, β | 230 min |

More budget reaches zero error, and reaches it away from the published structure: diopside's
remaining failure is a zero-error impostor, not an optimiser shortfall (ADR-0007's ladder again).
A better optimiser would arrive there faster; it would not arrive at the published structure.
