# Automatic-search fixtures from derived constraints and a density per formula unit

ADR-0026's exhaustive runs found TiO2's and ZnO's zero-error class far from unique: 736 TiO2
entries at or below 1e-3 with a median shortest O-O of 1.81 A, almost half on a lattice bound;
163 ZnO entries with a median Zn-Zn of 2.37 A (wurtzite 3.210). Design review of 2026-10-17.

## Measured first (recorded zero-error entries re-scored, structure fixed)

| constraints | TiO2 survivors | ZnO survivors | published structures |
| --- | --- | --- | --- |
| hand fixture, as run | 735 | 163 | anatase 0.0020 (Ti BVS 4.28 against 4 +- 0.1) |
| + measured `auto_coupling_cutoff` | 673 | 51, 3 classes, none on a bound | — |
| + O-Ti-O band [75, 105] | 43, 8 classes | — | — |
| derived (ADR-0028): composition, charges, cited CN | 52, 8 classes | 51, 3 classes | rutile, anatase, wurtzite all 0 |

Why ZnO's anion angle was off: with no `auto_coupling_cutoff`, the X-O-X angle's cutoff was the
bond's `distance_max` (2.04 A); the optimiser puts bonds on their band's long end, exactly at
2.04, and the strict `< cutoff` test drops them, so most O-centred angles were never counted.

The eight TiO2 classes left by the derived constraints include rutile's and anatase's (checked by
topology key) and a CdI2-like layered class whose volume per formula unit runs from 32.8 to
104 A^3. The volume per formula unit separates most of the rest (published: rutile 31.2,
anatase 33.5-34.1 A^3); an anion-anion floor does not (survivors' O-O 2.08-2.74 A, a distorted
anatase-class member at 2.33), as ADR-0027 found for silicates.

## Decision

- **The ZnO and TiO2 automatic fixtures state only composition, charges, the cited cation CN
  and the measured density**; everything else is derived (ADR-0028). NaCl stays hand-written
  (Na's s0 = 0.167 is below ADR-0028's validated range); Si has no anion and is already unique.
- **The anion coupling cutoff defaults to the largest measured shell cutoff that reaches the
  anion** (`config._measured_cutoff_for`: a coordination constraint's `cutoff_distance` or a bond
  constraint's `cutoff`), and to `distance_max` only when no constraint states one.
- **A density bounds the volume per formula unit.** `density_g_cm3` or `density_u_A3`, a value
  (+-2 %, ADR-0017) or [min, max]; exactly one of them, both is an error. Internally u/A^3
  (1 g/cm^3 = 0.602214076 u/A^3, exact), so V_fu = M / rho with M in u
  (`mizuho/data/atomic_weights.py`, IUPAC 2021, shared with `scripts/density_to_volume.py`).
  Penalty as `volume_constraint` (weight 5, midpoint scale), in `objective/cell_terms.py`, which
  energy.py and breakdown.py both call. It holds for every formula-unit count, which a cell
  volume (`volume_constraint`) cannot in the automatic search.
- **TiO2 is split** into `tio2_rutile.yaml` (expected 136; D(meas.) 4.23(2) g/cm^3, Handbook of
  Mineralogy, `references/rutile.pdf`; V_fu 31.20-31.50 A^3, the Handbook's own cell 31.22) and
  `tio2_anatase.yaml` (expected 141; 3.79-3.97 g/cm^3, `references/anatase.pdf`; 33.41-34.99).
  One fixture cannot state the density of two polymorphs. ZnO keeps {186, 216} with zincite's
  5.66(2) g/cm^3 (`references/zincite.pdf`; 23.79-23.96 A^3).

## Consequences

- The c = 2.500 A rutile-family entry of the first TiO2 run scores 0.0033 with rutile's density.
- What the derived constraints and density leave is measured by rerunning the exhaustive search
  (ADR-0026's table), one fixture at a time, one worker.
- A CdI2-like layered class with closed gaps at anatase's volume is not separated by density or
  by the void term of ADR-0031: first-shell constraints cannot tell it from anatase, as they
  cannot tell wurtzite from zinc blende (ADR-0027's limit).

## Measured: exhaustive reruns (2026-10-20/21, 6a56473, one worker each, three at once)

| fixture | entries | at or below 1e-3 | on a bound | record rank 1 | published group: record / merged rank | time |
| --- | --- | --- | --- | --- | --- | --- |
| `tio2_rutile` | 5507 | 206 | 33 | P6_3/mmc (194), Ti@a O@f | 136: 5 / 3 | 28.3 h |
| `tio2_anatase` | 5507 | 187 | 40 | P6_3/mmc (194), Ti@a O@f | 141: 4 / 2 | 28.2 h |
| `zno` | 4642 | 330 | 13 | F-43m (216), Zn@a O@c | 216: 1 / 1; 186: 39 / 8 | 16.3 h |

Both TiO2 fixtures give nearly the same merged ranking: a layered P6_3/mmc structure whose
octahedra share only edges (CdI2-like), then anatase's class, then rutile's. The density does not
pick the polymorph: within the derived bond and angle bands each topology stretches or squeezes
into either window. At rutile's density the rutile entry has c = 2.731 A (published 2.959) and
the anatase entry c = 8.09 A (9.51); at anatase's density the rutile entry swells to
V_fu 33.4 A^3. The layered class ranks first by ADR-0026's symmetry-first rule (order 24 against
16). Shortest O-O over the zero-error entries: median 2.42 A (rutile run), 2.50 A (anatase run).

ZnO's zero-error entries doubled from the hand constraints' 163 to 330: the derived bands (Zn-O
1.86-2.10 A, O-Zn-O [95, 125] deg) are wider than the hand ones (1.93-2.04 A, [105, 115]) and
admit edge-sharing tetrahedra (median shortest Zn-Zn 2.56 A, wurtzite 3.210). Merged, zinc blende
is first, tetragonal tetrahedral nets (137, 136, 131) follow, wurtzite is 8th.

So first-shell constraints and a density narrow each compound to a handful of zero-error
topology classes that contain the published one; they do not rank it first.
