# Known-answer fixtures for the automatic space-group search

No `spacegroup`: the search chooses it. Success is judged on the **detected**
symmetry of the returned structure, not the label the search reports
([ADR-0010](../../../docs/adr/0010-success-is-the-detected-symmetry.md)):

| fixture | expected | why a set |
|---|---|---|
| `nacl.yaml` | 225 | — |
| `cscl.yaml` | 221 | — |
| `si.yaml` | {227, 194} | cubic / hexagonal diamond are polytypes |
| `zno.yaml` | {186, 216} | wurtzite / zinc blende are polytypes |
| `tio2_rutile.yaml` | 136 | — (its density excludes anatase) |
| `tio2_anatase.yaml` | 141 | — (its density excludes rutile) |

**ZnO and TiO2 (ADR-0030)** state only composition, charges, the cited cation
coordination number and the measured density (Handbook of Mineralogy,
`references/`): every other constraint is derived (ADR-0028). With the hand
constraints below, hundreds of unphysical structures reached zero error
(ADR-0026); the density fixes the volume per formula unit, which one TiO2
fixture could not do for two polymorphs, hence the split.

The hand constraints of NaCl, CsCl and Si: unlike `../nacl.yaml`, which states only a bond band (an isolated Na-Cl pair
in a large cell scores zero against it), each fixture here states the cited
coordination numbers, so the answer is determined by first-shell geometry.
Cutoffs sit at the midpoint of the first/second shell gap of the ideal
structure (ADR-0004); bond bands are the first-shell span widened by 2%
(ADR-0007):

| compound | reference lattice | first shell | second shell | cutoff |
|---|---|---|---|---|
| NaCl | a = 5.6402 A | Na-Cl 2.820 A (6) | 4.885 A | 3.85 |
| CsCl | a = 4.123 A | Cs-Cl 3.571 A (8) | 6.837 A | 5.20 |
| Si | a = 5.431 A | Si-Si 2.352 A (4) | 3.840 A | 3.10 |
| ZnO (before ADR-0030) | `../reference/wurtzite.cif` | Zn-O 1.975-1.991 A (4) | 3.218 A | 2.60 |
| TiO2 (before ADR-0030) | rutile a = 4.594, c = 2.959, x = 0.3049 | Ti-O 1.948-1.981 A (6) | 3.487 A | 2.73 |
