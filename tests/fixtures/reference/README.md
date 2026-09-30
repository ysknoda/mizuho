# Reference structures

Published experimental structures used as the correctness gate of
[ADR-0003](../../../docs/adr/0003-experimental-structure-is-the-gate.md): the objective is correct
only if it is approximately zero when evaluated here, with every coordination term reporting the
cited coordination number.

One file per mineral, named after the fixture it validates, so the mapping is mechanical:

| reference file | fixture |
|---|---|
| `adamite.cif` | `../adamite.yaml` |
| `topaz.cif` | `../topaz.yaml` |
| `wollastonite.cif` | `../wollastonite_casio3.yaml` |

### Added for untested space groups

See `docs/plan-verification-and-tree-search.md` for the verification numbers. Each pair is two
**separate refinements**, not one dataset in two settings, so pair tests compare within the
difference between refinements rather than asserting identity.

| file | source | role |
|---|---|---|
| `corundum_H.cif` | COD 1000017 | reference — R-3c on hexagonal axes |
| `corundum_R.cif` | COD 1010914 | CIF-reader test: rhombohedral to hexagonal axes |
| `quartz_sg152.cif` | COD 1011097 | reference — P3₁21 |
| `quartz_sg154.cif` | COD 1011159 | not used: Si-O spans 0.068 A against 0.010 A in the 152 file |
| `anatase_origin2.cif` | COD 1526931 | reference — I4₁/amd, origin choice 2 to mizuho's origin choice 1 |
| `anatase_origin1.cif` | COD 1010942 | equivalent-assignment case: maps to `Ti@a O@e`, the reference to `Ti@b O@e` |
| `forsterite_Pbnm.cif` | COD 9016385 | reference — Pnma supplied in the Pbnm setting |
| `forsterite_Pnma.cif` | COD 9015658 | setting-independence cross-check |
| `wurtzite.cif` | COD 1577381 | reference — P6₃mc, u = 0.3823(10), Zn-O 1.975 / 1.991 A. Special positions rounded to 0.3333 / 0.6667: idealise before use |

`spinel_mgal2o4` and `diopside_camgsi2o6` are currently reconstructed in code rather than from a
CIF, having been verified against their published bond lengths (spinel Mg-O 1.918 A vs 1.92,
Al-O 1.929 A vs 1.93; diopside Si-O, Mg-O and Ca-O ranges all within 0.01 A).

## Every file here is verified before use

A CIF states coordinates in its own space-group setting and origin choice, which need not match
mizuho's Wyckoff database. Reconstructing spinel from memory in the wrong SG227 origin produced an
Mg-O distance of 0.21 A, so nothing in this directory is trusted until it reproduces the published
bond lengths and coordination numbers of the mineral it claims to be. That check is
setting-independent, which is exactly why it is the one used.
