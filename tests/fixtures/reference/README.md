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

`spinel_mgal2o4` and `diopside_camgsi2o6` are currently reconstructed in code rather than from a
CIF, having been verified against their published bond lengths (spinel Mg-O 1.918 A vs 1.92,
Al-O 1.929 A vs 1.93; diopside Si-O, Mg-O and Ca-O ranges all within 0.01 A).

## Every file here is verified before use

A CIF states coordinates in its own space-group setting and origin choice, which need not match
mizuho's Wyckoff database. Reconstructing spinel from memory in the wrong SG227 origin produced an
Mg-O distance of 0.21 A, so nothing in this directory is trusted until it reproduces the published
bond lengths and coordination numbers of the mineral it claims to be. That check is
setting-independent, which is exactly why it is the one used.
