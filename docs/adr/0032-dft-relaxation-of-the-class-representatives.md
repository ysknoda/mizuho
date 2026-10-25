# DFT relaxation of the zero-error class representatives

ADR-0030 found 11 TiO2 and 14 ZnO zero-error topology classes. Design review of 2026-10-22
(Q21-Q25): do they relax, in DFT, to minima, and are classes judged different also different in
total energy (tolerance 1.0 meV/atom)?

## Method

- The class representatives of `scripts/export_classes.py` (primitive cells) and three published
  structures as anchors: rutile (Handbook of Mineralogy cell, O x 0.3049), anatase
  (`anatase_origin2.cif`), wurtzite (`wurtzite.cif`). Inputs and runner in `runs/dft_classes/`.
- Settings of `runs/dft_basin` (design review 2026-10-02): PBEsol, ENCUT 520 eV, Ti_pv / O / Zn,
  KSPACING 0.25 1/A, EDIFF 1e-6, EDIFFG -0.01 eV/A, ISIF 3 with the symmetry kept: a coarse stage
  and two relaxations. POTIM is VASP's default, 0.5.
- Then every atom displaced by up to 0.03 A (fixed seed), symmetry off (ISYM = 0) and relaxed
  again: first at the coarse stage's settings, then at the agreed ones (design review 2026-10-25;
  without symmetry a 12-atom cell has 112 k-points against 34). A drop of more than 1 meV/atom
  marks the symmetry-kept result as a saddle.
- Two relaxed structures are the same when their topology key (the fixture's cutoffs), space group
  at 0.1 A and energy per atom (within 1 meV/atom) all agree. 0.1 A, not 1e-3: the symmetry-free
  relaxation leaves displacements under the force criterion that report rutile as Amm2 at 0.01 A.
  A confirmed minimum is judged on its symmetry-kept structure.

## Measured (2026-10-22 to 25)

Energies per atom, TiO2 relative to anatase, ZnO to wurtzite.

| TiO2 class | meV/atom | result |
| --- | --- | --- |
| 2 anatase type | 0 | minimum; = published anatase (0.01) |
| 7 TiO2-II (Pbcn) | +11 | minimum |
| 3 rutile type | +19 | minimum; = published rutile (0.01) |
| 8 (I4_1md) | +19 | saddle (-296); relaxes to rutile |
| 6 (Pnma) | +55 | minimum |
| 5 (P-31m) | +82 | minimum |
| 4 (P-3m1) | +102 | saddle (-7.8); distorts to P1, same topology |
| 10 (P6_4) | <= +143 | saddle (<= -55); not converged in 200 + 200 steps |
| 1 layers (P6_3/mmc) | +154 | minimum |
| 9 (P4_122) | +216 | minimum, after its topology changed in the symmetry-kept relaxation |
| 11 (P312) | +229 | minimum |

| ZnO class | meV/atom | result |
| --- | --- | --- |
| 8 wurtzite | 0 | minimum; = published wurtzite (0.01) |
| 7 layers (P-3m1) | 0 | saddle (-60); relaxes to wurtzite |
| 1 zinc blende | +6.7 | minimum |
| 3 (P4_2/mnm) | +31 | minimum |
| 13 (Ama2) and 9 (Cmmm, saddle -627) | +60 | one structure; topology changed |
| 10 (P3_221) | +88.2 | minimum |
| 11 (P2_12_12_1) | +89.4 | minimum, topology changed |
| 14 (P222_1) | +110 | minimum, topology changed (dense, near rock salt) |
| 12 (Aem2) | +112 | minimum, topology changed |
| 5 (P4_2/mmc) | +175 | minimum |
| 4 (P4_2/mmc) | +315 | minimum |
| 2 layers (P4_2/nmc) | +393 | minimum (-0.39) |
| 6 chains (P4_2/mmc) | +489 | ambiguous: coarse rattle stopped 0.3 below; the first, full-setting rattle fell >= 248 and did not converge |

- TiO2's 11 classes give 10 DFT structures, ZnO's 14 give 11. Every pair left distinct differs by
  more than 1 meV/atom (closest: ZnO 10 and 11, 1.1, with different keys and groups). The three
  merges are saddles relaxing onto another class (TiO2 8 to rutile, ZnO 7 to wurtzite, ZnO 9 to
  13).
- Classes that are DFT minima in their own topology: 7 of 11 TiO2, 9 of 14 ZnO (ZnO 6 counted).
- Every published anchor and its class agree to 0.02 meV/atom: the search's squeezed or stretched
  cells relax to the published structures.
- PBEsol puts anatase 19 meV/atom below rutile, the known ordering error of GGA functionals for
  TiO2; wurtzite below zinc blende by 6.7 meV/atom agrees with experiment. With no dispersion
  correction, the layer and chain classes' energies are indicative only.

## Consequences

- A zero-error class is a candidate, not a minimum: about a third of them are saddles or change
  their bonding on relaxation. A DFT relaxation with a symmetry-free stage is the check.
- Open: TiO2 class 10 and ZnO class 6 are not converged to a final structure.
