# Adamite's reference is a DFT relaxation with its hydrogen located

Adamite's reference was the 1937 refinement (COD 1010925, Kokkoros), and it was the exception in
every gate:

- its H is undetermined (dummy coordinates), so the gates set `charge_balance_weight` to 0;
- its As-O spans 1.588-1.814 A, which ADR-0007's span +/- 2% rule copied into a 0.415 A wide Zn-O
  band;
- it needed a bond-valence tolerance of 0.60 on Zn and a reference ceiling of 5e-2;
- it was left out of the DFT basin test (ADR-0019), since its H was not modelled.

## Decision

- **The reference is AFLOW's PBE relaxation of ICSD 34868**
  (`tests/fixtures/reference/adamite_aflow_icsd34868.cif`), an exception to "a reference structure
  is experimental" (CONTEXT.md; the wurtzite note in `plan-verification-and-tree-search.md`). It is
  accepted because mizuho's output is a starting structure for a DFT relaxation (ADR-0019), so a
  DFT minimum is a fair target. The cost is stated: PBE lengthens As-O by about 0.03 A, which enters
  the measured cutoffs and bands and lowers As's bond-valence sum to 4.47.
- **Verified before use (ADR-0003).**
  - The file declares P1; the structure is Pnnm at every tolerance from 1e-4 to 1e-1 A.
  - In mizuho's setting it is As@g H@g O@g x3 O@h Zn@e Zn@g. That is the equivalence class
    (ADR-0011) of the 1937 assignment, whose Zn@f is this Zn@e.
  - Cell 8.357 / 8.493 / 6.026 A: within 0.6% of the 1937 cell, volume -0.2%, inside the
    density-derived bound (ADR-0017).
  - As CN 4, Zn CN 5 and 6, O(As) in {0, 1}, as cited.
  - O-H 0.978 A, a DFT bond length (X-ray refinements place H at 0.8-0.9 A from O).
- **H is modelled explicitly.**
  - O-H bond [0.95, 1.00] A, cutoff 1.650 A; H(O) CN 1; O(H) in {0, 1}, since a second H on one O
    would model water rather than (OH).
  - Anion-centred constraints name `[Zn, As]` instead of `X`, with `auto_anion_coupling: false`.
    X would include H, and the hydrogen bond O...H at 2.323 A lies 0.085 A past the longest Zn-O
    bond (2.238 A), so no cutoff gives a shell containing H an empty gray band (ADR-0015).
- **Re-measured by ADR-0004 and ADR-0007:** As-O [1.68, 1.77] A, cutoff 2.435 A; Zn-O
  [1.94, 2.29] A, cutoff 2.829 A; O([Zn, As]) cutoff 2.690 A. Cited bands are kept, since the
  reference lies inside them: O-As-O [100, 120] deg, O-Zn-O [75, 130] deg and every coordination
  range. The M-O-M band [80, 160] deg, formerly injected automatically, is now stated with the
  same values (the reference realises 93.7-128.1 deg).
- **Bond-valence tolerances:** Zn 0.60 -> 0.21; As 0.53, set by PBE's long As-O, which leaves As's
  term nearly inert; H scores 0.928, within the cited 0.1.
- At the new reference every term is zero (18 of 18 constraints). The charge term is back on in
  every gate, and adamite's reference ceiling is the common 1e-3.
- **The adamite impostor records are retired** (`adamite.json`, `adamite_round2.json`,
  `adamite_round2_seedfix.json`). Their `published_vector` is the 1937 one, which ADR-0018's
  matching rule no longer accepts, and they were found under the 1937 bands. The x4 gate on the
  new reference records whatever remains.
- The 1937 file stays, as a CIF-reader case (a dummy H the reader skips) and as a pair test against
  the new reference.
- `cif_to_mizuho` treats a file declaring P1 as declaring no group and detects it, since database
  exports write every structure as P1.

## Consequences

- Tier 2 for adamite compares against a PBE lattice. Here it lies within 0.6% of the 1937 cell, so
  the 3% target keeps its meaning.
- Adamite can now enter the DFT basin test of ADR-0019.
- Phase 1 now scores the H constraints, so the published topology's rank can move either way. The
  rank-gate xfail stays, with a reason naming the old measurement, until ADR-0020's measurement 3
  re-measures it.
