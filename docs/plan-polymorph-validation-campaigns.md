# Polymorph validation campaigns after Al-O-N

Design review of 2026-10-28/29 (grilling, rounds 1-5). Goal: run searches that, like `runs/alon`,
extract many polymorphs from one constraint setting, both to validate mizuho and to show what it
can do across oxides, nitrides, sulfides and covalent systems. Nothing here runs before the Al-O-N
run ends, except short pre-gate checks.

## Success criterion

- **Gate: known-polymorph recall.** Every gate polymorph of a system must appear as a zero-error
  class (error <= 1e-3). Matched by bond-graph topology class (`mizuho compare`, ADR-0019); after
  MLIP relaxation also by StructureMatcher (secondary).
- **Report:** number of distinct classes, MLIP energy ranking (known phases near the top?),
  novel candidates. For single-composition systems also the recall fraction against databases:
  recovered / database entries inside the search's bounds (atoms per primitive cell, sites in the
  highest group, density range). Databases: IZA (SiO2 frameworks), COD, SACADA (carbon);
  AFLOW optional; Materials Project only if the user sets an API key himself.

## Pre-gate (before every production run)

1. `mizuho check` of each gate polymorph's COD CIF under the run's config: zero error required.
2. Phase 1 rank of the gate assignment within `max_candidates`, using the equivalent
   fixed-composition config (ADR-0033 Oracle 2; `candidates` rejects composition-free configs).
   A gate outside the cap raises `max_candidates` for the whole system.
3. Phase 1 counts over all 230 groups, for the run-time estimate.
4. `mizuho fit --budget 4` only where 2 shows a problem.

## Rules shared by every system

- **Density** (Q9): the span of the cited densities of the gate polymorphs, no margin; for a
  composition-free run the span of its end members instead, as Al-O-N (2026-10-29: a gate-only
  span would reject every other composition on the tie line).
- **CN given as a range** (Q3, Q10): ADR-0035 (to write): the derived bond band is the union of
  the bands of every CN in the range, s in [V/CN_max - 0.16, V/CN_min + 0.16]; no X-M-X angle
  bands for a range (as now); centring uses the converged site's integer CN. Measured motive:
  quartz under Si CN [4, 6] scores 0.0027 (band [1.639, 1.789] A against its 1.576-1.644 A
  bonds), 0.0000 under Si CN 4; corundum's 5.2e-4 in Al-O-N has the same cause. Pre-validated
  before implementation by writing the union bands as explicit `bond_constraints` (with cutoffs).
- **Low-s0 cations** (Q30): Li CN 6 (s0 0.167) and Cs CN 8-12 fall below `MIN_S0` = 0.25;
  their bond bands are stated explicitly, measured on COD reference structures.
- **Code state:** production runs from a worktree pinned at a recorded commit
  (`runs/<system>/code_state.txt`), record JSONL and resume as `runs/alon`.

## Compute (Q17, Q21, Q28, Q29)

- This Mac only for now; Genkai is considered separately if results come too slowly.
- At most 6 busy cores; heavy jobs one at a time. Short tasks (check, Phase 1 counts, pytest)
  may run beside a production run, briefly reaching 7-8 cores. Production runs after Al-O-N use
  5 workers.
- DFT (VASP locally, 4 ranks): first 3 Al-O-N structures to measure the time per structure,
  then decide Mac or Genkai for the rest. Estimate from `runs/dft_classes` (25 classes of about
  8 atoms, 56.9 h): 16-41 h per 21-atom structure if cost goes as N^2-N^3.

## Downstream (Q6, Q16, Q23, Q24)

- **Full procedure** (as Al-O-N: export --best 3, MACE medium-mpa-0, DFT stage 1/2) for the
  multi-component composition-free systems: Ga-O-N, Si-O-N, Cu-Zn-Sn-S, Li-Zr-P-O, Li-P-S,
  Cs-Pb-I, Zn-Ge-P. DFT on the Mac waits for the measurement above.
- **Reduced** (MLIP ranking; DFT only for gate phases and a few top novel ones): SiO2 (both
  runs), B-N, C, ice.
- Energy reference: tie line for single-cation systems (Si-O-N: alpha-quartz + beta-Si3N4;
  Ga-O-N: beta-Ga2O3 + GaN); convex hull of the formal-charge end members plus known COD
  ternary/quaternary phases otherwise, all relaxed with the same MLIP. The hull analysis becomes
  a separate package, planned in its own session.

## First wave, in this order (Q22)

| # | system | settings | gate polymorphs | downstream |
|---|---|---|---|---|
| 1 | Cu-Zn-Sn-S | Cu, Zn, Sn, S all CN 4; max_atoms 16, max_sites 8; all elements required; the 38 derived groups, P222 included however long its Phase 1 takes; density [4.096 sphalerite, 4.719 Cu2SnS3 COD 7246217]; config `runs/czts/czts.yaml` | kesterite (COD 4328658, pre-gate error 3e-4 without the density term); stannite-type, PMCA and the wurtz orderings reported from ideal enumerated models | full |
| 2 | B-N | composition-free [B, N]; B, N CN [3, 4]; max_atoms as C | h-BN, c-BN, w-BN | reduced |
| 3 | C | zero-charge, single element; CN [3, 4]; ADR-0036 primitive-cell bound, ADR-0037 C-C bond valence (R0 1.5445 A from diamond, b 0.37, V 4: union band [1.396, 1.609] A, graphite predicted 1.438 vs 1.421 A); max_atoms after Phase 1 counts | graphite 2H/3R, diamond, lonsdaleite; SACADA entries in bounds reported | reduced |
| 4 | SiO2 dense | Si CN [4, 6], O [2, 3]; max_atoms 24, max_sites 7; density [2.28, 4.294] | alpha-quartz, alpha-/beta-cristobalite, HP-tridymite, coesite, stishovite, seifertite; moganite after its CIF check | reduced |
| 5 | Si-O-N | all elements required; Si [4, 6], O [2, 3], N [3, 4]; max_atoms 28, max_sites 6; density [2.66, 4.29] | sinoite Si2N2O | full |
| 6 | Ga-O-N | as Al-O-N (Ga [4, 6], O [2, 4], N [3, 4], 21 atoms, 5 sites) with the new derivation; Al-O-N's zero-error classes re-scored with `check` | Ga3O3N spinel if a CIF is found | full |
| 7 | SiO2 frameworks | Si CN 4, O 2 (angle bands and ADR-0027 T-X-T active); max_atoms 36, max_sites 8, max_candidates 200; density [lowest IZA framework in bounds, 2.92 coesite] | IZA frameworks in bounds (SOD, CHA, ...) | reduced |

**Cu-Zn-Sn-S space groups are derived, not exhaustive** (2026-10-29). Over all 230 groups at
16 atoms / 8 sites Phase 1 reached 19.0 million assignments by SG 48 (Cmmm 9.8 million in 239 s,
Immm 2.9 million) and Pmmm ran over 6 h at 5.2 GB, so the run visits only the groups of cation
orderings on zinc blende and ideal wurtzite: `scripts/derive_ordering_spacegroups.py --cations
Cu:1 Zn:2 Sn:4 --anion S:-2 --max-atoms 16` (every HNF supercell up to 16 atoms, every
charge-neutral decoration, spglib at 1e-3) gives 38 groups from 131364 decorations
(`runs/czts/ordering/`), all non-centrosymmetric, containing kesterite 82, stannite 121, PMCA
111, wurtz-kesterite 7 and wurtz-stannite 31. 4H and other tetrahedral nets are out of scope
(B-N and C cover nets). COD's three I-42m Cu2ZnSnS4 entries (1550125, 2105813, 9004750) are Cu/Zn-disordered (occupancy
0.5 on 4d), which mizuho cannot read, so no ordered stannite-type gate exists. `mizuho/data/
atomic_weights.py` lacks Cu and Sn (the density term raises); add them (IUPAC 2021 abridged: Cu
63.546, Sn 118.71) once Al-O-N ends. COD has no wurtz-kesterite or wurtz-stannite: their ideal enumerated
models are checked and reported, not gates (ADR-0003).

Cu-Zn-Sn-S and SiO2 frameworks run on the current code; the others need ADR-0035-0037,
implemented in a worktree while Cu-Zn-Sn-S runs. Si3N4 polymorphs are outside every run (Q20:
Si-O-N requires all elements).

## Second wave (details after the first wave)

Li-P-S (beta/gamma-Li3PS4, Li7PS6), Li-Zr-P-O (NASICON, rhombohedral only: at max_atoms 36 and
5 sites R-3c gives 26 assignments, 5 compositions including LiZr2P3O12, while P-1, P21/c, C2/c
and Pnma give none), Cs-Pb-I (alpha/beta/gamma/delta-CsPbI3), ice (XI, Ic, IX, VIII), Zn-Ge-P
(ZnGeP2). Organic molecular crystals are out of scope: constraints are per element pair, and
packing is set by van der Waals contacts.

## Reference sources

Handbook of Mineralogy PDFs in `references/` (cell, space group, density; no coordinates):
quartz 2.66, tridymite 2.28 (low-temperature F1 cell), cristobalite 2.33 (alpha), moganite 2.55
as printed but 2.617 recomputed from its own cell (unexplained, UNCERTAIN), coesite 2.92,
stishovite 4.29, seifertite 4.294, sinoite 2.84, kesterite 4.524 and stannite 4.490 (both
natural Fe-bearing; stannite is Cu2FeSnS4, so CZTS densities come from COD). Every CIF used by
`check` comes from COD, each download approved by the user.

## Measured Phase 1 sizes (probes, 2026-10-28)

| config | 194 | 166 | 15 | 2 |
|---|---|---|---|---|
| SiO2 24/7 | | | 419 | 2272 |
| SiO2 36/8 | 9444 | 3412 | 647 | 3176 |
| SiO2 48/10 | 51879 | 14957 | 1268 | > 80 s |
| Si-O-N 28/6 | 3436 (6 comps) | 1743 (20) | 515 | 3573 |
| Cu-Zn-Sn-S 16/8 | I-4: 12214 (16 comps), Pc: 1 | | | |

Al-O-N's Phase 2 cost: 30 core-s per entry (21 atoms, 31290 entries, 43.9 h on 6 workers).
