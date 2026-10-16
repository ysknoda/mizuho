# Bond-valence sums at cation centres; anion sums reported only

ADR-0014 left same-topology impostors no first-shell term could see. A cation's bond-valence
sum ties its bond lengths to its formal valence, which the ±2% bond bands do not.

## Decision

- R0 from Brese & O'Keeffe, *Acta Cryst.* (1991) B47, 192, transcribed in full
  (`mizuho/data/bond_valence.py`): Table 2 109 cations × {O, F, Cl} = 327 entries (48 italic,
  i.e. interpolated), Table 3 71 cations × 9 anions = 639; each checked against the PDF's OCR
  layer (two OCR misreads, Pb(II)–O and Lu–N, resolved from the page image). The abstract's
  "969 pairs" does not match the 966 tabulated; untabulated values in the text are not used.
  b = 0.37 A.
- Bonds: each bond constraint's pairs within its measured cutoff, weighted by ADR-0015's
  smootherstep so the sum is continuous. Cations in the objective, |BVS − V| beyond 0.1 v.u.,
  scaled by V, weight 1.0 (`bond_valence: {weight, tolerance}`, `false` to disable). Anion
  sums reported only.
- Where a fixture's own reference exceeds 0.1, the measured deviation rounded outward replaces
  it, with the reason beside it: adamite Zn 0.60, anatase Ti 0.28, diopside Mg 0.14,
  forsterite Mg 0.12 / Si 0.13, spinel Al 0.17 / Mg 0.18, topaz Si 0.17, wollastonite Ca 0.11,
  pyrite Fe 2.54 (Table 3's Fe–S R0 does not fit low-spin Fe(II); the term is inert there).

## Measured (x4 global-minimum gate, seed 42, with ADR-0015)

| mineral | stage 1 (reached ≤ published + 1e-3) | stage 2 (lattice) |
| --- | --- | --- |
| fluorite, sphalerite, pyrite, spinel, wurtzite | pass | pass |
| anatase | pass (both 0) | a +2.6 / c +4.0 % — impostor |
| corundum | pass (both 0) | a +4.7 / c −8.8 % — impostor |
| quartz | pass (both 0) | c +6.5 % — impostor |
| diopside, topaz, adamite, forsterite | **fail**: reached 2.3e-3, 4.2e-2, 1.6e-3, 1.4e-3 against published 0 | (bounds) |

For the four large minerals the objective now ranks the published structure strictly below what
the optimiser reaches: their earlier impostors are gone, and the remaining failure is the
optimiser's (ADR-0005 stage 1), not the objective's. The three small oxides still have
zero-error impostors; anatase's are closer to Ti's formal valence than the published structure
(BVS 4.28 published, 3.86–4.14 impostors), so bond valence favours them.

Rank gate (five seeds): all pass except diopside (worst 154) and adamite (worst 3657, xfail);
identical to the run without bond valence, which Phase 1's local-shell scoring does not see.

## Consequence

Impostors remain in the objective's added terms (design review Q18 b): the work stops here and
the next step is a decision, not a parameter. Recorded as `*_round3_bvs` impostor fixtures.

## Amended by ADR-0020 and ADR-0025

Stage 1 stopped converging on the large minerals in the run above, which added compact support and
bond valence together. ADR-0020's term ablation separates the two, and demotes this term to
report-only if removing it improves tier 1 while topaz's round-2 impostor stays rejected.
Adamite's tolerances are re-measured on its new reference (ADR-0025): Zn 0.21, As 0.53.
