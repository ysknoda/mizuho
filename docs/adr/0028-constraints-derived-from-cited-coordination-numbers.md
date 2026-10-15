# Constraints are derived from cited coordination numbers, for any compound

Every fixture so far took its cutoffs, bond bands and bond-valence tolerances from its own
reference structure (ADR-0004, ADR-0007, ADR-0016). A compound being predicted has no reference.
Design review of 2026-10-14/15: settle how constraints are set for an arbitrary compound.

## Decision

- **Input (Q1).** The user states the composition, the ionic charges and, per cation, its
  coordination number (`coordination_numbers: {Si: 4, Ca: [7, 8]}`), a cited fact as ADR-0004
  requires; a measured density is optional (ADR-0017). Everything below is derived
  (`mizuho/constraints.py`, called first in `build_sg_config`). A constraint the config states
  for the same pair or element wins.
- **Bond band (Q2).** Each bond's valence within s0 +/- 0.16 v.u. of s0 = V / CN, turned into
  distances with Brese & O'Keeffe's R0 and b = 0.37 A (the table ADR-0016 transcribed). Measured
  on twelve references, every bond lies within 0.153 v.u. of s0 (largest: diopside Si-O,
  -0.153); in distance units the same spread is 0.6-10.7 %, so one valence width fits small and
  large cations alike. 0.16 is chosen from these data, not cited. The band is loose per bond on
  purpose: the bond-valence sum holds the mean, so a polyhedron stretched to its band edges (the
  topaz impostor of ADR-0007) is rejected while a real long/short split is not.
- **Cutoff (Q3).** The band's long end plus 3 smoothing widths (0.3 A), so the gray band ends
  where the band does. Empty gray band at 18 of 19 reference pairs; wollastonite Ca-O misses by
  0.024 A (second shell at 3.251 A), costing about 3e-5 at its reference, and is recorded as the
  exception.
- **Hydrogen (Q4).** Whenever H is present beside other cations, the auto anion constraints name
  the framework cations instead of X (ADR-0025's rule, now general). `coordination_numbers: {H: 1}`
  gives H its shell, and each anion gets X(H) in [0, 1], one H per anion; hydrates switch that off
  with `auto_hydroxyl: false`.
- **Bond-valence tolerance (Q5).** 10 % of the cation's valence (Si 0.4, Al 0.3, Mg 0.2 v.u.):
  the smallest fraction every experimental reference meets (spinel Mg 9 %, anatase Ti 7 %,
  diopside Mg 7 %). Chosen from data.
- **Range.** Cations with s0 < 0.25 (Na at CN 6: 0.167) are outside what was validated, where the
  band's long end runs away; they must state their bond constraints, and an error says so.

## Measured

Every reference regenerated from composition, charges, cited CN and density-derived volume alone
(`test_derived_config_scores_zero_at_the_reference`): twelve of thirteen score at most 3.8e-5.
Pyrite fails (1.38): Brese & O'Keeffe's Fe-S R0 does not fit low-spin Fe(II) (ADR-0016), so its
band (d0 2.566 A against the measured 2.264 A) and sum both miss; recorded as a strict xfail.

## Open

- O-M-O angle bands at cations, from the ideal polyhedron of the cited CN.
- The anion-centred CN range from composition and cation CN (Pauling's second rule), replacing
  the auto default [2, 4]; polymerisation constraints derived the same way (ADR-0007 showed they
  are fixed by composition and CN).
- Whether a search with derived constraints does as well as with the hand-measured fixtures:
  the second half of the validation.
