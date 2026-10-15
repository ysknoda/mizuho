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
- **Polyhedron angles (Q6).** X-M-X bands only where the CN fixes the polyhedron: CN exactly 4,
  [95, 125] deg (references deviate -7.7 / +12.9 deg from 109.47); CN exactly 6, [70, 110] deg with
  trans angles above 150 deg left out (references -18.1 / +20.0 deg from 90). CN 5, 7, 8 and CN
  ranges get none: their references span 60-144 deg. Chosen from data. The hand fixtures' cited
  tetrahedral band [100, 120] was narrower than wollastonite's 122.4 deg.
- **Anion CN and polymerisation (Q7).** The composition and the cited cation CN fix the mean
  number of framework cations per anion, m = bonds / anions; each anion's CN is held to
  [floor(m), ceil(m)] (H left out), replacing the generic auto [2, 4]. For high-valence tetrahedral
  cations (ADR-0027's T) the same count gives the polymerisation X(T). No number here is read off a
  reference; all thirteen references fall inside both ranges (diopside O 3-4 and O(Si) 1-2, topaz
  2-3 and 0-1, corundum and spinel exactly 4). With several anions (topaz O and F) one combined
  range is used, since which cation bonds which anion is not an input.
- **Range.** Cations with s0 < 0.25 (Na at CN 6: 0.167) are outside what was validated, where the
  band's long end runs away; they must state their bond constraints, and an error says so.

## Measured

Every reference regenerated from composition, charges, cited CN and density-derived volume alone
(`test_derived_config_scores_zero_at_the_reference`): twelve of thirteen score at most 3.8e-5.
Pyrite fails (1.38): Brese & O'Keeffe's Fe-S R0 does not fit low-spin Fe(II) (ADR-0016), so its
band (d0 2.566 A against the measured 2.264 A) and sum both miss; recorded as a strict xfail.

## Measured: search with derived constraints (2026-10-15, x4, seed 42)

`scripts/ablate_terms.py <mineral> derived`: the fixture keeps only composition, charges,
spacegroup, density and the cited CN; everything else is derived. Published assignment fixed.
Hand-fixture column: ADR-0019's x4 gate.

| mineral | derived: S1 / tier 1 | derived: lattice | hand fixture: tier 2 |
| --- | --- | --- | --- |
| anatase | yes / pass | a +2.4, c -2.7 % (tier 2) | pass |
| fluorite | yes / pass | +0.02 % (tier 2) | pass |
| sphalerite | yes / pass | +0.7 % (tier 2) | pass |
| spinel | yes / pass | +1.2 % (tier 2) | pass |
| quartz | yes / pass | a -0.3, c -1.0 % (tier 2) | a -1.2 / c +3.7 % |
| corundum | yes / pass | a -2.9, c +3.9 % | c +11.6 %, on a bound |
| forsterite | yes / pass | a -8.2, c +5.6 % | a on a bound |
| wurtzite | yes / pass | a -5.3, c +11.3 %, c on its bound | pass |

Tier 1 holds on all eight (as with the hand fixtures); tier 2 on five of eight (hand fixtures:
five of eight, a different five). Wurtzite's c runs to its bound; why the
derived bands allow that strained cell and the hand bands did not is not yet examined.

## Open

- Diopside, topaz and adamite with derived constraints, seeds 42-46: running.
