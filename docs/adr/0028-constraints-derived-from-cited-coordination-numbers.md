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

## Measured: the three hard minerals with derived constraints (2026-10-15, x4, seeds 42-46)

Same script and settings. "Edge" lists lattice parameters that ended on their search bound.

| mineral | seed | error | S1 | tier 1 | lattice |
| --- | --- | --- | --- | --- | --- |
| diopside | 42 | 2.5e-2 | no | fail | b, c on edge |
| diopside | 43 | 2.1e-2 | no | fail | a, b, beta on edge |
| diopside | 44 | 1.5e-22 | yes | pass | a, beta on edge |
| diopside | 45 | 0 | yes | pass | a, b, beta on edge |
| diopside | 46 | 2.3e-7 | yes | pass | c on edge (+6.6 %) |
| topaz | 42 | 0 | yes | fail | a, b on edge |
| topaz | 43 | 0 | yes | fail | a on edge |
| topaz | 44 | 4.4e-4 | yes | fail | a -13, b +14 %, b on edge |
| topaz | 45 | 0 | yes | fail | b, c on edge (c +14 %) |
| topaz | 46 | 0 | yes | fail | b, c on edge (c +14 %) |
| adamite | 42 | 1.6e-2 | no | fail | a, c on edge |
| adamite | 43 | 0 | yes | fail | b on edge |
| adamite | 44 | 0 | yes | fail | within 1.3 %, no edge |
| adamite | 45 | 0 | yes | fail | c on edge |
| adamite | 46 | 1.7e-4 | yes | fail | a -5.6, c +6.1 % |

Totals: diopside S1 3/5, tier 1 3/5 (hand fixture with ADR-0027's band: 3/5, 2/5); topaz S1
5/5, tier 1 0/5; adamite S1 4/5, tier 1 0/5; tier 2 0/15. Adamite's published structure scores
3.8e-5 under the derived config, so its zero-error other-class points sit below it.

- Tier 1 with derived constraints is no worse than with the hand fixtures: the hard minerals
  fail for the same reason (zero-error points of other classes, ADR-0027), not for the input.
- The derived bands are wider than the hand-measured ones, and lattices run to their bounds
  far more often (14 of 15 here; wurtzite among the easy ones). Tier 2 is where derived
  constraints lose.
- Adamite seed 44 reaches the published lattice within 1.3 % in another class: the lattice
  alone does not tell the classes apart.

## Measured: are the derived configs' end points the hand fixtures' ones? (2026-10-15)

Every recorded end point of the three minerals (ablation, CONT, backends, class enumeration,
ADR-0027's runs, and the fifteen derived fits) was classed with the hand fixture's cutoffs and
scored under both configs. Points per mineral: diopside 29, topaz 25, adamite 25.

- **Diopside: the derived config rejects every hand zero-error other-class point.** All five
  such points (0 to 4.1e-8 under the hand fixture) score 8.8e-3 to 0.38 derived; the two
  broken down lose it almost entirely to the derived X-Mg-X octahedral band (observed 63-144 deg; the hand fixture has no Mg angle
  band). This is why tier 1 holds 3/5 derived, and why all three derived tier-1 hits are new
  points.
- **Topaz and adamite: the hand zero-error points stay zero derived**, and the derived fits
  add new ones. Of ten derived end points, two fall in classes the hand runs had reached at zero
  error (adamite seeds 43, 45 with CONT seeds 42, 46); the rest are new classes. The hand
  fixture rejects the new ones for two reasons the derived inputs do not carry:
  - **Which anion a cation bonds.** Topaz seed 45 (1.10 under the hand fixture) has Si with two O
    and, by its derived CN 4 over all anions, two F; the hand fixture counts Si(O) only. ADR-0028
    Q7 states one combined anion range because this is not an input.
  - **Hydrogen's second neighbour.** Adamite seed 44 (1.44) puts a second O within the hand H(O)
    cutoff of 1.65 A (soft CN 1.87); the derived H shell ends at 1.25 A, so nothing keeps a
    second O away from H. Its Zn polyhedra also go unconstrained (O-Zn-O 44-147 deg): adamite's
    Zn is cited as CN [5, 6], which gets no angle band.

## Decided: anion partition (a) and hydrogen-bond geometry (b) (2026-10-15)

- **(a) `bonded_anions`.** Optional, per cation, the anions it bonds, a cited fact like its CN
  (topaz: `{Si: [O]}`). The cation's bands, CN and angles run over those anions only, and every
  other anion is held outside the shell it would have (CN [0, 0] at that pair's derived cutoff).
  The anion CN and polymerisation ranges are counted over the anions the cations bond, so topaz's
  polymerisation becomes O(Si) exactly 1 (4 Si-O bonds over 4 O) instead of [0, 1] over O and F.
  Without the key a cation bonds every anion, as before. The CN term's scale gets a floor of half
  a neighbour, so a [0, 0] band does not divide by zero (every band with cn_max >= 1 already had
  a midpoint of at least 0.5: unchanged). The reference gate fills it from the hand fixtures
  (topaz Si); every reference still scores as before.
- **(b) Hydrogen-bond angle.** With H at CN 1, any anion within 2.2 A of H sits at 130-180 deg
  from the O-H bond: moderate and strong hydrogen bonds have H...A <= 2.2 A and D-H...A >= 130
  deg; weak ones (beyond 2.2 A) get no band (Jeffrey 1997, as tabulated by Steiner, *Angew.
  Chem. Int. Ed.* 41, 48, 2002, Table 1; not read in the original from this environment).
  Two acceptors both inside 2.2 A cannot meet it, so short three-centre bonds are excluded with
  it; `auto_hydrogen_bond: false` switches it off. Adamite's reference (acceptor at 2.32 A) is
  unaffected.
- **Effect on the recorded points.** Topaz derived seed 45 (Si with two F) scores 25 instead of
  0. Adamite derived seed 44 (two O at 1.67 A, 71 deg from the O-H bond) scores 0.30 instead of 0;
  seed 46 0.084 instead of 1.7e-4; seed 43 3e-4 instead of 0. The other topaz points (seeds 42,
  43, 46) and adamite seed 45 stay at zero: they differ from the published structure in ways
  neither rule touches.

## Measured: centring inside the bands (c) (2026-10-15)

`scripts/centring_experiment.py`: from each derived tier-1 end point, and from the published
structure, L-BFGS-B on E + 0.01 E_ideal, where E_ideal is the derived objective with each derived
bond band shrunk to d0 = R0 - b ln(V / CN) ("bonds"), each derived polyhedron angle band to its
ideal angle ("angles"), or both. Lattice deviation from the published cell, in %; every point
keeps E <= 1e-4 and its class.

| point | before | both | bonds | angles |
| --- | --- | --- | --- | --- |
| wurtzite end | a -5.3, c +11.3 | -0.7, +1.2 | -0.7, +1.3 | -2.4, +5.5 |
| forsterite end | a -8.2, c +5.6 | within 1.7 | a -8.3, c +5.3 | within 1.6 |
| corundum end | a -2.9, c +3.9 | -3.4, +4.1 | -2.8, +3.9 | -4.0, +5.5 |
| anatase end | a +2.4, c -2.7 | +6.4, -12.0 | +2.7, -2.7 | +4.8, -9.2 |
| diopside seed 45 | up to 7.4 | within 1.5 | up to 7.4 | not run |
| diopside seeds 44, 46 | up to 7.4 / 6.6 | unchanged / 6.3 | unchanged / 6.5 | not run |
| spinel, sphalerite, quartz, fluorite | within 1.2 | within 1.5 | within 1.0 | not run |
| anatase published | 0 | +6.4, -12.0 | +2.1, +0.1 | +4.8, -9.2 |
| corundum published | 0 | -2.9, +5.5 | 0 | -4.0, +5.5 |
| others, published | 0 | within 2.0 | within 0.2 | |

- Ideal bond lengths pull a strained cell back (wurtzite) and leave every published structure
  within 2.1 %. Ideal angles pull a sheared cell back (forsterite, diopside 45) but push the
  published anatase and corundum 5-12 % off: their octahedra are distorted by edge and face
  sharing, and an ideal octahedron is the wrong target.
- A local minimisation does not move points pinned on several bounds (diopside 44).
- Easy minerals' tier 2: 5/8 before; 6/8 with "bonds" or "both" (different sixes).

## Decided: centring on ideal bond lengths (c) (2026-10-15)

Option "bonds" above. After Stage 2, each refined point is moved by L-BFGS-B on
E + 0.01 E_ideal, E_ideal the objective with every derived bond band shrunk to its ideal length
(`distance_ideal` = R0 - b ln(V / CN), carried by each derived bond). The move is kept only if E
rises by at most 1e-4 and the topology class is unchanged (`mizuho/optimize/driver.py` `_centre`).
On by default; `optimization.centring: false` switches it off; a no-op for configs without
derived bonds (every hand fixture). Ideal angles are left out: they move the published anatase
and corundum 5-12 %. Sheared cells (forsterite, diopside) are not fixed by this; they are a
search problem, not a centring one.

## Measured: topaz and adamite with (a), (b) and centring (2026-10-15, x4, seeds 42-46)

| mineral | seed | error | S1 | tier 1 | lattice (%) |
| --- | --- | --- | --- | --- | --- |
| topaz | 42 | 3e-6 | yes | fail | a +3.9, b -4.7, c +0.8 |
| topaz | 43 | 2.2e-2 | no | fail | b +14.3, c -13.7 |
| topaz | 44 | 1e-8 | yes | fail | b +8.7, c -10.9 |
| topaz | 45 | 4.2e-6 | yes | fail | a -13.6, c +14.0 |
| topaz | 46 | 0 | yes | fail | a -13.6, c +13.9 |
| adamite | 42 | 1.6e-2 | no | fail | within 1.3 |
| adamite | 43 | 1.9e-10 | yes | fail | b -5.8, c +7.2 |
| adamite | 44 | 7.8e-6 | yes | fail | a +5.3, b -5.8 |
| adamite | 45 | 0 | yes | fail | a -6.6, c +7.9 |
| adamite | 46 | 3.3e-6 | yes | fail | a +3.8, b -5.8, c +5.0 |

Tier 1 stays 0/5 for both (before: 0/5, 0/5); S1 4/5 for both. (a) and (b) remove the points they
were written for, and the search lands on other zero-error classes instead: the distinguishability
problem of ADR-0027 is not one missing rule. Centring leaves these cells 4-14 % off; they are
other classes, so no published cell is there to be pulled back to.

## Open

- Sheared cells inside the derived bands (forsterite, diopside): centring on bond lengths does
  not reach them; ideal angles do but break distorted octahedra.
