# Like-charge minimum distances are derived from the shared ion's bands

ADR-0028 derives bonds, coordination, angles and bond-valence sums from the cited cation CN, but
no like-charge distance: below the bands, only the hard-sphere overlap (covalent-sized radii,
`HARD_RADII`: Al 0.50, N 0.65, O 0.66 A) keeps two anions or two cations apart. In the Al-O-N
composition-free run (ADR-0033), 2,062 of the 2,568 zero-error classes have an anion pair below
2.0 A (Al5O3N3 class 7, SG 157: N-O 1.370 A, an NO3-like unit), and 2,448 an Al-Al pair below
corundum's 2.655 A, down to 1.00 A, the hard-sphere limit. The MLIP and DFT relaxations then change
the coordination (ADR-0035). Stating `repulsion_constraints` by hand (N-O/N-N/O-O >= 2.4 A,
Al-Al >= 2.6 A) rejects class 7 and leaves corundum at 5.2e-4; this ADR derives such entries.

## Decision

- **Which pairs.** Every anion pair present (like and mixed: O-O, N-N, N-O) and every cation pair
  (like and mixed) that some counter-ion bonds both of (`bonded_anions` respected). One
  `repulsion_constraints` entry each, weight 1.0, the existing one-sided term (energy.py
  section 5). **Opt-in:** only with `auto_repulsion: true` (default `false`, decided after the
  pilot below); without it the objective is exactly the previous one. A repulsion the config
  states for a pair wins. A pair no counter-ion bonds both of gets none.
- **Distance (the rule).** Two ions A, B bonded to one shared ion C, by bonds no shorter than
  their bands' short ends a = d_min(A-C), b = d_min(B-C), at an angle of at least theta at C, are
  at least

  d_min(A-B) = sqrt(a^2 + b^2 - 2ab cos theta)

  apart, or max(a, b) sin theta where one bond's foot falls short of the other's end
  (a < b cos theta). The distance grows with both bonds and the angle, so this is the minimum
  over the bands (checked against a brute-force grid in
  `test_shared_ion_distance_is_the_minimum_over_longer_bonds_and_wider_angles`). The entry takes
  the smallest value over every shared ion C, rounded down to 1 mA. The bands are the final ones
  (a stated bond's distance_min wins), so a later change of the derived bands (e.g. the union band
  for CN ranges planned in docs/plan-polymorph-validation-campaigns.md) carries over.
- **Angle floor at a cation (anion pairs).** The Tammes angle of the cation's largest CN (the
  largest smallest angle that many points on a sphere can have) less 20 deg, never below 60 deg:
  CN 4 89.47, CN 5-6 70, CN 7 and above 60. The 20 deg is the octahedral band's own allowance
  (ADR-0028: 90 -> 70 deg); 60 deg means the two anions are no closer to each other than the
  shorter bond is long. A cation of CN 1 (H) bonds one anion, so no anion pair meets at it.
- **Angle floor at an anion (cation pairs).** The anion-centred band's lower bound
  (`auto_coupling_angle_range`, 80 deg by default, or a lower stated band at that anion covering
  both cations), which the objective already holds within the anion's cutoff; 60 deg where no
  band covers the pair (pairs with H, whose shell the auto band leaves out; the band switched off).
- **Why not ionic radii.** Shannon (1976) O2- is 1.35-1.42 A (CN 2-8), so 2 r(O) = 2.70-2.84 A,
  above corundum's shared-face O-O of 2.526 A and every reference's shortest O-O (2.456-2.625 A);
  a sum of radii would need a factor read off the references. The shared-ion rule uses only the
  derived bands and an angle, so it scales with the cation (Si-O short, Ca-O long) as the bands do.

### Values, sources and confidence

| quantity | value | source | confidence |
|---|---|---|---|
| bond-band short end | R0 - b ln(s0 + 0.16) | ADR-0028 (Brese & O'Keeffe 1991) | as ADR-0028 |
| Tammes angles, N = 2-12 (deg) | 180, 120, 109.47, 90, 90, 77.87, 74.86, 70.53, 66.15, 63.43, 63.43 | Fejes Toth 1943 (N <= 6, 12), Schuette & van der Waerden 1951 (N = 7-9), Danzer 1963 (N = 10-11); not read in the originals in this environment | ~95 % for N <= 9; UNCERTAIN (~85 %) for N = 10-11, which the 60 deg floor overrides anyway |
| allowance | 20 deg | chosen: ADR-0028's octahedral band (90 -> 70), not cited | UNCERTAIN for CN 5 and 7-12: tested only by adamite Zn (CN 5-6), wollastonite Ca (7), diopside and fluorite Ca (8). A survey of X-M-X minima per CN (e.g. ICSD polyhedra) would resolve it |
| unbanded floor | 60 deg | chosen: anion-anion no shorter than the shorter bond | UNCERTAIN (~85 %): diopside Ca reaches 60.03 deg, a 0.03 deg margin. Below it a pair still scores zero unless it also falls under every other shared cation's bound |
| anion-centred floor | 80 deg | the existing auto band ([80, 160], not cited; ADR-0027) | already enforced by the objective; smallest reference corundum O, 84.6 deg |

## Measured: the references

Every reference regenerated from composition, charges and cited CN (`_minimal_config`, 12 of 13;
pyrite below): every derived entry lies below the reference's shortest pair, and every derived
repulsion term is exactly 0 (`test_derived_like_charge_repulsion_is_zero_at_the_reference`).
Smallest margins:

| reference | pair | shortest (A) | derived (A) | margin (A) |
|---|---|---|---|---|
| adamite | O-O | 2.456 | 2.228 | 0.228 |
| anatase | O-O | 2.468 | 2.162 | 0.306 |
| corundum | Al-Al (face-sharing) | 2.655 | 2.320 | 0.335 |
| wollastonite | O-O | 2.552 | 2.208 | 0.344 |
| forsterite | O-O / Mg-Si | 2.555 / 2.695 | 2.208 / 2.283 | 0.347 / 0.412 |
| corundum | O-O (shared face) | 2.526 | 2.070 | 0.456 |
| topaz | O-O / F-F / F-O | 2.507 / 2.517 / 2.546 | 2.070 / 1.949 / 2.011 | >= 0.437 |
| adamite | H-Zn | 2.423 | 1.682 | 0.741 |

45 pairs in all, the largest margin topaz Si-Si (2.412 A). Smallest angles at the shared ion
against the floors: diopside Ca (CN 8) 60.03 / 60, wollastonite Ca (CN 7) 61.2 / 60, forsterite Mg
71.9 / 70, adamite Zn 75.1 / 70, corundum Al 79.6 / 70; at anions, corundum O 84.6 / 80.

**Exception: pyrite.** Its S-S of 2.162 A is the S2(2-) dimer's covalent bond, not two anions
meeting at Fe (S pairs sharing Fe are 3.078 A apart; derived 2.777 A). Its derived score goes from
1.379 to 1.428. It was already a strict xfail (Fe-S R0 misfit, ADR-0016) and stays one. A compound
with a homonuclear polyanion (S2, peroxide O2, azide N3, C2) must state a repulsion for that pair
or switch the derived ones off: composition and charges do not tell such a pair apart. One rule
does not cover it, and no rule was tuned to.

**`mizuho check` and the gate, before (HEAD) and after with the terms on,** every fixture with a reference
structure: unchanged to the printed digit, apart from pyrite's derived score. Hand fixtures (no
`coordination_numbers`) are untouched by construction. The auto fixtures: ZnO on wurtzite.cif 0,
anatase on anatase_origin1/2.cif 8.3e-3 / 0, rutile (a 4.594, c 2.959, x 0.3049) 0. The Al-O-N
config: corundum_H.cif 5.185e-4, corundum_R.cif 1.240e-3, wurtzite AlN (a 3.112, c 4.982, u 0.382;
UNCERTAIN, not read from a source here) 4.497e-6, all repulsion terms exactly 0.

## Measured: the Al-O-N config

Derived: O-O 2.010, N-O 2.093, N-N 2.170, Al-Al 2.253 A (Al CN [4, 6]: Al-O short end 1.753,
Al-N 1.892 A; 70 deg at Al, 80 deg at O and N). Lower than the hand values (2.4, 2.6 A): the
derived entries reject contacts the bands make impossible for ions sharing a neighbour, not ones
that are merely unusual.

- Class 7 (SG 157, N-O 1.370 A): 1.0e-7 -> 0.119 (N-O term). Class 241 (SG 81, Al-Al 1.004 A):
  6.2e-5 -> 0.107 (Al-Al term 0.102). Regression tests on copies of both CIFs
  (tests/fixtures/regression).
- All 2,565 zero-error class representatives re-scored at their recorded points (no
  re-optimisation): 434 stay at or below 1e-3. Terms above 1e-3: N-O 1,180 classes, O-O 1,180,
  N-N 644, Al-Al 308 (46 by Al-Al alone).
- Against the MACE pre-screen (runs/alon/mlip/ranking.csv): 7 of the 36 ranked zero-error classes
  within 100 meV/atom of the tie line, and 3 of the 19 stage-1 picks, stay at their recorded
  points. Their MACE-relaxed structures have anion-anion >= 2.41 A; the recorded points do not.
  A recorded point lies anywhere in a wide zero-error region: the side measurement on branch
  claude/modest-swirles-7b56e6 (docs/measure-extra-terms-alon.md) found that local
  re-optimisation recovers most classes an anion-anion 2.3 A term removes, and that the term
  removed low- and high-energy classes alike. Whether a search with these terms still reaches the
  low-energy classes is what the pilot measures.

## Measured: pilot search

Runs/alon/alon.yaml restricted to one space group, 1 worker (the Mac's other cores ran VASP),
with the derived terms; the baseline is the same group's record in runs/alon/alon.jsonl (same
mizuho/ code, commit 2e4e88c, same seed). Classes: export_classes over each record (error
<= 1e-3); distances on the exported, symmetrised CIFs (`Al-Al`, and anion-anion over every anion
pair). MACE labels by the runs/alon class numbering (runs/alon/mlip).

| | SG 157 base | SG 157 with | SG 59 base | SG 59 with |
|---|---|---|---|---|
| zero-error classes | 26 | 15 (6 common, 9 new) | 74 | 58 (29 common, 29 new) |
| shortest anion-anion, min / median (A) | 1.300 / 1.712 | 2.007 / 2.095 | 1.300 / 1.668 | 1.994 / 2.093 |
| classes with anion-anion < 2.0 A | 22 | 0 | 55 | 1 (1.994) |
| shortest Al-Al, min / median (A) | 1.687 / 2.436 | 2.252 / 2.482 | 1.561 / 2.417 | 2.234 / 2.474 |
| MACE <= 100 meV/atom (stage-1 picks) | 1 (0) | 0 | 5 (3) | 1 (1) |
| stage-1 picks, all | 2 | 0 | 3 | 1 |
| MACE-excluded for anion-anion < 2.0 A | 3 | 1 | 11 | 5 |
| wall time, 1 worker | (815 s on 6) | 7,536 s | (725 s on 6) | 7,663 s |

- The terms do what they are for: no class leaves mizuho with an anion pair or an Al-Al pair
  below the derived minima (the 1.994 and 2.234 A are 1 % below them, inside the soft term's
  1e-3 threshold, after symmetrisation). The distances pile up at the minima (median
  anion-anion 2.09 A): the one-sided term holds the contacts at its bound, it does not push them
  to the 2.4-2.6 A of relaxed structures.
- They do not select for MACE energy. Of the 6 baseline classes within 100 meV/atom, 1 is found
  (Al4O3N2 class 635, 65 meV, a pick); lost are Al3O3N classes 48 (85 meV), 118 (69, pick), 129
  (98), 131 (66, pick) and Al4O3N2 class 394 (79). Of the 5 picks in the two groups, 1 remains.
  6 of 14 classes MACE excluded for anion-anion contacts are still found: their zero-error points
  now keep 2.0 A, and MACE still collapses them.
  - This agrees with the measurement on claude/modest-swirles-7b56e6 (an anion-anion 2.3 A term
    removed low- and high-energy classes alike).
  - Part of the class turnover is the search's own path dependence (a changed objective changes
    every CMA-ES trajectory): 38 new classes appear, mostly MACE saddles at 122-1,290 meV/atom
    where labelled. No baseline rerun separates that noise here (a second baseline run of each
    group would; ~2 h each on 1 worker).
- SG 189 and SG 166 were not run (about 6 and 14 h on 1 worker; the cores were busy).

**Decided (2026-11-02): opt-in, off by default.** The terms remove what they were derived to
remove and cost nothing at any reference, but on this pilot they shrink the low-energy share
rather than raise it (low-energy classes 6 -> 1, picks 5 -> 1). `auto_repulsion: true` turns them
on, e.g. where the zero-error structures themselves are the output, without an MLIP relaxation.

## Consequences

- Off by default, so existing configs and runs are unchanged; every measurement above was made
  with `auto_repulsion: true`.
- `init_guess.py` reads repulsion distance_min for its cell floor, but only as 0.6 d_min against
  the largest bond distance_max; at the default floors (theta <= 90 deg) the derived d_min is at
  most sqrt(2) times a band's short end, so 0.6 d_min stays below the bond term and starting
  cells are unchanged.
