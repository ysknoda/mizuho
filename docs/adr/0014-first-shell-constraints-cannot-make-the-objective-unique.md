# First-shell constraints have not made the objective unique; the search stops here

ADR-0005 caps the impostor hunt at two rounds: a third round means a third mechanism exists and
the design, not its parameters, needs revisiting. This records that the cap has been reached and
what was measured on the way. The design review of 2026-10-01 is summarised at the end.

## Two rounds

**Round 1** — uniqueness plan phases A-G (measured cutoffs, auto-coupling keyed on the
(centre, neighbours) pair, polymerisation constraints, integrality, bands = reference span ±2%,
bound-contact and integer-CN verdicts). The x4 global-minimum gate, latest code, seed 42:

| mineral | lattice reached | verdict |
| --- | --- | --- |
| quartz, spinel, wurtzite | within 3% | pass |
| anatase | +3.6 / −1.6 % | impostor, every term zero |
| corundum | −2.8 / +6.6 % | impostor, every term zero |
| forsterite | b −9.7 %, on a bound | impostor, scores below the published structure |
| topaz | −12.7 / +14.3 / +14.0 %, on bounds | impostor |
| diopside | b +5.6 %, β on a bound | impostor |
| adamite | +5.1 / +5.8 / +7.3 %, on bounds | impostor |

**Round 2** — a gray-band penalty: a neighbour inside cutoff ± 3 smoothing widths is penalised,
in the parent constraint's weight. It closes the hole integrality left open — an even number of
atoms parked at a cutoff gives an integral soft count (Si: four neighbours 0.0002 A outside the
cutoff, soft CN 4.000 at integer CN 2). Screened on the impostor regression set first, as agreed:

- it newly rejects one impostor (adamite, round 2);
- seven remain, with every one of their terms already zero or below the published structure:
  anatase ×2, corundum, diopside, forsterite ×2, adamite;
- topaz's latest impostor scores 1.93e-7 against the published 1.91e-7. The regression gate
  counts that as rejected (its threshold is a strict `>`), but a 1% difference at the noise
  floor separates nothing.

So the x4 gate was not re-run for round 2: the screen already shows the cap reached.

## What the impostors have in common

Polyhedral connectivity (corner / edge / face sharing between cation polyhedra), counted for every
impostor against its published structure:

- **same connectivity**: anatase (both), corundum (both), the latest adamite and forsterite — six
  of twelve. A connectivity constraint (Pauling's third rule) cannot reject them.
- **different connectivity**: diopside (SiO4 edge-sharing), topaz (AlO6–SiO4 edge-sharing), the
  earlier adamite and forsterite impostors.

Half the impostors are therefore the published topology, connected the published way, with every
first-shell quantity inside its band: bond lengths, angles, coordination numbers. Bond (±2%) and
angle (rounded outward to 5°) slack compound into a lattice more than 3% off. No first-shell term
can see this, because none of the first-shell quantities is wrong.

## Decision

- The gray band is kept, **cation centres in the objective**, and the version including anion
  centres is computed and **reported only** (`gray_band_report_all_centres` in `error_breakdown`,
  shown by `check` / `fit` / `run`). The references are all oxides, so whether anion shells should
  be held to the gray band cannot be decided yet: diopside's O shell has a 0.446 A gap, narrower
  than the 0.6 A band, so the published structure itself would be penalised (4.6e-4). Two
  impostors (adamite, diopside) are separated by the anion version and not by the cation one —
  the disagreement the review set as the trigger to reconsider which version is added.
- **Step 7 of `docs/plan-verification-and-tree-search.md` does not start.** Its precondition is a
  unique objective.
- The seven surviving impostors are pinned as `xfail` in the regression gate
  (`tests/fixtures/impostors/*_round2_seedfix.json` and earlier rounds).

## Ruled out in the design review (2026-10-01)

| idea | why not |
| --- | --- |
| narrower angle bands from the reference | new constraints must be citable facts (review Q2); narrower cited bands put references outside them (adamite Zn 78.6–125.9°) |
| O–O minimum distance | Shannon's 2 × 1.40 A rejects corundum (2.526 A); ideal-polyhedron edges reject anatase, corundum, adamite (shared edges shorten); the law-of-cosines bound adds nothing the bond and angle bands do not already imply |
| connectivity (second shell) | half the impostors share the published connectivity |

## Open

- Non-oxide references (a fluoride, a sulfide, and one with broad anion coordination) to decide
  the anion-centred gray band. Requested; CIFs to be supplied.
- What information *would* separate same-topology, same-connectivity impostors. Candidates not
  yet discussed: a volume or density target with a cited source, and second-refinement checks
  (ADR-0007's tautology) on corundum, anatase and forsterite.
