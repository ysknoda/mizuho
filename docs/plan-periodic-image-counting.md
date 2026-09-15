# Coding plan — periodic-image counting in the objective function

Decisions behind this plan: [ADR-0001](adr/0001-count-every-periodic-image.md),
[ADR-0002](adr/0002-no-compatibility-flag.md),
[ADR-0003](adr/0003-experimental-structure-is-the-gate.md),
[ADR-0004](adr/0004-cutoffs-measured-targets-cited.md).

## Problem

The objective's minimum is not at the real structure. Pre-change baseline
(backend `cma`, `normalize_penalties: true`, after the separate minimum-image angle fix):

| fixture | err_norm | searched lattice | literature | pinned at a bound | literature inside box |
|---|---|---|---|---|---|
| wollastonite_casio3 | 0.776406 | a=**3.50** b=3.43 c=3.43 | 7.925 / 7.32 / 7.065 (Z=6) | a | n/a (Z=1 model) |
| diopside_camgsi2o6 | 0.302825 | 9.83 / **9.40** / **5.60**, beta=**98.0** | 9.746 / 8.899 / 5.251, beta=105.79 | 3 of 4 | yes |
| adamite | 0.261599 | **8.80** / 8.10 / 6.13 | 8.306 / 8.524 / 6.043 | a | yes |
| spinel_mgal2o4 | 0.004053 | a=7.712 | 8.083 (−4.6%) | none | yes |
| topaz | 0.058538 | 8.97 / **9.60** / 4.30 | 8.80 / 8.39 / 4.65 (b +14%) | b | yes |

For every real-Z fixture the literature answer lies **inside** the search box, and the optimiser
goes to the box edge instead.

Root cause: the objective measures one periodic image per atom pair and never an atom against its
own images. Two confirmed consequences:

1. **Coordination numbers are undercounted** — NaCl 3 vs 6, CsCl 1 vs 8, fcc Cu 3 vs 12.
2. **The hard-sphere overlap term has no effect on cell collapse** — a single Na atom in a cubic
   cell scores an overlap penalty of 0.0000 all the way down to a = 0.2 A.

## Agreed approach

| Decision | Choice |
|---|---|
| Scope | All distance-based terms: overlap, bond, CN, repulsion, angle |
| CN semantics | Unbounded sum over all images (the crystallographic definition) |
| Self-images | Exclude the identity image only; other images of the same atom count |
| Compatibility | Unconditional replacement, no flag (ADR-0002) |
| Correctness gate | Objective approximately zero at the reference structure (ADR-0003) |
| Accuracy target | Searched lattice constants within 3% of literature |
| Parameter policy | `cutoff_distance` measured; `cn_min`/`cn_max` cited (ADR-0004) |
| Performance budget | Accept up to 2x `compute_error` cost; beyond that, optimise before proceeding |
| Sequencing | Incremental, so each change's effect stays attributable |

## Phases

### Phase 1 — overlap + CN

Introduce the neighbour-image primitive in `lattice.py` and move the overlap and coordination
terms onto it, mirrored in `objective/breakdown.py` (the module must keep agreeing with
`objective/energy.py`).

Validation: a new physics test asserting the objective reproduces the textbook coordination number
for NaCl (6), CsCl (8), CaF2 (8), diamond Si (4) and fcc Cu (12), and that the overlap term
penalises a cell collapsing onto its own periodic image. Then re-run the five fixtures and record
the change.

### Phase 2 — bond, repulsion, angle

Move the remaining distance-based terms onto the same primitive. The bond term's effective weight
rises, since more pairs now fall inside its cutoff; measure this rather than pre-compensating.

Validation: full suite, then the five fixtures again, with Phase 1's numbers as the comparison
point so the two effects stay separable.

### Phase 3 — reference structures

Write down the experimental structure for each of NaCl, CsCl, CaF2, diamond Si, fcc Cu (certain by
hand) and then adamite, spinel, topaz, diopside, wollastonite. **Each mineral is verified against
published bond lengths and coordination numbers before it is used for anything.** A first attempt
at spinel produced an Mg–O distance of 0.21 A from a wrong SG227 origin choice, so reconstruction
from memory is treated as unreliable by default and every value is reported with its source and a
confidence flag.

### Phase 4 — measure cutoffs, then evaluate the gate

Read each `cutoff_distance` off its reference structure (ADR-0004), update the fixtures, then
evaluate the ADR-0003 gate: objective approximately zero at the reference structure, with every
coordination term reporting the cited coordination number.

### Phase 5 — re-run and report

Re-run the five fixtures. Report against all three measures: the gate, whether any parameter is
still pinned at a bound, and the percentage error in the searched lattice constants.

## Outcome

All five phases are implemented. Fast suite 834 passed; `compute_error` costs 1.12-1.91x its
former time, inside the agreed 2x budget.

### The physics gate (phase 1)

`tests/unit/test_periodic_image_counting.py` asserts the objective reproduces each structure's
textbook coordination number:

| structure | true CN | before | after |
|---|---|---|---|
| NaCl | 6 | 3 | 6 |
| CsCl | 8 | 1 | 8 |
| fcc Cu | 12 | 3 | 12 |
| CaF2 | 8 | 8 | 8 |
| diamond Si | 4 | 4 | 4 |

It also pins the overlap term's new ability to see an atom's own images: a lone Na in a cubic cell
used to score exactly 0.0 down to a = 0.2 A.

### The ADR-0003 gate (phases 3-4)

`tests/integration/test_reference_structure_gate.py` holds the four reference structures, verifies
each against its mineral's published bond lengths before use, and asserts the objective is
approximately zero there:

| fixture | objective at its reference structure |
|---|---|
| topaz | 0.000003 |
| diopside_camgsi2o6 | 0.000023 |
| spinel_mgal2o4 | 0.000055 |
| adamite | 0.0292, plus 80.00 that is the CIF's undetermined hydrogen |

Reaching this took three separate corrections, only the first of which was the original diagnosis:

1. Counting every periodic image (ADR-0001).
2. Measuring coordination cutoffs, which had been too tight for the logistic soft count to
   saturate. Spinel's Mg-O reported 3.78 where the structure has 4.
3. Measuring bond cutoffs, which had been too loose. Diopside's Ca-O default of
   `2.85 * 1.5 = 4.275 A` swept in the whole second shell and contributed **12.13 of a 12.13
   total** at the experimental structure.

A fourth, the same too-tight cutoff in the anion constraints `augment_coupling_constraints`
injects, was found in topaz and fixed by stating those constraints explicitly.

### Two claims that verification overturned

Both were caught by checking a structure against its own published geometry rather than trusting a
reconstruction, which is why ADR-0003 makes that check mandatory:

- A first spinel reconstruction gave an **Mg-O distance of 0.21 A**: the SG227 origin choice was
  wrong. Identifying the assignment by which one reproduces the published bond lengths found
  Mg@a / Al@d / O@e with u = 0.3870, i.e. the published origin-choice-2 value plus 1/8.
- SG58 was reported here as missing the ITA position 4h (x, y, 1/2), which would have meant mizuho
  could not express adamite at all. **That was wrong.** Pnnm's n-glide `1/2+x, 1/2-y, 1/2-z` maps
  the z = 0 plane onto z = 1/2, so a single orbit contains both and letter `g` covers them. Adamite
  builds correctly as As@g, Zn@g, Zn@f, O@g x3, O@h, and the table needs no change.

### Fixture results

Both gates pass. End-to-end accuracy did **not** follow: searched against the full pipeline
(`normalize_penalties: true`, as every search run in this project uses), only spinel reaches the 3%
lattice target.

| fixture | free params | searched lattice vs literature | 3% target |
|---|---|---|---|
| spinel_mgal2o4 | 2 | a +0.3% | met |
| diopside_camgsi2o6 | 18 | a +2.1% b +5.6% c +6.6% beta -6.7% | missed |
| adamite | 17 | a +5.9% b +3.8% c -0.7% | missed |
| topaz | 18 | a -13.6% b +14.3% c +14.0% | missed |

The cause is **not** this plan's subject and not the optimiser. Handing the optimiser the published
topology at quadruple budget reaches error 0.000000 on diopside -- at a lattice 5-7% away from the
published one. The objective is minimised at the published structure but is not minimised *only*
there. That is taken up in `docs/plan-objective-uniqueness.md` and ADR-0007.

One inconsistency surfaced while measuring this and is carried into that plan: the numbers in this
section were produced with `normalize_penalties: true`, while the ADR-0003 gate above runs at the
fixtures' own setting, which is `false`. The two objectives are not the same.

## Risks


- **Not yet established that the gate is reachable.** Phases 1–2 remove a known error floor; they
  do not prove the remaining terms are consistent at the real structure. Phase 4 is the first point
  where that is actually tested, and it may expose further mis-specification.
- **Reference-structure reconstruction is the weakest link**, already demonstrated once. Treated as
  untrusted until each structure passes its own bond-length check.
- **Fixture targets may prove unreachable** even with correct counting, for example if a chosen
  Wyckoff topology cannot host the cited coordination. Per ADR-0004 that is reported as a finding,
  not fixed by moving the target.
- **`wollastonite_casio3` stays at Z=1.** Its lattice constants are not comparable to the real Z=6
  mineral; it validates the mechanism (can a structure satisfying every constraint be found?), and
  is excluded from the 3% accuracy measure.
