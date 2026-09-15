# Coding plan — the published structure as the objective's unique minimum

Decisions behind this plan: [ADR-0005](adr/0005-five-gates-for-the-search.md),
[ADR-0006](adr/0006-a-solution-on-a-lattice-bound-is-a-failure.md),
[ADR-0007](adr/0007-the-real-structure-must-be-the-unique-minimum.md),
[ADR-0008](adr/0008-enumerate-per-formula-unit-count.md), plus the amendment to
[ADR-0004](adr/0004-cutoffs-measured-targets-cited.md).

Successor to [plan-periodic-image-counting.md](plan-periodic-image-counting.md), which fixed the
objective's periodic-image counting and its cutoffs. Those gates pass; the 3% lattice target is
still missed on three of four minerals.

## Problem

The objective is approximately zero at each published structure **and also at a different structure
whose lattice is 5-14% away**. Measured by handing the optimiser the published topology at x4
budget (`n_restarts` and `cma_maxiter` both x4):

| fixture | free params | error found | error at published | lattice deviation |
|---|---|---|---|---|
| diopside | 18 | **0.00000000** | 0.00000066 | −5.6 / +4.0 / +6.6 / +6.8 % |
| topaz | 18 | **0.00000000** | 0.00000013 | −13.6 / +14.3 / +14.0 % |
| adamite | 17 | **0.00564958** | 0.00805674 | +5.9 / +5.8 / −5.3 % |

(`normalize_penalties: true`; diopside and topaz hold under `false` too, adamite does not — see
phase 0 below.)

The optimiser is not the bottleneck. The budget ladder on diopside's published topology:

| budget | error | lattice |
|---|---|---|
| x1 (8 restarts x 300 iter, 153 s) | 0.019499 | +2.1 / +5.6 / +6.6 / −6.7 % |
| x4 (32 x 1200, 2128 s) | **0.000000** | −5.6 / +4.0 / +6.6 / +6.8 % |

**The error fell to zero and the lattice got worse.** More compute arrives at the wrong answer
faster.

### Two mechanisms

1. **The soft coordination count is gameable.** Diopside's impostor parks two oxygens 0.001 A
   outside the 3.095 A Ca cutoff, each worth `expit(-0.01) = 0.4975`, and passes `cn_min: 7,
   cn_max: 8` with a hard coordination of 6 (soft 7.187). Adamite does the same on Zn.
2. **Bond bands are 15-33x wider than the chemistry.** Topaz's Si-O spans 0.006 A at its published
   structure against a configured band of 0.20 A. Its impostor distorts every polyhedron to the band
   edges and buys a 12.7% larger cell for free. Its coordination numbers are exactly integral, so
   mechanism 1's fix does not touch it.

### Secondary failures, already measured

- **Ranking.** The published topology's rank in Phase 1: spinel 2/22, diopside 25/39,650, topaz
  50/3,234, **adamite 2950/69,974**. Only adamite fails the `N = max_candidates = 50` gate. Topaz
  passes with zero margin and only because that fixture sets `feasibility_shortlist: 60`.
- **Wollastonite has never been gated.** Its reference structure (COD 9005777, Ohashi 1984, P-1,
  Z = 6, V = 396.960 A^3 reproduced exactly) scores **6.683659** — four constraint values are wrong,
  see phase A.

## Gate before fix

Five gates (ADR-0005), built and run first, with the failing baseline recorded. No fix starts until
the baseline exists.

| Gate | File | Cost |
|---|---|---|
| Reference structure (existing) | `tests/integration/test_reference_structure_gate.py` | one `compute_error` |
| Impostor regression (new) | same file | two `compute_error` |
| Rank (new) | `tests/integration/test_search_gates.py` | 81-1152 s per mineral |
| Global minimum, two-stage (new) | same file | ~35 min per mineral |
| End-to-end proxy (new) | same file | ~3.5 h per mineral |

Known failures are written at the correct threshold and marked `xfail`: adamite's rank (2950), and
whichever impostor gates fail before their fix lands.

## Phases

Ordered by dependency, **not** by expected impact, so each effect is a measured difference rather
than an inference. Gates are re-run between phases.

### Phase 0 — settle `normalize_penalties`

Every search run in this project uses `true`; the fixtures set it nowhere and so default to `false`,
which is what the ADR-0003 gate measures. Adamite's impostor exists under `true` and not under
`false`, so the two objectives genuinely disagree about which structure is better. **This is decided
before anything else**, because every gate below measures whichever one we pick. Blocked on a
decision; see "Open question".

### Phase A — ADR-0004 cutoff corrections

No dependencies.

| fixture | key | from | to | measured |
|---|---|---|---|---|
| wollastonite | `bond` Ca-O `cutoff` | 3.90 (auto) | **2.945** | 2.639 -> 3.251 |
| wollastonite | `cn` Si(O) `cutoff_distance` | 1.85 | **2.580** | 1.664 -> 3.497 |
| wollastonite | `cn` Ca(O) `cutoff_distance` | 2.75 | **2.945** | as above |
| wollastonite | `angle` O-Si-O band | [100, 120] | **[100, 125]** | 102.7-122.4 deg |
| wollastonite | `bond` Ca-O band | [2.30, 2.60] | **[2.25, 2.65]** | 2.266-2.639 |
| spinel | auto O(X) `cutoff` | 2.050 | **2.630** | 1.929 -> 3.332 |
| diopside | auto O(X) `cutoff` | 2.850 | **2.942** | 2.719 -> 3.165 |

Also add wollastonite to `_REFS` (reference-structure gate only — ADR-0005; its 51 free parameters
put the rank and global-minimum gates out of reach, and that exclusion is stated in the test).

Expected: wollastonite's reference score falls from 6.683659 towards zero; spinel's O(X) soft count
goes 3.102 -> 3.999.

### Phase B — auto-injection trigger and cutoff

`augment_coupling_constraints` currently skips an anion that appears as **any** `center_atom`, in
either `coordination_constraints` or `angle_constraints`. Writing one explicit anion coordination
constraint therefore silently deletes both the auto `O(X)` coordination constraint *and* the auto
`X-O-X` angle constraint. Change the test to the `(center, neighbor)` pair, and add
`auto_coupling_cutoff` so a fixture can supply an ADR-0004-measured cutoff instead of
`_max_bond_distance_for`'s systematically too-tight one.

Prerequisite for phase C. Adamite already carries an explicit `O(X)`, so it gains an auto
`X-O-X` angle constraint here — which is why the cutoff key must land in the same phase.

### Phase C — anion polymerisation constraints

Add `center_atom: <anion>, neighbor_atom: <cation>` to all five fixtures, with ADR-0004 cutoffs:

| fixture | constraint | `cn_min` | `cn_max` | measured at the reference |
|---|---|---|---|---|
| diopside | O(Si) | 1 | 2 | {1: 16, 2: 8} — single chain |
| wollastonite | O(Si) | 1 | 2 | {1: 12, 2: 6} — single chain |
| topaz | O(Si) | 1 | 1 | {1: 16} — isolated tetrahedra |
| adamite | O(As) | 0 | 1 | {0: 4, 1: 16} — isolated AsO4 + hydroxyl |
| spinel | O(Mg), O(Al) | cited | cited | control: already satisfied |

The per-atom band looks loose and is not: combined with the composition and the cation coordination
number it forces the distribution exactly (diopside, 8 Si x CN 4 = 32 bonds over 24 O in [1, 2],
gives 16 and 8 uniquely). **This does not restore uniqueness** — every impostor reproduces its
mineral's distribution — and ADR-0007 says so explicitly. It is added because it is a real chemical
fact no fixture states, and its effect on Phase 1 ranking is measured here via the rank gate.

Spinel is included as a control: if adding an already-satisfied constraint breaks it, the constraint
or its cutoff is wrong.

### Phase D — integrality penalty

Depends on phase A (wollastonite's cation cutoffs must be right first).

For every **cation-centred** coordination constraint, penalise `|CN_soft − round(CN_soft)|` outside
a dead band of `[0, 0.05]`, weight inherited from the parent constraint. Mirrored in
`objective/breakdown.py` and `objective/energy_jax.py`.

Reference values for the ceiling: worst deviation across the four minerals is 0.0179 (adamite Zn);
the impostors sit at 0.1865 (diopside Ca) and 0.4773 (adamite Zn). Anion-centred constraints are
exempt — diopside's O deviates 0.127 at its own published structure.

### Phase E — bond band widths

Independent. Every bond band becomes the span measured at the reference, widened by 2% of the bond
length and rounded outward. Rejects topaz's impostor on all three bonds and diopside's on Mg-O and
Ca-O.

### Phase F — lattice-bound assertion

Independent. Hard failure in the gates, warning in the report. Lattice parameters plus any free
coordinate whose `param_bounds` were narrowed below `[0, 1]`. Tolerance 1% of the interval width.

### Phase G — integer coordination verdict

Independent. Gates and the production report judge by the integer coordination number and flag
disagreement with the soft one.

### Phase H — ADR-0008 only

Record the Z decision; no implementation this round.

## Completion

Not "all gates green" alone. Changing the objective changes its landscape.

1. All five gates green except recorded `xfail`s.
2. Re-run the x4 probes on diopside, topaz and adamite; every reached lattice within 3%.
3. Any newly found impostor is added to the regression gate and the cycle repeats — **capped at two
   rounds**. Anything still standing is recorded as `xfail` with its parameter vector, which
   constraints it satisfied, and which measure could separate it.

A third round means a third mechanism exists, and the design rather than the parameters needs
revisiting.

## Open question

**Phase 0 blocks everything.** `normalize_penalties` is `true` in every search run and `false` in
every fixture and therefore in the ADR-0003 gate. Adamite's impostor exists under one and not the
other. Every threshold in ADR-0007 was deliberately chosen against geometry (soft coordination
numbers, bond distances) rather than against error values, so none of them move with this choice —
but the gates cannot be trusted while they score a different objective from the one being optimised.

## Out of scope

- Optimiser budget and backend. The ladder shows 18 dimensions are solvable at x4; more optimiser
  strength currently makes the prediction worse, not better.
- Phase 1 ranking. Only adamite fails the rank gate, and phase C may move it; that is measured
  before anything is designed for it.
- ADR-0008's implementation.
