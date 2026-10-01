# Coding plan — verification in untested space groups, tree-search verification, AB-MCTS

Decisions behind this plan: [ADR-0009](adr/0009-always-normalise-penalties.md),
[ADR-0010](adr/0010-success-is-the-detected-symmetry.md),
[ADR-0011](adr/0011-equivalent-assignments-via-the-euclidean-normalizer.md),
[ADR-0012](adr/0012-phase-1-is-seeded-and-ranked-over-several-seeds.md),
[ADR-0013](adr/0013-cli-subcommands.md).

Runs alongside [plan-objective-uniqueness.md](plan-objective-uniqueness.md). Functional
verification (can a structure be built correctly; does a search return a known answer) does not
depend on the objective being unique, so it goes first. Anything that measures **accuracy** or
**strengthens the search** waits for the uniqueness plan: strengthening the search before the
objective is unique was measured to make predictions worse (diopside, x1 → x4 budget, error
0.019499 → 0.000000 while the lattice moved further from the published one).

## What was found

- **Hexagonal and trigonal space groups are untested.** Fixtures cover 2, 15, 58, 62, 136, 221,
  225, 227 — five of seven crystal systems. The Wyckoff database itself is validated against
  spglib for all 230 groups (1,731 entries); what is untested is the optimisation path — hexagonal
  lattice parametrisation, R lattices, floating origins.
- **Three tree searches exist, and the default space-group search has no known-answer test.** The
  hierarchical PUCT search (`search/spacegroup_tree.py`) is the default for automatic space-group
  search. Its only end-to-end test asserts `1 <= sg <= 230`; the CLI has no successful auto-mode
  test at all.
- **The PUCT search is structurally limited.** The tree is fixed at 230 leaves; each space group
  optimises only its single best-by-surrogate assignment; a confirmed leaf is never re-evaluated;
  confirmation costs 3 restarts x 20 iterations (about 1/40 of the budget already shown too small
  at 18 dimensions); and its reward mixes surrogate scores and real errors through one
  `1/(1+x)`.
- **Phase 1 is unseeded** — ADR-0012.
- **The surrogate systematically ranks high-free-parameter candidates low.** Grouping candidates by
  coordinate free-parameter count (one unseeded run each):

  | fixture | overall rank | free params | stratum size | rank within stratum | strata |
  |---|---|---|---|---|---|
  | spinel | 1 | 1 | 10 | 1 | 2 |
  | topaz | 26 | 15 | 23 | 2 | 18 |
  | diopside | 163 | 14 | 1 | 1 | 12 |
  | adamite | 9745 | 15 | 202 | 147 | 17 |

  In adamite the strata with 3-6 free parameters hold the top of the ranking and no stratum with 9
  or more reaches higher than rank 169.
- No floating-origin handling, no runtime symmetry detection, no awareness of enantiomorph pairs.

## Reference structures for the new space groups

Every file was verified before use (ADR-0003): compound, space group, Z, stated against recomputed
volume, first-shell bond lengths and coordination numbers, and the Wyckoff letters spglib assigns in
**mizuho's** setting (the first Hall number of each group).

| file | source | role | in mizuho's setting | cation–O, CN | first/second shell gap → cutoff (A) |
|---|---|---|---|---|---|
| `corundum_H.cif` | COD 1000017 | **reference**, R-3c (167), R lattice on hexagonal axes | Al@c O@e | Al 6, 1.855–1.972 | 1.972 → 3.222, **2.597** |
| `corundum_R.cif` | COD 1010914 | CIF reader: rhombohedral → hexagonal axes | Al@c O@e | Al 6, 1.843–1.982 | 2.590 |
| `quartz_sg152.cif` | COD 1011097 | **reference**, P3₁21 (152), trigonal P + screw axis | Si@a O@c | Si 4, 1.614–1.624 | 1.624 → 3.488, **2.556** |
| `quartz_sg154.cif` | COD 1011159 | **not used** — Si–O spans 0.068 A against 0.010 A in the 152 file | Si@a O@c | Si 4, 1.576–1.644 | — |
| `anatase_origin2.cif` | COD 1526931 | **reference**, I4₁/amd (141), I lattice + origin choice 2 → mizuho's origin choice 1 | Ti@b O@e | Ti 6, 1.924–1.975 | 1.975 → 3.824, **2.899** |
| `anatase_origin1.cif` | COD 1010942 | ADR-0011 real-data case: same mineral, other representative | Ti@a O@e | Ti 6, 1.874–1.923 | 2.889 |
| `forsterite_Pbnm.cif` | COD 9016385 | **reference**, Pnma (62) supplied in the Pbnm setting | Mg@a Mg@c Si@c O@c x2 O@d | Mg 6, 2.045–2.210; Si 4, 1.614–1.655 | Mg **2.819**, Si **2.397** |
| `forsterite_Pnma.cif` | COD 9015658 | setting-independence cross-check | same | Mg 6, 2.067–2.210; Si 4, 1.601–1.628 | Mg 2.815, Si 2.391 |
| `wurtzite.cif` | COD 1577381 | **reference**, P6₃mc (186), hexagonal + polar floating origin | Zn@b O@b | Zn 4, 1.975 ×3 / 1.991 | 1.991 → 3.218, **2.604** |

- **Wurtzite.** `wurtzite.cif` (COD 1577381, Managave et al., *Ceramics International* 51, 2025)
  has u = 0.3823(10), a = 3.251 A, c = 5.209 A, Zn–O 1.975 A ×3 and 1.991 A. It is a sol-gel
  nanoparticle sample, probably a powder refinement, but its lattice matches bulk ZnO within 0.05%,
  so no size-induced strain shows. Its deposited volume (47.598 A³) disagrees with its own a and c
  by 0.17%; COD's recomputed 47.678 A³ is consistent and is the one used. A reference structure must
  be experimental (`CONTEXT.md`): a DFT-relaxed structure would put the functional's lattice error
  into the fixture's measured cutoffs and bands (ADR-0004).
- **Published CIFs round special positions.** `wurtzite.cif` writes 1/3 and 2/3 as 0.3333 and
  0.6667. Built as written, spglib finds only Cmc2₁ (36, an orthorhombic subgroup) at 1e-5 and
  1e-4 A, and P6₃mc from 1e-3 A. The CIF reader must therefore idealise special positions before
  use, or ADR-0010's tolerance ladder will report a spurious subgroup.
- Each secondary file is a separate refinement, not the same data in another setting. A reader test
  therefore asserts the same Wyckoff assignment (up to ADR-0011 equivalence) and lattice and bond
  lengths that agree within the difference between refinements — not identity.
- The enantiomorph test does not use `quartz_sg154.cif`. It inverts the 152 structure
  (x, y, z → −x, −y, −z maps P3₁21 onto P3₂21), which isolates handedness from refinement noise.
- Cutoffs above follow ADR-0004 (midpoint of the first/second shell gap). Coordination numbers are
  cited facts: Al 6 (corundum), Si 4 (quartz, forsterite), Ti 6 (anatase), Mg 6 (forsterite),
  Zn 4 (wurtzite). Bands follow ADR-0004's amendment. Constraints are written by hand; generating
  them automatically (`suggest`) waits for the uniqueness plan.

## Steps

Ordered by dependency. Gates are re-run between steps so every effect is a measured difference.

### Step 0 — close the unverified parts first

- Code review (code-check) of the tree-search modules never reviewed: `search/base.py`,
  `search/dfs.py`, `search/best_first.py`, `geometry/local_shell.py` (1010 lines),
  `optimize/budget_allocator.py`, `optimize/cma.py`. The unseeded Phase 1 is the kind of defect
  this should catch before anything is built on it.
- Run the 39 `slow` tests once and record the result as the baseline. They were not run for the
  0.1.0a1 release.
- Install-test the light extras `bayesian`, `mcmc`, `numba` in a throwaway environment. `gp` is left
  out (it pulls in torch).

### Step 1 — always normalise (ADR-0009)

First, because every baseline recorded after this is in normalised units. Remove the key from
`energy.py`, `breakdown.py`, `energy_jax.py`, reject it in configs, update the tests whose premise
was the switch, rewrite README's "Penalty normalisation" section.

### Step 2 — seed Phase 1, require spglib (ADR-0012, ADR-0010)

Pass `optimization.seed` through `search_and_rank`. Move `spglib` from the `test` extra into
`dependencies`.

### Step 3 — CIF reader, `check`, reference gates for the new minerals

- CIF reader, written in-house: cell, symmetry operations (both `_symmetry_equiv_pos_as_xyz` and
  `_space_group_symop_operation_xyz`), the atom-site loop, standard uncertainties in parentheses,
  quoted values. Setting conversion and Wyckoff letters come from spglib's dataset at mizuho's Hall
  number, **with special positions idealised** (spglib standardisation) because published CIFs
  round them — see wurtzite above. Tested on every supplied CIF; the pair tests above cover
  rhombohedral axes, origin choice and axis permutation, and wurtzite covers rounded special
  positions.
- `mizuho check` (ADR-0013).
- Fixtures and `_REFS` entries for corundum, quartz, anatase and forsterite.
- Two independent oracles per mineral: **geometric** — the built structure, handed to spglib,
  returns the same space group and Wyckoff letters; **chemical** — the objective is approximately
  zero at the published structure (ADR-0003).
- Enantiomorph test on quartz: the 152 structure and its inversion score identically.

### Step 4 — equivalent assignments, rank gate over seeds, `candidates`

- Euclidean normalizer (ADR-0011), validated three ways.
- Merge equivalent assignments before ranking; measure the reduction per fixture.
- Rank gate: equivalence-class identity, worst rank over five seeds, spread recorded (ADR-0012).
- `mizuho candidates`.

### Step 5 — remaining functional checks and commands

- Floating origin: wurtzite with and without pinning the polar coordinate.
- Detection tolerance for ADR-0010: measure the optimiser's coordinate noise on reference-structure
  runs; set the pass threshold from it.
- PUCT known-answer tests, judged on detected symmetry (ADR-0010): NaCl → 225 and CsCl → 221
  strictly; Si → {227, 194}, ZnO → {186, 216}, TiO₂ → {136, 141} as polytype families.
- `mizuho fit`, `mizuho wyckoff`, the move to subcommands.

### Step 6 — the uniqueness plan

Phases A-H of `plan-objective-uniqueness.md`. Its Phase 0 is resolved by ADR-0009. Corundum, quartz,
anatase and forsterite join its fixtures.

### Step 7 — after the objective is unique

- Accuracy benchmarks on the new minerals: the global-minimum gate and the 3% lattice target.
- AB-MCTS, below.

## AB-MCTS — a hypothesis, not a decision

Source: Inoue, Misaki, Imajuku, Kuroki, Nakamura, Akiba, "Wider or Deeper? Scaling LLM
Inference-Time Compute with Adaptive Branching Tree Search", Sakana AI, arXiv:2503.04412v5 (local
copy in `references/`, gitignored).

At every node, a GEN child stands for "generate a new child here" (go wider) beside the existing
children (go deeper); the choice between them is made by Thompson sampling on Bayesian posterior
predictives of the score, because UCT does not apply when arms appear during the search (§3.2).
AB-MCTS-M fits a hierarchical mixed model by MCMC at every selection (PyMC); AB-MCTS-A gathers the
children under a CONT node and uses conjugate priors — Gaussian with normal-inverse-χ² (m = 0,
κ = 1, ν = 1, τ² = 0.1) or Beta (α = β = 0.5) — and so updates analytically (§3.3–3.4, App. A.4,
B.2). The authors' own stated limitation is that the method assumes a reliable score evaluator
(§5), which is precisely what ADR-0007 found mizuho lacks today. That is why this waits for step 6.

### Mapping chosen

| | GEN — go wider | CONT — go deeper |
|---|---|---|
| **L2**, Phase 2 budget in a fixed space group (first) | draw the next candidate from Phase 1's ranking, **within a free-parameter stratum** | resume the best candidate: CMA-ES restarted at its best point with a smaller step size |
| L1, space-group search (after L2) | try a new space group | spend more on a space group already started, through L2 |

- **L2 first.** Phase 1's ranking behaves as an unbounded generator of new answers (69,974 for
  adamite), and continuing a started candidate is refinement with feedback — the paper's setting.
  L1's existing priors per crystal system have no natural place in AB-MCTS's selection rule, and
  L1's leaf evaluation is itself an L2 problem.
- **Strata.** Drawing Phase 1's list in order would need 2950-9745 expansions to reach adamite's
  published topology, so plain AB-MCTS would not reach it. The root's children are strata of equal
  free-parameter count; Thompson sampling learns which stratum yields low errors, and GEN draws the
  next candidate within it. Adopted only if the five-seed measurement shows within-stratum ranks are
  stable.
- **AB-MCTS-A, Gaussian, written in numpy** from equations (12)–(17). AB-MCTS-M tends to perform
  better (App. C.5) but needs PyMC; it is kept as a comparison if A disappoints. TreeQuest is not
  used as a dependency.
- **Score.** `1/(1+error)` is flat near zero, where the published structure and an impostor differ.
  The score is a standardised log error, e.g. `clip(−log10(error + ε) / k, 0, 1)`, with ε and k set
  from the measured error distribution.
- **Budget unit: objective evaluations.** One node is a fixed number of evaluations, so a node
  costs the same at 2 and at 18 dimensions — the paper counts nodes, and mizuho's costs per
  candidate otherwise differ by orders of magnitude.

### Measurement and adoption

Baseline: top 50 plus successive halving, which today restarts every round from scratch and
discards its best point. Compared at equal total objective evaluations, five seeds, on the
fixtures whose objective is unique by then. The measure is whether the published structure is
reached. Adopted at L2 if it reaches it where the baseline does not, or reaches it with fewer
evaluations; L1 is then built on L2.

## Decided by measurement

| question | measured by | decided how |
|---|---|---|
| remove the floating origin | wurtzite, pinned vs free | remove it via the normalizer's continuous part if convergence differs |
| detection tolerance | optimiser coordinate noise on reference runs | noise plus margin |
| strengthen Phase 1's feasibility solve | rank spread over five seeds | more restarts if the spread is large |
| stratified GEN | within-stratum rank over five seeds | adopt if stable |
| score transform ε, k | error distribution | set from it |
| adopt AB-MCTS | baseline comparison above | reaches the published structure more often or sooner |

## Out of scope

- `suggest` (constraint generation): after the uniqueness plan.
- A bond-valence-sum term: held until a third impostor mechanism appears (ADR-0005's two-round
  exit).
- Checkpoint and resume; Z enumeration (ADR-0008); removing the flat space-group search: separate
  small pieces of work.
- Second-shell constraints to separate polytypes: a future feature (ADR-0010).

## Status (2026-10-01)

Steps 0-5 and step 6's phases A-G are implemented. Step 7 is **not started**: the
objective is not unique (below), which is its precondition.

| step | state |
|---|---|
| 0 | slow baseline at 0.1.0a2: 38/39 (`test_plateau_patience...`, 1e-6 gap, pre-existing). Extras `bayesian`, `mcmc`, `numba` install and pass in a clean venv. Review of the tree-search modules found and fixed: wall-clock-dependent Phase 1 shortlist (now counted evaluations x a fitted cost model), pycma seed 0 = clock seed, `seed + restart` stream sharing across consecutive seeds (now `restart_seed` via SeedSequence) |
| 1 | `normalize_penalties` removed; rejected with an error naming ADR-0009 |
| 2 | Phase 1 seeded from `optimization.seed`; spglib required |
| 3 | `mizuho/io/cif.py`, `mizuho/symmetry.py`; all 12 CIFs map to the expected assignment (wurtzite and quartz idealised at 1e-3 A); fixtures and `_REFS` for corundum, quartz, anatase, forsterite, wurtzite; enantiomorph test |
| 4 | `mizuho/normalizer.py` (indices P-1 8, Pnma 8, Fd-3m 2; all 230 groups); merge reduction: adamite 69,974 -> 19,152, diopside 39,650 -> 11,230, topaz 3,234 -> 1,644, wollastonite 8,192 -> 1,024; `candidates` |
| 5 | floating origin kept (pinned vs free: same convergence, 10 seeds); success tolerance 1e-3 A (optimiser noise <= 1e-9 A, band-slack distortions 1e-2-1e-1 A); supergroup promotion in `run`; `fit`, `wyckoff`, `suggest` reserved |
| 6 | phases A-G implemented; wollastonite reference 6.68 -> 1.4e-5 |

### Gates after step 6 (latest code, x4 budget, seed 42; rank over seeds 42-46)

| mineral | rank (worst of 5) | global minimum: lattice reached | verdict |
|---|---|---|---|
| quartz | 2 | +0.9 / -0.2 % | pass |
| spinel | 1 | +0.6 % | pass |
| wurtzite | 2 | -0.7 / -2.3 % | pass |
| anatase | 1 | +3.6 / -1.6 % | impostor |
| corundum | 1 | -2.8 / +6.6 % | impostor |
| forsterite | 14 | b -9.7 %, on bound | impostor |
| topaz | 42 | -12.7 / +14.3 / +14.0 %, on bounds | impostor |
| diopside | **103** (8-103) | b +5.6 %, beta on bound | impostor, rank fail |
| adamite | **4036** (1553-4036) | +5.1 / +5.8 / +7.3 %, on bounds | impostor, rank xfail |

Automatic space-group search, judged on detected symmetry at 1e-3 A: NaCl -> 99,
CsCl -> 51, Si -> 123, ZnO -> 123, TiO2 -> 47. None reaches its known answer.

### Design review and round 2 (2026-10-01)

Gray-band penalty added (cation centres in the objective, all centres reported); it rejects one
more impostor and leaves seven. ADR-0005's two-round cap is reached:
[ADR-0014](adr/0014-first-shell-constraints-cannot-make-the-objective-unique.md).

### Findings that need a design decision

1. Six of nine minerals still have impostors after phases A-G: the second round of
   ADR-0005's two-round cap. The design, not the parameters, needs revisiting.
2. Integrality (ADR-0007) is defeated by an EVEN number of atoms parked at the cutoff:
   Si with four neighbours 0.0002 A outside it scores soft CN 4.000 at integer CN 2.
3. Since ADR-0009, Phase 1's local-shell score no longer separates the documented
   feasible and infeasible AsO4 shells (two geometry tests xfail).
4. Bond (2%) and angle (5 deg rounding) slack compound: anatase's impostor sits at zero
   with every term satisfied.

