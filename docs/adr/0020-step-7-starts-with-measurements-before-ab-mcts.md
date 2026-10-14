# Step 7 starts, with three measurements before AB-MCTS

ADR-0014 held step 7 of `docs/plan-verification-and-tree-search.md` (accuracy benchmarks,
AB-MCTS) until the objective was unique. ADR-0019 moved success to the bond-graph topology class,
and on that criterion the precondition holds: wherever Stage 1 converged in the x4 gate, the point
reached is in the published class (nine of twelve minerals). The three failures (diopside, adamite,
topaz) are Stage 1 not converging. The ground for the hold is gone too: "strengthening the search
made the prediction worse" was measured as lattice deviation, and diopside's x8 and x16 points,
worse by that measure, relax to the published structure under DFT.

Design review of 2026-10-14.

## Decision

- **Step 7 starts at L2 only** (Phase 2 budget within a fixed space group). L1, the space-group
  search, waits for ADR-0021.
- **The precondition becomes:** tier 1 on every mineral whose Stage 1 converges.
- **Standing DFT spot-check.** The first time a mineral reaches a same-class, zero-error point
  whose lattice is more than 3% from the published one, that point is relaxed with
  `scripts/dft_basin/`. ADR-0019's evidence is thirteen oxide points relaxed in their own space
  group; this keeps it from being generalised silently.
- **Adoption measure for every step-7 comparison: tier 1**, reported beside Stage-1 convergence
  (`error <= error(published) + 1e-3`), for the reason ADR-0005 keeps its two stages apart. Tier 2
  is reported and not used for adoption: it measures the objective's slack, not the search. There
  is no per-mineral allow-list of classes DFT found equivalent; diopside's x8 and x16 points count
  as failures and are annotated as such. A false negative can only make adoption harder.

## Order of measurement

Each rule below is fixed before its measurement runs.

1. **Term ablation.** x4, seed 42, published assignment fixed; diopside, topaz, adamite. Four
   configurations: (i) compact support only (ADR-0015), (ii) + bond valence (ADR-0016),
   (iii) + volume (ADR-0017; the current code), (iv) the current code without the integrality
   penalty, (v) the current code without bond valence (added once (i)-(iii) showed volume alone
   rejects the impostor, so bond valence's removal had to be measured with volume in). Measured: Stage-1 convergence, tier 1, and the score of topaz's round-2 impostor, the
   one point DFT confirmed to be a different structure (ADR-0019).
   - A term whose removal keeps that impostor above the published structure **and** improves
     tier 1 is demoted to report-only. A term whose removal lets it through stays.
   - "Above" means by more than ADR-0005's converged margin: `error(impostor) >
     error(published) + 1e-3`. A strict `>` would count noise as rejection; without volume the
     impostor scores 4.9e-11 (compact support only) and 7.5e-4 (+ bond valence) against a
     published 0, both inside the margin within which the optimiser cannot tell two points apart.
   - Integrality is removed when (iv) matches (iii) on Stage-1 convergence and tier 1 (the
     condition ADR-0015 left pending).

   Why first: Stage 1 stopped converging on the large minerals between ADR-0014 and ADR-0016, a
   step that added compact support and bond valence together, against ADR-0005's one fix at a
   time. ADR-0015's own x4 run was stopped before the large minerals finished. ADR-0016 read the
   failure as the optimiser's by inference. If a term causes it, an optimiser measured against it
   is being measured against our own landscape.

2. **The CONT primitive alone**, on whichever of the three still fail Stage 1: independent restarts
   (the current method) against resuming the best point with a reduced step size, with and without
   the CMA-ES state carried over. Equal objective evaluations, five seeds, the published assignment
   fixed. The three failures are single-candidate convergence, which only AB-MCTS's CONT arm could
   address; whether it beats independent restarts has never been measured.
   - CONT wins: AB-MCTS-A is built at L2.
   - CONT does not win: AB-MCTS is dropped, and L2's problem becomes the per-candidate optimiser.

3. **Phase 1's feasibility solve** (ADR-0012's remedy, chosen there before measuring). In parallel
   with 1-2, since it measures the rank gate rather than the global-minimum gate. A budget ladder
   x1 / x2 / x4 on the local-shell solve finds the budget at which the two AsO4 xfails in
   `tests/unit/test_geometry.py` pass (since ADR-0009 the infeasible shell's 0.0359 sits inside the
   feasible one's 0.030-0.051; the tests already xpass on some runs). The five-seed rank spread of
   diopside and adamite is then measured at that budget. Phase 1 taking 2-4x longer is accepted.
   - Rejected: a Phase-1-only score (raw angles, a hard feasibility test). It would reintroduce the
     two objectives ADR-0009 removed.
   - Rejected as the remedy: drawing by free-parameter stratum. It lifts diopside, alone in its
     stratum, but not adamite, 147th of 202 within its own.

## Correction to the plan

The plan gives AB-MCTS's baseline as "top 50 plus successive halving, which today restarts every
round from scratch and discards its best point". Mode A has no successive halving:
`pipeline.optimise_candidates` gives each of the top `max_candidates` the same budget, and ADR-0019's
early stopping is its only adaptivity. Successive halving runs only in the space-group searches
(`search/spacegroup_search.py`, `search/spacegroup_tree.py`). The L2 baseline is a uniform budget
over the top 50 with early stopping.

## Consequences

- Adamite takes part in all three measurements: its 1937 reference, which would have kept it out
  of the adoption decisions, is replaced by ADR-0025.
- AB-MCTS's score transform is not settled here. The plan's `-log10(error + eps)` meets end points
  that are exactly zero since ADR-0015; the transform is designed only if measurement 2 adopts
  CONT.
- ADR-0008's per-Z report (as ADR-0022 restates it) and `suggest` (ADR-0023) come after these
  measurements; no gate depends on them.
