# The bond-graph topology class is the unit of success and of de-duplication

ADR-0005 to 0017 judged a search by whether the structure it reached had every lattice length
within 3 % of the published one, and called the zero-error points that missed it impostors.
mizuho's output is a starting structure for a DFT relaxation (README), so the question that
matters is different: does the point reached relax to the published structure?

## Measured: DFT relaxation of the impostors

Every recorded impostor with a reference in `_REFS` (adamite excluded: its H is not modelled),
relaxed against its published structure with identical settings — PBEsol, 520 eV, PAW PBE.64 as
in MPRelaxSet, KSPACING 0.25 /A, |F| <= 0.01 eV/A, full cell relaxation in the fixed space group
(`scripts/dft_basin/`, README there for the procedure and every deviation):

| impostor | lattice off before | topology class | after DFT relaxation |
| --- | --- | --- | --- |
| anatase ×3 | 3.6-4.9 % | same | the published structure |
| corundum ×2 | 8.8-11.6 % | same | the published structure |
| forsterite ×2 | 9.7-9.9 % | same | the published structure |
| quartz ×2 | 3.7-6.5 % | same | the published structure (at \|F\| <= 0.002 eV/A) |
| diopside ×3 (x8, x16, round 2) | 4.2-5.6 % | different | the published structure |
| topaz round 2 | 14.3 % | different | another minimum, +277 meV/atom, volume +29 % |

Same structure: same topology class, lattice within 0.5 % and 0.5 deg, every atom within
0.02 A after alignment, energy within 1 meV/atom.

1. **Every zero-error impostor is the published structure for a DFT starting point** —
   corundum's at c +11.6 % included. The one true impostor, topaz's, scores 6.8e-2; the
   objective already ranks it below the published structure.
2. **Same topology class implies the same relaxed structure** in all nine cases.
3. **A different class does not imply a different structure**: diopside's three relax back,
   by re-forming Ca/Mg-O bonds; topaz's, with SiO4 sharing edges, does not. No graph quantity
   separates them, so the class is a safe criterion for merging, not a complete one.

The sample is thirteen impostors, all oxides, relaxed in their space group.

## Decision

- **Success has two tiers** (design review Q15). Tier 1, the pass: the structure reached is in
  the published structure's topology class. Tier 2, the higher grade: lattice within 3 % and no
  bound touched (ADR-0006). The global-minimum gate asserts both separately.
- **The impostor regression gate is split** (Q20). A different-class impostor must still score
  above the published structure; a same-class one is not compared, and the nine DFT-checked ones
  are pinned to the published class. The four round-3/4 xfails (corundum, forsterite, quartz)
  are resolved: they are the published structure.
- **Endpoints are grouped by topology class during the search** (Q16, Q19, Q22-Q26,
  `mizuho/optimize/endpoints.py`). Every backend hands each Stage-1 end point to an
  EndpointLedger: the four restart backends each restart's end point, in restart order; MCMC each
  walker's lowest point per segment; Optuna each trial. MCMC and Optuna points are classified only
  at or below `zero_error_threshold`, and when none reaches it their 20 lowest are classified and
  marked not converged. An assignment's restarts stop once `early_stop_hits` endpoints at or
  below the threshold share one class; restarts run in waves of `restart_workers` and results
  after the stopping one are dropped, so the outcome does not depend on `restart_workers`.
  Without early stopping, or without a stop, results are bit-identical to before.

  ```yaml
  optimization:
    topology_grouping:
      zero_error_threshold: 1.0e-3   # ADR-0005's converged margin
      early_stop: true
      early_stop_hits: 2
  ```

  A config that states no cutoff has no bonds and so no classes; the ledger then does nothing.
- **Classes are refined and ranked** (Q24). `optimize_assignment` refines the best endpoint of
  up to `max_classes` classes; `run --top N` asks for N, and the global ranking keeps one row per
  class across assignments and space groups ("Same" counts the merged rows).
- **Gates run with early stopping, as in production** (Q27); `early_stop: false` re-measures a
  budget ladder.
- `mizuho compare` stays, as the display of `mizuho/bond_graph.py` (Q21).

## Measured: the x4 global-minimum gate on the two tiers (2026-10-08, seed 42, early stopping on)

| mineral | Stage 1 converged | tier 1 (class) | tier 2 (3 %, no bound) |
| --- | --- | --- | --- |
| anatase, fluorite, pyrite, sphalerite, spinel, wurtzite | yes | pass | pass |
| quartz | yes | pass | a −1.2 / c +3.7 % |
| corundum | yes | pass | c +11.6 %, on a bound |
| forsterite | yes | pass | a on a bound |
| diopside | 8.3e-3 | different class | on bounds |
| adamite | 1.0e-2 | different class | on a bound |
| topaz | 4.1e-2 | different class (Al CN 5) | on bounds |

Nine of twelve pass tier 1; six pass tier 2. Every tier-1 failure is the optimiser's: Stage 1
stopped above the published structure's error, as at x4 before (ADR-0017). The three shards ran
in 32-34 min each.

## Consequences

- ADR-0007's "the real structure must be the unique minimum" is satisfied at the level that
  matters: the objective's zero set lies in the published DFT basin wherever measured, except
  where it already scores a rival class worse.
- ADR-0017's conclusion — that no term can see a volume-preserving change of shape — stands, but
  for these minerals it no longer needs a term: DFT removes the change.
- The open problem moves from the objective to the class criterion: a merge rule that also joins
  diopside's impostors to the published class without joining topaz's would need information
  beyond the bond graph.

## Followed by ADR-0020

Tier 1 is the adoption measure of every step-7 comparison, with Stage-1 convergence beside it, and
a new mineral's first same-class zero-error point more than 3% off the published lattice is
relaxed by DFT before the class criterion is trusted for it.
