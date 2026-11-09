# Plan: stop hopeless Phase-2 restarts early

Design review of 2026-11-02/03 (grilling session). Phase 2 stops a Wyckoff assignment's restarts
early only on success: ADR-0019's endpoint ledger skips the remaining restarts once two
zero-error endpoints share a topology class. Nothing stops a restart that is not getting
anywhere, and `plateau_patience` (total error only) is off in Phase 2 by default.

## Measured before the design (Al-O-N record, runs/alon)

- 38,975 record rows, which are topology classes; 17,843 distinct assignments. The bond-count
  pruning of ADR-0037 rejects 5,656 assignments and none with a zero-error class.
- 300 non-zero endpoints, sampled: the coordination term is 63 % of the error on average and over
  half of it in 68 %; 267 of 300 end with some centre's integer coordination number outside its
  range (median error 0.19, against 0.019 for the 33 that do not).

## Decisions

1. **Target**: throughput of composition-free exhaustive searches. The twelve-mineral gates are a
   no-regression condition.
2. **Bond-count pruning first** (ADR-0037, merged). Early stopping is designed and measured on the
   assignments that survive it.
3. **Unit**: one CMA-ES restart, stopped between generations. Skipping later restarts from the
   outcome of earlier ones is not implemented (restarts start from deliberately different points)
   but is evaluated in the replay; reallocating budget between candidates is out of scope
   (ADR-0020's L2).
4. **Strong constraints**: integer coordination numbers of cation and anion centres. The
   coordination deficit of a point is
   D = sum over every centre of every coordination constraint of max(0, cn_min - n, n - cn_max),
   n the number of neighbours closer than the cutoff, every periodic image counted
   (`compute_error(..., terms=...)["cn_deficit"]`).
5. **Stop condition**: a restart whose best point has D > 0 stops when the best
   (D, soft coordination term) has not improved for K generations. A restart whose best has
   D = 0 is never stopped by this rule. (D alone) and (D, total) are measured alongside.
6. **Weak constraints** (bond, angle, bond valence, density, ...) stay in the objective only.
   `plateau_patience` stays off and is measured as the baseline.
7. **Patience**: K = max(15, ceil(c n)), n the parameter count. c is the smallest value that loses
   no zero-error class on the calibration half, times 1.5.
8. **A stopped restart's** best point is an ordinary endpoint (ledger, Stage 2, centring
   unchanged). The record gains the number of stopped restarts and their generations.
9. **Adoption** as the Phase-2 default, all four required, otherwise opt-in:
   (1) no zero-error class lost on the validation half; (2) Stage-1 evaluations down by at least
   1.5x; (3) more saving than `plateau_patience` at its own zero-loss patience; (4) the gates at x4,
   seed 42, run with the rule on, give the same tier 1 and Stage-1 convergence as without.
10. **Measurement**: full trajectories recorded at the production budget with the ledger's early
    stop and `plateau_patience` off, then replayed offline. A seeded restart stopped at generation
    g is exactly the first g generations of its full run (`tests/unit/test_replay_early_stop.py`),
    so the replay is exact for Stage 1.
11. **Implementation**: `compute_error` gains an optional `terms` output (total unchanged); the
    rule is in `optimize/cma.py`'s generation loop, CMA-ES only, a function of the restart's own
    history alone, so ADR-0034's batching can take it over unchanged.

## Measurement

`scripts/record_stage1_trajectories.py runs/alon/alon.yaml runs/alon/alon.jsonl --out
runs/earlystop --fraction 0.2`: 20 % of each space group's unpruned assignments (2,446), the
space groups split alternately within each crystal system into calibration (1,433 assignments)
and validation (1,013). `scripts/replay_early_stop.py runs/alon/alon.yaml runs/earlystop` writes
`replay.csv`.

## Results

Recorded 2026-11-09: 2,446 assignments, 9,784 Stage-1 restarts, no failures. Reference Stage-1
cost (ledger on, no rule): 7.01 M evaluations on the calibration half.

Restarts: 73.3 % end (best total error) with D > 0, but 58.6 % evaluate at least one D = 0 point
(median first such generation 3 of 150), and the rule never stops those. Only ~41 % of restarts
are stoppable, which caps the saving at ~1.7-1.8x even at the smallest patience.

Smallest zero-loss c on the calibration half, times 1.5 (decision 7), then the validation half:

| rule | zero-loss c (cal.) | chosen c | saving cal. | saving val. | zero classes lost (val.) | near classes lost (val.) |
|---|---|---|---|---|---|---|
| deficit | 8 | 12 | 1.23x | 1.20x | 0 / 351 | 0 / 490 |
| deficit_cn | 6 | 9 | 1.24x | 1.22x | 0 / 351 | 0 / 490 |
| deficit_f | 6 | 9 | 1.19x | 1.17x | 0 / 351 | 0 / 490 |
| plateau | 12 | 18 | 1.005x | 1.005x | 0 / 351 | 0 / 490 |

Against decision 9 for deficit_cn at c = 9: (1) met; (2) not met, 1.22x < 1.5x; (3) met, 1.22x
against 1.005x. The rule therefore stays opt-in: `optimization.cn_stop`, ADR-0041 (the gates,
criterion 4, were not run). Full grid: runs/earlystop/replay.csv; chosen
patiences: runs/earlystop/replay_chosen.csv.
