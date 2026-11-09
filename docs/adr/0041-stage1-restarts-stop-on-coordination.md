# Stage-1 restarts may stop on an unmet coordination range (opt-in)

Phase 2 stops a Wyckoff assignment's restarts early only on success: ADR-0019's endpoint ledger
skips the remaining restarts once two zero-error endpoints share a topology class. A restart that
is getting nowhere runs its full `cma_maxiter` generations, and `plateau_patience` (relative
improvement of the total error) is off in Phase 2. In the Al-O-N composition-free run (ADR-0033)
the coordination term is 63 % of a non-zero endpoint's error on average, and 267 of 300 sampled
non-zero endpoints end with some centre's integer coordination number outside its range. Design
and measurement: docs/plan-phase2-early-stop.md.

## Decision

- **Strong constraint.** The coordination deficit of a point,
  D = sum over every centre of every coordination constraint (cation and anion centres) of
  max(0, cn_min - n, n - cn_max), n the neighbours closer than the cutoff over every periodic
  image (ADR-0001). `compute_error(..., terms=...)` returns it as `terms["cn_deficit"]`, with
  every term's contribution beside it; the total is unchanged bit for bit. Weak constraints
  (bonds, angles, bond valence, density, ...) stay in the objective only.
- **Rule.** A CMA-ES restart stops between generations when its lexicographic best
  (D, soft coordination term) has not improved for K generations (the soft term by more than
  `rel_tol` relative) while that best has D > 0. A restart that has evaluated a point with D = 0
  is never stopped by it. Its best point is an ordinary endpoint: ledger, Stage 2 and centring
  are unchanged. The rule depends on the restart's own history alone. CMA-ES only.
- **Patience.** K = max(`min_patience`, ceil(`patience_factor` n)), n the parameter count.
  Defaults 15 (Phase 1's local-shell patience) and 9: the smallest factor that lost no zero-error
  class on the calibration half below (6), times 1.5.
- **Opt-in.** On only with an `optimization.cn_stop` mapping (keys `patience_factor`,
  `min_patience`, `rel_tol`; `cn_stop: {}` takes the defaults); without it Stage 1 is exactly the
  previous one, the objective called with the parameter vector alone. Bad keys or values are a
  ConfigError before the optimisation starts (ADR-0036).
- **Record.** `AssignmentResult.cn_stopped` and, in the exhaustive run's record, each entry's
  `cn_stopped` field list the generations at which restarts were stopped; both are absent (None)
  with the rule off.

## Measurement

Full Stage-1 trajectories of 2,446 Al-O-N assignments that survive ADR-0037's bond-count pruning
(20 % per space group; the groups split alternately within each crystal system into calibration,
1,433, and validation, 1,013), at the production budget (`n_restarts` 4, `cma_maxiter` 150,
seed 42), with the ledger's early stop and `plateau_patience` off; then replayed offline with the
ledger on (`scripts/record_stage1_trajectories.py`, `scripts/replay_early_stop.py`). A seeded
restart stopped at generation g is exactly the first g generations of its full run, so the replay
is exact (`tests/unit/test_replay_early_stop.py`, `tests/unit/test_cn_stop.py`).

| rule (patience factor) | saving, calibration | saving, validation | zero-error classes lost (val.) |
|---|---|---|---|
| (D, soft CN), 9 | 1.24x | 1.22x | 0 / 351 |
| D alone, 12 | 1.23x | 1.20x | 0 / 351 |
| (D, total), 9 | 1.19x | 1.17x | 0 / 351 |
| `plateau_patience`, 18 | 1.005x | 1.005x | 0 / 351 |

Saving is Stage-1 evaluations without the rule over those with it. No near-zero class (error at
or below 1e-2) is lost either.

## Why not the default

The adoption criteria set before the measurement were all four of: no zero-error class lost on
the validation half; Stage-1 evaluations down at least 1.5x; more saving than `plateau_patience`
at its own zero-loss patience; the twelve-mineral gates unchanged. The first and third hold; the
second does not (1.22x). The saving is capped by the restarts the rule may stop: 73 % of restarts
end with D > 0, but 59 % evaluate a D = 0 point on the way (median first at generation 3 of 150)
and are never stopped, which bounds the saving at about 1.7x even at the smallest patience. The
gates were not run.

## Consequences

- An exhaustive composition-free run can take `optimization.cn_stop: {}` for about a fifth less
  Stage-1 work at no measured loss on Al-O-N. Other chemistries are unmeasured.
- `compute_error`'s `terms` output is available to any later per-term analysis.
