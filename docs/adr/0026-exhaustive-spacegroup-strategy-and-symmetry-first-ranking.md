# An exhaustive space-group strategy, and symmetry first among satisfied entries

ADR-0021 made the automatic space-group search opt-in: none of its five known answers is reached.
Two of its misses are Phase 0's own: CsCl's 221 "never enters Phase 0's top five", and every
miss is a group Phase 0 chose. Mode A, given the right group, does not have that failure.

## Decision

- **`spacegroup_search.strategy: exhaustive`** runs every one of the 230 groups through mode A's
  pipeline: Phase 1, Phase 2, supergroup promotion (ADR-0010) and one global ranking.
  `spacegroup_search.spacegroups` restricts it to a named subset, for a partial or resumed run. A
  group that no assignment of the composition fits is skipped by Phase 1 in milliseconds. It
  needs `mode: auto_spacegroup` like the other strategies (ADR-0021).
- **Among entries that satisfy the constraints, the highest symmetry ranks first.** The global
  ranking puts entries at or below `optimization.topology_grouping.zero_error_threshold` first,
  ordered by point-group order (Fm-3m 48, P4mm 8, P1 1) and then by error. Every other entry
  ranks by error alone. This applies to every automatic strategy.

  Why: a structure can also be written in every subgroup of its group, with more freedom, so it
  and its band-slack distortions reach zero error there too (ADR-0010). They share one topology
  class, so ADR-0019's merge keeps one of them, and before this rule that was whichever sorted
  first. The answer ADR-0010 names is the highest group in which the constraints are met.

## What exhaustive does and does not guarantee

The premise is that if mode A works in each group, the published structure is reached somewhere.
That holds only where mode A holds:

- **Phase 1 must rank the published assignment within `max_candidates`.** It does not for
  diopside (worst rank 185 of 11,230 over five seeds, measured 2026-10-14) or adamite (ADR-0020,
  measurement 3).
- **Phase 2 must converge.** It does not for topaz or adamite at x4 (ADR-0020's term ablation).

Exhaustive removes Phase 0's misses and keeps mode A's. Its cost is the sum of mode A over every
group the composition fits, on the order of hours for a binary compound.

## Measured

NaCl, the `tests/fixtures/auto/nacl.yaml` constraints, seed 42, restricted to P4mm (99) and Fm-3m
(225): Fm-3m ranks first at error 0 and is detected as 225 from 1e-4 to 1e-1 A; P4mm's best
reaches 0.036. 367 s. The run over all 230 groups is recorded when it finishes.

## Consequences

- The exhaustive results are a ground-truth table for L1 (ADR-0020): for each known-answer
  compound, the error and class every group reaches. Any Phase 0 strategy (PUCT, flat, AB-MCTS
  at L1) can be replayed against that table instead of re-optimising.
- `run`'s ranking order changes for every automatic strategy: a constraint-satisfying entry in a
  higher group outranks a lower-error one in a lower group.
