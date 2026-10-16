# An exhaustive space-group strategy, and symmetry first among satisfied entries

ADR-0021 made the automatic space-group search opt-in: none of its five known answers is reached.
Two of its misses are Phase 0's own: CsCl's 221 "never enters Phase 0's top five", and every
miss is a group Phase 0 chose. Mode A, given the right group, does not have that failure.

## Decision

- **`spacegroup_search.strategy: exhaustive`** runs every one of the 230 groups through mode A's
  pipeline: Phase 1, Phase 2, supergroup promotion (ADR-0010) and one global ranking.
  `spacegroup_search.spacegroups` restricts it to a named subset. A group that no assignment of
  the composition fits is skipped by Phase 1 in milliseconds. It needs `mode: auto_spacegroup`
  like the other strategies (ADR-0021).
- **Groups run from the highest point-group order down**, so a partial run has covered the groups
  the answer is most likely in.
- **`spacegroup_search.record: path.jsonl` records each finished group** (every Phase 2 entry's
  assignment, error, parameters, topology class and detected group) and a run with an existing
  record resumes after its last finished group. A first NaCl run over all 230 groups reached group
  16 of 230 in 1 h 40 min, about 6 min a group (P2/m alone has 5,048 assignments, 50 optimised),
  and was lost to a container restart; without a record a run of a day cannot survive one.
  `--workers N` (Phase 2's `parallel.candidate_workers`) spreads a group's candidates over cores.
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
reaches 0.036. 367 s.

All 230 groups, one worker, seed 42, the known-answer constraints (`tests/fixtures/auto/*.yaml`),
ranked by `_ranking_key` (threshold 1e-3):

| compound | entries | at or below 1e-3 | rank 1 | published group's best rank | time |
| --- | --- | --- | --- | --- | --- |
| NaCl | 4642 | 738 | Fm-3m (225), Na@a Cl@b, 0 | 1 | 15.0 h |
| CsCl | 1421 | 148 | Pm-3m (221), Cs@a Cl@b, 0 | 1 | 2.3 h |
| ZnO | 4642 | 181 | F-43m (216), Zn@a O@c, 0 (zinc blende) | 14 (P6_3mc, 186, Zn@b O@b, 0) | 13.9 h |

NaCl and CsCl: the published group ranks first. ZnO: wurtzite (186) reaches error 0 but ranks
14th, behind zinc blende (216, 215), a P6_3/mmc entry (194) and tetragonal groups, all at zero
error. Wurtzite and zinc blende are two stackings of the same corner-sharing ZnO4 tetrahedra
(zinc blende ZnO is a known metastable polymorph); the constraints, which state only CN 4 and
the Zn-O band, cannot tell them apart, and the symmetry-first ranking then prefers the cubic
group. This is the distinguishability limit of ADR-0027 in the space-group search: telling
polytypes apart needs information beyond first-shell constraints (energy, e.g. DFT).

## Consequences

- The exhaustive results are a ground-truth table for L1 (ADR-0020): for each known-answer
  compound, the error and class every group reaches. Any Phase 0 strategy (PUCT, flat, AB-MCTS
  at L1) can be replayed against that table instead of re-optimising.
- `run`'s ranking order changes for every automatic strategy: a constraint-satisfying entry in a
  higher group outranks a lower-error one in a lower group.

## The plan for L1 (agreed 2026-10-14)

In AB-MCTS at L1, GEN widens (a new space group) and CONT deepens (more budget for a group already
started, through L2). The known-answer misses are of two kinds. CsCl's 221 never enters Phase 0's
top five, which allocation can address. NaCl -> 99 and Si -> 74 stop in subgroups of the answer,
which it cannot: a subgroup has more freedom and reaches zero error more easily, so a reward on
error alone pulls any allocation, AB-MCTS's included, towards subgroups. The order of work:

1. Run the five known-answer compounds with `strategy: exhaustive`, one at a time, and record each
   group's error and topology class: the ground-truth table.
2. Compare L1 strategies (PUCT, flat, AB-MCTS) by replaying them on that table, without
   re-optimising.
3. Try a top-down order first, from the highest point-group order down, with a reward that
   carries symmetry as this ADR's ranking does. GEN and CONT are tuned after that.

