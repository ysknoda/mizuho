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
| Si | 925 | 53 | Fd-3m (227), Si@a, 0 | 227: 1; 194: 6 (Si@f, 0) | 0.5 h |
| ZnO | 4642 | 163 | F-43m (216), Zn@a O@c, 0 (zinc blende) | 216: 1; 186: 13 (Zn@a Zn@b O@a O@b, 0) | 12.2 h |
| TiO2 | 5507 | 736 | Pn-3m (224), Ti@b O@e, 0 | 141: 36 (Ti@a O@e, 0); 136: 39 (Ti@a O@f, 0) | 19.4 h |

Ranks are over the raw record (`scripts/exhaustive_table.py`), before supergroup promotion and
the topology-class merge; `run`'s merged ranking is given below where it differs. Si, ZnO and
TiO2 ran on 2026-10-16/17 on a local machine, three runs at once, one worker each; NaCl, CsCl
and the first ZnO run ran in a container, so the times compare only roughly. TiO2's promotion
and ranking after the last group took a further 6.4 h (2,673 re-optimisations). Si and ZnO ran
at a9b05d7, after the angle-term fix below; TiO2's fixture has no angle constraint.

NaCl, CsCl and Si: the published group ranks first. Si's merged ranking is Fd-3m, then P6_3/mmc
(194, hexagonal diamond), its polytype; the P4_132 and P4_332 entries at zero error between them
in the record have diamond's coordination sequence (4, 12, 24, 42, 64, ...) and merge into
Fd-3m's class.

ZnO: wurtzite (186) reaches error 0 but ranks 13th, behind zinc blende (216, 215) and
tetragonal groups, all at zero error; merged, 216 is first, 137, 136, 138 and 131 follow, and
186 is 6th. Wurtzite and zinc blende are two stackings of the same corner-sharing ZnO4 tetrahedra
(zinc blende ZnO is a known metastable polymorph); the constraints, which state only CN 4 and
the Zn-O band, cannot tell them apart, and the symmetry-first ranking then prefers the cubic
group. This is the distinguishability limit of ADR-0027 in the space-group search: telling
polytypes apart needs information beyond first-shell constraints (energy, e.g. DFT). The
tetragonal zero-error classes are not characterised.

TiO2: every one of the first 37 merged entries is at zero error. Anatase (141) is 8th and
rutile (136) 11th, behind cubic (224, 213, 205) and hexagonal (194, 191) structures.

With derived constraints and a measured density (ADR-0030, 6a56473): rutile's fixture ranks 136
fifth (merged third), anatase's ranks 141 fourth (merged second), behind a layered P6_3/mmc class
in both; ZnO ranks 216 first and 186 39th (merged eighth). See ADR-0030's measured section.

### The angle term skipped a lone neighbour atom (fixed in a9b05d7)

The first Si run put a one-atom P4/mmm square net (Si@a, angles 90 deg against [105, 115]) third
in the merged ranking at error 8e-6: the angle term skipped any constraint whose neighbour
species had one atom in the cell, although that atom's images are neighbours (ADR-0001). After
the fix it scores 0.082. Si's record is otherwise unchanged (the false entry reached the threshold
only after promotion). ZnO's zero-error entries fell from 181 to 163 and 186's rank from 14 to
13; the earlier ZnO row is replaced by the rerun.

### Zero error is not the published structure (measured 2026-10-17)

The constraints state the cation-anion band, CN and (Si, ZnO) the cation-centred angle; nothing
bounds the like-ion distances. For every recorded entry at or below 1e-3:

| compound | entries | on a `lattice_bounds` edge | shortest like-ion distance, median [10-90 %] | published |
| --- | --- | --- | --- | --- |
| Si | 53 | 0 | Si-Si 2.34 [2.30, 2.38] A | Si-Si 2.352 A (bond) |
| ZnO | 163 | 27 | Zn-Zn 2.37 [2.27, 3.17] A | Zn-Zn 3.210 A (`wurtzite.cif`) |
| TiO2 | 736 | 342 | O-O 1.81 [1.38, 2.46] A | O-O 2.535 A rutile, 2.468 A anatase |

Most ZnO entries share tetrahedron edges (Zn-O-Zn, the O-centred angle, is unconstrained). Most
TiO2 entries put two oxide ions 1.4-2.5 A apart, and almost half are on a bound (ADR-0006: the
box is doing the physics). None of the four zero-error entries in 136 or 141 is rutile or anatase:
136 Ti@a O@f has c = 2.500 A, the tetragonal lower bound; 141 Ti@a O@e has O-O 1.57 A. Published
rutile scores 0 under the fixture, so a family of zero-error structures runs through it and the
optimiser stops anywhere along it. Published anatase (`anatase_origin2.cif`) scores 0.0020,
above the threshold, from the bond-valence term (Ti BVS 4.28 against 4 +- 0.1).

So TiO2's ranking measures the constraints, not the search: CN 6/3 and a Ti-O band admit hundreds
of zero-error structures, most of them unphysical, and the symmetry-first rule picks the most
symmetric. The exhaustive table is a ground truth for the search only where the published
structure is the objective's unique zero-error class (ADR-0007). This holds for Si (up to
polytypes) and not for ZnO or TiO2; NaCl and CsCl were not measured this way, their records
having been lost with the container.

## Consequences

- The exhaustive results are a ground-truth table for L1 (ADR-0020): for each known-answer
  compound, the error and class every group reaches. Any Phase 0 strategy (PUCT, flat, AB-MCTS
  at L1) can be replayed against that table instead of re-optimising.
  It is a ground truth for the search only where the published structure is the unique
  zero-error class; for ZnO and TiO2 it is not yet (see "Zero error is not the published
  structure").
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

