# Promotion reuses the exhaustive run's own optimisations; no cross-group early stopping

An exhaustive run (ADR-0026) searches the groups from the highest point-group order down, and a
structure found in a group is found again in its subgroups. In the Al-O-N run (`runs/alon`,
ADR-0033) 8,591 of 38,975 recorded entries (22 %) were detected in a group of higher order than
the one searched. The question (design review 2026-11-03): can the search stop work on what a
higher group has already found, without losing a topology class (ADR-0019)?

## Measured: what the subgroups re-find

`scripts/cross_group_redundancy.py record`, threshold 1e-3, a class being (composition,
topology class) as in `scripts/export_classes.py`. Al-O-N, 230 groups, 56.1 h, `--top 3`:

| | count |
| --- | --- |
| entries / assignments | 38,975 / 17,843 |
| assignments reaching zero error | 3,169 (18 %) |
| zero-error entries | 4,735 |
| ... class first found in this group | 2,864 (2,565 distinct classes) |
| ... known from a group of higher order | 1,150, in 940 assignments |
| ... known from an earlier group of equal order | 721 |
| detected in a higher group | 8,591, of which 436 at zero error, 8,155 above it |

The 22 % are not re-findings: 95 % of them are restarts that stopped above the threshold at a
point of higher symmetry. Re-found zero-error classes sit in 940 assignments (5.3 %), 6.7 % of
the run's time with each group's time split equally over its assignments. That is the most any
rule acting on them can save.

The fixed-composition records (`--top 1`) show the same shape: known-higher entries are 82-92 %
of the zero-error entries, but the assignments holding them take 2.2-12.7 % of the time
(TiO2 ADR-0026 12.7 %, ZnO ADR-0030 11.5 %, rutile 2.6 %, anatase 2.2 %, Si 4.9 %).

### Stopping an assignment at a known class loses classes

The proposal: stop an assignment's restarts at its first Stage-1 endpoint at zero error in a
class known from a higher group, as ADR-0019's early stop does after two hits in one class.

- 253 of the 940 assignments also reached a class first found in that group; 241 classes
  (9.4 % of 2,565) are found in their first group only by such assignments. A stop could lose
  each of them, depending on restart order.
- Replayed (`replay`, 16 of the 940 assignments drawn at random, 8 of them from the 253 that
  also reached a new class; 14 reproduce the record exactly, the two others lack Phase 1's shell
  seeds): the stop saves 55 % of their time, at most 3-4 % of the run, and 3 of 16 lose a
  zero-error class a later restart finds. SG 23, Al4O3N2, #53: the
  known class at restart 2, a class first found in SG 23 at restart 3.
- Known classes are often reached only by Stage 2 (SG 23 #98: a Stage-1 endpoint at 0.5 refined
  to 3.8e-8 in a known class; SG 81 #57: 2.8e-2 to 1.7e-8), so a Stage-1 test misses them.

Skipping the centring (ADR-0028 (c)) of a refined point in a known class loses nothing, since
centring never changes the class and the higher group's entry represents it in the ranking, but
saves about 1 %: 1,150 entries at 1-13 s each.

### Promotion repeats optimisations the run has done

`run`'s promotion stage (ADR-0010) re-optimises, from scratch, every entry whose detected group
G' differs from its own, in the G' assignment it maps to. In an exhaustive run G' has been
searched already. `promotion` maps every such entry (first step) to its G' assignment:

| record | promotions | G' assignment already optimised in the run | of them at zero error |
| --- | --- | --- | --- |
| Al-O-N | 11,040 | 5,915 (54 %) | 251 |
| rutile (ADR-0030) | 3,047 | 1,687 (55 %) | 92 |

The promotion stage took 6.4 h after TiO2's 19.4 h search (ADR-0026, 2,673 re-optimisations);
for Al-O-N it was estimated above 17 h and the run was stopped before it (design review Q26).
The other 46 % are not repeats: G''s Phase 1 kept other assignments (`max_candidates`), and 420
zero-error Al-O-N entries (255 classes first found in their group) get their symmetry there.

## Decision

- **No cross-group early stopping.** Neither the restart stop (it loses classes) nor the
  centring skip (about 1 %) is built.
- **In an exhaustive run, promotion takes an assignment the run has already optimised from the
  run itself** (`pipeline.best_entry_per_assignment`, `promote_to_detected_supergroup(...,
  optimised=)`): the assignment's lowest-error entry, recorded or just optimised, stands in for
  the re-optimisation, which `optimize_one_assignment` would return as its best class. The
  symmetrised point still wins when it scores at least as well, as before. Assignments the run
  has not optimised are re-optimised as before. No key: other strategies are unchanged.
- The recorded entry is taken as the reference. For a fixed composition it is bit-identical to
  the re-optimisation (`tests/unit/test_promotion_from_record.py`, NaCl in Fm-3m). In a
  composition-free run Phase 2 also used the composition's `max_formula_units` and Phase 1's
  shell seeds, which promotion's re-optimisation does not, so there the record is the more
  faithful of the two.

## Measured: validation

`run --workers 1 --top 3`, seed 42, the code before (50e45c2) and after this change, one core,
on subsets of the auto fixtures (`spacegroup_search.spacegroups`, run highest order first):

| fixture, groups | record | promotions (from the run) | ranking | after the search |
| --- | --- | --- | --- | --- |
| rutile: 225, 209, 205, 202, 141, 166, 122, 98, 88, 155, 148 | identical | 8 (8) | identical, 6 classes (2 at zero error) | 57 s -> 5 s |
| ZnO: 216, 215, 198, 196, 195, 107, 104, 95, 91, 63, 60, 56 | identical | 70 (6) | identical, 25 classes (3 at zero error) | 225 s -> 238 s |

Every row of both rankings (group, assignment, error, merged count, dimensionality, sharing) is
the same before and after; so is every zero-error class. In the ZnO subset 64 of the 70
promotions land in groups outside it (221 twenty times, 65, 225, 139 eight times each), which
are re-optimised as before, so its time does not change (the difference is noise). A full
exhaustive run has searched every group a promotion can land in; there the share taken from the
run is the 54-55 % measured above.

## Next: centring above the threshold

`replay` of 16 Al-O-N assignments chosen at random: Stage 2 takes 18.2 % of an assignment's time,
centring of points at or below the threshold 3.3 %, centring of points above it 10.9 %. None of
those 34 centring calls brought a point below the threshold, and none changed a class. Centring
is a tie-break inside the zero-error region (ADR-0028 (c)); restricting it to points at or below
the threshold would save about a tenth of a run. Decided in ADR-0040.

## Consequences

- An exhaustive run's promotion stage roughly halves; the run can finish instead of being
  stopped before it.
- `run`'s output names each promotion "already optimised in this run" or "re-optimised".
