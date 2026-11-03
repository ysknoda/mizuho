# Centring acts on zero-error points only

ADR-0028 (c) centres each refined point inside the derived bond bands: L-BFGS-B on
E + 0.01 E_ideal, kept when E rises by at most 1e-4 and the topology class is unchanged. It was
measured and adopted on tier-1 end points, all at zero error, as a tie-break inside the region
where the constraints are met. The code centred every refined point, at any error. ADR-0039's
replays found that centring points above the threshold takes a tenth of an assignment's time.

## Measured

`scripts/cross_group_redundancy.py replay`, one core, every Stage-2 and centring call recorded
with its error and class before and after (threshold 1e-3):

| sample | assignments | centring above the threshold, share of time | calls | brought below the threshold | class changed |
| --- | --- | --- | --- | --- | --- |
| Al-O-N, random (seed 2) | 40 | 8.2 % | 79 | 0 | 0 |
| Al-O-N, random (ADR-0039 batch B) | 16 | 10.9 % | 34 | 0 | 0 |
| rutile (ADR-0030 record), random | 20 | 5.4 % | 20 | 0 | 0 |

Above the threshold, centring lowered the error in 6 of 133 calls, by at most 9e-4 (1 %
relative); in 63 it raised it, within the 1e-4 it allows. Centring at or below the threshold
takes 0-8.5 % and is kept. Stage 2 (16.6 % of Al-O-N time) can bring a point from 0.5 to zero
error (ADR-0039, SG 23 #98) and is kept for every point.

## Decision

- **`optimization.centring: true` (the default) centres a refined point only when its Stage-2
  error is at or below `topology_grouping.zero_error_threshold`** (`mizuho/optimize/driver.py`,
  `centring_mode`). `"all"` centres every refined point, the behaviour before this ADR;
  `false` none. Any other value is an error.
- Every strategy: centring above the threshold has no measured use in any of them.

What changes: an entry above the threshold is recorded at its Stage-2 point, not its centred
point. Its class is the same (centring never changes it), its error differs by at most 1e-4 up or
9e-4 down in the samples. `scripts/export_classes.py --best` exports such entries for
compositions without a zero-error class: 3 of Al-O-N's 2,568 classes.

## Measured: validation

`run --workers 1 --top 3`, seed 42, one core, before (7329396) and after this change, on
ADR-0039's subsets of the auto fixtures:

| fixture, groups | zero-error entries | every entry's class | ranking | search |
| --- | --- | --- | --- | --- |
| rutile: 225, 209, 205, 202, 141, 166, 122, 98, 88, 155, 148 | 7, identical to the bit | identical | 6 classes, zero-error rows identical | 58 s -> 55 s |
| ZnO: 216, 215, 198, 196, 195, 107, 104, 95, 91, 63, 60, 56 | 15, identical to the bit | identical | 25 classes, zero-error rows identical | 303 s -> 290 s |

The 89 entries above the threshold keep their class; their errors move by at most 2.8e-6 and
none crosses the threshold. The saving on these binaries (4-5 %) matches rutile's replay
(5.4 %) and is below Al-O-N's (8-11 %).

## Consequences

- A search is 4-11 % shorter on these samples (Al-O-N 8-11 %, the binaries 4-5 %).
- Entries above the threshold are no longer pulled towards ideal bond lengths; their recorded
  points are Stage 2's local minima of the objective itself.
