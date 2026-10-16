# The formula unit count is reported per topology class

ADR-0008 decided `tree_search.min_formula_units` and "the best k candidates per Z", and deferred
both while the objective was non-unique. ADR-0019 and ADR-0020 remove that ground. Before
implementing it, the report rule has to meet ADR-0019.

A topology class's signature is normalised by the gcd of its type counts (`mizuho/bond_graph.py`),
so one net described at Z = 1 and at Z = 2 is one class. `pipeline.merge_same_topology` merges
entries by class regardless of Z and keeps the lowest-error one. A larger-Z description has more
free parameters and tends to reach the lower error, so `run` can write a crystal that fits in a
Z = 1 cell as a Z = 2 cell. Read literally, ADR-0008's per-Z report would list one class again
under every Z it was reached at.

## Decision

- **Classes first.** One row per class, annotated with the set of Z it was reached at. For each Z
  searched, the report states how many classes were reached and how many first appeared at that Z.
  That is the direct answer to ADR-0008's concern: whether a larger-Z answer existed at all.
- **Representative of a class:** the smallest-Z entry among those at or below
  `optimization.topology_grouping.zero_error_threshold`; when none is, the lowest-error entry.
  Among entries that satisfy the constraints, the smaller cell is preferred to the lower error.
- `min_formula_units` as ADR-0008 decided: one key gives a range or, with `min == max`, a pin.
- Implemented after ADR-0020's measurements; no gate depends on it.

## Consequences

- ADR-0008's "best k per Z" is replaced by the per-Z counts above; `--top N` keeps ADR-0019's
  meaning, N distinct classes.
- The representative rule fixes the larger-cell output independently of `min_formula_units`.
