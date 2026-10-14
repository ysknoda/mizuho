# Phase 1 is seeded, and the rank gate judges the worst rank over five seeds

ADR-0005's rank gate asserts that the published topology sits within the first `max_candidates`
of Phase 1's ranking. Its recorded state — spinel, diopside and topaz passing, adamite failing —
does not reproduce. Run twice on identical code and fixtures:

| fixture | first run | second run | gate (N = 50) |
| --- | --- | --- | --- |
| spinel | 2 | 1 | pass, pass |
| diopside | 25 | **163** | **pass, fail** |
| topaz | 50 | 26 | pass, pass |
| adamite | 2950 | 9745 | fail, fail |

The cause: `pipeline.search_and_rank` passes no seed to `precompute_letter_feasibility` or to
`rank_by_feasibility_staged`. Both default to `seed=None`, so the small CMA-ES solves that score
each candidate's local geometry are unseeded, and the configuration's `optimization.seed` never
reaches Phase 1.

## Decision

- **Pass `optimization.seed` into Phase 1.** A run is then reproducible.
- **Judge the rank gate on the worst rank over five seeds, and record the spread.** Seeding alone
  would only make the flakiness reproducible: the gate would pass on seed 42, fail on 43, and
  nothing would reveal that. A diopside rank that moves from 25 to 163 suggests that near the
  published topology the feasibility scores differ by less than the solver's own noise. The spread
  across seeds measures whether the ranking carries a signal there at all.
- If the spread is large, the remedy is a stronger feasibility solve (more restarts per candidate).
  That is decided after measuring, not before.

## Consequences

- ADR-0005's recorded rank-gate results are withdrawn as unreproducible, including the note that
  topaz passes "with zero margin" at rank 50: that was one unseeded draw.
- The rank gate becomes five times as expensive. Topaz's Phase 1 took 601-1152 s per run, so its
  rank gate is of the order of an hour. It stays in the slow tier (`test_search_gates.py`).
- ADR-0011 changes what the gate compares (equivalence classes rather than literal letters); the
  two changes land together.

## Amended by ADR-0020

The spread is large (diopside 8-103 over five seeds after the uniqueness plan, worst 154 after
ADR-0016; adamite 1553-4036), so the remedy decided above, a stronger
feasibility solve, is measured as a budget ladder: first against the two AsO4 xfails, then on the
five-seed rank spread.
