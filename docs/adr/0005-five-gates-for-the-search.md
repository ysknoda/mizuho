# The search side is judged by five gates, not one number

ADR-0003 established a single gate: evaluate the objective at the published structure and require
it to be approximately zero. That gate now passes on every fixture, and end-to-end accuracy still
misses the 3% lattice target on three of four minerals. One number cannot say which stage is at
fault, so the search side gets its own gates — one per property, each failing for its own reason.

The evidence that one gate is not enough: with the published topology handed to the optimiser at
the production budget, diopside reaches 0.019499 while the published structure itself scores
0.000001. Raising the budget fourfold drops that to 0.000000 — at a lattice 5.6/4.0/6.6/6.8% away
from the published one. The error fell and the answer got worse. No single assertion distinguishes
"the optimiser stopped early" from "the optimiser arrived somewhere wrong".

## The five gates

| Gate | Asserts | Cost | Where |
| --- | --- | --- | --- |
| Reference structure (ADR-0003) | objective ≈ 0 at the published structure | one `compute_error` | `test_reference_structure_gate.py` |
| Impostor regression | `error(known impostor) > error(published)` | two `compute_error` | `test_reference_structure_gate.py` |
| Rank | the published topology appears within the first `max_candidates` of Phase 1's ranking | one Phase 1 run (81–1152 s) | `test_search_gates.py` |
| Global minimum | two stages, see below | one x4 optimisation (~35 min) | `test_search_gates.py` |
| End-to-end proxy | the published topology beats the top 5 ranked ones under the same budget | six x4 optimisations (~3.5 h) | `test_search_gates.py` |

The two fast gates stay with the `_REFS` table they need; the three slow ones import it. Splitting
by file rather than by marker means the runtime is legible before the run starts: opening
`test_search_gates.py` tells you this file takes hours.

### Thresholds

- **Reference structure**: unchanged from ADR-0003.
- **Rank**: `N = max_candidates` (default 50). That boundary is physically meaningful — inside it a
  candidate still receives Phase 2 budget and can win; outside it the candidate is provably
  eliminated. `N = 1` would fold Phase 2's quality into a Phase 1 measurement; `N = 10` has no
  basis.
- **Global minimum**: the budget is x4 (`n_restarts` and `cma_maxiter` both multiplied by four),
  the budget at which diopside provably reaches zero. The assertion has **two stages, reported
  separately**:
  1. the optimiser converged — `error ≤ error(published) + 1e-3`;
  2. the converged point's lattice is within **3%** of the published one.

  Stage 1 failing means the optimiser did not find a global minimum, and the gate says nothing
  about uniqueness. Stage 2 failing means the objective's minimum is not the real structure. Fusing
  them would let a future optimiser regression read as a uniqueness failure. Skipping on stage-1
  failure is worse still: it hides that regression entirely.

  The `+ 1e-3` is relative to what the objective itself achieves at the published structure, not an
  absolute figure. Adamite's reference scores 0.008057 because the 1937 refinement leaves H
  undetermined; demanding `< 1e-3` there would demand better than the data.

### Recorded failures

A gate that currently fails is written at its correct threshold and marked `xfail`, never relaxed
to fit. Adamite's published topology ranks 2950 of 69,974 — 59x outside the rank gate — and is
recorded that way. `xfail` also detects the repair: the test reports `xpass` the moment it starts
passing.

Topaz's rank position is 50 of 3,234, i.e. it passes with **zero margin**, and only because that
fixture sets `feasibility_shortlist: 60`. The gate carries that fact in a comment so a future
reader does not mistake a tuned pass for a robust one.

## Consequences

- Gates are built and run **before** any fix, and the failing baseline is recorded. The diopside
  x1 → x4 result above is precisely the kind of regression that is invisible without a recorded
  baseline.
- Each fix is applied on its own, with the gates re-run in between, so its effect is a difference
  rather than an inference.
- Work is sequenced by what the gates measure, not by which failure looked largest first. An
  earlier plan put the optimiser first; the budget ladder showed the optimiser is not the
  bottleneck, and that ordering was withdrawn. See ADR-0007.
- Completion is not "all gates green" alone. Changing the objective changes its landscape, so new
  impostors may appear. The rule is: fix, re-run the x4 probes, add any newly found impostor to the
  regression gate, and repeat — **capped at two rounds**. Anything still standing is recorded as
  `xfail` with its full diagnosis. Reaching a third round means a third mechanism exists and the
  design, not the parameters, needs revisiting.
