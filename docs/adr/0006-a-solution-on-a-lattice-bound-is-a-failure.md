# A converged solution sitting on a `lattice_bounds` edge is a failure, not a result

ADR-0003 listed "no lattice parameter pinned at a `lattice_bounds` edge" as a *secondary
indicator*. It is promoted here to a hard assertion, because measurement showed it is the single
most reliable signal that an answer is wrong.

At the published topology, optimised at the production budget:

| fixture | free params | pinned lattice params | volume vs published |
| --- | --- | --- | --- |
| spinel | 2 | **0 / 1** | +0.8% |
| adamite | 17 | 1 / 3 (`a` = upper) | +9.1% |
| diopside | 18 | 2 / 4 (`b`, `c` = upper) | +18.1% |
| topaz | 18 | **3 / 3** (`a` = lower, `b`, `c` = upper) | +12.7% |

The one fixture that reaches the real structure is the one fixture that touches no bound. Every
failure touches at least one. The three impostors of ADR-0007 pin bounds too — topaz's pins all
three even at quadruple budget.

The reasoning is not statistical. A solution on the boundary means the objective is still pushing
in that direction and nothing inside the model stops it: **the box is doing the physics**. The box
is ours — `lattice_bounds` in these fixtures was drawn from literature values, and the literature
value lies comfortably inside it for every real-Z fixture. An answer that lands on the wall we
built is not evidence about the mineral.

## Decision

- **Strength: both.** In the gates this is a hard failure. In a production report the solution is
  still returned, with a warning saying the answer sits against a bound — the user's own bounds may
  genuinely be too narrow, and silently discarding their answer would be worse than telling them.
- **Scope: lattice parameters, plus any free coordinate whose `param_bounds` were narrowed below
  the default `[0, 1]`.** A coordinate at 0 or 1 under the default bounds is not against a wall at
  all — coordinates are periodic. A fixture that deliberately wrote a narrow `param_bounds` did
  build a wall, and that one counts.
- **Tolerance: 1% of the interval's own width.** Exact equality misses it — floating-point
  convergence lands at 7.60001, not 7.6. Five percent is too coarse: on topaz's `a` interval
  [7.6, 9.2] that is 0.08 A, which starts to overlap the 3% accuracy target itself.
- **This applies even when the answer is already within 3% of the literature.** A within-3% answer
  that sits on a bound is within 3% because of where we drew the box, and calling that a pass
  restores "the number looks good" as the measure of correctness — the exact criterion ADR-0003
  exists to reject.

## Consequences

- The rank gate cannot carry this assertion: Phase 1 ranks topologies and has no lattice at all.
- A fixture whose bounds are genuinely too tight will now fail loudly instead of silently returning
  a wall-shaped answer. That is the intended trade: widening bounds is a deliberate, recorded act.
