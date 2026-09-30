# Penalty normalisation is always on; `normalize_penalties` is removed

Two different objectives had been in use without anyone choosing that. Every search run in this
project set `normalize_penalties: true`. No fixture sets the key, so the fixtures — and therefore
the ADR-0003 gate — scored the default, `false`. The two objectives genuinely disagree about which
structure is better:

| adamite, published topology at x4 budget | error found | error at published |
| --- | --- | --- |
| `normalize_penalties: true` | 0.00564958 | 0.00805674 — the found point wins |
| `normalize_penalties: false` | 0.17092684 | 0.02917244 — the published structure wins |

The gates were therefore testing a different objective from the one being optimised.

The default was `false` for one stated reason, in README.md: it "reproduces every prior release's
exact numeric behaviour". That is a compatibility argument, and ADR-0002 already decided
compatibility is not a reason to keep a switch. The same README measures what `false` costs: the
coordination-number term is 65-99% of the total across this project's fixtures, so every other
term's `weight` is nominal. Normalising divides each deviation by that term's own natural scale
(the band midpoint; the cell's total ionic charge for charge balance) so the terms are comparable
and the weights mean what they say.

## Decision

- Remove the key. The objective always normalises, in `objective/energy.py`,
  `objective/breakdown.py` and `objective/energy_jax.py` alike (the C-2 mirror rule).
- A config that still contains `normalize_penalties` is rejected with an error naming this ADR,
  rather than silently ignored. Ignoring it would let a user believe `false` still does something.

## Consequences

- Every recorded error value changes unit. This lands **first** in
  `docs/plan-verification-and-tree-search.md`, before any new gate baseline is recorded, so that
  no baseline has to be recorded twice.
- ADR-0003's ceilings still hold. At the published structures the normalised objective is lower
  than the raw one on every fixture (topaz 0.000000, diopside 0.000001, spinel 0.000003, adamite
  0.008057 against raw 0.000003, 0.000023, 0.000055, 0.029172).
- Adamite's impostor now exists in the default objective. It is not a regression: `false` hid it
  only because the coordination term dominated and happened to favour the published structure.
  ADR-0007's integrality check rejects it on its own terms (Zn deviates 0.4773 from an integer
  against a 0.05 ceiling), and that check is geometric, so it does not depend on this choice.
- Resolves the open question left by ADR-0007 and Phase 0 of `docs/plan-objective-uniqueness.md`.
