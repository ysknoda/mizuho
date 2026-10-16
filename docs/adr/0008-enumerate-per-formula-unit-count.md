# The formula unit count is an ambiguity to report, not to collapse silently

A composition, a space group and a set of bonding constraints do not determine Z. `Zn2(AsO4)(OH)`
in Pnnm is a statement about ratios; nothing in it says the cell holds four formula units rather
than one. mizuho currently caps Z from above (`tree_search.max_formula_units`) and has **no way to
set a floor**, so the search may always descend to the smallest cell that satisfies the
constraints, and the output collapses to a single answer with no sign that the question was
under-determined.

## Decision

- **Axis: the formula unit count Z.** The input is already a composition formula, and Z is the
  same axis `max_formula_units` already uses. Counting total atoms instead would admit values that
  do not divide into whole formula units and would not line up with the search's own pruning.
- **Knob: add `tree_search.min_formula_units`**, symmetric with the existing ceiling. One key
  expresses both a range and a pin (`min == max`). A separate `formula_units:` key would let the
  same thing be written two ways and create a precedence rule that means nothing.
- **Report: the best k candidates *per Z*.** `--top` keeps its meaning, applied within each Z. What
  went wrong was never that a small-Z answer is invalid — it is that the user cannot see whether a
  larger-Z answer existed at all.

## Implementation is deferred

This decision is recorded and **not implemented in this round of work**. The evidence originally
cited for it does not support it: a wollastonite run returning a Z = 1 cell was presented as
search bias, and the fixture in fact caps `max_formula_units` to 1 deliberately, documented in its
own header as a tractability simplification. The retraction removes the urgency, not the decision.

Across the four gated minerals, Z ambiguity costs nothing measurable — every published topology sits
in the candidate pool at the correct Z. Meanwhile the objective is demonstrably non-unique
(ADR-0007), which is an active loss of accuracy. Limited verification budget goes to the active
problem.

## Consequences

- An ADR can record a decision that is deliberately not acted on yet. That is the point of writing
  it down rather than leaving it in a conversation.
- When this is implemented, it changes the CLI's output contract. Per ADR-0002 there will be no
  compatibility flag: the new behaviour becomes the default and affected tests are updated, because
  the current silent collapse is the defect and leaving it behind a flag would leave every default
  user with it.

## Amended by ADR-0022

The report is per topology class, not per Z: a class is the same net at any Z (its signature is
gcd-normalised), so per-Z lists would repeat it. Each class carries the set of Z it was reached at,
each Z searched reports how many classes first appeared there, and a class is represented by its
smallest-Z entry among those that satisfy the constraints. The deferral ends with ADR-0020; the
implementation follows its measurements.
