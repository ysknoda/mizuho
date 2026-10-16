# Tabu by topology class: where it could act (draft)

ADR-0028 left topaz and adamite at tier 1 0/5 with derived constraints: removing one kind of
zero-error other-class point sends the search to another. Design review 2026-10-15: should the
search avoid topology classes it has already reached (tabu), and where?

## Measured (2026-10-15)

**Across independent restarts** (`scripts/enumerate_classes.py`, x4, 32 restarts, early stop
off, every class refined):

| run | classes reached | restarts landing in a class already reached | zero-error (S1) | published class |
| --- | --- | --- | --- | --- |
| diopside seeds 42-44 (hand fixture) | 32, 30, 31 | 3 of 96 | 2, 1, 0 | never |
| topaz seed 42 (derived) | 28 | 4 of 32 | 1 | never |
| topaz seed 43 (derived) | 24 | 8 of 32 | 0 | never |
| adamite seed 42 (derived) | 32 | 0 of 32 | 0 | never |

Restarts almost never revisit a class. Most end at a local minimum with error 1e-2 to 5e-2; a
zero-error other class is the rarer failure. The landscape has many classes and the published
one is rarely hit, so a tabu across restarts has almost nothing to act on.

**Inside a basin-hopping chain** (`scripts/bh_revisits.py`, diopside derived, seed 42, T 0.01,
step 0.1, 168 hops before a container restart): 88 distinct classes, 74 hops (46 %) landing in a
class already visited, 18 hops accepted, lowest error 0.048, published class never reached.
Revisits are frequent inside a chain, but this chain barely moves (11 % accepted) and stays far
from zero error.

## Direction

- No tabu across restarts: measured to have nothing to act on.
- Inside a chain a tabu could act, but rejecting a move does not save the evaluations spent
  landing in a visited class; its effect is to push the chain on, like a higher temperature.
- Dual annealing reached the published class on diopside seeds 42 and 43 where CMA-ES did not
  (ADR-0020's backend comparison, hand fixture). The comparison is resumed with derived configs,
  recording every dual-annealing minimum with its class, to measure the revisit rate inside its
  chain before any tabu is built into it.

## Measured: backends with derived configs, dual-annealing minima by class

`scripts/compare_backends.py --variant derived`, x4, equal evaluations; every minimum dual
annealing reports is classed.

| run | CMA-ES | basin hopping | dual annealing | DA minima: reported / distinct / revisits |
| --- | --- | --- | --- | --- |
| topaz seed 42 (312k evaluations) | 0, S1, other class | 1.3e-2 | 0, S1, other class | 13 / 11 / 2 |
| diopside seed 44 (377k evaluations) | 1.5e-22, S1, **published** | 4.4e-5, S1, other class | 0, S1, **published** | 184 / 17 / 167 |

Topaz: dual annealing reports few minima (13 in 312k evaluations) and revisits 2; the
published class is not among them.

Diopside: the chain revisits heavily. One class at error 0.112 takes 121 of the 184 minima,
from evaluation 18.8k to 129k (30 % of the budget); the chain then passes two classes near
2.4e-2 and reaches the published class (error 0) at minimum 178, evaluation 214k. This is the
case a tabu inside the chain is for: a chain held in one non-zero class for most of its budget.
A candidate mechanism: when one class has taken more than a set number of a chain's minima at
non-zero error, stop that chain and give its remaining evaluations to a new start (scipy's
dual_annealing stops when its callback returns True).
