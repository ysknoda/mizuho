# Correctness is judged at the experimental structure, not by the error value falling

The obvious success measure for "the error function does not come down" is "the error value came
down". We reject that, because it is circular and the data contradicts it: `spinel_mgal2o4` has by
far the **lowest** error of the five fixtures (`err_norm` 0.004053) and still lands on a = 7.712 A
against a literature 8.083 A, a 4.6% error. Optimising harder against a mis-specified objective
does not produce a better structure.

The gate is therefore: **build the experimental structure and evaluate the objective there.** Each
coordination term must report the literature coordination number, and `compute_error` must be
approximately zero. Only once the objective's minimum demonstrably sits at the real structure does
improving the search become meaningful.

## Consequences

- Secondary indicator: no lattice parameter pinned at a `lattice_bounds` edge. In the pre-change
  baseline, four of five fixtures pinned at least one parameter (diopside pinned three of four)
  **even though the literature value lies inside the box for every real-Z fixture** — direct
  evidence that the objective's minimum is not at the real structure.
- Final target: searched lattice constants within **3%** of the literature values.
- Reference structures are written down in this repository and each one is checked against
  published bond lengths and coordination numbers before it is trusted. This is not optional
  caution: a first attempt at reconstructing spinel from memory produced an Mg–O distance of
  0.21 A, because the SG227 origin choice was wrong.

## Amended by ADR-0025

Adamite's reference is a DFT relaxation (AFLOW, PBE), the one exception to an experimental
reference: mizuho's output is a DFT starting structure, and the 1937 refinement left H
undetermined.
