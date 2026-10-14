# The soft coordination count has compact support

The soft coordination count weighted each neighbour by the logistic `expit((cutoff - d) / width)`.
Its tails never vanish, and two defects followed from them:

- **The count is not the integer count even at a clean reference.** Pyrite's S(S) cutoff sits at
  the measured mid-gap, 2.620 A (ADR-0004), between the S–S bond at 2.162 A and six S at 3.078 A.
  Each of the six adds `expit(-4.58) = 0.010`, so the published structure scored soft CN ~1.05 at
  integer CN 1 and failed the ADR-0003 gate. ADR-0004's rule of thumb (cutoff at least three
  smoothing widths past the first shell) was satisfied — 4.6 widths each side — and still did not
  saturate the count, because the second shell is populous.
- **Parked atoms buy coordination.** ADR-0007's integrality penalty and the gray-band penalty
  (ADR-0014) were both patches on this.

## Decision

- Each neighbour weighs Perlin's quintic smootherstep `6t^5 - 15t^4 + 10t^3`, with
  `t = clip((cutoff + 3w - d) / 6w, 0, 1)`: exactly 1 inside `cutoff - 3w`, exactly 0 beyond
  `cutoff + 3w`, 0.5 at the cutoff, continuous to the second derivative at both band edges (Stage 2
  refines by finite differences, so smoothness there matters). `w` is the existing
  `smoothing_width`; no new parameter. Mirrored in `objective/energy_jax.py`; its gradient is
  checked against finite differences inside the band.
- **Whenever the gray band is empty, the soft count equals the integer count.** This replaces
  ADR-0004's three-widths rule of thumb: the condition on a measured cutoff is now exact — no atom
  of the reference within `cutoff ± 3w`. All thirteen references satisfy it for cation centres;
  for anion centres only diopside's O(X) does not (its O–cation gap is 0.446 A, narrower than the
  0.6 A band), which is why anion centres stay report-only for the gray band (ADR-0014).
- The integrality penalty is kept for now. Inside the band the gray-band penalty already acts, and
  outside it the count is integral, so integrality can only act where the band does. Measured on
  the twelve recorded impostors, removing it changes no verdict (8 of 12 rejected either way). It
  is left in until the x4 gate confirms the same, then removed by a separate decision.

## Consequences

- Every reference structure now scores exactly zero in its coordination terms; pyrite's xfail is
  released.
- Outside the band the count carries no gradient. The logistic's slope there was below
  `expit(-3) = 0.047` per neighbour and fell exponentially; CMA-ES does not use gradients, and
  Stage 2 starts from Stage 1's point.
- Impostors that were exactly zero stay zero. Three recorded impostors (diopside, forsterite ×2)
  and topaz's latest now score 1e-12 to 5e-10 above their published structures — margins at the
  optimiser's own tolerance (`refine_ftol = 1e-10`). Whether that separation is real is decided by
  re-running the x4 global-minimum gate, not by the regression gate.
- Second refinements, scored with constraints measured on the first (ADR-0007's tautology check):
  corundum_R 0.000000, anatase_origin1 0.000019, forsterite_Pnma 0.000011 — recorded, not judged.

## Measured: the x4 global-minimum gate with this change alone (2026-10-01)

Published structures and the points reached now both score exactly zero, and the lattices
reached moved *further* from the published ones on several minerals: anatase c −12%, corundum
c +11.6%, quartz a −5.0%, wurtzite c +4.8% (fluorite, sphalerite, pyrite, spinel within 1%).
The logistic's tails had been leaving a slight slope toward the published structures; compact
support removes it and widens the zero-error plateau. The change is kept for what it fixes (an
exact integer count, pyrite's reference, the parking hole), and is not by itself a uniqueness
measure. The run was stopped once this was clear and the bond-valence term (ADR-0016) measured
on top of it.

## Amended by ADR-0020

The integrality penalty's removal is decided by ADR-0020's term ablation: configuration (iv), the
current code without it, against (iii). It is removed if Stage-1 convergence and tier 1 match.

The ablation decided it: the integrality penalty is removed (ADR-0020's "Decided after the
ablation").
