# Cutoffs and angle bands are measured from the reference structure; `cn_min`/`cn_max` are cited

Two different YAML keys decide which neighbours a term looks at:
`coordination_constraints.cutoff_distance` and `bond_constraints.cutoff` (whose default is
`distance_max * 1.5`). Both are the same physical quantity — where the first coordination shell
ends — so both are re-derived by measuring the reference structure of ADR-0003 and placing the
cutoff at the midpoint of the gap between the first and second shells.

Neither could be carried over unexamined, and both were wrong in ways that only showed up once the
objective was evaluated at a real structure:

- **Coordination cutoffs were too tight.** Spinel's Mg–O sat at 2.2 A, inside the 1.918 → 3.384 A
  gap but only 0.28 A above the first shell. The coordination count is a logistic soft count with a
  0.10 A smoothing width, so it had not saturated: it reported 3.78 where the structure plainly has
  4, and charged a permanent penalty at the correct answer. A cutoff needs at least ~3 smoothing
  widths of clearance above the first shell, which the measured midpoint gives automatically.
- **Bond cutoffs were too loose.** Diopside's Ca–O default of `2.85 * 1.5 = 4.275 A` swept in the
  entire second shell (which starts at 3.470 A) and then penalised those 40 non-bonded contacts for
  not being 2.3–2.85 A. That single term contributed **12.13 of a 12.13 total** at the experimental
  structure. The bond term's own docstring already warned about exactly this failure mode; nothing
  had ever checked it against a real structure.

**Angle bands are measured too.** Adamite's `O-Zn-O` band of 75-120 deg was 34.87 of its residual:
the reference structure realises 78.6-125.9 deg, because Zn1 is a trigonal bipyramid whose
equatorial angles are ~120 deg and legitimately exceed that ceiling. Bands are read off the angles
the term actually evaluates, i.e. after `ignore_trans_above` has removed the trans angles (an
octahedron's ~180 deg ones), and rounded outward.

The same too-tight cutoff appears a third time in the O/F-centred constraints that
`augment_coupling_constraints` injects automatically, whose cutoff is `max(bond distance_max)` for
that anion. In topaz that put the F cutoff 0.15 A past the real Al-F bond, leaving the soft count
unsaturated and charging 0.1235 at the correct structure. Those fixtures now state the anion
constraint explicitly with a measured cutoff, which suppresses the auto-injection.

With all three measured, the objective at the reference structure falls to 0.000055 (spinel),
0.000023 (diopside), 0.000003 (topaz) and 0.0292 (adamite).

`cn_min` and `cn_max` are treated differently and are **not** re-fitted. A coordination number is a
cited chemical fact about the mineral, so re-deriving it from our own model would let the model
choose its own exam questions. If a cited coordination number cannot be reproduced at the reference
structure, that is a finding about the objective, not a licence to move the target.

## What a residual still means

Adamite keeps 0.0292 from its bond terms and a separate 80.00 from charge balance. Neither is a
defect in the objective:

- The 80.00 is the CIF's undetermined hydrogen (`calc_flag='dum'`, coordinates -1,-1,-1). Without
  the 4 H the cell carries a net -4, and `5.0 * 4**2 = 80` exactly. The charge term simply cannot
  be evaluated at this reference.
- The 0.0292 is the 1937 refinement's precision: its As-O distances span 1.588-1.814 A, wider than
  the cited 1.60-1.75 A band. A modern refinement would likely close it.

Both are recorded rather than tuned away, since moving a target to absorb a reference structure's
own limitations is exactly what this ADR forbids.

## Amendment: when a band may be re-measured, and how it is rounded

This ADR left two things unstated, both settled later.

**A band stays cited unless the reference contradicts it.** Bond bands (`distance_min` /
`distance_max`) and angle bands are cited chemical facts by default. When the reference structure
falls **outside** the band, the band is wrong: it is replaced by the measured range and the reason
is recorded in a comment beside it, so the fact that the citation failed survives. Wollastonite's
Ca-O band of [2.30, 2.60] against a measured 2.266-2.639 A is such a case, and so is the O-Si-O
band of [100, 120] deg against a measured 102.7-122.4 deg.

**Rounding is outward and mechanical.** Distances round outward to two decimals (2.266 -> 2.25,
2.639 -> 2.65); angles round outward to 5 deg (102.7 -> 100, 122.4 -> 125). Leaving a band exactly
at the measured extremes puts the reference on the boundary, where the sign of a floating-point
error decides the verdict. A percentage margin was rejected here as an unjustified free parameter.

**A band that is far too WIDE is not caught by any of the above**, since the reference sits
comfortably inside it. ADR-0007 adds the missing rule — a bond band is the reference span widened
by 2% of the bond length — and explains why: topaz's bands were 15-33x wider than the chemistry
they describe, and an impostor used exactly that slack.
