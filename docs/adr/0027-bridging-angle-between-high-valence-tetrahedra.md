# The bridging angle between high-valence tetrahedra is constrained for any composition

ADR-0020's CONT measurement left tier 1 blocked by zero-error end points of other classes: 12 of
13 converged end points on adamite and topaz. Diopside's recorded points are the same. The
question (design review, 2026-10-14) was whether constraints on cation-cation and anion-anion
geometry separate them, written generically for any composition.

## Measured

On the 13 references and every recorded end point (`mizuho/bond_graph.py` cutoffs):

- **A plain cation-cation minimum distance cannot separate them.** Correct structures reach
  2.655 A (corundum Al-Al, face-sharing) and 2.695 A (forsterite Mg-Si, edge-sharing); other-class
  points span 2.25-2.77 A. Any single threshold rejects a reference or passes a point.
- **Other-class points share edges with a high-valence tetrahedron.** Topaz: Al-Si edges, Al-O-Si
  80.2-84.2 deg (published: corners only, 127.3 deg). Adamite: As-Zn edges, As-O-Zn 80.3-88.1 deg
  (published: corners only, 115.3 deg). Diopside: Si-Si edges at 80.0 deg, Mg-Si edges at 82.4-87.3
  deg (published: Si-O-Si 135.9 deg, Mg-Si corners only). Most sit at 80.0-80.5 deg, the floor of
  the generic auto X-O-X band [80, 160] (`auto_coupling_angle_range`, not a cited value), which is
  where an undistorted tetrahedron and octahedron sharing an edge put the bridging angle (80.3 deg).
- **Correct structures do share edges with tetrahedra**, at no less than 90.3 deg: forsterite
  Mg-O-Si 90.3, wollastonite Ca-O-Si 90.3, diopside Ca-O-Si 91.3. Between two tetrahedra the
  references realise 135.9-141.9 deg.
- **An anion-anion floor separates nothing.** The shortest polyhedral edge the first-shell bands
  allow (2 d_min sin(theta_min / 2)) is below every reference's shortest anion-anion distance, as
  it must be, and also below every converged other-class point's (topaz floor 2.203 A, points from
  2.243 A; adamite 2.362 A, points from 2.460 A; diopside 2.375 A, points from 2.375 A).
- The hard-sphere overlap term does not act as cation-cation or anion-anion repulsion: its radii
  (O 0.66, Si 0.40 A) only bite below 1.32 A (O-O) and 0.80 A (Si-Si).

## Decision

- **(A) Tetrahedron-tetrahedron, generic and automatic.** For every anion X that two or more
  cations may share, where each cation has formal valence >= 4 and a coordination constraint
  citing CN exactly 4 to X, an X-centred angle constraint holds the T-X-T angle to [120, 180] deg
  (`config.augment_tetrahedral_bridging`, called by `build_sg_config`). It applies to silicates,
  germanates, phosphates, arsenates and sulfates alike, from inputs the user already states. The
  cutoff is the largest of those T-X shells; an explicit constraint over the same neighbours wins;
  it replaces the generic auto X-O-X band where both cover the same neighbours (quartz);
  `auto_tetrahedral_bridging: false` switches it off. Every reference still scores 0.
  - Source: Si-O-Si angles below 120 deg are uncommon (Liebau 1961, as cited by Gibbs, Prewitt &
    Baldwin, *Am. Mineral.* 57, 1578, 1972); tetrahedra of small high-charge cations share corners
    only (Pauling's third rule). The 180 deg ceiling is geometric. Not yet read in the original:
    the hosts serving both papers are blocked from this environment.
  - Diopside's hand-written constraint of the same form is replaced by the automatic one; the
    objective is identical at the reference and at every recorded point.
- **(B) Tetrahedron-other cation: on hold.** A floor of about 90 deg would reject nearly every
  other-class point, but 90 deg is read off these 13 references, and the margin is 2 deg (88.1
  against 90.3). Pauling's third rule says why edge sharing with a high-charge cation widens the
  bridging angle (cation-cation repulsion shortens the shared edge); no quantitative bound was
  found to cite. It waits for one.
- **No anion-anion term.** Measured to separate nothing above.

## Consequences

- The objective has no cation-cation repulsion. The bands (A) and, if a source is found, (B) stand
  in for it on the bridging angle; this is recorded as the mechanism, not hidden.
- Diopside is re-measured with (A) at x4 over seeds 42-46 (ADR-0020's five-seed rule). Recorded
  here when it completes.
