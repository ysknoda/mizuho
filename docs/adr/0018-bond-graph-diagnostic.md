# A bond-graph diagnostic separates topology from geometry

ADR-0017 ended with zero-error impostors on quartz, corundum, forsterite and diopside, and with
the claim that what remains is "a volume-preserving change of shape". That claim was read off the
lattice alone. Before deciding what information to add next (design review Q4), the impostors
need comparing with the published structures term by term: same connectivity or not, and if so,
which bond lengths, angles and polyhedral distortions moved.

## Decision

`mizuho compare cfg.yaml ref.cif cand...` (`mizuho/bond_graph.py`). A diagnostic only: nothing
here enters the objective.

- **Bond graph.** Atoms are vertices; an edge joins two atoms (any of the 27 periodic images,
  ADR-0001) within their pair's cutoff, with the cell translation on the edge (the periodic
  quotient graph). The cutoff is the config's own, most specific first: a coordination constraint
  naming the pair, then one reaching it through the wildcard `X`, then a bond constraint's
  `cutoff`; within a level the largest. A wildcard cutoff is set for all of an anion's cations at
  once and must not override one measured for the pair: under "largest wins" diopside's 2.73 A
  Si-O contact, outside Si(O) 2.425 A but inside O(X) 2.942 A, made a five-coordinate silicon.
  Contacts between the graph's cutoff and the largest cutoff any constraint applies are reported
  as **grey-band contacts**: the objective's constraints disagree whether they are bonds.
- **Topology class.** Each atom's coordination sequence to `--depth` bonds (default 10;
  Brunner & Laves 1971, O'Keeffe 1995) and each cation's polyhedral linkage (cation pairs sharing
  1, 2, >= 3 anions). Equal (element, sequence) defines a topological atom type; the class is the
  multiset of types, normalised by their gcd so supercells and atom order do not matter. Distinct
  classes are certainly distinct nets; equal classes are equal to that depth (coordination
  sequences are an invariant, not a complete one — no isomorphism test is attempted, and no graph
  library is used).
- **Geometry, per type label** (element plus first-shell composition), so two structures line up
  without an atom correspondence or a common setting: bond lengths, all bond angles, Baur's (1974)
  distortion index for every cation polyhedron, Robinson, Gibbs & Ribbe's (1971) quadratic
  elongation and angle variance for tetrahedra and octahedra, and the Lagrangian cell strain split
  into volume and the norm of its deviatoric part (same space group only). Compared also across
  classes when the label sets agree, since a difference far out in the net can hide a local one.

Verified against fixed values: rock salt N_k = 4k^2 + 2 and 12 edges + 6 corners shared; quartz Si
4, 12, 30, 52, 80 (RCSR qtz); corundum one face, three edges, nine corners; regular polyhedra
lambda = 1, sigma^2 = 0; the elongated-octahedron and flattened-tetrahedron closed forms
(`tests/unit/test_bond_graph.py`).

## Measured (fixture cutoffs, depth 10)

| impostor | class | volume | shape | what moved |
| --- | --- | --- | --- | --- |
| quartz round 4 | same | +1.1 % | 0.041 | SiO4 sigma^2 1.2 -> 12.1, O-Si-O 105.1-112.7 deg, inside the 105-115 band. The independent SG154 refinement has sigma^2 10.1 |
| corundum round 3 / 4 | same | +0.0 / −0.2 % | 0.108 / 0.143 | Al-O-Al up to 5.9 deg, AlO6 sigma^2 67 -> 88-98 |
| forsterite round 4 | same | +0.0 % | 0.122 | Mg2-O1 2.045 -> 2.224 A, Mg-O-Si 122.5 -> 143.1 deg, MgO6 sigma^2 89 -> 200 |
| diopside x8 | different from shell 2 | −2.9 % | 0.112 | Si shares 3 corners + 4 edges (published 7 + 2) |
| diopside x16 | different from shell 4 | +1.9 % | 0.112 | first shells and linkage identical; MgO6 sigma^2 17 -> 590 (cis 67.8 deg, trans 141 deg); eight grey-band Si-O 2.728 A |
| topaz round 2, adamite rounds 1 and 2 | different | +13.8, +6.1, +10.4 % | | — |
| adamite round 2 seedfix | same | +19.2 % | 0.017 | bonds up to 16 %, angles up to 13 deg |
| anatase rounds 1-3 (now rejected) | same | +5.7 to +9.4 % | | |

Three conclusions, for whoever designs the next term:

1. The same-topology impostors are not "shape only" in the cell; they are **polyhedral
   distortion** — the angle variance grows 2-10x while every angle stays inside its band. Angle
   bands are per-angle; nothing measures a polyhedron's distortion as a whole.
2. Diopside's x16 point keeps every first shell and distorts MgO6 almost beyond recognition; no
   Mg angle constraint exists.
3. Diopside x8 and the earlier topaz and adamite (rounds 1, 2) impostors are **different nets**: there the first
   shell is wrong, not merely distorted.

## Consequences

- Deciding which term to add next is now a matter of reading this table, not of reconstruction.
- Grey-band contacts expose disagreements between a fixture's own cutoffs; diopside's Si-O band
  (2.425, 2.942] is one.
- A JSON record is accepted only against a reference whose vector equals its `published_vector`;
  references reconstructed in code (spinel, diopside) go through `write_cif` first.

## Amended by ADR-0019

The bond graph is no longer only a diagnostic: its topology class groups Stage-1 endpoints,
stops restarts early and de-duplicates the ranking (`mizuho/optimize/endpoints.py`). The
objective is unchanged. The measured table above predates ADR-0019's DFT test, which found the
same-class impostors to be the published structure and two of the three diopside points too.
