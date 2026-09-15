# Coordination number counts every periodic image, not the nearest one per site

`lattice.pairwise_distances_pbc` returns one number per (centre, neighbour) **atom pair**: the
distance to that neighbour's nearest periodic image. Every distance-based penalty term consumed
that matrix, so a neighbour site whose coordination shell contains several of its own periodic
images was counted **once**. Measured against structures whose coordination number is a textbook
fact, the objective reported NaCl Na–Cl as **3 instead of 6**, CsCl Cs–Cl as **1 instead of 8**,
and fcc Cu as **3 instead of 12** (CaF2 and diamond Si happen to come out right, because there
every first-shell neighbour is a distinct atom in the cell). We therefore count **every periodic
image inside the cutoff**, which is the standard crystallographic definition of coordination
number and reproduces all five reference values exactly.

## Consequences

- A coordination-number target can now exceed the number of neighbour **atoms** in the cell. Under
  the old rule `CN <= N_neighbour_atoms` held by construction, which made
  `wollastonite_casio3.yaml` (Z=1, 3 oxygen atoms, Si needs 4 and Ca needs 6) carry a permanent,
  irreducible error floor. Counting images removes that floor: Ca CN=8 and Si CN=4 are
  simultaneously reachable inside that fixture's own `lattice_bounds`.
- An atom now sees its **own** images. Only the identity image (same atom, zero shift) is excluded.
  This is required, not incidental: fcc Cu's twelve nearest neighbours *are* images of the atom
  itself, and the hard-sphere overlap term can only detect a cell collapsing onto its own periodic
  copy if self-images are visible. Before this change a single Na atom in a cubic cell could shrink
  to a = 0.2 A with an overlap penalty of exactly 0.0.
- Every fixture's error value changes. Existing numbers are not a regression baseline; see
  ADR-0003 for what replaces them.
