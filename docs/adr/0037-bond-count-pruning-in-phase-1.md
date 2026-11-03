# Phase 1 drops Wyckoff assignments that bond counting proves cannot reach zero error

Phase 2 optimises up to `tree_search.max_candidates` assignments per (space group, composition)
(ADR-0033). In the Al-O-N exhaustive run (`runs/alon`, Al CN [4, 6], O [2, 4], N [3, 4]) a
prototype check found assignments whose multiplicities alone rule out the coordination numbers:
one Al orbit of multiplicity 1 cannot share bonds with one O orbit of multiplicity 8 when O needs
two to four of them. Such assignments take Phase-2 slots and optimiser time and can never score
zero. The check had to be exact: an assignment it drops must be one no coordinates can bring to
zero error, decided from measurements rather than assumed.

## Decision

Phase 1 rejects an assignment when no integer bond counts satisfy two conditions together with
every coordination range the objective can only meet at an integer count
(`mizuho/search/bond_counting.py`).

- **Bonds.** A cation-anion pair is bonded when closer than the cutoff of the first
  cation-centred coordination constraint that counts that anion. The relation is symmetric and
  invariant under every isometry, so the space group maps bonds onto bonds.
- **Condition 1, orbit bond balance.** All atoms of orbit i have the same number n_ij of bonds
  to orbit j, and counting the bonds between the two orbits per conventional cell from either
  end gives m_i n_ij = m_j n_ji (m the Wyckoff multiplicities).
- **Condition 2, site symmetry.** The site-symmetry group S_x of an atom x fixes x and maps
  orbit j and distances onto themselves, so x's neighbours on orbit j are a union of S_x-orbits.
  Grouping the bonds between two orbits by space-group orbit: a bond (x, y) with stabiliser
  S_x & S_y of order h gives |S_x|/h bonds at x and |S_y|/h at y, so
  n_ij/|S_x| = n_ji/|S_y| = sum of 1/h over the bond orbits. Every h is the order of a stabiliser
  in S_x of some point of orbit j and of one in S_y of some point of orbit i. Condition 1 follows
  from this.
- **Stabilisers from mizuho's own tables.** S_x is read off `SYMMETRY_OPS_OF_SG` for each
  `WYCKOFF_DB` position (asserting |S_x| m = number of operations, all 1,731 positions). The
  stabiliser of x + d in S_x is {R : R d = d}; it is exactly the set of elements fixing its
  fixed subspace pointwise, so the fixed subspaces of S_x's elements, closed under intersection,
  list every stabiliser of a point other than x. One is reachable on orbit j only if x + K meets
  the orbit: some image g(c_j + A_j v) + L lies on c_i + K, decided exactly as
  d in K + span(R_g A_j) + Z^3 with the integer annihilator of that subspace. Every coordinate
  constant and translation in the tables is a multiple of 1/24 (checked), so all of it is
  integer arithmetic. No external Wyckoff table.
- **Which ranges apply, at exactly zero error.**
  - A cation-centred count is an exact integer count of bonds. Its gray-band penalty is zero
    only when no neighbour lies within 3 smoothing widths of the cutoff, and there the soft
    count weighs 1 below the cutoff and 0 above it. A constraint that also counts non-anions
    keeps only its upper bound. A later constraint giving the same pair another cutoff is
    dropped: fewer constraints are a weaker, still necessary, condition.
  - An anion-centred count is soft, so each side carries over only when it provably applies to
    the integer count. The upper bound carries over when cn_max + 1 bonded contacts must weigh
    more than cn_max. Bonded contacts lie at or below the cation cutoff minus its half-width,
    outside the windows (distance_max, cutoff] that the pair's bond constraints forbid. The
    lower bound carries over when every cation it counts is bonded through a cation constraint
    and every non-bonded contact weighs exactly 0.
  - In the Al-O-N config every range applies. O's cutoff of 2.255 A ends its full-weight zone at
    1.955 A, the end of the Al-O band. The Al-O bond constraint forbids (1.955, 2.255], so no
    bonded Al-O contact weighs less than 1. bonded_anions (ADR-0028 (a)) enters through the CN-0
    constraint of each non-bonded pair.
- **Rules per composition.** The ranges are read from the constraints `build_sg_config` hands
  the objective, from ADR-0028's derivation and the anion coupling. In the composition-free
  search they are derived per reduced formula. A test asserts the rules equal those of
  `build_sg_config`.
- **Where.** At the accepted leaf of both strategies (`_leaf_status`), so a rejected leaf does
  not count against `max_candidates`. There letters stand for their shape groups, and condition
  2 takes the union over the letters each site could take, so a representative is rejected only
  when every concrete choice would be. After `expand_letter_choices` each concrete assignment is
  checked on its own.
- **Never guessing.** An instance that exceeds the search's work limits (20,000 rows per cation
  orbit, 200,000 nodes) is kept. `tree_search.bond_count_pruning: false` switches the pruning off
  (default on).

## Measured (Al-O-N config, `runs/alon/alon.yaml`, max_atoms 21, max_sites 5)

`scripts/measure_bond_count_pruning.py`, one process.

- **Phase 1, all 230 groups.** 926,790 concrete assignments, 112,331 removed (12.1%): 31,095 by
  condition 1, 81,236 more by condition 2.

  | Crystal system | Removed |
  |---|---|
  | Triclinic | 0% |
  | Monoclinic | 0% |
  | Orthorhombic | 7.7% |
  | Tetragonal | 18.9% |
  | Trigonal | 47.7% |
  | Hexagonal | 54.9% |
  | Cubic | 68.5% |

  Six groups lose every assignment: 202, 209, 218, 223, 225 and 226. Pmmm (47) keeps 272,256 of
  319,824. By composition the share removed runs from 4.4% (Al3O3N) to 88.5% (Al8O3N6). In every
  group, both strategies return exactly the unpruned enumeration minus what the check rejects.
- **Cost.** Enumeration time summed over the 230 groups goes from 17.9 s to 67.4 s with the
  check (Pmmm 2.8 s to 14.6 s). This is against Phase 1's feasibility ranking, which took about
  3,000 s on Pmmm alone in production and now ranks 15% fewer assignments there.
- **The run's optimised set.** The record holds 17,843 (group, assignment) pairs. The check
  rejects 1,990 by condition 1 (the prototype's number) and 3,666 by condition 2, 31.7% in all.
  It rejects none of the 3,169 that reached zero error (<= 1e-3).
- **The premises at the recorded points.** All 38,975 recorded entries were rebuilt with every
  periodic image in reach.
  - 86 entries, all non-zero error, have an orbit collapsed onto a special position: fewer atoms
    than the multiplicities say. They are left out of the checks below.
  - In every other entry, every orbit has equal per-atom bond counts, m_i n_ij = m_j n_ji holds,
    and n_ij is in condition 2's set.
  - In all 4,735 zero-error entries the integer counts meet every range the check uses.
- **References.** All 13 ADR-0003 gate structures pass, under the fixture's constraints and
  under the derived ones (`test_bond_count_pruning_keeps_the_reference_assignment`).
- **Property test.** In 12 groups (P-1 to Ia-3d), random structures at every distance threshold
  between shells obey both conditions (`tests/unit/test_bond_counting.py`).

## Limits

- **Exact zero versus the 1e-3 threshold.** The argument is exact at zero error. Zero-error
  classes are judged at 1e-3, where a contact may sit slightly inside a gray band or a band end
  (corundum's 1.972 A Al-O, 0.017 A past the band). The record shows no rejection of a class
  within 1e-3, and the integer counts at all 4,735 such points meet the ranges. This is
  measured, not proven, for the tolerance.
- **The objective's periodic images.** It counts the 27 images around the fractional minimum
  image. That covers every contact within reach when every interplanar spacing exceeds
  (cutoff + 3 widths) / 1.5. Two zero-error entries in P-1 fall below that sufficient bound
  (1.67 and 1.75 A against 1.80 A); their measured counts still meet every range.
- **Collapsed orbits.** A structure whose free parameters reach a special position builds fewer
  atoms than its assignment states (`expand_orbit` merges points within 1e-4). It is a different
  assignment and is not covered. None of the zero-error entries is one.
- **Plan numbering.** `docs/plan-polymorph-validation-campaigns.md` still names ADR-0035-0037 for
  the union bond band, the primitive-cell bound and C-C bond valence. 0035 and 0036 went to other
  decisions and this is 0037, so those need new numbers.
