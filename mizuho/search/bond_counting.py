"""Exact Phase-1 pruning of Wyckoff assignments by bond counting (ADR-0037).

Two necessary conditions on a Wyckoff assignment, decided from the space
group's operations and the Wyckoff multiplicities alone, before Phase 2
optimises a single coordinate:

1. Orbit bond balance. The space group carries every atom of orbit i onto
   every other one, so all of them have the same number n_ij of bonded
   neighbours on orbit j. Counting the bonds between the two orbits per
   conventional cell from either end gives m_i n_ij = m_j n_ji, m the
   Wyckoff multiplicities.
2. Site symmetry. The site-symmetry group S_x of an atom x on orbit i fixes x
   and maps orbit j and every distance onto themselves, so x's n_ij
   neighbours on orbit j are a union of S_x-orbits. Grouped by space-group
   orbit of bonds instead: a bond (x, y) whose stabiliser S_x & S_y has order
   h contributes |S_x|/h bonds at x and |S_y|/h at y, so
   n_ij/|S_x| = n_ji/|S_y| = sum over the bond orbits of 1/h, every h the
   order of a subgroup of S_x that fixes some point of orbit j and of a
   subgroup of S_y that fixes some point of orbit i (`bond_stabiliser_orders`).
   Condition 1 follows from this one.

The bonds are the atom pairs closer than the cation-centred cutoff of their
pair. A coordination constraint the objective can only meet at an integer
count of those bonds gives a range for sums of n_ij (`bond_rules`, whose
docstring has the argument); an assignment for which no integer n_ij meet
every range under conditions 1-2 cannot reach zero error at any coordinates,
and is not handed to Phase 2.

Everything is exact integer arithmetic: every coordinate constant and
translation in mizuho's tables is a multiple of 1/24 (`_D`), so coordinates
are held as integers in units of 1/24.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache
from itertools import product
from math import ceil, floor, gcd
from typing import Optional

import numpy as np

from ..composition import parse_composition
from ..data import WYCKOFF_DB, WyckoffEntry
from ..data.symmetry_ops import SYMMETRY_OPS_OF_SG
from ..objective.penalties import (
    GRAY_BAND_WIDTHS,
    _expand_wildcard,
    _smootherstep_weight,
)

# Common denominator of every coordinate constant and operation translation.
_D = 24
_E = ((1, 0, 0), (0, 1, 0), (0, 0, 1))

Vec = tuple[int, int, int]
# A linear subspace of R^3 with an integer basis, in canonical form:
# (3, None) everything, (2, n) the plane with primitive normal n, (1, u) the
# line along the primitive vector u, (0, None) the origin.
Subspace = tuple[int, Vec | None]
_ALL: Subspace = (3, None)
_ZERO: Subspace = (0, None)

BALANCE = "balance"
SITE_SYMMETRY = "site_symmetry"


# ---------------------------------------------------------------------------
# Integer linear algebra in R^3
# ---------------------------------------------------------------------------


def _scaled(x: float) -> int:
    v = round(x * _D)
    if abs(v - x * _D) > 1e-6:
        raise ValueError(f"{x} is not a multiple of 1/{_D}")
    return v


def _dot(a, b) -> int:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b) -> Vec:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _matvec(R, v) -> Vec:
    return (_dot(R[0], v), _dot(R[1], v), _dot(R[2], v))


def _primitive(v) -> Vec:
    """v divided by the gcd of its entries, first non-zero entry positive."""
    g = gcd(gcd(abs(v[0]), abs(v[1])), abs(v[2]))
    v = tuple(x // g for x in v)
    first = next(x for x in v if x)
    return v if first > 0 else tuple(-x for x in v)


def _span(vectors) -> Subspace:
    vs = [tuple(v) for v in vectors if any(v)]
    if not vs:
        return _ZERO
    normal = next((c for c in (_cross(vs[0], v) for v in vs[1:]) if any(c)), None)
    if normal is None:
        return (1, _primitive(vs[0]))
    if all(_dot(normal, v) == 0 for v in vs):
        return (2, _primitive(normal))
    return _ALL


def _basis(K: Subspace) -> tuple[Vec, ...]:
    """Integer vectors spanning K (u x e_k span the plane normal to u)."""
    dim, v = K
    if dim == 3:
        return _E
    if dim == 2:
        return tuple(c for c in (_cross(v, e) for e in _E) if any(c))
    if dim == 1:
        return (v,)
    return ()


def _kernel(M) -> Subspace:
    """Null space of an integer 3x3 matrix, from its row space."""
    dim, v = _span(M)
    if dim == 0:
        return _ALL
    if dim == 1:
        return (2, v)
    if dim == 2:
        return (1, v)
    return _ZERO


def _intersect(K1: Subspace, K2: Subspace) -> Subspace:
    (d1, v1), (d2, v2) = K1, K2
    if d1 == 3:
        return K2
    if d2 == 3:
        return K1
    if d1 == 0 or d2 == 0:
        return _ZERO
    if d1 == 2 and d2 == 2:
        return K1 if v1 == v2 else (1, _primitive(_cross(v1, v2)))
    if d1 == 1 and d2 == 1:
        return K1 if v1 == v2 else _ZERO
    plane, line = (v1, v2) if d1 == 2 else (v2, v1)
    return (1, line) if _dot(plane, line) == 0 else _ZERO


def _in_span_plus_lattice(d: Vec, vectors) -> bool:
    """Whether d/_D lies in span(vectors) + Z^3. For a rational subspace W,
    W + Z^3 is the set on which every integer functional vanishing on W
    takes integer values; W's integer annihilator is Z n for a plane with
    primitive normal n, and is spanned by u x e_k for a line along a
    primitive u."""
    dim, v = _span(vectors)
    if dim == 3:
        return True
    if dim == 2:
        return _dot(v, d) % _D == 0
    if dim == 1:
        return all(_dot(_cross(v, e), d) % _D == 0 for e in _E)
    return all(x % _D == 0 for x in d)


# ---------------------------------------------------------------------------
# Site-symmetry groups and the stabilisers of bonds
# ---------------------------------------------------------------------------


@cache
def _ops(sg: int):
    rotations, translations = SYMMETRY_OPS_OF_SG[sg]
    return tuple(
        (tuple(tuple(int(x) for x in row) for row in R), tuple(_scaled(x) for x in t))
        for R, t in zip(rotations, translations)
    )


def _entry(sg: int, letter: str) -> WyckoffEntry:
    return next(e for e in WYCKOFF_DB[sg] if e.letter == letter)


def _template(entry: WyckoffEntry) -> tuple[Vec, tuple[Vec, ...]]:
    """(c, A): the position is c/_D + A u, one column of A per free parameter."""
    c, cols = [], {p: [0, 0, 0] for p in entry.free_params}
    for k, slot in enumerate(entry.coord_template):
        if isinstance(slot, tuple):
            coeffs, offset = slot
        elif isinstance(slot, dict):
            coeffs, offset = slot, 0.0
        else:
            coeffs, offset = {}, slot
        c.append(_scaled(offset))
        for p, a in coeffs.items():
            cols[p][k] += int(a)
    return tuple(c), tuple(tuple(cols[p]) for p in entry.free_params)


@dataclass(frozen=True)
class SiteSymmetry:
    """The site-symmetry group S of a Wyckoff position, by its linear parts
    (fractional basis), with the fixed subspaces of its subgroups.

    The stabiliser in S of a point x + d is {R in S : R d = d}. Every such
    stabiliser H fixes K_H, the intersection of ker(R - I) over R in H, and
    is exactly the set of elements fixing K_H pointwise; so the fixed
    subspaces of S's elements, closed under intersection, list every
    stabiliser of a point other than x (`subspaces`, the origin left out,
    with `stabiliser_orders` |H| for each)."""

    order: int
    rotations: tuple
    subspaces: tuple[Subspace, ...]
    stabiliser_orders: tuple[int, ...]


@cache
def site_symmetry(sg: int, letter: str) -> SiteSymmetry:
    entry = _entry(sg, letter)
    c, A = _template(entry)
    ops = _ops(sg)
    # The operations fixing every point of the position, modulo the
    # lattice: R A = A and c - R c - t integer.
    rotations = [
        R
        for R, t in ops
        if all(_matvec(R, a) == a for a in A)
        and all((ci - rci - ti) % _D == 0 for ci, rci, ti in zip(c, _matvec(R, c), t))
    ]
    if len(rotations) * entry.multiplicity != len(ops):
        raise AssertionError(
            f"SG {sg} {letter}: {len(rotations)} site operations x "
            f"multiplicity {entry.multiplicity} != {len(ops)} operations"
        )
    subspaces = {
        _kernel(tuple(tuple(R[a][b] - _E[a][b] for b in range(3)) for a in range(3)))
        for R in rotations
    }
    frontier = set(subspaces)
    while frontier:
        new = {_intersect(K1, K2) for K1 in frontier for K2 in subspaces} - subspaces
        subspaces |= new
        frontier = new
    subspaces.discard(_ZERO)
    ordered = tuple(sorted(subspaces, key=lambda K: (-K[0], K[1] or ())))
    orders = tuple(
        sum(all(_matvec(R, b) == b for b in _basis(K)) for R in rotations)
        for K in ordered
    )
    return SiteSymmetry(len(rotations), tuple(rotations), ordered, orders)


@cache
def contact_stabiliser_orders(sg: int, letter_i: str, letter_j: str) -> frozenset[int]:
    """Orders of the stabilisers, in the site-symmetry group S of a point x
    on `letter_i`, that a point y of `letter_j`'s orbit can have: a superset
    of every |S & S_y| over y != x in the orbit, whatever the free
    parameters of either position.

    A stabiliser H with fixed subspace K is reachable only if x + K meets
    the orbit, i.e. some image g(c_j + A_j v) + L of the position lies on
    c_i + K (K contains the position's own directions A_i, so x's free
    parameters do not move it). An H fixing x alone (K the origin) is never
    reachable: y would sit on x."""
    site = site_symmetry(sg, letter_i)
    ci, _ = _template(_entry(sg, letter_i))
    cj, Aj = _template(_entry(sg, letter_j))
    images = set()
    for R, t in _ops(sg):
        cp = _matvec(R, cj)
        d = tuple((cp[k] + t[k] - ci[k]) % _D for k in range(3))
        images.add((d, _span([_matvec(R, a) for a in Aj])))
    out = set()
    for K, h in zip(site.subspaces, site.stabiliser_orders):
        if h in out:
            continue
        if any(
            _in_span_plus_lattice(d, _basis(K) + _basis(span)) for d, span in images
        ):
            out.add(h)
    return frozenset(out)


@cache
def bond_stabiliser_orders(sg: int, letter_i: str, letter_j: str) -> frozenset[int]:
    """The orders a bond's stabiliser S_x & S_y can have between an atom x
    on `letter_i` and an atom y on `letter_j`: seen from x it is the
    stabiliser of y, seen from y that of x."""
    return contact_stabiliser_orders(
        sg, letter_i, letter_j
    ) & contact_stabiliser_orders(sg, letter_j, letter_i)


@cache
def _semigroup_counts(
    orders: frozenset[int], site_order: int, cap: int
) -> tuple[int, ...]:
    """The bond counts n <= cap of the form site_order * sum(1/h), h in
    `orders` with repetition (n = 0 included)."""
    if not orders:
        return (0,)
    L = 1
    for h in orders:
        L = L * h // gcd(L, h)
    # n = site_order * t / L, t a non-negative combination of the L/h.
    steps = sorted({L // h for h in orders})
    t_max = cap * L // site_order
    reach = [False] * (t_max + 1)
    reach[0] = True
    for t in range(1, t_max + 1):
        reach[t] = any(t >= s and reach[t - s] for s in steps)
    return tuple(
        n
        for n in range(cap + 1)
        if (n * L) % site_order == 0 and reach[n * L // site_order]
    )


def _shape_members(sg: int, letter: str) -> tuple[str, ...]:
    """The letters a shape-group representative stands for (base._shape_key)."""
    from .base import _shape_key

    key = _shape_key(_entry(sg, letter))
    return tuple(e.letter for e in WYCKOFF_DB[sg] if _shape_key(e) == key)


@cache
def _representative_bond_orders(sg: int, rep_i: str, rep_j: str) -> frozenset[int]:
    """bond_stabiliser_orders united over the letters the two
    representatives stand for: a representative assignment is rejected only
    when no letter choice could pass."""
    out = set()
    for li in _shape_members(sg, rep_i):
        for lj in _shape_members(sg, rep_j):
            out |= bond_stabiliser_orders(sg, li, lj)
    return frozenset(out)


# ---------------------------------------------------------------------------
# Coordination ranges an integer bond count must meet at zero error
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CountRule:
    """lo <= (bonds from a `centre` atom to atoms of `partners`) <= hi;
    None where the objective does not imply that side."""

    centre: str
    partners: tuple[str, ...]
    lo: int | None
    hi: int | None


@dataclass(frozen=True)
class BondRules:
    cations: frozenset[str]
    anions: frozenset[str]
    pairs: frozenset[tuple[str, str]]  # (cation, anion) with a bond relation
    cation_rules: tuple[CountRule, ...]
    anion_rules: tuple[CountRule, ...]


def _forbidden_windows(config: dict, present: set, charges: dict, a: str, b: str):
    """Distance intervals (lo, hi] no a-b contact can have at zero error: a
    bond constraint penalises every contact within its cutoff outside
    [distance_min, distance_max]."""
    out = []
    for bc in config.get("bond_constraints", []):
        if bc.get("weight", 1.0) <= 0:
            continue
        a1 = bc["atom1"]
        partners = _expand_wildcard(bc["atom2"], present, a1, charges)
        if not ((a1 == a and b in partners) or (a1 == b and a in partners)):
            continue
        dmax = bc["distance_max"]
        cutoff = bc["cutoff"] if "cutoff" in bc else 1.5 * dmax
        out.append((dmax, float("inf") if cutoff is None else cutoff))
    return out


def _escape_down(x: float, windows) -> float:
    """The largest distance <= x outside every forbidden window."""
    moved = True
    while moved:
        moved = False
        for lo, hi in windows:
            if lo < x <= hi:
                x, moved = lo, True
    return x


def _escape_up(x: float, windows) -> float:
    """A lower bound of the distances >= x outside every forbidden window."""
    moved = True
    while moved:
        moved = False
        for lo, hi in windows:
            if lo < x <= hi:
                x, moved = hi, True
    return x


def bond_rules(config: dict) -> BondRules | None:
    """The coordination ranges of `config` (with its composition) that bound
    integer bond counts at zero error, or None without any.

    The bond relation is per (cation, anion) pair: closer than the cutoff
    of the first cation-centred constraint counting that anion. At zero
    error a cation-centred count is the exact integer count of that
    relation: the gray-band penalty (cation centres, ADR-0024) is zero only
    with no neighbour within 3 smoothing widths of the cutoff, where the
    soft count weighs 1 below and 0 above. A later constraint of the same
    cation giving one of its anions another cutoff is dropped (a subset of
    the constraints is a weaker, still necessary, condition), and one also
    counting non-anions keeps only its upper bound.

    An anion-centred count is soft at zero error, so its range carries over
    side by side. The upper bound cn_max does when cn_max + 1 bonded
    contacts must weigh more than cn_max: every bonded contact lies at or
    below the cation cutoff minus its gray half-width, outside the windows
    the pair's bond constraints forbid (`_forbidden_windows`), and the
    longest such contact is weighed. The lower bound does when every cation
    the constraint counts has a bond relation to the anion and every
    non-bonded contact (at or beyond the cation cutoff plus its half-width,
    outside the windows) weighs exactly 0. Constraints of zero weight say
    nothing and are skipped."""
    charges = config.get("ionic_charges") or {}
    composition = config.get("composition")
    if not composition or not charges:
        return None
    present = set(parse_composition(composition))
    cations = {e for e in present if charges.get(e, 0) > 0}
    anions = {e for e in present if charges.get(e, 0) < 0}
    if not cations or not anions:
        return None

    relation: dict[tuple[str, str], tuple[float, float]] = {}
    cation_rules: list[CountRule] = []
    for cc in config.get("coordination_constraints", []):
        c = cc["center_atom"]
        if c not in cations or cc.get("weight", 1.0) <= 0:
            continue
        neigh = _expand_wildcard(cc["neighbor_atom"], present, c, charges)
        ours = sorted(x for x in neigh if x in anions)
        if not ours:
            continue
        cut = (
            float(cc["cutoff_distance"]),
            GRAY_BAND_WIDTHS * float(cc.get("smoothing_width", 0.10)),
        )
        if any(relation.get((c, x), cut) != cut for x in ours):
            continue
        for x in ours:
            relation[(c, x)] = cut
        lo = ceil(cc["cn_min"] - 1e-9) if len(ours) == len(neigh) else None
        cation_rules.append(CountRule(c, tuple(ours), lo, floor(cc["cn_max"] + 1e-9)))

    anion_rules: list[CountRule] = []
    for cc in config.get("coordination_constraints", []):
        a = cc["center_atom"]
        if a not in anions or cc.get("weight", 1.0) <= 0:
            continue
        neigh = _expand_wildcard(cc["neighbor_atom"], present, a, charges)
        tracked = sorted(x for x in neigh if (x, a) in relation)
        if not tracked:
            continue
        r_a = float(cc["cutoff_distance"])
        w_a = float(cc.get("smoothing_width", 0.10))
        cn_lo, cn_hi = cc["cn_min"], cc["cn_max"]
        hi = floor(cn_hi + 1e-9)
        ub_ok, lb_ok = True, len(tracked) == len(neigh)
        for x in tracked:
            rho, half = relation[(x, a)]
            windows = _forbidden_windows(config, present, charges, x, a)
            # The longest bonded contact and the shortest non-bonded one.
            longest = _escape_down(rho - half, windows)
            shortest = _escape_up(rho + half, windows)
            w_long = float(_smootherstep_weight(np.array([longest]), r_a, w_a)[0])
            w_short = float(_smootherstep_weight(np.array([shortest]), r_a, w_a)[0])
            ub_ok = ub_ok and (hi + 1) * w_long > cn_hi
            lb_ok = lb_ok and w_short == 0.0
        if not (ub_ok or lb_ok):
            continue
        anion_rules.append(
            CountRule(
                a,
                tuple(tracked),
                ceil(cn_lo - 1e-9) if lb_ok else None,
                hi if ub_ok else None,
            )
        )

    if not cation_rules:
        return None
    return BondRules(
        frozenset(cations),
        frozenset(anions),
        frozenset(relation),
        tuple(cation_rules),
        tuple(anion_rules),
    )


def objective_constraints(config: dict) -> dict:
    """The constraints pipeline.build_sg_config hands the objective for this
    config's composition: ADR-0028's derivation and the anion coupling (the
    bridging-angle augmentation adds angles only)."""
    from ..config import augment_coupling_constraints
    from ..constraints import derive_constraints

    return augment_coupling_constraints(derive_constraints(config))


# ---------------------------------------------------------------------------
# Feasibility of one assignment
# ---------------------------------------------------------------------------


def _feasible(
    orbits, rules: BondRules, allowed: Callable[[int, int, int], tuple[int, ...]]
) -> bool:
    """Whether integer bond counts n_ij exist for `orbits` [(element,
    multiplicity)] meeting every rule, n_ij drawn from allowed(i, j, cap),
    each a multiple of m_j / gcd(m_i, m_j), and n_ji = n_ij m_i / m_j."""
    by_cation: dict[str, list[CountRule]] = {}
    for r in rules.cation_rules:
        by_cation.setdefault(r.centre, []).append(r)
    by_anion: dict[str, list[CountRule]] = {}
    for r in rules.anion_rules:
        by_anion.setdefault(r.centre, []).append(r)

    cat = [i for i, (e, _) in enumerate(orbits) if e in by_cation]
    an = [j for j, (e, _) in enumerate(orbits) if e in rules.anions]

    # Every cation orbit's admissible rows (n_ij over its partner orbits),
    # kept as the counts n_ji they give the anion orbits.
    rows_of = []
    for i in cat:
        e_i = orbits[i][0]
        crules = by_cation[e_i]
        partners = [j for j in an if (e_i, orbits[j][0]) in rules.pairs]
        caps = [
            min(r.hi for r in crules if orbits[j][0] in r.partners) for j in partners
        ]
        rows = []
        for combo in product(*(allowed(i, j, cap) for j, cap in zip(partners, caps))):
            sums = [
                sum(n for j, n in zip(partners, combo) if orbits[j][0] in r.partners)
                for r in crules
            ]
            if all(
                (r.lo is None or s >= r.lo) and s <= r.hi for r, s in zip(crules, sums)
            ):
                rows.append(
                    [
                        (j, n * orbits[i][1] // orbits[j][1])
                        for j, n in zip(partners, combo)
                        if n
                    ]
                )
        if not rows:
            return False
        rows_of.append((e_i, rows))
    rows_of.sort(key=lambda x: len(x[1]))

    checks = [(j, r) for j in an for r in by_anion.get(orbits[j][0], ())]
    if not checks:
        return True
    index: dict[int, list[int]] = {}
    for k, (j, _) in enumerate(checks):
        index.setdefault(j, []).append(k)

    def search(level: int, sums: list[int]) -> bool:
        if level == len(rows_of):
            return all(r.lo is None or s >= r.lo for (_, r), s in zip(checks, sums))
        e_i, rows = rows_of[level]
        for row in rows:
            new = list(sums)
            ok = True
            for j, n_ji in row:
                for k in index.get(j, ()):
                    r = checks[k][1]
                    if e_i in r.partners:
                        new[k] += n_ji
                        ok = ok and (r.hi is None or new[k] <= r.hi)
            if ok and search(level + 1, new):
                return True
        return False

    return search(0, [0] * len(checks))


@dataclass
class BondCountCheck:
    """Rejects Wyckoff assignments that cannot reach zero error (module
    docstring). `rules_for(formula)` gives a composition's rules (the
    composition-free search derives them per reduced formula, ADR-0033);
    decisions are cached per (composition, sites)."""

    sg: int
    rules_for: Callable[[str], BondRules | None]
    elements: list[str]
    site_symmetry: bool = True
    _cache: dict[tuple, str | None] = field(default_factory=dict, repr=False)

    @classmethod
    def from_config(
        cls, config: dict, elements: list[str], composition_free: bool
    ) -> Optional["BondCountCheck"]:
        """The check for a search config, or None when it is switched off
        (`tree_search.bond_count_pruning: false`) or nothing can be pruned.
        Rules are derived on first use, as pipeline.build_sg_config derives
        the config of each composition."""
        ts_cfg = config.get("tree_search") or {}
        if ts_cfg.get("bond_count_pruning", True) is False:
            return None
        if not (
            config.get("ionic_charges")
            and (
                config.get("coordination_constraints")
                or config.get("coordination_numbers")
            )
        ):
            return None
        rules_cache: dict[str, BondRules | None] = {}

        def rules_for(formula: str) -> BondRules | None:
            if formula not in rules_cache:
                cfg = dict(config, composition=formula) if composition_free else config
                try:
                    rules_cache[formula] = bond_rules(objective_constraints(cfg))
                except ValueError:
                    # No config can be built for this composition; the
                    # pipeline raises when it tries, so nothing is pruned here.
                    rules_cache[formula] = None
            return rules_cache[formula]

        if composition_free:
            return cls(config["spacegroup"], rules_for, list(elements))
        fixed = config["composition"]
        return cls(
            config["spacegroup"], lambda _formula: rules_for(fixed), list(elements)
        )

    def _formula(self, assignment) -> str:
        counts: dict[str, int] = {}
        for a in assignment:
            counts[a.element] = counts.get(a.element, 0) + a.entry.multiplicity
        g = 0
        for n in counts.values():
            g = gcd(g, n)
        return "".join(
            f"{e}{'' if counts[e] // g == 1 else counts[e] // g}"
            for e in self.elements
            if counts.get(e)
        )

    def rejection(self, assignment, representative: bool = False) -> str | None:
        """None when the assignment passes, else the first condition it
        fails: BALANCE (condition 1) or SITE_SYMMETRY (condition 2). With
        `representative`, letters stand for their shape groups
        (base.canonical_representatives) and condition 2 takes the union
        over the letters each site could take."""
        formula = self._formula(assignment)
        sites = tuple(sorted((a.element, a.entry.letter) for a in assignment))
        key = (formula, sites, representative)
        if key not in self._cache:
            rules = self.rules_for(formula)
            self._cache[key] = (
                None if rules is None else self._decide(sites, rules, representative)
            )
        return self._cache[key]

    def accepts(self, assignment, representative: bool = False) -> bool:
        return self.rejection(assignment, representative) is None

    def _decide(self, sites, rules: BondRules, representative: bool) -> str | None:
        sg = self.sg
        mult = {e.letter: e.multiplicity for e in WYCKOFF_DB[sg]}
        orbits = [(el, mult[le]) for el, le in sites]

        def balanced(i, j, cap):
            mi, mj = orbits[i][1], orbits[j][1]
            return tuple(n for n in range(cap + 1) if (n * mi) % mj == 0)

        if not _feasible(orbits, rules, balanced):
            return BALANCE
        if not self.site_symmetry:
            return None
        orders = (
            _representative_bond_orders if representative else bond_stabiliser_orders
        )

        def symmetric(i, j, cap):
            li, lj = sites[i][1], sites[j][1]
            return _semigroup_counts(
                orders(sg, li, lj), site_symmetry(sg, li).order, cap
            )

        if not _feasible(orbits, rules, symmetric):
            return SITE_SYMMETRY
        return None
