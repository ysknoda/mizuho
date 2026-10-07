"""Periodic bond graph, topology class and geometric fingerprint (ADR-0018).

A diagnostic, never part of the objective. It answers one question about two
structures: do they differ in how atoms are connected, or only in how the same
connection pattern sits in space -- and if the latter, which bond lengths,
angles, polyhedral distortions and cell-shape components differ, by how much.

- Bond graph: atoms are vertices, an edge joins two atoms (any periodic image)
  within the measured first-shell cutoff of their pair (ADR-0004). Edges carry
  the integer cell translation, so the graph is the periodic quotient graph of
  crystal-net analysis.
- Topology class: every atom's coordination sequence (Brunner & Laves 1971;
  O'Keeffe 1995, Z. Kristallogr. 210, 905) plus the polyhedral linkage (how
  many anions two cations share). Equal classes mean the same multiset of
  atom environments to `depth` bonds; distinct classes are certainly different
  topologies. Atoms with equal (element, coordination sequence) form one
  topological type, which is what geometry is compared over -- so the
  comparison needs no atom correspondence and no common setting.
- Geometry: bond lengths and bond angles per topological type, polyhedral
  distortion per cation type (Baur 1974 distortion index for every polyhedron;
  quadratic elongation and angle variance of Robinson, Gibbs & Ribbe 1971,
  Science 172, 567, for tetrahedra and octahedra), and the Lagrangian strain
  between the two cells split into volume and shape.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from math import gcd

import numpy as np
from scipy.spatial import ConvexHull

from .lattice import _IMAGE_SHIFTS
from .objective.penalties import _expand_wildcard

DEFAULT_DEPTH = 10

# Regular polyhedra for Robinson et al. (1971): ideal cis angle and the volume
# of the regular polyhedron with centre-to-vertex distance l0, V = k * l0^3.
# Tetrahedron: edge l0*sqrt(8/3), V = edge^3 / (6 sqrt 2) = 8 / (9 sqrt 3) l0^3.
# Octahedron: two square pyramids, V = 4/3 l0^3.
_IDEAL = {
    4: {
        "angle": float(np.degrees(np.arccos(-1.0 / 3.0))),
        "n_cis": 6,
        "k": 8.0 / (9.0 * np.sqrt(3.0)),
    },
    6: {"angle": 90.0, "n_cis": 12, "k": 4.0 / 3.0},
}


@dataclass
class Crystal:
    """A periodic structure: row lattice vectors (A), species, fractional coords."""

    lattice: np.ndarray
    species: list
    frac: np.ndarray
    name: str = ""


@dataclass
class BondGraph:
    crystal: Crystal
    # adjacency[i] = [(j, t, d)]: atom j displaced by integer translation t,
    # at distance d from atom i in the home cell.
    adjacency: list
    cutoffs: dict
    cations: set = field(default_factory=set)


# ---------------------------------------------------------------------------
# Cutoffs and edges
# ---------------------------------------------------------------------------


def pair_cutoffs(config: dict, elements) -> tuple:
    """(cutoffs, ranges): {frozenset({A, B}): cutoff} for the bond graph, and
    {frozenset({A, B}): (lowest, highest)} over every cutoff the config applies
    to that pair.

    Sources, most specific first, the first that covers a pair deciding it:
    1. a coordination constraint naming both elements (`cutoff_distance`, the
       shell the integer coordination number counts), in either direction;
    2. a coordination constraint reaching the pair through a wildcard "X"
       (expanded exactly as the objective expands it);
    3. a bond constraint's explicit `cutoff`.
    Within one level the largest value is taken: each is a measured gap between
    first and second shell (ADR-0004). A wildcard cutoff is set for every
    cation of an anion at once, so it is not allowed to override a cutoff
    measured for the pair itself -- that is what made a 2.73 A Si-O contact a
    bond under diopside's O(X) cutoff 2.942 A against its Si(O) cutoff 2.425 A.
    A contact between the two ends of `ranges` is one the objective's own
    constraints disagree about (`grey_contacts`)."""
    present = set(elements)
    charges = config.get("ionic_charges") or {}
    levels = [{}, {}, {}]
    ranges = {}

    def put(level, a, b, r):
        key = frozenset((a, b))
        levels[level][key] = max(levels[level].get(key, 0.0), float(r))
        lo, hi = ranges.get(key, (float(r), float(r)))
        ranges[key] = (min(lo, float(r)), max(hi, float(r)))

    for cc in config.get("coordination_constraints", []):
        c, r = cc["center_atom"], cc.get("cutoff_distance")
        if r is None or c not in present:
            continue
        level = 1 if cc["neighbor_atom"] == "X" else 0
        for n in _expand_wildcard(cc["neighbor_atom"], present, c, charges):
            put(level, c, n, r)
    for bc in config.get("bond_constraints", []):
        a, r = bc["atom1"], bc.get("cutoff")
        if r is None or a not in present:
            continue
        for b in _expand_wildcard(bc["atom2"], present, a, charges):
            put(2, a, b, r)
    cutoffs = {}
    for level in levels:
        for key, r in level.items():
            cutoffs.setdefault(key, r)
    return cutoffs, ranges


def grey_contacts(crystal: "Crystal", cutoffs: dict, ranges: dict) -> dict:
    """{frozenset({A, B}): [distances]} of contacts beyond the graph's cutoff
    but within the highest cutoff any constraint applies to the pair, each
    contact once."""
    hi = {k: ranges[k][1] for k in cutoffs if ranges[k][1] > cutoffs[k]}
    if not hi:
        return {}
    wide = build_graph(crystal, hi)
    out = defaultdict(list)
    sp = crystal.species
    for i, nb in enumerate(wide.adjacency):
        for j, t, d in nb:
            key = frozenset((sp[i], sp[j]))
            if d > cutoffs[key] and (i, t) < (j, tuple(-x for x in t)):
                out[key].append(d)
    return dict(out)


def build_graph(crystal: Crystal, cutoffs: dict, cations=()) -> BondGraph:
    """Edges to every periodic image within the pair cutoff. The 27 images of
    mizuho.lattice are searched, as everywhere else in mizuho (ADR-0001)."""
    lat = np.asarray(crystal.lattice, float)
    frac = np.asarray(crystal.frac, float) % 1.0
    shifts = _IMAGE_SHIFTS.astype(int)
    n = len(crystal.species)
    adjacency = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            r = cutoffs.get(frozenset((crystal.species[i], crystal.species[j])))
            if r is None:
                continue
            d = frac[j] - frac[i]
            base = -np.round(d).astype(int)
            t_all = base[None, :] + shifts  # (27, 3)
            dist = np.linalg.norm((d[None, :] + t_all) @ lat, axis=1)
            for t, dd in zip(t_all, dist):
                if dd <= r and not (i == j and not t.any()):
                    adjacency[i].append((j, tuple(int(x) for x in t), float(dd)))
    return BondGraph(crystal, adjacency, cutoffs, set(cations))


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------


def coordination_sequence(graph: BondGraph, i: int, depth: int) -> tuple:
    """Number of atoms exactly k bonds from atom i, k = 1..depth, counting
    periodic images as distinct vertices (breadth-first on the infinite net)."""
    start = (i, (0, 0, 0))
    seen = {start}
    frontier = [start]
    seq = []
    for _ in range(depth):
        nxt = []
        for a, ta in frontier:
            for b, tb, _ in graph.adjacency[a]:
                v = (b, (ta[0] + tb[0], ta[1] + tb[1], ta[2] + tb[2]))
                if v not in seen:
                    seen.add(v)
                    nxt.append(v)
        seq.append(len(nxt))
        frontier = nxt
    return tuple(seq)


def linkage(graph: BondGraph) -> dict:
    """{cation index: Counter({shared-anion count: number of cation pairs})}.
    1 = corner-, 2 = edge-, >= 3 = face-sharing (Pauling's third rule)."""
    out = {}
    sp = graph.crystal.species
    for i, el in enumerate(sp):
        if el not in graph.cations:
            continue
        shared = Counter()
        for a, ta, _ in graph.adjacency[i]:
            if sp[a] in graph.cations:
                continue
            for c, tc, _ in graph.adjacency[a]:
                if sp[c] not in graph.cations:
                    continue
                tt = (ta[0] + tc[0], ta[1] + tc[1], ta[2] + tc[2])
                if c == i and tt == (0, 0, 0):
                    continue
                shared[(c, tt)] += 1
        out[i] = Counter(shared.values())
    return out


@dataclass
class Topology:
    types: list  # per atom: topological type key (element, CS)
    labels: dict  # type key -> readable label
    signature: tuple  # class key, independent of cell size
    linkage: dict  # type key -> {shared count: pairs per atom}


def _env_label(graph: BondGraph, i: int) -> str:
    sp = graph.crystal.species
    c = Counter(sp[j] for j, _, _ in graph.adjacency[i])
    return f"{sp[i]}(" + "".join(f"{e}{c[e]}" for e in sorted(c)) + ")"


def topology(graph: BondGraph, depth: int = DEFAULT_DEPTH) -> Topology:
    sp = graph.crystal.species
    types = [(sp[i], coordination_sequence(graph, i, depth)) for i in range(len(sp))]
    count = Counter(types)
    g = 0
    for v in count.values():
        g = gcd(g, v)
    link = linkage(graph)
    per_type = defaultdict(Counter)
    for i, c in link.items():
        per_type[types[i]].update(c)
    link_norm = {
        k: {s: v / count[k] for s, v in sorted(c.items())} for k, c in per_type.items()
    }
    signature = tuple(
        sorted(
            (k, count[k] // g, tuple(sorted(link_norm.get(k, {}).items())))
            for k in count
        )
    )
    labels = {}
    by_label = defaultdict(list)
    for k in sorted(count):
        by_label[_env_label(graph, types.index(k))].append(k)
    for lab, keys in by_label.items():
        for n, k in enumerate(keys):
            labels[k] = lab if len(keys) == 1 else f"{lab}#{n + 1}"
    return Topology(types, labels, signature, link_norm)


def describe_difference(ref: Topology, other: Topology) -> list:
    """Readable lines saying where two topology classes part: for each
    reference type, the candidate type with the same first-shell label and the
    first shell at which their coordination sequences differ, or that the
    label is missing; then any linkage change."""
    def by_label(t):
        out = {}
        for k in set(t.types):
            out[t.labels[k]] = (k, t.types.count(k), t.linkage.get(k, {}))
        return out

    a, b = by_label(ref), by_label(other)
    lines = []
    for lab in sorted(set(a) | set(b)):
        if lab not in b:
            lines.append(f"{lab}: only in reference")
            continue
        if lab not in a:
            lines.append(f"{lab}: only in candidate")
            continue
        (ka, na, la), (kb, nb, lb) = a[lab], b[lab]
        diff = [n for n, (x, y) in enumerate(zip(ka[1], kb[1])) if x != y]
        if diff:
            n = diff[0]
            lines.append(f"{lab}: coordination sequence differs from shell {n + 1} "
                         f"({ka[1][n]} vs {kb[1][n]})")
        if la != lb:
            lines.append(f"{lab}: polyhedron sharing {dict(la)} vs {dict(lb)}")
    return lines


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def _neighbour_vectors(graph: BondGraph, i: int):
    lat = np.asarray(graph.crystal.lattice, float)
    frac = np.asarray(graph.crystal.frac, float) % 1.0
    out = []
    for j, t, _ in graph.adjacency[i]:
        out.append((j, (frac[j] + np.array(t) - frac[i]) @ lat))
    return out


def polyhedron(vectors: np.ndarray) -> dict:
    """Distortion of one coordination polyhedron from centre-to-ligand vectors.

    `delta`: Baur (1974) distortion index, mean |l_i - <l>| / <l>.
    `lambda`, `sigma2` (CN 4 and 6 only): Robinson et al. (1971) quadratic
    elongation <(l_i / l_0)^2>, l_0 the centre-to-vertex distance of the regular
    polyhedron of equal volume, and the variance of the cis angles about the
    ideal angle with n_cis - 1 degrees of freedom."""
    v = np.asarray(vectors, float)
    lengths = np.linalg.norm(v, axis=1)
    out = {
        "cn": len(v),
        "delta": float(np.mean(np.abs(lengths - lengths.mean())) / lengths.mean()),
    }
    ideal = _IDEAL.get(len(v))
    if ideal is None:
        return out
    try:
        vol = ConvexHull(v).volume
    except Exception:  # a flat or degenerate polyhedron has no volume
        return out
    l0 = (vol / ideal["k"]) ** (1.0 / 3.0)
    unit = v / lengths[:, None]
    angles = sorted(
        float(np.degrees(np.arccos(np.clip(unit[a] @ unit[b], -1.0, 1.0))))
        for a, b in combinations(range(len(v)), 2)
    )
    cis = np.array(angles[: ideal["n_cis"]])
    out["lambda"] = float(np.mean((lengths / l0) ** 2))
    out["sigma2"] = float(np.sum((cis - ideal["angle"]) ** 2) / (ideal["n_cis"] - 1))
    return out


def _stats(values) -> tuple:
    a = np.asarray(values, float)
    return (float(a.min()), float(a.mean()), float(a.max()), len(a))


def geometry(graph: BondGraph, topo: Topology) -> dict:
    """Per-type geometry, keyed by readable labels so two structures of one
    class line up without an atom correspondence.

    bonds[(A, B)]       distances from cation type A to neighbour type B
    angles[(B, A, C)]   angles B-A-C at every atom of type A (all neighbour pairs)
    polyhedra[A]        delta / lambda / sigma2 over the cations of type A"""
    lab = [topo.labels[k] for k in topo.types]
    bonds, angles, poly = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(lambda: defaultdict(list)),
    )
    for i in range(len(lab)):
        nb = _neighbour_vectors(graph, i)
        if not nb:
            continue
        sp = graph.crystal.species
        for (j, vj), (_, _, d) in zip(nb, graph.adjacency[i]):
            # Each bond once, seen from its cation (from the lower label
            # when neither or both ends are cations).
            if (sp[i] in graph.cations) != (sp[j] in graph.cations):
                keep = sp[i] in graph.cations
            else:
                keep = (lab[i], i) <= (lab[j], j)
            if keep:
                bonds[(lab[i], lab[j])].append(d)
        for (j, vj), (k, vk) in combinations(nb, 2):
            cosang = vj @ vk / (np.linalg.norm(vj) * np.linalg.norm(vk))
            a, c = sorted((lab[j], lab[k]))
            angles[(a, lab[i], c)].append(
                float(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
            )
        if graph.crystal.species[i] in graph.cations:
            for key, val in polyhedron(np.array([v for _, v in nb])).items():
                poly[lab[i]][key].append(val)
    return {
        "bonds": {k: _stats(v) for k, v in bonds.items()},
        "angles": {k: _stats(v) for k, v in angles.items()},
        "polyhedra": {
            k: {q: _stats(v) for q, v in d.items() if q != "cn"}
            for k, d in poly.items()
        },
    }


def cell_strain(lat_ref, lat) -> dict:
    """Lagrangian strain E = (F^T F - I) / 2 taking the reference cell to the
    other: with the lattice vectors as columns, A1 = F A0. Valid only when both
    cells use the same axes (same setting).
    `volume` is det F - 1; `shape` the Frobenius norm of the deviatoric part of
    E, zero for a pure scaling."""
    A0 = np.asarray(lat_ref, float).T
    A1 = np.asarray(lat, float).T
    F = A1 @ np.linalg.inv(A0)
    E = 0.5 * (F.T @ F - np.eye(3))
    dev = E - np.trace(E) / 3.0 * np.eye(3)
    w = np.linalg.eigvalsh(E)
    return {
        "volume": float(np.linalg.det(F) - 1.0),
        "shape": float(np.linalg.norm(dev)),
        "principal": [float(x) for x in w],
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _fmt_key(k) -> str:
    return k if isinstance(k, str) else "-".join(k)


def compare_geometry(ref: dict, other: dict) -> dict:
    """Rows (quantity, key, ref (min, mean, max), other (min, mean, max), score),
    one table per category, sorted by score: the largest shift of min, mean or
    max -- relative (%) for bond lengths, absolute for angles (deg) and for the
    dimensionless distortion measures."""
    tables = {}
    for cat in ("bonds", "angles"):
        rows = []
        for k in sorted(set(ref[cat]) & set(other[cat])):
            r, o = ref[cat][k], other[cat][k]
            diff = [o[n] - r[n] for n in range(3)]
            score = (
                max(abs(x) / r[1] * 100 for x in diff)
                if cat == "bonds"
                else max(abs(x) for x in diff)
            )
            rows.append((cat, _fmt_key(k), r, o, score))
        tables[cat] = sorted(rows, key=lambda x: -x[4])
    rows = []
    for k in sorted(set(ref["polyhedra"]) & set(other["polyhedra"])):
        for q in sorted(set(ref["polyhedra"][k]) & set(other["polyhedra"][k])):
            r, o = ref["polyhedra"][k][q], other["polyhedra"][k][q]
            rows.append((q, k, r, o, max(abs(o[n] - r[n]) for n in range(3))))
    tables["polyhedra"] = rows
    return tables
