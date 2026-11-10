"""Space groups of every cation ordering on zinc-blende and wurtzite parents.

A composition-free search of an all-tetrahedral compound such as Cu-Zn-Sn-S
(every cation and every anion at CN 4) blows up in Phase 1 in groups that no
tetrahedral ordering can have: at 16 atoms and 8 sites Cmmm alone gives 9.8
million assignments and Pmmm did not finish in 6 h. This script lists the
groups the search should visit instead (design review 2026-10-29).

Parents: zinc blende (F-43m, 2 atoms per primitive cell) and ideal wurtzite
(P63mc, u = 3/8, c/a = sqrt(8/3), 4 atoms). The anion sublattice is kept;
every supercell with at most --max-atoms atoms (all Hermite normal forms of
each index) has its cation sites decorated with every charge-neutral
assignment of the given cations, and spglib names the group of each
decoration. Both parent groups are non-centrosymmetric, so is every
subgroup: no centrosymmetric group can appear.

    python scripts/derive_ordering_spacegroups.py \\
        --cations Cu:1 Zn:2 Sn:4 --anion S:-2 --max-atoms 16 --out runs/czts/ordering

Writes spacegroups.csv (group, parent, decorations, smallest cell,
compositions), spacegroups.txt (the group numbers, one line, for
spacegroup_search.spacegroups) and one ideal POSCAR per (parent,
composition, group): the first decoration found, unit cation-anion bond.
"""

import argparse
import csv
import itertools
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import spglib
import spglib.error

spglib.error.OLD_ERROR_HANDLING = False

BOND = (
    1.0  # cation-anion bond length of the ideal parents (A); groups do not depend on it
)


def zincblende():
    """Primitive zinc-blende cell: lattice rows, fractional sites, is_cation."""
    a = 4.0 * BOND / math.sqrt(3.0)
    lat = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float)
    frac = np.array([[0, 0, 0], [0.25, 0.25, 0.25]])
    return lat, frac, np.array([True, False])


def wurtzite():
    """Ideal wurtzite: u = 3/8, c/a = sqrt(8/3), so the cation-anion bond along c is u c."""
    c = BOND / 0.375
    a = c / math.sqrt(8.0 / 3.0)
    lat = np.array([[a, 0, 0], [-a / 2, a * math.sqrt(3) / 2, 0], [0, 0, c]])
    frac = np.array(
        [
            [1 / 3, 2 / 3, 0],
            [2 / 3, 1 / 3, 0.5],
            [1 / 3, 2 / 3, 0.375],
            [2 / 3, 1 / 3, 0.875],
        ]
    )
    return lat, frac, np.array([True, True, False, False])


PARENTS = {"zb": zincblende, "wz": wurtzite}


def hnf_matrices(n):
    """Every lower-triangular Hermite normal form of determinant n (column convention)."""
    for a in range(1, n + 1):
        if n % a:
            continue
        for c in range(1, n // a + 1):
            if (n // a) % c:
                continue
            f = n // (a * c)
            for b in range(c):
                for d in range(f):
                    for e in range(f):
                        yield np.array([[a, 0, 0], [b, c, 0], [d, e, f]])


def supercell(lat, frac, is_cat, h):
    """Supercell of the Hermite normal form h, sites as fractional coordinates of it.

    h follows the column convention of the HNF enumeration (superlattice vectors
    are the columns of A h), so with lattice vectors as rows the supercell is
    h.T @ lat. Using h itself would repeat sublattices and miss others.
    """
    h = np.asarray(h).T
    n = round(abs(np.linalg.det(h)))
    hinv = np.linalg.inv(h)
    # Lattice points of the parent inside the supercell: integer t with t @ hinv in [0, 1).
    span = int(np.abs(h).sum(axis=0).max())
    pts = set()
    for t in itertools.product(range(-span, span + 1), repeat=3):
        s = np.array(t) @ hinv
        s = s - np.floor(s + 1e-9)
        pts.add(tuple(np.round(s, 9) % 1.0))
    assert len(pts) == n, (len(pts), n)
    new_frac, new_cat = [], []
    for p in sorted(pts):
        for x, cat in zip(frac, is_cat):
            q = (x @ hinv + np.array(p)) % 1.0
            new_frac.append(q)
            new_cat.append(cat)
    return h @ lat, np.array(new_frac), np.array(new_cat)


def count_vectors(n_cat, n_anion, cations, anion_q, require_all):
    """Cation counts (one per cation) summing to n_cat and neutralising the anions."""
    names = list(cations)
    lo = 1 if require_all else 0
    for counts in itertools.product(range(lo, n_cat + 1), repeat=len(names)):
        if sum(counts) != n_cat:
            continue
        if sum(k * cations[e] for k, e in zip(counts, names)) + n_anion * anion_q == 0:
            yield dict(zip(names, counts))


def decorations(n_sites, counts):
    """Every assignment of the counted species to n_sites ordered sites."""
    species = [e for e in counts if counts[e]]

    def rec(free, i):
        if i == len(species) - 1:
            yield {s: species[i] for s in free}
            return
        for chosen in itertools.combinations(free, counts[species[i]]):
            rest = [s for s in free if s not in chosen]
            for tail in rec(rest, i + 1):
                tail.update({s: species[i] for s in chosen})
                yield tail

    yield from rec(list(range(n_sites)), 0)


def formula(counts, anion, n_anion):
    g = math.gcd(*[k for k in counts.values() if k], n_anion)
    parts = [(e, k // g) for e, k in counts.items() if k] + [(anion, n_anion // g)]
    return "".join(e + (str(k) if k > 1 else "") for e, k in parts)


def enumerate_groups(cations, anion, anion_q, max_atoms, parents, symprec, require_all):
    """{(parent, sg): {"n": decorations, "atoms": smallest cell, "comps": set}} and
    {(parent, composition, sg): (lattice, frac, symbols)} representatives."""
    found = defaultdict(lambda: {"n": 0, "atoms": None, "comps": set()})
    reps = {}
    for pname in parents:
        lat, frac, is_cat = PARENTS[pname]()
        per = len(frac)
        for index in range(1, max_atoms // per + 1):
            for h in hnf_matrices(index):
                slat, sfrac, scat = supercell(lat, frac, is_cat, h)
                cat_idx = np.flatnonzero(scat)
                n_an = int((~scat).sum())
                for counts in count_vectors(
                    len(cat_idx), n_an, cations, anion_q, require_all
                ):
                    comp = formula(counts, anion, n_an)
                    for dec in decorations(len(cat_idx), counts):
                        symbols = [anion] * len(sfrac)
                        for k, e in dec.items():
                            symbols[cat_idx[k]] = e
                        z = [ZNUM[s] for s in symbols]
                        sg = spglib.get_symmetry_dataset(
                            (slat, sfrac, z), symprec=symprec
                        ).number
                        rec = found[(pname, sg)]
                        rec["n"] += 1
                        rec["comps"].add(comp)
                        natoms = len(sfrac)
                        rec["atoms"] = (
                            natoms
                            if rec["atoms"] is None
                            else min(rec["atoms"], natoms)
                        )
                        reps.setdefault(
                            (pname, comp, sg), (slat, sfrac.copy(), symbols)
                        )
    return found, reps


ZNUM = {
    "H": 1,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
}


def write_poscar(path, lat, frac, symbols):
    order = sorted(range(len(symbols)), key=lambda i: (ZNUM[symbols[i]], i))
    species = list(dict.fromkeys(symbols[i] for i in order))
    lines = [path.stem, "1.0"] + [" ".join(f"{x:.10f}" for x in row) for row in lat]
    lines += [
        " ".join(species),
        " ".join(str(symbols.count(s)) for s in species),
        "Direct",
    ]
    lines += [" ".join(f"{x:.10f}" for x in frac[i]) for i in order]
    path.write_text("\n".join(lines) + "\n")


def parse_species(items):
    out = {}
    for it in items:
        e, q = it.split(":")
        out[e] = int(q)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--cations",
        nargs="+",
        required=True,
        help="element:charge, e.g. Cu:1 Zn:2 Sn:4",
    )
    ap.add_argument("--anion", required=True, help="element:charge, e.g. S:-2")
    ap.add_argument(
        "--max-atoms",
        type=int,
        default=16,
        help="atoms per supercell (primitive bound)",
    )
    ap.add_argument(
        "--parents", nargs="+", default=["zb", "wz"], choices=sorted(PARENTS)
    )
    ap.add_argument(
        "--symprec", type=float, default=1e-3, help="spglib tolerance (ideal sites)"
    )
    ap.add_argument(
        "--allow-missing",
        action="store_true",
        help="decorations need not contain every cation",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    cations = parse_species(args.cations)
    ((anion, anion_q),) = parse_species([args.anion]).items()
    found, reps = enumerate_groups(
        cations,
        anion,
        anion_q,
        args.max_atoms,
        args.parents,
        args.symprec,
        not args.allow_missing,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "spacegroups.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "spacegroup",
                "parent",
                "decorations",
                "smallest_cell_atoms",
                "compositions",
            ]
        )
        for (p, sg), r in sorted(found.items(), key=lambda kv: (-kv[0][1], kv[0][0])):
            w.writerow([sg, p, r["n"], r["atoms"], " ".join(sorted(r["comps"]))])
    groups = sorted({sg for _, sg in found}, reverse=True)
    (args.out / "spacegroups.txt").write_text(" ".join(map(str, groups)) + "\n")
    pos = args.out / "poscar"
    pos.mkdir(exist_ok=True)
    for (p, comp, sg), (lat, frac, sym) in reps.items():
        write_poscar(pos / f"{p}_{comp}_sg{sg}.vasp", lat, frac, sym)
    print(f"{len(groups)} groups: {groups}")
    print(
        f"{sum(r['n'] for r in found.values())} decorations, {len(reps)} representatives -> {args.out}"
    )


if __name__ == "__main__":
    main()
