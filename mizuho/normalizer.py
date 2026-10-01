"""The Euclidean normalizer of a space group, and equivalent Wyckoff
assignments (docs/adr/0011-equivalent-assignments-via-the-euclidean-normalizer.md).

The normalizer is derived from mizuho's own symmetry-operation tables, by
testing N G N^-1 = G directly: N = (P, s) runs over the point operations
of the crystal family's holohedry (the metric-preserving ones for the
family's GENERAL metric, see the ADR) and over origin shifts s on a grid.
No external table is used, the same rule the Wyckoff database follows.

- Discrete part: maps that permute Wyckoff letters. Two assignments
  related by one describe the same crystal and are merged before Phase 1
  ranking.
- Continuous part: the floating origin of a polar group -- the
  directions along which translating the whole structure changes nothing.
  They are the subspace every rotation of the group fixes.
"""

from collections.abc import Sequence
from functools import cache
from itertools import product

import numpy as np

from .data import WYCKOFF_DB, build_point, get_ops
from .lattice import get_lattice_system
from .symmetry import solve_free_params

# A space group whose rotations are exactly a family's holohedry, in
# mizuho's axes for that family: its rotation set IS the holohedry.
_HOLOHEDRY_SG = {
    "triclinic": 2,
    "monoclinic": 10,
    "orthorhombic": 47,
    "tetragonal": 123,
    "trigonal": 191,
    "hexagonal": 191,
    "cubic": 221,
}

_KEY_DIGITS = 6


def _op_key(R: np.ndarray, t: np.ndarray) -> tuple:
    tt = np.round(np.asarray(t) % 1.0, _KEY_DIGITS) % 1.0
    return tuple(np.rint(R).astype(int).ravel()) + tuple(tt)


@cache
def _op_set(sg: int):
    rot, tr = get_ops(sg)
    return frozenset(_op_key(R, t) for R, t in zip(rot, tr))


@cache
def holohedry(sg: int) -> tuple[np.ndarray, ...]:
    rot, _ = get_ops(_HOLOHEDRY_SG[get_lattice_system(sg)])
    uniq = {tuple(np.rint(R).astype(int).ravel()): R for R in rot}
    return tuple(np.array(R, dtype=float) for R in uniq.values())


@cache
def polar_directions(sg: int) -> np.ndarray:
    """Basis (k, 3) of the floating-origin subspace: vectors every rotation
    of the group leaves fixed. Empty for a non-polar group."""
    rot, _ = get_ops(sg)
    M = np.vstack([R - np.eye(3) for R in rot])
    _, s, vt = np.linalg.svd(M)
    rank = int(np.sum(s > 1e-8))
    return vt[rank:]


def _shift_grid(sg: int) -> list[np.ndarray]:
    """Candidate origin shifts. Multiples of 1/8 reach every normalizer
    shift of the non-hexagonal families; hexagonal axes also need thirds
    and sixths, so 1/24 there. Components along a polar direction are
    continuous and held at 0 (they are reported separately)."""
    den = 24 if get_lattice_system(sg) in ("hexagonal", "trigonal") else 8
    polar = polar_directions(sg)
    free_axes = [i for i in range(3) if not any(abs(v[i]) > 1e-8 for v in polar)]
    if any(np.count_nonzero(np.abs(v) > 1e-8) > 1 for v in polar):
        raise NotImplementedError(f"SG {sg}: polar direction not along an axis")
    grid = []
    for combo in product(range(den), repeat=len(free_axes)):
        s = np.zeros(3)
        for ax, k in zip(free_axes, combo):
            s[ax] = k / den
        grid.append(s)
    return grid


@cache
def euclidean_normalizer(sg: int) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Every (P, s), s modulo the lattice and the polar directions, with
    N G N^-1 = G. Includes G's own operations (they normalise G trivially)."""
    rot, tr = get_ops(sg)
    ops = list(zip(rot, tr))
    target = _op_set(sg)
    # Test the non-identity ops first: identity always passes.
    ops.sort(
        key=lambda op: np.allclose(op[0], np.eye(3)) and np.allclose(op[1] % 1.0, 0)
    )
    found = []
    for P in holohedry(sg):
        Pinv = np.linalg.inv(P)
        conj = [(P @ R @ Pinv, P @ t) for R, t in ops]
        for s in _shift_grid(sg):
            ok = True
            for Rp, Pt in conj:
                # N (R, t) N^-1 = (P R P^-1, P t + s - P R P^-1 s)
                if _op_key(Rp, Pt + s - Rp @ s) not in target:
                    ok = False
                    break
            if ok:
                found.append((P, s))
    return tuple(found)


@cache
def coset_representatives(sg: int) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """One normalizer element per coset N G. Two elements N1, N2 share a
    coset iff N1^-1 N2 lies in G, and elements of one coset induce the
    same letter permutation, so only these need mapping."""
    target = _op_set(sg)
    reps: list[tuple[np.ndarray, np.ndarray]] = []
    for P, s in euclidean_normalizer(sg):
        for Q, u in reps:
            Qinv = np.linalg.inv(Q)
            # (Q, u)^-1 (P, s) = (Q^-1 P, Q^-1 (s - u)); modulo the polar
            # directions, which euclidean_normalizer already factored out.
            d = Qinv @ (s - u)
            for v in polar_directions(sg):
                d = d - (d @ v) * v
            if _op_key(Qinv @ P, d) in target:
                break
        else:
            reps.append((P, s))
    return tuple(reps)


def _generic_values(entry, rng) -> dict[str, float]:
    return {p: float(rng.uniform(0.07, 0.43)) for p in entry.free_params}


@cache
def letter_permutations(sg: int) -> tuple[dict[str, str], ...]:
    """The distinct permutations of Wyckoff letters the normalizer induces.

    A letter's image is found by mapping a generic point of it (irrational-
    looking free parameters, so it sits on no special position of the
    image) and asking which letter's template the image lies on."""
    entries = WYCKOFF_DB[sg]
    rng = np.random.default_rng(12345)
    points = {
        e.letter: build_point(e.coord_template, _generic_values(e, rng))
        for e in entries
    }
    perms = set()
    for P, s in coset_representatives(sg):
        perm = {}
        for e in entries:
            q = P @ points[e.letter] + s
            image = [
                f.letter
                for f in entries
                if f.multiplicity == e.multiplicity
                and len(f.free_params) == len(e.free_params)
                and solve_free_params(sg, f, q) is not None
            ]
            if len(image) != 1:
                raise RuntimeError(
                    f"SG {sg}: {e.letter} maps onto {image} under P={P.tolist()}, s={s}"
                )
            perm[e.letter] = image[0]
        perms.add(tuple(sorted(perm.items())))
    return tuple(dict(p) for p in sorted(perms))


def canonical_key(
    sg: int, pairs: Sequence[tuple[str, str]]
) -> tuple[tuple[str, str], ...]:
    """Canonical representative of an assignment's equivalence class: the
    smallest sorted (element, letter) tuple over every letter permutation
    the normalizer induces."""
    return min(
        tuple(sorted((el, perm[le]) for el, le in pairs))
        for perm in letter_permutations(sg)
    )


def assignment_key(sg: int, assignment) -> tuple[tuple[str, str], ...]:
    """canonical_key for a list of SiteAssignment."""
    return canonical_key(sg, [(a.element, a.entry.letter) for a in assignment])


def merge_equivalent(sg: int, assignments: list) -> list:
    """Drop every assignment equivalent to one earlier in the list, keeping
    enumeration order among the survivors."""
    seen, out = set(), []
    for asg in assignments:
        k = assignment_key(sg, asg)
        if k not in seen:
            seen.add(k)
            out.append(asg)
    return out
