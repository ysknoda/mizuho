"""Crystal-system classification and lattice/distance geometry helpers.

CRYSTAL_SYSTEM_OF_SG is the single source of truth for space-group ->
crystal-system classification. The original mizuho.py had two
byte-identical copies of this table (LATTICE_SYSTEM and _CS_OF_SG,
used inconsistently by different call sites) -- consolidated here into
one name during the package split.
"""

import math

import numpy as np

CRYSTAL_SYSTEM_OF_SG = {
    **{i: "triclinic"    for i in range(1,   3)},
    **{i: "monoclinic"   for i in range(3,  16)},
    **{i: "orthorhombic" for i in range(16,  75)},
    **{i: "tetragonal"   for i in range(75, 143)},
    **{i: "trigonal"     for i in range(143, 168)},
    **{i: "hexagonal"    for i in range(168, 195)},
    **{i: "cubic"        for i in range(195, 231)},
}


def get_lattice_system(sg: int) -> str:
    return CRYSTAL_SYSTEM_OF_SG.get(sg, "triclinic")


# ==============================================================================
# Lattice geometry helpers
# ==============================================================================

def build_lattice_matrix(lp: dict) -> np.ndarray:
    a, b, c = lp["a"], lp["b"], lp["c"]
    alpha = math.radians(lp.get("alpha", 90.0))
    beta  = math.radians(lp.get("beta",  90.0))
    gamma = math.radians(lp.get("gamma", 90.0))
    cos_a, cos_b, cos_g = math.cos(alpha), math.cos(beta), math.cos(gamma)
    sin_g = math.sin(gamma)
    cx = c * cos_b
    cy = c * (cos_a - cos_b * cos_g) / sin_g
    cz = math.sqrt(max(c**2 - cx**2 - cy**2, 0.0))
    return np.array([[a, 0.0, 0.0],
                     [b*cos_g, b*sin_g, 0.0],
                     [cx, cy, cz]])


_IMAGE_SHIFTS = np.array([[i, j, k]
                          for i in (-1, 0, 1)
                          for j in (-1, 0, 1)
                          for k in (-1, 0, 1)], dtype=float)  # (27, 3)

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


_PAIRWISE_DOC = """
    Minimum-image pairwise distances (Angstrom).
    fa: (Na, 3) fractional coords of group A
    fb: (Nb, 3) fractional coords of group B
    Returns: (Na, Nb) distance matrix.

    Explicitly checks all 27 neighbouring lattice images (+/-1 in each
    of the 3 directions) and keeps the minimum Cartesian distance,
    rather than rounding each fractional component to its nearest
    integer independently. The two agree for orthogonal cells, but
    component-wise rounding is not guaranteed to find the true nearest
    periodic image for skewed (monoclinic/triclinic, or tetragonal/
    hexagonal near their skew limits) cells -- the fractional lattice
    point nearest in fractional-coordinate space is not always the one
    nearest in real (Cartesian) space once the lattice vectors are not
    mutually perpendicular. At the tool's assumed scale (tens of atoms
    per cell) the extra factor of 27 is negligible.
    """


def _pairwise_distances_pbc_numpy(fa: np.ndarray, fb: np.ndarray,
                                   lat: np.ndarray) -> np.ndarray:
    fa = np.asarray(fa, dtype=float)
    fb = np.asarray(fb, dtype=float)
    # diff shape: (Na, Nb, 3)
    diff = fa[:, None, :] - fb[None, :, :]
    diff -= np.round(diff)  # centre on the fractional minimum image first
    # Add each of the 27 candidate image shifts -> (Na, Nb, 27, 3)
    shifted = diff[:, :, None, :] + _IMAGE_SHIFTS[None, None, :, :]
    cart = shifted @ lat  # (Na, Nb, 27, 3)
    dist2 = np.einsum('ijkl,ijkl->ijk', cart, cart)  # (Na, Nb, 27)
    return np.sqrt(dist2.min(axis=2))


_pairwise_distances_pbc_numpy.__doc__ = _PAIRWISE_DOC + """
    Pure-NumPy broadcast implementation: allocates a transient
    (Na, Nb, 27, 3) array per call. Always available, used directly
    when numba is not installed and kept importable under this name
    even when it is, so the two implementations can be cross-checked
    against each other in the same process (see test_lattice.py).
    """


if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _pairwise_distances_pbc_numba_kernel(fa, fb, lat, shifts):
        Na = fa.shape[0]
        Nb = fb.shape[0]
        out = np.empty((Na, Nb), dtype=np.float64)
        for i in range(Na):
            for j in range(Nb):
                dx = fa[i, 0] - fb[j, 0]
                dy = fa[i, 1] - fb[j, 1]
                dz = fa[i, 2] - fb[j, 2]
                dx -= np.round(dx)
                dy -= np.round(dy)
                dz -= np.round(dz)
                best2 = np.inf
                for s in range(shifts.shape[0]):
                    sx = dx + shifts[s, 0]
                    sy = dy + shifts[s, 1]
                    sz = dz + shifts[s, 2]
                    cx = sx * lat[0, 0] + sy * lat[1, 0] + sz * lat[2, 0]
                    cy = sx * lat[0, 1] + sy * lat[1, 1] + sz * lat[2, 1]
                    cz = sx * lat[0, 2] + sy * lat[1, 2] + sz * lat[2, 2]
                    d2 = cx * cx + cy * cy + cz * cz
                    if d2 < best2:
                        best2 = d2
                out[i, j] = math.sqrt(best2)
        return out

    def _pairwise_distances_pbc_numba(fa: np.ndarray, fb: np.ndarray,
                                       lat: np.ndarray) -> np.ndarray:
        fa = np.ascontiguousarray(fa, dtype=np.float64)
        fb = np.ascontiguousarray(fb, dtype=np.float64)
        lat = np.ascontiguousarray(lat, dtype=np.float64)
        return _pairwise_distances_pbc_numba_kernel(fa, fb, lat, _IMAGE_SHIFTS)

    _pairwise_distances_pbc_numba.__doc__ = _PAIRWISE_DOC + """
    Numba-JIT implementation: an explicit per-(i, j) loop over the 27
    image shifts that keeps only a running minimum, so it never
    materialises the (Na, Nb, 27, 3) intermediate the NumPy broadcast
    version does -- measured 14-21x faster than
    _pairwise_distances_pbc_numpy depending on N (see
    scripts/benchmark_lattice.py), and verified numerically identical
    to it (max diff ~1e-15, floating-point noise from a different
    summation order) across all 7 crystal systems and N=1..56 (see
    test_lattice.py). fastmath is deliberately NOT enabled: it would
    relax IEEE 754 semantics enough to risk the minimum-image
    selection disagreeing with the NumPy version on near-tie cases,
    and this project relies on bit-for-bit-reproducible optimisation
    results (see phase G-8's plateau_patience measurements) -- not
    worth trading that away for a speedup already measured at 14-21x
    without it.
    """


def minimum_image_vectors(fa: np.ndarray, fb: np.ndarray,
                           lat: np.ndarray) -> tuple:
    """
    Minimum-image pairwise DISPLACEMENT VECTORS and their lengths.

    fa: (Na, 3) fractional coords of group A
    fb: (Nb, 3) fractional coords of group B
    Returns (vecs, dists):
      vecs[i, j]  -- (3,) Cartesian vector from fb[j] to fa[i], taking
                     whichever periodic image of fb[j] is nearest.
      dists[i, j] -- its length, identical to
                     pairwise_distances_pbc(fa, fb, lat)[i, j].

    Angle terms need the VECTOR, not just the scalar distance, so they
    cannot use pairwise_distances_pbc directly. Before this helper
    existed they each open-coded a *component-wise* `diff -= round(diff)`
    minimum image instead -- which picks the fractional-space nearest
    image, not the real-space nearest one. Those two agree only for
    orthogonal cells (there the squared distance is separable per axis,
    so minimising each |Delta f_i| independently is exactly right); for a
    skewed monoclinic/triclinic cell they can disagree badly. Measured
    inside tests/fixtures/wollastonite_casio3.yaml's OWN allowed
    lattice_bounds: a pair 2.10 A apart by the true minimum image is
    reported as 3.65 A by component-wise rounding, far enough to push a
    genuine bond outside any sane angle cutoff and silently delete a
    real bond angle from the penalty sum -- while the CN/bond terms,
    using pairwise_distances_pbc, still counted that same atom as a
    neighbour. Sharing this helper is what keeps every term looking at
    the same periodic image of the same atom.

    Deliberately mirrors _pairwise_distances_pbc_numpy's own 27-image
    construction expression-for-expression, so the returned `dists` and
    pairwise_distances_pbc cannot drift apart under future edits (a
    regression test asserts they agree across all crystal systems).
    """
    fa = np.asarray(fa, dtype=float)
    fb = np.asarray(fb, dtype=float)
    diff = fa[:, None, :] - fb[None, :, :]
    diff -= np.round(diff)  # centre on the fractional minimum image first
    shifted = diff[:, :, None, :] + _IMAGE_SHIFTS[None, None, :, :]
    cart = shifted @ lat                              # (Na, Nb, 27, 3)
    dist2 = np.einsum('ijkl,ijkl->ijk', cart, cart)   # (Na, Nb, 27)
    best = np.argmin(dist2, axis=2)                   # (Na, Nb)
    vecs = np.take_along_axis(cart, best[:, :, None, None], axis=2)[:, :, 0, :]
    dists = np.sqrt(np.take_along_axis(dist2, best[:, :, None], axis=2)[:, :, 0])
    return vecs, dists


ZERO_SHIFT_INDEX = 13   # index of (0, 0, 0) within _IMAGE_SHIFTS


def all_image_distances(fa: np.ndarray, fb: np.ndarray,
                         lat: np.ndarray) -> np.ndarray:
    """
    Distances to EVERY periodic image, shape (Na, Nb, 27).

    fa: (Na, 3) fractional coords of group A
    fb: (Nb, 3) fractional coords of group B
    Returns out[i, j, k] -- the distance from fa[i] to the image of fb[j]
    displaced by _IMAGE_SHIFTS[k]. `pairwise_distances_pbc` is exactly
    `out.min(axis=2)`.

    A coordination number is a count over neighbouring ATOMS, and an
    atom's periodic images are separate neighbours. Reducing to the
    nearest image first -- which is what pairwise_distances_pbc does, and
    what every penalty term used to consume -- collapses all images of one
    site into a single countable neighbour, so a first coordination shell
    containing several images of the same site is undercounted. Measured
    against structures whose coordination number is a textbook fact, that
    reduction reported NaCl Na-Cl as 3 instead of 6, CsCl Cs-Cl as 1
    instead of 8, and fcc Cu as 3 instead of 12. See
    docs/adr/0001-count-every-periodic-image.md.

    Callers that compare an atom against its own group must mask the
    identity image (i == j at k == ZERO_SHIFT_INDEX), and ONLY that one:
    the other 26 images of the same atom are genuine neighbours -- fcc
    Cu's twelve nearest neighbours are images of the atom itself.
    """
    fa = np.asarray(fa, dtype=float)
    fb = np.asarray(fb, dtype=float)
    diff = fa[:, None, :] - fb[None, :, :]
    diff -= np.round(diff)
    shifted = diff[:, :, None, :] + _IMAGE_SHIFTS[None, None, :, :]
    cart = shifted @ lat
    return np.sqrt(np.einsum('ijkl,ijkl->ijk', cart, cart))


def all_image_vectors(fa: np.ndarray, fb: np.ndarray,
                       lat: np.ndarray) -> tuple:
    """
    Displacement vectors to EVERY periodic image, and their lengths.

    Returns (vecs, dists) of shapes (Na, Nb, 27, 3) and (Na, Nb, 27):
    vecs[i, j, k] is the Cartesian vector from the image of fb[j]
    displaced by _IMAGE_SHIFTS[k] to fa[i]. `minimum_image_vectors` is
    this reduced along axis 2 by `argmin(dists)`.

    The vector-returning counterpart of all_image_distances, for terms
    that need directions rather than only separations -- angles, and the
    hybridisation geometry. A bond angle at a centre atom is formed with
    each neighbouring ATOM, and images of one neighbour site are separate
    atoms, so enumerating only the nearest image of each site silently
    drops every angle subtended by the others.
    """
    fa = np.asarray(fa, dtype=float)
    fb = np.asarray(fb, dtype=float)
    diff = fa[:, None, :] - fb[None, :, :]
    diff -= np.round(diff)
    shifted = diff[:, :, None, :] + _IMAGE_SHIFTS[None, None, :, :]
    cart = shifted @ lat                              # (Na, Nb, 27, 3)
    dist = np.sqrt(np.einsum('ijkl,ijkl->ijk', cart, cart))
    return cart, dist


def self_image_distances(lat: np.ndarray) -> np.ndarray:
    """
    Distances from any atom to its own periodic images, shape (13,).

    The displacement between an atom and its own image is a lattice
    vector, independent of where in the cell the atom sits, so this is a
    property of the CELL alone -- the same 13 numbers serve every atom.

    Only half of the 26 non-zero shifts are returned: a shift and its
    negation describe the same physical contact seen from either end, so
    returning both would double-count it.
    """
    half = _IMAGE_SHIFTS[ZERO_SHIFT_INDEX + 1:]      # 13 shifts, one per +/- pair
    return np.linalg.norm(half @ lat, axis=1)


def shortest_lattice_translation(lat: np.ndarray) -> float:
    """
    Length of the shortest non-zero lattice translation, searched over the
    same 27 images as everything else in this module.

    This is how far an atom sits from its own nearest periodic image, and
    it is the SAME for every atom in the cell -- the displacement between
    an atom and its own image is a lattice vector, independent of where in
    the cell the atom sits. That makes the hard-sphere check against
    self-images a single scalar comparison rather than an N-body one.
    """
    d = np.linalg.norm(_IMAGE_SHIFTS @ lat, axis=1)
    return float(d[d > 1e-12].min())


# Bound once at import time (not re-checked per call): this function is
# invoked thousands to tens of thousands of times per structure
# candidate during Stage-1 optimisation (see phase G's own Context
# section), so even a per-call `if NUMBA_AVAILABLE` branch would add up.
# Falls back to the pure-NumPy implementation, unchanged from every
# prior phase, when numba is not installed -- mizuho[numba] is an
# optional extra, never a hard dependency.
pairwise_distances_pbc = (
    _pairwise_distances_pbc_numba if NUMBA_AVAILABLE
    else _pairwise_distances_pbc_numpy)
