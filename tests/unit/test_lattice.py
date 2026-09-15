"""Unit tests for mizuho.lattice."""

import numpy as np
import pytest

from mizuho.lattice import (
    CRYSTAL_SYSTEM_OF_SG,
    NUMBA_AVAILABLE,
    _pairwise_distances_pbc_numpy,
    build_lattice_matrix,
    get_lattice_system,
    pairwise_distances_pbc,
)

if NUMBA_AVAILABLE:
    from mizuho.lattice import _pairwise_distances_pbc_numba


def test_crystal_system_of_sg_covers_all_230():
    for sg in range(1, 231):
        assert get_lattice_system(sg) in (
            "triclinic", "monoclinic", "orthorhombic",
            "tetragonal", "trigonal", "hexagonal", "cubic")


def test_crystal_system_boundaries():
    assert get_lattice_system(1) == "triclinic"
    assert get_lattice_system(2) == "triclinic"
    assert get_lattice_system(3) == "monoclinic"
    assert get_lattice_system(15) == "monoclinic"
    assert get_lattice_system(16) == "orthorhombic"
    assert get_lattice_system(74) == "orthorhombic"
    assert get_lattice_system(75) == "tetragonal"
    assert get_lattice_system(142) == "tetragonal"
    assert get_lattice_system(143) == "trigonal"
    assert get_lattice_system(167) == "trigonal"
    assert get_lattice_system(168) == "hexagonal"
    assert get_lattice_system(194) == "hexagonal"
    assert get_lattice_system(195) == "cubic"
    assert get_lattice_system(230) == "cubic"


def test_build_lattice_matrix_cubic_volume():
    lp = {"a": 5.0, "b": 5.0, "c": 5.0, "alpha": 90, "beta": 90, "gamma": 90}
    lat = build_lattice_matrix(lp)
    vol = abs(np.dot(lat[0], np.cross(lat[1], lat[2])))
    assert vol == pytest.approx(125.0, rel=1e-6)


def test_build_lattice_matrix_triclinic_volume():
    # Triple product formula for a triclinic cell (cross-check against
    # the standard closed-form volume expression).
    a, b, c = 5.0, 6.0, 7.0
    alpha, beta, gamma = 80.0, 95.0, 100.0
    lp = {"a": a, "b": b, "c": c, "alpha": alpha, "beta": beta, "gamma": gamma}
    lat = build_lattice_matrix(lp)
    vol = abs(np.dot(lat[0], np.cross(lat[1], lat[2])))

    ca = np.cos(np.radians(alpha))
    cb = np.cos(np.radians(beta))
    cg = np.cos(np.radians(gamma))
    expected_vol = a * b * c * np.sqrt(
        1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg)
    assert vol == pytest.approx(expected_vol, rel=1e-6)


def _brute_force_min_image(fa, fb, lat):
    """Oracle: explicit +/-1 image enumeration (27 combinations)."""
    best = np.inf
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                shift = np.array([i, j, k], dtype=float)
                diff = (fa - fb) - shift
                cart = diff @ lat
                d = np.linalg.norm(cart)
                best = min(best, d)
    return best


def test_pairwise_distances_pbc_matches_brute_force_orthogonal_cell():
    lp = {"a": 5.0, "b": 5.0, "c": 5.0, "alpha": 90, "beta": 90, "gamma": 90}
    lat = build_lattice_matrix(lp)
    fa = np.array([[0.9, 0.1, 0.5]])
    fb = np.array([[0.05, 0.95, 0.5]])
    dm = pairwise_distances_pbc(fa, fb, lat)
    oracle = _brute_force_min_image(fa[0], fb[0], lat)
    assert dm[0, 0] == pytest.approx(oracle, rel=1e-6)


def test_pairwise_distances_pbc_matches_brute_force_skewed_cell():
    """A heavily skewed triclinic cell (alpha=beta=gamma=30 deg) where
    naive component-wise rounding of the fractional difference picks
    the wrong periodic image: the fractional lattice point nearest in
    fractional-coordinate space is not always nearest in real
    (Cartesian) space once the lattice vectors are far from
    orthogonal. This is exactly the case the 27-image search (rather
    than a single rounded image) was added to handle correctly."""
    lp = {"a": 6.0, "b": 6.0, "c": 6.0, "alpha": 30.0, "beta": 30.0, "gamma": 30.0}
    lat = build_lattice_matrix(lp)
    fa = np.array([[0.1, 0.1, 0.1]])
    fb = np.array([[0.9, 0.9, 0.9]])

    naive_diff = fa[0] - fb[0]
    naive_diff -= np.round(naive_diff)
    naive_dist = np.linalg.norm(naive_diff @ lat)

    oracle = _brute_force_min_image(fa[0], fb[0], lat)
    dm = pairwise_distances_pbc(fa, fb, lat)

    assert naive_dist > oracle + 0.1, (
        "test fixture no longer demonstrates the naive-rounding failure "
        f"(naive={naive_dist}, oracle={oracle})")
    assert dm[0, 0] == pytest.approx(oracle, rel=1e-6)


# ==============================================================================
# Phase H-2: numba-accelerated pairwise_distances_pbc
# ==============================================================================

_CRYSTAL_SYSTEM_LATTICE_PARAMS = {
    "triclinic":    {"a": 5.0, "b": 6.0, "c": 7.0, "alpha": 80.0, "beta": 95.0, "gamma": 100.0},
    "monoclinic":   {"a": 5.0, "b": 6.0, "c": 7.0, "beta": 100.0},
    "orthorhombic": {"a": 5.0, "b": 6.0, "c": 7.0},
    "tetragonal":   {"a": 5.0, "b": 5.0, "c": 7.0},
    "trigonal":     {"a": 5.0, "b": 5.0, "c": 5.0, "alpha": 70.0, "beta": 70.0, "gamma": 70.0},
    "hexagonal":    {"a": 5.0, "b": 5.0, "c": 7.0, "gamma": 120.0},
    "cubic":        {"a": 5.0, "b": 5.0, "c": 5.0},
}


def test_pairwise_distances_pbc_dispatches_to_the_available_implementation():
    """The public name must be bound, once at import time, to whichever
    implementation NUMBA_AVAILABLE selects -- not re-checked per call."""
    if NUMBA_AVAILABLE:
        assert pairwise_distances_pbc is _pairwise_distances_pbc_numba
    else:
        assert pairwise_distances_pbc is _pairwise_distances_pbc_numpy


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba is not installed")
@pytest.mark.parametrize("system", sorted(_CRYSTAL_SYSTEM_LATTICE_PARAMS))
@pytest.mark.parametrize("n", [1, 2, 8, 36, 56])
def test_numba_matches_numpy_implementation(system, n):
    """The numba kernel (explicit loop, no (Na,Nb,27,3) intermediate)
    must agree with the NumPy broadcast version to floating-point
    noise across every crystal system and the tool's realistic N range
    (fixtures span N=2..56, see phase H's Context section) -- this is
    the actual correctness guarantee phase H-2 depends on, not just
    the two spot-checks the pre-existing brute-force tests exercise."""
    lat = build_lattice_matrix(_CRYSTAL_SYSTEM_LATTICE_PARAMS[system])
    rng = np.random.default_rng(hash((system, n)) % (2**32))
    fa = rng.uniform(0.0, 1.0, size=(n, 3))
    fb = rng.uniform(0.0, 1.0, size=(n, 3))

    d_numpy = _pairwise_distances_pbc_numpy(fa, fb, lat)
    d_numba = _pairwise_distances_pbc_numba(fa, fb, lat)

    assert d_numpy.shape == d_numba.shape == (n, n)
    assert np.allclose(d_numpy, d_numba, atol=1e-9)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba is not installed")
def test_numba_and_numpy_implementations_handle_empty_input():
    lat = build_lattice_matrix(_CRYSTAL_SYSTEM_LATTICE_PARAMS["cubic"])
    fa = np.empty((0, 3))
    fb = np.empty((0, 3))
    assert _pairwise_distances_pbc_numpy(fa, fb, lat).shape == (0, 0)
    assert _pairwise_distances_pbc_numba(fa, fb, lat).shape == (0, 0)
