"""Density (volume per formula unit, ADR-0030) and largest empty sphere
(ADR-0031)."""

import glob
import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mizuho.io.cif import read_cif
from mizuho.objective.cell_terms import (
    DEFAULT_R_HOLE_MAX,
    density_band_u_a3,
    density_term,
    r_hole,
    r_hole_bounds,
    void_settings,
    void_term,
    volume_per_formula_unit_band,
)

ROOT = Path(__file__).resolve().parents[2]
REFERENCES = sorted(glob.glob(str(ROOT / "tests/fixtures/reference/*.cif")))


# ----------------------------------------------------------------------
# Density
# ----------------------------------------------------------------------


def test_density_units_agree_and_match_the_conversion_script():
    """Rutile D(meas.) 4.23(2) g/cm^3 (Handbook of Mineralogy,
    references/rutile.pdf): both units give the band the script gives."""
    g = volume_per_formula_unit_band(
        {"composition": "TiO2", "density_g_cm3": [4.21, 4.25]}
    )
    u = volume_per_formula_unit_band(
        {
            "composition": "TiO2",
            "density_u_A3": [4.21 * 0.602214076, 4.25 * 0.602214076],
        }
    )
    assert g == pytest.approx(u, rel=1e-12)
    out = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/density_to_volume.py"),
            "TiO2",
            "1",
            "4.21",
            "4.25",
        ],
        capture_output=True,
        text=True,
    ).stdout
    lo, hi = (float(x) for x in out.split("[")[1].split("]")[0].split(","))
    assert g[0] == pytest.approx(lo, abs=0.005) and g[1] == pytest.approx(hi, abs=0.005)
    # The Handbook's own rutile cell (a 4.5937, c 2.9587, Z 2) lies inside.
    assert g[0] <= 4.5937**2 * 2.9587 / 2 <= g[1]


def test_single_density_gets_two_percent_and_both_units_is_an_error():
    lo, hi = density_band_u_a3({"density_g_cm3": 4.0})
    assert lo / 0.602214076 == pytest.approx(
        3.92
    ) and hi / 0.602214076 == pytest.approx(4.08)
    with pytest.raises(ValueError):
        density_band_u_a3({"density_g_cm3": 4.0, "density_u_A3": 2.4})
    assert density_band_u_a3({}) is None


def test_density_term_holds_for_any_formula_unit_count():
    """Rutile written with Z = 2 and as a doubled cell (Z = 4) has the same
    volume per formula unit, so the same penalty: zero."""
    cfg = {"composition": "TiO2", "density_g_cm3": [4.21, 4.25]}
    lat = np.diag([4.5937, 4.5937, 2.9587])
    assert density_term(lat, ["Ti"] * 2 + ["O"] * 4, cfg)["penalty"] == 0.0
    lat2 = np.diag([4.5937, 4.5937, 2 * 2.9587])
    assert density_term(lat2, ["Ti"] * 4 + ["O"] * 8, cfg)["penalty"] == 0.0
    squeezed = np.diag([4.5937, 4.5937, 2.5])
    assert density_term(squeezed, ["Ti"] * 2 + ["O"] * 4, cfg)["penalty"] > 0.0


# ----------------------------------------------------------------------
# Largest empty sphere
# ----------------------------------------------------------------------

_DIAMOND = [
    [0, 0, 0],
    [0, 0.5, 0.5],
    [0.5, 0, 0.5],
    [0.5, 0.5, 0],
    [0.25, 0.25, 0.25],
    [0.25, 0.75, 0.75],
    [0.75, 0.25, 0.75],
    [0.75, 0.75, 0.25],
]
_ROCK_SALT = [list(p) for p in itertools.product((0, 0.5), repeat=3)]


@pytest.mark.parametrize(
    "name,lat,frac,expected",
    [
        # one atom per cubic cell: the body centre, a sqrt(3)/2
        ("simple cubic", np.eye(3) * 3.0, [[0, 0, 0]], 3.0 * np.sqrt(3) / 2),
        # diamond Si: the empty tetrahedral site, a sqrt(3)/4
        ("diamond Si", np.eye(3) * 5.431, _DIAMOND, 5.431 * np.sqrt(3) / 4),
        # rock salt as one species: the tetrahedral site, a sqrt(3)/4
        ("rock salt", np.eye(3) * 5.64, _ROCK_SALT, 5.64 * np.sqrt(3) / 4),
        # a 2.5 A square net 20 A apart: the gap's centre above a square's centre
        (
            "open layer",
            np.diag([2.5, 2.5, 20.0]),
            [[0, 0, 0]],
            np.sqrt(2 * 1.25**2 + 10.0**2),
        ),
    ],
)
def test_r_hole_matches_analytic_values(name, lat, frac, expected):
    assert r_hole(lat, frac) == pytest.approx(expected, abs=1e-6)
    lower, upper = r_hole_bounds(lat, frac)
    assert lower - 1e-9 <= expected <= upper + 1e-9


def test_r_hole_is_bracketed_by_the_grid_bounds_in_a_skewed_cell():
    lat = np.array([[3.0, 0, 0], [1.5, 2.8, 0], [0.7, 0.9, 4.1]])
    frac = [[0.1, 0.2, 0.3], [0.6, 0.55, 0.8]]
    lower, upper = r_hole_bounds(lat, frac)
    exact = r_hole(lat, frac)
    # A fine brute-force grid never exceeds the exact value.
    sh = np.array(list(itertools.product(range(-3, 4), repeat=3)))
    pts = ((np.array(frac)[None] + sh[:, None]).reshape(-1, 3)) @ lat
    g = (
        np.stack(np.meshgrid(*[np.arange(60) / 60] * 3, indexing="ij"), -1).reshape(
            -1, 3
        )
        @ lat
    )
    brute = np.min(np.linalg.norm(g[:, None] - pts[None], axis=-1), axis=1).max()
    assert lower <= exact <= upper
    assert brute <= exact + 1e-9 and exact - brute < 0.05


@pytest.mark.parametrize("path", REFERENCES, ids=[Path(p).stem for p in REFERENCES])
def test_default_void_bound_holds_at_every_reference(path):
    """Every reference with all its atoms is inside the default band (the
    H-less adamite.cif leaves H's site empty and is the one exception)."""
    lat, frac, _ = read_cif(path).expand()
    r = r_hole(lat, frac)
    if Path(path).name == "adamite.cif":
        assert r > DEFAULT_R_HOLE_MAX
    else:
        assert r <= DEFAULT_R_HOLE_MAX


def test_void_defaults_only_when_volume_is_not_otherwise_bounded():
    assert void_settings({}) == (0.0, DEFAULT_R_HOLE_MAX, 1.0)
    assert void_settings({"density_g_cm3": 4.2}) is None
    assert (
        void_settings({"volume_constraint": {"volume_min": 1, "volume_max": 2}}) is None
    )
    assert void_settings({"void": False}) is None
    assert void_settings({"density_g_cm3": 4.2, "void": {"r_hole_min": 3.0}}) == (
        3.0,
        np.inf,
        1.0,
    )
    with pytest.raises(ValueError):
        void_settings({"void": {"r_hole_min": 3.0, "r_hole_max": 2.0}})


def test_void_penalty_follows_the_analytic_r_hole():
    """Open layer of the test above, c varied: the penalty is
    w ((R - r_max) / r_max)^2 with R analytic, so it is continuous and its
    finite-difference slope matches the analytic one."""
    cfg = {"void": {"r_hole_max": 2.65}}
    R = lambda c: np.sqrt(2 * 1.25**2 + (c / 2) ** 2)
    pen = lambda c: void_term(np.diag([2.5, 2.5, c]), [[0, 0, 0]], cfg)["penalty"]
    for c in (6.0, 9.0, 14.0):
        assert pen(c) == pytest.approx(((R(c) - 2.65) / 2.65) ** 2, rel=1e-9)
        h = 1e-4
        fd = (pen(c + h) - pen(c - h)) / (2 * h)
        analytic = 2 * (R(c) - 2.65) / 2.65**2 * (c / 4) / R(c)
        assert fd == pytest.approx(analytic, rel=1e-5)
    # Dense enough: exactly zero, decided by the grid bound alone.
    # a = 2.5: R = 2.165, grid upper bound below 2.65.
    t = void_term(np.eye(3) * 2.5, [[0, 0, 0]], cfg)
    assert t["penalty"] == 0.0 and not t["exact"]
    # a = 3.0: R = 2.598, too close for the bound; computed exactly.
    t = void_term(np.eye(3) * 3.0, [[0, 0, 0]], cfg)
    assert t["penalty"] == 0.0 and t["exact"]


def test_r_hole_min_asks_for_an_open_structure():
    cfg = {"void": {"r_hole_min": 4.0}}
    dense = void_term(np.eye(3) * 3.0, [[0, 0, 0]], cfg)
    open_ = void_term(np.diag([2.5, 2.5, 20.0]), [[0, 0, 0]], cfg)
    assert dense["penalty"] > 0.0 and open_["penalty"] == 0.0


def test_void_term_skips_a_collapsed_cell():
    """A singular lattice (CMA-ES can propose one in triclinic groups)
    must not raise; it crashed the first ADR-0030 ZnO run in P-1."""
    lat = np.array([[3.0, 0, 0], [0, 3.0, 0], [3.0, 3.0, 0]])
    assert void_term(lat, [[0, 0, 0]], {}) is None


def test_local_shell_configs_switch_the_void_bound_off():
    """Phase 1's local-shell problems are not whole cells; the default void
    bound slowed and broke them (ADR-0031)."""
    from mizuho.geometry.local_shell import build_mini_config
    group = {"bond_constraints": [], "angle_constraints": [], "coordination_constraints": []}
    mini = build_mini_config({"spacegroup": 1, "lattice_bounds": {}}, [{}], group, [0])
    assert void_settings(mini) is None
