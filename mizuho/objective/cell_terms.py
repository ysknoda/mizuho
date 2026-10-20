"""Whole-cell terms: volume per formula unit from a measured density
(ADR-0030) and the largest empty sphere (ADR-0031).

Both are evaluated here once and called by energy.py and breakdown.py, so the
objective and its report cannot disagree.

Density (ADR-0030). `density_g_cm3` or `density_u_A3`, a value or [min, max];
exactly one of the two. Internally u/A^3 (1 g/cm^3 = 0.602214076 u/A^3), so
the volume per formula unit is V_fu = M / rho with M in u. A single value gets
+-2 % (ADR-0017). Unlike `volume_constraint` (a cell volume) it holds for any
formula-unit count, which the automatic space-group search needs.

Largest empty sphere (ADR-0031). R_hole is the largest distance from a point
of the cell to the nearest atom centre, no radii. `void: {r_hole_min,
r_hole_max, weight}` bounds it; without a stated density or
`volume_constraint`, `void` defaults to r_hole_max = 2.65 A, chosen from data
(eleven references span 1.85-2.61 A); `void: false` switches it off.
"""

import itertools
from functools import lru_cache

import numpy as np
from scipy.spatial import QhullError, Voronoi, cKDTree

from ..composition import parse_composition
from ..data.atomic_weights import ATOMIC_WEIGHT, G_CM3_TO_U_A3
from .penalties import _midpoint_scale, _penalty

DENSITY_WEIGHT = 5.0  # as volume_constraint's default weight
SINGLE_VALUE_MARGIN = 0.02  # ADR-0017
DEFAULT_R_HOLE_MAX = 2.65  # A; ADR-0031, chosen from data, not cited
VOID_WEIGHT = 1.0
_GRID_STEP = 0.5  # A; coarse grid of the cheap bound


# ----------------------------------------------------------------------
# Density -> volume per formula unit
# ----------------------------------------------------------------------


def _band(value):
    if isinstance(value, (int, float)):
        return value * (1 - SINGLE_VALUE_MARGIN), value * (1 + SINGLE_VALUE_MARGIN)
    lo, hi = (float(v) for v in value)
    if not 0 < lo <= hi:
        raise ValueError(f"density band must satisfy 0 < min <= max, got {value!r}")
    return lo, hi


@lru_cache(maxsize=64)
def formula_mass(composition: str) -> float:
    """Mass of one formula unit in u."""
    comp = parse_composition(composition)
    missing = sorted(set(comp) - set(ATOMIC_WEIGHT))
    if missing:
        raise ValueError(
            f"no standard atomic weight for {missing}; add it to "
            "mizuho/data/atomic_weights.py"
        )
    return sum(ATOMIC_WEIGHT[e] * n for e, n in comp.items())


def density_band_u_a3(config: dict):
    """(rho_min, rho_max) in u/A^3, or None when no density is stated."""
    g, u = config.get("density_g_cm3"), config.get("density_u_A3")
    if g is not None and u is not None:
        raise ValueError("state density_g_cm3 or density_u_A3, not both")
    if g is not None:
        lo, hi = _band(g)
        return lo * G_CM3_TO_U_A3, hi * G_CM3_TO_U_A3
    if u is not None:
        return _band(u)
    return None


def volume_per_formula_unit_band(config: dict):
    """(V_min, V_max) per formula unit in A^3, or None."""
    rho = density_band_u_a3(config)
    if rho is None:
        return None
    m = formula_mass(config["composition"])
    return m / rho[1], m / rho[0]


def formula_units(species, composition: str) -> float:
    comp = parse_composition(composition)
    return len(species) / sum(comp.values())


def density_term(lat, species, config: dict):
    """{'V_fu', 'V_min', 'V_max', 'penalty'} or None."""
    band = volume_per_formula_unit_band(config)
    if band is None:
        return None
    v_fu = abs(float(np.linalg.det(lat))) / formula_units(
        species, config["composition"]
    )
    lo, hi = band
    pen = float(_penalty(v_fu, lo, hi, DENSITY_WEIGHT, scale=_midpoint_scale(lo, hi)))
    return {"V_fu": v_fu, "V_min": lo, "V_max": hi, "penalty": pen}


# ----------------------------------------------------------------------
# Largest empty sphere
# ----------------------------------------------------------------------


def _face_spacings(lat):
    """Spacings of the cell's three pairs of faces."""
    return 1.0 / np.linalg.norm(np.linalg.inv(lat), axis=0)


def _images(lat, frac, reach):
    """Cartesian atoms with enough images that every point of the cell sees
    every atom within `reach`, and no image farther from the cell than that
    (a margin of reach / face spacing in each fractional direction)."""
    inv = np.linalg.inv(lat)
    widths = 1.0 / np.linalg.norm(inv, axis=0)       # spacings of the cell's faces
    margin = reach / widths
    n = np.maximum(np.ceil(margin - 1e-9).astype(int), 1)
    shifts = np.array(list(itertools.product(*[range(-k, k + 1) for k in n])))
    f = (frac[None] + shifts[:, None]).reshape(-1, 3)
    keep = np.all((f >= -margin - 1e-9) & (f <= 1 + margin + 1e-9), axis=1)
    return f[keep] @ lat, inv, widths


def r_hole_bounds(lat, frac):
    """(lower, upper) bounds on R_hole from a coarse grid: the grid's largest
    nearest-atom distance, plus half the longest diagonal of a grid cell."""
    lat = np.asarray(lat, float)
    frac = np.asarray(frac, float) % 1.0
    lengths = np.linalg.norm(lat, axis=1)
    k = np.maximum(np.ceil(lengths / _GRID_STEP).astype(int), 2)
    step = lat / k[:, None]
    half_diag = 0.5 * max(
        np.linalg.norm(s @ step)
        for s in ((1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1))
    )
    grid = np.stack(
        np.meshgrid(*[np.arange(m) / m for m in k], indexing="ij"), -1
    ).reshape(-1, 3)
    reach = _face_spacings(lat).min()
    pts, _, _ = _images(lat, frac, reach)
    while True:
        d, _ = cKDTree(pts).query(grid @ lat)
        lower = float(d.max())
        if lower + half_diag <= reach:
            return lower, lower + half_diag
        reach = (lower + half_diag) * 1.05
        pts, _, _ = _images(lat, frac, reach)


def r_hole(lat, frac, upper=None):
    """Exact R_hole: the largest empty sphere is centred on a Voronoi vertex
    of the periodic point set, so it is the farthest such vertex inside the
    cell from its nearest atom.

    With every image within reach >= R_hole of the cell present, the
    vertex of the largest sphere is a vertex of that finite set, at its true
    distance. reach is the grid's upper bound (`upper`, from r_hole_bounds),
    which R_hole cannot exceed."""
    lat = np.asarray(lat, float)
    frac = np.asarray(frac, float) % 1.0
    if upper is None:
        upper = r_hole_bounds(lat, frac)[1]
    reach = upper * (1 + 1e-6)
    pts, inv, _ = _images(lat, frac, reach)
    while True:
        try:
            verts = Voronoi(pts).vertices
        except QhullError:      # a flat set of images; more images make it 3-D
            reach *= 1.5
            pts, _, _ = _images(lat, frac, reach)
            continue
        f = verts @ inv
        verts = verts[np.all((f >= -1e-9) & (f < 1 - 1e-9), axis=1)]
        if len(verts):
            d, _ = cKDTree(pts).query(verts)
            return float(d.max())
        reach *= 1.5            # no vertex inside the cell from too few images
        pts, _, _ = _images(lat, frac, reach)


def void_settings(config: dict):
    """(r_min, r_max, weight) or None. Defaults to r_max = 2.65 A unless the
    cell volume is already bounded by a density or volume_constraint."""
    v = config.get("void")
    if v is False:
        return None
    if v is None:
        if (
            config.get("density_g_cm3") is not None
            or config.get("density_u_A3") is not None
            or config.get("volume_constraint") is not None
        ):
            return None
        v = {"r_hole_max": DEFAULT_R_HOLE_MAX}
    r_min = float(v.get("r_hole_min", 0.0))
    r_max = float(v.get("r_hole_max", np.inf))
    if not r_min <= r_max:
        raise ValueError(f"void needs r_hole_min <= r_hole_max, got {v!r}")
    return r_min, r_max, float(v.get("weight", VOID_WEIGHT))


def void_term(lat, frac, config: dict):
    """{'R_hole', 'r_min', 'r_max', 'exact', 'penalty'} or None. The exact
    Voronoi value is computed only when the grid bounds cannot show the
    penalty to be zero."""
    s = void_settings(config)
    if s is None:
        return None
    if abs(np.linalg.det(np.asarray(lat, float))) < 1e-6:
        return None             # a collapsed cell has no voids; cell validity charges it
    r_min, r_max, w = s
    lower, upper = r_hole_bounds(lat, frac)
    if upper <= r_max and lower >= r_min:
        return {
            "R_hole": lower,
            "r_min": r_min,
            "r_max": r_max,
            "exact": False,
            "penalty": 0.0,
        }
    r = r_hole(lat, frac, upper)
    scale = (
        _midpoint_scale(r_min, r_max)
        if np.isfinite(r_max) and r_min > 0
        else (r_max if np.isfinite(r_max) else r_min)
    )
    pen = float(_penalty(r, r_min, r_max, w, scale=max(scale, 1e-6)))
    return {"R_hole": r, "r_min": r_min, "r_max": r_max, "exact": True, "penalty": pen}
