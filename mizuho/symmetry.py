"""Symmetry detection and the mapping "structure in any setting -> mizuho's
Wyckoff assignment".

mizuho's Wyckoff tables (data/) were generated from spglib's operations
for the FIRST Hall number of each space group. spglib standardises a
structure into exactly that setting when asked for that Hall number, so
one mechanism serves both callers named in
docs/adr/0010-success-is-the-detected-symmetry.md: reading a CIF written
in another setting or origin choice, and re-expressing an optimised
structure in a detected supergroup.

Standardisation also idealises positions. Published CIFs round special
positions (wurtzite writes 1/3 as 0.3333); built as written, spglib finds
only an orthorhombic subgroup at 1e-4 A. Idealising at the tolerance that
recovers the declared group removes that rounding before the structure is
used.
"""

from collections.abc import Sequence
import warnings
from dataclasses import dataclass
from functools import cache
from itertools import product

import numpy as np
import spglib
import spglib.error

from .data import WYCKOFF_DB, build_point, expand_orbit, get_ops
from .data._orbit import _frac_delta
from .lattice import get_lattice_system

# Tolerances (A) at which the detected space group is always reported
# (ADR-0010). The one success is judged at is measured, not chosen here.
TOLERANCE_LADDER = (1e-4, 1e-3, 1e-2, 1e-1)

# Tolerance (A) at which success is judged and a higher detected symmetry
# is acted on (ADR-0010). Measured, not chosen: where the optimiser reaches
# a higher-symmetry point it does so to <= 1e-9 A (~1e-5 A after a CIF's
# 6-decimal rounding), while zero-error structures that are genuinely
# distorted by band slack sit 1e-2 to 1e-1 A from the higher group
# (CsCl/Si/NaCl automatic runs, docs/plan-verification-and-tree-search.md).
# 1e-3 A clears the first by 100x and the second by 10x.
SUCCESS_TOLERANCE = 1e-3

# Tolerances tried, smallest first, when idealising a structure whose
# space group is known (a CIF declares one). 1e-5 A first so an exact
# structure is never perturbed by a looser tolerance than it needs.
_IDEALISE_LADDER = (1e-5, 1e-4, 1e-3, 1e-2, 5e-2)

# Fractional tolerance for matching an idealised position to a Wyckoff
# template. Idealised positions are exact to ~1e-10; this is loose only
# so float noise never rejects a correct match.
_MATCH_TOL = 1e-5


@cache
def hall_number_of_sg(sg: int) -> int:
    """First Hall number of a space group: the setting mizuho's tables use
    (scripts/generate_symmetry_ops_table.py applies the same rule)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for hall in range(1, 531):
            if spglib.get_spacegroup_type(hall).number == sg:
                return hall
    raise ValueError(f"no Hall number for space group {sg}")


def _dataset(cell, symprec: float, hall_number: int = 0):
    """spglib's symmetry dataset, or None when it finds no symmetry.

    spglib 2.x warns on every call made under its legacy error handling;
    the warning concerns spglib's own API migration, not the result, so it
    is silenced here and failures of either kind are reported as None."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            return spglib.get_symmetry_dataset(cell, symprec=symprec, hall_number=hall_number)
        except spglib.error.SpglibError:
            return None


def _spglib_cell(lattice: np.ndarray, frac: np.ndarray, species: Sequence[str]):
    kinds = list(dict.fromkeys(species))
    types = [kinds.index(s) + 1 for s in species]
    return (np.asarray(lattice, float), np.asarray(frac, float) % 1.0, types), kinds


def point_group_order(sg: int) -> int:
    """Number of distinct rotations in `sg`: its point group's order, the
    measure of symmetry the global ranking prefers among constraint-
    satisfying entries (ADR-0026). Fm-3m 48, P4mm 8, P1 1."""
    rot, _ = get_ops(sg)
    return len({tuple(np.rint(r).astype(int).ravel()) for r in rot})


def detect_spacegroup(
    lattice, frac, species, symprec: float
) -> tuple[int, str] | None:
    """(number, international symbol) spglib detects at `symprec` A, or
    None if it finds no symmetry at all (degenerate input)."""
    cell, _ = _spglib_cell(lattice, frac, species)
    ds = _dataset(cell, symprec)
    if ds is None:
        return None
    return int(ds.number), str(ds.international)


def symmetry_ladder(
    lattice, frac, species, tolerances: Sequence[float] = TOLERANCE_LADDER
) -> dict[float, tuple[int, str] | None]:
    """Detected space group at every tolerance of the ladder (ADR-0010)."""
    return {tol: detect_spacegroup(lattice, frac, species, tol) for tol in tolerances}


@dataclass
class MizuhoStructure:
    """A structure expressed in mizuho's own setting: the Wyckoff
    assignment and the parameter vector build_structure takes."""

    spacegroup: int
    specs: list[dict]  # [{"element", "letter"}, ...]
    param_vector: np.ndarray
    symprec: float  # tolerance the structure was idealised at

    def label(self) -> str:
        return " ".join(f"{s['element']}@{s['letter']}" for s in self.specs)

    def config(self) -> dict:
        """Minimal config for build_structure / the objective."""
        from .structure import assignments_to_sites, resolve_fixed_assignment

        asg = resolve_fixed_assignment(self.spacegroup, self.specs)
        return {
            "spacegroup": self.spacegroup,
            "wyckoff_sites": assignments_to_sites(asg),
        }


def _lattice_params(lattice: np.ndarray) -> dict[str, float]:
    a, b, c = np.linalg.norm(lattice, axis=1)

    def angle(u, v):
        return float(
            np.degrees(
                np.arccos(
                    np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)
                )
            )
        )

    return {
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "alpha": angle(lattice[1], lattice[2]),
        "beta": angle(lattice[0], lattice[2]),
        "gamma": angle(lattice[0], lattice[1]),
    }


LATTICE_KEYS = {
    "cubic": ("a",),
    "tetragonal": ("a", "c"),
    "orthorhombic": ("a", "b", "c"),
    "hexagonal": ("a", "c"),
    "trigonal": ("a", "c"),
    "monoclinic": ("a", "b", "c", "beta"),
    "triclinic": ("a", "b", "c", "alpha", "beta", "gamma"),
}


def lattice_vector(sg: int, lattice: np.ndarray) -> list[float]:
    """The lattice part of mizuho's parameter vector for `sg`."""
    lp = _lattice_params(lattice)
    return [lp[k] for k in LATTICE_KEYS[get_lattice_system(sg)]]


def _affine_template(template) -> tuple[list[str], np.ndarray, np.ndarray]:
    """coord_template -> (names, A, b) with point = A @ p + b."""
    names: list[str] = []
    for slot in template:
        coeffs = (
            slot[0]
            if isinstance(slot, tuple)
            else slot
            if isinstance(slot, dict)
            else {}
        )
        for n in coeffs:
            if n not in names:
                names.append(n)
    A = np.zeros((3, len(names)))
    b = np.zeros(3)
    for i, slot in enumerate(template):
        if isinstance(slot, tuple):
            coeffs, b[i] = slot
        elif isinstance(slot, dict):
            coeffs = slot
        else:
            coeffs, b[i] = {}, float(slot)
        for n, k in coeffs.items():
            A[i, names.index(n)] = k
    return names, A, b


def solve_free_params(
    sg: int, entry, point: np.ndarray, tol: float = _MATCH_TOL
) -> dict[str, float] | None:
    """Free-parameter values that put `entry`'s representative point on the
    orbit of `point`, or None if `point` does not lie on that Wyckoff
    position. Tries every symmetry image of `point`, because the
    representative point is one specific member of the orbit."""
    names, A, b = _affine_template(entry.coord_template)
    rotations, translations = get_ops(sg)
    free_rows = [i for i in range(3) if np.any(A[i])]
    fixed_rows = [i for i in range(3) if not np.any(A[i])]
    # Template coefficients are at most 2 in magnitude, so the integer
    # lattice shift absorbed by a free row is within [-2, 2].
    shifts = list(product(range(-2, 3), repeat=len(free_rows)))
    for R, t in zip(rotations, translations):
        q = R @ point + t
        if any(abs(_frac_delta(q[i], b[i])) > tol for i in fixed_rows):
            continue
        if not names:
            return {}
        for n in shifts:
            rhs = q[free_rows] - b[free_rows] + np.array(n)
            p, *_ = np.linalg.lstsq(A[free_rows], rhs, rcond=None)
            p = p % 1.0
            values = dict(zip(names, p))
            if np.all(
                np.abs(_frac_delta(build_point(entry.coord_template, values), q)) < tol
            ):
                return {k: float(v) for k, v in values.items()}
    return None


def _standardise(lattice, frac, species, sg: int, symprec: float):
    cell, kinds = _spglib_cell(lattice, frac, species)
    # Plain detection first: asking spglib for a Hall number the structure
    # does not have costs ~2 s per call, a plain detection well under 1 ms.
    plain = _dataset(cell, symprec)
    if plain is None or plain.number != sg:
        return None
    hall = hall_number_of_sg(sg)
    ds = _dataset(cell, symprec, hall)
    if ds is None or ds.number != sg:
        return None
    std = (ds.std_lattice, ds.std_positions, ds.std_types)
    # Wyckoff letters of the standardised atoms themselves (ds.wyckoffs
    # indexes the input atoms, which may sit in another cell).
    ds_std = _dataset(std, symprec, hall)
    if ds_std is None or ds_std.number != sg:
        return None
    return ds_std, std, kinds


def to_mizuho(
    lattice, frac, species, spacegroup: int, symprec: float | None = None
) -> MizuhoStructure:
    """Express a structure in mizuho's setting of `spacegroup`.

    With `symprec` None, the smallest tolerance of _IDEALISE_LADDER at
    which spglib recovers `spacegroup` is used, so rounding in the input
    is idealised away but no more symmetry is imposed than needed.
    Raises ValueError when the group is not recovered or an idealised
    site does not land on the Wyckoff position spglib names -- both mean
    the input is not the structure it claims to be.
    """
    ladder = (symprec,) if symprec is not None else _IDEALISE_LADDER
    for tol in ladder:
        res = _standardise(lattice, frac, species, spacegroup, tol)
        if res is not None:
            break
    else:
        found = {t: detect_spacegroup(lattice, frac, species, t) for t in ladder}
        raise ValueError(
            f"space group {spacegroup} not recovered at any tolerance "
            f"{list(ladder)} A; detected {found}"
        )
    ds, (std_lat, std_pos, std_types), kinds = res

    entries = {e.letter: e for e in WYCKOFF_DB[spacegroup]}
    specs, params = [], []
    for rep in sorted(set(int(i) for i in ds.equivalent_atoms)):
        letter = ds.wyckoffs[rep]
        element = kinds[std_types[rep] - 1]
        entry = entries[letter]
        n_orbit = int(np.sum(ds.equivalent_atoms == rep))
        if n_orbit != entry.multiplicity:
            raise ValueError(
                f"{element}@{letter}: {n_orbit} atoms in the standard cell "
                f"but multiplicity {entry.multiplicity} in mizuho's table"
            )
        values = solve_free_params(spacegroup, entry, std_pos[rep])
        if values is None:
            raise ValueError(
                f"{element} at {std_pos[rep]} does not lie on "
                f"SG {spacegroup} Wyckoff {letter} in mizuho's setting"
            )
        # Check the rebuilt orbit is exactly the standard cell's orbit.
        rot, tr = get_ops(spacegroup)
        orbit = np.array(
            expand_orbit(rot, tr, build_point(entry.coord_template, values))
        )
        target = std_pos[ds.equivalent_atoms == rep]
        for p in target:
            if not np.any(np.all(np.abs(_frac_delta(orbit, p)) < 1e-4, axis=1)):
                raise ValueError(f"{element}@{letter}: rebuilt orbit misses {p}")
        specs.append({"element": element, "letter": letter})
        params.extend(values[n] for n in entry.free_params)

    pv = np.array(lattice_vector(spacegroup, std_lat) + params, dtype=float)
    return MizuhoStructure(spacegroup, specs, pv, tol)
