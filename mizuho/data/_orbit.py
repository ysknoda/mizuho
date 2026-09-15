"""Generic Wyckoff-orbit expansion, driven entirely by a space group's
true symmetry operations (rotation+translation pairs, see
symmetry_ops.py) plus a WyckoffEntry's coord_template.

This replaces the original per-position hand-written generator lambdas
(1,301 of them) with ONE mechanism: apply every operation to the
representative point and deduplicate modulo the lattice. Correctness
no longer depends on any single hand-transcribed formula -- it follows
directly from the space group's operation list, which is generated
mechanically from spglib (see scripts/generate_symmetry_ops_table.py)
rather than typed by hand per Wyckoff position.

Kept dependency-free (no spglib import here): SYMMETRY_OPS_OF_SG is
static data computed once at development time, so end users of the
installed package never need spglib at runtime.
"""

from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np

from .symmetry_ops import SYMMETRY_OPS_OF_SG


@lru_cache(maxsize=None)
def get_ops(sg: int) -> Tuple[np.ndarray, np.ndarray]:
    """Cached numpy view of SYMMETRY_OPS_OF_SG[sg] (stored as plain
    nested lists so the data file stays simple, generated source)."""
    rotations, translations = SYMMETRY_OPS_OF_SG[sg]
    return (np.array(rotations, dtype=float), np.array(translations, dtype=float))


def build_point(coord_template: Tuple, param_values: Dict[str, float]) -> np.ndarray:
    """coord_template: 3-tuple, each element either:
      - a fixed float, or
      - a {param_name: coefficient} dict describing a linear combination
        of free parameters (usually one entry with coefficient 1, e.g.
        {"x": 1}; coupled positions like (x,x,z) reuse the same name in
        two slots, and a few hexagonal positions need a coefficient of
        2, e.g. (x,2x,z)), or
      - a (coeffs_dict, offset) pair when a coupled slot has a non-zero
        additive constant relative to the other slot(s) sharing that
        parameter, e.g. (x, x+1/2, z): here the SAME parameter is used
        with a genuinely different offset in different slots, which is
        NOT droppable the way a single free parameter's own arbitrary
        origin is -- y=x and y=x+1/2 are different geometric lines, not
        two labellings of the same one, since dropping the offset would
        change which points are reachable relative to the OTHER slot(s)
        using that same parameter."""
    out = []
    for slot in coord_template:
        if isinstance(slot, tuple):
            coeffs, offset = slot
            out.append(sum(coeff * param_values[name] for name, coeff in coeffs.items())
                       + offset)
        elif isinstance(slot, dict):
            out.append(sum(coeff * param_values[name] for name, coeff in slot.items()))
        else:
            out.append(float(slot))
    return np.array(out, dtype=float)


def _frac_delta(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    d = p - q
    d -= np.round(d)
    return d


def expand_orbit(rotations, translations, point: np.ndarray,
                  tol: float = 1e-4) -> List[Tuple[float, float, float]]:
    """Apply every (R, t) op to `point`, dedupe modulo the lattice."""
    uniq: List[np.ndarray] = []
    for R, t in zip(rotations, translations):
        p2 = (R @ point + t) % 1.0
        if not any(np.all(np.abs(_frac_delta(p2, q)) < tol) for q in uniq):
            uniq.append(p2)
    return [tuple(p) for p in uniq]


def true_stabilizer_size(rotations, translations, point: np.ndarray,
                          tol: float = 1e-4) -> int:
    """Number of ops that fix `point` modulo the lattice -- used only by
    the dev-time regeneration/validation script to cross-check declared
    multiplicities, not by the runtime structure builder."""
    count = 0
    for R, t in zip(rotations, translations):
        d = _frac_delta(R @ point + t, point)
        if np.all(np.abs(d) < tol):
            count += 1
    return count
