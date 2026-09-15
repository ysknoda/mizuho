"""Wyckoff position database for all 230 space groups.

The data itself lives in seven per-crystal-system modules (triclinic,
monoclinic, orthorhombic, tetragonal, trigonal, hexagonal, cubic) --
split along the same SG-number ranges used elsewhere in this package
(see mizuho.lattice.CRYSTAL_SYSTEM_OF_SG) -- purely to keep any single
file from growing unmanageably large. This module merges them into one
WYCKOFF_DB dict, built once at import time, exactly as the original
single-file implementation's module-level WYCKOFF_DB = _make_db() did.
"""

from typing import Dict, List

from ._wyckoff_entry import WyckoffEntry
from ._orbit import build_point, expand_orbit, get_ops
from . import triclinic, monoclinic, orthorhombic, tetragonal
from . import trigonal, hexagonal, cubic
from .point_groups import IS_SYMMORPHIC_OF_SG, POINT_GROUP_OF_SG
from .point_group_priors import space_group_prior_weight

__all__ = [
    "WyckoffEntry", "WYCKOFF_DB",
    "POINT_GROUP_OF_SG", "IS_SYMMORPHIC_OF_SG",
    "build_point", "expand_orbit", "get_ops",
    "space_group_prior_weight",
]

_MODULES = (
    triclinic, monoclinic, orthorhombic, tetragonal,
    trigonal, hexagonal, cubic,
)


def _build_db() -> Dict[int, List[WyckoffEntry]]:
    db: Dict[int, List[WyckoffEntry]] = {}
    for module in _MODULES:
        db.update(module.get_entries())
    return db


WYCKOFF_DB: Dict[int, List[WyckoffEntry]] = _build_db()
