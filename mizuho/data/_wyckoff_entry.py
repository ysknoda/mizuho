"""The WyckoffEntry data model shared by every per-crystal-system module.

Kept in its own leaf module (rather than in __init__.py) so the seven
crystal-system data modules can import it directly without going through
the data package's __init__.py -- this avoids a circular import between
__init__.py (which imports the crystal-system modules to build
WYCKOFF_DB) and the crystal-system modules themselves.

No per-entry generator callable: the orbit for any WyckoffEntry is
expanded generically (data/_orbit.py) from coord_template plus the
space group's true symmetry operations (data/symmetry_ops.py), so
there is exactly one orbit-expansion mechanism for all 230 space
groups instead of 1,301 hand-written lambdas.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

# Each of the 3 slots is one of:
#   - a fixed fractional value
#   - a dict of {free_param_name: integer_coefficient} describing a
#     linear combination of free parameters (almost always a single
#     entry with coefficient 1; a handful of hexagonal/trigonal
#     positions need coefficient 2, e.g. (x, 2x, z); coupled positions
#     like (x,x,z) reuse the same name in two slots)
#   - a (coeffs_dict, offset) pair for a coupled slot with a non-zero
#     additive constant relative to another slot sharing that
#     parameter, e.g. (x, x+1/2, z) -- see data/_orbit.py's build_point
#     docstring for why this offset is NOT droppable in general.
CoordSlot = Union[float, Dict[str, int], Tuple[Dict[str, int], float]]


@dataclass
class WyckoffEntry:
    letter:         str
    multiplicity:   int
    free_params:    List[str]           # unique free-parameter names, e.g. [], ["x"], ["x","z"]
    coord_template: Tuple[CoordSlot, CoordSlot, CoordSlot]
