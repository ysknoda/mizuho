"""Human-readable structure summary and penalty-breakdown reporting."""

from collections import Counter
from typing import List, Optional

import numpy as np

from ..objective.breakdown import error_breakdown, print_error_breakdown
from ..structure import SiteAssignment


def print_summary(structure: dict, assignments: List[SiteAssignment],
                  config: dict, error: float,
                  param_vector: Optional[np.ndarray] = None):
    """
    Print a full structure summary including the penalty breakdown table.
    If param_vector is provided, error_breakdown() is called to produce
    the detailed per-term table.
    """
    lp  = structure["lattice_params"]
    lat = structure["lattice_matrix"]
    sp  = structure["species"]
    fc  = structure["frac_coords"]

    vol = abs(float(np.dot(lat[0], np.cross(lat[1], lat[2]))))
    print(f"\n{'='*65}")
    print("  STRUCTURE SUMMARY")
    print(f"{'='*65}")
    print(f"  Composition : {config['composition']}")
    print(f"  Space group : {config.get('spacegroup','')} "
          f"({config.get('spacegroup_symbol','')})")
    print(f"  Wyckoff sites:")
    for a in assignments:
        print(f"    {a.element} @ {a.entry.letter} "
              f"(mult={a.entry.multiplicity}, "
              f"free_params={a.entry.free_params})")
    print(f"\n  Lattice constants:")
    print(f"    a={lp['a']:.5f}  b={lp['b']:.5f}  c={lp['c']:.5f} A")
    print(f"    alpha={lp.get('alpha',90):.3f}  beta={lp.get('beta',90):.3f}  "
          f"gamma={lp.get('gamma',90):.3f} deg")
    print(f"    Volume = {vol:.3f} A^3")
    print(f"  Atom count  : {dict(Counter(sp))}")
    print(f"\n  Fractional coordinates:")
    print(f"  {'No.':>4}  {'Elem':>4}  {'x':>10}  {'y':>10}  {'z':>10}")
    print(f"  {'-'*52}")
    for i, (s, f) in enumerate(zip(sp, fc)):
        print(f"  {i+1:>4}  {s:>4}  {f[0]:10.6f}  {f[1]:10.6f}  {f[2]:10.6f}")
    print(f"\n  Total error : {error:.6f}")

    # Penalty breakdown table
    if param_vector is not None:
        bd = error_breakdown(param_vector, config)
        print_error_breakdown(bd)

    print(f"{'='*65}")
