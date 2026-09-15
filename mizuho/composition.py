"""Composition-string parsing and composition-space discovery."""

import math
import re
from itertools import combinations_with_replacement
from typing import Dict, List, Tuple

from .data import WYCKOFF_DB


_COMPOSITION_TOKEN = re.compile(r'([A-Z][a-z]?)(\d*)')


def parse_composition(composition: str) -> Dict[str, int]:
    """
    Parse a composition string like 'TiO2', 'La2CuO4', 'BaTiO3'.
    Returns {element: count}.

    Raises ValueError on input this grammar cannot account for, rather
    than returning a partial parse. The previous re.findall version
    silently DISCARDED anything that did not match: 'tio2' returned {}
    (an empty composition that then propagated through the whole
    pipeline), 'TiO2.5' quietly became TiO2, and 'Ti0O2' produced a
    zero-count element. A composition string is the single input that
    defines the entire search problem, so a typo in it must fail loudly
    at the entry point instead of silently redefining the target.

    Note the grammar alone cannot catch every typo: 'NACL' tiles
    perfectly as N + A + C + L and is accepted, inventing the
    non-elements 'A' and 'L'. Rejecting those would need a full
    periodic table; the element-property tables in objective.penalties
    deliberately cover only 73 elements and would wrongly reject
    legitimate ones outside that set. Unknown elements do get logged
    (at DEBUG) by _hard_radius / _partial_volume when they fall back to
    generic values.
    """
    text = composition.strip()
    result: Dict[str, int] = {}
    pos = 0
    for m in _COMPOSITION_TOKEN.finditer(text):
        if m.start() != pos:
            raise ValueError(
                f"Invalid composition {composition!r}: cannot parse "
                f"{text[pos:m.start()]!r} at position {pos}. Expected an "
                f"element symbol (a capital letter optionally followed by "
                f"one lower-case letter) with an optional count, e.g. "
                f"'TiO2' or 'La2CuO4'.")
        pos = m.end()
        elem, cnt = m.group(1), m.group(2)
        count = int(cnt) if cnt else 1
        if count == 0:
            raise ValueError(
                f"Invalid composition {composition!r}: element {elem!r} has "
                f"a count of 0. Omit the element instead of writing "
                f"'{elem}0'.")
        result[elem] = result.get(elem, 0) + count

    if pos != len(text):
        raise ValueError(
            f"Invalid composition {composition!r}: trailing {text[pos:]!r} "
            f"could not be parsed as an element symbol with an optional "
            f"count. Element symbols must start with a capital letter "
            f"(e.g. 'TiO2', not 'tio2'), and counts must be whole numbers.")
    if not result:
        raise ValueError(
            f"Invalid composition {composition!r}: no element symbols "
            f"found. Element symbols must start with a capital letter "
            f"(e.g. 'TiO2', not 'tio2').")
    return result


def _all_charges_zero(ionic_q: dict, elements: List[str]) -> bool:
    """Return True when every element in the composition has charge == 0."""
    return all(ionic_q.get(e, 0) == 0 for e in elements)


def wyckoff_reachable_counts(sg: int,
                              max_sites_per_elem: int,
                              max_total: int) -> List[int]:
    """
    Return all atom counts achievable for one element by placing
    1..max_sites_per_elem Wyckoff sites in space group sg.
    Includes 0 (no atoms of this element).
    """
    mults = sorted(set(e.multiplicity for e in WYCKOFF_DB.get(sg, [])))
    if not mults:
        return [0]
    reachable: set = {0}
    for n in range(1, max_sites_per_elem + 1):
        for combo in combinations_with_replacement(mults, n):
            s = sum(combo)
            if s <= max_total:
                reachable.add(s)
    return sorted(reachable)


def discover_compositions(
        sg: int,
        elements: List[str],
        ionic_charges: Dict[str, int],
        max_sites_per_elem: int = 3,
        max_atoms: int = 36,
        max_compositions: int = 50,
        require_all_elements: bool = True,
) -> List[str]:
    """
    Enumerate all charge-neutral compositions that are geometrically
    reachable given the Wyckoff multiplicities of space group sg.

    Parameters
    ----------
    sg                  : space-group number
    elements            : list of element symbols, e.g. ['Na','Zr','P','O']
    ionic_charges       : {element: charge_value}
    max_sites_per_elem  : maximum number of Wyckoff sites per element
    max_atoms           : upper bound on total atoms in the unit cell
    max_compositions    : cap on number of returned compositions
    require_all_elements: if True, every element must appear at least once

    Returns sorted list of composition strings, e.g. ['Na2ZrP2O8', ...].

    Two operating modes are selected automatically, mirroring
    wyckoff_tree_search()'s ionic / zero-charge (covalent) modes:

    IONIC MODE (at least one element has a non-zero ionic_charges entry)
        The element with the largest |charge| is treated as the "solved"
        element and its atom count is derived from the charge-neutrality
        equation given the other elements' atom counts.

    ZERO-CHARGE MODE (every element has charge 0, or ionic_charges is
    empty/omitted)
        There is no charge equation to solve, so every combination of
        per-element reachable atom counts is enumerated directly and
        reduced to its primitive (GCD-reduced) formula.
    """
    reachable = {e: wyckoff_reachable_counts(sg, max_sites_per_elem, max_atoms)
                 for e in elements}

    seen_prim: set = set()
    results: List[Tuple[int, str]] = []

    if _all_charges_zero(ionic_charges, elements):
        # Covalent / zero-charge systems: no charge equation to solve, so
        # enumerate all combinations of reachable atom counts directly.
        elem_reach = [reachable[e] for e in elements]

        def _iter_zero(depth: int, partial_n: List[int], running_atoms: int):
            if depth == len(elements):
                if require_all_elements and any(n == 0 for n in partial_n):
                    return
                if all(n == 0 for n in partial_n):
                    return
                counts = dict(zip(elements, partial_n))

                # Primitive formula (GCD reduction)
                vals = [counts[e] for e in elements]
                g    = math.gcd(*vals)
                prim = tuple(v // g for v in vals)
                if prim in seen_prim:
                    return
                seen_prim.add(prim)

                comp_str = "".join(
                    f"{e}{counts[e]}" for e in elements if counts[e] > 0)
                results.append((running_atoms, comp_str))
                return

            for n in elem_reach[depth]:
                if n == 0 and require_all_elements:
                    continue
                new_atoms = running_atoms + n
                if new_atoms > max_atoms:
                    break   # reachable is sorted, so larger n won't help
                _iter_zero(depth + 1, partial_n + [n], new_atoms)

        _iter_zero(0, [], 0)

    else:
        # Ionic mode: pick the element with the largest |charge| as the
        # "solved" element to maximise pruning efficiency, and solve for
        # its atom count from the charge-neutrality equation given the
        # other elements' atom counts.
        charges   = [ionic_charges.get(e, 0) for e in elements]
        abs_c     = [abs(c) for c in charges]
        pivot_idx = abs_c.index(max(abs_c))   # element we solve for
        pivot_elem = elements[pivot_idx]
        pivot_charge = charges[pivot_idx]     # guaranteed != 0: not all-zero

        other_elems   = [e for i, e in enumerate(elements) if i != pivot_idx]
        other_charges = [c for i, c in enumerate(charges)  if i != pivot_idx]
        other_reach   = [reachable[e] for e in other_elems]

        def _iter(depth: int, partial_n: List[int], running_charge: int,
                  running_atoms: int):
            if depth == len(other_elems):
                # Solve for pivot element
                needed_charge = -running_charge
                if needed_charge % pivot_charge != 0:
                    return
                n_pivot = needed_charge // pivot_charge
                if n_pivot <= 0:
                    return
                if n_pivot not in reachable[pivot_elem]:
                    return
                total = running_atoms + n_pivot
                if total > max_atoms:
                    return

                # Build full count dict
                counts = {}
                for i, e in enumerate(other_elems):
                    counts[e] = partial_n[i]
                counts[pivot_elem] = n_pivot

                # Require all elements present
                if require_all_elements and any(counts.get(e, 0) == 0
                                                for e in elements):
                    return

                # Primitive formula (GCD reduction)
                vals = [counts[e] for e in elements]
                g    = math.gcd(*vals)
                prim = tuple(v // g for v in vals)
                if prim in seen_prim:
                    return
                seen_prim.add(prim)

                comp_str = "".join(
                    f"{e}{counts[e]}" for e in elements if counts[e] > 0)
                results.append((total, comp_str))
                return

            elem   = other_elems[depth]
            charge = other_charges[depth]
            for n in other_reach[depth]:
                if n == 0 and require_all_elements:
                    continue
                new_atoms  = running_atoms  + n
                new_charge = running_charge + charge * n
                if new_atoms > max_atoms:
                    break   # reachable is sorted, so larger n won't help
                _iter(depth + 1, partial_n + [n], new_charge, new_atoms)

        _iter(0, [], 0, 0)

    results.sort(key=lambda x: x[0])
    return [comp for _, comp in results[:max_compositions]]
