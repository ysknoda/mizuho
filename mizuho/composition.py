"""Composition-string parsing."""

import re
from typing import Dict, List


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
