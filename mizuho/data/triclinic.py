"""Wyckoff position database: triclinic space groups (SG 1-2).

Derived first-principles from spglib's true symmetry operations
(see scripts/generate_wyckoff_db.py): every space group's full set
of Wyckoff positions is found by collecting each operation's fixed-
point subspace (via Smith Normal Form over the integers), closing
under pairwise intersection, deduplicating positions related by a
group operation, and confirming each survivor's multiplicity and
canonical letter through an independent spglib re-analysis. No
hand-transcribed data and no external Wyckoff-table dependency.
No generator callables: orbits are expanded generically at
runtime via data/_orbit.py.
"""

from typing import Dict, List

from ._wyckoff_entry import WyckoffEntry


def get_entries() -> Dict[int, List[WyckoffEntry]]:
    """Return the Wyckoff entries for SG 1-2."""
    db: Dict[int, List[WyckoffEntry]] = {}

    db[1] = [
        WyckoffEntry('a', 1, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[2] = [
        WyckoffEntry('a', 1, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 1, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 1, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('d', 1, [], (0.5, 0.0, 0.0)),
        WyckoffEntry('e', 1, [], (0.5, 0.5, 0.0)),
        WyckoffEntry('f', 1, [], (0.5, 0.0, 0.5)),
        WyckoffEntry('g', 1, [], (0.0, 0.5, 0.5)),
        WyckoffEntry('h', 1, [], (0.5, 0.5, 0.5)),
        WyckoffEntry('i', 2, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    return db
