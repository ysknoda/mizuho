"""Wyckoff position database: monoclinic space groups (SG 3-15).

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
    """Return the Wyckoff entries for SG 3-15."""
    db: Dict[int, List[WyckoffEntry]] = {}

    db[3] = [
        WyckoffEntry('a', 1, ['x'], (0.0, {'x': 1}, 0.0)),
        WyckoffEntry('b', 1, ['x'], (0.0, {'x': 1}, 0.5)),
        WyckoffEntry('c', 1, ['x'], (0.5, {'x': 1}, 0.0)),
        WyckoffEntry('d', 1, ['x'], (0.5, {'x': 1}, 0.5)),
        WyckoffEntry('e', 2, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[4] = [
        WyckoffEntry('a', 2, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[5] = [
        WyckoffEntry('a', 2, ['x'], (0.0, {'x': 1}, 0.0)),
        WyckoffEntry('b', 2, ['x'], (0.0, {'x': 1}, 0.5)),
        WyckoffEntry('c', 4, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[6] = [
        WyckoffEntry('a', 1, ['x', 'y'], ({'x': 1}, 0.0, {'y': 1})),
        WyckoffEntry('b', 1, ['x', 'y'], ({'x': 1}, 0.5, {'y': 1})),
        WyckoffEntry('c', 2, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[7] = [
        WyckoffEntry('a', 2, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[8] = [
        WyckoffEntry('a', 2, ['x', 'y'], ({'x': 1}, 0.0, {'y': 1})),
        WyckoffEntry('b', 4, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[9] = [
        WyckoffEntry('a', 4, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[10] = [
        WyckoffEntry('a', 1, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 1, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('c', 1, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('d', 1, [], (0.5, 0.0, 0.0)),
        WyckoffEntry('e', 1, [], (0.5, 0.5, 0.0)),
        WyckoffEntry('f', 1, [], (0.0, 0.5, 0.5)),
        WyckoffEntry('g', 1, [], (0.5, 0.0, 0.5)),
        WyckoffEntry('h', 1, [], (0.5, 0.5, 0.5)),
        WyckoffEntry('i', 2, ['x'], (0.0, {'x': 1}, 0.0)),
        WyckoffEntry('j', 2, ['x'], (0.5, {'x': 1}, 0.0)),
        WyckoffEntry('k', 2, ['x'], (0.0, {'x': 1}, 0.5)),
        WyckoffEntry('l', 2, ['x'], (0.5, {'x': 1}, 0.5)),
        WyckoffEntry('m', 2, ['x', 'y'], ({'x': 1}, 0.0, {'y': 1})),
        WyckoffEntry('n', 2, ['x', 'y'], ({'x': 1}, 0.5, {'y': 1})),
        WyckoffEntry('o', 4, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[11] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.5, 0.0, 0.0)),
        WyckoffEntry('c', 2, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('d', 2, [], (0.5, 0.0, 0.5)),
        WyckoffEntry('e', 2, ['x', 'y'], ({'x': 1}, 0.25, {'y': 1})),
        WyckoffEntry('f', 4, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[12] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('c', 2, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('d', 2, [], (0.0, 0.5, 0.5)),
        WyckoffEntry('e', 4, [], (0.25, 0.25, 0.0)),
        WyckoffEntry('f', 4, [], (0.25, 0.25, 0.5)),
        WyckoffEntry('g', 4, ['x'], (0.0, {'x': 1}, 0.0)),
        WyckoffEntry('h', 4, ['x'], (0.0, {'x': 1}, 0.5)),
        WyckoffEntry('i', 4, ['x', 'y'], ({'x': 1}, 0.0, {'y': 1})),
        WyckoffEntry('j', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[13] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.5, 0.5, 0.0)),
        WyckoffEntry('c', 2, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('d', 2, [], (0.5, 0.0, 0.0)),
        WyckoffEntry('e', 2, ['x'], (0.0, {'x': 1}, 0.25)),
        WyckoffEntry('f', 2, ['x'], (0.5, {'x': 1}, 0.25)),
        WyckoffEntry('g', 4, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[14] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.5, 0.0, 0.0)),
        WyckoffEntry('c', 2, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('d', 2, [], (0.5, 0.0, 0.5)),
        WyckoffEntry('e', 4, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[15] = [
        WyckoffEntry('a', 4, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 4, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('c', 4, [], (0.25, 0.25, 0.0)),
        WyckoffEntry('d', 4, [], (0.25, 0.25, 0.5)),
        WyckoffEntry('e', 4, ['x'], (0.0, {'x': 1}, 0.25)),
        WyckoffEntry('f', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    return db
