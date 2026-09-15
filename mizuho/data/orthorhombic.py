"""Wyckoff position database: orthorhombic space groups (SG 16-74).

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
    """Return the Wyckoff entries for SG 16-74."""
    db: Dict[int, List[WyckoffEntry]] = {}

    db[16] = [
        WyckoffEntry('a', 1, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 1, [], (0.5, 0.0, 0.0)),
        WyckoffEntry('c', 1, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('d', 1, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('e', 1, [], (0.5, 0.5, 0.0)),
        WyckoffEntry('f', 1, [], (0.5, 0.0, 0.5)),
        WyckoffEntry('g', 1, [], (0.0, 0.5, 0.5)),
        WyckoffEntry('h', 1, [], (0.5, 0.5, 0.5)),
        WyckoffEntry('i', 2, ['x'], ({'x': 1}, 0.0, 0.0)),
        WyckoffEntry('j', 2, ['x'], ({'x': 1}, 0.0, 0.5)),
        WyckoffEntry('k', 2, ['x'], ({'x': 1}, 0.5, 0.0)),
        WyckoffEntry('l', 2, ['x'], ({'x': 1}, 0.5, 0.5)),
        WyckoffEntry('m', 2, ['x'], (0.0, {'x': 1}, 0.0)),
        WyckoffEntry('n', 2, ['x'], (0.0, {'x': 1}, 0.5)),
        WyckoffEntry('o', 2, ['x'], (0.5, {'x': 1}, 0.0)),
        WyckoffEntry('p', 2, ['x'], (0.5, {'x': 1}, 0.5)),
        WyckoffEntry('q', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('r', 2, ['x'], (0.5, 0.0, {'x': 1})),
        WyckoffEntry('s', 2, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('t', 2, ['x'], (0.5, 0.5, {'x': 1})),
        WyckoffEntry('u', 4, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[17] = [
        WyckoffEntry('a', 2, ['x'], ({'x': 1}, 0.0, 0.0)),
        WyckoffEntry('b', 2, ['x'], ({'x': 1}, 0.5, 0.0)),
        WyckoffEntry('c', 2, ['x'], (0.0, {'x': 1}, 0.25)),
        WyckoffEntry('d', 2, ['x'], (0.5, {'x': 1}, 0.25)),
        WyckoffEntry('e', 4, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[18] = [
        WyckoffEntry('a', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 2, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('c', 4, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[19] = [
        WyckoffEntry('a', 4, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[20] = [
        WyckoffEntry('a', 4, ['x'], ({'x': 1}, 0.0, 0.0)),
        WyckoffEntry('b', 4, ['x'], (0.0, {'x': 1}, 0.25)),
        WyckoffEntry('c', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[21] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('c', 2, [], (0.0, 0.5, 0.5)),
        WyckoffEntry('d', 2, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('e', 4, ['x'], ({'x': 1}, 0.0, 0.0)),
        WyckoffEntry('f', 4, ['x'], ({'x': 1}, 0.0, 0.5)),
        WyckoffEntry('g', 4, ['x'], (0.0, {'x': 1}, 0.0)),
        WyckoffEntry('h', 4, ['x'], (0.0, {'x': 1}, 0.5)),
        WyckoffEntry('i', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('j', 4, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('k', 4, ['x'], (0.25, 0.25, {'x': 1})),
        WyckoffEntry('l', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[22] = [
        WyckoffEntry('a', 4, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 4, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 4, [], (0.25, 0.25, 0.25)),
        WyckoffEntry('d', 4, [], (0.75, 0.25, 0.25)),
        WyckoffEntry('e', 8, ['x'], ({'x': 1}, 0.0, 0.0)),
        WyckoffEntry('f', 8, ['x'], (0.0, {'x': 1}, 0.0)),
        WyckoffEntry('g', 8, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('h', 8, ['x'], (0.25, 0.25, {'x': 1})),
        WyckoffEntry('i', 8, ['x'], (0.25, {'x': 1}, 0.25)),
        WyckoffEntry('j', 8, ['x'], ({'x': 1}, 0.25, 0.25)),
        WyckoffEntry('k', 16, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[23] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.0, 0.5, 0.5)),
        WyckoffEntry('c', 2, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('d', 2, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('e', 4, ['x'], ({'x': 1}, 0.0, 0.0)),
        WyckoffEntry('f', 4, ['x'], ({'x': 1}, 0.0, 0.5)),
        WyckoffEntry('g', 4, ['x'], (0.0, {'x': 1}, 0.0)),
        WyckoffEntry('h', 4, ['x'], (0.0, {'x': 1}, 0.5)),
        WyckoffEntry('i', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('j', 4, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('k', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[24] = [
        WyckoffEntry('a', 4, ['x'], ({'x': 1}, 0.0, 0.25)),
        WyckoffEntry('b', 4, ['x'], (0.25, {'x': 1}, 0.0)),
        WyckoffEntry('c', 4, ['x'], (0.0, 0.25, {'x': 1})),
        WyckoffEntry('d', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[25] = [
        WyckoffEntry('a', 1, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 1, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('c', 1, ['x'], (0.5, 0.0, {'x': 1})),
        WyckoffEntry('d', 1, ['x'], (0.5, 0.5, {'x': 1})),
        WyckoffEntry('e', 2, ['x', 'y'], ({'x': 1}, 0.0, {'y': 1})),
        WyckoffEntry('f', 2, ['x', 'y'], ({'x': 1}, 0.5, {'y': 1})),
        WyckoffEntry('g', 2, ['x', 'y'], (0.0, {'x': 1}, {'y': 1})),
        WyckoffEntry('h', 2, ['x', 'y'], (0.5, {'x': 1}, {'y': 1})),
        WyckoffEntry('i', 4, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[26] = [
        WyckoffEntry('a', 2, ['x', 'y'], (0.0, {'x': 1}, {'y': 1})),
        WyckoffEntry('b', 2, ['x', 'y'], (0.5, {'x': 1}, {'y': 1})),
        WyckoffEntry('c', 4, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[27] = [
        WyckoffEntry('a', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 2, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('c', 2, ['x'], (0.5, 0.0, {'x': 1})),
        WyckoffEntry('d', 2, ['x'], (0.5, 0.5, {'x': 1})),
        WyckoffEntry('e', 4, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[28] = [
        WyckoffEntry('a', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 2, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('c', 2, ['x', 'y'], (0.25, {'x': 1}, {'y': 1})),
        WyckoffEntry('d', 4, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[29] = [
        WyckoffEntry('a', 4, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[30] = [
        WyckoffEntry('a', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 2, ['x'], (0.5, 0.0, {'x': 1})),
        WyckoffEntry('c', 4, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[31] = [
        WyckoffEntry('a', 2, ['x', 'y'], (0.0, {'x': 1}, {'y': 1})),
        WyckoffEntry('b', 4, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[32] = [
        WyckoffEntry('a', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 2, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('c', 4, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[33] = [
        WyckoffEntry('a', 4, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[34] = [
        WyckoffEntry('a', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 2, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('c', 4, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[35] = [
        WyckoffEntry('a', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 2, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('c', 4, ['x'], (0.25, 0.25, {'x': 1})),
        WyckoffEntry('d', 4, ['x', 'y'], ({'x': 1}, 0.0, {'y': 1})),
        WyckoffEntry('e', 4, ['x', 'y'], (0.0, {'x': 1}, {'y': 1})),
        WyckoffEntry('f', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[36] = [
        WyckoffEntry('a', 4, ['x', 'y'], (0.0, {'x': 1}, {'y': 1})),
        WyckoffEntry('b', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[37] = [
        WyckoffEntry('a', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 4, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('c', 4, ['x'], (0.25, 0.25, {'x': 1})),
        WyckoffEntry('d', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[38] = [
        WyckoffEntry('a', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 2, ['x'], (0.5, 0.0, {'x': 1})),
        WyckoffEntry('c', 4, ['x', 'y'], ({'x': 1}, 0.0, {'y': 1})),
        WyckoffEntry('d', 4, ['x', 'y'], (0.0, {'x': 1}, {'y': 1})),
        WyckoffEntry('e', 4, ['x', 'y'], (0.5, {'x': 1}, {'y': 1})),
        WyckoffEntry('f', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[39] = [
        WyckoffEntry('a', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 4, ['x'], (0.5, 0.0, {'x': 1})),
        WyckoffEntry('c', 4, ['x', 'y'], ({'x': 1}, 0.25, {'y': 1})),
        WyckoffEntry('d', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[40] = [
        WyckoffEntry('a', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 4, ['x', 'y'], (0.25, {'x': 1}, {'y': 1})),
        WyckoffEntry('c', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[41] = [
        WyckoffEntry('a', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[42] = [
        WyckoffEntry('a', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 8, ['x'], (0.25, 0.25, {'x': 1})),
        WyckoffEntry('c', 8, ['x', 'y'], (0.0, {'x': 1}, {'y': 1})),
        WyckoffEntry('d', 8, ['x', 'y'], ({'x': 1}, 0.0, {'y': 1})),
        WyckoffEntry('e', 16, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[43] = [
        WyckoffEntry('a', 8, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 16, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[44] = [
        WyckoffEntry('a', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 2, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('c', 4, ['x', 'y'], ({'x': 1}, 0.0, {'y': 1})),
        WyckoffEntry('d', 4, ['x', 'y'], (0.0, {'x': 1}, {'y': 1})),
        WyckoffEntry('e', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[45] = [
        WyckoffEntry('a', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 4, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('c', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[46] = [
        WyckoffEntry('a', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 4, ['x', 'y'], (0.25, {'x': 1}, {'y': 1})),
        WyckoffEntry('c', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[47] = [
        WyckoffEntry('a', 1, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 1, [], (0.5, 0.0, 0.0)),
        WyckoffEntry('c', 1, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('d', 1, [], (0.5, 0.0, 0.5)),
        WyckoffEntry('e', 1, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('f', 1, [], (0.5, 0.5, 0.0)),
        WyckoffEntry('g', 1, [], (0.0, 0.5, 0.5)),
        WyckoffEntry('h', 1, [], (0.5, 0.5, 0.5)),
        WyckoffEntry('i', 2, ['x'], ({'x': 1}, 0.0, 0.0)),
        WyckoffEntry('j', 2, ['x'], ({'x': 1}, 0.0, 0.5)),
        WyckoffEntry('k', 2, ['x'], ({'x': 1}, 0.5, 0.0)),
        WyckoffEntry('l', 2, ['x'], ({'x': 1}, 0.5, 0.5)),
        WyckoffEntry('m', 2, ['x'], (0.0, {'x': 1}, 0.0)),
        WyckoffEntry('n', 2, ['x'], (0.0, {'x': 1}, 0.5)),
        WyckoffEntry('o', 2, ['x'], (0.5, {'x': 1}, 0.0)),
        WyckoffEntry('p', 2, ['x'], (0.5, {'x': 1}, 0.5)),
        WyckoffEntry('q', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('r', 2, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('s', 2, ['x'], (0.5, 0.0, {'x': 1})),
        WyckoffEntry('t', 2, ['x'], (0.5, 0.5, {'x': 1})),
        WyckoffEntry('u', 4, ['x', 'y'], (0.0, {'x': 1}, {'y': 1})),
        WyckoffEntry('v', 4, ['x', 'y'], (0.5, {'x': 1}, {'y': 1})),
        WyckoffEntry('w', 4, ['x', 'y'], ({'x': 1}, 0.0, {'y': 1})),
        WyckoffEntry('x', 4, ['x', 'y'], ({'x': 1}, 0.5, {'y': 1})),
        WyckoffEntry('y', 4, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.0)),
        WyckoffEntry('z', 4, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.5)),
        WyckoffEntry('A', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[48] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.0, 0.5, 0.5)),
        WyckoffEntry('c', 2, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('d', 2, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('e', 4, [], (0.25, 0.25, 0.25)),
        WyckoffEntry('f', 4, [], (0.25, 0.25, 0.75)),
        WyckoffEntry('g', 4, ['x'], ({'x': 1}, 0.0, 0.0)),
        WyckoffEntry('h', 4, ['x'], ({'x': 1}, 0.0, 0.5)),
        WyckoffEntry('i', 4, ['x'], (0.0, {'x': 1}, 0.0)),
        WyckoffEntry('j', 4, ['x'], (0.0, {'x': 1}, 0.5)),
        WyckoffEntry('k', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('l', 4, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('m', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[49] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.5, 0.5, 0.0)),
        WyckoffEntry('c', 2, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('d', 2, [], (0.5, 0.0, 0.0)),
        WyckoffEntry('e', 2, [], (0.0, 0.0, 0.25)),
        WyckoffEntry('f', 2, [], (0.5, 0.0, 0.25)),
        WyckoffEntry('g', 2, [], (0.0, 0.5, 0.25)),
        WyckoffEntry('h', 2, [], (0.5, 0.5, 0.25)),
        WyckoffEntry('i', 4, ['x'], ({'x': 1}, 0.0, 0.25)),
        WyckoffEntry('j', 4, ['x'], ({'x': 1}, 0.5, 0.25)),
        WyckoffEntry('k', 4, ['x'], (0.0, {'x': 1}, 0.25)),
        WyckoffEntry('l', 4, ['x'], (0.5, {'x': 1}, 0.25)),
        WyckoffEntry('m', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('n', 4, ['x'], (0.5, 0.5, {'x': 1})),
        WyckoffEntry('o', 4, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('p', 4, ['x'], (0.5, 0.0, {'x': 1})),
        WyckoffEntry('q', 4, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.0)),
        WyckoffEntry('r', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[50] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('c', 2, [], (0.0, 0.5, 0.5)),
        WyckoffEntry('d', 2, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('e', 4, [], (0.25, 0.25, 0.0)),
        WyckoffEntry('f', 4, [], (0.25, 0.25, 0.5)),
        WyckoffEntry('g', 4, ['x'], ({'x': 1}, 0.0, 0.0)),
        WyckoffEntry('h', 4, ['x'], ({'x': 1}, 0.0, 0.5)),
        WyckoffEntry('i', 4, ['x'], (0.0, {'x': 1}, 0.0)),
        WyckoffEntry('j', 4, ['x'], (0.0, {'x': 1}, 0.5)),
        WyckoffEntry('k', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('l', 4, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('m', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[51] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('c', 2, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('d', 2, [], (0.0, 0.5, 0.5)),
        WyckoffEntry('e', 2, ['x'], (0.25, 0.0, {'x': 1})),
        WyckoffEntry('f', 2, ['x'], (0.25, 0.5, {'x': 1})),
        WyckoffEntry('g', 4, ['x'], (0.0, {'x': 1}, 0.0)),
        WyckoffEntry('h', 4, ['x'], (0.0, {'x': 1}, 0.5)),
        WyckoffEntry('i', 4, ['x', 'y'], ({'x': 1}, 0.0, {'y': 1})),
        WyckoffEntry('j', 4, ['x', 'y'], ({'x': 1}, 0.5, {'y': 1})),
        WyckoffEntry('k', 4, ['x', 'y'], (0.25, {'x': 1}, {'y': 1})),
        WyckoffEntry('l', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[52] = [
        WyckoffEntry('a', 4, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 4, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 4, ['x'], (0.25, 0.0, {'x': 1})),
        WyckoffEntry('d', 4, ['x'], ({'x': 1}, 0.25, 0.25)),
        WyckoffEntry('e', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[53] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 2, [], (0.0, 0.5, 0.5)),
        WyckoffEntry('d', 2, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('e', 4, ['x'], ({'x': 1}, 0.0, 0.0)),
        WyckoffEntry('f', 4, ['x'], ({'x': 1}, 0.5, 0.0)),
        WyckoffEntry('g', 4, ['x'], (0.25, {'x': 1}, 0.25)),
        WyckoffEntry('h', 4, ['x', 'y'], (0.0, {'x': 1}, {'y': 1})),
        WyckoffEntry('i', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[54] = [
        WyckoffEntry('a', 4, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 4, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('c', 4, ['x'], (0.0, {'x': 1}, 0.25)),
        WyckoffEntry('d', 4, ['x'], (0.25, 0.0, {'x': 1})),
        WyckoffEntry('e', 4, ['x'], (0.25, 0.5, {'x': 1})),
        WyckoffEntry('f', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[55] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 2, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('d', 2, [], (0.0, 0.5, 0.5)),
        WyckoffEntry('e', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 4, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('g', 4, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.0)),
        WyckoffEntry('h', 4, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.5)),
        WyckoffEntry('i', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[56] = [
        WyckoffEntry('a', 4, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 4, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 4, ['x'], (0.25, 0.25, {'x': 1})),
        WyckoffEntry('d', 4, ['x'], (0.25, 0.75, {'x': 1})),
        WyckoffEntry('e', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[57] = [
        WyckoffEntry('a', 4, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 4, [], (0.5, 0.0, 0.0)),
        WyckoffEntry('c', 4, ['x'], ({'x': 1}, 0.25, 0.0)),
        WyckoffEntry('d', 4, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.25)),
        WyckoffEntry('e', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[58] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 2, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('d', 2, [], (0.0, 0.5, 0.5)),
        WyckoffEntry('e', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 4, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('g', 4, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.0)),
        WyckoffEntry('h', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[59] = [
        WyckoffEntry('a', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 2, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('c', 4, [], (0.25, 0.25, 0.0)),
        WyckoffEntry('d', 4, [], (0.25, 0.25, 0.5)),
        WyckoffEntry('e', 4, ['x', 'y'], (0.0, {'x': 1}, {'y': 1})),
        WyckoffEntry('f', 4, ['x', 'y'], ({'x': 1}, 0.0, {'y': 1})),
        WyckoffEntry('g', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[60] = [
        WyckoffEntry('a', 4, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 4, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('c', 4, ['x'], (0.0, {'x': 1}, 0.25)),
        WyckoffEntry('d', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[61] = [
        WyckoffEntry('a', 4, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 4, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[62] = [
        WyckoffEntry('a', 4, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 4, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 4, ['x', 'y'], ({'x': 1}, 0.25, {'y': 1})),
        WyckoffEntry('d', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[63] = [
        WyckoffEntry('a', 4, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 4, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('c', 4, ['x'], (0.0, {'x': 1}, 0.25)),
        WyckoffEntry('d', 8, [], (0.25, 0.25, 0.0)),
        WyckoffEntry('e', 8, ['x'], ({'x': 1}, 0.0, 0.0)),
        WyckoffEntry('f', 8, ['x', 'y'], (0.0, {'x': 1}, {'y': 1})),
        WyckoffEntry('g', 8, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.25)),
        WyckoffEntry('h', 16, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[64] = [
        WyckoffEntry('a', 4, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 4, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 8, [], (0.25, 0.25, 0.0)),
        WyckoffEntry('d', 8, ['x'], ({'x': 1}, 0.0, 0.0)),
        WyckoffEntry('e', 8, ['x'], (0.25, {'x': 1}, 0.25)),
        WyckoffEntry('f', 8, ['x', 'y'], (0.0, {'x': 1}, {'y': 1})),
        WyckoffEntry('g', 16, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[65] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('c', 2, [], (0.0, 0.5, 0.5)),
        WyckoffEntry('d', 2, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('e', 4, [], (0.25, 0.25, 0.0)),
        WyckoffEntry('f', 4, [], (0.25, 0.25, 0.5)),
        WyckoffEntry('g', 4, ['x'], ({'x': 1}, 0.0, 0.0)),
        WyckoffEntry('h', 4, ['x'], ({'x': 1}, 0.0, 0.5)),
        WyckoffEntry('i', 4, ['x'], (0.0, {'x': 1}, 0.0)),
        WyckoffEntry('j', 4, ['x'], (0.0, {'x': 1}, 0.5)),
        WyckoffEntry('k', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('l', 4, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('m', 8, ['x'], (0.25, 0.25, {'x': 1})),
        WyckoffEntry('n', 8, ['x', 'y'], (0.0, {'x': 1}, {'y': 1})),
        WyckoffEntry('o', 8, ['x', 'y'], ({'x': 1}, 0.0, {'y': 1})),
        WyckoffEntry('p', 8, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.0)),
        WyckoffEntry('q', 8, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.5)),
        WyckoffEntry('r', 16, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[66] = [
        WyckoffEntry('a', 4, [], (0.0, 0.0, 0.25)),
        WyckoffEntry('b', 4, [], (0.0, 0.5, 0.25)),
        WyckoffEntry('c', 4, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('d', 4, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('e', 4, [], (0.25, 0.25, 0.0)),
        WyckoffEntry('f', 4, [], (0.25, 0.25, 0.5)),
        WyckoffEntry('g', 8, ['x'], ({'x': 1}, 0.0, 0.25)),
        WyckoffEntry('h', 8, ['x'], (0.0, {'x': 1}, 0.25)),
        WyckoffEntry('i', 8, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('j', 8, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('k', 8, ['x'], (0.25, 0.25, {'x': 1})),
        WyckoffEntry('l', 8, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.0)),
        WyckoffEntry('m', 16, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[67] = [
        WyckoffEntry('a', 4, [], (0.25, 0.0, 0.0)),
        WyckoffEntry('b', 4, [], (0.25, 0.0, 0.5)),
        WyckoffEntry('c', 4, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('d', 4, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('e', 4, [], (0.25, 0.25, 0.0)),
        WyckoffEntry('f', 4, [], (0.25, 0.25, 0.5)),
        WyckoffEntry('g', 4, ['x'], (0.0, 0.25, {'x': 1})),
        WyckoffEntry('h', 8, ['x'], ({'x': 1}, 0.0, 0.0)),
        WyckoffEntry('i', 8, ['x'], ({'x': 1}, 0.0, 0.5)),
        WyckoffEntry('j', 8, ['x'], (0.25, {'x': 1}, 0.0)),
        WyckoffEntry('k', 8, ['x'], (0.25, {'x': 1}, 0.5)),
        WyckoffEntry('l', 8, ['x'], (0.25, 0.0, {'x': 1})),
        WyckoffEntry('m', 8, ['x', 'y'], (0.0, {'x': 1}, {'y': 1})),
        WyckoffEntry('n', 8, ['x', 'y'], ({'x': 1}, 0.25, {'y': 1})),
        WyckoffEntry('o', 16, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[68] = [
        WyckoffEntry('a', 4, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 4, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 8, [], (0.25, 0.0, 0.25)),
        WyckoffEntry('d', 8, [], (0.0, 0.25, 0.25)),
        WyckoffEntry('e', 8, ['x'], ({'x': 1}, 0.0, 0.0)),
        WyckoffEntry('f', 8, ['x'], (0.0, {'x': 1}, 0.0)),
        WyckoffEntry('g', 8, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('h', 8, ['x'], (0.25, 0.25, {'x': 1})),
        WyckoffEntry('i', 16, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[69] = [
        WyckoffEntry('a', 4, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 4, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 8, [], (0.0, 0.25, 0.25)),
        WyckoffEntry('d', 8, [], (0.25, 0.0, 0.25)),
        WyckoffEntry('e', 8, [], (0.25, 0.25, 0.0)),
        WyckoffEntry('f', 8, [], (0.25, 0.25, 0.25)),
        WyckoffEntry('g', 8, ['x'], ({'x': 1}, 0.0, 0.0)),
        WyckoffEntry('h', 8, ['x'], (0.0, {'x': 1}, 0.0)),
        WyckoffEntry('i', 8, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('j', 16, ['x'], (0.25, 0.25, {'x': 1})),
        WyckoffEntry('k', 16, ['x'], (0.25, {'x': 1}, 0.25)),
        WyckoffEntry('l', 16, ['x'], ({'x': 1}, 0.25, 0.25)),
        WyckoffEntry('m', 16, ['x', 'y'], (0.0, {'x': 1}, {'y': 1})),
        WyckoffEntry('n', 16, ['x', 'y'], ({'x': 1}, 0.0, {'y': 1})),
        WyckoffEntry('o', 16, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.0)),
        WyckoffEntry('p', 32, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[70] = [
        WyckoffEntry('a', 8, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 8, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 16, [], (0.125, 0.125, 0.125)),
        WyckoffEntry('d', 16, [], (0.125, 0.125, 0.625)),
        WyckoffEntry('e', 16, ['x'], ({'x': 1}, 0.0, 0.0)),
        WyckoffEntry('f', 16, ['x'], (0.0, {'x': 1}, 0.0)),
        WyckoffEntry('g', 16, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('h', 32, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[71] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.0, 0.5, 0.5)),
        WyckoffEntry('c', 2, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('d', 2, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('e', 4, ['x'], ({'x': 1}, 0.0, 0.0)),
        WyckoffEntry('f', 4, ['x'], ({'x': 1}, 0.0, 0.5)),
        WyckoffEntry('g', 4, ['x'], (0.0, {'x': 1}, 0.0)),
        WyckoffEntry('h', 4, ['x'], (0.0, {'x': 1}, 0.5)),
        WyckoffEntry('i', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('j', 4, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('k', 8, [], (0.25, 0.25, 0.25)),
        WyckoffEntry('l', 8, ['x', 'y'], (0.0, {'x': 1}, {'y': 1})),
        WyckoffEntry('m', 8, ['x', 'y'], ({'x': 1}, 0.0, {'y': 1})),
        WyckoffEntry('n', 8, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.0)),
        WyckoffEntry('o', 16, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[72] = [
        WyckoffEntry('a', 4, [], (0.0, 0.0, 0.25)),
        WyckoffEntry('b', 4, [], (0.0, 0.5, 0.25)),
        WyckoffEntry('c', 4, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('d', 4, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('e', 8, [], (0.25, 0.25, 0.25)),
        WyckoffEntry('f', 8, ['x'], ({'x': 1}, 0.0, 0.25)),
        WyckoffEntry('g', 8, ['x'], (0.0, {'x': 1}, 0.25)),
        WyckoffEntry('h', 8, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('i', 8, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('j', 8, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.0)),
        WyckoffEntry('k', 16, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[73] = [
        WyckoffEntry('a', 8, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 8, [], (0.25, 0.25, 0.25)),
        WyckoffEntry('c', 8, ['x'], ({'x': 1}, 0.0, 0.25)),
        WyckoffEntry('d', 8, ['x'], (0.25, {'x': 1}, 0.0)),
        WyckoffEntry('e', 8, ['x'], (0.0, 0.25, {'x': 1})),
        WyckoffEntry('f', 16, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[74] = [
        WyckoffEntry('a', 4, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 4, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 4, [], (0.25, 0.25, 0.25)),
        WyckoffEntry('d', 4, [], (0.25, 0.25, 0.75)),
        WyckoffEntry('e', 4, ['x'], (0.0, 0.25, {'x': 1})),
        WyckoffEntry('f', 8, ['x'], ({'x': 1}, 0.0, 0.0)),
        WyckoffEntry('g', 8, ['x'], (0.25, {'x': 1}, 0.25)),
        WyckoffEntry('h', 8, ['x', 'y'], (0.0, {'x': 1}, {'y': 1})),
        WyckoffEntry('i', 8, ['x', 'y'], ({'x': 1}, 0.25, {'y': 1})),
        WyckoffEntry('j', 16, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    return db
