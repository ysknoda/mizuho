"""Wyckoff position database: trigonal space groups (SG 143-167).

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
    """Return the Wyckoff entries for SG 143-167."""
    db: Dict[int, List[WyckoffEntry]] = {}

    db[143] = [
        WyckoffEntry('a', 1, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 1, ['x'], (0.3333333333333333, 0.6666666666666666, {'x': 1})),
        WyckoffEntry('c', 1, ['x'], (0.6666666666666666, 0.3333333333333333, {'x': 1})),
        WyckoffEntry('d', 3, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[144] = [
        WyckoffEntry('a', 3, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[145] = [
        WyckoffEntry('a', 3, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[146] = [
        WyckoffEntry('a', 3, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 9, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[147] = [
        WyckoffEntry('a', 1, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 1, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('d', 2, ['x'], (0.6666666666666666, 0.3333333333333333, {'x': 1})),
        WyckoffEntry('e', 3, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('f', 3, [], (0.0, 0.5, 0.5)),
        WyckoffEntry('g', 6, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[148] = [
        WyckoffEntry('a', 3, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 3, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 6, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('d', 9, [], (0.0, 0.5, 0.5)),
        WyckoffEntry('e', 9, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('f', 18, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[149] = [
        WyckoffEntry('a', 1, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 1, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 1, [], (0.3333333333333333, 0.6666666666666666, 0.0)),
        WyckoffEntry('d', 1, [], (0.3333333333333333, 0.6666666666666666, 0.5)),
        WyckoffEntry('e', 1, [], (0.6666666666666666, 0.3333333333333333, 0.0)),
        WyckoffEntry('f', 1, [], (0.6666666666666666, 0.3333333333333333, 0.5)),
        WyckoffEntry('g', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('h', 2, ['x'], (0.3333333333333333, 0.6666666666666666, {'x': 1})),
        WyckoffEntry('i', 2, ['x'], (0.6666666666666666, 0.3333333333333333, {'x': 1})),
        WyckoffEntry('j', 3, ['x'], ({'x': -1}, {'x': 1}, 0.0)),
        WyckoffEntry('k', 3, ['x'], ({'x': -1}, {'x': 1}, 0.5)),
        WyckoffEntry('l', 6, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[150] = [
        WyckoffEntry('a', 1, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 1, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('d', 2, ['x'], (0.6666666666666666, 0.3333333333333333, {'x': 1})),
        WyckoffEntry('e', 3, ['x'], ({'x': 1}, {'x': 1}, 0.0)),
        WyckoffEntry('f', 3, ['x'], ({'x': 1}, {'x': 1}, 0.5)),
        WyckoffEntry('g', 6, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[151] = [
        WyckoffEntry('a', 3, ['x'], ({'x': -1}, {'x': 1}, 0.3333333333333333)),
        WyckoffEntry('b', 3, ['x'], ({'x': -1}, {'x': 1}, 0.8333333333333334)),
        WyckoffEntry('c', 6, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[152] = [
        WyckoffEntry('a', 3, ['x'], ({'x': 1}, {'x': 1}, 0.0)),
        WyckoffEntry('b', 3, ['x'], ({'x': 1}, {'x': 1}, 0.5)),
        WyckoffEntry('c', 6, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[153] = [
        WyckoffEntry('a', 3, ['x'], ({'x': -1}, {'x': 1}, 0.6666666666666666)),
        WyckoffEntry('b', 3, ['x'], ({'x': -1}, {'x': 1}, 0.16666666666666666)),
        WyckoffEntry('c', 6, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[154] = [
        WyckoffEntry('a', 3, ['x'], ({'x': 1}, {'x': 1}, 0.0)),
        WyckoffEntry('b', 3, ['x'], ({'x': 1}, {'x': 1}, 0.5)),
        WyckoffEntry('c', 6, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[155] = [
        WyckoffEntry('a', 3, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 3, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 6, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('d', 9, ['x'], ({'x': 1}, {'x': 1}, 0.0)),
        WyckoffEntry('e', 9, ['x'], ({'x': 1}, {'x': 1}, 0.5)),
        WyckoffEntry('f', 18, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[156] = [
        WyckoffEntry('a', 1, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 1, ['x'], (0.3333333333333333, 0.6666666666666666, {'x': 1})),
        WyckoffEntry('c', 1, ['x'], (0.6666666666666666, 0.3333333333333333, {'x': 1})),
        WyckoffEntry('d', 3, ['x', 'y'], ({'x': -1}, {'x': 1}, {'y': 1})),
        WyckoffEntry('e', 6, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[157] = [
        WyckoffEntry('a', 1, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 2, ['x'], (0.6666666666666666, 0.3333333333333333, {'x': 1})),
        WyckoffEntry('c', 3, ['x', 'y'], ({'x': 1}, {'x': 1}, {'y': 1})),
        WyckoffEntry('d', 6, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[158] = [
        WyckoffEntry('a', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 2, ['x'], (0.3333333333333333, 0.6666666666666666, {'x': 1})),
        WyckoffEntry('c', 2, ['x'], (0.6666666666666666, 0.3333333333333333, {'x': 1})),
        WyckoffEntry('d', 6, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[159] = [
        WyckoffEntry('a', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 2, ['x'], (0.6666666666666666, 0.3333333333333333, {'x': 1})),
        WyckoffEntry('c', 6, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[160] = [
        WyckoffEntry('a', 3, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 9, ['x', 'y'], ({'x': -1}, {'x': 1}, {'y': 1})),
        WyckoffEntry('c', 18, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[161] = [
        WyckoffEntry('a', 6, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 18, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[162] = [
        WyckoffEntry('a', 1, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 1, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 2, [], (0.6666666666666666, 0.3333333333333333, 0.0)),
        WyckoffEntry('d', 2, [], (0.6666666666666666, 0.3333333333333333, 0.5)),
        WyckoffEntry('e', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 3, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('g', 3, [], (0.0, 0.5, 0.5)),
        WyckoffEntry('h', 4, ['x'], (0.6666666666666666, 0.3333333333333333, {'x': 1})),
        WyckoffEntry('i', 6, ['x'], ({'x': -1}, {'x': 1}, 0.0)),
        WyckoffEntry('j', 6, ['x'], ({'x': -1}, {'x': 1}, 0.5)),
        WyckoffEntry('k', 6, ['x', 'y'], ({'x': 1}, {'x': 1}, {'y': 1})),
        WyckoffEntry('l', 12, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[163] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.25)),
        WyckoffEntry('b', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('c', 2, [], (0.6666666666666666, 0.3333333333333333, 0.75)),
        WyckoffEntry('d', 2, [], (0.6666666666666666, 0.3333333333333333, 0.25)),
        WyckoffEntry('e', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 4, ['x'], (0.6666666666666666, 0.3333333333333333, {'x': 1})),
        WyckoffEntry('g', 6, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('h', 6, ['x'], ({'x': -1}, {'x': 1}, 0.25)),
        WyckoffEntry('i', 12, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[164] = [
        WyckoffEntry('a', 1, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 1, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('d', 2, ['x'], (0.6666666666666666, 0.3333333333333333, {'x': 1})),
        WyckoffEntry('e', 3, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('f', 3, [], (0.0, 0.5, 0.5)),
        WyckoffEntry('g', 6, ['x'], ({'x': 1}, {'x': 1}, 0.0)),
        WyckoffEntry('h', 6, ['x'], ({'x': 1}, {'x': 1}, 0.5)),
        WyckoffEntry('i', 6, ['x', 'y'], ({'x': -1}, {'x': 1}, {'y': 1})),
        WyckoffEntry('j', 12, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[165] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.25)),
        WyckoffEntry('b', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('c', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('d', 4, ['x'], (0.6666666666666666, 0.3333333333333333, {'x': 1})),
        WyckoffEntry('e', 6, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('f', 6, ['x'], ({'x': 1}, {'x': 1}, 0.25)),
        WyckoffEntry('g', 12, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[166] = [
        WyckoffEntry('a', 3, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 3, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 6, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('d', 9, [], (0.0, 0.5, 0.5)),
        WyckoffEntry('e', 9, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('f', 18, ['x'], ({'x': 1}, {'x': 1}, 0.0)),
        WyckoffEntry('g', 18, ['x'], ({'x': 1}, {'x': 1}, 0.5)),
        WyckoffEntry('h', 18, ['x', 'y'], ({'x': -1}, {'x': 1}, {'y': 1})),
        WyckoffEntry('i', 36, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[167] = [
        WyckoffEntry('a', 6, [], (0.0, 0.0, 0.25)),
        WyckoffEntry('b', 6, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('c', 12, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('d', 18, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('e', 18, ['x'], ({'x': 1}, {'x': 1}, 0.25)),
        WyckoffEntry('f', 36, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    return db
