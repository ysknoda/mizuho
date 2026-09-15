"""Wyckoff position database: hexagonal space groups (SG 168-194).

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
    """Return the Wyckoff entries for SG 168-194."""
    db: Dict[int, List[WyckoffEntry]] = {}

    db[168] = [
        WyckoffEntry('a', 1, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 2, ['x'], (0.6666666666666666, 0.3333333333333333, {'x': 1})),
        WyckoffEntry('c', 3, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('d', 6, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[169] = [
        WyckoffEntry('a', 6, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[170] = [
        WyckoffEntry('a', 6, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[171] = [
        WyckoffEntry('a', 3, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 3, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('c', 6, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[172] = [
        WyckoffEntry('a', 3, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 3, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('c', 6, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[173] = [
        WyckoffEntry('a', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 2, ['x'], (0.6666666666666666, 0.3333333333333333, {'x': 1})),
        WyckoffEntry('c', 6, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[174] = [
        WyckoffEntry('a', 1, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 1, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 1, [], (0.3333333333333333, 0.6666666666666666, 0.0)),
        WyckoffEntry('d', 1, [], (0.3333333333333333, 0.6666666666666666, 0.5)),
        WyckoffEntry('e', 1, [], (0.6666666666666666, 0.3333333333333333, 0.0)),
        WyckoffEntry('f', 1, [], (0.6666666666666666, 0.3333333333333333, 0.5)),
        WyckoffEntry('g', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('h', 2, ['x'], (0.3333333333333333, 0.6666666666666666, {'x': 1})),
        WyckoffEntry('i', 2, ['x'], (0.6666666666666666, 0.3333333333333333, {'x': 1})),
        WyckoffEntry('j', 3, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.0)),
        WyckoffEntry('k', 3, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.5)),
        WyckoffEntry('l', 6, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[175] = [
        WyckoffEntry('a', 1, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 1, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 2, [], (0.3333333333333333, 0.6666666666666666, 0.0)),
        WyckoffEntry('d', 2, [], (0.6666666666666666, 0.3333333333333333, 0.5)),
        WyckoffEntry('e', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 3, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('g', 3, [], (0.0, 0.5, 0.5)),
        WyckoffEntry('h', 4, ['x'], (0.6666666666666666, 0.3333333333333333, {'x': 1})),
        WyckoffEntry('i', 6, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('j', 6, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.0)),
        WyckoffEntry('k', 6, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.5)),
        WyckoffEntry('l', 12, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[176] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.25)),
        WyckoffEntry('b', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('c', 2, [], (0.6666666666666666, 0.3333333333333333, 0.75)),
        WyckoffEntry('d', 2, [], (0.6666666666666666, 0.3333333333333333, 0.25)),
        WyckoffEntry('e', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 4, ['x'], (0.6666666666666666, 0.3333333333333333, {'x': 1})),
        WyckoffEntry('g', 6, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('h', 6, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.25)),
        WyckoffEntry('i', 12, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[177] = [
        WyckoffEntry('a', 1, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 1, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 2, [], (0.6666666666666666, 0.3333333333333333, 0.0)),
        WyckoffEntry('d', 2, [], (0.6666666666666666, 0.3333333333333333, 0.5)),
        WyckoffEntry('e', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 3, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('g', 3, [], (0.0, 0.5, 0.5)),
        WyckoffEntry('h', 4, ['x'], (0.6666666666666666, 0.3333333333333333, {'x': 1})),
        WyckoffEntry('i', 6, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('j', 6, ['x'], (0.0, {'x': 1}, 0.0)),
        WyckoffEntry('k', 6, ['x'], (0.0, {'x': 1}, 0.5)),
        WyckoffEntry('l', 6, ['x'], ({'x': -1}, {'x': 1}, 0.0)),
        WyckoffEntry('m', 6, ['x'], ({'x': -1}, {'x': 1}, 0.5)),
        WyckoffEntry('n', 12, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[178] = [
        WyckoffEntry('a', 6, ['x'], (0.0, {'x': 1}, 0.3333333333333333)),
        WyckoffEntry('b', 6, ['x'], ({'x': -1}, {'x': 1}, 0.4166666666666667)),
        WyckoffEntry('c', 12, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[179] = [
        WyckoffEntry('a', 6, ['x'], (0.0, {'x': 1}, 0.16666666666666666)),
        WyckoffEntry('b', 6, ['x'], ({'x': -1}, {'x': 1}, 0.08333333333333333)),
        WyckoffEntry('c', 12, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[180] = [
        WyckoffEntry('a', 3, [], (0.0, 0.0, 0.3333333333333333)),
        WyckoffEntry('b', 3, [], (0.0, 0.0, 0.8333333333333334)),
        WyckoffEntry('c', 3, [], (0.0, 0.5, 0.6666666666666666)),
        WyckoffEntry('d', 3, [], (0.0, 0.5, 0.16666666666666666)),
        WyckoffEntry('e', 6, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 6, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('g', 6, ['x'], (0.0, {'x': 1}, 0.6666666666666666)),
        WyckoffEntry('h', 6, ['x'], (0.0, {'x': 1}, 0.16666666666666666)),
        WyckoffEntry('i', 6, ['x'], ({'x': -1}, {'x': 1}, 0.3333333333333333)),
        WyckoffEntry('j', 6, ['x'], ({'x': -1}, {'x': 1}, 0.8333333333333334)),
        WyckoffEntry('k', 12, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[181] = [
        WyckoffEntry('a', 3, [], (0.0, 0.0, 0.6666666666666666)),
        WyckoffEntry('b', 3, [], (0.0, 0.0, 0.16666666666666666)),
        WyckoffEntry('c', 3, [], (0.0, 0.5, 0.3333333333333333)),
        WyckoffEntry('d', 3, [], (0.0, 0.5, 0.8333333333333334)),
        WyckoffEntry('e', 6, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 6, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('g', 6, ['x'], (0.0, {'x': 1}, 0.3333333333333333)),
        WyckoffEntry('h', 6, ['x'], (0.0, {'x': 1}, 0.8333333333333334)),
        WyckoffEntry('i', 6, ['x'], ({'x': -1}, {'x': 1}, 0.6666666666666666)),
        WyckoffEntry('j', 6, ['x'], ({'x': -1}, {'x': 1}, 0.16666666666666666)),
        WyckoffEntry('k', 12, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[182] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.0, 0.0, 0.25)),
        WyckoffEntry('c', 2, [], (0.6666666666666666, 0.3333333333333333, 0.75)),
        WyckoffEntry('d', 2, [], (0.6666666666666666, 0.3333333333333333, 0.25)),
        WyckoffEntry('e', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 4, ['x'], (0.6666666666666666, 0.3333333333333333, {'x': 1})),
        WyckoffEntry('g', 6, ['x'], (0.0, {'x': 1}, 0.0)),
        WyckoffEntry('h', 6, ['x'], ({'x': -1}, {'x': 1}, 0.25)),
        WyckoffEntry('i', 12, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[183] = [
        WyckoffEntry('a', 1, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 2, ['x'], (0.6666666666666666, 0.3333333333333333, {'x': 1})),
        WyckoffEntry('c', 3, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('d', 6, ['x', 'y'], ({'x': 1}, {'x': 1}, {'y': 1})),
        WyckoffEntry('e', 6, ['x', 'y'], ({'x': 2}, {'x': 1}, {'y': 1})),
        WyckoffEntry('f', 12, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[184] = [
        WyckoffEntry('a', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 4, ['x'], (0.6666666666666666, 0.3333333333333333, {'x': 1})),
        WyckoffEntry('c', 6, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('d', 12, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[185] = [
        WyckoffEntry('a', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 4, ['x'], (0.6666666666666666, 0.3333333333333333, {'x': 1})),
        WyckoffEntry('c', 6, ['x', 'y'], ({'x': 1}, {'x': 1}, {'y': 1})),
        WyckoffEntry('d', 12, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[186] = [
        WyckoffEntry('a', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 2, ['x'], (0.6666666666666666, 0.3333333333333333, {'x': 1})),
        WyckoffEntry('c', 6, ['x', 'y'], ({'x': 2}, {'x': 1}, {'y': 1})),
        WyckoffEntry('d', 12, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[187] = [
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
        WyckoffEntry('l', 6, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.0)),
        WyckoffEntry('m', 6, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.5)),
        WyckoffEntry('n', 6, ['x', 'y'], ({'x': 2}, {'x': 1}, {'y': 1})),
        WyckoffEntry('o', 12, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[188] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.0, 0.0, 0.25)),
        WyckoffEntry('c', 2, [], (0.3333333333333333, 0.6666666666666666, 0.0)),
        WyckoffEntry('d', 2, [], (0.3333333333333333, 0.6666666666666666, 0.25)),
        WyckoffEntry('e', 2, [], (0.6666666666666666, 0.3333333333333333, 0.0)),
        WyckoffEntry('f', 2, [], (0.6666666666666666, 0.3333333333333333, 0.75)),
        WyckoffEntry('g', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('h', 4, ['x'], (0.3333333333333333, 0.6666666666666666, {'x': 1})),
        WyckoffEntry('i', 4, ['x'], (0.6666666666666666, 0.3333333333333333, {'x': 1})),
        WyckoffEntry('j', 6, ['x'], ({'x': -1}, {'x': 1}, 0.0)),
        WyckoffEntry('k', 6, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.25)),
        WyckoffEntry('l', 12, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[189] = [
        WyckoffEntry('a', 1, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 1, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 2, [], (0.3333333333333333, 0.6666666666666666, 0.0)),
        WyckoffEntry('d', 2, [], (0.6666666666666666, 0.3333333333333333, 0.5)),
        WyckoffEntry('e', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 3, ['x'], (0.0, {'x': 1}, 0.0)),
        WyckoffEntry('g', 3, ['x'], (0.0, {'x': 1}, 0.5)),
        WyckoffEntry('h', 4, ['x'], (0.6666666666666666, 0.3333333333333333, {'x': 1})),
        WyckoffEntry('i', 6, ['x', 'y'], ({'x': 1}, {'x': 1}, {'y': 1})),
        WyckoffEntry('j', 6, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.0)),
        WyckoffEntry('k', 6, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.5)),
        WyckoffEntry('l', 12, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[190] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.0, 0.0, 0.25)),
        WyckoffEntry('c', 2, [], (0.6666666666666666, 0.3333333333333333, 0.75)),
        WyckoffEntry('d', 2, [], (0.6666666666666666, 0.3333333333333333, 0.25)),
        WyckoffEntry('e', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 4, ['x'], (0.6666666666666666, 0.3333333333333333, {'x': 1})),
        WyckoffEntry('g', 6, ['x'], (0.0, {'x': 1}, 0.0)),
        WyckoffEntry('h', 6, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.25)),
        WyckoffEntry('i', 12, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[191] = [
        WyckoffEntry('a', 1, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 1, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 2, [], (0.3333333333333333, 0.6666666666666666, 0.0)),
        WyckoffEntry('d', 2, [], (0.6666666666666666, 0.3333333333333333, 0.5)),
        WyckoffEntry('e', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 3, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('g', 3, [], (0.0, 0.5, 0.5)),
        WyckoffEntry('h', 4, ['x'], (0.6666666666666666, 0.3333333333333333, {'x': 1})),
        WyckoffEntry('i', 6, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('j', 6, ['x'], (0.0, {'x': 1}, 0.0)),
        WyckoffEntry('k', 6, ['x'], (0.0, {'x': 1}, 0.5)),
        WyckoffEntry('l', 6, ['x'], ({'x': -1}, {'x': 1}, 0.0)),
        WyckoffEntry('m', 6, ['x'], ({'x': -1}, {'x': 1}, 0.5)),
        WyckoffEntry('n', 12, ['x', 'y'], ({'x': 1}, {'x': 1}, {'y': 1})),
        WyckoffEntry('o', 12, ['x', 'y'], ({'x': 2}, {'x': 1}, {'y': 1})),
        WyckoffEntry('p', 12, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.0)),
        WyckoffEntry('q', 12, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.5)),
        WyckoffEntry('r', 24, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[192] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.25)),
        WyckoffEntry('b', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('c', 4, [], (0.6666666666666666, 0.3333333333333333, 0.25)),
        WyckoffEntry('d', 4, [], (0.6666666666666666, 0.3333333333333333, 0.5)),
        WyckoffEntry('e', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 6, [], (0.0, 0.5, 0.25)),
        WyckoffEntry('g', 6, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('h', 8, ['x'], (0.6666666666666666, 0.3333333333333333, {'x': 1})),
        WyckoffEntry('i', 12, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('j', 12, ['x'], (0.0, {'x': 1}, 0.25)),
        WyckoffEntry('k', 12, ['x'], ({'x': -1}, {'x': 1}, 0.25)),
        WyckoffEntry('l', 12, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.0)),
        WyckoffEntry('m', 24, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[193] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.25)),
        WyckoffEntry('b', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('c', 4, [], (0.6666666666666666, 0.3333333333333333, 0.75)),
        WyckoffEntry('d', 4, [], (0.6666666666666666, 0.3333333333333333, 0.0)),
        WyckoffEntry('e', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 6, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('g', 6, ['x'], (0.0, {'x': 1}, 0.25)),
        WyckoffEntry('h', 8, ['x'], (0.6666666666666666, 0.3333333333333333, {'x': 1})),
        WyckoffEntry('i', 12, ['x'], ({'x': -1}, {'x': 1}, 0.0)),
        WyckoffEntry('j', 12, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.25)),
        WyckoffEntry('k', 12, ['x', 'y'], ({'x': 1}, {'x': 1}, {'y': 1})),
        WyckoffEntry('l', 24, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[194] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.0, 0.0, 0.25)),
        WyckoffEntry('c', 2, [], (0.6666666666666666, 0.3333333333333333, 0.75)),
        WyckoffEntry('d', 2, [], (0.6666666666666666, 0.3333333333333333, 0.25)),
        WyckoffEntry('e', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 4, ['x'], (0.6666666666666666, 0.3333333333333333, {'x': 1})),
        WyckoffEntry('g', 6, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('h', 6, ['x'], ({'x': -1}, {'x': 1}, 0.25)),
        WyckoffEntry('i', 12, ['x'], (0.0, {'x': 1}, 0.0)),
        WyckoffEntry('j', 12, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.25)),
        WyckoffEntry('k', 12, ['x', 'y'], ({'x': 2}, {'x': 1}, {'y': 1})),
        WyckoffEntry('l', 24, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    return db
