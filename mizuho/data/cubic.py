"""Wyckoff position database: cubic space groups (SG 195-230).

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
    """Return the Wyckoff entries for SG 195-230."""
    db: Dict[int, List[WyckoffEntry]] = {}

    db[195] = [
        WyckoffEntry('a', 1, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 1, [], (0.5, 0.5, 0.5)),
        WyckoffEntry('c', 3, [], (0.0, 0.5, 0.5)),
        WyckoffEntry('d', 3, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('e', 4, ['x'], ({'x': 1}, {'x': 1}, {'x': 1})),
        WyckoffEntry('f', 6, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('g', 6, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('h', 6, ['x'], (0.5, 0.0, {'x': 1})),
        WyckoffEntry('i', 6, ['x'], (0.5, 0.5, {'x': 1})),
        WyckoffEntry('j', 12, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[196] = [
        WyckoffEntry('a', 4, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 4, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 4, [], (0.25, 0.25, 0.25)),
        WyckoffEntry('d', 4, [], (0.75, 0.75, 0.75)),
        WyckoffEntry('e', 16, ['x'], ({'x': 1}, {'x': 1}, {'x': 1})),
        WyckoffEntry('f', 24, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('g', 24, ['x'], ({'x': 1}, 0.25, 0.25)),
        WyckoffEntry('h', 48, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[197] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 6, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 8, ['x'], ({'x': 1}, {'x': 1}, {'x': 1})),
        WyckoffEntry('d', 12, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('e', 12, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('f', 24, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[198] = [
        WyckoffEntry('a', 4, ['x'], ({'x': 1}, {'x': 1}, {'x': 1})),
        WyckoffEntry('b', 12, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[199] = [
        WyckoffEntry('a', 8, ['x'], ({'x': 1}, {'x': 1}, {'x': 1})),
        WyckoffEntry('b', 12, ['x'], (0.0, 0.25, {'x': 1})),
        WyckoffEntry('c', 24, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[200] = [
        WyckoffEntry('a', 1, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 1, [], (0.5, 0.5, 0.5)),
        WyckoffEntry('c', 3, [], (0.0, 0.5, 0.5)),
        WyckoffEntry('d', 3, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('e', 6, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 6, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('g', 6, ['x'], (0.5, 0.0, {'x': 1})),
        WyckoffEntry('h', 6, ['x'], (0.5, 0.5, {'x': 1})),
        WyckoffEntry('i', 8, ['x'], ({'x': 1}, {'x': 1}, {'x': 1})),
        WyckoffEntry('j', 12, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.0)),
        WyckoffEntry('k', 12, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.5)),
        WyckoffEntry('l', 24, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[201] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 4, [], (0.25, 0.25, 0.25)),
        WyckoffEntry('c', 4, [], (0.25, 0.25, 0.75)),
        WyckoffEntry('d', 6, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('e', 8, ['x'], ({'x': 1}, {'x': 1}, {'x': 1})),
        WyckoffEntry('f', 12, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('g', 12, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('h', 24, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[202] = [
        WyckoffEntry('a', 4, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 4, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 8, [], (0.25, 0.25, 0.25)),
        WyckoffEntry('d', 24, [], (0.0, 0.25, 0.25)),
        WyckoffEntry('e', 24, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 32, ['x'], ({'x': 1}, {'x': 1}, {'x': 1})),
        WyckoffEntry('g', 48, ['x'], ({'x': 1}, 0.25, 0.25)),
        WyckoffEntry('h', 48, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.0)),
        WyckoffEntry('i', 96, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[203] = [
        WyckoffEntry('a', 8, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 8, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 16, [], (0.125, 0.125, 0.125)),
        WyckoffEntry('d', 16, [], (0.125, 0.125, 0.625)),
        WyckoffEntry('e', 32, ['x'], ({'x': 1}, {'x': 1}, {'x': 1})),
        WyckoffEntry('f', 48, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('g', 96, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[204] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 6, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 8, [], (0.25, 0.25, 0.25)),
        WyckoffEntry('d', 12, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('e', 12, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('f', 16, ['x'], ({'x': 1}, {'x': 1}, {'x': 1})),
        WyckoffEntry('g', 24, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.0)),
        WyckoffEntry('h', 48, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[205] = [
        WyckoffEntry('a', 4, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 4, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 8, ['x'], ({'x': 1}, {'x': 1}, {'x': 1})),
        WyckoffEntry('d', 24, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[206] = [
        WyckoffEntry('a', 8, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 8, [], (0.75, 0.75, 0.25)),
        WyckoffEntry('c', 16, ['x'], ({'x': 1}, {'x': 1}, {'x': 1})),
        WyckoffEntry('d', 24, ['x'], (0.0, 0.25, {'x': 1})),
        WyckoffEntry('e', 48, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[207] = [
        WyckoffEntry('a', 1, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 1, [], (0.5, 0.5, 0.5)),
        WyckoffEntry('c', 3, [], (0.5, 0.5, 0.0)),
        WyckoffEntry('d', 3, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('e', 6, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 6, ['x'], (0.5, 0.5, {'x': 1})),
        WyckoffEntry('g', 8, ['x'], ({'x': 1}, {'x': 1}, {'x': 1})),
        WyckoffEntry('h', 12, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('i', 12, ['x'], ({'x': -1}, {'x': 1}, 0.0)),
        WyckoffEntry('j', 12, ['x'], ({'x': -1}, {'x': 1}, 0.5)),
        WyckoffEntry('k', 24, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[208] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 4, [], (0.25, 0.25, 0.25)),
        WyckoffEntry('c', 4, [], (0.75, 0.75, 0.75)),
        WyckoffEntry('d', 6, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('e', 6, [], (0.0, 0.5, 0.25)),
        WyckoffEntry('f', 6, [], (0.5, 0.0, 0.25)),
        WyckoffEntry('g', 8, ['x'], ({'x': 1}, {'x': 1}, {'x': 1})),
        WyckoffEntry('h', 12, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('i', 12, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('j', 12, ['x'], (0.5, 0.0, {'x': 1})),
        WyckoffEntry('k', 12, ['x'], (({'x': -1}, 0.5), {'x': 1}, 0.25)),
        WyckoffEntry('l', 12, ['x'], (({'x': -1}, 0.5), {'x': 1}, 0.75)),
        WyckoffEntry('m', 24, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[209] = [
        WyckoffEntry('a', 4, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 4, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 8, [], (0.25, 0.25, 0.25)),
        WyckoffEntry('d', 24, [], (0.25, 0.75, 0.0)),
        WyckoffEntry('e', 24, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 32, ['x'], ({'x': 1}, {'x': 1}, {'x': 1})),
        WyckoffEntry('g', 48, ['x'], ({'x': -1}, {'x': 1}, 0.0)),
        WyckoffEntry('h', 48, ['x'], ({'x': -1}, {'x': 1}, 0.5)),
        WyckoffEntry('i', 48, ['x'], ({'x': 1}, 0.25, 0.25)),
        WyckoffEntry('j', 96, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[210] = [
        WyckoffEntry('a', 8, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 8, [], (0.5, 0.0, 0.0)),
        WyckoffEntry('c', 16, [], (0.125, 0.125, 0.125)),
        WyckoffEntry('d', 16, [], (0.625, 0.625, 0.625)),
        WyckoffEntry('e', 32, ['x'], ({'x': 1}, {'x': 1}, {'x': 1})),
        WyckoffEntry('f', 48, ['x'], ({'x': 1}, 0.0, 0.0)),
        WyckoffEntry('g', 48, ['x'], (({'x': -1}, 0.75), 0.125, {'x': 1})),
        WyckoffEntry('h', 96, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[211] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 6, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 8, [], (0.25, 0.25, 0.25)),
        WyckoffEntry('d', 12, [], (0.0, 0.5, 0.25)),
        WyckoffEntry('e', 12, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 16, ['x'], ({'x': 1}, {'x': 1}, {'x': 1})),
        WyckoffEntry('g', 24, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('h', 24, ['x'], ({'x': -1}, {'x': 1}, 0.0)),
        WyckoffEntry('i', 24, ['x'], (({'x': -1}, 0.5), {'x': 1}, 0.25)),
        WyckoffEntry('j', 48, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[212] = [
        WyckoffEntry('a', 4, [], (0.125, 0.125, 0.125)),
        WyckoffEntry('b', 4, [], (0.875, 0.375, 0.125)),
        WyckoffEntry('c', 8, ['x'], ({'x': 1}, {'x': 1}, {'x': 1})),
        WyckoffEntry('d', 12, ['x'], (({'x': -1}, 0.25), {'x': 1}, 0.125)),
        WyckoffEntry('e', 24, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[213] = [
        WyckoffEntry('a', 4, [], (0.375, 0.375, 0.375)),
        WyckoffEntry('b', 4, [], (0.625, 0.125, 0.375)),
        WyckoffEntry('c', 8, ['x'], ({'x': 1}, {'x': 1}, {'x': 1})),
        WyckoffEntry('d', 12, ['x'], (({'x': -1}, 0.75), {'x': 1}, 0.375)),
        WyckoffEntry('e', 24, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[214] = [
        WyckoffEntry('a', 8, [], (0.125, 0.125, 0.125)),
        WyckoffEntry('b', 8, [], (0.375, 0.375, 0.375)),
        WyckoffEntry('c', 12, [], (0.125, 0.0, 0.25)),
        WyckoffEntry('d', 12, [], (0.625, 0.0, 0.25)),
        WyckoffEntry('e', 16, ['x'], ({'x': 1}, {'x': 1}, {'x': 1})),
        WyckoffEntry('f', 24, ['x'], ({'x': 1}, 0.0, 0.25)),
        WyckoffEntry('g', 24, ['x'], (({'x': 1}, 0.25), 0.125, {'x': 1})),
        WyckoffEntry('h', 24, ['x'], (({'x': -1}, 0.25), {'x': 1}, 0.125)),
        WyckoffEntry('i', 48, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[215] = [
        WyckoffEntry('a', 1, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 1, [], (0.5, 0.5, 0.5)),
        WyckoffEntry('c', 3, [], (0.5, 0.5, 0.0)),
        WyckoffEntry('d', 3, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('e', 4, ['x'], ({'x': 1}, {'x': 1}, {'x': 1})),
        WyckoffEntry('f', 6, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('g', 6, ['x'], (0.5, 0.5, {'x': 1})),
        WyckoffEntry('h', 12, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('i', 12, ['x', 'y'], ({'x': 1}, {'x': 1}, {'y': 1})),
        WyckoffEntry('j', 24, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[216] = [
        WyckoffEntry('a', 4, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 4, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 4, [], (0.25, 0.25, 0.25)),
        WyckoffEntry('d', 4, [], (0.25, 0.25, 0.75)),
        WyckoffEntry('e', 16, ['x'], ({'x': 1}, {'x': 1}, {'x': 1})),
        WyckoffEntry('f', 24, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('g', 24, ['x'], ({'x': 1}, 0.25, 0.25)),
        WyckoffEntry('h', 48, ['x', 'y'], ({'x': 1}, {'x': 1}, {'y': 1})),
        WyckoffEntry('i', 96, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[217] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 6, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 8, ['x'], ({'x': 1}, {'x': 1}, {'x': 1})),
        WyckoffEntry('d', 12, [], (0.5, 0.0, 0.25)),
        WyckoffEntry('e', 12, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 24, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('g', 24, ['x', 'y'], ({'x': 1}, {'x': 1}, {'y': 1})),
        WyckoffEntry('h', 48, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[218] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 6, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 6, [], (0.5, 0.0, 0.25)),
        WyckoffEntry('d', 6, [], (0.0, 0.5, 0.25)),
        WyckoffEntry('e', 8, ['x'], ({'x': 1}, {'x': 1}, {'x': 1})),
        WyckoffEntry('f', 12, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('g', 12, ['x'], (0.5, 0.0, {'x': 1})),
        WyckoffEntry('h', 12, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('i', 24, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[219] = [
        WyckoffEntry('a', 8, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 8, [], (0.25, 0.25, 0.25)),
        WyckoffEntry('c', 24, [], (0.25, 0.75, 0.0)),
        WyckoffEntry('d', 24, [], (0.25, 0.0, 0.0)),
        WyckoffEntry('e', 32, ['x'], ({'x': 1}, {'x': 1}, {'x': 1})),
        WyckoffEntry('f', 48, ['x'], ({'x': 1}, 0.0, 0.0)),
        WyckoffEntry('g', 48, ['x'], (0.25, 0.25, {'x': 1})),
        WyckoffEntry('h', 96, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[220] = [
        WyckoffEntry('a', 12, [], (0.5, 0.25, 0.625)),
        WyckoffEntry('b', 12, [], (0.5, 0.25, 0.125)),
        WyckoffEntry('c', 16, ['x'], ({'x': 1}, {'x': 1}, {'x': 1})),
        WyckoffEntry('d', 24, ['x'], (0.0, 0.25, {'x': 1})),
        WyckoffEntry('e', 48, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[221] = [
        WyckoffEntry('a', 1, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 1, [], (0.5, 0.5, 0.5)),
        WyckoffEntry('c', 3, [], (0.0, 0.5, 0.5)),
        WyckoffEntry('d', 3, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('e', 6, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 6, ['x'], (0.5, 0.5, {'x': 1})),
        WyckoffEntry('g', 8, ['x'], ({'x': 1}, {'x': 1}, {'x': 1})),
        WyckoffEntry('h', 12, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('i', 12, ['x'], ({'x': -1}, {'x': 1}, 0.0)),
        WyckoffEntry('j', 12, ['x'], ({'x': -1}, {'x': 1}, 0.5)),
        WyckoffEntry('k', 24, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.0)),
        WyckoffEntry('l', 24, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.5)),
        WyckoffEntry('m', 24, ['x', 'y'], ({'x': 1}, {'x': 1}, {'y': 1})),
        WyckoffEntry('n', 48, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[222] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 6, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 8, [], (0.25, 0.25, 0.25)),
        WyckoffEntry('d', 12, [], (0.5, 0.0, 0.25)),
        WyckoffEntry('e', 12, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 16, ['x'], ({'x': 1}, {'x': 1}, {'x': 1})),
        WyckoffEntry('g', 24, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('h', 24, ['x'], ({'x': -1}, {'x': 1}, 0.0)),
        WyckoffEntry('i', 48, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[223] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 6, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 6, [], (0.0, 0.5, 0.25)),
        WyckoffEntry('d', 6, [], (0.5, 0.0, 0.25)),
        WyckoffEntry('e', 8, [], (0.25, 0.25, 0.25)),
        WyckoffEntry('f', 12, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('g', 12, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('h', 12, ['x'], (0.5, 0.0, {'x': 1})),
        WyckoffEntry('i', 16, ['x'], ({'x': 1}, {'x': 1}, {'x': 1})),
        WyckoffEntry('j', 24, ['x'], (({'x': -1}, 0.5), {'x': 1}, 0.25)),
        WyckoffEntry('k', 24, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.0)),
        WyckoffEntry('l', 48, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[224] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 4, [], (0.25, 0.25, 0.25)),
        WyckoffEntry('c', 4, [], (0.25, 0.25, 0.75)),
        WyckoffEntry('d', 6, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('e', 8, ['x'], ({'x': 1}, {'x': 1}, {'x': 1})),
        WyckoffEntry('f', 12, [], (0.0, 0.5, 0.25)),
        WyckoffEntry('g', 12, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('h', 24, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('i', 24, ['x'], (({'x': -1}, 0.5), {'x': 1}, 0.25)),
        WyckoffEntry('j', 24, ['x'], (({'x': -1}, 0.5), {'x': 1}, 0.75)),
        WyckoffEntry('k', 24, ['x', 'y'], ({'x': 1}, {'x': 1}, {'y': 1})),
        WyckoffEntry('l', 48, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[225] = [
        WyckoffEntry('a', 4, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 4, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 8, [], (0.25, 0.25, 0.25)),
        WyckoffEntry('d', 24, [], (0.0, 0.25, 0.25)),
        WyckoffEntry('e', 24, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 32, ['x'], ({'x': 1}, {'x': 1}, {'x': 1})),
        WyckoffEntry('g', 48, ['x'], ({'x': 1}, 0.25, 0.25)),
        WyckoffEntry('h', 48, ['x'], ({'x': -1}, {'x': 1}, 0.0)),
        WyckoffEntry('i', 48, ['x'], ({'x': -1}, {'x': 1}, 0.5)),
        WyckoffEntry('j', 96, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.0)),
        WyckoffEntry('k', 96, ['x', 'y'], ({'x': 1}, {'x': 1}, {'y': 1})),
        WyckoffEntry('l', 192, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[226] = [
        WyckoffEntry('a', 8, [], (0.25, 0.25, 0.25)),
        WyckoffEntry('b', 8, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('c', 24, [], (0.25, 0.0, 0.0)),
        WyckoffEntry('d', 24, [], (0.25, 0.75, 0.0)),
        WyckoffEntry('e', 48, ['x'], ({'x': 1}, 0.0, 0.0)),
        WyckoffEntry('f', 48, ['x'], (0.25, 0.25, {'x': 1})),
        WyckoffEntry('g', 64, ['x'], ({'x': 1}, {'x': 1}, {'x': 1})),
        WyckoffEntry('h', 96, ['x'], ({'x': 1}, 0.25, {'x': 1})),
        WyckoffEntry('i', 96, ['x', 'y'], (0.0, {'x': 1}, {'y': 1})),
        WyckoffEntry('j', 192, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[227] = [
        WyckoffEntry('a', 8, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 8, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 16, [], (0.125, 0.125, 0.125)),
        WyckoffEntry('d', 16, [], (0.125, 0.125, 0.625)),
        WyckoffEntry('e', 32, ['x'], ({'x': 1}, {'x': 1}, {'x': 1})),
        WyckoffEntry('f', 48, ['x'], ({'x': 1}, 0.0, 0.0)),
        WyckoffEntry('g', 96, ['x', 'y'], (({'x': -1}, 0.5), {'x': 1}, {'y': 1})),
        WyckoffEntry('h', 96, ['x'], (({'x': -1}, 0.75), 0.125, {'x': 1})),
        WyckoffEntry('i', 192, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[228] = [
        WyckoffEntry('a', 16, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 32, [], (0.125, 0.125, 0.125)),
        WyckoffEntry('c', 32, [], (0.375, 0.125, 0.125)),
        WyckoffEntry('d', 48, [], (0.25, 0.75, 0.0)),
        WyckoffEntry('e', 64, ['x'], ({'x': 1}, {'x': 1}, {'x': 1})),
        WyckoffEntry('f', 96, ['x'], ({'x': 1}, 0.0, 0.0)),
        WyckoffEntry('g', 96, ['x'], (({'x': -1}, 0.75), 0.125, {'x': 1})),
        WyckoffEntry('h', 192, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[229] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 6, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 8, [], (0.25, 0.25, 0.25)),
        WyckoffEntry('d', 12, [], (0.5, 0.0, 0.25)),
        WyckoffEntry('e', 12, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 16, ['x'], ({'x': 1}, {'x': 1}, {'x': 1})),
        WyckoffEntry('g', 24, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('h', 24, ['x'], ({'x': -1}, {'x': 1}, 0.0)),
        WyckoffEntry('i', 48, ['x'], (({'x': -1}, 0.5), {'x': 1}, 0.25)),
        WyckoffEntry('j', 48, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.0)),
        WyckoffEntry('k', 48, ['x', 'y'], ({'x': 1}, {'x': 1}, {'y': 1})),
        WyckoffEntry('l', 96, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[230] = [
        WyckoffEntry('a', 16, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 16, [], (0.125, 0.125, 0.125)),
        WyckoffEntry('c', 24, [], (0.125, 0.0, 0.25)),
        WyckoffEntry('d', 24, [], (0.5, 0.25, 0.125)),
        WyckoffEntry('e', 32, ['x'], ({'x': 1}, {'x': 1}, {'x': 1})),
        WyckoffEntry('f', 48, ['x'], ({'x': 1}, 0.0, 0.25)),
        WyckoffEntry('g', 48, ['x'], (({'x': -1}, 0.25), {'x': 1}, 0.125)),
        WyckoffEntry('h', 96, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    return db
