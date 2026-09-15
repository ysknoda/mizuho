"""Wyckoff position database: tetragonal space groups (SG 75-142).

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
    """Return the Wyckoff entries for SG 75-142."""
    db: Dict[int, List[WyckoffEntry]] = {}

    db[75] = [
        WyckoffEntry('a', 1, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 1, ['x'], (0.5, 0.5, {'x': 1})),
        WyckoffEntry('c', 2, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('d', 4, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[76] = [
        WyckoffEntry('a', 4, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[77] = [
        WyckoffEntry('a', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 2, ['x'], (0.5, 0.5, {'x': 1})),
        WyckoffEntry('c', 2, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('d', 4, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[78] = [
        WyckoffEntry('a', 4, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[79] = [
        WyckoffEntry('a', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 4, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('c', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[80] = [
        WyckoffEntry('a', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[81] = [
        WyckoffEntry('a', 1, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 1, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 1, [], (0.5, 0.5, 0.0)),
        WyckoffEntry('d', 1, [], (0.5, 0.5, 0.5)),
        WyckoffEntry('e', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 2, ['x'], (0.5, 0.5, {'x': 1})),
        WyckoffEntry('g', 2, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('h', 4, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[82] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 2, [], (0.5, 0.0, 0.75)),
        WyckoffEntry('d', 2, [], (0.5, 0.0, 0.25)),
        WyckoffEntry('e', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 4, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('g', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[83] = [
        WyckoffEntry('a', 1, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 1, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 1, [], (0.5, 0.5, 0.0)),
        WyckoffEntry('d', 1, [], (0.5, 0.5, 0.5)),
        WyckoffEntry('e', 2, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('f', 2, [], (0.0, 0.5, 0.5)),
        WyckoffEntry('g', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('h', 2, ['x'], (0.5, 0.5, {'x': 1})),
        WyckoffEntry('i', 4, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('j', 4, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.0)),
        WyckoffEntry('k', 4, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.5)),
        WyckoffEntry('l', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[84] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.5, 0.5, 0.0)),
        WyckoffEntry('c', 2, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('d', 2, [], (0.0, 0.5, 0.5)),
        WyckoffEntry('e', 2, [], (0.0, 0.0, 0.25)),
        WyckoffEntry('f', 2, [], (0.5, 0.5, 0.25)),
        WyckoffEntry('g', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('h', 4, ['x'], (0.5, 0.5, {'x': 1})),
        WyckoffEntry('i', 4, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('j', 4, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.0)),
        WyckoffEntry('k', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[85] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 2, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('d', 4, [], (0.25, 0.25, 0.0)),
        WyckoffEntry('e', 4, [], (0.25, 0.25, 0.5)),
        WyckoffEntry('f', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('g', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[86] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 4, [], (0.25, 0.25, 0.25)),
        WyckoffEntry('d', 4, [], (0.25, 0.25, 0.75)),
        WyckoffEntry('e', 4, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('f', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('g', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[87] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 4, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('d', 4, [], (0.5, 0.0, 0.25)),
        WyckoffEntry('e', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 8, [], (0.25, 0.25, 0.25)),
        WyckoffEntry('g', 8, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('h', 8, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.0)),
        WyckoffEntry('i', 16, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[88] = [
        WyckoffEntry('a', 4, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 4, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 8, [], (0.0, 0.25, 0.125)),
        WyckoffEntry('d', 8, [], (0.0, 0.25, 0.625)),
        WyckoffEntry('e', 8, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 16, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[89] = [
        WyckoffEntry('a', 1, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 1, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 1, [], (0.5, 0.5, 0.0)),
        WyckoffEntry('d', 1, [], (0.5, 0.5, 0.5)),
        WyckoffEntry('e', 2, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('f', 2, [], (0.0, 0.5, 0.5)),
        WyckoffEntry('g', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('h', 2, ['x'], (0.5, 0.5, {'x': 1})),
        WyckoffEntry('i', 4, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('j', 4, ['x'], ({'x': -1}, {'x': 1}, 0.0)),
        WyckoffEntry('k', 4, ['x'], ({'x': -1}, {'x': 1}, 0.5)),
        WyckoffEntry('l', 4, ['x'], ({'x': 1}, 0.0, 0.0)),
        WyckoffEntry('m', 4, ['x'], ({'x': 1}, 0.5, 0.5)),
        WyckoffEntry('n', 4, ['x'], ({'x': 1}, 0.0, 0.5)),
        WyckoffEntry('o', 4, ['x'], ({'x': 1}, 0.5, 0.0)),
        WyckoffEntry('p', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[90] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 2, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('d', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('e', 4, ['x'], ({'x': -1}, {'x': 1}, 0.0)),
        WyckoffEntry('f', 4, ['x'], ({'x': -1}, {'x': 1}, 0.5)),
        WyckoffEntry('g', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[91] = [
        WyckoffEntry('a', 4, ['x'], ({'x': 1}, 0.0, 0.25)),
        WyckoffEntry('b', 4, ['x'], ({'x': 1}, 0.5, 0.25)),
        WyckoffEntry('c', 4, ['x'], ({'x': -1}, {'x': 1}, 0.125)),
        WyckoffEntry('d', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[92] = [
        WyckoffEntry('a', 4, ['x'], ({'x': -1}, {'x': 1}, 0.25)),
        WyckoffEntry('b', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[93] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.5, 0.5, 0.0)),
        WyckoffEntry('c', 2, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('d', 2, [], (0.0, 0.5, 0.5)),
        WyckoffEntry('e', 2, [], (0.0, 0.0, 0.25)),
        WyckoffEntry('f', 2, [], (0.5, 0.5, 0.25)),
        WyckoffEntry('g', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('h', 4, ['x'], (0.5, 0.5, {'x': 1})),
        WyckoffEntry('i', 4, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('j', 4, ['x'], ({'x': 1}, 0.0, 0.0)),
        WyckoffEntry('k', 4, ['x'], ({'x': 1}, 0.5, 0.5)),
        WyckoffEntry('l', 4, ['x'], ({'x': 1}, 0.0, 0.5)),
        WyckoffEntry('m', 4, ['x'], ({'x': 1}, 0.5, 0.0)),
        WyckoffEntry('n', 4, ['x'], ({'x': -1}, {'x': 1}, 0.75)),
        WyckoffEntry('o', 4, ['x'], ({'x': -1}, {'x': 1}, 0.25)),
        WyckoffEntry('p', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[94] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('d', 4, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('e', 4, ['x'], ({'x': -1}, {'x': 1}, 0.5)),
        WyckoffEntry('f', 4, ['x'], ({'x': -1}, {'x': 1}, 0.0)),
        WyckoffEntry('g', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[95] = [
        WyckoffEntry('a', 4, ['x'], ({'x': 1}, 0.0, 0.25)),
        WyckoffEntry('b', 4, ['x'], ({'x': 1}, 0.5, 0.25)),
        WyckoffEntry('c', 4, ['x'], ({'x': -1}, {'x': 1}, 0.375)),
        WyckoffEntry('d', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[96] = [
        WyckoffEntry('a', 4, ['x'], ({'x': -1}, {'x': 1}, 0.25)),
        WyckoffEntry('b', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[97] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 4, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('d', 4, [], (0.0, 0.5, 0.25)),
        WyckoffEntry('e', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 8, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('g', 8, ['x'], ({'x': -1}, {'x': 1}, 0.0)),
        WyckoffEntry('h', 8, ['x'], ({'x': 1}, 0.0, 0.0)),
        WyckoffEntry('i', 8, ['x'], ({'x': 1}, 0.0, 0.5)),
        WyckoffEntry('j', 8, ['x'], (({'x': -1}, 0.5), {'x': 1}, 0.25)),
        WyckoffEntry('k', 16, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[98] = [
        WyckoffEntry('a', 4, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 4, [], (0.5, 0.5, 0.0)),
        WyckoffEntry('c', 8, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('d', 8, ['x'], (({'x': -1}, 0.5), {'x': 1}, 0.25)),
        WyckoffEntry('e', 8, ['x'], ({'x': -1}, {'x': 1}, 0.0)),
        WyckoffEntry('f', 8, ['x'], ({'x': 1}, 0.25, 0.125)),
        WyckoffEntry('g', 16, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[99] = [
        WyckoffEntry('a', 1, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 1, ['x'], (0.5, 0.5, {'x': 1})),
        WyckoffEntry('c', 2, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('d', 4, ['x', 'y'], ({'x': 1}, {'x': 1}, {'y': 1})),
        WyckoffEntry('e', 4, ['x', 'y'], (0.0, {'x': 1}, {'y': 1})),
        WyckoffEntry('f', 4, ['x', 'y'], (0.5, {'x': 1}, {'y': 1})),
        WyckoffEntry('g', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[100] = [
        WyckoffEntry('a', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 2, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('c', 4, ['x', 'y'], (({'x': 1}, 0.5), {'x': 1}, {'y': 1})),
        WyckoffEntry('d', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[101] = [
        WyckoffEntry('a', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 2, ['x'], (0.5, 0.5, {'x': 1})),
        WyckoffEntry('c', 4, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('d', 4, ['x', 'y'], ({'x': 1}, {'x': 1}, {'y': 1})),
        WyckoffEntry('e', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[102] = [
        WyckoffEntry('a', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 4, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('c', 4, ['x', 'y'], ({'x': 1}, {'x': 1}, {'y': 1})),
        WyckoffEntry('d', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[103] = [
        WyckoffEntry('a', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 2, ['x'], (0.5, 0.5, {'x': 1})),
        WyckoffEntry('c', 4, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('d', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[104] = [
        WyckoffEntry('a', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 4, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('c', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[105] = [
        WyckoffEntry('a', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 2, ['x'], (0.5, 0.5, {'x': 1})),
        WyckoffEntry('c', 2, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('d', 4, ['x', 'y'], (0.0, {'x': 1}, {'y': 1})),
        WyckoffEntry('e', 4, ['x', 'y'], (0.5, {'x': 1}, {'y': 1})),
        WyckoffEntry('f', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[106] = [
        WyckoffEntry('a', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 4, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('c', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[107] = [
        WyckoffEntry('a', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 4, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('c', 8, ['x', 'y'], ({'x': 1}, {'x': 1}, {'y': 1})),
        WyckoffEntry('d', 8, ['x', 'y'], (0.0, {'x': 1}, {'y': 1})),
        WyckoffEntry('e', 16, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[108] = [
        WyckoffEntry('a', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 4, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('c', 8, ['x', 'y'], (({'x': 1}, 0.5), {'x': 1}, {'y': 1})),
        WyckoffEntry('d', 16, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[109] = [
        WyckoffEntry('a', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 8, ['x', 'y'], (0.0, {'x': 1}, {'y': 1})),
        WyckoffEntry('c', 16, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[110] = [
        WyckoffEntry('a', 8, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('b', 16, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[111] = [
        WyckoffEntry('a', 1, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 1, [], (0.5, 0.5, 0.5)),
        WyckoffEntry('c', 1, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('d', 1, [], (0.5, 0.5, 0.0)),
        WyckoffEntry('e', 2, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('f', 2, [], (0.0, 0.5, 0.5)),
        WyckoffEntry('g', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('h', 2, ['x'], (0.5, 0.5, {'x': 1})),
        WyckoffEntry('i', 4, ['x'], ({'x': 1}, 0.0, 0.0)),
        WyckoffEntry('j', 4, ['x'], ({'x': 1}, 0.5, 0.5)),
        WyckoffEntry('k', 4, ['x'], ({'x': 1}, 0.0, 0.5)),
        WyckoffEntry('l', 4, ['x'], ({'x': 1}, 0.5, 0.0)),
        WyckoffEntry('m', 4, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('n', 4, ['x', 'y'], ({'x': 1}, {'x': 1}, {'y': 1})),
        WyckoffEntry('o', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[112] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.25)),
        WyckoffEntry('b', 2, [], (0.0, 0.5, 0.75)),
        WyckoffEntry('c', 2, [], (0.5, 0.5, 0.25)),
        WyckoffEntry('d', 2, [], (0.0, 0.5, 0.25)),
        WyckoffEntry('e', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('f', 2, [], (0.5, 0.5, 0.0)),
        WyckoffEntry('g', 4, ['x'], ({'x': 1}, 0.0, 0.25)),
        WyckoffEntry('h', 4, ['x'], ({'x': 1}, 0.5, 0.75)),
        WyckoffEntry('i', 4, ['x'], ({'x': 1}, 0.5, 0.25)),
        WyckoffEntry('j', 4, ['x'], ({'x': 1}, 0.0, 0.75)),
        WyckoffEntry('k', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('l', 4, ['x'], (0.5, 0.5, {'x': 1})),
        WyckoffEntry('m', 4, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('n', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[113] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 2, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('d', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('e', 4, ['x', 'y'], (({'x': 1}, 0.5), {'x': 1}, {'y': 1})),
        WyckoffEntry('f', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[114] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('d', 4, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('e', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[115] = [
        WyckoffEntry('a', 1, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 1, [], (0.5, 0.5, 0.0)),
        WyckoffEntry('c', 1, [], (0.5, 0.5, 0.5)),
        WyckoffEntry('d', 1, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('e', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 2, ['x'], (0.5, 0.5, {'x': 1})),
        WyckoffEntry('g', 2, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('h', 4, ['x'], ({'x': -1}, {'x': 1}, 0.0)),
        WyckoffEntry('i', 4, ['x'], ({'x': -1}, {'x': 1}, 0.5)),
        WyckoffEntry('j', 4, ['x', 'y'], (0.0, {'x': 1}, {'y': 1})),
        WyckoffEntry('k', 4, ['x', 'y'], (0.5, {'x': 1}, {'y': 1})),
        WyckoffEntry('l', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[116] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.25)),
        WyckoffEntry('b', 2, [], (0.5, 0.5, 0.25)),
        WyckoffEntry('c', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('d', 2, [], (0.5, 0.5, 0.0)),
        WyckoffEntry('e', 4, ['x'], ({'x': -1}, {'x': 1}, 0.75)),
        WyckoffEntry('f', 4, ['x'], ({'x': -1}, {'x': 1}, 0.25)),
        WyckoffEntry('g', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('h', 4, ['x'], (0.5, 0.5, {'x': 1})),
        WyckoffEntry('i', 4, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('j', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[117] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 2, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('d', 2, [], (0.0, 0.5, 0.5)),
        WyckoffEntry('e', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 4, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('g', 4, ['x'], (({'x': -1}, 0.5), {'x': 1}, 0.0)),
        WyckoffEntry('h', 4, ['x'], (({'x': -1}, 0.5), {'x': 1}, 0.5)),
        WyckoffEntry('i', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[118] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 2, [], (0.0, 0.5, 0.25)),
        WyckoffEntry('d', 2, [], (0.0, 0.5, 0.75)),
        WyckoffEntry('e', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 4, ['x'], (({'x': -1}, 0.5), {'x': 1}, 0.25)),
        WyckoffEntry('g', 4, ['x'], (({'x': -1}, 0.5), {'x': 1}, 0.75)),
        WyckoffEntry('h', 4, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('i', 8, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[119] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 2, [], (0.5, 0.0, 0.75)),
        WyckoffEntry('d', 2, [], (0.5, 0.0, 0.25)),
        WyckoffEntry('e', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 4, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('g', 8, ['x'], ({'x': -1}, {'x': 1}, 0.0)),
        WyckoffEntry('h', 8, ['x'], (({'x': -1}, 0.5), {'x': 1}, 0.25)),
        WyckoffEntry('i', 8, ['x', 'y'], (0.0, {'x': 1}, {'y': 1})),
        WyckoffEntry('j', 16, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[120] = [
        WyckoffEntry('a', 4, [], (0.0, 0.0, 0.25)),
        WyckoffEntry('b', 4, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('c', 4, [], (0.5, 0.0, 0.25)),
        WyckoffEntry('d', 4, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('e', 8, ['x'], ({'x': -1}, {'x': 1}, 0.25)),
        WyckoffEntry('f', 8, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('g', 8, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('h', 8, ['x'], (({'x': -1}, 0.5), {'x': 1}, 0.0)),
        WyckoffEntry('i', 16, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[121] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 4, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('d', 4, [], (0.5, 0.0, 0.25)),
        WyckoffEntry('e', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 8, ['x'], ({'x': 1}, 0.0, 0.0)),
        WyckoffEntry('g', 8, ['x'], ({'x': 1}, 0.0, 0.5)),
        WyckoffEntry('h', 8, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('i', 8, ['x', 'y'], ({'x': 1}, {'x': 1}, {'y': 1})),
        WyckoffEntry('j', 16, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[122] = [
        WyckoffEntry('a', 4, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 4, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 8, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('d', 8, ['x'], ({'x': 1}, 0.25, 0.125)),
        WyckoffEntry('e', 16, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[123] = [
        WyckoffEntry('a', 1, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 1, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 1, [], (0.5, 0.5, 0.0)),
        WyckoffEntry('d', 1, [], (0.5, 0.5, 0.5)),
        WyckoffEntry('e', 2, [], (0.0, 0.5, 0.5)),
        WyckoffEntry('f', 2, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('g', 2, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('h', 2, ['x'], (0.5, 0.5, {'x': 1})),
        WyckoffEntry('i', 4, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('j', 4, ['x'], ({'x': -1}, {'x': 1}, 0.0)),
        WyckoffEntry('k', 4, ['x'], ({'x': -1}, {'x': 1}, 0.5)),
        WyckoffEntry('l', 4, ['x'], ({'x': 1}, 0.0, 0.0)),
        WyckoffEntry('m', 4, ['x'], ({'x': 1}, 0.0, 0.5)),
        WyckoffEntry('n', 4, ['x'], ({'x': 1}, 0.5, 0.0)),
        WyckoffEntry('o', 4, ['x'], ({'x': 1}, 0.5, 0.5)),
        WyckoffEntry('p', 8, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.0)),
        WyckoffEntry('q', 8, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.5)),
        WyckoffEntry('r', 8, ['x', 'y'], ({'x': 1}, {'x': 1}, {'y': 1})),
        WyckoffEntry('s', 8, ['x', 'y'], (0.0, {'x': 1}, {'y': 1})),
        WyckoffEntry('t', 8, ['x', 'y'], (0.5, {'x': 1}, {'y': 1})),
        WyckoffEntry('u', 16, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[124] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.25)),
        WyckoffEntry('b', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('c', 2, [], (0.5, 0.5, 0.25)),
        WyckoffEntry('d', 2, [], (0.5, 0.5, 0.0)),
        WyckoffEntry('e', 4, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('f', 4, [], (0.0, 0.5, 0.25)),
        WyckoffEntry('g', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('h', 4, ['x'], (0.5, 0.5, {'x': 1})),
        WyckoffEntry('i', 8, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('j', 8, ['x'], ({'x': -1}, {'x': 1}, 0.25)),
        WyckoffEntry('k', 8, ['x'], ({'x': 1}, 0.0, 0.25)),
        WyckoffEntry('l', 8, ['x'], ({'x': 1}, 0.5, 0.25)),
        WyckoffEntry('m', 8, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.0)),
        WyckoffEntry('n', 16, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[125] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 2, [], (0.5, 0.0, 0.0)),
        WyckoffEntry('d', 2, [], (0.5, 0.0, 0.5)),
        WyckoffEntry('e', 4, [], (0.25, 0.25, 0.0)),
        WyckoffEntry('f', 4, [], (0.25, 0.25, 0.5)),
        WyckoffEntry('g', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('h', 4, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('i', 8, ['x'], ({'x': -1}, {'x': 1}, 0.0)),
        WyckoffEntry('j', 8, ['x'], ({'x': -1}, {'x': 1}, 0.5)),
        WyckoffEntry('k', 8, ['x'], ({'x': 1}, 0.0, 0.0)),
        WyckoffEntry('l', 8, ['x'], ({'x': 1}, 0.0, 0.5)),
        WyckoffEntry('m', 8, ['x', 'y'], (({'x': 1}, 0.5), {'x': 1}, {'y': 1})),
        WyckoffEntry('n', 16, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[126] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 4, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('d', 4, [], (0.5, 0.0, 0.25)),
        WyckoffEntry('e', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 8, [], (0.25, 0.25, 0.25)),
        WyckoffEntry('g', 8, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('h', 8, ['x'], ({'x': -1}, {'x': 1}, 0.0)),
        WyckoffEntry('i', 8, ['x'], ({'x': 1}, 0.0, 0.0)),
        WyckoffEntry('j', 8, ['x'], ({'x': 1}, 0.0, 0.5)),
        WyckoffEntry('k', 16, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[127] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 2, [], (0.0, 0.5, 0.5)),
        WyckoffEntry('d', 2, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('e', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 4, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('g', 4, ['x'], (({'x': -1}, 0.5), {'x': 1}, 0.0)),
        WyckoffEntry('h', 4, ['x'], (({'x': -1}, 0.5), {'x': 1}, 0.5)),
        WyckoffEntry('i', 8, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.0)),
        WyckoffEntry('j', 8, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.5)),
        WyckoffEntry('k', 8, ['x', 'y'], (({'x': 1}, 0.5), {'x': 1}, {'y': 1})),
        WyckoffEntry('l', 16, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[128] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 4, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('d', 4, [], (0.0, 0.5, 0.25)),
        WyckoffEntry('e', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 8, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('g', 8, ['x'], (({'x': -1}, 0.5), {'x': 1}, 0.25)),
        WyckoffEntry('h', 8, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.0)),
        WyckoffEntry('i', 16, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[129] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 2, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('d', 4, [], (0.25, 0.25, 0.0)),
        WyckoffEntry('e', 4, [], (0.25, 0.25, 0.5)),
        WyckoffEntry('f', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('g', 8, ['x'], ({'x': -1}, {'x': 1}, 0.0)),
        WyckoffEntry('h', 8, ['x'], ({'x': -1}, {'x': 1}, 0.5)),
        WyckoffEntry('i', 8, ['x', 'y'], (0.0, {'x': 1}, {'y': 1})),
        WyckoffEntry('j', 8, ['x', 'y'], (({'x': 1}, 0.5), {'x': 1}, {'y': 1})),
        WyckoffEntry('k', 16, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[130] = [
        WyckoffEntry('a', 4, [], (0.0, 0.0, 0.25)),
        WyckoffEntry('b', 4, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('c', 4, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('d', 8, [], (0.25, 0.25, 0.0)),
        WyckoffEntry('e', 8, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 8, ['x'], ({'x': -1}, {'x': 1}, 0.25)),
        WyckoffEntry('g', 16, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[131] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.5, 0.5, 0.0)),
        WyckoffEntry('c', 2, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('d', 2, [], (0.0, 0.5, 0.5)),
        WyckoffEntry('e', 2, [], (0.0, 0.0, 0.25)),
        WyckoffEntry('f', 2, [], (0.5, 0.5, 0.25)),
        WyckoffEntry('g', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('h', 4, ['x'], (0.5, 0.5, {'x': 1})),
        WyckoffEntry('i', 4, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('j', 4, ['x'], ({'x': 1}, 0.0, 0.0)),
        WyckoffEntry('k', 4, ['x'], ({'x': 1}, 0.5, 0.5)),
        WyckoffEntry('l', 4, ['x'], ({'x': 1}, 0.0, 0.5)),
        WyckoffEntry('m', 4, ['x'], ({'x': 1}, 0.5, 0.0)),
        WyckoffEntry('n', 8, ['x'], ({'x': -1}, {'x': 1}, 0.25)),
        WyckoffEntry('o', 8, ['x', 'y'], (0.0, {'x': 1}, {'y': 1})),
        WyckoffEntry('p', 8, ['x', 'y'], (0.5, {'x': 1}, {'y': 1})),
        WyckoffEntry('q', 8, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.0)),
        WyckoffEntry('r', 16, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[132] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.0, 0.0, 0.25)),
        WyckoffEntry('c', 2, [], (0.5, 0.5, 0.0)),
        WyckoffEntry('d', 2, [], (0.5, 0.5, 0.25)),
        WyckoffEntry('e', 4, [], (0.0, 0.5, 0.25)),
        WyckoffEntry('f', 4, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('g', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('h', 4, ['x'], (0.5, 0.5, {'x': 1})),
        WyckoffEntry('i', 4, ['x'], ({'x': -1}, {'x': 1}, 0.5)),
        WyckoffEntry('j', 4, ['x'], ({'x': -1}, {'x': 1}, 0.0)),
        WyckoffEntry('k', 8, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('l', 8, ['x'], ({'x': 1}, 0.0, 0.25)),
        WyckoffEntry('m', 8, ['x'], ({'x': 1}, 0.5, 0.25)),
        WyckoffEntry('n', 8, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.0)),
        WyckoffEntry('o', 8, ['x', 'y'], ({'x': 1}, {'x': 1}, {'y': 1})),
        WyckoffEntry('p', 16, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[133] = [
        WyckoffEntry('a', 4, [], (0.0, 0.5, 0.25)),
        WyckoffEntry('b', 4, [], (0.0, 0.0, 0.25)),
        WyckoffEntry('c', 4, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('d', 4, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('e', 8, [], (0.25, 0.25, 0.25)),
        WyckoffEntry('f', 8, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('g', 8, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('h', 8, ['x'], ({'x': 1}, 0.0, 0.25)),
        WyckoffEntry('i', 8, ['x'], ({'x': 1}, 0.0, 0.75)),
        WyckoffEntry('j', 8, ['x'], (({'x': -1}, 0.5), {'x': 1}, 0.0)),
        WyckoffEntry('k', 16, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[134] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 4, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('d', 4, [], (0.0, 0.5, 0.25)),
        WyckoffEntry('e', 4, [], (0.25, 0.25, 0.25)),
        WyckoffEntry('f', 4, [], (0.25, 0.25, 0.75)),
        WyckoffEntry('g', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('h', 8, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('i', 8, ['x'], ({'x': 1}, 0.0, 0.0)),
        WyckoffEntry('j', 8, ['x'], ({'x': 1}, 0.0, 0.5)),
        WyckoffEntry('k', 8, ['x'], (({'x': -1}, 0.5), {'x': 1}, 0.75)),
        WyckoffEntry('l', 8, ['x'], (({'x': -1}, 0.5), {'x': 1}, 0.25)),
        WyckoffEntry('m', 8, ['x', 'y'], ({'x': 1}, {'x': 1}, {'y': 1})),
        WyckoffEntry('n', 16, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[135] = [
        WyckoffEntry('a', 4, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 4, [], (0.0, 0.0, 0.25)),
        WyckoffEntry('c', 4, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('d', 4, [], (0.0, 0.5, 0.25)),
        WyckoffEntry('e', 8, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 8, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('g', 8, ['x'], (({'x': -1}, 0.5), {'x': 1}, 0.25)),
        WyckoffEntry('h', 8, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.0)),
        WyckoffEntry('i', 16, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[136] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 4, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('d', 4, [], (0.5, 0.0, 0.25)),
        WyckoffEntry('e', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 4, ['x'], ({'x': -1}, {'x': 1}, 0.5)),
        WyckoffEntry('g', 4, ['x'], ({'x': -1}, {'x': 1}, 0.0)),
        WyckoffEntry('h', 8, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('i', 8, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.0)),
        WyckoffEntry('j', 8, ['x', 'y'], ({'x': 1}, {'x': 1}, {'y': 1})),
        WyckoffEntry('k', 16, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[137] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('d', 4, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('e', 8, [], (0.25, 0.25, 0.25)),
        WyckoffEntry('f', 8, ['x'], ({'x': -1}, {'x': 1}, 0.0)),
        WyckoffEntry('g', 8, ['x', 'y'], (0.0, {'x': 1}, {'y': 1})),
        WyckoffEntry('h', 16, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[138] = [
        WyckoffEntry('a', 4, [], (0.0, 0.0, 0.25)),
        WyckoffEntry('b', 4, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('c', 4, [], (0.25, 0.25, 0.25)),
        WyckoffEntry('d', 4, [], (0.25, 0.25, 0.75)),
        WyckoffEntry('e', 4, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('f', 8, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('g', 8, ['x'], ({'x': -1}, {'x': 1}, 0.75)),
        WyckoffEntry('h', 8, ['x'], ({'x': -1}, {'x': 1}, 0.25)),
        WyckoffEntry('i', 8, ['x', 'y'], (({'x': 1}, 0.5), {'x': 1}, {'y': 1})),
        WyckoffEntry('j', 16, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[139] = [
        WyckoffEntry('a', 2, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 2, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 4, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('d', 4, [], (0.5, 0.0, 0.25)),
        WyckoffEntry('e', 4, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 8, [], (0.25, 0.25, 0.25)),
        WyckoffEntry('g', 8, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('h', 8, ['x'], ({'x': -1}, {'x': 1}, 0.0)),
        WyckoffEntry('i', 8, ['x'], ({'x': 1}, 0.0, 0.0)),
        WyckoffEntry('j', 8, ['x'], ({'x': 1}, 0.0, 0.5)),
        WyckoffEntry('k', 16, ['x'], (({'x': -1}, 0.5), {'x': 1}, 0.25)),
        WyckoffEntry('l', 16, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.0)),
        WyckoffEntry('m', 16, ['x', 'y'], ({'x': 1}, {'x': 1}, {'y': 1})),
        WyckoffEntry('n', 16, ['x', 'y'], (0.0, {'x': 1}, {'y': 1})),
        WyckoffEntry('o', 32, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[140] = [
        WyckoffEntry('a', 4, [], (0.0, 0.0, 0.25)),
        WyckoffEntry('b', 4, [], (0.5, 0.0, 0.25)),
        WyckoffEntry('c', 4, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('d', 4, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('e', 8, [], (0.25, 0.25, 0.25)),
        WyckoffEntry('f', 8, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('g', 8, ['x'], (0.0, 0.5, {'x': 1})),
        WyckoffEntry('h', 8, ['x'], (({'x': -1}, 0.5), {'x': 1}, 0.0)),
        WyckoffEntry('i', 16, ['x'], ({'x': -1}, {'x': 1}, 0.25)),
        WyckoffEntry('j', 16, ['x'], ({'x': 1}, 0.0, 0.25)),
        WyckoffEntry('k', 16, ['x', 'y'], ({'y': 1}, {'x': 1}, 0.0)),
        WyckoffEntry('l', 16, ['x', 'y'], (({'x': 1}, 0.5), {'x': 1}, {'y': 1})),
        WyckoffEntry('m', 32, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[141] = [
        WyckoffEntry('a', 4, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 4, [], (0.0, 0.0, 0.5)),
        WyckoffEntry('c', 8, [], (0.0, 0.25, 0.125)),
        WyckoffEntry('d', 8, [], (0.0, 0.25, 0.625)),
        WyckoffEntry('e', 8, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('f', 16, ['x'], ({'x': 1}, 0.25, 0.125)),
        WyckoffEntry('g', 16, ['x'], ({'x': -1}, {'x': 1}, 0.0)),
        WyckoffEntry('h', 16, ['x', 'y'], (0.0, {'x': 1}, {'y': 1})),
        WyckoffEntry('i', 32, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    db[142] = [
        WyckoffEntry('a', 8, [], (0.0, 0.0, 0.0)),
        WyckoffEntry('b', 8, [], (0.0, 0.5, 0.0)),
        WyckoffEntry('c', 16, [], (0.0, 0.25, 0.125)),
        WyckoffEntry('d', 16, ['x'], (0.0, 0.0, {'x': 1})),
        WyckoffEntry('e', 16, ['x'], ({'x': 1}, 0.25, 0.375)),
        WyckoffEntry('f', 16, ['x'], (({'x': -1}, 0.5), {'x': 1}, 0.0)),
        WyckoffEntry('g', 32, ['x', 'y', 'z'], ({'x': 1}, {'y': 1}, {'z': 1})),
    ]

    return db
