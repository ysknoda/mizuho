"""Euclidean normalizer and equivalent assignments (ADR-0011), validated
three ways: the defining property N G N^-1 = G, the spinel and anatase
cases, and spglib confirming that merged pairs are one crystal."""

import numpy as np
import pytest

from mizuho.data import WYCKOFF_DB, build_point, get_ops
from mizuho.lattice import all_image_distances
from mizuho.normalizer import (
    _op_key,
    canonical_key,
    coset_representatives,
    letter_permutations,
    merge_equivalent,
    polar_directions,
)
from mizuho.structure import build_structure
from mizuho.symmetry import to_mizuho

FIXTURE_GROUPS = [2, 15, 58, 62, 136, 141, 152, 167, 186, 221, 225, 227]


def _conjugates_onto_itself(sg, P, s):
    rot, tr = get_ops(sg)
    ops = {_op_key(R, t) for R, t in zip(rot, tr)}
    Pinv = np.linalg.inv(P)
    images = {
        _op_key(P @ R @ Pinv, P @ t + s - P @ R @ Pinv @ s) for R, t in zip(rot, tr)
    }
    return images == ops


@pytest.mark.parametrize("sg", FIXTURE_GROUPS)
def test_every_normalizer_element_maps_the_group_onto_itself(sg):
    for P, s in coset_representatives(sg):
        assert abs(abs(np.linalg.det(P)) - 1) < 1e-9
        assert _conjugates_onto_itself(sg, P, s)


@pytest.mark.slow
@pytest.mark.parametrize("sg", range(1, 231))
def test_normalizer_is_well_defined_for_every_space_group(sg):
    for P, s in coset_representatives(sg):
        assert _conjugates_onto_itself(sg, P, s)
    perms = letter_permutations(sg)
    letters = sorted(e.letter for e in WYCKOFF_DB[sg])
    for perm in perms:
        assert sorted(perm.values()) == letters  # a bijection


@pytest.mark.parametrize(
    "sg,index",
    [
        (2, 8),  # P-1: every half-integer origin shift
        (62, 8),  # Pnma: normalizer Pmmm with halved axes (ADR-0011)
        (227, 2),  # Fd-3m: normalizer Im-3m
    ],
)
def test_index_over_the_group(sg, index):
    assert len(coset_representatives(sg)) == index


def test_polar_directions():
    assert polar_directions(186).shape == (1, 3)
    np.testing.assert_allclose(np.abs(polar_directions(186)[0]), [0, 0, 1], atol=1e-12)
    assert polar_directions(62).shape == (0, 3)
    assert polar_directions(227).shape == (0, 3)
    assert polar_directions(1).shape == (3, 3)


def test_spinel_representatives_are_one_crystal():
    """Phase 1's top candidate and the published assignment (ADR-0011)."""
    assert canonical_key(227, [("Mg", "a"), ("Al", "d"), ("O", "e")]) == canonical_key(
        227, [("Mg", "b"), ("Al", "c"), ("O", "e")]
    )
    assert canonical_key(227, [("Mg", "a"), ("Al", "d"), ("O", "e")]) != canonical_key(
        227, [("Mg", "a"), ("Al", "c"), ("O", "e")]
    )


def test_anatase_cifs_are_one_crystal():
    assert canonical_key(141, [("Ti", "a"), ("O", "e")]) == canonical_key(
        141, [("Ti", "b"), ("O", "e")]
    )


def _generic_structure(sg, specs, rng):
    entries = {e.letter: e for e in WYCKOFF_DB[sg]}
    from mizuho.structure import assignments_to_sites, resolve_fixed_assignment

    cfg = {
        "spacegroup": sg,
        "wyckoff_sites": assignments_to_sites(resolve_fixed_assignment(sg, specs)),
    }
    # A lattice with no accidental extra metric symmetry.
    lat = {
        "cubic": [7.3],
        "tetragonal": [5.1, 7.9],
        "orthorhombic": [5.3, 6.7, 8.1],
        "hexagonal": [5.2, 8.3],
        "trigonal": [5.2, 8.3],
        "monoclinic": [5.3, 6.7, 8.1, 101.3],
        "triclinic": [5.3, 6.7, 8.1, 83.0, 97.0, 104.0],
    }
    from mizuho.lattice import get_lattice_system

    params = []
    for s in specs:
        params += list(rng.uniform(0.05, 0.45, len(entries[s["letter"]].free_params)))
    pv = np.array(lat[get_lattice_system(sg)] + params)
    return cfg, pv


def _distance_signature(lat, frac, species):
    out = []
    for el in sorted(set(species)):
        sp = np.array(species)
        d = all_image_distances(frac[sp == el], frac, lat).ravel()
        out.append(np.sort(d[(d > 1e-6) & (d < 6.0)]))
    return out


@pytest.mark.parametrize(
    "sg,specs",
    [
        (
            227,
            [
                {"element": "Mg", "letter": "a"},
                {"element": "Al", "letter": "d"},
                {"element": "O", "letter": "e"},
            ],
        ),
        (141, [{"element": "Ti", "letter": "a"}, {"element": "O", "letter": "e"}]),
        (136, [{"element": "Ti", "letter": "a"}, {"element": "O", "letter": "f"}]),
        (
            62,
            [
                {"element": "Mg", "letter": "a"},
                {"element": "Si", "letter": "c"},
                {"element": "O", "letter": "d"},
            ],
        ),
        (
            15,
            [
                {"element": "Ca", "letter": "a"},
                {"element": "Mg", "letter": "c"},
                {"element": "O", "letter": "f"},
            ],
        ),
        (
            58,
            [
                {"element": "Zn", "letter": "a"},
                {"element": "As", "letter": "c"},
                {"element": "O", "letter": "g"},
            ],
        ),
    ],
)
def test_spglib_confirms_mapped_assignments_are_the_same_crystal(sg, specs):
    """Apply each normalizer element to a generic structure: spglib re-reads
    the image on exactly the permuted letters, and every interatomic
    distance is unchanged -- the same crystal."""
    rng = np.random.default_rng(7)
    cfg, pv = _generic_structure(sg, specs, rng)
    st = build_structure(pv, cfg)
    lat, frac, sp = st["lattice_matrix"], st["frac_coords"], st["species"]
    sig = _distance_signature(lat, frac, sp)
    original = sorted((s["element"], s["letter"]) for s in specs)
    for P, s in coset_representatives(sg):
        image = (frac @ P.T + s) % 1.0
        m = to_mizuho(lat, image, sp, sg, symprec=1e-5)
        got = sorted((x["element"], x["letter"]) for x in m.specs)
        assert canonical_key(sg, got) == canonical_key(sg, original)
        mapped = [(el, _image_letter(sg, P, s, le)) for el, le in original]
        assert got == sorted(mapped)
        for a, b in zip(sig, _distance_signature(lat, image, sp)):
            np.testing.assert_allclose(a, b, atol=1e-8)


def _image_letter(sg, P, s, letter):
    from mizuho.symmetry import solve_free_params

    entries = WYCKOFF_DB[sg]
    e = next(x for x in entries if x.letter == letter)
    q = (
        P
        @ build_point(
            e.coord_template,
            {p: 0.1234 + 0.071 * i for i, p in enumerate(e.free_params)},
        )
        + s
    )
    return next(
        f.letter
        for f in entries
        if f.multiplicity == e.multiplicity
        and len(f.free_params) == len(e.free_params)
        and solve_free_params(sg, f, q) is not None
    )


def test_merge_keeps_first_of_each_class_in_order():
    from mizuho.structure import resolve_fixed_assignment

    a1 = resolve_fixed_assignment(
        227,
        [
            {"element": "Mg", "letter": "b"},
            {"element": "Al", "letter": "c"},
            {"element": "O", "letter": "e"},
        ],
    )
    a2 = resolve_fixed_assignment(
        227,
        [
            {"element": "Mg", "letter": "a"},
            {"element": "Al", "letter": "c"},
            {"element": "O", "letter": "e"},
        ],
    )
    a3 = resolve_fixed_assignment(
        227,
        [
            {"element": "Mg", "letter": "a"},
            {"element": "Al", "letter": "d"},
            {"element": "O", "letter": "e"},
        ],
    )
    assert merge_equivalent(227, [a1, a2, a3]) == [a1, a2]
