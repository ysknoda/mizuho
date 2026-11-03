"""Exact bond-counting pruning of Wyckoff assignments (ADR-0037).

Hand-checkable cases for both conditions, the coordination ranges they read
from a config, and a property test of the site-symmetry orbit sizes against
neighbour counts measured in built structures.
"""

import itertools

import numpy as np
import pytest
from mizuho.search.bond_counting import (
    BALANCE,
    SITE_SYMMETRY,
    BondCountCheck,
    _semigroup_counts,
    bond_rules,
    bond_stabiliser_orders,
    objective_constraints,
    site_symmetry,
)

from mizuho.data import WYCKOFF_DB
from mizuho.pipeline import build_sg_config
from mizuho.search.base import canonical_representatives, expand_letter_choices
from mizuho.search.dfs import wyckoff_tree_search
from mizuho.structure import SiteAssignment, assignments_to_sites, build_structure


def _sites(sg, *specs):
    entries = {e.letter: e for e in WYCKOFF_DB[sg]}
    return [SiteAssignment(el, entries[le]) for el, le in specs]


def _check(sg, composition, charges, cns, anion_cns=None, **extra):
    config = {
        "spacegroup": sg,
        "composition": composition,
        "ionic_charges": charges,
        "coordination_numbers": cns,
        **extra,
    }
    if anion_cns:
        config["anion_coordination_numbers"] = anion_cns
    return BondCountCheck.from_config(config, list(charges), composition_free=False)


AL_O = {"Al": 3, "O": -2}


def test_one_to_eight_multiplicities_cannot_balance():
    """Pm-3m Al@1a + O@8g: 1 n_AlO = 8 n_OAl, so O's 2-4 bonds force Al to
    16-32, beyond its 6."""
    check = _check(221, "AlO8", AL_O, {"Al": [4, 6]}, {"O": [2, 4]})
    assert check.rejection(_sites(221, ("Al", "a"), ("O", "g"))) == BALANCE


def test_corundum_passes():
    """R-3c Al@12c O@18e: 12 x 6 = 18 x 4 Al-O bonds per cell."""
    check = _check(167, "Al2O3", AL_O, {"Al": 6}, {"O": 4})
    assert check.rejection(_sites(167, ("Al", "c"), ("O", "e"))) is None


def test_an_m3m_site_cannot_be_tetrahedral():
    """Fm-3m Al@4a + O@8c balances at Al CN 4, O CN 2 (4 x 4 = 8 x 2), but
    the eight 8c points around 4a are one orbit of its m-3m site group."""
    check = _check(225, "AlO2", AL_O, {"Al": 4}, {"O": 2})
    assert check.rejection(_sites(225, ("Al", "a"), ("O", "c"))) == SITE_SYMMETRY
    fluorite = _check(225, "CaF2", {"Ca": 2, "F": -1}, {"Ca": 8})
    assert fluorite.rejection(_sites(225, ("Ca", "a"), ("F", "c"))) is None


def test_rock_salt_neighbour_counts():
    """Fm-3m 4a around 4b: shells of 6 (axes), 8 (body diagonals), then
    sums; never 4 or 5 (m-3m has no orbit of that size off its centre)."""
    site = site_symmetry(225, "a")
    assert site.order == 48
    assert _semigroup_counts(bond_stabiliser_orders(225, "a", "b"), 48, 14) == (
        0,
        6,
        8,
        12,
        14,
    )
    assert _semigroup_counts(bond_stabiliser_orders(225, "a", "c"), 48, 16) == (
        0,
        8,
        16,
    )


def test_site_group_orders_match_multiplicities():
    """|S| x multiplicity = number of operations, in every group (the
    assertion inside site_symmetry)."""
    for sg, entries in WYCKOFF_DB.items():
        for e in entries:
            site_symmetry(sg, e.letter)


def test_bonded_anions_hold_a_pair_at_zero():
    """ADR-0028 (a): Si bonds O only, so Si-F bonds are held at 0."""
    config = {
        "spacegroup": 1,
        "composition": "Al2SiO4F2",
        "ionic_charges": {"Al": 3, "Si": 4, "O": -2, "F": -1},
        "coordination_numbers": {"Al": 6, "Si": 4},
        "bonded_anions": {"Si": ["O"]},
    }
    rules = bond_rules(objective_constraints(config))
    si = {r.partners: (r.lo, r.hi) for r in rules.cation_rules if r.centre == "Si"}
    assert si == {("O",): (4, 4), ("F",): (0, 0)}
    # F's derived range counts Al, the cation bonding it; the auto-coupling
    # F(X) also counts Si, whose bonds to F the Si rule holds at 0.
    assert ("F", ("Al",)) in {(r.centre, r.partners) for r in rules.anion_rules}


def test_derived_anion_band_edge_keeps_both_bounds():
    """Al-O-N: O's cutoff 2.255 A ends its full-weight zone at 1.955 A, the
    Al-O band's end, and Al's 2.394 A cutoff is longer; the Al-O bond
    constraint closes the gap, so O keeps its [2, 4] range."""
    config = {
        "spacegroup": 1,
        "composition": "Al3O3N",
        "ionic_charges": {"Al": 3, "O": -2, "N": -3},
        "coordination_numbers": {"Al": [4, 6]},
        "anion_coordination_numbers": {"O": [2, 4], "N": [3, 4]},
    }
    rules = bond_rules(objective_constraints(config))
    anions = {r.centre: (r.lo, r.hi) for r in rules.anion_rules}
    assert anions == {"O": (2, 4), "N": (3, 4)}


def test_rules_match_the_pipeline_config():
    """The search derives each composition's rules as build_sg_config does."""
    base = {
        "elements": ["Al", "O", "N"],
        "ionic_charges": {"Al": 3, "O": -2, "N": -3},
        "coordination_numbers": {"Al": [4, 6]},
        "anion_coordination_numbers": {"O": [2, 4], "N": [3, 4]},
    }
    for comp in ("Al3O3N", "Al5O6N", "Al8O3N6"):
        assert bond_rules(
            objective_constraints(dict(base, composition=comp))
        ) == bond_rules(build_sg_config(base, 225, composition=comp))


@pytest.mark.parametrize("sg", [72, 225, 221, 139])
def test_representative_is_never_stricter_than_its_letters(sg):
    """A representative leaf is rejected only when every concrete letter
    choice is (Ibam's 4a 222 and 4c 2/m share one shape group)."""
    check = _check(sg, "AlO", AL_O, {"Al": [4, 6]}, {"O": [2, 4]})
    reps = [
        e for e in canonical_representatives(WYCKOFF_DB[sg]) if e.multiplicity <= 16
    ]
    for ea, eo in itertools.product(reps, reps):
        rep = [SiteAssignment("Al", ea), SiteAssignment("O", eo)]
        if any(check.accepts(c) for c in expand_letter_choices(rep, WYCKOFF_DB[sg])):
            assert check.accepts(rep, representative=True)


def test_pruned_search_is_the_filtered_unpruned_search():
    """Both strategies return exactly the unpruned enumeration minus the
    assignments the check rejects."""
    base = {
        "spacegroup": 167,
        "elements": ["Al", "O", "N"],
        "ionic_charges": {"Al": 3, "O": -2, "N": -3},
        "coordination_numbers": {"Al": [4, 6]},
        "anion_coordination_numbers": {"O": [2, 4], "N": [3, 4]},
        "tree_search": {"max_atoms": 21, "max_sites": 5},
    }
    off = dict(base, tree_search=dict(base["tree_search"], bond_count_pruning=False))
    unpruned = wyckoff_tree_search(off)
    check = BondCountCheck.from_config(base, base["elements"], composition_free=True)
    expected = {
        tuple(sorted((a.element, a.entry.letter) for a in asg))
        for asg in unpruned
        if check.accepts(asg)
    }
    assert 0 < len(expected) < len(unpruned)
    for strategy in ("best_first", "exhaustive"):
        cfg = dict(base, tree_search=dict(base["tree_search"], strategy=strategy))
        got = {
            tuple(sorted((a.element, a.entry.letter) for a in asg))
            for asg in wyckoff_tree_search(cfg)
        }
        assert got == expected, strategy


# ---------------------------------------------------------------------------
# Property test: neighbour counts measured in built structures
# ---------------------------------------------------------------------------

_LATTICE = {
    "cubic": [5.3],
    "tetragonal": [5.1, 6.7],
    "hexagonal": [5.2, 6.9],
    "trigonal": [5.2, 6.9],
    "orthorhombic": [5.1, 5.9, 6.7],
    "monoclinic": [5.1, 5.9, 6.7, 101.0],
    "triclinic": [5.1, 5.9, 6.7, 84.0, 97.0, 103.0],
}


def _orbit(sg, letter, rng, lattice):
    from mizuho.lattice import get_lattice_system

    asg = _sites(sg, ("X", letter))
    cfg = {"spacegroup": sg, "wyckoff_sites": assignments_to_sites(asg)}
    pv = np.array(
        _LATTICE[get_lattice_system(sg)]
        + list(rng.uniform(0.03, 0.47, len(asg[0].entry.free_params)))
    )
    s = build_structure(pv, cfg)
    return s["frac_coords"], s["lattice_matrix"]


def _distances(x, frac_j, lat):
    shifts = np.array(list(itertools.product(range(-3, 4), repeat=3)))
    d = (frac_j[:, None, :] + shifts[None, :, :] - x) @ lat
    return np.linalg.norm(d, axis=2).ravel()


@pytest.mark.parametrize("sg", [221, 225, 227, 229, 139, 194, 167, 62, 72, 12, 15, 2])
def test_measured_neighbour_counts_obey_both_conditions(sg):
    """For random free parameters and every distance threshold between
    measured shells, the number of orbit-j atoms within it of an orbit-i
    atom lies in the computed set, and m_i n_ij = m_j n_ji."""
    rng = np.random.default_rng(sg)
    entries = WYCKOFF_DB[sg]
    for _ in range(12):
        ei, ej = rng.choice(len(entries), 2)
        ei, ej = entries[ei], entries[ej]
        fi, lat = _orbit(sg, ei.letter, rng, None)
        fj, _ = _orbit(sg, ej.letter, rng, None)
        if len(fi) != ei.multiplicity or len(fj) != ej.multiplicity:
            continue  # a collapsed orbit
        di = np.sort(_distances(fi[0], fj, lat))
        if di[0] < 1e-3:
            continue                                   # orbits on top of each other
        dj = np.sort(_distances(fj[0], fi, lat))
        gaps = [
            (a + b) / 2 for a, b in zip(di[:-1], di[1:]) if b - a > 1e-4 and a > 1e-3
        ][:6]
        orders = bond_stabiliser_orders(sg, ei.letter, ej.letter)
        order = site_symmetry(sg, ei.letter).order
        for r in gaps:
            if r > 9.0:
                break
            n_ij = int(np.sum(di < r))
            n_ji = int(np.sum(dj < r))
            assert ei.multiplicity * n_ij == ej.multiplicity * n_ji, (
                sg,
                ei.letter,
                ej.letter,
                r,
            )
            assert n_ij in _semigroup_counts(orders, order, n_ij), (
                sg,
                ei.letter,
                ej.letter,
                r,
                n_ij,
            )
