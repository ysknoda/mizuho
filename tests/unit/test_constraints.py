"""Constraints derived from cited coordination numbers (ADR-0028)."""

import math

import pytest

from mizuho.constraints import BAND_VU, bond_band, derive_constraints


def _silicate(**extra):
    cfg = {"composition": "Mg2SiO4", "ionic_charges": {"Mg": 2, "Si": 4, "O": -2},
           "coordination_numbers": {"Si": 4, "Mg": 6}}
    cfg.update(extra)
    return cfg


def test_bond_band_is_the_bond_valence_band():
    """Si-O at CN 4: s0 = 1, R0 = 1.624 (Brese & O'Keeffe), band s0 +- 0.16
    v.u., cutoff 0.3 A past the long end."""
    b = bond_band("Si", 4, "O", 4, 4)
    assert b["distance_min"] == pytest.approx(1.624 - 0.37 * math.log(1 + BAND_VU), abs=1e-3)
    assert b["distance_max"] == pytest.approx(1.624 - 0.37 * math.log(1 - BAND_VU), abs=1e-3)
    assert b["cutoff"] == pytest.approx(b["distance_max"] + 0.3, abs=1e-3)


def test_derived_constraints_cover_every_cation_and_tolerance():
    out = derive_constraints(_silicate())
    assert {(b["atom1"], b["atom2"]) for b in out["bond_constraints"]} == {("Mg", "O"), ("Si", "O")}
    cns = {c["center_atom"]: (c["cn_min"], c["cn_max"]) for c in out["coordination_constraints"]
           if c["center_atom"] != "O"}
    assert cns == {"Mg": (6, 6), "Si": (4, 4)}
    assert out["bond_valence"]["tolerance"] == {"Mg": 0.2, "Si": 0.4}
    assert set(out["auto_coupling_cutoff"]) == {"O"}


def test_stated_constraints_win():
    stated = {"atom1": "Si", "atom2": "O", "distance_min": 1.6, "distance_max": 1.65, "cutoff": 2.4}
    out = derive_constraints(_silicate(bond_constraints=[stated],
                                       bond_valence={"tolerance": {"Si": 0.17}}))
    si_o = [b for b in out["bond_constraints"] if b["atom1"] == "Si"]
    assert si_o == [stated]
    assert out["bond_valence"]["tolerance"]["Si"] == 0.17


def test_without_the_key_nothing_changes():
    cfg = {"composition": "SiO2", "ionic_charges": {"Si": 4, "O": -2}}
    assert derive_constraints(cfg) is cfg


def test_low_bond_valence_is_outside_the_validated_range():
    cfg = {"composition": "NaCl", "ionic_charges": {"Na": 1, "Cl": -1},
           "coordination_numbers": {"Na": 6}}
    with pytest.raises(ValueError, match="0.25"):
        derive_constraints(cfg)


def test_hydroxyl_rule_and_its_opt_out():
    cfg = {"composition": "Zn2AsO5H", "ionic_charges": {"Zn": 2, "As": 5, "O": -2, "H": 1},
           "coordination_numbers": {"Zn": [5, 6], "As": 4, "H": 1}}
    out = derive_constraints(cfg)
    assert any(c["center_atom"] == "O" and c["neighbor_atom"] == "H" and c["cn_max"] == 1
               for c in out["coordination_constraints"])
    off = derive_constraints(dict(cfg, auto_hydroxyl=False))
    assert not any(c["neighbor_atom"] == "H" for c in off["coordination_constraints"])


def test_polyhedron_angle_bands_only_for_cn_4_and_6():
    cfg = {"composition": "CaMgSi2O6", "ionic_charges": {"Ca": 2, "Mg": 2, "Si": 4, "O": -2},
           "coordination_numbers": {"Si": 4, "Mg": 6, "Ca": [7, 8]}}
    angles = {a["atom_center"]: (a["angle_min"], a["angle_max"])
              for a in derive_constraints(cfg)["angle_constraints"]}
    assert angles == {"Si": (95.0, 125.0), "Mg": (70.0, 110.0)}


def test_anion_cn_and_polymerisation_are_counted_from_the_composition():
    """Diopside: (4*2 + 6 + 7.5) bonds over 6 O -> O CN in [3, 4]; 8 Si-O
    bonds over 6 O -> O(Si) in [1, 2] (ADR-0007's {1: 16, 2: 8}). The
    derived anion constraint replaces the generic auto [2, 4]."""
    from mizuho.config import augment_coupling_constraints
    cfg = {"composition": "CaMgSi2O6", "ionic_charges": {"Ca": 2, "Mg": 2, "Si": 4, "O": -2},
           "coordination_numbers": {"Si": 4, "Mg": 6, "Ca": [7, 8]}}
    out = augment_coupling_constraints(derive_constraints(cfg))
    o = {tuple(sorted(c["neighbor_atom"])) if isinstance(c["neighbor_atom"], list)
         else (c["neighbor_atom"],): (c["cn_min"], c["cn_max"])
         for c in out["coordination_constraints"] if c["center_atom"] == "O"}
    assert o == {("Ca", "Mg", "Si"): (3, 4), ("Si",): (1, 2)}


def _topaz(**extra):
    cfg = {"composition": "Al2SiO4F2", "ionic_charges": {"Al": 3, "Si": 4, "O": -2, "F": -1},
           "coordination_numbers": {"Al": 6, "Si": 4}}
    cfg.update(extra)
    return cfg


def test_bonded_anions_restrict_a_cation_to_its_anions():
    """ADR-0028 (a): Si bonds O only, so Si gets no Si-F band, its CN and
    angles run over O, and F is held outside the Si shell at CN 0."""
    out = derive_constraints(_topaz(bonded_anions={"Si": ["O"]}))
    pairs = {(b["atom1"], b["atom2"]) for b in out["bond_constraints"]}
    assert pairs == {("Al", "F"), ("Al", "O"), ("Si", "O")}
    si = {c["label"]: c for c in out["coordination_constraints"] if c["center_atom"] == "Si"}
    assert si["Si(X) [derived]"]["neighbor_atom"] == "O"
    assert (si["Si(F) [derived, not bonded]"]["cn_min"],
            si["Si(F) [derived, not bonded]"]["cn_max"]) == (0, 0)
    angle = [a for a in out["angle_constraints"] if a["atom_center"] == "Si"]
    assert angle[0]["atom_neighbor"] == "O"
    # Polymerisation over the anions Si bonds: 4 Si-O bonds over 4 O.
    poly = [c for c in out["coordination_constraints"] if "polymerisation" in c["label"]]
    assert [(c["center_atom"], c["cn_min"], c["cn_max"]) for c in poly] == [("O", 1, 1)]


def test_without_bonded_anions_a_cation_bonds_every_anion():
    out = derive_constraints(_topaz())
    assert ("Si", "F") in {(b["atom1"], b["atom2"]) for b in out["bond_constraints"]}
    poly = [c for c in out["coordination_constraints"] if "polymerisation" in c["label"]]
    assert {c["center_atom"] for c in poly} == {"O", "F"}


def test_bonded_anions_must_name_anions_present():
    with pytest.raises(ValueError, match="bonded_anions"):
        derive_constraints(_topaz(bonded_anions={"Si": ["Cl"]}))


def _hydroxide(**extra):
    cfg = {"composition": "Zn2AsO5H", "ionic_charges": {"Zn": 2, "As": 5, "O": -2, "H": 1},
           "coordination_numbers": {"As": 4, "Zn": [5, 6], "H": 1}}
    cfg.update(extra)
    return cfg


def test_hydrogen_bond_angle_band_and_its_opt_out():
    """ADR-0028 (b): an acceptor within 2.2 A of H sits at 130 deg or more
    from the O-H bond (Jeffrey 1997, via Steiner 2002)."""
    out = derive_constraints(_hydroxide())
    hb = [a for a in out["angle_constraints"] if a["atom_center"] == "H"]
    assert len(hb) == 1
    assert (hb[0]["angle_min"], hb[0]["angle_max"], hb[0]["angle_cutoff"]) == (130.0, 180.0, 2.2)
    off = derive_constraints(_hydroxide(auto_hydrogen_bond=False))
    assert not [a for a in off["angle_constraints"] if a["atom_center"] == "H"]


def test_ideal_bond_config_shrinks_only_derived_bands():
    """ADR-0028 (c): derived bonds carry distance_ideal (the bond at s0),
    which centring shrinks the band to; a stated bond is left alone."""
    from mizuho.constraints import ideal_bond_config
    stated = {"atom1": "Si", "atom2": "O", "distance_min": 1.6, "distance_max": 1.65, "cutoff": 2.4}
    out = derive_constraints(_silicate(bond_constraints=[stated]))
    ideal = {b["atom1"]: b for b in ideal_bond_config(out)["bond_constraints"]}
    mg = bond_band("Mg", 2, "O", 6, 6)
    assert ideal["Mg"]["distance_min"] == ideal["Mg"]["distance_max"] == mg["distance_ideal"]
    assert (ideal["Si"]["distance_min"], ideal["Si"]["distance_max"]) == (1.6, 1.65)
    assert ideal_bond_config({"bond_constraints": [stated]}) is None


def _alon(**extra):
    cfg = {"composition": "Al3O3N", "ionic_charges": {"Al": 3, "O": -2, "N": -3},
           "coordination_numbers": {"Al": [4, 6]}}
    cfg.update(extra)
    return cfg


def _anion_cn(out):
    return {c["center_atom"]: (c["cn_min"], c["cn_max"], c["cutoff_distance"])
            for c in out["coordination_constraints"] if c["center_atom"] in ("O", "N")}


def test_anion_coordination_numbers_split_the_combined_range():
    """ADR-0033: the mean m fixes only the anions' combined count (Al3O3N
    at Al CN 4-6: m = 3-4.5, so [3, 5] for both); the stated ranges replace
    it per anion, at the derived cutoffs."""
    plain = _anion_cn(derive_constraints(_alon()))
    assert plain["O"][:2] == plain["N"][:2] == (3, 5)
    split = _anion_cn(derive_constraints(_alon(
        anion_coordination_numbers={"O": [2, 4], "N": [3, 4]})))
    assert split["O"][:2] == (2, 4) and split["N"][:2] == (3, 4)
    assert split["O"][2] == plain["O"][2] and split["N"][2] == plain["N"][2]


def test_anion_coordination_numbers_names_anions_only():
    with pytest.raises(ValueError, match="not anions"):
        derive_constraints(_alon(anion_coordination_numbers={"Al": 4}))


# ---------------------------------------------------------------------------
# ADR-0037: like-charge minimum distances
# ---------------------------------------------------------------------------

def _repulsions(out):
    return {frozenset((r["atom1"], r["atom2"])): r["distance_min"]
            for r in out.get("repulsion_constraints", [])}


def test_shared_ion_distance_is_the_minimum_over_longer_bonds_and_wider_angles():
    """Brute force over bonds no shorter than a, b and angles no smaller
    than theta: the closed form is that grid's minimum, never above it."""
    import numpy as np
    from mizuho.constraints import _shared_ion_distance
    rng = np.random.default_rng(0)
    for _ in range(200):
        a, b = rng.uniform(0.8, 2.6, 2)
        theta = rng.uniform(50.0, 120.0)
        d1 = np.linspace(a, a + 3, 301)[:, None, None]
        d2 = np.linspace(b, b + 3, 301)[None, :, None]
        t = np.radians(np.linspace(theta, 180.0, 61))[None, None, :]
        grid = np.sqrt(d1 ** 2 + d2 ** 2 - 2 * d1 * d2 * np.cos(t)).min()
        got = _shared_ion_distance(a, b, theta)
        assert got <= grid + 1e-9
        assert got == pytest.approx(grid, abs=2e-3)


def test_every_like_charge_pair_gets_a_derived_repulsion():
    """Al-O-N (Al3O3N): O-O, N-O, N-N through Al (CN up to 6: 90 - 20 = 70 deg) and
    Al-Al through either anion (the auto anion band's 80 deg), both bonds
    at their band's short end."""
    out = derive_constraints(_alon(auto_repulsion=True))
    reps = _repulsions(out)
    assert set(reps) == {frozenset(p) for p in (("O",), ("N",), ("N", "O"), ("Al",))}
    short = {b["atom2"]: b["distance_min"] for b in out["bond_constraints"]}
    s70, s80 = math.sin(math.radians(35.0)), math.sin(math.radians(40.0))
    assert reps[frozenset(("O",))] == pytest.approx(2 * short["O"] * s70, abs=1e-3)
    assert reps[frozenset(("N",))] == pytest.approx(2 * short["N"] * s70, abs=1e-3)
    a, b, c = short["N"], short["O"], math.cos(math.radians(70.0))
    assert reps[frozenset(("N", "O"))] == pytest.approx(math.sqrt(a * a + b * b - 2 * a * b * c),
                                                       abs=1e-3)
    # the smaller of the two shared anions' bounds: O, whose Al-O bond is shorter
    assert reps[frozenset(("Al",))] == pytest.approx(2 * min(short.values()) * s80, abs=1e-3)


def test_mixed_cation_pairs_and_cn_specific_angle_floors():
    """Forsterite: Mg-Mg, Mg-Si, Si-Si through O at 80 deg; O-O at the
    weaker of Si (CN 4: 109.47 - 20) and Mg (CN 6: 90 - 20)."""
    out = derive_constraints(_silicate(auto_repulsion=True))
    reps = _repulsions(out)
    assert set(reps) == {frozenset(p) for p in (("O",), ("Mg",), ("Si",), ("Mg", "Si"))}
    short = {b["atom1"]: b["distance_min"] for b in out["bond_constraints"]}
    oo = min(2 * short["Si"] * math.sin(math.radians(89.47 / 2)),
             2 * short["Mg"] * math.sin(math.radians(35.0)))
    assert reps[frozenset(("O",))] == pytest.approx(oo, abs=1e-3)
    a, b, c = short["Mg"], short["Si"], math.cos(math.radians(80.0))
    assert reps[frozenset(("Mg", "Si"))] == pytest.approx(math.sqrt(a * a + b * b - 2 * a * b * c),
                                                         abs=1e-3)


def test_a_stated_repulsion_wins_for_its_pair_only():
    stated = {"atom1": "O", "atom2": "N", "distance_min": 2.4, "weight": 1.0}
    out = derive_constraints(_alon(auto_repulsion=True, repulsion_constraints=[stated]))
    no = [r for r in out["repulsion_constraints"] if {r["atom1"], r["atom2"]} == {"N", "O"}]
    assert no == [stated]
    assert {frozenset((r["atom1"], r["atom2"])) for r in out["repulsion_constraints"]} == \
        {frozenset(p) for p in (("O",), ("N",), ("N", "O"), ("Al",))}


def test_repulsions_are_opt_in():
    assert "repulsion_constraints" not in derive_constraints(_alon())
    assert "repulsion_constraints" not in derive_constraints(_alon(auto_repulsion=False))
    stated = [{"atom1": "Al", "atom2": "Al", "distance_min": 2.6}]
    out = derive_constraints(_alon(repulsion_constraints=stated))
    assert out["repulsion_constraints"] == stated


def test_anion_band_settings_set_the_cation_angle_floor():
    """The cation-cation floor follows the anion-centred band: its lower
    bound when overridden, 60 deg when the band is switched off."""
    short = min(b["distance_min"] for b in derive_constraints(_alon(auto_repulsion=True))["bond_constraints"])
    for extra, angle in (({"auto_coupling_angle_range": [90.0, 160.0]}, 90.0),
                         ({"auto_anion_coupling_angle": False}, 60.0)):
        reps = _repulsions(derive_constraints(_alon(auto_repulsion=True, **extra)))
        assert reps[frozenset(("Al",))] == pytest.approx(
            2 * short * math.sin(math.radians(angle / 2)), abs=1e-3)


def test_hydrogen_pairs_meet_at_the_unbanded_floor_and_no_anion_pair_meets_at_h():
    """H at CN 1 bonds one anion, so no anion pair is bounded through it;
    the auto anion band names the framework cations only, so pairs with H
    take the 60 deg floor."""
    from mizuho.constraints import _shared_ion_distance
    cfg = {"composition": "Zn2AsO5H", "ionic_charges": {"Zn": 2, "As": 5, "O": -2, "H": 1},
           "coordination_numbers": {"Zn": [5, 6], "As": 4, "H": 1}, "auto_repulsion": True}
    out = derive_constraints(cfg)
    reps = _repulsions(out)
    short = {b["atom1"]: b["distance_min"] for b in out["bond_constraints"]}
    assert reps[frozenset(("H", "Zn"))] == pytest.approx(
        _shared_ion_distance(short["H"], short["Zn"], 60.0), abs=1e-3)
    assert reps[frozenset(("O",))] == pytest.approx(
        min(2 * short["As"] * math.sin(math.radians(89.47 / 2)),
            2 * short["Zn"] * math.sin(math.radians(35.0))), abs=1e-3)
