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
