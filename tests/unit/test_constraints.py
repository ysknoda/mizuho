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
    cns = {c["center_atom"]: (c["cn_min"], c["cn_max"]) for c in out["coordination_constraints"]}
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
