"""Unit tests for mizuho.config."""

from mizuho.config import (
    _build_generic_lattice_bounds, _lattice_bounds_for_system,
    augment_coupling_constraints,
)


def test_lattice_bounds_for_system_keeps_user_bounds_when_complete():
    user_bounds = {"a": [4.0, 5.0], "c": [3.0, 4.0]}
    result = _lattice_bounds_for_system(user_bounds, "tetragonal")
    assert result == user_bounds


def test_lattice_bounds_for_system_falls_back_when_incomplete():
    """Regression guard: a cubic-only {"a": [...]} from the user's config
    must NOT be reused unmodified for a tetragonal/orthorhombic/...
    candidate space group during auto space-group search -- this
    previously reached structure.build_bounds() and raised a bare
    KeyError (e.g. `lb["c"]`) instead of falling back to generic
    bounds for that crystal system."""
    cubic_only = {"a": [4.5, 6.5]}
    result = _lattice_bounds_for_system(cubic_only, "tetragonal")
    assert result == _build_generic_lattice_bounds("tetragonal")
    assert "c" in result


def test_lattice_bounds_for_system_falls_back_when_empty():
    result = _lattice_bounds_for_system({}, "orthorhombic")
    assert result == _build_generic_lattice_bounds("orthorhombic")


def test_lattice_bounds_for_system_cubic_is_always_sufficient():
    # A cubic-only bounds dict is complete for a cubic SG itself.
    cubic_only = {"a": [4.5, 6.5]}
    result = _lattice_bounds_for_system(cubic_only, "cubic")
    assert result == cubic_only


# ---------------------------------------------------------------------------
# augment_coupling_constraints: see mizuho.config.augment_coupling_constraints's
# docstring. Injects a default anion-centred coordination_constraint (+
# angle_constraint, wildcard "X" neighbour) whenever an existing CATION-
# centred constraint already references that anion as a neighbour but
# nothing constrains the anion itself -- otherwise cation polyhedra sharing
# that anion have nothing coupling them together (confirmed empirically
# this session: best error 15.04 -> 8.42 for adamite after adding exactly
# this).
# ---------------------------------------------------------------------------

def _rutile_like_config():
    return {
        "composition": "TiO2",
        "spacegroup": 136,
        "ionic_charges": {"Ti": 4, "O": -2},
        "bond_constraints": [
            {"atom1": "Ti", "atom2": "O", "distance_min": 1.85, "distance_max": 2.05},
        ],
        "coordination_constraints": [
            {"center_atom": "Ti", "neighbor_atom": "O",
             "cn_min": 6, "cn_max": 6, "cutoff_distance": 2.2},
        ],
    }


def test_coupling_triggers_for_cation_constraint_referencing_anion():
    cfg = _rutile_like_config()
    new_cfg = augment_coupling_constraints(cfg)
    assert new_cfg is not cfg  # new dict returned, something was injected

    coord = new_cfg["coordination_constraints"]
    assert any(c["center_atom"] == "O" and c["neighbor_atom"] == "X"
               for c in coord)
    angle = new_cfg["angle_constraints"]
    assert any(a["atom_center"] == "O" and a["atom_neighbor"] == "X"
               for a in angle)

    # Ti's own original constraint must be untouched.
    assert cfg["coordination_constraints"][0] in coord


def test_coupling_no_trigger_when_no_coordination_or_angle_constraints():
    # nacl/cscl-style: ionic_charges present, but nothing references O
    # as a neighbour of any cation-centred constraint -- nothing to
    # couple, so this must be a true no-op (same object back).
    cfg = {
        "composition": "NaCl",
        "spacegroup": 225,
        "ionic_charges": {"Na": 1, "Cl": -1},
        "bond_constraints": [
            {"atom1": "Na", "atom2": "Cl", "distance_min": 2.5, "distance_max": 3.2},
        ],
    }
    assert augment_coupling_constraints(cfg) is cfg


def test_coupling_no_trigger_without_ionic_charges():
    # diamond_si-style: covalent, no ionic_charges at all.
    cfg = {"composition": "Si", "spacegroup": 227}
    assert augment_coupling_constraints(cfg) is cfg


def test_coupling_opt_out_via_auto_anion_coupling_false():
    cfg = _rutile_like_config()
    cfg["auto_anion_coupling"] = False
    assert augment_coupling_constraints(cfg) is cfg


def test_coupling_user_supplied_anion_constraint_suppresses_injection():
    cfg = _rutile_like_config()
    cfg["coordination_constraints"].append({
        "center_atom": "O", "neighbor_atom": "Ti",
        "cn_min": 3, "cn_max": 3, "cutoff_distance": 2.2,
    })
    new_cfg = augment_coupling_constraints(cfg)
    # O already has its own constraint -- must not get a second,
    # auto-injected one.
    assert new_cfg is cfg
    o_constraints = [c for c in cfg["coordination_constraints"]
                     if c["center_atom"] == "O"]
    assert len(o_constraints) == 1


def test_coupling_angle_constraint_can_be_disabled_independently():
    cfg = _rutile_like_config()
    cfg["auto_anion_coupling_angle"] = False
    new_cfg = augment_coupling_constraints(cfg)
    assert any(c["center_atom"] == "O" for c in new_cfg["coordination_constraints"])
    assert not any(a["atom_center"] == "O" for a in new_cfg["angle_constraints"])


def test_coupling_cutoff_derived_from_existing_bond_constraint():
    cfg = _rutile_like_config()
    new_cfg = augment_coupling_constraints(cfg)
    o_cst = next(c for c in new_cfg["coordination_constraints"]
                 if c["center_atom"] == "O")
    assert o_cst["cutoff_distance"] == 2.05  # Ti-O distance_max


def test_coupling_defaults_are_overridable():
    cfg = _rutile_like_config()
    cfg["auto_coupling_cn_range"] = [3, 5]
    cfg["auto_coupling_angle_range"] = [90.0, 150.0]
    cfg["auto_coupling_weight"] = 2.5
    new_cfg = augment_coupling_constraints(cfg)
    o_cst = next(c for c in new_cfg["coordination_constraints"]
                 if c["center_atom"] == "O")
    assert (o_cst["cn_min"], o_cst["cn_max"]) == (3, 5)
    assert o_cst["weight"] == 2.5
    o_angle = next(a for a in new_cfg["angle_constraints"]
                   if a["atom_center"] == "O")
    assert (o_angle["angle_min"], o_angle["angle_max"]) == (90.0, 150.0)


# ---------------------------------------------------------------------------
# auto_coupling_exclude_hh: "X" on an anion-centred angle_constraint now
# resolves by charge sign (see objective.penalties._expand_wildcard), so it
# legitimately includes H alongside the framework cations -- but an H-H
# angle pair models a water molecule, not the cation-anion-cation bridging
# geometry this auto-injected constraint targets, so it is excluded by
# default whenever H is one of the coupled cations.
# ---------------------------------------------------------------------------

def _adamite_like_config():
    return {
        "composition": "O5Zn2AsH",
        "spacegroup": 58,
        "ionic_charges": {"Zn": 2, "As": 5, "O": -2, "H": 1},
        "bond_constraints": [
            {"atom1": "As", "atom2": "O", "distance_min": 1.60, "distance_max": 1.75},
            {"atom1": "Zn", "atom2": "O", "distance_min": 1.90, "distance_max": 2.15},
        ],
        "coordination_constraints": [
            {"center_atom": "As", "neighbor_atom": "O",
             "cn_min": 4, "cn_max": 4, "cutoff_distance": 1.9},
            {"center_atom": "Zn", "neighbor_atom": "O",
             "cn_min": 5, "cn_max": 6, "cutoff_distance": 2.3},
        ],
    }


def test_coupling_injects_hh_exclusion_when_hydrogen_present():
    cfg = _adamite_like_config()
    new_cfg = augment_coupling_constraints(cfg)
    o_angle = next(a for a in new_cfg["angle_constraints"]
                   if a["atom_center"] == "O")
    assert o_angle.get("exclude_neighbor_pairs") == [["H", "H"]]


def test_coupling_no_hh_exclusion_when_hydrogen_absent():
    # rutile_tio2-style: no H in the composition at all -- the H-H
    # exclusion default must not appear (it would be meaningless noise).
    cfg = _rutile_like_config()
    new_cfg = augment_coupling_constraints(cfg)
    o_angle = next(a for a in new_cfg["angle_constraints"]
                   if a["atom_center"] == "O")
    assert "exclude_neighbor_pairs" not in o_angle


def test_coupling_hh_exclusion_opt_out():
    cfg = _adamite_like_config()
    cfg["auto_coupling_exclude_hh"] = False
    new_cfg = augment_coupling_constraints(cfg)
    o_angle = next(a for a in new_cfg["angle_constraints"]
                   if a["atom_center"] == "O")
    assert "exclude_neighbor_pairs" not in o_angle
