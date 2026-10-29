"""Input errors stop at load time with one `ERROR:` line (ADR-0036).

Before ADR-0036 a bond or repulsion entry without a required key raised a
KeyError inside every candidate's optimisation, each reported as
"optimisation failed", and `run` exited 0 with error 1e10 everywhere; a lone
volume_min was ignored in favour of the automatic band.
"""

from pathlib import Path

import pytest
import yaml
from mizuho.errors import ConfigError

from mizuho.cli import main
from mizuho.objective.penalties import REQUIRED_ENTRY_KEYS, validate_constraint_entries
from mizuho.optimize import driver
from mizuho.search.dfs import wyckoff_tree_search

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

FULL_ENTRY = {
    "bond_constraints": {
        "atom1": "Na",
        "atom2": "Cl",
        "distance_min": 2.5,
        "distance_max": 3.0,
    },
    "coordination_constraints": {
        "center_atom": "Na",
        "neighbor_atom": "Cl",
        "cn_min": 6,
        "cn_max": 6,
        "cutoff_distance": 3.3,
    },
    "angle_constraints": {
        "atom_center": "Na",
        "atom_neighbor": "Cl",
        "angle_min": 80,
        "angle_max": 100,
    },
    "repulsion_constraints": {"atom1": "Na", "atom2": "Na", "distance_min": 3.5},
    "hybridisation_constraints": {"element": "Na", "hybridisation": "sp3"},
}


def test_every_list_has_a_complete_example():
    assert set(FULL_ENTRY) == set(REQUIRED_ENTRY_KEYS)
    for name, entry in FULL_ENTRY.items():
        validate_constraint_entries({name: [entry]})


@pytest.mark.parametrize(
    "name,key", [(n, k) for n, keys in REQUIRED_ENTRY_KEYS.items() for k in keys]
)
def test_a_missing_required_key_is_named(name, key):
    entry = {k: v for k, v in FULL_ENTRY[name].items() if k != key}
    with pytest.raises(ConfigError, match=rf"{name}\[1\] is missing {key}"):
        validate_constraint_entries({name: [FULL_ENTRY[name], entry]})


def test_every_problem_is_listed_at_once():
    with pytest.raises(ConfigError) as e:
        validate_constraint_entries(
            {
                "bond_constraints": [{"atom1": "Na"}],
                "volume_constraint": {"volume_max": 100.0},
            }
        )
    msg = str(e.value)
    assert "bond_constraints[0] is missing atom2, distance_min, distance_max" in msg
    assert "volume_constraint states only volume_max" in msg


@pytest.mark.parametrize(
    "vc,ok",
    [
        ({}, True),
        ({"weight": 2.0}, True),
        ({"volume_min": 1.0, "volume_max": 2.0}, True),
        ({"volume_min": 1.0}, False),
        ({"volume_max": 2.0}, False),
    ],
)
def test_volume_constraint_states_both_bounds_or_neither(vc, ok):
    if ok:
        validate_constraint_entries({"volume_constraint": vc})
    else:
        with pytest.raises(ConfigError, match="states only"):
            validate_constraint_entries({"volume_constraint": vc})


@pytest.mark.parametrize(
    "cfg",
    [
        {"bond_constraints": {"atom1": "Na"}},
        {"bond_constraints": ["Na-Cl"]},
        {"volume_constraint": [100, 200]},
    ],
)
def test_wrong_shapes_are_refused(cfg):
    with pytest.raises(ConfigError, match="must be a"):
        validate_constraint_entries(cfg)


@pytest.mark.parametrize(
    "path",
    sorted(FIXTURES.glob("*.yaml")) + sorted(FIXTURES.glob("auto/*.yaml")),
    ids=lambda p: p.stem,
)
def test_every_fixture_is_valid(path):
    validate_constraint_entries(yaml.safe_load(path.read_text()))


def test_config_error_is_a_value_error():
    """Callers that catch ValueError keep working."""
    assert issubclass(ConfigError, ValueError)


def _run(tmp_path, capsys, cfg):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(SystemExit) as e:
        main(["run", str(path), "--output", str(tmp_path / "out.vasp")])
    return e.value.code, capsys.readouterr().out


NACL = {
    "composition": "NaCl",
    "spacegroup": 225,
    "ionic_charges": {"Na": 1, "Cl": -1},
    "tree_search": {"max_sites": 2, "max_formula_units": 4},
}


def test_run_stops_on_a_missing_key(tmp_path, capsys):
    code, out = _run(
        tmp_path,
        capsys,
        dict(
            NACL, bond_constraints=[{"atom1": "Na", "atom2": "Cl", "distance_max": 3.0}]
        ),
    )
    assert code == 1
    assert "ERROR: bond_constraints[0] is missing distance_min" in out
    assert "Phase 1" not in out


def test_run_prints_a_derivation_error_before_searching(tmp_path, capsys):
    """coordination_numbers without an anion charge used to raise a bare
    ValueError traceback."""
    code, out = _run(
        tmp_path,
        capsys,
        {"composition": "NaCl", "spacegroup": 225, "coordination_numbers": {"Na": 6}},
    )
    assert code == 1
    assert (
        "ERROR: coordination_numbers needs ionic_charges with at least one anion" in out
    )
    assert "Phase 1" not in out


def _nacl_problem(nacl_config):
    asgn = wyckoff_tree_search(nacl_config)[0]
    return asgn, dict(nacl_config["optimization"], n_restarts=1, cma_maxiter=5)


@pytest.mark.parametrize(
    "exc", [KeyError("distance_min"), TypeError("bad"), ConfigError("bad input")]
)
def test_optimiser_lets_input_errors_and_defects_through(monkeypatch, nacl_config, exc):
    def broken(**kwargs):
        raise exc

    monkeypatch.setitem(driver._CORE_BACKENDS, "cma", broken)
    asgn, opt = _nacl_problem(nacl_config)
    with pytest.raises(type(exc)):
        driver.optimize_assignment(asgn, nacl_config, opt)


def test_optimiser_reports_a_numerical_failure(monkeypatch, nacl_config, capsys):
    """A failure that is not an input error or a defect still marks only
    this candidate as failed, and now says why even without --verbose."""

    def diverged(**kwargs):
        raise FloatingPointError("overflow in CMA-ES")

    monkeypatch.setitem(driver._CORE_BACKENDS, "cma", diverged)
    asgn, opt = _nacl_problem(nacl_config)
    assert driver.optimize_assignment(asgn, nacl_config, opt, label="Na@a + Cl@b") == []
    assert (
        "Optimisation failed for Na@a + Cl@b: FloatingPointError: overflow"
        in capsys.readouterr().out
    )
