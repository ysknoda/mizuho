"""CLI smoke test: invokes the installed `mizuho` console script directly
via subprocess, validating the packaging/entry-point wiring itself.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "mizuho", "--help"],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0
    for command in ("run", "check", "fit", "candidates", "wyckoff", "suggest"):
        assert command in result.stdout


def test_cli_nacl_end_to_end(tmp_path):
    out_path = tmp_path / "nacl.vasp"
    result = subprocess.run(
        [sys.executable, "-m", "mizuho", "run",
         str(FIXTURES_DIR / "nacl.yaml"),
         "--output", str(out_path), "--seed", "42"],
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert out_path.exists()
    content = out_path.read_text()
    assert "Na" in content
    assert "Cl" in content


def test_cli_mode_single_spacegroup_explicit_matches_implicit(tmp_path):
    """mode: single_spacegroup is an optional, explicit spelling of
    "spacegroup is given" -- it must not change behaviour."""
    with open(FIXTURES_DIR / "nacl.yaml") as fh:
        config = yaml.safe_load(fh)
    config["mode"] = "single_spacegroup"
    config_path = tmp_path / "nacl_explicit_mode.yaml"
    with open(config_path, "w") as fh:
        yaml.safe_dump(config, fh)

    out_path = tmp_path / "nacl.vasp"
    result = subprocess.run(
        [sys.executable, "-m", "mizuho", "run", str(config_path),
         "--output", str(out_path), "--seed", "42"],
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert out_path.exists()


def test_cli_workers_flag_matches_sequential_result(tmp_path):
    """Phase H-1: --workers must not change WHICH structure is found.
    Every Phase 2 candidate shares the SAME opt_cfg['seed'] regardless
    of process/dispatch order (see optimize.parallel's module
    docstring), so the parallel run must be bit-identical to the
    sequential one, and must actually take the parallel code path
    (not silently fall back)."""
    seq_out = tmp_path / "nacl_seq.vasp"
    par_out = tmp_path / "nacl_par.vasp"

    seq = subprocess.run(
        [sys.executable, "-m", "mizuho", "run", str(FIXTURES_DIR / "nacl.yaml"),
         "--output", str(seq_out), "--seed", "42"],
        capture_output=True, text=True, timeout=120)
    assert seq.returncode == 0, seq.stderr
    assert "in parallel" not in seq.stdout

    par = subprocess.run(
        [sys.executable, "-m", "mizuho", "run", str(FIXTURES_DIR / "nacl.yaml"),
         "--output", str(par_out), "--seed", "42", "--workers", "2"],
        capture_output=True, text=True, timeout=120)
    assert par.returncode == 0, par.stderr
    assert "in parallel (2 workers)" in par.stdout

    assert seq_out.read_text() == par_out.read_text()


def test_cli_mode_conflicting_with_spacegroup_is_rejected(tmp_path):
    """mode: auto_spacegroup while "spacegroup" is also set is a
    contradictory config and must be rejected with a clear error,
    not silently pick one interpretation."""
    with open(FIXTURES_DIR / "nacl.yaml") as fh:
        config = yaml.safe_load(fh)
    config["mode"] = "auto_spacegroup"  # nacl.yaml already sets spacegroup: 225
    config_path = tmp_path / "nacl_conflicting_mode.yaml"
    with open(config_path, "w") as fh:
        yaml.safe_dump(config, fh)

    result = subprocess.run(
        [sys.executable, "-m", "mizuho", "run", str(config_path)],
        capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "auto_spacegroup" in result.stdout


def test_cli_without_spacegroup_requires_opting_in(tmp_path):
    """ADR-0021: the automatic space-group search has not reached its known
    answers, so omitting "spacegroup" is an error unless the config opts in
    with mode: auto_spacegroup."""
    with open(FIXTURES_DIR / "nacl.yaml") as fh:
        config = yaml.safe_load(fh)
    del config["spacegroup"]
    config_path = tmp_path / "nacl_no_spacegroup.yaml"
    with open(config_path, "w") as fh:
        yaml.safe_dump(config, fh)

    result = subprocess.run(
        [sys.executable, "-m", "mizuho", "run", str(config_path)],
        capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "mode: auto_spacegroup" in result.stdout
    assert "adr/0021" in result.stdout


def test_cli_auto_spacegroup_opt_in_warns(tmp_path):
    """The opt-in passes the mode check and prints that the search is
    experimental. Checked on _check_mode alone: a full automatic search
    takes minutes."""
    from mizuho.cli import AUTO_SPACEGROUP_WARNING, _check_mode
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _check_mode({"mode": "auto_spacegroup", "composition": "NaCl"})
    assert AUTO_SPACEGROUP_WARNING in buf.getvalue()


def test_exhaustive_strategy_runs_every_group_or_the_named_subset():
    """ADR-0026: `spacegroup_search.strategy: exhaustive` runs every group in
    mizuho's tables, or the subset `spacegroups` names."""
    from mizuho.cli import _exhaustive_spacegroups
    from mizuho.data import WYCKOFF_DB

    assert _exhaustive_spacegroups({}) == sorted(WYCKOFF_DB)
    assert len(_exhaustive_spacegroups({})) == 230
    cfg = {"spacegroup_search": {"strategy": "exhaustive", "spacegroups": [225, 99, 225]}}
    assert _exhaustive_spacegroups(cfg) == [99, 225]


def test_unknown_spacegroup_strategy_is_rejected(tmp_path):
    with open(FIXTURES_DIR / "auto" / "nacl.yaml") as fh:
        config = yaml.safe_load(fh)
    config["spacegroup_search"] = {"strategy": "everything"}
    config_path = tmp_path / "nacl_bad_strategy.yaml"
    with open(config_path, "w") as fh:
        yaml.safe_dump(config, fh)
    result = subprocess.run(
        [sys.executable, "-m", "mizuho", "run", str(config_path)],
        capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "exhaustive" in result.stdout


def test_ranking_prefers_the_higher_group_among_satisfied_entries():
    """ADR-0026: at or below the threshold, symmetry first (Fm-3m 48 over
    P4mm 8) even against a lower error; above it, error alone."""
    from mizuho.cli import _ranking_key

    entries = [(1e-12, 99), (0.0, 1), (5e-4, 225), (0.02, 225), (0.01, 99)]
    order = sorted(entries, key=lambda e: _ranking_key(e[0], e[1], 1e-3))
    assert order == [(5e-4, 225), (1e-12, 99), (0.0, 1), (0.01, 99), (0.02, 225)]


@pytest.mark.slow
def test_exhaustive_nacl_over_99_and_225_returns_fm3m(tmp_path):
    """Restricted to P4mm and Fm-3m, the exhaustive strategy's answer is
    detected as Fm-3m (225): measured 2026-10-14, error 0 in 225 against
    0.036 at best in 99."""
    from mizuho.io.cif import read_cif
    from mizuho.symmetry import SUCCESS_TOLERANCE, detect_spacegroup

    with open(FIXTURES_DIR / "auto" / "nacl.yaml") as fh:
        config = yaml.safe_load(fh)
    config["spacegroup_search"] = {"strategy": "exhaustive", "spacegroups": [99, 225]}
    config_path = tmp_path / "nacl_exhaustive.yaml"
    with open(config_path, "w") as fh:
        yaml.safe_dump(config, fh)
    out = tmp_path / "nacl.cif"
    result = subprocess.run(
        [sys.executable, "-m", "mizuho", "run", str(config_path), "--format", "cif",
         "--output", str(out), "--seed", "42"],
        capture_output=True, text=True, timeout=3600)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    lat, frac, species = read_cif(out).expand()
    assert detect_spacegroup(lat, frac, species, SUCCESS_TOLERANCE)[0] == 225


# ---------------------------------------------------------------------------
# Subcommands (docs/adr/0013-cli-subcommands.md)
# ---------------------------------------------------------------------------

REFERENCE_DIR = FIXTURES_DIR / "reference"


def _mizuho(*args, timeout=120):
    return subprocess.run([sys.executable, "-m", "mizuho", *map(str, args)],
                          capture_output=True, text=True, timeout=timeout)


def test_bare_yaml_argument_is_not_an_implicit_run():
    """No compatibility special case: `mizuho input.yaml` stops working."""
    result = _mizuho(FIXTURES_DIR / "nacl.yaml", timeout=30)
    assert result.returncode != 0
    assert "invalid choice" in result.stderr


def test_run_writes_cif_that_reads_back(tmp_path):
    from mizuho.io.cif import cif_to_mizuho
    out = tmp_path / "nacl.cif"
    result = _mizuho("run", FIXTURES_DIR / "nacl.yaml", "--format", "cif",
                     "--output", out, "--seed", "42")
    assert result.returncode == 0, result.stderr
    m = cif_to_mizuho(out)
    assert m.spacegroup == 225
    assert sorted(m.label().split()) == ["Cl@b", "Na@a"]
    # The run report states the detected symmetry ladder.
    assert "Detected space group by tolerance" in result.stdout


def test_check_reports_objective_breakdown_and_symmetry():
    result = _mizuho("check", FIXTURES_DIR / "wurtzite_zno.yaml",
                     REFERENCE_DIR / "wurtzite.cif", timeout=60)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "Zn@b O@b" in out
    assert "Objective   : 0.0000" in out
    assert "integer [4]" in out
    assert "186 (P6_3mc)" in out


def test_check_rejects_removed_config_key(tmp_path):
    config = yaml.safe_load(open(FIXTURES_DIR / "nacl.yaml"))
    config["normalize_penalties"] = False
    path = tmp_path / "old.yaml"
    path.write_text(yaml.safe_dump(config))
    result = _mizuho("run", path, timeout=30)
    assert result.returncode != 0
    assert "0009" in result.stdout


def test_fit_compares_with_the_reference(tmp_path):
    out = tmp_path / "fit.cif"
    result = _mizuho("fit", FIXTURES_DIR / "wurtzite_zno.yaml",
                     REFERENCE_DIR / "wurtzite.cif", "--seed", "1", "--output", out,
                     timeout=300)
    assert result.returncode == 0, result.stderr
    assert "Objective at reference" in result.stdout
    assert "lattice within 3% of reference" in result.stdout
    assert out.exists()


def test_candidates_lists_merged_classes_with_strata():
    result = _mizuho("candidates", FIXTURES_DIR / "rutile_tio2.yaml", "--seeds", "2",
                     timeout=120)
    assert result.returncode == 0, result.stderr
    assert "equivalence classes" in result.stdout
    assert "free-parameter strata" in result.stdout
    assert "spread" in result.stdout


def test_candidates_find_accepts_any_representative():
    a = _mizuho("candidates", FIXTURES_DIR / "rutile_tio2.yaml", "--find", "Ti@a O@f")
    b = _mizuho("candidates", FIXTURES_DIR / "rutile_tio2.yaml", "--find", "Ti@b O@g")
    assert a.returncode == 0 and b.returncode == 0, a.stderr + b.stderr
    assert a.stdout.splitlines()[-1] == b.stdout.splitlines()[-1]


def test_wyckoff_lists_positions_and_equivalences():
    result = _mizuho("wyckoff", "227", timeout=60)
    assert result.returncode == 0, result.stderr
    assert "setting choice '1'" in result.stdout
    line_a = next(l for l in result.stdout.splitlines() if l.split()[:1] == ["a"])
    assert line_a.split()[-1] == "b"


def test_suggest_is_reserved():
    result = _mizuho("suggest", REFERENCE_DIR / "wurtzite.cif", timeout=30)
    assert result.returncode == 2
    assert "reserved" in result.stdout
