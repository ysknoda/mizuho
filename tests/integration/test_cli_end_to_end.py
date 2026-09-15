"""CLI smoke test: invokes the installed `mizuho` console script directly
via subprocess, validating the packaging/entry-point wiring itself.
"""

import subprocess
import sys
from pathlib import Path

import yaml

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "mizuho", "--help"],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0
    assert "input" in result.stdout.lower()


def test_cli_nacl_end_to_end(tmp_path):
    out_path = tmp_path / "nacl.vasp"
    result = subprocess.run(
        [sys.executable, "-m", "mizuho",
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
        [sys.executable, "-m", "mizuho", str(config_path),
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
        [sys.executable, "-m", "mizuho", str(FIXTURES_DIR / "nacl.yaml"),
         "--output", str(seq_out), "--seed", "42"],
        capture_output=True, text=True, timeout=120)
    assert seq.returncode == 0, seq.stderr
    assert "in parallel" not in seq.stdout

    par = subprocess.run(
        [sys.executable, "-m", "mizuho", str(FIXTURES_DIR / "nacl.yaml"),
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
        [sys.executable, "-m", "mizuho", str(config_path)],
        capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "auto_spacegroup" in result.stdout
