"""Known-answer tests for the automatic space-group search (the PUCT tree
search, the default). Judged on the DETECTED symmetry of the returned
structure at the measured success tolerance, not on the label the search
reports (docs/adr/0010-success-is-the-detected-symmetry.md).

Minutes per compound, so they run with the other search gates:
`pytest --run-gates tests/integration/test_auto_spacegroup_known_answers.py`.
Fixtures and the provenance of their constraints: tests/fixtures/auto/.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from mizuho.io.cif import read_cif
from mizuho.symmetry import SUCCESS_TOLERANCE, detect_spacegroup

pytestmark = [pytest.mark.slow, pytest.mark.gate]

AUTO = Path(__file__).parent.parent / "fixtures" / "auto"

# Strict answers, and polytype families first-shell constraints cannot
# separate (ADR-0010).
EXPECTED = {
    "nacl": {225},
    "cscl": {221},
    "si": {227, 194},
    "zno": {186, 216},
    "tio2": {136, 141},
}

# Recorded failures at the correct threshold (ADR-0005), with what was seen.
XFAIL = {
    "cscl": "returns a band-slack distortion detected as Pmma (51); 221 never "
    "enters Phase 0's top five",
    "si": "returns a P4/mmm (123) net whose four second neighbours sit 0.0002 A "
    "outside the cutoff: soft CN 4.000, integer CN 2",
}


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(n, marks=pytest.mark.xfail(reason=XFAIL[n], strict=False))
        if n in XFAIL
        else n
        for n in sorted(EXPECTED)
    ],
)
def test_detected_symmetry_is_the_known_answer(name, tmp_path):
    out = tmp_path / f"{name}.cif"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mizuho",
            "run",
            str(AUTO / f"{name}.yaml"),
            "--format",
            "cif",
            "--output",
            str(out),
            "--seed",
            "42",
        ],
        capture_output=True,
        text=True,
        timeout=7200,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    lat, frac, species = read_cif(out).expand()
    found = detect_spacegroup(lat, frac, species, SUCCESS_TOLERANCE)
    assert found is not None and found[0] in EXPECTED[name], (
        f"{name}: detected {found} at {SUCCESS_TOLERANCE} A, expected {sorted(EXPECTED[name])}"
    )
