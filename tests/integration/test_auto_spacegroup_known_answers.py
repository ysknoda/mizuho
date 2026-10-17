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
    "tio2_rutile": {136},
    "tio2_anatase": {141},
}

# Recorded failures at the correct threshold (ADR-0005), with what was seen.
XFAIL = {
    "nacl": "detected P4mm (99) at 1e-3 A, re-measured 2026-10-14 (ADR-0021)",
    "cscl": "returns a band-slack distortion detected as Pmma (51); 221 never "
    "enters Phase 0's top five",
    "zno": "detected Pmc2_1 (26) at 1e-3 A with the hand constraints, 2026-10-14 "
    "(ADR-0021); not re-measured since the derived constraints and density (ADR-0030)",
    "tio2_rutile": "not measured since the split into rutile and anatase (ADR-0030); "
    "the combined fixture detected Pmc2_1 (26), 2026-10-14",
    "tio2_anatase": "not measured since the split into rutile and anatase (ADR-0030); "
    "the combined fixture detected Pmc2_1 (26), 2026-10-14",
    "si": "detected Imma (74) at 1e-4 to 1e-1 A with integer CN 4, re-measured "
    "2026-10-14; the former P4/mmm net with soft CN 4 at integer CN 2 is closed "
    "by the gray band and compact support",
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
