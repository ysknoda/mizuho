"""Known-answer tests for the automatic space-group search (the PUCT tree
search, the default). Judged on the DETECTED symmetry at the measured
success tolerance, not on the label the search reports
(docs/adr/0010-success-is-the-detected-symmetry.md), of every returned
structure that satisfies the constraints: the published structure must be
among the zero-error topology classes, not necessarily first, since other
structures with the same bonding (polymorphs, metastable or not) are
legitimate answers too (ADR-0030).

Minutes per compound, so they run with the other search gates:
`pytest --run-gates tests/integration/test_auto_spacegroup_known_answers.py`.
Fixtures and the provenance of their constraints: tests/fixtures/auto/.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from mizuho.io.cif import read_cif
from mizuho.symmetry import SUCCESS_TOLERANCE, detect_spacegroup

pytestmark = [pytest.mark.slow, pytest.mark.gate]

ROOT = Path(__file__).resolve().parents[2]
AUTO = ROOT / "tests" / "fixtures" / "auto"

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


ZERO_ERROR = 1e-3   # ADR-0026's threshold
TOP = 20


def _zero_error_ranks(stdout: str) -> list:
    """Ranks of the GLOBAL RANKING rows at or below ZERO_ERROR."""
    lines = stdout.split("GLOBAL RANKING", 1)[1].splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip().startswith("---")) + 1
    out = []
    for line in lines[start:]:
        if not line.strip():
            break                       # the table ends at the first blank line
        rank, error = line.split()[:2]
        if float(error) <= ZERO_ERROR:
            out.append(int(rank))
    return out


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(n, marks=pytest.mark.xfail(reason=XFAIL[n], strict=False))
        if n in XFAIL
        else n
        for n in sorted(EXPECTED)
    ],
)
def test_published_structure_is_among_the_zero_error_classes(name, tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "mizuho", "run", str(AUTO / f"{name}.yaml"),
         "--format", "cif", "--top", str(TOP), "--seed", "42"],
        capture_output=True, text=True, timeout=7200, cwd=tmp_path,
        # The top-N files land in the working directory; keep mizuho importable.
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(
            [str(ROOT)] + [p for p in [os.environ.get("PYTHONPATH")] if p])),
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    found = []
    for rank in _zero_error_ranks(result.stdout):
        path = next(tmp_path.glob(f"*_rank{rank}.cif"), None)
        if path is None:
            continue
        lat, frac, species = read_cif(path).expand()
        hit = detect_spacegroup(lat, frac, species, SUCCESS_TOLERANCE)
        found.append(None if hit is None else hit[0])
    assert set(found) & EXPECTED[name], (
        f"{name}: zero-error structures detected as {found} at {SUCCESS_TOLERANCE} A, "
        f"expected one of {sorted(EXPECTED[name])}"
    )
