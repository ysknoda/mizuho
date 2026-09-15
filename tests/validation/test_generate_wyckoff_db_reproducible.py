"""Regression guard for scripts/generate_wyckoff_db.py: re-running the
ab-initio derivation for a sample of space groups (spanning simple,
non-symmorphic, and high-symmetry cubic cases) must reproduce exactly
what's currently committed in mizuho.data.WYCKOFF_DB. Protects against
the shipped per-crystal-system files silently drifting out of sync
with the generator (e.g. a manual edit) without needing to pay the
full ~100s cost of regenerating all 230 space groups on every test run.
"""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("spglib")

from mizuho.data import WYCKOFF_DB

SCRIPT_PATH = (Path(__file__).resolve().parent.parent.parent
               / "scripts" / "generate_wyckoff_db.py")

SAMPLE_SGS = [1, 2, 14, 21, 62, 75, 99, 194, 221, 225, 227, 230]


@pytest.fixture(scope="module")
def gen_module():
    spec = importlib.util.spec_from_file_location("generate_wyckoff_db", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.slow
@pytest.mark.parametrize("sg", SAMPLE_SGS)
def test_regenerated_matches_shipped(gen_module, sg):
    from mizuho.data.symmetry_ops import SYMMETRY_OPS_OF_SG
    import numpy as np

    rotations, translations = SYMMETRY_OPS_OF_SG[sg]
    rotations = np.array(rotations, dtype=float)
    translations = np.array(translations, dtype=float)

    regenerated = gen_module.derive_space_group(sg, rotations, translations)
    regen_set = {(e["letter"], e["multiplicity"]) for e in regenerated}
    shipped_set = {(e.letter, e.multiplicity) for e in WYCKOFF_DB[sg]}

    assert regen_set == shipped_set
