"""Centring acts on zero-error points only (ADR-0039).

Above the threshold centring never brought a point below it and was never
measured to help (ADR-0028 (c) measured it on zero-error points), so
`optimization.centring: true` skips it there; "all" restores centring of
every refined point and false switches it off.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from mizuho.optimize import driver
from mizuho.optimize.driver import centring_mode, optimize_assignment
from mizuho.pipeline import build_sg_config
from mizuho.structure import resolve_fixed_assignment

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "auto" / "tio2_rutile.yaml"
RUTILE = [{"element": "Ti", "letter": "a"}, {"element": "O", "letter": "f"}]


@pytest.mark.parametrize("value", [True, False, "all"])
def test_centring_mode_accepts_the_three_settings(value):
    assert centring_mode({"centring": value}) == value


def test_centring_mode_defaults_to_zero_error_only():
    assert centring_mode({}) is True


@pytest.mark.parametrize("value", [1, 0, "zero", None, 1.0])
def test_centring_mode_rejects_anything_else(value):
    with pytest.raises(ValueError, match="optimization.centring"):
        centring_mode({"centring": value})


def _centred_after_stage2(monkeypatch, stage2_error, mode):
    """Whether optimize_assignment centres a Stage-2 point of the given
    error. Stage 2 is replaced so the refined error is the test's own."""
    config = yaml.safe_load(FIXTURE.read_text())
    cfg = build_sg_config(config, 136)
    opt = {"n_restarts": 1, "cma_maxiter": 3, "refine_maxiter": 3, "seed": 1}
    if mode is not None:
        opt["centring"] = mode
    calls = []
    monkeypatch.setattr(driver, "_stage2", lambda obj, cfg, x0, f0, *a: (np.asarray(x0), stage2_error))
    monkeypatch.setattr(driver, "_centre",
                        lambda obj, cfg, ccfg, p, e, *a: calls.append(e) or (p, e))
    res = optimize_assignment(resolve_fixed_assignment(136, RUTILE), cfg, opt)
    assert res and res[0].error == stage2_error
    return bool(calls)


@pytest.mark.parametrize("stage2_error, mode, centred", [
    (5e-4, None, True),     # at or below the threshold: centred by default
    (1e-3, None, True),     # the threshold itself counts as zero error
    (2e-2, None, False),    # above it: not centred by default
    (2e-2, "all", True),    # "all": every refined point, as before ADR-0039
    (5e-4, False, False),   # false: none
])
def test_centring_only_at_or_below_the_threshold(monkeypatch, stage2_error, mode, centred):
    assert _centred_after_stage2(monkeypatch, stage2_error, mode) is centred
