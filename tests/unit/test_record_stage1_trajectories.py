"""scripts/record_stage1_trajectories.py: the recorded Stage-1 trajectories
are complete and consistent with what the optimiser itself returns."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import record_stage1_trajectories as rec

from mizuho.objective.energy import compute_error
from mizuho.pipeline import build_sg_config
from mizuho.structure import (
    assignments_to_sites,
    resolve_fixed_assignment,
)

_BASE = {
    "elements": ["Al", "O", "N"],
    "ionic_charges": {"Al": 3, "O": -2, "N": -3},
    "charge_tolerance_min": 0,
    "charge_tolerance_max": 0,
    "coordination_numbers": {"Al": [4, 6]},
    "anion_coordination_numbers": {"O": [2, 4], "N": [3, 4]},
    "density_g_cm3": [3.26, 4.10],
    "optimization": {
        "n_restarts": 3,
        "cma_maxiter": 8,
        "refine_maxiter": 5,
        "seed": 42,
    },
}
_SPECS = [{"element": "Al", "letter": "b"}, {"element": "N", "letter": "b"}]


def test_every_stage1_evaluation_is_recorded(tmp_path, monkeypatch):
    import mizuho.optimize.cma as cma_mod
    import mizuho.optimize.driver as driver_mod

    # the hooks patch module globals; monkeypatch restores them afterwards
    monkeypatch.setattr(cma_mod, "_cma_es", cma_mod._cma_es)
    monkeypatch.setattr(driver_mod, "compute_error", driver_mod.compute_error)
    rec._install_hooks()
    row = rec._record_one(("c0", _BASE, 186, "AlN", _SPECS, str(tmp_path)))
    assert row["n_restarts"] == 3
    d = np.load(tmp_path / "traj" / "c0.npz")
    cfg = build_sg_config(_BASE, 186, composition="AlN")
    cfg["wyckoff_sites"] = assignments_to_sites(resolve_fixed_assignment(186, _SPECS))
    for k in range(3):
        f, lam = d[f"f_{k}"], int(d[f"popsize_{k}"])
        # the start point, then whole generations of `lam` evaluations
        assert (len(f) - 1) % lam == 0 and 1 <= (len(f) - 1) // lam <= 8
        # the restart's own result is the best evaluation recorded for it
        assert float(d[f"end_f_{k}"]) == f.min()
        # the last improvement is the returned point, and it re-evaluates
        idx, xs = d[f"improve_idx_{k}"], d[f"improve_x_{k}"]
        assert idx[-1] == int(np.argmin(f))
        terms = {}
        assert compute_error(xs[-1], cfg, terms=terms) == f.min()
        assert terms["cn_deficit"] == d[f"deficit_{k}"][idx[-1]]
        assert np.allclose(xs[-1], d[f"end_x_{k}"])
