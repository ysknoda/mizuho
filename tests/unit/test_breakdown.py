"""Unit tests for mizuho.objective.breakdown.

This module previously had zero test coverage (grep confirmed zero
references to error_breakdown/print_error_breakdown anywhere under
tests/ -- see the mizuho-py-wychoff-position plan's phase I-1 finding).
These tests establish the module's core invariant (its "total" must
always match compute_error's value for the same param_vector/config --
the whole point of a diagnostic breakdown is to explain, not
recompute, the number actually optimised) and pin down a double-
counting bug found while auditing the module for phase I-1.
"""

import numpy as np
import pytest

from mizuho.objective.breakdown import error_breakdown
from mizuho.objective.energy import compute_error
from mizuho.structure import assignments_to_sites, resolve_fixed_assignment


def _single_atom_config(extra=None):
    sites = assignments_to_sites(resolve_fixed_assignment(1, [{"element": "Na", "letter": "a"}]))
    cfg = {"spacegroup": 1, "wyckoff_sites": sites}
    if extra:
        cfg.update(extra)
    return cfg


def _single_atom_param_vector(a=10.0):
    return np.array([a, a, a, 90.0, 90.0, 90.0, 0.5, 0.5, 0.5])


def _si_o3_config(extra=None):
    """SG1: Si centre + 3 O neighbours along +/-x, +y -- enough for the
    hybridisation term's CN/angle sub-penalties to evaluate without
    crashing (K>=2)."""
    sites = assignments_to_sites(resolve_fixed_assignment(
        1, [{"element": "Si", "letter": "a"}, {"element": "O", "letter": "a"},
            {"element": "O", "letter": "a"}, {"element": "O", "letter": "a"}]))
    cfg = {"spacegroup": 1, "wyckoff_sites": sites}
    if extra:
        cfg.update(extra)
    return cfg


def _si_o3_param_vector():
    L = 20.0
    c = 0.5
    d = 1.6 / L
    coords = [c, c, c,
              c + d, c, c,
              c - d, c, c,
              c, c + d, c]
    return np.array([L, L, L, 90.0, 90.0, 90.0] + coords)


@pytest.mark.parametrize("extra", [
    {},
    {"ionic_charges": {"Na": 3.0}, "charge_balance_weight": 2.0},
    {"volume_constraint": {"volume_min": 200.0, "volume_max": 800.0, "weight": 3.0}},
    {"aspect_ratio_constraint": {"ratio_max": 1.5, "weight": 4.0}},
    {"ionic_charges": {"Na": 3.0}, "charge_balance_weight": 2.0,
     "volume_constraint": {"volume_min": 200.0, "volume_max": 800.0, "weight": 3.0}},
    {"ionic_charges": {"Na": 3.0}, "charge_balance_weight": 2.0,
     "volume_constraint": {"volume_min": 200.0, "volume_max": 800.0, "weight": 3.0},
     "aspect_ratio_constraint": {"ratio_max": 1.5, "weight": 4.0}},
    {"cell_validity_constraint": {"g_min": 0.99, "weight": 2.0}},
    {"ionic_charges": {"Na": 3.0}, "charge_balance_weight": 2.0,
     "volume_constraint": {"volume_min": 200.0, "volume_max": 800.0, "weight": 3.0},
     "aspect_ratio_constraint": {"ratio_max": 1.5, "weight": 4.0},
     "cell_validity_constraint": {"g_min": 0.99, "weight": 2.0}},
    {"normalize_penalties": True,
     "ionic_charges": {"Na": 3.0}, "charge_balance_weight": 2.0,
     "volume_constraint": {"volume_min": 200.0, "volume_max": 800.0, "weight": 3.0},
     "aspect_ratio_constraint": {"ratio_max": 1.5, "weight": 4.0},
     "cell_validity_constraint": {"g_min": 0.99, "weight": 2.0}},
])
def test_error_breakdown_total_matches_compute_error(extra):
    cfg = _single_atom_config(extra)
    pv  = _single_atom_param_vector()
    assert error_breakdown(pv, cfg)["total"] == pytest.approx(compute_error(pv, cfg), abs=1e-9)


def test_error_breakdown_total_matches_compute_error_with_hybridisation_only():
    # hybridisation_constraints alone (no volume_constraint) exercises
    # the hybrid_err re-run path without the double-counting bug
    # condition (which needs volume_constraint too -- see below).
    cfg = _si_o3_config({
        "hybridisation_constraints": [
            {"element": "Si", "hybridisation": "sp3", "cutoff": 2.5}],
    })
    pv = _si_o3_param_vector()
    assert error_breakdown(pv, cfg)["total"] == pytest.approx(compute_error(pv, cfg), abs=1e-6)


def test_error_breakdown_volume_double_counting_bug_is_fixed():
    """Regression test for a bug found while auditing breakdown.py for
    phase I-1: when BOTH volume_constraint and hybridisation_constraints
    are configured, error_breakdown's "total" used to double-count the
    volume penalty -- once folded into hybrid_err (because sub_total,
    used to back out the hybridisation-only contribution from a fresh
    compute_error call, omitted volume), and once added again
    explicitly. This fires even when the true hybridisation penalty is
    exactly 0, since the bug's magnitude is entirely the (nonzero)
    volume penalty leaking into hybrid_err. Also covers
    aspect_ratio_constraint / cell_validity_constraint (phase I-3),
    added to sub_total from the start specifically to avoid repeating
    this same bug class.
    """
    cfg = _si_o3_config({
        "hybridisation_constraints": [
            {"element": "Si", "hybridisation": "sp3", "cutoff": 2.5}],
        # Deliberately violated bounds so E_vol/E_ar/E_cv != 0 and the
        # bug (additive with each) would be visible if reintroduced.
        "volume_constraint": {"volume_min": 1.0, "volume_max": 2.0, "weight": 5.0},
        "aspect_ratio_constraint": {"ratio_max": 1.001, "weight": 5.0},
        "cell_validity_constraint": {"g_min": 0.99999, "weight": 2.0},
    })
    pv = _si_o3_param_vector()

    bd = error_breakdown(pv, cfg)

    # 1) External consistency: must match the value actually optimised.
    #    This is the assertion that actually catches the bug -- the old
    #    code's "total" exceeded compute_error's by exactly the (nonzero)
    #    volume penalty whenever both constraint types were configured.
    assert bd["total"] == pytest.approx(compute_error(pv, cfg), abs=1e-6)

    # 2) Schema self-consistency: result["total"] is documented as the
    #    sum of every per-term value the dict itself reports, each
    #    counted once -- confirm a caller reconstructing the total from
    #    the parts gets the same number "total" already holds.
    reconstructed = (bd["overlap"]
                      + sum(b["penalty"] for b in bd["bond"])
                      + sum(a["penalty"] for a in bd["angle"])
                      + sum(c["penalty"] for c in bd["cn"])
                      + sum(r["penalty"] for r in bd["repulsion"])
                      + bd["charge"]
                      + bd["volume"]["penalty"]
                      + bd["aspect_ratio"]["penalty"]
                      + bd["cell_validity"]["penalty"]
                      + bd["hybrid"])
    assert reconstructed == pytest.approx(bd["total"], abs=1e-9)


def test_error_breakdown_schema_has_repulsion_and_volume_keys():
    # The module docstring previously omitted "repulsion"/"volume" from
    # the documented return schema even though both keys are always
    # present -- lock down the real schema so a future docstring/code
    # divergence is caught here rather than discovered by a caller.
    cfg = _single_atom_config()
    pv  = _single_atom_param_vector()
    bd  = error_breakdown(pv, cfg)
    for key in ("total", "overlap", "bond", "angle", "cn", "repulsion",
                "charge", "volume", "aspect_ratio", "cell_validity", "hybrid"):
        assert key in bd
    assert bd["repulsion"] == []
    assert bd["volume"] is None         # no volume_constraint configured
    assert bd["aspect_ratio"] is None   # no aspect_ratio_constraint configured
    assert bd["cell_validity"] is None  # no cell_validity_constraint configured
