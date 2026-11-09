"""compute_error's optional `terms` output: the per-term contributions and
the integer coordination-number deficit the Phase-2 early-stop measurement
records (docs/plan-phase2-early-stop.md)."""

import numpy as np
import pytest

from mizuho.objective.breakdown import error_breakdown
from mizuho.objective.energy import compute_error
from mizuho.pipeline import build_sg_config
from mizuho.structure import (
    assignments_to_sites,
    build_bounds,
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
}

# (space group, composition, Wyckoff specs): wurtzite AlN, and an Al-O-N
# assignment with a free coordinate on every site.
_CASES = [
    (186, "AlN", [{"element": "Al", "letter": "b"}, {"element": "N", "letter": "b"}]),
    (
        186,
        "Al3O3N",
        [
            {"element": "Al", "letter": "c"},
            {"element": "O", "letter": "c"},
            {"element": "N", "letter": "b"},
        ],
    ),
]


def _config(sg, comp, specs):
    cfg = build_sg_config(_BASE, sg, composition=comp)
    cfg["wyckoff_sites"] = assignments_to_sites(resolve_fixed_assignment(sg, specs))
    return cfg


def _points(cfg, n=20, seed=0):
    rng = np.random.default_rng(seed)
    bounds = build_bounds(cfg)
    return [np.array([rng.uniform(lo, hi) for lo, hi in bounds]) for _ in range(n)]


@pytest.mark.parametrize("case", _CASES)
def test_total_is_unchanged_by_terms(case):
    cfg = _config(*case)
    for x in _points(cfg):
        assert compute_error(x, cfg, terms={}) == compute_error(x, cfg)


@pytest.mark.parametrize("case", _CASES)
def test_terms_sum_to_total(case):
    cfg = _config(*case)
    for x in _points(cfg):
        terms = {}
        total = compute_error(x, cfg, terms=terms)
        parts = sum(v for k, v in terms.items() if k != "cn_deficit")
        assert parts == pytest.approx(total, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("case", _CASES)
def test_cn_deficit_matches_integer_counts_of_breakdown(case):
    cfg = _config(*case)
    for x in _points(cfg):
        terms = {}
        compute_error(x, cfg, terms=terms)
        expected = sum(
            max(0, c["cn_min_target"] - n, n - c["cn_max_target"])
            for c in error_breakdown(x, cfg)["cn"]
            for n in c["cn_per_atom"]
        )
        assert terms["cn_deficit"] == expected


def test_cn_deficit_counts_anion_centres():
    """Wurtzite AlN with N given CN 5-6: Al is satisfied (4 in [4, 6]) and
    every N centre (CN 4) falls short by one."""
    base = dict(_BASE, anion_coordination_numbers={"O": [2, 4], "N": [5, 6]})
    cfg = build_sg_config(base, 186, composition="AlN")
    cfg["wyckoff_sites"] = assignments_to_sites(
        resolve_fixed_assignment(
            186, [{"element": "Al", "letter": "b"}, {"element": "N", "letter": "b"}]
        )
    )
    a, c, u = 3.112, 4.982, 0.382  # AlN, a and c as in runs/alon/alon.yaml
    x = np.array([a, c, 0.0, u])
    terms = {}
    compute_error(x, cfg, terms=terms)
    per_n = [
        n
        for s in error_breakdown(x, cfg)["cn"]
        if s["center_atom"] == "N"
        for n in s["cn_per_atom"]
    ]
    assert per_n and all(n == 4 for n in per_n)
    assert terms["cn_deficit"] == len(per_n)


def test_terms_left_empty_for_an_unbuildable_structure():
    cfg = _config(*_CASES[0])
    terms = {}
    assert compute_error(np.array([np.nan, np.nan, np.nan]), cfg, terms=terms) == 1e10
    assert terms == {}
