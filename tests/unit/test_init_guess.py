"""Unit tests for mizuho.optimize.init_guess._coordination_shell_seed_x0
(Phase D: seed Phase 2's CMA-ES starting point from geometry.
solve_local_shell's own per-group solutions instead of a blind
box-centre/random guess).
"""
import numpy as np
import pytest

from mizuho.objective.energy import compute_error
from mizuho.optimize.init_guess import _coordination_shell_seed_x0
from mizuho.structure import assignments_to_sites, build_bounds, resolve_fixed_assignment

_ASO4_SPECS = [
    {"element": "As", "letter": "g"},
    {"element": "O", "letter": "g"},
    {"element": "O", "letter": "g"},
    {"element": "O", "letter": "h"},
]

_ASO4_CONFIG_BASE = {
    "spacegroup": 58,
    "composition": "AsO3",   # loose -- only used here to derive the present-species set
    "lattice_bounds": {"a": [7.8, 8.8], "b": [8.0, 9.0], "c": [5.6, 6.5]},
    "bond_constraints": [
        {"atom1": "As", "atom2": "O", "distance_min": 1.60, "distance_max": 1.75},
    ],
    "angle_constraints": [
        {"atom_center": "As", "atom_neighbor": "O",
         "angle_min": 100.0, "angle_max": 120.0},
    ],
    "coordination_constraints": [
        {"center_atom": "As", "neighbor_atom": "O",
         "cn_min": 4, "cn_max": 4, "cutoff_distance": 1.9},
    ],
}


def _config_with_sites(specs, sg=58):
    asgn = resolve_fixed_assignment(sg, specs)
    cfg = dict(_ASO4_CONFIG_BASE)
    cfg["spacegroup"] = sg
    cfg["wyckoff_sites"] = assignments_to_sites(asgn)
    return cfg


def _box_centre(bounds):
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    return (lo + hi) / 2.0


def test_seed_is_noop_without_wyckoff_sites():
    cfg = dict(_ASO4_CONFIG_BASE)   # no "wyckoff_sites" key at all
    x0 = np.array([8.0, 8.5, 6.0])
    bounds = [(7.8, 8.8), (8.0, 9.0), (5.6, 6.5)]
    result = _coordination_shell_seed_x0(x0, bounds, cfg, None)
    assert np.array_equal(result, x0)


def test_seed_is_noop_without_composition():
    cfg = _config_with_sites(_ASO4_SPECS)
    del cfg["composition"]
    bounds = build_bounds(cfg)
    x0 = _box_centre(bounds)
    result = _coordination_shell_seed_x0(x0, bounds, cfg, None)
    assert np.array_equal(result, x0)


def test_seed_is_noop_without_coupled_groups():
    # nacl-style: no coordination_constraints/angle_constraints at all.
    asgn = resolve_fixed_assignment(
        225, [{"element": "Na", "letter": "a"}, {"element": "Cl", "letter": "b"}])
    cfg = {
        "spacegroup": 225,
        "composition": "NaCl",
        "lattice_bounds": {"a": [4.5, 6.5]},
        "wyckoff_sites": assignments_to_sites(asgn),
    }
    bounds = build_bounds(cfg)
    x0 = _box_centre(bounds)
    result = _coordination_shell_seed_x0(x0, bounds, cfg, None)
    assert np.array_equal(result, x0)


def test_seed_reuses_shell_seed_info_without_recomputing():
    """When shell_seed_info already has this assignment's group(s)
    solved (exactly as geometry.rank_by_feasibility would produce), no
    optimize_fn call should happen at all -- avoids paying for the
    same local-shell solve twice."""
    from mizuho.composition import parse_composition
    from mizuho.geometry import (build_mini_config, group_coupled_constraints,
                                  sites_for_group, solve_local_shell)

    cfg = _config_with_sites(_ASO4_SPECS)
    bounds = build_bounds(cfg)
    box_centre = _box_centre(bounds)

    def fake_opt(objective, bounds, n_restarts, maxiter, seed, config, periodic_mask,
                 **_ignored):
        p = np.array([b[0] for b in bounds])
        return p, objective(p)

    present = set(parse_composition(cfg["composition"]).keys())
    groups = group_coupled_constraints(cfg, present)
    assert len(groups) == 1
    sites = cfg["wyckoff_sites"]
    indices = sites_for_group(sites, groups[0])
    mini_config = build_mini_config(cfg, sites, groups[0], indices)
    best_p, _ = solve_local_shell(mini_config, fake_opt)
    shell_seed_info = {0: best_p}

    calls = []

    def counting_opt(**kwargs):
        calls.append(1)
        return fake_opt(**kwargs)

    _coordination_shell_seed_x0(
        box_centre, bounds, cfg, counting_opt, shell_seed_info=shell_seed_info)

    assert calls == []   # no fresh solve -- the provided seed was reused entirely


def test_seed_writes_only_bounds_respecting_values():
    """Even with a deliberately tiny, fast solve budget, every value
    written into x0 must respect its own parameter's bounds (the
    function clips explicitly; this guards against a future regression
    silently dropping that clip)."""
    from mizuho.optimize.cma import multistart_cma_es

    cfg = _config_with_sites(_ASO4_SPECS)
    bounds = build_bounds(cfg)
    box_centre = _box_centre(bounds)

    seeded = _coordination_shell_seed_x0(
        box_centre, bounds, cfg, multistart_cma_es,
        n_restarts=1, maxiter=20, seed=1)

    for v, (lo_b, hi_b) in zip(seeded, bounds):
        assert lo_b - 1e-9 <= v <= hi_b + 1e-9


def test_seed_lattice_is_refined_beyond_the_plain_group_average():
    """Phase I-3: after averaging each group's own proposed lattice
    (a no-op here, since this config has exactly one coupled group --
    see group_coupled_constraints in the module docstring), the seed
    now also applies a few _apply_lattice_gradient_hint passes using
    the group's own just-solved (physically meaningful) atomic
    positions. Confirms that step actually fires and moves the lattice
    part away from the raw single-group proposal, while still
    respecting bounds -- the behavioural change this phase introduces,
    as distinct from test_seed_writes_only_bounds_respecting_values
    (bounds compliance, already covered) and
    test_seed_scores_meaningfully_better_than_box_centre (overall
    quality, unaffected -- see that test, still passing unmodified)."""
    from mizuho.composition import parse_composition
    from mizuho.geometry import (build_mini_config, group_coupled_constraints,
                                  sites_for_group, solve_local_shell)
    from mizuho.optimize.cma import multistart_cma_es

    cfg = _config_with_sites(_ASO4_SPECS)
    bounds = build_bounds(cfg)
    box_centre = _box_centre(bounds)

    present = set(parse_composition(cfg["composition"]).keys())
    groups = group_coupled_constraints(cfg, present)
    assert len(groups) == 1   # single group -> averaging alone is a no-op
    sites = cfg["wyckoff_sites"]
    indices = sites_for_group(sites, groups[0])
    mini_config = build_mini_config(cfg, sites, groups[0], indices)
    raw_proposal, _ = solve_local_shell(
        mini_config, multistart_cma_es, n_restarts=8, maxiter=250, seed=42)
    n_lat = 3   # SG58 is orthorhombic
    shell_seed_info = {0: raw_proposal}

    seeded = _coordination_shell_seed_x0(
        box_centre, bounds, cfg, multistart_cma_es, shell_seed_info=shell_seed_info)

    raw_lattice = raw_proposal[:n_lat]
    refined_lattice = seeded[:n_lat]
    assert not np.allclose(refined_lattice, raw_lattice)
    for v, (lo_b, hi_b) in zip(seeded, bounds):
        assert lo_b - 1e-9 <= v <= hi_b + 1e-9


@pytest.mark.slow
def test_seed_scores_meaningfully_better_than_box_centre():
    """Deterministic (fixed seed) comparison: the seeded x0, fed
    DIRECTLY into compute_error with NO further CMA-ES search at all,
    must score clearly better than an unmodified box-centre x0 for a
    real coupled AsO4 tetrahedron -- isolating the seed's own quality
    from CMA-ES's later stochastic refinement, per the plan's
    requirement for a CMA-ES-randomness-independent check."""
    from mizuho.optimize.cma import multistart_cma_es

    cfg = _config_with_sites(_ASO4_SPECS)
    bounds = build_bounds(cfg)
    box_centre = _box_centre(bounds)

    seeded = _coordination_shell_seed_x0(
        box_centre, bounds, cfg, multistart_cma_es,
        n_restarts=8, maxiter=250, seed=42)

    err_box_centre = compute_error(box_centre, cfg)
    err_seeded = compute_error(seeded, cfg)

    assert err_seeded < err_box_centre
    assert err_seeded < 10.0   # a genuinely solved AsO4 shell, not just "less bad"
