"""Unit tests for mizuho.objective.energy_jax.

Every test in this module is skipped (not failed) when the optional
'jax' package is not installed, via the module-level
pytest.importorskip below (mirrors phase G's own optuna/emcee handling
in test_optuna_backend.py/test_mcmc.py).
"""
import numpy as np
import pytest
from scipy.optimize import approx_fprime

pytest.importorskip("jax")

from mizuho.objective import energy_jax
from mizuho.objective.energy import compute_error
from mizuho.objective.energy_jax import make_gradient_fn
from mizuho.structure import assignments_to_sites, resolve_fixed_assignment


def _center_with_neighbors_config(n_neighbors=3, d=1.5, cn_min=2, cn_max=2,
                                   volume_bounds=None):
    """SG1: centre 'C' + n_neighbors 'Nb' atoms -- mirrors
    test_objective.py's _center_with_neighbors_case."""
    specs = ([{"element": "C", "letter": "a"}]
             + [{"element": "Nb", "letter": "a"} for _ in range(n_neighbors)])
    sites = assignments_to_sites(resolve_fixed_assignment(1, specs))
    L = 20.0
    c = 0.5
    coords = [c, c, c]
    directions = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1)]
    for dx, dy, dz in directions[:n_neighbors]:
        coords += [c + dx * d / L, c + dy * d / L, c + dz * d / L]
    lattice = [L, L, L, 90.0, 90.0, 90.0]
    pv = np.array(lattice + coords)

    cfg = {"spacegroup": 1, "wyckoff_sites": sites,
           "coordination_constraints": [
               {"center_atom": "C", "neighbor_atom": "Nb", "cn_min": cn_min,
                "cn_max": cn_max, "cutoff_distance": 2.0, "weight": 1.5}]}
    if volume_bounds is not None:
        cfg["volume_constraint"] = {"volume_min": volume_bounds[0],
                                     "volume_max": volume_bounds[1], "weight": 2.0}
    return cfg, pv


def _nacl_config():
    """SG225 (Fm-3m): a real multi-operation space group with
    multiplicity-4 sites and ZERO free Wyckoff parameters (Na@a/Cl@b
    are both fully fixed) -- exercises real orbit-expansion
    deduplication (many symmetry ops -> 4 unique points per site)
    cheaply, without needing an expensive Phase-1 search, AND (with 4
    distinct Na centre atoms) the CN term's per-centre normalisation
    (raw_cn / len(ci_arr)). A single-centre config like
    _center_with_neighbors_config's cannot catch a missing or wrong
    normalisation divisor, since norm=1 is a no-op there -- this is
    exactly how an earlier version of this module's missing
    `/ max(len(ci_arr), 1)` division was caught (it inflated the CN
    gradient here by a factor of 4)."""
    asgn = resolve_fixed_assignment(
        225, [{"element": "Na", "letter": "a"}, {"element": "Cl", "letter": "b"}])
    sites = assignments_to_sites(asgn)
    cfg = {"spacegroup": 225, "wyckoff_sites": sites,
           "coordination_constraints": [
               {"center_atom": "Na", "neighbor_atom": "Cl", "cn_min": 6, "cn_max": 6,
                "cutoff_distance": 3.5, "weight": 1.5}],
           "volume_constraint": {"volume_min": 150.0, "volume_max": 250.0, "weight": 2.0}}
    pv = np.array([5.6])
    return cfg, pv


@pytest.mark.parametrize("cfg_factory", [
    lambda: _center_with_neighbors_config(),
    lambda: _center_with_neighbors_config(volume_bounds=(7000.0, 7900.0)),
    lambda: _nacl_config(),
    # Neighbours 0.05 A inside the cutoff: soft CN 3 x 0.622 = 1.87, 0.13
    # from an integer, so the integrality term (ADR-0007) is active and its
    # JAX gradient is checked against finite differences here.
    lambda: _center_with_neighbors_config(d=1.95),
])
def test_gradient_matches_full_finite_difference(cfg_factory):
    cfg, pv = cfg_factory()
    grad_fn = make_gradient_fn(cfg, pv)
    g_hybrid = grad_fn(pv)
    g_fd = approx_fprime(pv, lambda x: compute_error(x, cfg))

    assert np.all(np.isfinite(g_hybrid))
    # approx_fprime itself carries ~1e-8 relative precision; the CN/
    # volume terms are exact via JAX here, so any residual difference
    # is purely finite-difference noise on the OTHER (still-FD) terms
    # -- a generous but not vacuous tolerance.
    tol = 1e-4 * (np.abs(g_fd).max() + 1.0)
    assert np.abs(g_hybrid - g_fd).max() < tol


def test_gradient_matches_full_finite_difference_at_nearby_point():
    """The reference point (used to resolve which symmetry operations
    produce each site's distinct orbit points) and the point grad_fn is
    actually EVALUATED at need not coincide -- this is the realistic
    usage (L-BFGS-B calls jac at many points during its own
    iterations, not just x0)."""
    cfg, pv = _center_with_neighbors_config(volume_bounds=(7000.0, 7900.0))
    grad_fn = make_gradient_fn(cfg, pv)

    rng = np.random.RandomState(0)
    nearby = pv + 1e-3 * rng.randn(len(pv))
    g_hybrid = grad_fn(nearby)
    g_fd = approx_fprime(nearby, lambda x: compute_error(x, cfg))

    assert np.all(np.isfinite(g_hybrid))
    tol = 1e-4 * (np.abs(g_fd).max() + 1.0)
    assert np.abs(g_hybrid - g_fd).max() < tol


def test_gradient_is_a_descent_direction():
    """A cheap end-to-end sanity check independent of the finite-
    difference comparisons above: stepping a small distance opposite
    the gradient must reduce compute_error's actual value."""
    cfg, pv = _center_with_neighbors_config(volume_bounds=(7000.0, 7900.0))
    grad_fn = make_gradient_fn(cfg, pv)
    g = grad_fn(pv)

    err_before = compute_error(pv, cfg)
    stepped = pv - 1e-4 * g / (np.linalg.norm(g) + 1e-12)
    err_after = compute_error(stepped, cfg)
    assert err_after < err_before


def test_no_coordination_or_volume_constraints_reduces_to_pure_finite_difference():
    """With neither coordination_constraints nor volume_constraint
    configured, the JAX half contributes nothing (grad_cn_volume is
    identically zero, since _cn_volume_total's constraint loops never
    execute) -- the combined gradient must equal a plain finite-
    difference gradient of the (unchanged) full compute_error."""
    sites = assignments_to_sites(resolve_fixed_assignment(1, [{"element": "Na", "letter": "a"}]))
    pv = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0, 0.5, 0.5, 0.5])
    cfg = {"spacegroup": 1, "wyckoff_sites": sites,
           "ionic_charges": {"Na": 3.0}, "charge_balance_weight": 2.0}

    grad_fn = make_gradient_fn(cfg, pv)
    g_hybrid = grad_fn(pv)
    g_fd = approx_fprime(pv, lambda x: compute_error(x, cfg))

    assert np.allclose(g_hybrid, g_fd, atol=1e-4)


def test_raises_clear_actionable_error_when_jax_unavailable(monkeypatch):
    monkeypatch.setattr(energy_jax, "JAX_AVAILABLE", False)
    cfg, pv = _center_with_neighbors_config()
    with pytest.raises(ImportError, match=r"mizuho\[jax\]"):
        make_gradient_fn(cfg, pv)
