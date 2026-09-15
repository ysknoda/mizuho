"""Correctness gate for each optimize_fn backend: confirms every
alternative to multistart_cma_es is a genuine drop-in replacement --
both at geometry.local_shell.solve_local_shell's dependency-injection
seam and at the full pipeline level (optimize.driver.optimize_one_assignment,
selected via the optimization.backend YAML key) -- by actually running
it (not a fake optimizer double) against small, fast fixtures/mini-shells
and checking it reaches a reasonably low error in reasonable time.

This is a correctness/regression gate, not a speed/accuracy comparison
across backends -- see scripts/benchmark_optimizers.py (plan phase G-2)
for that. optuna/torch/emcee are optional dependencies (pyproject.toml's
`bayesian`/`gp`/`mcmc` extras) -- their parametrize entries are skipped
automatically when the package is not installed.
"""
import pytest

from mizuho.geometry import solve_local_shell
from mizuho.optimize.basin_hopping import multistart_basin_hopping
from mizuho.optimize.cma import multistart_cma_es
from mizuho.optimize.differential_evolution import multistart_differential_evolution
from mizuho.optimize.driver import optimize_one_assignment
from mizuho.optimize.dual_annealing import multistart_dual_annealing
from mizuho.optimize.mcmc import EMCEE_AVAILABLE, multistart_mcmc
from mizuho.optimize.optuna_backend import (
    GP_AVAILABLE,
    OPTUNA_AVAILABLE,
    multistart_optuna_gp,
    multistart_optuna_tpe,
)
from mizuho.pipeline import build_sg_config
from mizuho.search.dfs import wyckoff_tree_search
from mizuho.structure import assignments_to_sites, resolve_fixed_assignment

_needs_optuna = pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="optuna not installed")
_needs_gp = pytest.mark.skipif(
    not GP_AVAILABLE, reason="optuna GPSampler needs torch (pip install mizuho[gp])")
_needs_emcee = pytest.mark.skipif(not EMCEE_AVAILABLE, reason="emcee not installed")

_BACKEND_FN_CASES = [
    pytest.param(multistart_cma_es, id="cma"),
    pytest.param(multistart_differential_evolution, id="differential_evolution"),
    pytest.param(multistart_basin_hopping, id="basin_hopping"),
    pytest.param(multistart_dual_annealing, id="dual_annealing"),
    pytest.param(multistart_optuna_tpe, marks=_needs_optuna, id="tpe"),
    pytest.param(multistart_optuna_gp, marks=_needs_gp, id="gp"),
    pytest.param(multistart_mcmc, marks=_needs_emcee, id="mcmc"),
]
_BACKEND_NAME_CASES = [
    pytest.param("cma", id="cma"),
    pytest.param("differential_evolution", id="differential_evolution"),
    pytest.param("basin_hopping", id="basin_hopping"),
    pytest.param("dual_annealing", id="dual_annealing"),
    pytest.param("tpe", marks=_needs_optuna, id="tpe"),
    pytest.param("gp", marks=_needs_gp, id="gp"),
    pytest.param("mcmc", marks=_needs_emcee, id="mcmc"),
]


@pytest.mark.slow
@pytest.mark.parametrize("optimize_fn", _BACKEND_FN_CASES)
def test_backend_solves_a_simple_local_shell(optimize_fn):
    """A minimal, fast NaCl-scale mini shell (Na@a + Cl@b, SG225: both
    Wyckoff sites are zero-free-parameter fixed points, so the ONLY
    free parameter is the cubic lattice constant `a`) fed directly
    through geometry.local_shell.solve_local_shell's optimize_fn
    dependency-injection seam. Confirms the drop-in optimize_fn
    contract genuinely holds for real (not faked) alternative backends
    on a real (if intentionally tiny/fast, 1-dimensional) geometry
    problem -- deliberately NOT the harder multi-parameter ASO4
    tetrahedron test_geometry.py's own CMA-ES-only regression test
    uses, since that one is specifically calibrated (per its own
    docstring) to need a much larger budget than a quick correctness
    gate should spend."""
    asgn = resolve_fixed_assignment(225, [
        {"element": "Na", "letter": "a"}, {"element": "Cl", "letter": "b"}])
    sites = assignments_to_sites(asgn)
    mini_config = {
        "spacegroup": 225,
        "wyckoff_sites": sites,
        "lattice_bounds": {"a": [4.0, 8.0]},
        "bond_constraints": [
            {"atom1": "Na", "atom2": "Cl", "distance_min": 2.7, "distance_max": 2.9},
        ],
    }

    _, error = solve_local_shell(
        mini_config, optimize_fn, n_restarts=3, maxiter=40, seed=42)

    assert error < 1e-3


@pytest.mark.slow
@pytest.mark.parametrize("backend", _BACKEND_NAME_CASES)
def test_backend_reaches_a_low_error_nacl_structure(backend, nacl_config):
    """optimization.backend end-to-end, through the full Phase-2 driver
    (not just the geometry/ mini-shell seam): every backend must find
    the ordinary NaCl rock-salt structure (error == 0, both sites
    within their bond-distance constraint) for the legitimate
    (non-degenerate) Wyckoff assignment."""
    sg_config = build_sg_config(nacl_config, nacl_config["spacegroup"])
    valid = wyckoff_tree_search(sg_config)
    # Na@a + Cl@b (or the symmetric Na@b + Cl@a) -- the non-degenerate
    # assignment; Na@a + Cl@a places both species at the same fixed
    # point and is expected to score badly regardless of backend.
    asgn = next(a for a in valid
                if len({x.entry.letter for x in a}) == 2)

    opt_cfg = dict(sg_config["optimization"])
    opt_cfg["backend"] = backend

    _, best_e, structure = optimize_one_assignment(
        asgn, sg_config, opt_cfg, verbose=False, label=backend)

    assert structure is not None
    assert best_e < 1e-3
