"""Phase I-2 benchmark spike: does supplying Stage 2's L-BFGS-B with an
exact JAX gradient for the coordination-number + volume terms (the
"partial-gradient hybrid", see objective/energy_jax.py) actually beat
scipy's default finite-difference gradient on real fixtures?

For each of adamite / spinel_mgal2o4 / topaz (chosen for their range of
dimensionality and CN-domination, see the mizuho-py-wychoff-position
plan's phase I-2 section), this script:
  1. Obtains a real Stage-1-converged starting point (reusing the same
     cheap direct-reconstruction shortcuts the project's own earlier
     investigations used, to avoid Phase 1's full search cost).
  2. Runs Stage 2 (scipy.optimize.minimize, method="L-BFGS-B") TWICE
     from that SAME starting point: once with jac=None (today's
     default, scipy's own finite difference) and once with
     jac=energy_jax.make_gradient_fn(...) (including that function's
     own one-time JIT-compile cost in the timing, since that cost is
     real and would be paid in production too).
  3. Reports wall-clock time and final error for both, and evaluates
     the phase I-2 Go/No-Go criterion: >=1.5x speedup AND <0.1%
     relative error difference, on EVERY fixture, before recommending
     wiring optimization.refine_jac into driver.py as more than an
     experimental opt-in.
"""
import json
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import yaml
from scipy.optimize import minimize

from mizuho.objective.energy import compute_error
from mizuho.objective.energy_jax import make_gradient_fn
from mizuho.optimize.cma import multistart_cma_es
from mizuho.pipeline import build_sg_config, search_and_rank
from mizuho.search.dfs import wyckoff_tree_search
from mizuho.structure import (assignments_to_sites, build_bounds,
                               resolve_fixed_assignment)

FIXTURES_DIR = "/Users/yusuke/Claude_Code/Mizuho/tests/fixtures"


def _load(name):
    with open(f"{FIXTURES_DIR}/{name}.yaml") as f:
        return yaml.safe_load(f)


def _stage1_start(cfg, opt_cfg, seed=42):
    """A real Stage-1 (CMA-ES) result -- the actual starting point
    Stage 2 refinement begins from in production, not an arbitrary
    point -- computed directly (bypassing optimize_one_assignment's
    own Stage 2 call) so the SAME starting point can be fed into two
    DIFFERENT Stage-2 runs for a fair comparison."""
    bounds = build_bounds(cfg)
    n_wyckoff = sum(len(s["free_params"]) for s in cfg["wyckoff_sites"])
    n_lattice = len(bounds) - n_wyckoff
    periodic_mask = np.array([False] * n_lattice + [True] * n_wyckoff)
    best_p, best_e = multistart_cma_es(
        objective=lambda pv: compute_error(pv, cfg), bounds=bounds,
        n_restarts=opt_cfg.get("n_restarts", 8), sigma0=0.25,
        maxiter=opt_cfg.get("cma_maxiter", 300), tol=1e-8, seed=seed,
        verbose=False, config=cfg, periodic_mask=periodic_mask)
    return best_p, best_e, bounds


def _run_stage2(cfg, best_p, bounds, jac):
    t0 = time.perf_counter()
    ref = minimize(lambda pv: compute_error(pv, cfg), best_p, jac=jac,
                    method="L-BFGS-B", bounds=bounds,
                    options={"maxiter": 1000, "ftol": 1e-10})
    elapsed = time.perf_counter() - t0
    # scipy's own ref.fun can be STALE/WRONG when L-BFGS-B terminates
    # abnormally (status="ABNORMAL_TERMINATION_IN_LNSRCH", nit=0) --
    # empirically confirmed this happens whenever best_p starts exactly
    # AT a lattice_bounds edge (a real, pre-existing scipy/L-BFGS-B
    # quirk, unrelated to jac=None vs jac=<jax gradient>: reproduced
    # identically with jac=None). Always recompute the true value at
    # ref.x directly rather than trusting ref.fun, so this benchmark's
    # numbers are never corrupted by that quirk.
    true_err = compute_error(ref.x, cfg)
    return elapsed, true_err, int(ref.nit)


def _benchmark_one(name, cfg):
    print(f"=== {name} ===", flush=True)
    opt_cfg = cfg.get("optimization", {})
    t0 = time.time()
    best_p, stage1_err, bounds = _stage1_start(cfg, opt_cfg)
    print(f"  Stage 1 done in {time.time()-t0:.1f}s, error={stage1_err:.6f}, ndim={len(best_p)}")

    t_fd, err_fd, nit_fd = _run_stage2(cfg, best_p, bounds, jac=None)
    print(f"  Stage 2 (finite-difference baseline): {t_fd:.3f}s, "
          f"error={err_fd:.6f}, nit={nit_fd}")

    t0 = time.time()
    grad_fn = make_gradient_fn(cfg, best_p)
    t_setup = time.time() - t0
    t_hybrid_call, err_hybrid, nit_hybrid = _run_stage2(cfg, best_p, bounds, jac=grad_fn)
    t_hybrid = t_setup + t_hybrid_call
    print(f"  Stage 2 (JAX hybrid, incl. {t_setup:.2f}s jit-compile setup): "
          f"{t_hybrid:.3f}s, error={err_hybrid:.6f}, nit={nit_hybrid}")

    speedup = t_fd / t_hybrid
    rel_err_diff = abs(err_hybrid - err_fd) / (abs(err_fd) + 1e-12)
    go = (speedup >= 1.5) and (rel_err_diff < 0.001)
    print(f"  speedup={speedup:.2f}x, rel_err_diff={rel_err_diff*100:.4f}%, "
          f"Go(>=1.5x and <0.1%)={go}")

    return {
        "fixture": name, "ndim": len(best_p), "stage1_error": stage1_err,
        "fd_time_s": t_fd, "fd_error": err_fd, "fd_nit": nit_fd,
        "hybrid_time_s": t_hybrid, "hybrid_setup_s": t_setup,
        "hybrid_call_s": t_hybrid_call, "hybrid_error": err_hybrid,
        "hybrid_nit": nit_hybrid, "speedup": speedup,
        "rel_err_diff": rel_err_diff, "go": go,
    }


results = {}


def _save():
    with open("/private/tmp/claude-501/-Users-yusuke-Claude-Code-Mizuho/"
              "7b0028a7-7282-472f-963a-5292cda80810/scratchpad/"
              "jax_gradient_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)


# --- spinel_mgal2o4: cheap enough for a real search_and_rank ---
base = _load("spinel_mgal2o4")
sg_config = build_sg_config(base, base["spacegroup"])
valid_asgns, _, _ = search_and_rank(sg_config, max_cands=1)
cfg = dict(sg_config)
cfg["wyckoff_sites"] = assignments_to_sites(valid_asgns[0])
results["spinel_mgal2o4"] = _benchmark_one("spinel_mgal2o4", cfg)
_save()

# --- adamite: reconstruct the known B-4 best topology directly
# (bypasses Phase 1's ~19min search -- same shortcut this session's
# earlier error_breakdown_analysis.py used) ---
base = _load("adamite")
sg_config = build_sg_config(base, base["spacegroup"])
asgn = resolve_fixed_assignment(58, [
    {"element": "Zn", "letter": "a"}, {"element": "Zn", "letter": "b"},
    {"element": "Zn", "letter": "c"}, {"element": "Zn", "letter": "d"},
    {"element": "O", "letter": "g"}, {"element": "O", "letter": "h"},
    {"element": "O", "letter": "h"}, {"element": "As", "letter": "f"},
    {"element": "H", "letter": "g"},
])
cfg = dict(sg_config)
cfg["wyckoff_sites"] = assignments_to_sites(asgn)
results["adamite"] = _benchmark_one("adamite", cfg)
_save()

# --- topaz: wyckoff_tree_search directly (bypasses ranking, which is
# too expensive) + first non-degenerate candidate among the first 200
# -- same shortcut this session's earlier error_breakdown_analysis.py
# used ---
base = _load("topaz")
sg_config = build_sg_config(base, base["spacegroup"])
valid_asgns = wyckoff_tree_search(sg_config)
asgn = next((a for a in valid_asgns[:200]
             if len({x.entry.letter for x in a}) == len(a)), valid_asgns[0])
cfg = dict(sg_config)
cfg["wyckoff_sites"] = assignments_to_sites(asgn)
results["topaz"] = _benchmark_one("topaz", cfg)
_save()

print("\n=== Summary ===")
for name, r in results.items():
    print(f"  {name}: speedup={r['speedup']:.2f}x, rel_err_diff={r['rel_err_diff']*100:.4f}%, go={r['go']}")
overall_go = all(r["go"] for r in results.values())
print(f"\nOverall Go/No-Go (ALL fixtures must pass): {overall_go}")
