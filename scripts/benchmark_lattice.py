#!/usr/bin/env python3
"""Benchmark mizuho.lattice.pairwise_distances_pbc's NumPy vs numba
implementations -- plan phase H-2 (see the plan file's "フェーズH"
section).

Two parts:
  1. Micro-benchmark: pairwise_distances_pbc alone, at N=2/8/36/56
     (the tool's realistic atoms-per-cell range -- see fixtures), both
     wall-clock time and peak memory allocation (tracemalloc) per
     call, numpy vs numba.
  2. End-to-end: compute_error's own before/after speedup on 3 real,
     multi-constraint fixtures (spinel_mgal2o4, topaz, adamite), by
     temporarily monkeypatching objective.energy/breakdown's already-
     bound `pairwise_distances_pbc` name back to the pure-NumPy
     implementation and comparing against the numba-accelerated
     dispatch these modules normally get. This isolates the SAME
     effect a "numba not installed" environment would see, without
     needing to actually uninstall numba.

Requires numba to be installed (mizuho[numba]) -- this script measures
numba's OWN benefit, so there is nothing to benchmark without it.

Usage:
    python3 scripts/benchmark_lattice.py
    python3 scripts/benchmark_lattice.py --out results.json
"""
import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import yaml

import mizuho.objective.breakdown as breakdown_mod
import mizuho.objective.energy as energy_mod
from mizuho.lattice import (
    NUMBA_AVAILABLE, _pairwise_distances_pbc_numba, _pairwise_distances_pbc_numpy,
    build_lattice_matrix,
)
from mizuho.objective.energy import compute_error
from mizuho.pipeline import build_sg_config
from mizuho.search.dfs import wyckoff_tree_search
from mizuho.structure import assignments_to_sites, build_bounds

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
N_VALUES = [2, 8, 36, 56]
N_REPEATS = 500
ENDTOEND_FIXTURES = ["spinel_mgal2o4", "topaz", "adamite"]
ENDTOEND_N_EVALS = 100


def _load_fixture(name):
    with open(FIXTURES_DIR / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def benchmark_kernel():
    print("=== Part 1: pairwise_distances_pbc kernel (numpy vs numba) ===")
    lat = build_lattice_matrix({"a": 6.0, "b": 6.5, "c": 7.0,
                                 "alpha": 80.0, "beta": 95.0, "gamma": 100.0})
    rng = np.random.default_rng(0)
    rows = []
    for n in N_VALUES:
        fa = rng.uniform(0.0, 1.0, size=(n, 3))
        fb = rng.uniform(0.0, 1.0, size=(n, 3))

        # warm up JIT compilation before timing
        _pairwise_distances_pbc_numba(fa, fb, lat)

        t0 = time.perf_counter()
        for _ in range(N_REPEATS):
            _pairwise_distances_pbc_numpy(fa, fb, lat)
        t_numpy = (time.perf_counter() - t0) / N_REPEATS

        t0 = time.perf_counter()
        for _ in range(N_REPEATS):
            _pairwise_distances_pbc_numba(fa, fb, lat)
        t_numba = (time.perf_counter() - t0) / N_REPEATS

        tracemalloc.start()
        for _ in range(N_REPEATS):
            _pairwise_distances_pbc_numpy(fa, fb, lat)
        _, peak_numpy = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        tracemalloc.start()
        for _ in range(N_REPEATS):
            _pairwise_distances_pbc_numba(fa, fb, lat)
        _, peak_numba = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        speedup = t_numpy / t_numba if t_numba > 0 else float("inf")
        mem_ratio = peak_numba / peak_numpy if peak_numpy > 0 else 0.0
        print(f"  N={n:3d}  numpy={t_numpy*1e6:8.1f}us  numba={t_numba*1e6:7.1f}us  "
              f"speedup={speedup:5.1f}x  peak_mem numpy={peak_numpy}B numba={peak_numba}B "
              f"(ratio={mem_ratio:.2%})")
        rows.append({"n": n, "t_numpy_us": t_numpy * 1e6, "t_numba_us": t_numba * 1e6,
                      "speedup": speedup, "peak_mem_numpy_bytes": peak_numpy,
                      "peak_mem_numba_bytes": peak_numba})
    return rows


def _prepare_fixture_eval(name):
    base_config = _load_fixture(name)
    sg_config = build_sg_config(base_config, base_config["spacegroup"])
    # wyckoff_tree_search directly, NOT search_and_rank: this benchmark
    # only needs A representative candidate (for a realistic N/dimension
    # count), not the BEST one, so it must not pay for
    # precompute_letter_feasibility/rank_by_feasibility_staged's
    # expensive geometric re-ranking (minutes-to-tens-of-minutes on
    # topaz/adamite, see phase B-3/B-4) just to throw the ranking away.
    valid_asgns = wyckoff_tree_search(sg_config)
    if not valid_asgns:
        return None
    asgn = valid_asgns[0]
    cfg = dict(sg_config)
    cfg["wyckoff_sites"] = assignments_to_sites(asgn)
    bounds = build_bounds(cfg)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    rng = np.random.default_rng(0)
    param_vectors = [rng.uniform(lo, hi) for _ in range(ENDTOEND_N_EVALS)]
    return cfg, param_vectors


def benchmark_end_to_end():
    print("\n=== Part 2: compute_error end-to-end (numpy-only vs numba-accelerated) ===")
    rows = []
    for name in ENDTOEND_FIXTURES:
        prepared = _prepare_fixture_eval(name)
        if prepared is None:
            print(f"  [{name}] no valid assignment found -- skipping")
            continue
        cfg, param_vectors = prepared

        # Force the pure-NumPy kernel in both modules that call it directly.
        energy_mod.pairwise_distances_pbc = _pairwise_distances_pbc_numpy
        breakdown_mod.pairwise_distances_pbc = _pairwise_distances_pbc_numpy
        compute_error(param_vectors[0], cfg)  # warm-up (irrelevant for pure numpy, kept for symmetry)
        t0 = time.perf_counter()
        for pv in param_vectors:
            compute_error(pv, cfg)
        t_numpy = (time.perf_counter() - t0) / len(param_vectors)

        # Restore the numba-accelerated dispatch (the normal, installed state).
        energy_mod.pairwise_distances_pbc = _pairwise_distances_pbc_numba
        breakdown_mod.pairwise_distances_pbc = _pairwise_distances_pbc_numba
        compute_error(param_vectors[0], cfg)  # JIT warm-up
        t0 = time.perf_counter()
        for pv in param_vectors:
            compute_error(pv, cfg)
        t_numba = (time.perf_counter() - t0) / len(param_vectors)

        speedup = t_numpy / t_numba if t_numba > 0 else float("inf")
        print(f"  [{name}] compute_error: numpy-only={t_numpy*1e3:.3f}ms  "
              f"numba-accelerated={t_numba*1e3:.3f}ms  speedup={speedup:.2f}x")
        rows.append({"fixture": name, "t_numpy_only_ms": t_numpy * 1e3,
                      "t_numba_accelerated_ms": t_numba * 1e3, "speedup": speedup})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help="Write results as JSON")
    args = parser.parse_args()

    if not NUMBA_AVAILABLE:
        print("ERROR: numba is not installed -- nothing to benchmark. "
              "Install it with: pip install mizuho[numba]")
        sys.exit(1)

    kernel_rows = benchmark_kernel()
    e2e_rows = benchmark_end_to_end()

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"kernel": kernel_rows, "end_to_end": e2e_rows}, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
