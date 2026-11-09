"""Replay per-restart stopping rules on recorded Stage-1 trajectories.

Second half of the measurement in docs/plan-phase2-early-stop.md. Reads the
output of scripts/record_stage1_trajectories.py and, for every rule and
patience, computes exactly what Stage 1 would have done had the rule been on:
a seeded restart stopped at generation g has the first g generations of its
full run, so its end point is the best evaluation up to there.

Rules (a restart is never stopped while its best has cn_deficit 0):
    deficit     the best cn_deficit has not decreased for K generations
    deficit_cn  the best (cn_deficit, soft CN term) has not improved, the CN
                term by at least --rel-tol relative
    deficit_f   the best (cn_deficit, total error) has not improved, the
                total by at least --rel-tol relative
    plateau     the existing plateau_patience check of optimize.cma (total
                error only, relative tolerance --rel-tol), as the baseline;
                K is its patience

Patience: K = max(--k-min, ceil(c * n)), n the candidate's parameter count,
for every c of --c-grid (for plateau too, so the rules are compared at the
same budgets).

Stage 1 is replayed with the endpoint ledger on, as in production (ADR-0019):
restarts in order, the rest skipped once --hits endpoints at or below
--threshold share a topology class. A candidate's zero-error classes are the
classes of its Stage-1 endpoints at or below --threshold; the reference is
the replay without any rule. A class found in the reference and not under a
rule is lost.

Output: <out>/replay.csv, one row per (rule, c, split).

Usage:
    python scripts/replay_early_stop.py runs/alon/alon.yaml runs/earlystop \
        --workers 6
"""

import argparse
import csv
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RULES = ("deficit", "deficit_cn", "deficit_f", "plateau")
DEFAULT_C_GRID = (0.5, 1, 1.5, 2, 3, 4, 6, 8, 12)


# ----------------------------------------------------------------------
# Per-restart replay
# ----------------------------------------------------------------------


def last_improvement(f, deficit, cn, popsize, rule, rel_tol):
    """Per generation g = 0..G (0 = the start point): the generation at which
    the rule's best key last improved, and the best key's cn_deficit."""
    n_gen = (len(f) - 1) // popsize
    last = np.zeros(n_gen + 1, dtype=int)
    best_d = np.zeros(n_gen + 1, dtype=int)
    bd, bs = int(deficit[0]), float(f[0] if rule == "deficit_f" else cn[0])
    best_d[0] = bd
    li = 0
    for g in range(1, n_gen + 1):
        lo = 1 + (g - 1) * popsize
        improved = False
        for j in range(lo, lo + popsize):
            d = int(deficit[j])
            s = float(f[j] if rule == "deficit_f" else cn[j])
            if d < bd:
                bd, bs, improved = d, s, True
            elif d == bd and rule != "deficit" and s < bs - rel_tol * abs(bs) - 1e-15:
                bs, improved = s, True
        if improved:
            li = g
        last[g], best_d[g] = li, bd
    return last, best_d


def stop_generation(last, best_d, k):
    """First generation at which the rule stops the restart, or None."""
    g = np.arange(len(last))
    hit = np.nonzero((g >= 1) & (best_d > 0) & (g - last >= k))[0]
    return int(hit[0]) if len(hit) else None


def plateau_stop_generation(f, popsize, k, rel_tol):
    """optimize.cma._cma_es's plateau_patience check, replayed."""
    n_gen = (len(f) - 1) // popsize
    best = float(f[0])
    ref_f, ref_g = best, 0
    for g in range(1, n_gen + 1):
        lo = 1 + (g - 1) * popsize
        best = min(best, float(f[lo : lo + popsize].min()))
        if g - ref_g >= k:
            if (ref_f - best) / max(abs(ref_f), 1e-12) < rel_tol:
                return g
            ref_f, ref_g = best, g
    return None


# ----------------------------------------------------------------------
# Per-candidate replay
# ----------------------------------------------------------------------

_CFG_CACHE = {}


def _config(base, sg, comp, specs):
    from mizuho.pipeline import build_sg_config
    from mizuho.structure import assignments_to_sites, resolve_fixed_assignment

    key = (sg, comp)
    if key not in _CFG_CACHE:
        _CFG_CACHE[key] = build_sg_config(base, sg, composition=comp)
    cfg = dict(_CFG_CACHE[key])
    cfg["wyckoff_sites"] = assignments_to_sites(resolve_fixed_assignment(sg, specs))
    return cfg


def replay_candidate(task):
    """{(rule, c): (evals, frozenset of zero-error classes,
    frozenset of near-zero classes, best Stage-1 error)} plus the reference
    under (None, None)."""
    (cand, base, traj_dir, c_grid, k_min, rel_tol, threshold, near, hits) = task
    from mizuho.optimize.endpoints import topology_key

    d = np.load(Path(traj_dir) / f"{cand['id']}.npz")
    n_r = int(d["n_restarts"])
    restarts = []
    for k in range(n_r):
        restarts.append(
            {
                "f": d[f"f_{k}"],
                "deficit": d[f"deficit_{k}"],
                "cn": d[f"cn_{k}"],
                "popsize": int(d[f"popsize_{k}"]),
                "imp_idx": d[f"improve_idx_{k}"],
                "imp_x": d[f"improve_x_{k}"],
            }
        )
    n_dims = restarts[0]["imp_x"].shape[1] if n_r else 0
    cfg = _config(base, cand["sg"], cand["composition"], cand["specs"])
    class_cache = {}

    def endpoint(k, cutoff):
        r = restarts[k]
        pos = int(np.searchsorted(r["imp_idx"], cutoff, side="left")) - 1
        idx = int(r["imp_idx"][pos])
        return idx, float(r["f"][idx]), pos

    def cls(k, pos):
        if (k, pos) not in class_cache:
            class_cache[(k, pos)] = topology_key(restarts[k]["imp_x"][pos], cfg)
        return class_cache[(k, pos)]

    def run(cutoffs):
        """Stage 1 with the ledger: cutoffs[k] = evaluations restart k makes."""
        evals, zero, nearz, best = 0, set(), set(), math.inf
        counts = {}
        for k in range(n_r):
            evals += cutoffs[k]
            _, fe, pos = endpoint(k, cutoffs[k])
            best = min(best, fe)
            if fe <= near:
                key = cls(k, pos)
                nearz.add(key)
                if fe <= threshold:
                    zero.add(key)
                    counts[key] = counts.get(key, 0) + 1
                    if counts[key] >= hits:
                        break
        return evals, frozenset(zero), frozenset(nearz), best

    out = {(None, None): run([len(r["f"]) for r in restarts])}
    lasts = {
        rule: [
            last_improvement(r["f"], r["deficit"], r["cn"], r["popsize"], rule, rel_tol)
            for r in restarts
        ]
        for rule in RULES
        if rule != "plateau"
    }
    for c in c_grid:
        kk = max(k_min, math.ceil(c * n_dims))
        for rule in RULES:
            cutoffs = []
            for k, r in enumerate(restarts):
                if rule == "plateau":
                    g = plateau_stop_generation(r["f"], r["popsize"], kk, rel_tol)
                else:
                    g = stop_generation(*lasts[rule][k], kk)
                cutoffs.append(len(r["f"]) if g is None else 1 + g * r["popsize"])
            out[(rule, c)] = run(cutoffs)
    return cand["id"], n_dims, out


# ----------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config", help="the recorded run's input YAML")
    ap.add_argument(
        "record_dir", help="output directory of record_stage1_trajectories.py"
    )
    ap.add_argument(
        "--c-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_C_GRID),
        help="patience coefficients c, K = max(k_min, ceil(c * n))",
    )
    ap.add_argument(
        "--k-min",
        type=int,
        default=15,
        help="smallest patience in generations (default 15, as Phase 1's local shells)",
    )
    ap.add_argument(
        "--rel-tol",
        type=float,
        default=1e-3,
        help="relative improvement that counts (default 1e-3, plateau_rel_tol's default)",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=1e-3,
        help="zero-error threshold (default 1e-3, ADR-0005)",
    )
    ap.add_argument(
        "--near",
        type=float,
        default=1e-2,
        help="near-zero threshold, reported alongside (default 1e-2)",
    )
    ap.add_argument(
        "--hits", type=int, default=2, help="ledger early_stop_hits (default 2)"
    )
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)

    base = yaml.safe_load(Path(args.config).read_text())
    rdir = Path(args.record_dir)
    chosen = {c["id"]: c for c in json.loads((rdir / "selection.json").read_text())}
    done = [
        json.loads(l)
        for l in (rdir / "done.jsonl").read_text().splitlines()
        if l.strip()
    ]
    ok = [chosen[r["id"]] for r in done if "failed" not in r]
    print(f"candidates recorded: {len(done)}, usable: {len(ok)}", flush=True)
    tasks = [
        (
            c,
            base,
            str(rdir / "traj"),
            args.c_grid,
            args.k_min,
            args.rel_tol,
            args.threshold,
            args.near,
            args.hits,
        )
        for c in ok
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(replay_candidate, tasks, chunksize=8))

    rows = []
    for split in ("calibration", "validation", "all"):
        sel = [
            (cid, n, out)
            for cid, n, out in results
            if split == "all" or chosen[cid]["split"] == split
        ]
        ref_evals = sum(out[(None, None)][0] for _, _, out in sel)
        ref_zero = sum(len(out[(None, None)][1]) for _, _, out in sel)
        ref_near = sum(len(out[(None, None)][2]) for _, _, out in sel)
        for rule in RULES:
            for c in args.c_grid:
                evals = lost = gained = lost_near = cands_lost = 0
                worse = []
                for cid, n, out in sel:
                    e, z, nz, best = out[(rule, c)]
                    re_, rz, rnz, rbest = out[(None, None)]
                    evals += e
                    lost += len(rz - z)
                    gained += len(z - rz)
                    lost_near += len(rnz - nz)
                    cands_lost += bool(rz - z)
                    if rbest > args.threshold:
                        worse.append(best / max(rbest, 1e-12))
                rows.append(
                    {
                        "split": split,
                        "rule": rule,
                        "c": c,
                        "candidates": len(sel),
                        "evals": evals,
                        "ref_evals": ref_evals,
                        "saving": round(ref_evals / max(evals, 1), 3),
                        "zero_classes_ref": ref_zero,
                        "zero_classes_lost": lost,
                        "zero_classes_gained": gained,
                        "candidates_losing": cands_lost,
                        "near_classes_ref": ref_near,
                        "near_classes_lost": lost_near,
                        "nonzero_best_ratio_median": round(float(np.median(worse)), 4)
                        if worse
                        else "",
                    }
                )
    path = rdir / "replay.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
