"""Record full Stage-1 CMA-ES trajectories of recorded Phase-2 candidates.

The measurement step of docs/plan-phase2-early-stop.md. Candidates are drawn
from the record of a composition-free exhaustive run (ADR-0033, e.g.
runs/alon/alon.jsonl) and re-optimised exactly as Phase 2 does
(optimize.driver.optimize_assignment, the run's own `optimization` block),
except that the endpoint ledger's early stop and plateau_patience are off, so
every restart runs to its own end. Every objective evaluation of every Stage-1
restart is recorded with its integer coordination deficit and its soft
coordination term (compute_error's `terms`), so any per-restart stopping rule
can be replayed offline: a seeded restart stopped at generation g has exactly
the first g generations of its full run.

Selection: within every space group of the record, a fixed fraction of the
candidates that pass the bond-count pruning of ADR-0037 (seeded, at least one
per group that has any). Space groups are split alternately, within each
crystal system, into a calibration half and a validation half.

Output, under --out:
    selection.json            the chosen candidates and their split
    traj/<id>.npz             one file per candidate (see _save)
    done.jsonl                one line per finished candidate (resumable:
                              a rerun skips every id already listed)

Usage:
    python scripts/record_stage1_trajectories.py runs/alon/alon.yaml \
        runs/alon/alon.jsonl --out runs/earlystop --fraction 0.2 --workers 6
"""

import argparse
import functools
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mizuho.lattice import CRYSTAL_SYSTEM_OF_SG

_SYSTEMS = (
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "hexagonal",
    "cubic",
)


def crystal_system(sg: int) -> str:
    return CRYSTAL_SYSTEM_OF_SG.get(sg, "triclinic")


# ----------------------------------------------------------------------
# Recording hooks (installed in every worker process)
# ----------------------------------------------------------------------


class _Recorder:
    """Collects the evaluations of the Stage-1 restart currently running."""

    def __init__(self):
        self.active = False
        self.restarts = []

    def begin(self, n_dims: int, x0):
        self.active = True
        self.restarts.append(
            {
                "n_dims": n_dims,
                "popsize": 4 + int(3 * math.log(n_dims)),
                "f": [],
                "deficit": [],
                "cn": [],
                "improve_idx": [],
                "improve_x": [],
                "x0": np.asarray(x0, float).copy(),
            }
        )

    def end(self, best_x, best_f):
        r = self.restarts[-1]
        r["end_x"], r["end_f"] = np.asarray(best_x, float).copy(), float(best_f)
        self.active = False

    def log(self, x, f, terms):
        r = self.restarts[-1]
        best = min(r["f"]) if r["f"] else math.inf
        r["f"].append(f)
        r["deficit"].append(terms.get("cn_deficit", -1))
        r["cn"].append(terms.get("cn", math.nan))
        if f < best:
            r["improve_idx"].append(len(r["f"]) - 1)
            r["improve_x"].append(np.asarray(x, float).copy())


_REC = _Recorder()


def _recording_error(param_vector, config):
    """compute_error, logging every call made inside a Stage-1 restart."""
    from mizuho.objective.energy import compute_error

    terms = {}
    f = compute_error(param_vector, config, terms=terms)
    if _REC.active:
        _REC.log(param_vector, f, terms)
    return f


def _install_hooks():
    import mizuho.optimize.cma as cma_mod
    import mizuho.optimize.driver as driver_mod

    original = cma_mod._cma_es

    def cma_es(objective, x0, bounds, **kwargs):
        stage1 = (
            isinstance(objective, functools.partial)
            and objective.func is _recording_error
        )
        if not stage1:  # Phase-1-style local-shell solves of the seed
            return original(objective, x0, bounds, **kwargs)
        _REC.begin(len(x0), x0)
        try:
            res = original(objective, x0, bounds, **kwargs)
        except BaseException:
            _REC.active = False
            raise
        _REC.end(res[0], res[1])
        return res

    cma_mod._cma_es = cma_es
    driver_mod.compute_error = _recording_error


# ----------------------------------------------------------------------
# One candidate
# ----------------------------------------------------------------------


def _record_one(task):
    cid, base, sg, comp, specs, out_dir = task
    from mizuho.optimize.driver import optimize_assignment
    from mizuho.pipeline import build_sg_config
    from mizuho.structure import resolve_fixed_assignment

    _REC.restarts = []
    _REC.active = False
    sg_config = build_sg_config(base, sg, composition=comp)
    opt_cfg = dict(base.get("optimization", {}))
    opt_cfg["topology_grouping"] = {
        **(opt_cfg.get("topology_grouping") or {}),
        "early_stop": False,
    }
    opt_cfg.pop("plateau_patience", None)
    asgn = resolve_fixed_assignment(sg, specs)
    t0 = time.time()
    results = optimize_assignment(asgn, sg_config, opt_cfg, verbose=False, label=cid)
    seconds = time.time() - t0
    final = results[0] if results else None
    _save(Path(out_dir) / "traj" / f"{cid}.npz", _REC.restarts)
    return {
        "id": cid,
        "sg": sg,
        "composition": comp,
        "specs": specs,
        "seconds": round(seconds, 2),
        "n_restarts": len(_REC.restarts),
        "n_evals": int(sum(len(r["f"]) for r in _REC.restarts)),
        "final_error": None if final is None else float(final.error),
        "final_topology": None
        if final is None or final.topology is None
        else repr(final.topology),
    }


def _save(path: Path, restarts: list) -> None:
    """Per restart k: f_k, deficit_k, cn_k (one entry per evaluation; entry 0
    is the start point, then generation g is entries 1+g*popsize ...
    (g+1)*popsize), improve_idx_k / improve_x_k (every evaluation that set a
    new best-so-far total, with its parameter vector), x0_k, end_x_k, end_f_k,
    popsize_k."""
    arrays = {"n_restarts": np.array(len(restarts))}
    for k, r in enumerate(restarts):
        arrays[f"f_{k}"] = np.asarray(r["f"], float)
        arrays[f"deficit_{k}"] = np.asarray(r["deficit"], np.int32)
        arrays[f"cn_{k}"] = np.asarray(r["cn"], float)
        arrays[f"improve_idx_{k}"] = np.asarray(r["improve_idx"], np.int32)
        arrays[f"improve_x_{k}"] = (
            np.vstack(r["improve_x"]) if r["improve_x"] else np.zeros((0, r["n_dims"]))
        )
        arrays[f"x0_{k}"] = r["x0"]
        arrays[f"end_x_{k}"] = r.get("end_x", np.full(r["n_dims"], np.nan))
        arrays[f"end_f_{k}"] = np.array(r.get("end_f", np.nan))
        arrays[f"popsize_{k}"] = np.array(r["popsize"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _worker_init():
    _install_hooks()


# ----------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------


def select(base: dict, record: Path, fraction: float, seed: int) -> list:
    """Candidates of the record that pass the bond-count pruning, `fraction`
    of each space group's (at least one), with a calibration/validation split
    by space group, alternating within each crystal system."""
    from mizuho.search.bond_counting import BondCountCheck
    from mizuho.structure import resolve_fixed_assignment

    rng = np.random.default_rng(seed)
    per_sg = {}
    n_entries = n_total = n_pruned = 0
    for line in record.read_text().splitlines():
        row = json.loads(line)
        sg, entries = row["sg"], row["entries"]
        if not entries:
            continue
        check = BondCountCheck.from_config(
            dict(base, spacegroup=sg), base["elements"], composition_free=True
        )
        # A record row is one topology class; an assignment that reached
        # several classes has several rows. One candidate per assignment,
        # carrying its lowest recorded error.
        unique = {}
        for e in entries:
            n_entries += 1
            key = (e["composition"], json.dumps(e["specs"], sort_keys=True))
            if key not in unique or e["error"] < unique[key]["error"]:
                unique[key] = e
        kept = []
        for e in unique.values():
            n_total += 1
            if check is not None and not check.accepts(
                resolve_fixed_assignment(sg, e["specs"])
            ):
                n_pruned += 1
                continue
            kept.append(e)
        if kept:
            per_sg[sg] = kept
    chosen = []
    by_system = {s: [] for s in _SYSTEMS}
    for sg in sorted(per_sg):
        by_system[crystal_system(sg)].append(sg)
    split_of = {}
    for system, groups in by_system.items():
        order = list(rng.permutation(groups))
        for i, sg in enumerate(order):
            split_of[int(sg)] = "calibration" if i % 2 == 0 else "validation"
    for sg in sorted(per_sg):
        entries = per_sg[sg]
        n = max(1, round(fraction * len(entries)))
        idx = sorted(rng.choice(len(entries), size=n, replace=False))
        for i in idx:
            e = entries[int(i)]
            chosen.append(
                {
                    "id": f"sg{sg}_{int(i):05d}",
                    "sg": sg,
                    "system": crystal_system(sg),
                    "split": split_of[sg],
                    "composition": e["composition"],
                    "specs": e["specs"],
                    "recorded_error": e["error"],
                }
            )
    print(
        f"record: {n_entries} rows, {n_total} distinct assignments, "
        f"{n_pruned} pruned by bond counting, "
        f"{sum(len(v) for v in per_sg.values())} kept; selected {len(chosen)}"
    )
    return chosen


# ----------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config", help="the run's input YAML (composition-free mode)")
    ap.add_argument("record", help="the run's per-space-group record (JSONL)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument(
        "--fraction",
        type=float,
        default=0.2,
        help="fraction of each space group's unpruned candidates (default 0.2)",
    )
    ap.add_argument("--seed", type=int, default=0, help="selection seed (default 0)")
    ap.add_argument("--workers", type=int, default=6, help="processes (default 6)")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="record only the first N selected candidates (smoke test)",
    )
    ap.add_argument(
        "--select-only", action="store_true", help="write selection.json and stop"
    )
    args = ap.parse_args(argv)

    base = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sel_path = out / "selection.json"
    if sel_path.exists():
        chosen = json.loads(sel_path.read_text())
        print(f"selection: {len(chosen)} candidates (reused {sel_path})")
    else:
        chosen = select(base, Path(args.record), args.fraction, args.seed)
        sel_path.write_text(json.dumps(chosen, indent=1))
    if args.select_only:
        return 0

    done_path = out / "done.jsonl"
    done = set()
    if done_path.exists():
        done = {
            json.loads(l)["id"] for l in done_path.read_text().splitlines() if l.strip()
        }
    todo = [c for c in chosen if c["id"] not in done]
    if args.limit is not None:
        todo = todo[: max(0, args.limit - len(done))]
    print(f"to record: {len(todo)} (already done: {len(done)})", flush=True)
    tasks = [
        (c["id"], base, c["sg"], c["composition"], c["specs"], str(out)) for c in todo
    ]
    t0 = time.time()
    n = 0
    with (
        open(done_path, "a") as fh,
        ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as pool,
    ):
        futures = {pool.submit(_record_one, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            cid = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:  # recorded and skipped, never retried silently
                row = {"id": cid, "failed": f"{type(exc).__name__}: {exc}"}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            n += 1
            if n % 25 == 0 or n == len(tasks):
                rate = (time.time() - t0) / n
                print(
                    f"{n}/{len(tasks)} recorded, {rate:.1f} s each (wall), "
                    f"eta {rate * (len(tasks) - n) / 3600:.1f} h",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
