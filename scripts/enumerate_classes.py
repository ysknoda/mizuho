"""How often is the published class among the zero-error classes one run reaches?

Design review 2026-10-14, proposal 2: instead of returning one answer, report
every topology class the restarts reach at zero error and let DFT choose. Before
building a diversification mechanism, this measures how far the existing
restarts already get: the x4 fit with the published assignment fixed, early
stopping off so every restart runs, and every class refined (driver's
max_classes). Recorded per seed: each class reached, its refined error,
endpoints in it, and whether it is the published class.

    python scripts/enumerate_classes.py diopside_camgsi2o6 --seeds 42 43
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ablate_terms  # noqa: E402

from mizuho.objective.energy import compute_error  # noqa: E402
from mizuho.optimize.driver import optimize_assignment  # noqa: E402
from mizuho.optimize.endpoints import topology_key  # noqa: E402


def run(name, variant, seed, budget=4):
    cfg, pv_ref, asg = ablate_terms._config(name, variant)
    opt = dict(cfg.get("optimization", {}))
    opt["n_restarts"] = int(opt.get("n_restarts", 8) * budget)
    opt["cma_maxiter"] = int(opt.get("cma_maxiter", 300) * budget)
    opt["seed"] = seed
    grouping = dict(opt.get("topology_grouping") or {})
    grouping["early_stop"] = False
    opt["topology_grouping"] = grouping
    key_ref = topology_key(pv_ref, cfg)
    results = optimize_assignment(asg, cfg, opt, verbose=False, label=name, max_classes=10**6)
    classes = [{
        "error": float(r.error),
        "published_class": r.topology == key_ref,
        "n_endpoints": r.n_endpoints,
        "n_zero": r.n_zero,
        "param_vector": [float(v) for v in r.params],
    } for r in results]
    return {"mineral": name, "variant": variant, "seed": seed,
            "error_published": float(compute_error(pv_ref, cfg)),
            "n_stage1": results[0].n_stage1 if results else 0, "classes": classes}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mineral")
    ap.add_argument("--variant", default="iii", choices=ablate_terms.VARIANTS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--out", default="gate-results/classes")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        rec = run(args.mineral, args.variant, seed)
        (out / f"{args.mineral}_{args.variant}_seed{seed}.json").write_text(json.dumps(rec, indent=2))
        zero = [c for c in rec["classes"] if c["error"] <= rec["error_published"] + 1e-3]
        print(f"seed {seed}: {len(rec['classes'])} classes, {len(zero)} at zero error, "
              f"published among them: {any(c['published_class'] for c in zero)}")


if __name__ == "__main__":
    main()
