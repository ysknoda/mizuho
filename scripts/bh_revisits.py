"""How often does a basin-hopping chain revisit a topology class?

Tabu design (ADR-0029): independent restarts almost never revisit a
class, so a class tabu across restarts has nothing to act on. Inside a
chain it might. Records every local minimum the chain visits: its error,
its class, whether the hop was accepted.

    bh_revisits.py MINERAL VARIANT SEED NITER OUT.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import basinhopping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ablate_terms import _config  # noqa: E402
from mizuho.objective.energy import compute_error  # noqa: E402
from mizuho.optimize.endpoints import topology_key  # noqa: E402
from mizuho.structure import build_bounds  # noqa: E402

name, variant, seed, niter, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
cfg, pv, _ = _config(name, variant)
bounds = build_bounds(cfg)
lo, hi = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
rng = np.random.default_rng(seed)
x0 = rng.uniform(lo, hi)
kref = topology_key(pv, cfg)
keys, rows = {}, []
t0 = time.time()


def cb(x, f, accept):
    k = topology_key(np.clip(x, lo, hi), cfg)
    cid = keys.setdefault(k, len(keys))
    rows.append({"f": float(f), "cls": cid, "accept": bool(accept), "published": k == kref,
                 "t": round(time.time() - t0, 1)})
    Path(out).write_text(json.dumps({"mineral": name, "variant": variant, "seed": seed,
                                     "rows": rows}, indent=0))


basinhopping(lambda x: compute_error(x, cfg), x0, niter=niter, T=0.01, stepsize=0.1,
             minimizer_kwargs={"method": "L-BFGS-B", "bounds": list(bounds)},
             seed=seed, callback=cb)
