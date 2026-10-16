"""Experiment for ADR-0028 (c): does centring a zero-error point inside the
derived bands pull a strained lattice back?

F(x) = E(x) + mu * E_ideal(x), E_ideal the derived objective with every
derived bond band shrunk to d0 = R0 - b ln(V / CN) and every derived
polyhedron angle band to its ideal angle. L-BFGS-B from the end point.

Usage: centring_experiment.py RESULTS_DIR MU[,MU...] RESULT.json...
(RESULT.json from ablate_terms.py ... derived). IDEAL=bonds|angles|both.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests" / "integration"))
from ablate_terms import _config  # noqa: E402
from mizuho.constraints import _cn_range  # noqa: E402
from mizuho.data.bond_valence import B_BOND_VALENCE, r0  # noqa: E402
from mizuho.diagnostics import lattice_names  # noqa: E402
from mizuho.objective.energy import compute_error  # noqa: E402
from mizuho.structure import build_bounds  # noqa: E402
from mizuho.optimize.endpoints import topology_key  # noqa: E402

IDEAL = {4: 109.4712, 6: 90.0}


def ideal_config(cfg, base, which="both"):
    cns, q = base["coordination_numbers"], base["ionic_charges"]
    out = dict(cfg)
    bonds = []
    for b in cfg["bond_constraints"]:
        b = dict(b)
        if which in ("both", "bonds") and "[derived]" in b.get("label", "") and b["atom1"] in cns:
            lo, hi = _cn_range(cns[b["atom1"]])
            s0 = q[b["atom1"]] / ((lo + hi) / 2)
            d0 = r0(b["atom1"], int(q[b["atom1"]]), b["atom2"]) - B_BOND_VALENCE * math.log(s0)
            b["distance_min"] = b["distance_max"] = d0
        bonds.append(b)
    angles = []
    for a in cfg["angle_constraints"]:
        a = dict(a)
        if which in ("both", "angles") and a.get("label", "").startswith("X-") and "[derived]" in a["label"]:
            lo, hi = _cn_range(cns[a["atom_center"]])
            a["angle_min"] = a["angle_max"] = IDEAL[lo]
        angles.append(a)
    out["bond_constraints"], out["angle_constraints"] = bonds, angles
    return out


def main():
    S = Path(sys.argv[1])
    mus = [float(m) for m in sys.argv[2].split(",")]
    files = sys.argv[3:]
    import yaml
    from test_reference_structure_gate import _REFS, FIXTURES, _minimal_config
    for f in files:
        d = json.load(open(S / f))
        name = d["mineral"]
        cfg, pv, _ = _config(name, "derived")
        base = _minimal_config(yaml.safe_load(open(f"{ROOT}/{FIXTURES}/{_REFS[name].fixture}.yaml")))
        import os
        icfg = ideal_config(cfg, base, os.environ.get("IDEAL", "both"))
        bounds = build_bounds(cfg)
        names = lattice_names(cfg["spacegroup"])
        kref = topology_key(pv, cfg)
        for label, x0 in (("end", np.asarray(d["param_vector"])), ("published", np.asarray(pv))):
            for mu in mus:
                fun = lambda x: compute_error(x, cfg) + mu * compute_error(x, icfg)  # noqa: E731
                r = minimize(fun, x0, method="L-BFGS-B", bounds=bounds,
                             options={"maxiter": 3000, "maxfun": 200000})
                x = r.x
                dev0 = {k: round(100 * (x0[i] - pv[i]) / pv[i], 1) for i, k in enumerate(names)}
                dev1 = {k: round(100 * (x[i] - pv[i]) / pv[i], 1) for i, k in enumerate(names)}
                print(f"{f:40s} {label:9s} mu={mu:g} E {compute_error(x0, cfg):.1e}->{compute_error(x, cfg):.1e}"
                      f" Eideal {compute_error(x0, icfg):.1e}->{compute_error(x, icfg):.1e}"
                      f" T1 {topology_key(x0, cfg) == kref}->{topology_key(x, cfg) == kref}"
                      f" lattice% {dev0} -> {dev1}", flush=True)


main()
