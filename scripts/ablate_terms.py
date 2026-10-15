"""Term ablation of ADR-0020, measurement 1.

Runs the x4 global-minimum fit (published assignment fixed, as in
tests/integration/test_search_gates.py) under one of four objectives:

  i    compact support only: no bond-valence term, no volume constraint
  ii   i + bond valence (ADR-0016)
  iii  ii + volume (ADR-0017): the current code
  iv   iii without the integrality penalty (ADR-0015's pending removal;
       since ADR-0020 removed the penalty, iv is the current code)
  v    iii without bond valence: whether bond valence is needed once
       volume is in (the variants above add terms cumulatively)
  notxt the current code without the automatic T-X-T bridging band
       (ADR-0027), its comparison run
  derived the config regenerated from composition, charges, cited CN and
       the density-derived volume alone (ADR-0028), with the fixture's
       optimisation settings

and records Stage-1 convergence, tier 1 (topology class) and tier 2
(lattice, bounds) to <out>/<mineral>_<variant>.json.

`--impostor-only` instead scores topaz's round-2 impostor, the one point DFT
confirmed to be a different structure (ADR-0019), against the published
structure under every variant. It takes seconds.

    python scripts/ablate_terms.py diopside_camgsi2o6 iii
    python scripts/ablate_terms.py --impostor-only
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests" / "integration"))

from test_reference_structure_gate import _REFS, FIXTURES  # noqa: E402

import mizuho.objective.breakdown as _breakdown  # noqa: E402
import mizuho.objective.energy as _energy  # noqa: E402
from mizuho.diagnostics import bound_contacts, lattice_names  # noqa: E402
from mizuho.objective.energy import compute_error  # noqa: E402
from mizuho.optimize.driver import optimize_one_assignment  # noqa: E402
from mizuho.optimize.endpoints import topology_key  # noqa: E402
from mizuho.pipeline import build_sg_config  # noqa: E402
from mizuho.structure import assignments_to_sites  # noqa: E402

VARIANTS = ("i", "ii", "iii", "iv", "v", "notxt", "derived")
BUDGET = 4
SEED = 42
CONVERGED_MARGIN = 1e-3
TOPAZ_IMPOSTOR = ROOT / FIXTURES / "impostors" / "topaz_round2_seedfix.json"


def _apply(variant: str, base: dict) -> dict:
    """The fixture config with the variant's terms removed. Integrality has
    no config key (ADR-0002), so variant iv patches it out in-process."""
    base = dict(base)
    if variant == "i":
        base.pop("volume_constraint", None)
        base["bond_valence"] = False
    elif variant == "ii":
        base.pop("volume_constraint", None)
    elif variant == "v":
        base["bond_valence"] = False
    elif variant == "notxt":
        base["auto_tetrahedral_bridging"] = False
    off = variant == "iv"
    for mod in (_energy, _breakdown):
        if not hasattr(mod, "_integrality_penalty"):
            continue  # removed by ADR-0020: variant iv is now the current code
        if not hasattr(mod, "_integrality_penalty_original"):
            mod._integrality_penalty_original = mod._integrality_penalty
        mod._integrality_penalty = (
            (lambda cn, w, scale: 0.0) if off else mod._integrality_penalty_original
        )
    return base


def _config(name: str, variant: str):
    ref = _REFS[name]
    base = yaml.safe_load(open(ROOT / FIXTURES / f"{ref.fixture}.yaml"))
    if variant == "derived":
        from test_reference_structure_gate import _minimal_config
        base = dict(_minimal_config(base), optimization=base.get("optimization", {}))
    base = _apply(variant, base)
    cfg = build_sg_config(base, ref.sg)
    _, pv, asg = ref.build()
    return dict(cfg, wyckoff_sites=assignments_to_sites(asg)), np.asarray(pv), asg


def run_fit(name: str, variant: str, out: Path, seed: int = SEED) -> dict:
    cfg, pv_ref, asg = _config(name, variant)
    opt = dict(cfg.get("optimization", {}))
    opt["n_restarts"] = int(opt.get("n_restarts", 8) * BUDGET)
    opt["cma_maxiter"] = int(opt.get("cma_maxiter", 300) * BUDGET)
    opt["seed"] = seed
    t0 = time.time()
    best_p, best_e, _ = optimize_one_assignment(asg, cfg, opt, verbose=False, label=name)
    elapsed = time.time() - t0
    best_p = np.asarray(best_p)
    err_ref = compute_error(pv_ref, cfg)
    names = lattice_names(cfg["spacegroup"])
    record = {
        "mineral": name,
        "variant": variant,
        "budget": BUDGET,
        "seed": seed,
        "seconds": elapsed,
        "error_found": float(best_e),
        "error_published": float(err_ref),
        "stage1_converged": bool(best_e <= err_ref + CONVERGED_MARGIN),
        "tier1_same_class": topology_key(best_p, cfg) == topology_key(pv_ref, cfg),
        "lattice_deviation": {
            k: float((best_p[i] - pv_ref[i]) / pv_ref[i]) for i, k in enumerate(names)
        },
        "bound_contacts": [str(c) for c in bound_contacts(best_p, cfg)],
        "param_vector": [float(x) for x in best_p],
        "published_vector": [float(x) for x in pv_ref],
    }
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{name}_{variant}" if seed == SEED else f"{name}_{variant}_seed{seed}"
    (out / f"{stem}.json").write_text(json.dumps(record, indent=2))
    return record


def score_impostor() -> dict:
    rec = json.loads(TOPAZ_IMPOSTOR.read_text())
    imp = np.asarray(rec["param_vector"])
    rows = {}
    for variant in VARIANTS:
        cfg, pv_ref, _ = _config(rec["reference"], variant)
        rows[variant] = {
            "impostor": float(compute_error(imp, cfg)),
            "published": float(compute_error(pv_ref, cfg)),
        }
        rows[variant]["rejected"] = (
            rows[variant]["impostor"] > rows[variant]["published"] + CONVERGED_MARGIN)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mineral", nargs="?")
    ap.add_argument("variant", nargs="?", choices=VARIANTS)
    ap.add_argument("--out", default="gate-results/ablation")
    ap.add_argument("--impostor-only", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    if args.impostor_only:
        print(json.dumps(score_impostor(), indent=2))
        return
    if not (args.mineral and args.variant):
        ap.error("mineral and variant are required unless --impostor-only")
    rec = run_fit(args.mineral, args.variant, Path(args.out), args.seed)
    print(json.dumps({k: v for k, v in rec.items()
                      if k not in ("param_vector", "published_vector")}, indent=2))


if __name__ == "__main__":
    main()
