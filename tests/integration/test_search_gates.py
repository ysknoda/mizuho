"""The slow search gates of ADR-0005. Minutes to hours per mineral.

Run with `pytest --run-gates tests/integration/test_search_gates.py`.
Opening this file tells you it takes hours; that is why the fast gates
(reference structure, impostor regression) live beside `_REFS` instead.

- Rank: the published topology's equivalence class (ADR-0011) sits within
  the first `max_candidates` of Phase 1's ranking, judged on the WORST rank
  over five seeds, with the spread recorded (ADR-0012).
- Global minimum, reported separately: (1) the optimiser, given the
  published topology at x4 budget, reaches an error no worse than the
  published structure's + 1e-3; (2) tier 1, the pass: the point it
  reaches is in the published structure's bond-graph topology class, i.e.
  a DFT starting point that relaxes to the published structure
  (ADR-0019); (3) tier 2, the higher grade: every lattice length within
  3% of the published one and no lattice bound touched (ADR-0006).
  Stage 1 stops early on the default optimization.topology_grouping, as
  in production; set its early_stop to false to re-measure a budget
  ladder (ADR-0019, Q27).
- End-to-end proxy: under the same budget, the published topology beats
  the top five ranked topologies.

A gate that currently fails is written at its correct threshold and
marked xfail, never relaxed (ADR-0005). Wollastonite is excluded from
all three: its 51 free parameters put them out of reach.
"""

import json
import os
from functools import cache
from pathlib import Path

import numpy as np
import pytest
import yaml
from test_reference_structure_gate import _REFS, FIXTURES

from mizuho.objective.energy import compute_error
from mizuho.optimize.driver import optimize_one_assignment
from mizuho.pipeline import build_sg_config, search_and_rank
from mizuho.structure import assignments_to_sites

pytestmark = [pytest.mark.slow, pytest.mark.gate]

SEEDS = (42, 43, 44, 45, 46)
BUDGET = 4  # ADR-0005: the budget at which diopside provably reaches zero
CONVERGED_MARGIN = 1e-3
LATTICE_TARGET = 0.03
BOUND_FRACTION = 0.01  # ADR-0006

# Where each gate run writes what it measured (ranks per seed, fitted
# parameter vectors). The fitted vectors of a stage-2 failure are the
# impostors the regression gate pins.
RESULTS = Path(os.environ.get("MIZUHO_GATE_RESULTS", "gate-results"))

MINERALS = sorted(n for n in _REFS if n != "wollastonite_casio3")

# Recorded failures (ADR-0005): the correct threshold, xfail rather than relaxed.
RANK_XFAIL = {
    "adamite": "published topology ranked 2950 of 69,974 before ADR-0011/0012"
}


def _record(name: str, payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(payload, indent=2, default=float))


def _full_config(name):
    ref = _REFS[name]
    base = yaml.safe_load(open(f"{FIXTURES}/{ref.fixture}.yaml"))
    if name == "adamite":
        base["charge_balance_weight"] = 0.0  # undetermined H, see the reference gate
    cfg = build_sg_config(base, ref.sg)
    _, pv, asg = ref.build()
    cfg = dict(cfg, wyckoff_sites=assignments_to_sites(asg))
    return base, cfg, pv, asg


def _mark(name, table):
    return (
        pytest.param(name, marks=pytest.mark.xfail(reason=table[name], strict=False))
        if name in table
        else name
    )


# ---------------------------------------------------------------------------
# Rank
# ---------------------------------------------------------------------------

def _located_key(sg, assignment, located):
    """Equivalence-class key (ADR-0011) of the sites whose element is in
    `located`."""
    from mizuho.normalizer import canonical_key
    return canonical_key(sg, [(a.element, a.entry.letter) for a in assignment
                              if a.element in located])



@pytest.mark.parametrize("name", [_mark(n, RANK_XFAIL) for n in MINERALS])
def test_rank_gate_worst_over_five_seeds(name):
    base, _, _, asg = _full_config(name)
    sg = _REFS[name].sg
    # Elements the composition has but the reference does not locate
    # (adamite's hydrogen, calc_flag 'dum') cannot be matched, so classes
    # are compared on the located elements only.
    located = {a.element for a in asg}
    target = _located_key(sg, asg, located)
    n_max = base.get("tree_search", {}).get("max_candidates", 50)
    ranks = []
    for seed in SEEDS:
        sg_config = build_sg_config(base, sg)
        sg_config["optimization"] = dict(sg_config.get("optimization", {}), seed=seed)
        asgns, n_found, _ = search_and_rank(sg_config, max_cands=10**9)
        keys = [_located_key(sg, a, located) for a in asgns]
        ranks.append(keys.index(target) + 1 if target in keys else None)
    _record(
        f"rank_{name}",
        {
            "seeds": SEEDS,
            "ranks": ranks,
            "classes": len(keys),
            "enumerated": n_found,
            "max_candidates": n_max,
        },
    )
    assert None not in ranks, (
        f"{name}: published topology absent in some seeds: {ranks}"
    )
    print(
        f"\n  {name}: ranks {ranks}, worst {max(ranks)}, spread {max(ranks) - min(ranks)} "
        f"of {len(keys)} classes"
    )
    assert max(ranks) <= n_max


# ---------------------------------------------------------------------------
# Global minimum
# ---------------------------------------------------------------------------


@cache
def _global_minimum_fit(name):
    _, cfg, pv_ref, asg = _full_config(name)
    opt = dict(cfg.get("optimization", {}))
    opt["n_restarts"] = int(opt.get("n_restarts", 8) * BUDGET)
    opt["cma_maxiter"] = int(opt.get("cma_maxiter", 300) * BUDGET)
    best_p, best_e, _ = optimize_one_assignment(
        asg, cfg, opt, verbose=False, label=name
    )
    err_ref = compute_error(pv_ref, cfg)
    _record(
        f"global_minimum_{name}",
        {
            "error_found": best_e,
            "error_published": err_ref,
            "param_vector": list(best_p),
            "published_vector": list(pv_ref),
        },
    )
    return cfg, np.asarray(best_p), best_e, np.asarray(pv_ref), err_ref


@pytest.mark.parametrize("name", MINERALS)
def test_global_minimum_stage1_optimiser_converged(name):
    _, _, best_e, _, err_ref = _global_minimum_fit(name)
    assert best_e <= err_ref + CONVERGED_MARGIN, (
        f"{name}: optimiser stopped at {best_e:.6f}, published scores {err_ref:.6f}"
    )


@pytest.mark.parametrize("name", MINERALS)
def test_global_minimum_tier1_reached_the_published_topology_class(name):
    from mizuho.optimize.endpoints import topology_key

    cfg, best_p, _, pv_ref, _ = _global_minimum_fit(name)
    assert topology_key(best_p, cfg) == topology_key(pv_ref, cfg), (
        f"{name}: the reached point is not in the published topology class")


@pytest.mark.parametrize("name", MINERALS)
def test_global_minimum_tier2_reached_lattice_is_the_published_one(name):
    from mizuho.diagnostics import bound_contacts, lattice_names

    cfg, best_p, _, pv_ref, _ = _global_minimum_fit(name)
    names = lattice_names(cfg["spacegroup"])
    devs = {
        k: (best_p[i] - pv_ref[i]) / pv_ref[i]
        for i, k in enumerate(names)
        if k in ("a", "b", "c")
    }
    contacts = bound_contacts(best_p, cfg)
    assert not contacts, f"{name}: on a bound: {contacts}"
    # Judged by the INTEGER coordination number, not the soft one the
    # objective uses (ADR-0007; uniqueness plan phase G).
    from mizuho.objective.breakdown import error_breakdown
    from mizuho.objective.penalties import _integrality_applies
    charges = cfg.get("ionic_charges", {})
    for e in error_breakdown(best_p, cfg)["cn"]:
        if _integrality_applies(e["center_atom"], charges) and e["cn_per_atom"]:
            assert e["cn_min_target"] <= min(e["cn_per_atom"]) and \
                max(e["cn_per_atom"]) <= e["cn_max_target"], (
                f"{name}: {e['label']} integer CN {sorted(set(e['cn_per_atom']))}")
    assert max(abs(d) for d in devs.values()) <= LATTICE_TARGET, (
        f"{name}: lattice deviation "
        + ", ".join(f"{k} {100 * d:+.1f}%" for k, d in devs.items())
    )


# ---------------------------------------------------------------------------
# End-to-end proxy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", MINERALS)
def test_end_to_end_published_topology_beats_top_five(name):
    base, cfg, _, asg = _full_config(name)
    sg = _REFS[name].sg
    located = {a.element for a in asg}
    target = _located_key(sg, asg, located)
    opt = dict(cfg.get("optimization", {}))
    opt["n_restarts"] = int(opt.get("n_restarts", 8) * BUDGET)
    opt["cma_maxiter"] = int(opt.get("cma_maxiter", 300) * BUDGET)

    sg_config = build_sg_config(base, sg)
    ranked, _, seeds = search_and_rank(sg_config, max_cands=6)
    rivals = [(a, s) for a, s in zip(ranked, seeds) if _located_key(sg, a, located) != target][
        :5
    ]
    _, published_e, _ = optimize_one_assignment(
        asg, cfg, opt, verbose=False, label=name
    )
    rival_errors = []
    for a, s in rivals:
        rcfg = dict(sg_config, wyckoff_sites=assignments_to_sites(a))
        _, e, _ = optimize_one_assignment(
            a, rcfg, opt, verbose=False, shell_seed_info=s
        )
        rival_errors.append(e)
    _record(f"end_to_end_{name}", {"published": published_e, "rivals": rival_errors})
    assert published_e < min(rival_errors), (
        f"{name}: published {published_e:.6f} vs best rival {min(rival_errors):.6f}"
    )
