"""Measure the bond-counting pruning of Phase 1 (ADR-0037) on a
composition-free run: how many assignments each condition removes, what it
costs, and whether it ever removes an assignment that reached zero error.

    phase1  every space group's Phase-1 enumeration without pruning,
            classified (kept / bond balance / site symmetry), and the
            enumeration timed with and without pruning; one JSON line per
            group in --ledger, resumable.
    record  every optimised (space group, assignment) of a run record,
            classified, crossed with whether any of its entries reached
            the zero-error threshold.
    verify  the premises at the recorded points: each entry's structure
            rebuilt, with its full atom count (no orbit collapsed onto a
            special position), every interplanar spacing wide enough for
            the objective's 27 images to hold every contact within reach,
            per-orbit bond counts equal over each orbit and obeying both
            conditions, and (zero-error entries) the coordination ranges
            the check derives met by those integer counts.

    python scripts/measure_bond_count_pruning.py phase1 --config runs/alon/alon.yaml \\
        --ledger phase1.jsonl
    python scripts/measure_bond_count_pruning.py record --config runs/alon/alon.yaml \\
        --record runs/alon/alon.jsonl
    python scripts/measure_bond_count_pruning.py verify --config runs/alon/alon.yaml \\
        --record runs/alon/alon.jsonl [--all]

Everything runs in one process.
"""

import argparse
import itertools
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mizuho.search.bond_counting import (
    BondCountCheck,
    _semigroup_counts,
    bond_stabiliser_orders,
    site_symmetry,
)

from mizuho.objective.penalties import GRAY_BAND_WIDTHS
from mizuho.pipeline import build_sg_config
from mizuho.search.base import reduced_formula
from mizuho.search.dfs import wyckoff_tree_search
from mizuho.structure import (
    assignments_to_sites,
    build_structure,
    resolve_fixed_assignment,
)

ZERO_ERROR = 1e-3  # optimize.endpoints.DEFAULT_THRESHOLD


def _enum_config(base: dict, sg: int, prune: bool) -> dict:
    """The enumeration config of pipeline.search_and_rank_compositions."""
    ts = {
        k: v
        for k, v in (base.get("tree_search") or {}).items()
        if k != "max_candidates"
    }
    ts["bond_count_pruning"] = prune
    return dict(base, spacegroup=sg, tree_search=ts)


def _key(asgn) -> tuple:
    return tuple(sorted((a.element, a.entry.letter) for a in asgn))


def cmd_phase1(args) -> None:
    base = yaml.safe_load(open(args.config))
    ledger = Path(args.ledger)
    done = set()
    if ledger.exists():
        done = {json.loads(line)["sg"] for line in ledger.read_text().splitlines()}
    for sg in range(1, 231):
        if sg in done:
            continue
        t0 = time.perf_counter()
        unpruned = wyckoff_tree_search(_enum_config(base, sg, False))
        t1 = time.perf_counter()
        pruned = wyckoff_tree_search(_enum_config(base, sg, True))
        t2 = time.perf_counter()
        check = BondCountCheck.from_config(
            _enum_config(base, sg, True), base["elements"], composition_free=True
        )
        reasons = Counter()
        per_comp = defaultdict(Counter)
        kept = set()
        for asgn in unpruned:
            why = check.rejection(asgn) if check else None
            reasons[why or "kept"] += 1
            per_comp[reduced_formula(asgn, base["elements"])][why or "kept"] += 1
            if why is None:
                kept.add(_key(asgn))
        got = {_key(a) for a in pruned}
        row = {
            "sg": sg,
            "n": len(unpruned),
            "kept": reasons["kept"],
            "balance": reasons["balance"],
            "site_symmetry": reasons["site_symmetry"],
            "seconds_unpruned": round(t1 - t0, 3),
            "seconds_pruned": round(t2 - t1, 3),
            "pruned_equals_filtered": got == kept,
            "compositions": {c: dict(v) for c, v in sorted(per_comp.items())},
        }
        with ledger.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(
            sg,
            row["n"],
            row["kept"],
            row["balance"],
            row["site_symmetry"],
            row["seconds_unpruned"],
            row["seconds_pruned"],
            row["pruned_equals_filtered"],
            flush=True,
        )
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    tot = Counter()
    for r in rows:
        for k in (
            "n",
            "kept",
            "balance",
            "site_symmetry",
            "seconds_unpruned",
            "seconds_pruned",
        ):
            tot[k] += r[k]
    print(
        json.dumps(
            {
                "groups": len(rows),
                **{k: round(v, 2) for k, v in tot.items()},
                "all_pruned_equal_filtered": all(
                    r["pruned_equals_filtered"] for r in rows
                ),
            }
        )
    )


def _record_assignments(path):
    """{(sg, sites): (composition, [entries])} of a run record."""
    out = {}
    for line in open(path):
        r = json.loads(line)
        for e in r["entries"]:
            sites = tuple((s["element"], s["letter"]) for s in e["specs"])
            out.setdefault((r["sg"], sites), (e["composition"], []))[1].append(e)
    return out


def cmd_record(args) -> None:
    base = yaml.safe_load(open(args.config))
    checks = {}
    table = Counter()
    rejected_zero = []
    for (sg, sites), (comp, entries) in sorted(
        _record_assignments(args.record).items()
    ):
        if sg not in checks:
            checks[sg] = BondCountCheck.from_config(
                _enum_config(base, sg, True), base["elements"], composition_free=True
            )
        asgn = resolve_fixed_assignment(
            sg, [{"element": e, "letter": le} for e, le in sites]
        )
        why = checks[sg].rejection(asgn) if checks[sg] else None
        zero = any(e["error"] is not None and e["error"] <= ZERO_ERROR for e in entries)
        table[(zero, why or "kept")] += 1
        if zero and why:
            rejected_zero.append((sg, sites, why))
    print(
        json.dumps(
            {
                f"{'zero' if z else 'nonzero'}/{w}": n
                for (z, w), n in sorted(table.items())
            },
            indent=1,
        )
    )
    print("zero-error assignments rejected:", len(rejected_zero))
    for r in rejected_zero[:20]:
        print("  ", r)


def _image_range(lat: np.ndarray, reach: float) -> np.ndarray:
    recip = np.linalg.inv(lat).T  # rows b_k with a_i . b_k = delta_ik
    n = [int(np.ceil(reach * np.linalg.norm(b))) + 1 for b in recip]
    return np.array(list(itertools.product(*(range(-k, k + 1) for k in n))))


def _counts(frac, lat, i_atoms, j_atoms, cutoff, shifts):
    """Bonds (distance < cutoff, identity excluded) from each atom of i_atoms
    to atoms of j_atoms."""
    out = []
    for a in i_atoms:
        d = (frac[j_atoms][:, None, :] + shifts[None, :, :] - frac[a]) @ lat
        r = np.linalg.norm(d, axis=2)
        out.append(int(np.sum((r < cutoff) & (r > 1e-8))))
    return out


def cmd_verify(args) -> None:
    base = yaml.safe_load(open(args.config))
    stats = Counter()
    failures = defaultdict(list)
    cfg_cache = {}
    for (sg, sites), (comp, entries) in sorted(
        _record_assignments(args.record).items()
    ):
        if (sg, comp) not in cfg_cache:
            cfg = build_sg_config(base, sg, composition=comp)
            check = BondCountCheck.from_config(
                cfg, base["elements"], composition_free=False
            )
            cfg_cache[(sg, comp)] = (cfg, check.rules_for(comp) if check else None)
        cfg, rules = cfg_cache[(sg, comp)]
        if rules is None:
            continue
        cut = dict(rules.cutoffs)
        half = GRAY_BAND_WIDTHS * 0.10
        reach = max(cut.values()) + half
        asgn = resolve_fixed_assignment(
            sg, [{"element": e, "letter": le} for e, le in sites]
        )
        mult = [a.entry.multiplicity for a in asgn]
        bounds = np.cumsum([0] + mult)
        orbit_atoms = [list(range(bounds[k], bounds[k + 1])) for k in range(len(asgn))]
        full = dict(cfg, wyckoff_sites=assignments_to_sites(asgn))
        for e in entries:
            zero = e["error"] is not None and e["error"] <= ZERO_ERROR
            if not (zero or args.all):
                continue
            stats["entries"] += 1
            stats["zero_entries"] += zero
            s = build_structure(np.array(e["params"]), full)
            frac, lat = np.asarray(s["frac_coords"]), np.asarray(s["lattice_matrix"])
            tag = (sg, sites, round(e["error"], 6))
            if len(frac) != bounds[-1]:
                failures["orbit_collapsed"].append(tag)
                continue
            spacing = 1.0 / np.linalg.norm(np.linalg.inv(lat).T, axis=1)
            if spacing.min() * 1.5 <= reach:
                failures["image_window_too_small"].append(
                    tag + (round(float(spacing.min()), 3),)
                )
            shifts = _image_range(lat, reach)
            n = {}
            for i, (ci, li) in enumerate(sites):
                for j, (aj, lj) in enumerate(sites):
                    if (ci, aj) not in cut:
                        continue
                    nij = _counts(
                        frac, lat, orbit_atoms[i], orbit_atoms[j], cut[(ci, aj)], shifts
                    )
                    nji = _counts(
                        frac, lat, orbit_atoms[j], orbit_atoms[i], cut[(ci, aj)], shifts
                    )
                    if len(set(nij)) > 1 or len(set(nji)) > 1:
                        failures["orbit_counts_differ"].append(tag)
                        continue
                    n[(i, j)] = (nij[0], nji[0])
                    if mult[i] * nij[0] != mult[j] * nji[0]:
                        failures["balance_identity"].append(tag)
                    allowed = _semigroup_counts(
                        bond_stabiliser_orders(sg, li, lj),
                        site_symmetry(sg, li).order,
                        nij[0],
                    )
                    if nij[0] not in allowed:
                        failures["site_symmetry_identity"].append(
                            tag + (li, lj, nij[0])
                        )
            if not zero:
                continue
            for r in rules.cation_rules:
                for i, (el, _) in enumerate(sites):
                    if el != r.centre:
                        continue
                    tot = sum(
                        v[0]
                        for (ii, j), v in n.items()
                        if ii == i and sites[j][0] in r.partners
                    )
                    if (r.lo is not None and tot < r.lo) or tot > r.hi:
                        failures["cation_range_at_zero_error"].append(tag + (i, tot))
            for r in rules.anion_rules:
                for j, (el, _) in enumerate(sites):
                    if el != r.centre:
                        continue
                    tot = sum(
                        v[1]
                        for (i, jj), v in n.items()
                        if jj == j and sites[i][0] in r.partners
                    )
                    if (r.lo is not None and tot < r.lo) or (
                        r.hi is not None and tot > r.hi
                    ):
                        failures["anion_range_at_zero_error"].append(tag + (j, tot))
    print(json.dumps(dict(stats)))
    print(json.dumps({k: len(v) for k, v in failures.items()}))
    for k, v in failures.items():
        print(k, v[:5])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser(
        "phase1", help="classify and time every group's Phase-1 enumeration"
    )
    a.add_argument("--config", required=True)
    a.add_argument(
        "--ledger", required=True, help="JSONL, one line per space group (resumable)"
    )
    b = sub.add_parser(
        "record", help="classify the optimised assignments of a run record"
    )
    b.add_argument("--config", required=True)
    b.add_argument("--record", required=True)
    c = sub.add_parser("verify", help="check the premises at the recorded points")
    c.add_argument("--config", required=True)
    c.add_argument("--record", required=True)
    c.add_argument(
        "--all", action="store_true", help="every entry, not only zero-error ones"
    )
    args = p.parse_args()
    {"phase1": cmd_phase1, "record": cmd_record, "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    main()
