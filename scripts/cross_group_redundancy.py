"""How much of an exhaustive run re-finds what a higher group already found
(ADR-0038).

Three measurements on an exhaustive run's record (`spacegroup_search.record`,
one JSON line per finished group, written in run order, highest point-group
order first):

    record     Zero-error entries whose topology class (with the composition)
               was first recorded in an earlier group of strictly higher
               point-group order; the assignments holding them, their share
               of the run's time (each group's time split equally over its
               assignments), and the classes first found in a group only by
               assignments that also hold such a known class -- the classes
               a stop at the first known class could lose.
    promotion  For every entry whose detected group differs from its own
               (what `run`'s promotion stage re-optimises, ADR-0010): whether
               the run already holds the same assignment in the detected group.
    replay     Re-optimise selected recorded assignments exactly as the run
               did, recording every Stage-1 endpoint (time, error, class) in
               restart order and every Stage-2 / centring call (time, error
               and class before and after). One JSON line per assignment,
               resumable. Runs in one process.

    python scripts/cross_group_redundancy.py record runs/alon/alon.jsonl
    python scripts/cross_group_redundancy.py promotion runs/alon/alon.jsonl \
        runs/alon/alon.yaml
    python scripts/cross_group_redundancy.py replay runs/alon/alon.jsonl \
        runs/alon/alon.yaml --select select.json --out replay.jsonl
    python scripts/cross_group_redundancy.py summary replay.jsonl
"""

import argparse
import collections
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mizuho.optimize import driver, endpoints  # noqa: E402
from mizuho.pipeline import build_sg_config, max_formula_units  # noqa: E402
from mizuho.search import wyckoff_tree_search  # noqa: E402
from mizuho.search.base import reduced_formula  # noqa: E402
from mizuho.structure import (  # noqa: E402
    assignments_to_sites, build_structure, resolve_fixed_assignment,
)
from mizuho.symmetry import SUCCESS_TOLERANCE, point_group_order, to_mizuho  # noqa: E402

THRESHOLD = 1e-3  # optimization.topology_grouping's default (ADR-0005)


def load(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def is_zero(e):
    return e["error"] is not None and e["error"] <= THRESHOLD and bool(e["topology"])


def assignments(recs):
    """[(sg, index, composition, specs, entries, known)] in run order. An
    assignment's entries are consecutive in the record (one per topology
    class it reached, `run --top`); known[i] is True when entry i is a
    zero-error class first recorded in a group of strictly higher order."""
    first = {}  # (composition, topology) -> (point-group order, sg) of its first group
    out = []
    for r in recs:
        sg, order = r["sg"], point_group_order(r["sg"])
        groups = itertools.groupby(r["entries"],
                                   key=lambda e: (e.get("composition"), json.dumps(e["specs"])))
        for i, ((comp, specs), es) in enumerate(groups):
            es = list(es)
            known = [is_zero(e) and first.get((comp, e["topology"]), (-1,))[0] > order
                     for e in es]
            out.append((sg, i, comp, json.loads(specs), es, known))
        for e in r["entries"]:
            if is_zero(e):
                first.setdefault((e.get("composition"), e["topology"]), (order, sg))
    return out, first


def cmd_record(args):
    recs = load(args.record)
    asg, first = assignments(recs)
    c = collections.Counter()
    seconds = {r["sg"]: r["seconds"] for r in recs}
    n_asg = collections.Counter(a[0] for a in asg)
    t_known = 0.0
    at_risk = 0
    for sg, group in itertools.groupby(asg, key=lambda a: a[0]):
        group = list(group)
        order = point_group_order(sg)
        new_in = collections.defaultdict(list)
        for k, (_, _, comp, _, es, known) in enumerate(group):
            c["assignments"] += 1
            if any(is_zero(e) for e in es):
                c["assignments reaching zero error"] += 1
            if any(known):
                c["assignments with a known class"] += 1
                t_known += seconds[sg] / n_asg[sg]
            for e, kn in zip(es, known):
                c["entries"] += 1
                if not is_zero(e):
                    continue
                c["zero-error entries"] += 1
                o = first[(comp, e["topology"])]
                if kn:
                    c["  known from a higher order"] += 1
                elif o[1] == sg:
                    c["  first found in this group"] += 1
                    new_in[(comp, e["topology"])].append(k)
                else:
                    c["  known from an equal order"] += 1
        for key, ks in new_in.items():
            if all(any(group[k][5]) for k in ks):
                at_risk += 1
    c["detected higher"] = c["detected higher, zero error"] = 0
    for r in recs:
        for e in r["entries"]:
            d = e["detected_sg"]
            if d is not None and point_group_order(d) > point_group_order(r["sg"]):
                c["detected higher"] += 1
                c["detected higher, zero error"] += is_zero(e)
    total = sum(r["seconds"] for r in recs)
    for k, v in c.items():
        print(f"  {k:<40} {v}")
    print(f"  {'zero-error classes':<40} {len(first)}")
    print(f"  {'classes at risk (stop on a known class)':<40} {at_risk}")
    print(f"  {'time':<40} {total / 3600:.1f} h; assignments with a known class "
          f"{100 * t_known / total:.1f} % (equal split)")


def cmd_promotion(args):
    recs = load(args.record)
    base = yaml.safe_load(open(args.config))
    held = {(r["sg"], e.get("composition"),
             tuple(sorted((s["element"], s["letter"]) for s in e["specs"])))
            for r in recs for e in r["entries"] if e["params"] is not None}
    cfgs, c = {}, collections.Counter()
    for r in recs:
        sg = r["sg"]
        for e in r["entries"]:
            d = e["detected_sg"]
            if e["params"] is None or d is None or d == sg:
                continue
            c["promotions"] += 1
            comp = e.get("composition")
            if (sg, comp) not in cfgs:
                cfgs[(sg, comp)] = build_sg_config(base, sg, composition=comp)
            cfg = dict(cfgs[(sg, comp)],
                       wyckoff_sites=assignments_to_sites(resolve_fixed_assignment(sg, e["specs"])))
            st = build_structure(np.array(e["params"]), cfg)
            try:
                m = to_mizuho(st["lattice_matrix"], st["frac_coords"], st["species"], d,
                              symprec=SUCCESS_TOLERANCE)
            except ValueError:
                c["not expressible in the detected group"] += 1
                continue
            key = (d, comp, tuple(sorted((s["element"], s["letter"]) for s in m.specs)))
            hit = "already optimised" if key in held else "not in the run"
            c[hit] += 1
            c[f"{hit}, {'zero' if is_zero(e) else 'non-zero'} error"] += 1
    for k, v in c.items():
        print(f"  {k:<45} {v}")


def short(key):
    return None if key is None else hashlib.sha1(repr(key).encode()).hexdigest()[:12]


def cmd_replay(args):
    recs = load(args.record)
    base = yaml.safe_load(open(args.config))
    asg, first = assignments(recs)
    index = {(a[0], a[1]): a for a in asg}
    done = set()
    if Path(args.out).exists():
        done = {(d["sg"], d["index"]) for d in load(args.out)}

    # One cached class per point: the ledger and this log share it.
    orig_key, memo = endpoints.topology_key, {}

    def cached_key(x, config, *a, **k):
        b = np.asarray(x, float).tobytes()
        if b not in memo:
            memo.clear()
            memo[b] = orig_key(x, config, *a, **k)
        return memo[b]

    endpoints.topology_key = cached_key
    log, calls = [], []
    orig_add = endpoints.EndpointLedger.add

    def add(self, x, f, converged=True):
        t = time.perf_counter()
        stop = orig_add(self, x, f, converged)
        log.append((t, float(f), cached_key(x, self.config) if self.enabled else None))
        return stop

    endpoints.EndpointLedger.add = add
    for name in ("_stage2", "_centre"):
        fn = getattr(driver, name)

        def timed(*a, _fn=fn, _name=name, **k):
            # _stage2(obj, cfg, p, e, ...) and _centre(obj, cfg, centring_cfg, p, e, ...)
            cfg, p_in, e_in = (a[1], a[2], a[3]) if _name == "_stage2" else (a[1], a[3], a[4])
            t0 = time.perf_counter()
            out = _fn(*a, **k)
            dt = time.perf_counter() - t0
            calls.append([_name, dt, float(e_in), float(out[1]),
                          short(endpoints.topology_key(p_in, cfg)),
                          short(endpoints.topology_key(out[0], cfg))])
            return out

        setattr(driver, name, timed)

    mfu = {}
    for key in [tuple(x) for x in json.load(open(args.select))]:
        if key in done:
            continue
        sg, i, comp, specs, es, known = index[key]
        cfg = build_sg_config(base, sg, composition=comp)
        if comp is not None and "elements" in base:
            # As search_and_rank_compositions builds a composition's config:
            # the local-shell seed reads tree_search.max_formula_units.
            if sg not in mfu:
                enum = dict(base, spacegroup=sg)
                enum["tree_search"] = {k: v for k, v in base["tree_search"].items()
                                       if k != "max_candidates"}
                groups = collections.defaultdict(list)
                for a in wyckoff_tree_search(enum):
                    groups[reduced_formula(a, base["elements"])].append(a)
                mfu[sg] = {c: max_formula_units(c, g) for c, g in groups.items()}
            cfg["tree_search"] = dict(cfg.get("tree_search") or {},
                                      max_formula_units=mfu[sg].get(comp, 4))
        log.clear(); calls.clear()
        t0 = time.perf_counter()
        res = driver.optimize_assignment(resolve_fixed_assignment(sg, specs), cfg,
                                         base["optimization"], max_classes=args.top)
        total = time.perf_counter() - t0
        order = point_group_order(sg)
        got = [None if r.topology is None else repr(r.topology) for r in res]
        out = {
            "sg": sg, "index": i, "composition": comp, "total_s": total,
            "endpoints": [{"t": t - t0, "f": f,
                           "known": k is not None and first.get((comp, repr(k)), (-1,))[0] > order,
                           "key": short(k)} for t, f, k in log],
            "refine": calls,
            "record": [[short(e["topology"]), e["error"], kn] for e, kn in zip(es, known)],
            "got": [[short(k), r.error] for k, r in zip(got, res)],
            "reproduced": got == [e["topology"] for e in es] and all(
                abs(r.error - e["error"]) <= 1e-9 * max(1.0, abs(e["error"]))
                for r, e in zip(res, es)),
        }
        with open(args.out, "a") as fh:
            fh.write(json.dumps(out) + "\n")
        print(f"SG {sg} #{i} {comp}: {total:.1f} s, {len(log)} endpoints, "
              f"reproduced={out['reproduced']}", flush=True)


def cmd_summary(args):
    """Time shares over replayed assignments: Stage 2, centring (split by
    whether the refined point is at or below the threshold), and what a stop
    at the first known-class Stage-1 endpoint would save and lose."""
    rows = load(args.replay)
    tot = sum(d["total_s"] for d in rows)
    s2 = sum(c[1] for d in rows for c in d["refine"] if c[0] == "_stage2")
    ce_zero = sum(c[1] for d in rows for c in d["refine"] if c[0] == "_centre" and c[2] <= THRESHOLD)
    ce_nz = sum(c[1] for d in rows for c in d["refine"] if c[0] == "_centre" and c[2] > THRESHOLD)
    crossed = sum(1 for d in rows for c in d["refine"]
                  if c[0] == "_centre" and c[2] > THRESHOLD >= c[3])
    save = lost = 0
    for d in rows:
        ep = d["endpoints"]
        hits = [i for i, e in enumerate(ep) if e["known"] and e["f"] <= THRESHOLD]
        if hits:
            i = hits[0]
            save += d["total_s"] - ep[i]["t"]
            later = ({e["key"] for e in ep[i + 1:] if e["f"] <= THRESHOLD}
                     - {e["key"] for e in ep[:i + 1]})
            lost += bool(later)
    print(f"  assignments {len(rows)} (reproduced {sum(d['reproduced'] for d in rows)}), "
          f"{tot:.0f} s")
    print(f"  Stage 2 {100 * s2 / tot:.1f} %; centring at or below the threshold "
          f"{100 * ce_zero / tot:.1f} %, above it {100 * ce_nz / tot:.1f} % "
          f"({crossed} centring call(s) crossed the threshold downwards)")
    print(f"  stop at the first known class: saves {100 * save / tot:.1f} %, "
          f"{lost} assignment(s) lose a later zero-error class")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("record"); p.add_argument("record")
    p = sub.add_parser("promotion"); p.add_argument("record"); p.add_argument("config")
    p = sub.add_parser("replay"); p.add_argument("record"); p.add_argument("config")
    p.add_argument("--select", required=True, help="JSON list of [sg, assignment index]")
    p.add_argument("--out", required=True)
    p.add_argument("--top", type=int, default=3, help="`run --top` of the recorded run")
    p = sub.add_parser("summary"); p.add_argument("replay")
    args = ap.parse_args()
    {"record": cmd_record, "promotion": cmd_promotion, "replay": cmd_replay,
     "summary": cmd_summary}[args.cmd](args)


if __name__ == "__main__":
    main()
