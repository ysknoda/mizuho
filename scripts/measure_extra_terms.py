"""Measure candidate objective terms on the zero-error entries of an
exhaustive-run record, without rerunning the search.

Every zero-error entry is rebuilt (mizuho.cli._entries_from_record, with its
composition's config) and scored under four candidate terms, each in the
functional form, weight (1.0) and normalisation of the existing term it
extends:

  bvs   Bond-valence sums at anion centres, |BVS - V| beyond tol = f |V|,
        scaled by |V| and averaged per element: the cation term of
        objective/bond_valence.py (ADR-0016, ADR-0028) applied to anions.
  aa    Anion-anion minimum distance d_min: repulsion_constraints O-O, O-N,
        N-N (objective/energy.py term 5), normalised by N(atom1).
  xalx  X-Al-X angle bands by each cation's actual CN: ADR-0028's CN 4
        [95, 125] deg and CN 6 [70, 110] deg, angles above 150 deg left out
        as compute_error does for both (xalx_t4: a tetrahedron's every angle
        held to its band); CN 5 gets none. Normalised by N(cation).
  axa   The automatic anion-centred band [80, a_max] evaluated over every
        angle (ignore_trans_above 180 instead of the default 150), as the
        increase over the configured anion angle term.

An entry survives a term (or a sum of terms) when its recorded error plus
the term's penalty stays at or below the zero-error threshold (1e-3). A class
survives when any of its zero-error entries does: the recorded points are
samples of the class's zero-error region, so a class counted as removed
because every sample fails is an upper bound on what the term would remove;
`reopt` tightens it by local re-optimisation from the recorded points.

    python scripts/measure_extra_terms.py score --config runs/alon/alon.yaml \\
        --record runs/alon/alon.jsonl --out OUT -j 6
    python scripts/measure_extra_terms.py verify --config ... --record ... --n 200
    python scripts/measure_extra_terms.py refs --config runs/alon/alon.yaml
"""

import argparse
import json
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from mizuho.cli import _entries_from_record, _run_entry_config
from mizuho.constraints import POLYHEDRON_ANGLES
from mizuho.lattice import (
    ZERO_SHIFT_INDEX,
    all_image_distances,
    all_image_vectors,
)
from mizuho.objective.bond_valence import bond_valence_term
from mizuho.objective.energy import compute_error
from mizuho.objective.penalties import (
    _midpoint_scale,
    _penalty,
    _soft_coordination_number,
)
from mizuho.pipeline import build_sg_config

THRESHOLD = 1e-3  # zero-error threshold of the run (ADR-0026)
# Bond-valence tolerance as a fraction of |V|: ADR-0028 chose 10 % for
# cations. 25 % and 30 % are included because a 2-coordinated O between CN-4
# Al cannot exceed 2 x 0.76 = 1.52 v.u. inside the derived bond band (s0 0.6
# +/- 0.16 v.u. at the range's CN 5), 24 % below its valence; N at CN 3 the
# same 2.28 against 3.
BVS_FRACTIONS = (0.10, 0.15, 0.20, 0.25, 0.30)
# Anion-anion minimum distance, A. Molecular units end near 1.5 A (O-O
# 1.2-1.5, N-O 1.2-1.4, N2 1.10); ionic contacts start near 2.5 A (corundum's
# shared-face O-O, corundum_H.cif). The hard-sphere term only acts below
# r_O + r_O = 1.32 A (penalties.HARD_RADII).
ANION_MIN = (2.0, 2.2, 2.3, 2.4, 2.5)
# Upper end of the anion-centred band, deg, evaluated over every angle; 160
# is the configured band (config.augment_coupling_constraints).
ANION_ANGLE_MAX = (150.0, 160.0, 170.0)
ANION_ANGLE_DEFAULT_TRANS = 150.0  # energy.py's ignore_trans_above default

TERM_NAMES = (
    [f"bvs_{f:.2f}" for f in BVS_FRACTIONS]
    + [f"aa_{d:.1f}" for d in ANION_MIN]
    + ["xalx", "xalx_t4"]
    + [f"axa_{a:.0f}" for a in ANION_ANGLE_MAX]
)


# ----------------------------------------------------------------------
# Constraint lookups in a built config
# ----------------------------------------------------------------------


def _charges(cfg):
    return cfg.get("ionic_charges") or {}


def _as_list(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


def cation_shell(cfg, cation):
    """(cutoff, anions) of the cation's derived coordination constraint."""
    for c in cfg["coordination_constraints"]:
        if c["center_atom"] == cation and c["cn_max"] > 0:
            return c["cutoff_distance"], _as_list(c["neighbor_atom"])
    raise ValueError(f"no coordination constraint for {cation}")


def anion_shell(cfg, anion):
    """Cutoff of the anion's coordination constraint over the cations."""
    for c in cfg["coordination_constraints"]:
        if c["center_atom"] == anion:
            return c["cutoff_distance"]
    raise ValueError(f"no coordination constraint for {anion}")


def anion_angle(cfg, anion):
    """The configured anion-centred angle constraint of `anion`."""
    for a in cfg.get("angle_constraints", []):
        if a["atom_center"] == anion:
            return a
    raise ValueError(f"no angle constraint for {anion}")


# ----------------------------------------------------------------------
# The candidate terms
# ----------------------------------------------------------------------


def bvs_penalty(values, valence, tol):
    """The cation bond-valence penalty of objective/bond_valence.py for one
    element: mean over its atoms of P(|BVS - V|, 0, tol), scaled by V."""
    if not len(values):
        return 0.0
    return sum(
        _penalty(abs(b - valence), 0.0, tol, 1.0, scale=float(valence)) for b in values
    ) / len(values)


def _angles(vectors):
    """Every angle (deg) between pairs of the given vectors."""
    k = len(vectors)
    if k < 2:
        return np.zeros(0)
    n = np.linalg.norm(vectors, axis=1)
    cos = np.clip(vectors @ vectors.T / (n[:, None] * n[None, :] + 1e-12), -1.0, 1.0)
    i, j = np.triu_indices(k, k=1)
    return np.degrees(np.arccos(cos[i, j]))


def _shells(frac, lat, centre_idx, neigh_idx, cutoff):
    """Per centre, the vectors to every neighbour image within `cutoff`
    (the angle term's convention: the identity image excluded)."""
    vecs, norms = all_image_vectors(frac[centre_idx], frac[neigh_idx], lat)
    out = []
    for li in range(len(centre_idx)):
        n = norms[li].ravel()
        out.append(vecs[li].reshape(-1, 3)[(n < cutoff) & (n > 1e-6)])
    return out


def describe(st, cfg):
    """Raw quantities and every candidate term's penalty for one structure."""
    lat = np.asarray(st["lattice_matrix"], float)
    frac = np.asarray(st["frac_coords"], float)
    sp = np.asarray(st["species"])
    q = _charges(cfg)
    present = sorted(set(sp.tolist()))
    cations = [e for e in present if q.get(e, 0) > 0]
    anions = [e for e in present if q.get(e, 0) < 0]
    out = {"n_atoms": len(sp), "terms": {}}

    # Bond-valence sums (the objective's own bonds and smoothing) and each
    # anion's soft CN over the cations at its coordination cutoff.
    bv = bond_valence_term(frac, lat, sp.tolist(), cfg)
    cat_idx = np.where(np.isin(sp, cations))[0]
    anion_rows = []
    for x in anions:
        idx = np.where(sp == x)[0]
        d = all_image_distances(frac[idx], frac[cat_idx], lat)
        cut = anion_shell(cfg, x)
        for i, b in zip(range(len(idx)), bv["anions"][x]["bvs"]):
            anion_rows.append(
                (
                    x,
                    round(_soft_coordination_number(d[i].ravel(), cut), 4),
                    round(float(b), 4),
                )
            )
    out["anions"] = anion_rows
    out["cation_bvs"] = {
        c: [round(float(b), 4) for b in v["bvs"]] for c, v in bv["cations"].items()
    }
    for f in BVS_FRACTIONS:
        out["terms"][f"bvs_{f:.2f}"] = sum(
            bvs_penalty(bv["anions"][x]["bvs"], -q[x], f * -q[x]) for x in anions
        )

    # Anion-anion contacts: repulsion_constraints O-O, O-N, N-N (atom1 <=
    # atom2 by name), every image, the identity image removed, normalised
    # by N(atom1); a like pair counts each contact from both ends, as
    # energy.py's term does.
    contacts = {}
    pair_d = []
    for a1 in anions:
        for a2 in anions:
            if a2 < a1:
                continue
            i1, i2 = np.where(sp == a1)[0], np.where(sp == a2)[0]
            dm = all_image_distances(frac[i1], frac[i2], lat)
            dm[i1[:, None] == i2[None, :], ZERO_SHIFT_INDEX] = np.inf
            d = dm[np.isfinite(dm)]
            pair_d.append((d, len(i1)))
            contacts[f"{a1}-{a2}"] = round(float(d.min()), 4) if len(d) else None
    out["anion_contacts"] = contacts
    for dmin in ANION_MIN:
        out["terms"][f"aa_{dmin:.1f}"] = sum(
            float(np.sum(((dmin - d[d < dmin]) / dmin) ** 2)) / n1 for d, n1 in pair_d
        )

    # Cation-centred angles by each cation's actual CN (ADR-0028's bands),
    # over the cation's coordination shell. "xalx" is the band as
    # compute_error applies it: a band without ignore_trans_above (CN 4)
    # still drops angles above energy.py's default 150 deg. "xalx_t4" also
    # holds a tetrahedron's angles above 150 deg to its band.
    xalx = {"xalx": 0.0, "xalx_t4": 0.0}
    cation_rows = []
    for c in cations:
        cut, neigh = cation_shell(cfg, c)
        ci = np.where(sp == c)[0]
        ni = np.where(np.isin(sp, neigh))[0]
        raw = dict.fromkeys(xalx, 0.0)
        for v in _shells(frac, lat, ci, ni, cut):
            th = _angles(v)
            band = POLYHEDRON_ANGLES.get(len(v))
            p = dict.fromkeys(xalx, 0.0)
            if band is not None:
                lo, hi, trans = band
                for name, t_max in (
                    ("xalx", trans or ANION_ANGLE_DEFAULT_TRANS),
                    ("xalx_t4", trans or 180.0),
                ):
                    p[name] = sum(
                        _penalty(t, lo, hi, 1.0, scale=_midpoint_scale(lo, hi))
                        for t in th[th <= t_max]
                    )
            for name in xalx:
                raw[name] += p[name]
            cation_rows.append(
                (
                    c,
                    len(v),
                    round(float(th.min()), 2) if len(th) else None,
                    round(float(th.max()), 2) if len(th) else None,
                    p["xalx"],
                )
            )
        for name in xalx:
            xalx[name] += raw[name] / max(len(ci), 1)
    out["cations"] = cation_rows
    out["terms"].update(xalx)

    # Anion-centred angles over every angle against [a_min, a_max], minus
    # the configured term (band [80, 160] up to ignore_trans_above).
    max_angle = {}
    deltas = {a: 0.0 for a in ANION_ANGLE_MAX}
    for x in anions:
        ac = anion_angle(cfg, x)
        lo, hi0 = ac["angle_min"], ac["angle_max"]
        trans0 = ac.get("ignore_trans_above", ANION_ANGLE_DEFAULT_TRANS)
        xi = np.where(sp == x)[0]
        th = [_angles(v) for v in _shells(frac, lat, xi, cat_idx, ac["angle_cutoff"])]
        allth = np.concatenate(th) if th else np.zeros(0)
        max_angle[x] = round(float(allth.max()), 2) if len(allth) else None
        old = sum(
            _penalty(t, lo, hi0, 1.0, scale=_midpoint_scale(lo, hi0))
            for t in allth[allth <= trans0]
        )
        for a in ANION_ANGLE_MAX:
            new = sum(
                _penalty(t, lo, a, 1.0, scale=_midpoint_scale(lo, a)) for t in allth
            )
            deltas[a] += (new - old) / max(len(xi), 1)
    out["anion_max_angle"] = max_angle
    for a in ANION_ANGLE_MAX:
        out["terms"][f"axa_{a:.0f}"] = deltas[a]
    return out


# ----------------------------------------------------------------------
# Record traversal and scoring
# ----------------------------------------------------------------------


def zero_entries(record_path, threshold=THRESHOLD):
    """[(sg, entry index within its group, entry)] of every zero-error entry."""
    out = []
    for line in open(record_path):
        r = json.loads(line)
        for k, e in enumerate(r["entries"]):
            if e["error"] is not None and e["error"] <= threshold and e["topology"]:
                out.append((int(r["sg"]), k, e))
    return out


def _score_group(task):
    sg, items, base = task
    rows = _entries_from_record(sg, [e for _, e in items], None, base)
    out = []
    for (k, e), row in zip(items, rows):
        full = _run_entry_config(row)
        d = describe(row[4], full)
        d.update(
            sg=sg,
            index=k,
            composition=e["composition"],
            topology=e["topology"],
            detected_sg=e["detected_sg"],
            error=e["error"],
            error_recomputed=compute_error(row[3], full),
        )
        out.append(d)
    return out


def cmd_score(a):
    base = yaml.safe_load(open(a.config))
    by_sg = defaultdict(list)
    for sg, k, e in zero_entries(a.record):
        by_sg[sg].append((k, e))
    tasks = [(sg, items, base) for sg, items in sorted(by_sg.items())]
    if a.limit:
        tasks = tasks[: a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0, n = time.time(), 0
    with open(out / "scores.jsonl", "w") as fh:
        if a.workers > 1:
            with ProcessPoolExecutor(a.workers) as ex:
                results = ex.map(_score_group, tasks)
        else:
            results = map(_score_group, tasks)
        for rows in results:
            fh.writelines(json.dumps(r) + "\n" for r in rows)
            n += len(rows)
    print(
        f"{n} zero-error entries scored in {time.time() - t0:.0f} s -> {out / 'scores.jsonl'}"
    )


# ----------------------------------------------------------------------
# Cross-check against compute_error for the terms a config can express
# ----------------------------------------------------------------------


def config_variants(full, d):
    """{term: config} for every term compute_error can evaluate itself:
    aa and axa always, xalx when every cation has the same banded CN."""
    q = _charges(full)
    anions = sorted({x for x, _, _ in d["anions"]})
    out = {}
    for dmin in ANION_MIN:
        reps = [
            {"atom1": a1, "atom2": a2, "distance_min": dmin, "weight": 1.0}
            for a1 in anions
            for a2 in anions
            if a2 >= a1
        ]
        out[f"aa_{dmin:.1f}"] = dict(full, repulsion_constraints=reps)
    for a in ANION_ANGLE_MAX:
        angles = [
            dict(ac, angle_max=a, ignore_trans_above=180.0)
            if q.get(ac["atom_center"], 0) < 0
            else ac
            for ac in full["angle_constraints"]
        ]
        out[f"axa_{a:.0f}"] = dict(full, angle_constraints=angles)
    cns = {(c, cn) for c, cn, _, _, _ in d["cations"]}
    if len(cns) == 1:
        c, cn = next(iter(cns))
        if cn in POLYHEDRON_ANGLES:
            lo, hi, trans = POLYHEDRON_ANGLES[cn]
            cut, neigh = cation_shell(full, c)
            extra = {
                "atom_center": c,
                "atom_neighbor": neigh,
                "angle_min": lo,
                "angle_max": hi,
                "angle_cutoff": cut,
                "weight": 1.0,
            }
            if trans is not None:
                extra["ignore_trans_above"] = trans
            out["xalx"] = dict(
                full, angle_constraints=full["angle_constraints"] + [extra]
            )
            extra = dict(extra, ignore_trans_above=trans or 180.0)
            out["xalx_t4"] = dict(
                full, angle_constraints=full["angle_constraints"] + [extra]
            )
    return out


def cmd_verify(a):
    base = yaml.safe_load(open(a.config))
    entries = zero_entries(a.record)
    rng = np.random.default_rng(a.seed)
    pick = sorted(rng.choice(len(entries), size=min(a.n, len(entries)), replace=False))
    worst, checked, nonzero = defaultdict(float), Counter(), Counter()
    for i in pick:
        sg, _, e = entries[i]
        row = _entries_from_record(sg, [e], None, base)[0]
        full, p, st = _run_entry_config(row), row[3], row[4]
        e0 = compute_error(p, full)
        d = describe(st, full)
        for term, cfg in config_variants(full, d).items():
            got = compute_error(p, cfg) - e0
            worst[term] = max(worst[term], abs(got - d["terms"][term]))
            checked[term] += 1
            nonzero[term] += got > 1e-12
        bv = bond_valence_term(
            np.asarray(st["frac_coords"]),
            np.asarray(st["lattice_matrix"]),
            st["species"],
            full,
        )
        for info in bv["cations"].values():
            mine = bvs_penalty(info["bvs"], info["valence"], info["tolerance"])
            worst["bvs, cation form"] = max(
                worst["bvs, cation form"], abs(mine - info["penalty"])
            )
            checked["bvs, cation form"] += 1
            nonzero["bvs, cation form"] += info["penalty"] > 1e-12
        worst["recorded error"] = max(worst["recorded error"], abs(e0 - e["error"]))
        checked["recorded error"] += 1
    for term in sorted(checked):
        print(
            f"  {term:<18} {checked[term]:>5} checked ({nonzero[term]:>4} non-zero), "
            f"largest |difference| {worst[term]:.3g}"
        )


# ----------------------------------------------------------------------
# End members
# ----------------------------------------------------------------------

# Wurtzite AlN, P6_3mc: a 3.112, c 4.982 A, u 0.382 (common literature
# values, as in scripts/mlip_screen.py and ADR-0033; not read from a source in
# this environment: UNCERTAIN, ~85 %).
ALN_CIF = """data_AlN
_cell_length_a 3.112
_cell_length_b 3.112
_cell_length_c 4.982
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 120
_space_group_IT_number 186
loop_
_space_group_symop_operation_xyz
x,y,z
-y,x-y,z
-x+y,-x,z
-x,-y,z+1/2
y,-x+y,z+1/2
x-y,x,z+1/2
-y,-x,z
-x+y,y,z
x,x-y,z
y,x,z+1/2
x-y,-y,z+1/2
-x,-x+y,z+1/2
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Al1 Al 0.333333 0.666667 0.0
N1 N 0.333333 0.666667 0.382
"""


def reference_scores(config_path):
    """{name: (objective error, describe())} for corundum and wurtzite AlN
    under the run's constraints, each with its own composition's config."""
    from mizuho.io.cif import cif_to_mizuho
    from mizuho.structure import build_structure

    base = yaml.safe_load(open(config_path))
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        aln = Path(tmp) / "aln.cif"
        aln.write_text(ALN_CIF)
        for name, path, comp in (
            ("corundum", REPO / "tests/fixtures/reference/corundum_H.cif", "Al2O3"),
            ("wurtzite AlN", aln, "AlN"),
        ):
            ref = cif_to_mizuho(str(path))
            cfg = build_sg_config(base, ref.spacegroup, composition=comp)
            cfg["wyckoff_sites"] = ref.config()["wyckoff_sites"]
            st = build_structure(ref.param_vector, cfg)
            out[name] = (compute_error(ref.param_vector, cfg), describe(st, cfg))
    return out


def cmd_refs(a):
    for name, (err, d) in reference_scores(a.config).items():
        print(f"{name}: objective {err:.3g}, {d['n_atoms']} atoms")
        bvs = defaultdict(list)
        for x, _, b in d["anions"]:
            bvs[x].append(b)
        print(
            "  anion CN",
            dict(Counter((x, round(c)) for x, c, _ in d["anions"])),
            " anion BVS",
            {x: (min(v), max(v)) for x, v in bvs.items()},
            " cation BVS",
            {c: (min(v), max(v)) for c, v in d["cation_bvs"].items()},
        )
        print(
            "  anion-anion shortest",
            d["anion_contacts"],
            " anion max angle",
            d["anion_max_angle"],
        )
        print("  cation CN, angle range", sorted({r[:4] for r in d["cations"]}))
        print("  " + "  ".join(f"{t} {v:.2g}" for t, v in d["terms"].items()))


# ----------------------------------------------------------------------
# Local re-optimisation of the classes a term removes at their recorded points
# ----------------------------------------------------------------------


def class_entries(scores_path):
    """{(composition, topology): [score rows]} from a `score` output."""
    out = defaultdict(list)
    for line in open(scores_path):
        d = json.loads(line)
        out[(d["composition"], d["topology"])].append(d)
    return out


def penalised(d, terms):
    """Recorded error plus the summed penalties of `terms`."""
    return d["error"] + sum(d["terms"][t] for t in terms)


def _reopt_one(task):
    """L-BFGS-B from one recorded point on the objective plus `terms`, as
    Phase 2's refinement (bounded, finite-difference gradient). Survives
    when the sum reaches the threshold in the same topology class."""
    from scipy.optimize import minimize

    from mizuho.optimize.endpoints import topology_key
    from mizuho.structure import build_bounds, build_structure

    sg, entry, base, terms, maxiter = task
    row = _entries_from_record(sg, [entry], None, base)[0]
    full, p0 = _run_entry_config(row), np.asarray(row[3], float)

    def objective(x):
        e = compute_error(x, full)
        if e >= 1e9:
            return e
        d = describe(build_structure(x, full), full)
        return e + sum(d["terms"][t] for t in terms)

    t0 = time.time()
    res = minimize(
        objective,
        p0,
        method="L-BFGS-B",
        bounds=build_bounds(full),
        options={"maxiter": maxiter, "ftol": 1e-10},
    )
    x = res.x
    end = objective(x)
    same = repr(topology_key(x, full)) == entry["topology"]
    return {
        "sg": sg,
        "start": objective(p0),
        "end": end,
        "base": compute_error(x, full),
        "same_topology": same,
        "survives": bool(end <= THRESHOLD and same),
        "nfev": int(res.nfev),
        "nit": int(res.nit),
        "status": int(res.status),  # 0 converged, 1 iteration limit
        "seconds": round(time.time() - t0, 1),
        "params": [float(v) for v in x],
    }


def cmd_reopt(a):
    terms = a.terms.split(",")
    unknown = set(terms) - set(TERM_NAMES)
    if unknown:
        raise SystemExit(f"unknown terms {sorted(unknown)}; known: {TERM_NAMES}")
    base = yaml.safe_load(open(a.config))
    record = defaultdict(dict)
    for sg, k, e in zero_entries(a.record):
        record[sg][k] = e
    classes = class_entries(a.scores)
    removed = sorted(
        ck
        for ck, ds in classes.items()
        if all(penalised(d, terms) > THRESHOLD for d in ds)
    )
    rng = np.random.default_rng(a.seed)
    chosen = (
        removed
        if not a.sample or a.sample >= len(removed)
        else [
            removed[i]
            for i in sorted(rng.choice(len(removed), size=a.sample, replace=False))
        ]
    )
    if a.include:
        wanted = {tuple(json.loads(line)) for line in open(a.include) if line.strip()}
        chosen = sorted(set(chosen) | (wanted & set(removed)))
    out = Path(a.out)
    done = set()
    if out.exists():
        done = {(r["composition"], r["topology"]) for r in map(json.loads, open(out))}
    todo = [ck for ck in chosen if ck not in done]
    print(
        f"{len(removed)} of {len(classes)} classes removed at their recorded points by "
        f"{'+'.join(terms)}; {len(chosen)} chosen, {len(todo)} to run"
    )
    tasks, owner = [], []
    for ck in todo:
        starts = sorted(classes[ck], key=lambda d: penalised(d, terms))[: a.starts]
        for d in starts:
            tasks.append((d["sg"], record[d["sg"]][d["index"]], base, terms, a.maxiter))
            owner.append(ck)
    results = defaultdict(list)
    left = Counter(owner)
    t0 = time.time()
    with open(out, "a") as fh, ProcessPoolExecutor(a.workers) as ex:
        for ck, res in zip(owner, ex.map(_reopt_one, tasks)):
            results[ck].append(res)
            left[ck] -= 1
            if left[ck] == 0:
                fh.write(
                    json.dumps(
                        {
                            "composition": ck[0],
                            "topology": ck[1],
                            "terms": terms,
                            "survives": any(r["survives"] for r in results[ck]),
                            "starts": results[ck],
                        }
                    )
                    + "\n"
                )
                fh.flush()
    n_ok = sum(any(r["survives"] for r in v) for v in results.values())
    print(f"{n_ok} of {len(results)} recovered in {time.time() - t0:.0f} s -> {out}")


# ----------------------------------------------------------------------
# Join with the exported classes and the MACE pre-screen (ADR-0035)
# ----------------------------------------------------------------------

LOW_MEV = 100.0  # meV/atom above the tie line: "low-energy" (MACE, final)
HIGH_MEV = 150.0  # meV/atom above the tie line: "high-energy"
# Primary threshold per term, for the combinations and the re-optimisation.
PRIMARY = ("bvs_0.20", "aa_2.3", "xalx", "axa_160")


def class_names(config_path, record_path, classes_dir, threshold=THRESHOLD, best=3):
    """{(composition, topology): exported name, e.g. Al3O3N__class_34_sg166},
    numbered as scripts/export_classes.py numbers them (its `collect`, the
    same order) and checked against its classes.csv."""
    import csv

    sys.path.insert(0, str(REPO / "scripts"))
    import export_classes as ec

    from mizuho.symmetry import point_group_order

    def rep_key(item):
        err, sg, e, _, _ = item
        if err > threshold:
            return (1, 0, err)
        return (0, -point_group_order(e["detected_sg"] or sg), err)

    classes = ec.collect([(config_path, record_path)], threshold, best)
    rows = {
        (r["composition"], int(r["class"])): r
        for r in csv.DictReader(open(Path(classes_dir) / "classes.csv"))
    }
    ordered = sorted(
        classes.items(), key=lambda kv: (kv[0][0] or "", min(rep_key(i) for i in kv[1]))
    )
    numbers, out = Counter(), {}
    for (comp, key), items in ordered:
        numbers[comp] += 1
        err, sg, e, _, _ = min(items, key=rep_key)
        r = rows[(comp, numbers[comp])]
        if (
            int(r["entries"]) != len(items)
            or r["error"] != f"{err:.2g}"
            or not r["file"].endswith(f"_sg{e['detected_sg'] or sg}")
        ):
            raise SystemExit(
                f"class numbering differs from classes.csv at {comp} {numbers[comp]}"
            )
        out[(comp, key)] = r["file"].replace("/", "__")
    return out


def mace_classes(mlip_dir, config_path):
    """{name: dict} for every relaxed class: energy above the tie line of the
    final and of the symmetric structure (meV/atom), saddle, convergence,
    the rank stage's exclusion and selection, and every term's penalty on
    the final (relaxed) structure under its composition's config."""
    import csv

    sys.path.insert(0, str(REPO / "scripts"))
    import mlip_screen as ms

    mlip = Path(mlip_dir)
    recs = ms.read_ledger(mlip / "ledger.jsonl")
    e_cor, e_aln = recs["ref_corundum"]["e_final"], recs["ref_AlN"]["e_final"]
    base = yaml.safe_load(open(config_path))
    cfgs, out = {}, {}
    for name, r in recs.items():
        if r.get("source") == "reference" or not r["ok"]:
            continue
        counts = ms.counts_of(r["symbols"])
        comp = r["composition"]
        if comp not in cfgs:
            cfgs[comp] = build_sg_config(base, 1, composition=comp)
        lat = np.asarray(r["cell"], float)
        frac = np.linalg.solve(lat.T, np.asarray(r["positions"], float).T).T % 1.0
        d = describe(
            {"lattice_matrix": lat, "frac_coords": frac, "species": r["symbols"]},
            cfgs[comp],
        )
        out[name] = {
            "saddle": bool(r["saddle"]),
            "converged": bool(r["converged_sym"] and r["converged_free"]),
            "e_hull": 1000
            * ms.energy_above_tie_line(r["e_final"], counts, e_cor, e_aln),
            "e_hull_sym": 1000
            * ms.energy_above_tie_line(r["e_sym"], counts, e_cor, e_aln),
            "excluded": "",
            "selected": "",
            "duplicate_of": "",
            "relaxed_terms": d["terms"],
            "relaxed_anion_cn": Counter(f"{x}{round(cn)}" for x, cn, _ in d["anions"]),
        }
    for r in csv.DictReader(open(mlip / "excluded.csv")):
        out[r["name"]]["excluded"] = r["reason"]
    for r in csv.DictReader(open(mlip / "ranking.csv")):
        out[r["name"]]["selected"] = r["selected"]
        for dup in r["duplicates"].split():
            out[dup]["duplicate_of"] = r["name"]
    return out


def categories():
    """(label, predicate on a relaxed class row) for the report's columns."""
    kept = lambda m: not m["excluded"]
    return [
        ("excl. anion-anion", lambda m: m["excluded"].startswith("anion-anion")),
        ("excl. not converged", lambda m: m["excluded"] == "not converged"),
        ("saddle", lambda m: m["saddle"]),
        (f"> {HIGH_MEV:.0f} meV", lambda m: kept(m) and m["e_hull"] > HIGH_MEV),
        (f"<= {LOW_MEV:.0f} meV", lambda m: kept(m) and m["e_hull"] <= LOW_MEV),
        ("stage-1 picks", lambda m: bool(m["selected"])),
    ]


def term_sets():
    """Single terms at every threshold, then the primary ones combined."""
    from itertools import combinations

    sets = [(t,) for t in TERM_NAMES]
    for k in range(2, len(PRIMARY) + 1):
        sets += list(combinations(PRIMARY, k))
    return sets


def _table(rows_out, header):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows_out]
    return "\n".join(lines)


def cmd_report(a):
    import csv

    names = class_names(a.config, a.record, a.classes)
    classes = class_entries(a.scores)
    mace = mace_classes(a.mlip, a.config)
    reopt = defaultdict(dict)  # term set -> {class key: survives}
    for path in a.reopt or []:
        for r in map(json.loads, open(path)):
            reopt["+".join(r["terms"])][(r["composition"], r["topology"])] = r[
                "survives"
            ]
    rows = []
    for ck, ds in classes.items():
        name = names[ck]
        rows.append(
            {
                "key": ck,
                "name": name,
                "composition": ck[0],
                "entries": ds,
                "mace": mace.get(name),
            }
        )
    relaxed = [r for r in rows if r["mace"]]
    cats = categories()
    totals = [sum(f(r["mace"]) for r in relaxed) for _, f in cats]
    md = [
        f"Zero-error classes: {len(rows)}; relaxed by MACE: {len(relaxed)}; "
        f"zero-error entries: {sum(len(r['entries']) for r in rows)}.",
        "",
    ]

    def removed_row(label, keep):
        cells = [label, sum(keep(r) for r in rows)]
        for (_, f), tot in zip(cats, totals):
            cells.append(f"{sum(f(r['mace']) and not keep(r) for r in relaxed)}/{tot}")
        energies = [
            r["mace"]["e_hull"]
            for r in relaxed
            if keep(r) and not r["mace"]["excluded"]
        ]
        cells.append(f"{np.median(energies):.0f}" if energies else "-")
        # Share of the kept classes at or below LOW_MEV; the unfiltered
        # share is the baseline a term must beat to select for energy.
        n_kept = sum(keep(r) for r in relaxed)
        n_low = sum(e <= LOW_MEV for e in energies)
        cells.append(f"{100 * n_low / n_kept:.1f}" if n_kept else "-")
        return cells

    header = (
        ["term(s)", "classes kept"]
        + [f"removed: {c}" for c, _ in cats]
        + ["median E kept (meV)", f"% kept <= {LOW_MEV:.0f} meV"]
    )
    md += ["## At the recorded points", ""]
    md.append(
        _table(
            [removed_row("none", lambda r: True)]
            + [
                removed_row(
                    "+".join(ts),
                    lambda r, ts=ts: any(
                        penalised(d, ts) <= THRESHOLD for d in r["entries"]
                    ),
                )
                for ts in term_sets()
            ],
            header,
        )
    )
    if reopt:
        md += ["", "## After local re-optimisation from the recorded points", ""]
        out_rows = []
        for tset, res in reopt.items():
            ts = tset.split("+")
            removed = [
                r
                for r in rows
                if all(penalised(d, ts) > THRESHOLD for d in r["entries"])
            ]
            missing = sum(r["key"] not in res for r in removed)

            def keep(r, ts=ts, res=res):
                return any(
                    penalised(d, ts) <= THRESHOLD for d in r["entries"]
                ) or res.get(r["key"], False)

            label = tset + (
                f" ({missing} removed classes not re-optimised)" if missing else ""
            )
            out_rows.append(removed_row(label, keep))
        md.append(_table(out_rows, header))

    # The terms on the relaxed structures, by energy band (kept classes).
    bands = [
        (-1e9, 75),
        (75, LOW_MEV),
        (LOW_MEV, HIGH_MEV),
        (HIGH_MEV, 300),
        (300, 1e9),
    ]
    kept = [r["mace"] for r in relaxed if not r["mace"]["excluded"]]
    head = ["term"] + [
        f"{'' if lo < -1e8 else f'{lo:.0f} < '}E{'' if hi > 1e8 else f' <= {hi:.0f}'}"
        f" (n={sum(lo < m['e_hull'] <= hi for m in kept)})"
        for lo, hi in bands
    ]
    trows = []
    for t in TERM_NAMES:
        cells = [t]
        for lo, hi in bands:
            s = [m for m in kept if lo < m["e_hull"] <= hi]
            cells.append(
                f"{np.mean([m['relaxed_terms'][t] <= THRESHOLD for m in s]):.2f}"
                if s
                else "-"
            )
        trows.append(cells)
    md += [
        "",
        "## Fraction of relaxed structures (not excluded) each term passes, by MACE energy",
        "",
    ]
    md.append(_table(trows, head))
    cn_rows = []
    for lo, hi in bands:
        c = Counter()
        for m in kept:
            if lo < m["e_hull"] <= hi:
                c.update(m["relaxed_anion_cn"])
        cn_rows.append(
            [f"{lo:.0f}..{hi:.0f}"]
            + [c.get(k, 0) for k in ("O2", "O3", "O4", "O5", "N3", "N4", "N5")]
        )
    md += ["", "## Anion CN of the relaxed structures (sites), by MACE energy", ""]
    md.append(
        _table(cn_rows, ["E band (meV/atom)", "O2", "O3", "O4", "O5", "N3", "N4", "N5"])
    )

    # Anion CN against BVS at the recorded points.
    sites = defaultdict(list)
    for r in rows:
        for d in r["entries"]:
            for x, cn, b in d["anions"]:
                sites[f"{x}{round(cn)}"].append(b)
    md += ["", "## Anion BVS by CN at the recorded points (all zero-error entries)", ""]
    md.append(
        _table(
            [
                [
                    k,
                    len(v),
                    f"{np.min(v):.2f}",
                    f"{np.median(v):.2f}",
                    f"{np.max(v):.2f}",
                ]
                for k, v in sorted(sites.items())
            ],
            ["anion CN", "sites", "BVS min", "median", "max"],
        )
    )
    for x, cn in (("O", 2), ("N", 3)):
        sel = [
            r
            for r in rows
            if all(
                any(s == x and round(c) == cn for s, c, _ in d["anions"])
                for d in r["entries"]
            )
        ]
        es = [
            r["mace"]["e_hull"] for r in sel if r["mace"] and not r["mace"]["excluded"]
        ]
        oth = [
            r["mace"]["e_hull"]
            for r in relaxed
            if r not in sel and not r["mace"]["excluded"]
        ]
        md.append(
            f"\nClasses with {x} at CN {cn} in every entry: {len(sel)}; MACE median "
            f"{np.median(es):.0f} meV/atom (others {np.median(oth):.0f}); "
            f"<= {LOW_MEV:.0f} meV: {sum(e <= LOW_MEV for e in es)} (others "
            f"{sum(e <= LOW_MEV for e in oth)})."
        )

    # The stage-1 picks.
    picks = sorted(
        (r for r in relaxed if r["mace"]["selected"]), key=lambda r: r["mace"]["e_hull"]
    )
    prow = []
    for r in picks:
        cells = [r["name"], f"{r['mace']['e_hull']:.1f}", r["mace"]["selected"]]
        for t in PRIMARY:
            rec = any(penalised(d, (t,)) <= THRESHOLD for d in r["entries"])
            ro = reopt.get(t, {}).get(r["key"])
            cells.append(
                "kept"
                if rec
                else (
                    "recovered"
                    if ro
                    else ("removed" if ro is False else "removed (not re-optimised)")
                )
            )
        cells.append(
            "pass"
            if all(r["mace"]["relaxed_terms"][t] <= THRESHOLD for t in PRIMARY)
            else ",".join(
                t for t in PRIMARY if r["mace"]["relaxed_terms"][t] > THRESHOLD
            )
        )
        prow.append(cells)
    md += ["", "## Stage-1 picks", ""]
    md.append(
        _table(
            prow,
            ["class", "E (meV/atom)", "why"] + list(PRIMARY) + ["relaxed structure"],
        )
    )

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(md) + "\n")
    with open(out / "classes_terms.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "name",
                "composition",
                "entries",
                "e_hull_meV",
                "e_hull_sym_meV",
                "saddle",
                "excluded",
                "selected",
                "duplicate_of",
            ]
            + [f"min_{t}" for t in TERM_NAMES]
            + [f"relaxed_{t}" for t in TERM_NAMES]
        )
        for r in rows:
            m = r["mace"] or {}
            w.writerow(
                [
                    r["name"],
                    r["composition"],
                    len(r["entries"]),
                    round(m["e_hull"], 2) if m else "",
                    round(m["e_hull_sym"], 2) if m else "",
                    m.get("saddle", ""),
                    m.get("excluded", ""),
                    m.get("selected", ""),
                    m.get("duplicate_of", ""),
                ]
                + [
                    f"{min(penalised(d, (t,)) for d in r['entries']):.3g}"
                    for t in TERM_NAMES
                ]
                + [f"{m['relaxed_terms'][t]:.3g}" if m else "" for t in TERM_NAMES]
            )
    print(
        f"{len(rows)} classes, {len(relaxed)} relaxed -> {out / 'report.md'}, "
        f"{out / 'classes_terms.csv'}"
    )


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="score every zero-error entry")
    s.add_argument("--config", required=True)
    s.add_argument("--record", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("-j", "--workers", type=int, default=1)
    s.add_argument(
        "--limit", type=int, default=0, help="first N space groups only (testing)"
    )
    v = sub.add_parser("verify", help="cross-check the terms against compute_error")
    v.add_argument("--config", required=True)
    v.add_argument("--record", required=True)
    v.add_argument("--n", type=int, default=200)
    v.add_argument("--seed", type=int, default=0)
    f = sub.add_parser("refs", help="corundum and wurtzite AlN under every term")
    f.add_argument("--config", required=True)
    o = sub.add_parser("reopt", help="re-optimise the classes a term set removes")
    o.add_argument("--config", required=True)
    o.add_argument("--record", required=True)
    o.add_argument("--scores", required=True, help="score's scores.jsonl")
    o.add_argument(
        "--terms", required=True, help="comma-separated, e.g. bvs_0.20,aa_2.3"
    )
    o.add_argument("--out", required=True, help="JSONL ledger, appended and resumed")
    o.add_argument(
        "--sample", type=int, default=0, help="random removed classes (0: all)"
    )
    o.add_argument("--include", help="JSON lines [composition, topology] always run")
    o.add_argument("--starts", type=int, default=1, help="recorded points per class")
    o.add_argument(
        "--maxiter", type=int, default=200, help="as the run's refine_maxiter"
    )
    o.add_argument("--seed", type=int, default=0)
    o.add_argument("-j", "--workers", type=int, default=1)
    r = sub.add_parser("report", help="join with the classes and the MACE pre-screen")
    r.add_argument("--config", required=True)
    r.add_argument("--record", required=True)
    r.add_argument("--scores", required=True, help="score's scores.jsonl")
    r.add_argument(
        "--classes", required=True, help="export_classes.py output directory"
    )
    r.add_argument(
        "--mlip",
        required=True,
        help="mlip_screen.py directory: ledger.jsonl, excluded.csv, ranking.csv",
    )
    r.add_argument("--reopt", nargs="*", help="reopt ledgers")
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    commands = {
        "score": cmd_score,
        "verify": cmd_verify,
        "refs": cmd_refs,
        "reopt": cmd_reopt,
        "report": cmd_report,
    }
    commands[a.cmd](a)


if __name__ == "__main__":
    main()
