#!/usr/bin/env python3
"""Did each impostor relax to the same structure as its published structure?
(design review 2026-10-02, Q11). Both endpoints are the relax2 CONTCARs (or
the tight ones, see final_stage); the
same structure means all four of:

1. the same bond-graph topology class (mizuho.bond_graph, fixture cutoffs);
2. conventional lattice lengths within 0.5 % and angles within 0.5 deg;
3. every atom within 0.02 A of a same-element atom after the best alignment
   over the lattice's point operations and origin shifts (this absorbs the
   Euclidean-normalizer equivalences of ADR-0011);
4. total energies within 1 meV/atom.
"""

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import spglib
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "runs" / "dft_basin"   # replaced by --workdir
sys.path.insert(0, str(ROOT))
from mizuho import bond_graph as bg  # noqa: E402
from mizuho.config import augment_coupling_constraints  # noqa: E402

TOL_LAT, TOL_ANG, TOL_DISP, TOL_E = 0.005, 0.5, 0.02, 1e-3
Z = {"O": 8, "F": 9, "Mg": 12, "Al": 13, "Si": 14, "Ca": 20, "Ti": 22}


def read_poscar(path):
    lines = Path(path).read_text().splitlines()
    scale = float(lines[1])
    lat = np.array([[float(x) for x in lines[i].split()] for i in range(2, 5)]) * scale
    els, counts = lines[5].split(), [int(x) for x in lines[6].split()]
    start = 8 if lines[7].strip()[0] in "DdCc" else 9
    n = sum(counts)
    frac = np.array([[float(x) for x in lines[start + i].split()[:3]] for i in range(n)])
    species = [e for e, c in zip(els, counts) for _ in range(c)]
    return lat, frac, species


def conventional(lat, frac, species):
    nums = [Z[e] for e in species]
    clat, cfrac, cnums = spglib.standardize_cell((lat, frac, nums), to_primitive=False,
                                                 no_idealize=False, symprec=1e-3)
    inv = {v: k for k, v in Z.items()}
    return clat, cfrac % 1.0, [inv[z] for z in cnums]


def params(lat):
    a, b, c = np.linalg.norm(lat, axis=1)
    ang = lambda u, v: np.degrees(np.arccos(u @ v / np.linalg.norm(u) / np.linalg.norm(v)))
    return np.array([a, b, c]), np.array([ang(lat[1], lat[2]), ang(lat[0], lat[2]), ang(lat[0], lat[1])])


def max_displacement(A, B):
    """Smallest, over aligning operations, of the largest distance from an atom
    of A to the nearest same-element atom of B (A's lattice is used for both)."""
    lat, fa, sa = A
    _, fb, sb = B
    sym = spglib.get_symmetry((lat, [[0, 0, 0]], [1]), symprec=1e-3)
    sa, sb = np.array(sa), np.array(sb)
    el = sa[0]
    best = np.inf
    images = np.array(list(product((-1, 0, 1), repeat=3)))
    for R in sym["rotations"]:
        rb = (fb @ R.T) % 1.0
        for t in (rb[sb == el] - fa[0]):        # map B's atom onto A's first atom
            shifted = (rb - t) % 1.0
            worst = 0.0
            for e in set(sa):
                d = fa[sa == e][:, None, :] - shifted[sb == e][None, :, :]
                d -= np.round(d)
                cart = (d[:, :, None, :] + images[None, None]) @ lat
                worst = max(worst, np.linalg.norm(cart, axis=-1).min(axis=(1, 2)).max())
                if worst >= best:
                    break
            best = min(best, worst)
    return float(best)


def topology_class(name, lat, frac, species):
    fixture = name.split("__")[0]
    cfg = augment_coupling_constraints(yaml.safe_load(open(ROOT / "tests" / "fixtures" / f"{fixture}.yaml")))
    cut, ranges = bg.pair_cutoffs(cfg, set(species))
    cations = {e for e, q in cfg["ionic_charges"].items() if q > 0}
    crystal = bg.Crystal(lat, species, frac)
    topo = bg.topology(bg.build_graph(crystal, cut, cations))
    return topo, sum(len(v) for v in bg.grey_contacts(crystal, cut, ranges).values())


def final_stage(pub, imp):
    """relax2, or `tight` when both structures were relaxed again to
    |F| <= 0.002 eV/A (quartz: a soft mode leaves a 0.02 A spread at the
    agreed 0.01 eV/A, README.md)."""
    if all((WORK / n / "tight" / "CONTCAR").exists() for n in (pub, imp)):
        return "tight"
    return "relax2"


def stage_energy(name, stage):
    import re
    text = (WORK / name / stage / "OSZICAR").read_text()
    return float(re.findall(r"E0=\s*([-\d.Ee+]+)", text)[-1])


def main():
    import argparse
    global WORK
    ap = argparse.ArgumentParser(description="Judge each impostor's relaxed structure (Q11).")
    ap.add_argument("--workdir", default=str(WORK), help="default: runs/dft_basin")
    WORK = Path(ap.parse_args().workdir).resolve()
    ledger = {}
    for r in map(json.loads, (WORK / "ledger.jsonl").read_text().splitlines()):
        if r.get("ok"):
            ledger[(r["name"], r["stage"])] = r
    names = sorted({n for n, s in ledger if s == "relax2"})
    rows = []
    for pub in [n for n in names if n.endswith("__published")]:
        fixture = pub.split("__")[0]
        print(f"\n{fixture}:")
        for imp in [n for n in names if n.startswith(fixture + "__") and n != pub]:
            stage = final_stage(pub, imp)
            P = read_poscar(WORK / pub / stage / "CONTCAR")
            I = read_poscar(WORK / imp / stage / "CONTCAR")
            Pc, Ic = conventional(*P), conventional(*I)
            tp, _ = topology_class(pub, *P)
            ti, gi = topology_class(imp, *I)
            ep = stage_energy(pub, stage) / len(P[2])
            ei = stage_energy(imp, stage) / len(I[2])
            lp, ap = params(Pc[0])
            li, ai = params(Ic[0])
            sgi = spglib.get_symmetry_dataset((I[0], I[1], [Z[e] for e in I[2]]), symprec=1e-3).number
            same_topo = ti.signature == tp.signature
            dlat = float(np.max(np.abs(li - lp) / lp))
            dang = float(np.max(np.abs(ai - ap)))
            same_cell = len(Ic[2]) == len(Pc[2])
            disp = max_displacement(Pc, Ic) if same_cell and dlat < 0.05 else float("nan")
            de = ei - ep
            same = (same_topo and dlat <= TOL_LAT and dang <= TOL_ANG
                    and disp <= TOL_DISP and abs(de) <= TOL_E)
            conv = all(ledger[(imp, s)]["converged"] for s in ("relax1", "relax2"))
            rows.append({"impostor": imp, "stage": stage, "sg": sgi,
                         "same_topology": same_topo, "dlat": dlat, "dang": dang,
                         "max_disp": disp, "dE_meV_atom": 1000 * de, "grey": gi,
                         "relax_converged": conv, "same_structure": bool(same)})
            print(f"  {imp.split('__')[1]:22s} [{stage}] SG {sgi:>3}  "
                  f"topo {'same' if same_topo else 'DIFF'}  "
                  f"dlat {100 * dlat:6.2f}%  dang {dang:5.2f}  disp {disp:7.3f} A  "
                  f"dE {1000 * de:+8.2f} meV/atom  grey {gi}  "
                  f"{'SAME' if same else 'different'}{'' if conv else '  (relax not converged)'}")
    (WORK / "basins.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
