"""Export one representative structure per zero-error topology class of
exhaustive-run records, as CIF and POSCAR, with a summary table.

Records of runs on the same composition (e.g. the rutile and anatase
fixtures) are merged: a class is one bond-graph topology key (ADR-0019),
whichever run found it. The representative is the class entry detected in
the highest point group, then the lowest error. Its structure is reduced to
the primitive cell and idealised by spglib at the success tolerance
(1e-3 A, ADR-0010), so no supercell is exported; the topology key of the
exported cell is recomputed and must equal the class's.

    python scripts/export_classes.py --out runs/classes/tio2 \\
        runs/exhaustive_adr0030/tio2_rutile.yaml:runs/exhaustive_adr0030/tio2_rutile.jsonl \\
        runs/exhaustive_adr0030/tio2_anatase.yaml:runs/exhaustive_adr0030/tio2_anatase.jsonl
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import spglib
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mizuho import bond_graph as bg
from mizuho.cli import _entries_from_record
from mizuho.pipeline import build_sg_config
from mizuho.symmetry import SUCCESS_TOLERANCE, point_group_order

SYMBOLS = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba"]
Z = {s: i + 1 for i, s in enumerate(SYMBOLS)}


def graph_of(lattice, species, frac, config):
    cut, _ = bg.pair_cutoffs(config, set(species))
    cations = {e for e, q in (config.get("ionic_charges") or {}).items() if q > 0}
    return bg.build_graph(
        bg.Crystal(np.asarray(lattice), list(species), np.asarray(frac)), cut, cations
    )


def collect(pairs, threshold):
    """{topology key: [(error, sg, entry, config, run name)]} over all records."""
    classes = defaultdict(list)
    for cfg_path, rec_path in pairs:
        config = yaml.safe_load(open(cfg_path))
        name = Path(rec_path).stem
        for line in open(rec_path):
            r = json.loads(line)
            for e in r["entries"]:
                if e["error"] is not None and e["error"] <= threshold and e["topology"]:
                    classes[e["topology"]].append(
                        (e["error"], int(r["sg"]), e, config, name)
                    )
    return classes


def primitive(structure):
    cell = (
        structure["lattice_matrix"],
        structure["frac_coords"],
        [Z[s] for s in structure["species"]],
    )
    lat, pos, nums = spglib.standardize_cell(
        cell, to_primitive=True, symprec=SUCCESS_TOLERANCE
    )
    data = spglib.get_symmetry_dataset((lat, pos, nums), symprec=SUCCESS_TOLERANCE)
    order = np.argsort(nums, kind="stable")  # by atomic number
    return (
        lat,
        pos[order] % 1.0,
        [SYMBOLS[n - 1] for n in np.asarray(nums)[order]],
        data,
    )


def write_poscar(path, title, lat, frac, species):
    els = list(dict.fromkeys(species))
    lines = [title, "1.0"] + ["  " + "  ".join(f"{x:15.10f}" for x in v) for v in lat]
    lines += [
        "  " + "  ".join(els),
        "  " + "  ".join(str(species.count(e)) for e in els),
        "Direct",
    ]
    lines += [
        "  " + "  ".join(f"{x:15.10f}" for x in f) + f"  {s}"
        for f, s in zip(frac, species)
    ]
    Path(path).write_text("\n".join(lines) + "\n")


def _symop(rot, trans):
    """'x,y+1/2,-z' from a rotation matrix and a translation."""
    from fractions import Fraction
    out = []
    for r, t in zip(rot, trans):
        terms = "".join(("+" if c > 0 else "-") + ("" if abs(c) == 1 else str(abs(c))) + v
                        for c, v in zip(r, "xyz") if c)
        f = Fraction(float(t) % 1.0).limit_denominator(12)
        if f:
            terms += f"+{f.numerator}/{f.denominator}"
        out.append(terms.lstrip("+"))
    return ",".join(out)


def write_cif(path, title, lat_p, frac_p, species_p):
    """CIF in the detected space group: spglib's standard conventional cell
    (ITA standard setting), the symmetry operations written out, and one
    atom per Wyckoff orbit with its letter and multiplicity."""
    nums = [Z[s] for s in species_p]
    prim = spglib.get_symmetry_dataset((lat_p, frac_p, nums), symprec=SUCCESS_TOLERANCE)
    std = (prim.std_lattice, prim.std_positions, prim.std_types)
    ds = spglib.get_symmetry_dataset(std, symprec=SUCCESS_TOLERANCE)
    lat, pos, types = std
    a, b, c = np.linalg.norm(lat, axis=1)
    ang = lambda u, v: np.degrees(np.arccos(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)))
    species = [SYMBOLS[t - 1] for t in types]
    counts = Counter(species)
    ops = list(dict.fromkeys(_symop(r, t) for r, t in zip(ds.rotations, ds.translations)))
    lines = [f"# {title}", f"# idealised by spglib at {SUCCESS_TOLERANCE} A; standard conventional cell",
             "data_" + title.split()[0],
             f"_chemical_formula_sum '{' '.join(f'{e}{n}' for e, n in counts.items())}'",
             f"_space_group_name_H-M_alt '{ds.international}'",
             f"_space_group_IT_number {ds.number}",
             f"_space_group_name_Hall '{ds.hall}'",
             f"_cell_length_a {a:.6f}", f"_cell_length_b {b:.6f}", f"_cell_length_c {c:.6f}",
             f"_cell_angle_alpha {ang(lat[1], lat[2]):.4f}", f"_cell_angle_beta {ang(lat[0], lat[2]):.4f}",
             f"_cell_angle_gamma {ang(lat[0], lat[1]):.4f}",
             f"_cell_volume {abs(np.linalg.det(lat)):.4f}",
             f"_cell_formula_units_Z {int(np.gcd.reduce(list(counts.values())))}",
             "loop_", "_space_group_symop_id", "_space_group_symop_operation_xyz"]
    lines += [f"{k} {op}" for k, op in enumerate(ops, 1)]
    lines += ["loop_", "_atom_site_label", "_atom_site_type_symbol", "_atom_site_symmetry_multiplicity",
              "_atom_site_Wyckoff_symbol", "_atom_site_fract_x", "_atom_site_fract_y",
              "_atom_site_fract_z", "_atom_site_occupancy"]
    seen = Counter()
    eq = np.asarray(ds.equivalent_atoms)
    for i in sorted(set(eq), key=lambda i: (types[i], i)):
        s = species[i]
        seen[s] += 1
        f = pos[i] % 1.0
        lines.append(f"{s}{seen[s]} {s} {int((eq == i).sum())} {ds.wyckoffs[i]} "
                     f"{f[0]:.8f} {f[1]:.8f} {f[2]:.8f} 1.0")
    Path(path).write_text("\n".join(lines) + "\n")


def check_cif(path, lat_p, frac_p, species_p, config, key):
    """Read the CIF back (mizuho's reader expands the operations) and
    require the same space group, composition per cell volume and topology."""
    from mizuho.io.cif import read_cif
    lat, frac, species = read_cif(path).expand()
    sg = spglib.get_symmetry_dataset((lat, frac, [Z[s] for s in species]), symprec=SUCCESS_TOLERANCE).number
    sg_p = spglib.get_symmetry_dataset((lat_p, frac_p, [Z[s] for s in species_p]),
                                       symprec=SUCCESS_TOLERANCE).number
    dens = len(species) / abs(np.linalg.det(lat))
    dens_p = len(species_p) / abs(np.linalg.det(lat_p))
    same_topo = repr(bg.topology(graph_of(lat, species, frac, config)).signature) == key
    if sg != sg_p or abs(dens / dens_p - 1) > 1e-6 or not same_topo:
        raise SystemExit(f"{path}: CIF read-back differs (SG {sg} vs {sg_p}, "
                         f"density ratio {dens / dens_p:.6f}, topology same: {same_topo})")
    return len(species)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("runs", nargs="+", help="config.yaml:record.jsonl, one per run")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--threshold", type=float, default=1e-3)
    a = p.parse_args(argv)

    classes = collect([r.split(":") for r in a.runs], a.threshold)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def rep_key(item):
        err, sg, e, _, _ = item
        det = e["detected_sg"] or sg
        return (-point_group_order(det), err)

    ordered = sorted(classes.items(), key=lambda kv: min(rep_key(i) for i in kv[1]))
    rows = []
    for n, (key, items) in enumerate(ordered, 1):
        err, sg, e, config, run = min(items, key=rep_key)
        sgc = build_sg_config(config, sg)
        st = _entries_from_record(sg, [e], sgc)[0][4]
        lat, frac, species, data = primitive(st)
        g = graph_of(lat, species, frac, sgc)
        topo = bg.topology(g)
        if repr(topo.signature) != key:
            raise SystemExit(
                f"class {n}: the primitive cell's topology differs from the class's"
            )
        comp = Counter(species)
        nfu = np.gcd.reduce(list(comp.values()))
        cations = {e for e, q in config["ionic_charges"].items() if q > 0}
        link = Counter()
        for i, c in bg.linkage(g).items():
            link.update(c)
        ncat = sum(comp[c] for c in cations)
        tag = f"class_{n:02d}_sg{data.number}"
        title = (
            f"{tag} {''.join(f'{e}{k}' for e, k in comp.items())} {data.international} "
            f"(from run {run}, optimised in SG {sg}, error {err:.2g})"
        )
        write_cif(out / f"{tag}.cif", title, lat, frac, species)
        n_conv = check_cif(out / f"{tag}.cif", lat, frac, species, sgc, key)
        write_poscar(out / f"{tag}.vasp", title, lat, frac, species)
        rows.append(
            {
                "class": n,
                "file": tag,
                "spacegroup": f"{data.international} ({data.number})",
                "atoms_primitive": len(species),
                "atoms_conventional": n_conv,
                "formula_units": int(nfu),
                "V_per_fu_A3": round(abs(np.linalg.det(lat)) / nfu, 3),
                "dimensionality": bg.dimensionality(g),
                "corner_edge_face_per_cation": "/".join(
                    f"{sum(v for k, v in link.items() if (k >= 3 if s == 3 else k == s)) / ncat:g}"
                    for s in (1, 2, 3)
                ),
                "entries": len(items),
                "runs": " ".join(sorted({i[4] for i in items})),
                "detected_groups": " ".join(
                    str(g_)
                    for g_, _ in Counter(
                        i[2]["detected_sg"] for i in items
                    ).most_common(6)
                ),
                "error": f"{err:.2g}",
            }
        )
    with open(out / "classes.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(
            f"{r['class']:>3} {r['spacegroup']:<14} atoms {r['atoms_primitive']:>3}  Z {r['formula_units']:>2}"
            f"  V/fu {r['V_per_fu_A3']:7.2f}  dim {r['dimensionality']}"
            f"  c/e/f {r['corner_edge_face_per_cation']:<10} n {r['entries']:>3}  {r['runs']}"
        )


if __name__ == "__main__":
    main()
