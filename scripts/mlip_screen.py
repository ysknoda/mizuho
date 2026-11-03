"""MLIP pre-screen of exported topology classes before DFT (design review
2026-10-28, Q14-Q25).

Two stages, each resumable:

  relax  Relax every selected class with a machine-learned potential (MACE
         medium-mpa-0 by default): symmetry kept (ase FixSymmetry +
         FrechetCellFilter) to fmax, then every atom displaced by up to
         `rattle` A and relaxed again without symmetry. A drop of more than
         `saddle_mev` meV/atom marks the symmetric result a saddle; the lower
         structure is kept. One JSON line per structure in the ledger; a
         rerun skips names already there. Corundum and wurtzite AlN are
         relaxed the same way as the tie-line end members.
  rank   Structures whose relaxation did not converge or that formed
         anion-anion bonds (< ANION_ANION_MIN) are excluded (excluded.csv);
         short Al-Al and energies far below the tie line are flagged.
         Energy above the Al2O3-AlN tie line for every structure (every
         charge-neutral Al-O-N composition is a Al2O3 + b AlN), duplicates
         merged (same composition, StructureMatcher match, energy within
         `dup_mev` meV/atom), and the DFT selection: each composition's
         `per_comp` lowest, then the lowest overall up to `n_select`.

    python scripts/mlip_screen.py relax --classes runs/alon/classes --out runs/alon/mlip -j 6
    python scripts/mlip_screen.py rank --out runs/alon/mlip --per-comp 1 --n-select 20

Every option can also come from a YAML file (--config); command-line flags win.
"""

import argparse
import csv
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

# Relaxation and comparison settings, all from the DFT protocol of ADR-0032
# (runs/dft_classes): EDIFFG -0.01 eV/A, 0.03 A rattle, 1 meV/atom.
DEFAULTS = {
    "model": "mace:medium-mpa-0",
    "fmax": 0.01,  # eV/A, ADR-0032's EDIFFG
    "steps": 2000,  # optimiser steps per stage; non-convergence is recorded
    "rattle": 0.03,  # A, ADR-0032
    "seed": 42,
    "saddle_mev": 1.0,  # meV/atom, ADR-0032
    "dup_mev": 1.0,  # meV/atom, ADR-0032's same-structure energy tolerance
    "near_misses": "zero-free",  # Q21: best classes only of compositions with no zero-error class
    "per_comp": 1,  # Q25 (iv), stage 1: each composition's lowest
    "n_select": 20,  # Q16: stage 1 size, references excluded
    "workers": 6,
    "symprec_keep": 1e-3,  # A, the success tolerance the classes were exported at (ADR-0010)
}

# Corundum: the reference CIF of the ADR-0003 gate (a 4.7606, c 12.9940 A).
CORUNDUM_CIF = REPO / "tests/fixtures/reference/corundum_H.cif"
# Wurtzite AlN starting cell, P6_3mc, Al (1/3, 2/3, 0), N (1/3, 2/3, u).
# a, c, u recalled from the common literature values (e.g. a 3.11, c 4.98 A,
# u 0.382), not read from a source in this environment: UNCERTAIN (~85 %).
# Only a starting point: the relaxation converges to the potential's own cell.
ALN_A, ALN_C, ALN_U = 3.112, 4.982, 0.382

# Screening of relaxed structures before ranking (design review 2026-10-30,
# Q27; ADR-0035). The potential can fall into unphysical holes far from its
# training data: one Al3O3N class reached -2.1 eV/atom below the tie line
# with Al-Al 1.82 A and an N-O bond of 1.34 A, its free relaxation unconverged.
# Anion-anion bonds mark molecular units (N2 1.10 A, N-O ~1.2-1.4 A,
# O-O ~1.2-1.5 A) outside the Al3+/O2-/N3- model the search assumed; ionic
# anion contacts start near 2.5 A (corundum O-O 2.53 A, corundum_H.cif).
# 2.0 A sits in that gap.
ANION_ANION_MIN = 2.0      # A, below it: excluded
# Corundum's face-sharing octahedra put Al at 2.655 A (corundum_H.cif; 2.683
# after MACE relaxation): shorter Al-Al is flagged, not excluded, since the
# energy already ranks a strained face-sharing pair down.
AL_AL_FLAG = 2.5           # A
# Far below the tie line is suspect rather than a discovery: flagged.
BELOW_TIE_LINE_FLAG = -50.0   # meV/atom
CONTACT_RANGE = 3.2        # A, neighbour search radius for the contact checks


# ----------------------------------------------------------------------
# Composition and tie-line arithmetic
# ----------------------------------------------------------------------


def counts_of(symbols):
    out = {}
    for s in symbols:
        out[s] = out.get(s, 0) + 1
    return out


def pseudobinary(counts):
    """(a, b) with Al_x O_y N_z = a Al2O3 + b AlN; raises if the composition
    is not charge-neutral with Al3+, O2-, N3- (then it is off the line)."""
    al, o, n = counts.get("Al", 0), counts.get("O", 0), counts.get("N", 0)
    if set(counts) - {"Al", "O", "N"}:
        raise ValueError(f"not an Al-O-N composition: {counts}")
    a, b = o / 3.0, float(n)
    if abs(2 * a + b - al) > 1e-9:
        raise ValueError(f"{counts} is not charge-neutral (3 Al != 2 O + 3 N)")
    return a, b


def energy_above_tie_line(e_per_atom, counts, e_al2o3, e_aln):
    """eV/atom above the Al2O3-AlN line, end members' energies per atom.
    Any per-element energy shift cancels: the composition is a fixed
    combination of the two end members."""
    a, b = pseudobinary(counts)
    n_atoms = sum(counts.values())
    return e_per_atom - (5 * a * e_al2o3 + 2 * b * e_aln) / n_atoms


def formula(counts):
    g = 0
    for v in counts.values():
        g = math.gcd(g, v)
    return "".join(
        f"{e}{'' if counts[e] // g == 1 else counts[e] // g}"
        for e in ("Al", "O", "N")
        if counts.get(e)
    )


# ----------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------


def read_classes(classes_dir, near_misses):
    """Rows of an export_classes.py classes.csv to relax, as
    [(name, composition, poscar path, mizuho error, within threshold)].

    near_misses: 'none' keeps zero-error classes only; 'zero-free' (Q21) adds
    the lowest-error classes of compositions with no zero-error class;
    'all' keeps every exported row."""
    classes_dir = Path(classes_dir)
    rows = list(csv.DictReader(open(classes_dir / "classes.csv")))
    has_zero = {
        r["composition"] for r in rows if r.get("within_threshold", "True") == "True"
    }
    out = []
    for r in rows:
        zero = r.get("within_threshold", "True") == "True"
        keep = (
            zero
            or near_misses == "all"
            or (near_misses == "zero-free" and r["composition"] not in has_zero)
        )
        if keep:
            out.append(
                (
                    r["file"].replace("/", "__"),
                    r["composition"],
                    classes_dir / f"{r['file']}.vasp",
                    float(r["error"]),
                    zero,
                )
            )
    return out


def reference_atoms():
    from ase.io import read
    from ase.spacegroup import crystal

    cor = read(CORUNDUM_CIF)
    aln = crystal(
        ["Al", "N"],
        [(1 / 3, 2 / 3, 0.0), (1 / 3, 2 / 3, ALN_U)],
        spacegroup=186,
        cellpar=[ALN_A, ALN_A, ALN_C, 90, 90, 120],
    )
    return {"ref_corundum": cor, "ref_AlN": aln}


# ----------------------------------------------------------------------
# Relaxation (worker side)
# ----------------------------------------------------------------------

_CALC = None


def make_calculator(spec):
    """'mace:<model>' or 'emt' (ase's EMT, for tests only)."""
    if spec == "emt":
        from ase.calculators.emt import EMT

        return EMT()
    if spec.startswith("mace:"):
        from mace.calculators import mace_mp

        return mace_mp(
            model=spec.split(":", 1)[1], default_dtype="float64", device="cpu"
        )
    raise ValueError(f"unknown calculator {spec!r}")


def _init_worker(spec):
    global _CALC
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass
    _CALC = make_calculator(spec)


def spacegroup_of(atoms, symprec):
    import spglib

    cell = (atoms.cell[:], atoms.get_scaled_positions(), atoms.numbers)
    ds = spglib.get_symmetry_dataset(cell, symprec=symprec)
    return None if ds is None else int(ds.number)


def relax_stage(atoms, calc, fmax, steps, keep_symmetry, symprec):
    """Relax positions and cell in place; returns (energy per atom, converged, steps)."""
    from ase.constraints import FixSymmetry
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE

    atoms.calc = calc
    atoms.set_constraint(FixSymmetry(atoms, symprec=symprec) if keep_symmetry else None)
    opt = FIRE(FrechetCellFilter(atoms), logfile=None)
    converged = bool(opt.run(fmax=fmax, steps=steps))
    n = opt.get_number_of_steps()
    atoms.set_constraint(None)
    return float(atoms.get_potential_energy()) / len(atoms), converged, n


def relax_one(task):
    """Both stages for one structure; a JSON-able ledger record."""
    name, atoms, meta, p = task
    t0 = time.time()
    rec = dict(
        meta,
        name=name,
        n_atoms=len(atoms),
        formula=formula(counts_of(atoms.get_chemical_symbols())),
        sg_start=spacegroup_of(atoms, p["symprec_keep"]),
    )
    try:
        e_sym, conv_sym, n_sym = relax_stage(
            atoms, _CALC, p["fmax"], p["steps"], True, p["symprec_keep"]
        )
        sym = atoms.copy()
        rng = np.random.default_rng(p["seed"])
        atoms.positions += rng.uniform(-p["rattle"], p["rattle"], atoms.positions.shape)
        e_free, conv_free, n_free = relax_stage(
            atoms, _CALC, p["fmax"], p["steps"], False, p["symprec_keep"]
        )
    except Exception as exc:  # a failed relaxation is recorded, the batch goes on
        rec.update(
            ok=False,
            error_message=f"{type(exc).__name__}: {exc}",
            seconds=round(time.time() - t0, 1),
        )
        return rec
    saddle = (e_sym - e_free) * 1000 > p["saddle_mev"]
    final = atoms if saddle else sym
    rec.update(
        ok=True,
        e_sym=e_sym,
        e_free=e_free,
        saddle=saddle,
        e_final=min(e_sym, e_free) if saddle else e_sym,
        converged_sym=conv_sym,
        converged_free=conv_free,
        steps_sym=n_sym,
        steps_free=n_free,
        sg_sym=spacegroup_of(sym, p["symprec_keep"]),
        sg_final_0p1=spacegroup_of(final, 0.1),  # ADR-0032 judges at 0.1 A
        cell=final.cell[:].tolist(),
        positions=final.positions.tolist(),
        symbols=final.get_chemical_symbols(),
        seconds=round(time.time() - t0, 1),
    )
    return rec


# ----------------------------------------------------------------------
# Stage drivers
# ----------------------------------------------------------------------


def read_ledger(path):
    path = Path(path)
    if not path.exists():
        return {}
    return {r["name"]: r for r in map(json.loads, path.read_text().splitlines()) if r}


def stage_relax(p):
    from ase.io import read

    out = Path(p["out"])
    out.mkdir(parents=True, exist_ok=True)
    ledger = out / "ledger.jsonl"
    done = read_ledger(ledger)
    tasks = []
    for name, atoms in reference_atoms().items():
        tasks.append(
            (
                name,
                atoms,
                {
                    "composition": formula(counts_of(atoms.get_chemical_symbols())),
                    "source": "reference",
                },
                p,
            )
        )
    for name, comp, path, err, zero in read_classes(p["classes"], p["near_misses"]):
        tasks.append(
            (
                name,
                read(path, format="vasp"),
                {
                    "composition": comp,
                    "source": str(path),
                    "mizuho_error": err,
                    "within_threshold": zero,
                },
                p,
            )
        )
    tasks = [t for t in tasks if t[0] not in done]
    if p.get("limit"):
        tasks = tasks[: p["limit"]]
    print(
        f"{len(done)} already in {ledger}; relaxing {len(tasks)} with {p['model']} "
        f"on {p['workers']} worker(s)",
        flush=True,
    )
    t0 = time.time()
    with (
        ProcessPoolExecutor(
            p["workers"], initializer=_init_worker, initargs=(p["model"],)
        ) as ex,
        open(ledger, "a") as fh,
    ):
        futs = [ex.submit(relax_one, t) for t in tasks]
        for k, f in enumerate(as_completed(futs), 1):
            rec = f.result()
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            status = (
                f"E {rec['e_final']:.4f} eV/atom{' saddle' if rec['saddle'] else ''}"
                if rec["ok"]
                else f"FAILED {rec['error_message']}"
            )
            print(
                f"[{k}/{len(tasks)}] {rec['name']} {rec['formula']} {status} "
                f"({rec['seconds']} s, {time.time() - t0:.0f} s total)",
                flush=True,
            )


def dedupe(records, dup_mev):
    """Groups of records that are one structure: same composition, a
    StructureMatcher match (pymatgen defaults) and energies within dup_mev
    meV/atom. Returns lists, each sorted lowest energy first."""
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Lattice, Structure

    sm = StructureMatcher()
    by_comp = {}
    for r in records:
        by_comp.setdefault(r["composition"], []).append(r)
    groups = []
    for recs in by_comp.values():
        recs = sorted(recs, key=lambda r: r["e_final"])
        reps = []  # (structure, group)
        for r in recs:
            st = Structure(
                Lattice(r["cell"]),
                r["symbols"],
                r["positions"],
                coords_are_cartesian=True,
            )
            for st0, g in reps:
                if abs(r["e_final"] - g[0]["e_final"]) * 1000 <= dup_mev and sm.fit(
                    st, st0
                ):
                    g.append(r)
                    break
            else:
                reps.append((st, [r]))
        groups.extend(g for _, g in reps)
    return groups


def select(ranked, per_comp, n_select):
    """[(record, reason)]: each composition's `per_comp` lowest, then the
    lowest overall until n_select; `ranked` is sorted by energy above the
    tie line."""
    chosen, seen = [], {}
    for r in ranked:
        k = seen.get(r["composition"], 0)
        if k < per_comp:
            chosen.append((r, f"composition #{k + 1}"))
            seen[r["composition"]] = k + 1
    picked = {id(r) for r, _ in chosen}
    for r in ranked:
        if len(chosen) >= n_select:
            break
        if id(r) not in picked:
            chosen.append((r, "overall"))
            picked.add(id(r))
    return sorted(chosen, key=lambda x: x[0]["e_hull"])


def shortest_contacts(rec):
    """Shortest Al-Al, anion-anion and Al-anion distances (A) of a ledger
    record's final structure, over every periodic image; 9.9 when none is
    within CONTACT_RANGE."""
    from ase import Atoms
    from ase.neighborlist import neighbor_list

    atoms = Atoms(rec["symbols"], positions=rec["positions"], cell=rec["cell"], pbc=True)
    i, j, d = neighbor_list("ijd", atoms, CONTACT_RANGE)
    cation = np.array([s == "Al" for s in rec["symbols"]])
    out = {}
    for key, mask in (("Al-Al", cation[i] & cation[j]), ("anion-anion", ~cation[i] & ~cation[j]),
                      ("Al-anion", cation[i] != cation[j])):
        out[key] = float(d[mask].min()) if mask.any() else 9.9
    return out


def screen(rec):
    """(exclusion reason or None, [flags]) for a relaxed candidate (Q27)."""
    c = rec["contacts"] = shortest_contacts(rec)
    flags = []
    if c["Al-Al"] < AL_AL_FLAG:
        flags.append(f"Al-Al {c['Al-Al']:.2f} A")
    if "e_hull" in rec and rec["e_hull"] * 1000 < BELOW_TIE_LINE_FLAG:
        flags.append(f"{rec['e_hull'] * 1000:.0f} meV/atom below the tie line")
    if not (rec["converged_sym"] and rec["converged_free"]):
        return "not converged", flags
    if c["anion-anion"] < ANION_ANION_MIN:
        return f"anion-anion {c['anion-anion']:.2f} A", flags
    return None, flags


def stage_rank(p):
    out = Path(p["out"])
    recs = list(read_ledger(out / "ledger.jsonl").values())
    refs = {r["name"]: r for r in recs if r.get("source") == "reference" and r["ok"]}
    if set(refs) != {"ref_corundum", "ref_AlN"}:
        raise SystemExit(
            "rank needs both references relaxed (run the relax stage first)"
        )
    e_cor, e_aln = refs["ref_corundum"]["e_final"], refs["ref_AlN"]["e_final"]
    cands = [r for r in recs if r.get("source") != "reference" and r["ok"]]
    failed = [r["name"] for r in recs if not r["ok"]]
    for r in cands:
        r["e_hull"] = energy_above_tie_line(
            r["e_final"], counts_of(r["symbols"]), e_cor, e_aln
        )
    excluded, kept = [], []
    for r in cands:
        why, r["flags"] = screen(r)
        (excluded if why else kept).append((r, why))
    with open(out / "excluded.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "composition", "reason", "e_above_tie_line_meV", "Al-Al_A",
                    "anion-anion_A", "flags"])
        for r, why in sorted(excluded, key=lambda x: x[0]["name"]):
            w.writerow([r["name"], r["composition"], why, round(r["e_hull"] * 1000, 2),
                        round(r["contacts"]["Al-Al"], 3), round(r["contacts"]["anion-anion"], 3),
                        "; ".join(r["flags"])])
    cands = [r for r, _ in kept]
    groups = dedupe(cands, p["dup_mev"])
    ranked = sorted((g[0] for g in groups), key=lambda r: r["e_hull"])
    members = {id(g[0]): g for g in groups}
    chosen = select(ranked, p["per_comp"], p["n_select"])
    reason = {id(r): why for r, why in chosen}

    fields = [
        "rank",
        "name",
        "composition",
        "e_above_tie_line_meV",
        "e_final_eV_atom",
        "saddle",
        "sg_sym",
        "sg_final_0p1",
        "n_atoms",
        "mizuho_error",
        "Al-Al_A",
        "anion-anion_A",
        "flags",
        "duplicates",
        "selected",
    ]
    with open(out / "ranking.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for k, r in enumerate(ranked, 1):
            w.writerow(
                {
                    "rank": k,
                    "name": r["name"],
                    "composition": r["composition"],
                    "e_above_tie_line_meV": round(r["e_hull"] * 1000, 2),
                    "e_final_eV_atom": round(r["e_final"], 6),
                    "saddle": r["saddle"],
                    "sg_sym": r["sg_sym"],
                    "sg_final_0p1": r["sg_final_0p1"],
                    "n_atoms": r["n_atoms"],
                    "mizuho_error": r.get("mizuho_error"),
                    "Al-Al_A": round(r["contacts"]["Al-Al"], 3),
                    "anion-anion_A": round(r["contacts"]["anion-anion"], 3),
                    "flags": "; ".join(r["flags"]),
                    "duplicates": " ".join(x["name"] for x in members[id(r)][1:]),
                    "selected": reason.get(id(r), ""),
                }
            )
    print(
        f"references: corundum {e_cor:.4f}, AlN {e_aln:.4f} eV/atom; "
        f"{len(excluded)} excluded (excluded.csv); "
        f"{len(cands)} structures -> {len(ranked)} after merging duplicates; "
        f"{len(failed)} failed{': ' + ', '.join(failed) if failed else ''}"
    )
    print(
        f"selected {len(chosen)} (per composition {p['per_comp']}, total {p['n_select']}):"
    )
    for r, why in chosen:
        print(
            f"  {r['composition']:<10} {r['e_hull'] * 1000:8.1f} meV/atom  SG {r['sg_final_0p1']:>3}  "
            f"{r['name']}  [{why}]" + (f"  FLAG {'; '.join(r['flags'])}" if r["flags"] else "")
        )
    return ranked, chosen


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("relax", "rank"))
    ap.add_argument("--config", help="YAML file supplying any of the options below")
    ap.add_argument("--classes", help="export_classes.py output directory (relax)")
    ap.add_argument("--out", help="output directory: ledger.jsonl, ranking.csv")
    ap.add_argument(
        "--model", help=f"'mace:<model>' or 'emt' (default {DEFAULTS['model']})"
    )
    ap.add_argument("-j", "--workers", type=int)
    ap.add_argument("--fmax", type=float, help="eV/A (default 0.01, ADR-0032)")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--rattle", type=float, help="A (default 0.03, ADR-0032)")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--saddle-mev", dest="saddle_mev", type=float)
    ap.add_argument("--dup-mev", dest="dup_mev", type=float)
    ap.add_argument(
        "--near-misses", dest="near_misses", choices=("none", "zero-free", "all")
    )
    ap.add_argument("--per-comp", dest="per_comp", type=int)
    ap.add_argument("--n-select", dest="n_select", type=int)
    ap.add_argument(
        "--limit", type=int, help="relax only the first N pending structures (timing)"
    )
    a = ap.parse_args(argv)
    p = dict(DEFAULTS)
    if a.config:
        import yaml

        p.update(yaml.safe_load(open(a.config)) or {})
    p.update({k: v for k, v in vars(a).items() if v is not None and k != "config"})
    if p["stage"] == "relax" and not p.get("classes"):
        ap.error("relax needs --classes")
    if not p.get("out"):
        ap.error("--out is required")
    return p


def main(argv=None):
    p = parse(argv)
    if p["stage"] == "relax":
        stage_relax(p)
    else:
        stage_rank(p)


if __name__ == "__main__":
    main()
