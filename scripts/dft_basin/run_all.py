#!/usr/bin/env python3
"""Relax every prepared structure in three stages (ISIF 3, fixed symmetry),
one at a time, recording each finished stage in ledger.jsonl so an interrupted run
resumes at the first unfinished stage.

    DFT_NP=4 nohup caffeinate -i -s python scripts/dft_basin/run_all.py \
        --workdir runs/dft_basin > runs/dft_basin/run_all.log 2>&1 &
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from best_frame import best_frame

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "runs" / "dft_basin"   # replaced by --workdir
MPIRUN = "/opt/homebrew/opt/open-mpi/bin/mpirun"
VASP = str(Path.home() / "vasp" / "vasp.6.6.1" / "bin" / "vasp_std")
NP = int(os.environ.get("DFT_NP", "4"))
# Design review 2026-10-02, Q12(b): a coarse stage finds the basin, then two
# relaxations at the agreed settings (Q10) finish it. Only the coarse stage
# overrides the INCAR; ENCUT and the functional are never changed.
STAGES = ("coarse", "relax1", "relax2")
PREVIOUS = {"coarse": None, "relax1": "coarse", "relax2": "relax1"}
OVERRIDES = {"coarse": {"KSPACING": "0.45", "EDIFF": "1E-5", "EDIFFG": "-0.02", "NSW": "300"}}


def done():
    if not (WORK / "ledger.jsonl").exists():
        return set()
    return {(r["name"], r["stage"]) for r in map(json.loads, (WORK / "ledger.jsonl").read_text().splitlines())
            if r.get("ok")}


def final_energy(oszicar: Path):
    lines = [l for l in oszicar.read_text().splitlines() if "F=" in l]
    return float(re.search(r"E0=\s*([-\d.Ee+]+)", lines[-1]).group(1)), len(lines)


# Design review 2026-10-02, Q13(a): every stage runs in chunks of CHUNK_NSW
# ionic steps, each restarted from the previous CONTCAR, so the plane-wave
# basis and the conjugate-gradient history are rebuilt for the current cell.
# With one long run the basis stays fixed at the starting cell, and the Pulay
# error then drives a large volume change (forsterite round2_seedfix collapsed
# from 292 to 250 A^3 at an unphysical -214.9 eV).
# Q14(a): shorter chunks, each restarted from its lowest-energy ionic step.
CHUNK_NSW = 10
MAX_CHUNKS = 20


def _stage_incar(name, stage):
    lines = []
    for line in (WORK / name / "INCAR").read_text().splitlines():
        key = line.split("=")[0].strip()
        if key in OVERRIDES.get(stage, {}):
            line = f"{key} = {OVERRIDES[stage][key]}"
        if key == "NSW":
            line = f"NSW = {CHUNK_NSW}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def run(name, stage):
    d = WORK / name / stage
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    prev = PREVIOUS[stage]
    start = WORK / name / "POSCAR" if prev is None else WORK / name / prev / "CONTCAR"
    if prev is None and (WORK / name / "coarse_part1" / "POSCAR.best").exists():
        # A coarse stage that reached MAX_CHUNKS continues from the best step
        # of its first part (diopside round2_seedfix).
        start = WORK / name / "coarse_part1" / "POSCAR.best"
    shutil.copy(start, d / "POSCAR")
    shutil.copy(WORK / name / "POTCAR", d / "POTCAR")
    (d / "INCAR").write_text(_stage_incar(name, stage))
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               PATH="/opt/homebrew/opt/open-mpi/bin:/opt/homebrew/bin:" + os.environ["PATH"])
    t0 = time.time()
    rc, converged, steps, chunks, energies = 0, False, 0, 0, []
    for chunk in range(1, MAX_CHUNKS + 1):
        chunks = chunk
        with open(d / "vasp.out", "a") as out:
            rc = subprocess.call([MPIRUN, "-np", str(NP), VASP], cwd=d, stdout=out,
                                 stderr=subprocess.STDOUT, env=env)
        if rc != 0 or not (d / "CONTCAR").exists():
            break
        e0, n = final_energy(d / "OSZICAR")
        steps += n
        energies.append(e0)
        converged = "reached required accuracy" in (d / "OUTCAR").read_text()
        for f in ("OSZICAR", "OUTCAR"):
            shutil.copy(d / f, d / f"{f}.{chunk:02d}")
        print(json.dumps({"name": name, "stage": stage, "chunk": chunk, "E0": e0,
                          "ionic_steps": n, "converged": converged}), flush=True)
        if converged:
            break
        # Restart from the best ionic step, not the last one (Q14(a)).
        k, e_best, text = best_frame(d)
        (d / "POSCAR").write_text(text)
        print(json.dumps({"name": name, "stage": stage, "chunk": chunk,
                          "restart_from_step": k + 1, "E0_restart": e_best}), flush=True)
    rec = {"name": name, "stage": stage, "rc": rc, "seconds": round(time.time() - t0, 1),
           "converged": converged, "chunks": chunks, "ionic_steps": steps,
           "ok": rc == 0 and converged}
    if energies:
        rec["E0"] = energies[-1]
        rec["E0_by_chunk"] = energies
    with open(WORK / "ledger.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
    return rec["ok"]


def main(names):
    finished = done()
    for name in names:
        for stage in STAGES:
            if (name, stage) in finished:
                continue
            if not run(name, stage):
                print(f"{name} {stage} failed; skipping its later stages", flush=True)
                break


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Relax every prepared structure (resumable).")
    ap.add_argument("names", nargs="*", help="structures to run (default: all)")
    ap.add_argument("--workdir", default=str(WORK),
                    help="directory written by prepare.py (default: runs/dft_basin)")
    args = ap.parse_args()
    WORK = Path(args.workdir).resolve()
    order = ["anatase", "quartz", "corundum", "forsterite", "topaz", "diopside"]
    all_names = sorted((p.name for p in WORK.iterdir() if (p / "meta.json").exists()),
                       key=lambda n: (order.index(n.split("_")[0]), "published" not in n, n))
    main(args.names or all_names)
