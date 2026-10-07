#!/usr/bin/env python3
"""Write VASP inputs for the basin test of ADR-0018's hypothesis: does a
same-topology impostor relax (DFT, fixed space group) to the published
structure? Every structure is built exactly as the impostor regression gate
builds it (tests/integration/test_reference_structure_gate.py::_REFS), then
reduced to its primitive cell with spglib.

Settings (design review 2026-10-02, Q7-Q11): PBEsol (GGA = PS), ENCUT 520 eV
(1.3 x max ENMAX, O/F 400 eV), MPRelaxSet POTCARs (Ca_sv, Mg_pv, Ti_pv, Al,
Si, O, F; pymatgen MPRelaxSet.yaml), KSPACING 0.25 1/A, ISMEAR 0 SIGMA 0.05,
EDIFF 1e-6, EDIFFG -0.01 eV/A, ISIF 3 with the input symmetry kept,
KPAR 4. POTIM was added later per structure (see README.md).

    python scripts/dft_basin/prepare.py --workdir runs/dft_basin
"""

import json
import sys
from pathlib import Path

import numpy as np
import spglib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "integration"))
sys.path.insert(0, str(ROOT))
from test_reference_structure_gate import _REFS  # noqa: E402

from mizuho.structure import build_structure  # noqa: E402

HERE = Path(__file__).resolve().parent   # holds the ladder records
WORK = ROOT / "runs" / "dft_basin"        # replaced by --workdir
IMP = ROOT / "tests" / "fixtures" / "impostors"
POTCAR_DIR = Path.home() / "vasp" / "potpaw_PBE.64"
POTCAR = {"Ca": "Ca_sv", "Mg": "Mg_pv", "Ti": "Ti_pv", "Al": "Al", "Si": "Si", "O": "O", "F": "F"}
Z = {"O": 8, "F": 9, "Mg": 12, "Al": 13, "Si": 14, "Ca": 20, "Ti": 22}

INCAR = """SYSTEM = {name}
GGA = PS
PREC = Accurate
ENCUT = 520
LASPH = .TRUE.
ISPIN = 1
ISMEAR = 0
SIGMA = 0.05
KSPACING = 0.25
KGAMMA = .TRUE.
EDIFF = 1E-6
EDIFFG = -0.01
IBRION = 2
ISIF = 3
NSW = 200
NELM = 120
ALGO = Fast
LREAL = .FALSE.
KPAR = 4
LWAVE = .FALSE.
LCHARG = .FALSE.
"""

CASES = {
    "anatase_tio2": ["anatase_tio2", "anatase_tio2_round2_seedfix", "anatase_tio2_round3_bvs"],
    "quartz_sio2": ["quartz_sio2_round3_bvs", "quartz_sio2_round4_volume"],
    "corundum_al2o3": ["corundum_al2o3_round3_bvs", "corundum_al2o3_round4_volume"],
    "forsterite_mg2sio4": ["forsterite_mg2sio4_round4_volume", "forsterite_mg2sio4_round2_seedfix"],
    "topaz": ["topaz_round2_seedfix"],
    "diopside_camgsi2o6": ["HERE:ladder_diopside_camgsi2o6_x8", "HERE:ladder_diopside_camgsi2o6_x16",
                           "diopside_camgsi2o6_round2_seedfix"],
}


def write(name, cfg, pv, sg_expected):
    st = build_structure(np.asarray(pv, float), cfg)
    species = st["species"]
    numbers = [Z[e] for e in species]
    cell = (st["lattice_matrix"], st["frac_coords"], numbers)
    sg = spglib.get_symmetry_dataset(cell, symprec=1e-3).number
    lat, pos, nums = spglib.standardize_cell(cell, to_primitive=True, no_idealize=True, symprec=1e-3)
    sg_prim = spglib.get_symmetry_dataset((lat, pos, nums), symprec=1e-3).number
    inv = {v: k for k, v in Z.items()}
    order = sorted(set(nums), key=lambda z: z)          # by atomic number
    d = WORK / name
    d.mkdir(exist_ok=True)
    lines = [name, "1.0"] + ["  " + " ".join(f"{x:.10f}" for x in row) for row in lat]
    lines.append("  " + " ".join(inv[z] for z in order))
    lines.append("  " + " ".join(str(int(np.sum(nums == z))) for z in order))
    lines.append("Direct")
    for z in order:
        for p in pos[nums == z]:
            lines.append("  " + " ".join(f"{x % 1.0:.10f}" for x in p))
    (d / "POSCAR").write_text("\n".join(lines) + "\n")
    (d / "INCAR").write_text(INCAR.format(name=name))
    with open(d / "POTCAR", "w") as fh:
        for z in order:
            fh.write((POTCAR_DIR / POTCAR[inv[z]] / "POTCAR").read_text())
    meta = {"name": name, "sg_mizuho": cfg["spacegroup"], "sg_detected": int(sg),
            "sg_primitive": int(sg_prim), "n_conv": len(species), "n_prim": int(len(nums)),
            "param_vector": [float(x) for x in pv]}
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    flag = "" if sg == sg_prim == sg_expected else "  <-- symmetry differs"
    print(f"{name:45s} SG {cfg['spacegroup']:>3} det {sg:>3} prim {sg_prim:>3}  "
          f"{len(species):>3} -> {len(nums):>3} atoms{flag}")


def main():
    import argparse
    global WORK
    ap = argparse.ArgumentParser(description="Write the VASP inputs of the basin test.")
    ap.add_argument("--workdir", default=str(WORK),
                    help="Directory to write one sub-directory per structure into "
                         "(default: runs/dft_basin)")
    WORK = Path(ap.parse_args().workdir).resolve()
    WORK.mkdir(parents=True, exist_ok=True)
    for ref_name, imps in CASES.items():
        ref = _REFS[ref_name]
        cfg, pv, _ = ref.build()
        write(f"{ref_name}__published", cfg, pv, ref.sg)
        for imp in imps:
            if imp.startswith("HERE:"):
                tag = imp[5:]
                rec = json.loads((HERE / f"{tag}.json").read_text())
                tag = tag.replace("ladder_diopside_camgsi2o6_", "ladder_")
            else:
                rec = json.loads((IMP / f"{imp}.json").read_text())
                pub = rec.get("published_vector")
                assert pub is None or np.allclose(pub, pv, atol=1e-3), imp
                tag = imp[len(ref_name):].lstrip("_") or "round1"
            write(f"{ref_name}__{tag}", cfg, rec["param_vector"], ref.sg)


if __name__ == "__main__":
    main()
