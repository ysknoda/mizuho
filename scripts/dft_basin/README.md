# DFT basin test of the recorded impostors (ADR-0019)

Does a structure the objective scores like the published one relax, under DFT, to the published
structure? Each impostor and its published structure are relaxed with the same settings and the
two end points compared.

## Procedure

```bash
python scripts/dft_basin/prepare.py --workdir runs/dft_basin
DFT_NP=4 nohup caffeinate -i -s python scripts/dft_basin/run_all.py --workdir runs/dft_basin > runs/dft_basin/run_all.log 2>&1 &
python scripts/dft_basin/compare_basins.py --workdir runs/dft_basin
```

- `prepare.py` builds every structure exactly as the impostor regression gate does
  (`tests/integration/test_reference_structure_gate.py::_REFS`), reduces it to its primitive cell
  (spglib) and writes POSCAR, INCAR and POTCAR. The two diopside budget-ladder end points are the
  `ladder_*.json` records here.
- `run_all.py` relaxes each structure in three stages, `coarse` (KSPACING 0.45, EDIFF 1e-5,
  EDIFFG -0.02) then `relax1` and `relax2` at the agreed settings, each stage in chunks of 10
  ionic steps restarted from the chunk's lowest-energy step (`best_frame.py`). Finished stages
  are appended to `ledger.jsonl`; a restart resumes at the first unfinished one.
- `compare_basins.py` judges each impostor against its published structure and writes
  `basins.json` (copied here).

Agreed settings (design review 2026-10-02, Q7-Q11): PBEsol (`GGA = PS`), ENCUT 520 eV (1.3 x the
largest ENMAX, O and F at 400 eV), PAW PBE.64 potentials as in pymatgen's MPRelaxSet (Ca_sv,
Mg_pv, Ti_pv, Al, Si, O, F), KSPACING 0.25 /A, ISMEAR 0 with SIGMA 0.05, EDIFF 1e-6,
EDIFFG -0.01 eV/A, ISIF 3 with the input space group kept, KPAR 4.

**Same structure** means all four of: the same bond-graph topology class; conventional lattice
lengths within 0.5 % and angles within 0.5 deg; every atom within 0.02 A of a same-element atom
after the best alignment over the lattice's point operations and origin shifts; total energies
within 1 meV/atom.

## Deviations from the plan, all recorded

| what | why | affected |
| --- | --- | --- |
| a coarse stage before `relax1` (Q12) | KSPACING 0.25 made a 30-100-step relaxation take hours | all |
| chunks of 15 ionic steps, each a fresh run (Q13) | one long run keeps the plane-wave basis of the starting cell; the Pulay error then collapsed forsterite round2_seedfix from 292 to 250 A^3 at an unphysical -214.9 eV | from forsterite's impostors on |
| chunks of 10, restarted from the lowest-energy step (Q14) | the conjugate-gradient line search went uphill at the end of each 15-step chunk on diopside round2_seedfix and the next chunk started there | diopside round2_seedfix |
| POTIM 0.2, then 0.1 | the default 0.5 overshot on forsterite; both values only change the path | topaz published (0.2); forsterite impostors, topaz round2_seedfix, all of diopside (0.1) |
| quartz relaxed again to \|F\| <= 0.002 eV/A (`tight_quartz.sh`) | a soft mode of alpha-quartz leaves a 0.02 A spread at 0.01 eV/A; the published structure was first relaxed to an energy criterion (`relax3`: EDIFFG = +1e-5) | quartz, all three structures |

`tight_quartz.sh` expects `quartz_sio2__published/relax3`, made from `relax2/CONTCAR` with the
same INCAR and `EDIFFG = 1E-5`, `NSW = 100`.

## Result (`basins.json`)

| impostor | topology before relaxation | result |
| --- | --- | --- |
| anatase round1, round2_seedfix, round3_bvs | same | same structure |
| corundum round3_bvs, round4_volume | same | same structure |
| forsterite round2_seedfix, round4_volume | same | same structure |
| quartz round3_bvs, round4_volume | same | same structure (tight) |
| diopside ladder_x16, ladder_x8, round2_seedfix | different | same structure |
| topaz round2_seedfix | different | different minimum, +277 meV/atom, volume +29 % |

Twelve of thirteen impostors relax to the published structure; every same-class impostor does.
