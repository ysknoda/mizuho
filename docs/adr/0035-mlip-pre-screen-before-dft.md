# A machine-learned potential ranks the zero-error classes before DFT

The Al-O-N composition-free run (ADR-0033) left 2,565 zero-error topology classes over twelve
compositions, and the three closest classes of Al7O9N, which never reached zero error. Zero error
means the bonding geometry is met, not that the structure is stable: ADR-0032's TiO2 classes
spread over 0-229 meV/atom in DFT. At ADR-0032's 0.9 h per 6-12 atom cell (and an estimated
3-5 h for these 18-21 atom cells) DFT cannot take them all. Design review of 2026-10-28/30
(Q14-Q27).

## Decision

- **Energy first (Q14).** Candidates are ranked by energy above the Al2O3-AlN tie line. Every
  charge-neutral Al-O-N composition is a Al2O3 + b AlN, so this compares all compositions, and
  any per-element energy shift (a reference or correction) cancels.
- **MACE medium-mpa-0 pre-screen (Q15, Q18).** `scripts/mlip_screen.py relax`, mace-torch 0.3.17,
  float64, CPU. Input: every zero-error class and, for compositions with none, the three
  lowest-error classes (Q21), as exported by `export_classes.py --best 3`; corundum
  (`corundum_H.cif`) and wurtzite AlN as the end members.
- **Relaxation (Q20)**, ADR-0032's protocol: symmetry kept (FixSymmetry, FrechetCellFilter) to
  0.01 eV/A, then every atom displaced by up to 0.03 A and relaxed without symmetry; a drop of
  more than 1 meV/atom marks a saddle and the lower structure is kept.
- **Screening (Q27).** A structure is excluded when either relaxation did not converge, or when
  two anions come within 2.0 A (molecular units: N2 1.10 A, N-O ~1.2-1.4 A, O-O ~1.2-1.5 A,
  against ionic contacts from about 2.5 A, corundum O-O 2.53 A). Al-Al under 2.5 A (corundum's
  face-sharing pair is 2.655 A) and energies more than 50 meV/atom below the tie line are flagged,
  not excluded. Exclusions and their reasons go to `excluded.csv`.
- **Duplicates (Q22)**: same composition, a pymatgen StructureMatcher match (defaults) and
  energies within 1 meV/atom are one structure; the lowest represents it and the others are
  listed.
- **No geometric pre-filter (Q24).** Layered and chain classes are relaxed like the rest; the
  energy judges them.
- **DFT in two stages (Q16, Q23, Q25 (iv)).** Stage 1: each composition's lowest structure plus
  the lowest overall to 20, with corundum and AlN, at ADR-0032's settings (PBEsol, ENCUT 520 eV,
  KSPACING 0.25 1/A, symmetry kept then rattled, Q17). Stage 2, if the MLIP and DFT orders agree:
  each composition's second and third plus the lowest overall to 50.

## Measured

- Model load 20 s; AlN single point -7.438 eV/atom; relaxed end members corundum -7.4799 and AlN
  -7.4410 eV/atom. Per 21-atom class 200-550 s on one worker (median 242 s), about 60 s per
  structure over six workers: about 41 h for the 2,570 structures.
- Of the first 983 classes, 666 (68 %) were saddles: their free relaxation fell 0.4-0.85 eV/atom
  in the first examples, to P1 or Cm. Five did not converge in 2000 steps; one of them,
  Al3O3N class 364 (SG 38), ended at -9.57 eV/atom, 2.1 eV/atom below the tie line, with
  Al-Al 1.82 A and an N-O bond of 1.34 A: a hole of the potential, which motivated Q27.
- On those 983: 100 excluded for anion-anion bonds (down to 1.106 A, an N2 unit), 5 for
  non-convergence; 902 left, 694 after merging duplicates. Nothing lay below the tie line.
