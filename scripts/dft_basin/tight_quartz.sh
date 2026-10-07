#!/bin/zsh
# Q17 follow-up: relax quartz published and both impostors to |F| <= 0.002 eV/A
# (from relax2, or relax3 for the published structure) to test whether the
# 0.02 A spread is the force tolerance on a soft mode.
export PATH="/opt/homebrew/opt/open-mpi/bin:/opt/homebrew/bin:$PATH" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${1:-runs/dft_basin}"
for n in published round3_bvs round4_volume; do
  d=quartz_sio2__$n; src=$d/relax2; [ $n = published ] && src=$d/relax3
  mkdir -p $d/tight && cp $src/CONTCAR $d/tight/POSCAR && cp $d/POTCAR $d/tight/
  sed -e 's/^EDIFFG = .*/EDIFFG = -0.002/' -e 's/^EDIFF = .*/EDIFF = 1E-8/' -e 's/^NSW = .*/NSW = 100/' -e 's/^POTIM.*//' $d/INCAR > $d/tight/INCAR
  (cd $d/tight && mpirun -np 4 "${VASP_STD:-$HOME/vasp/vasp.6.6.1/bin/vasp_std}" > vasp.out 2>&1)
  echo "$n done: $(grep -c 'F=' $d/tight/OSZICAR) steps, converged=$(grep -c 'reached required' $d/tight/OUTCAR)"
done
