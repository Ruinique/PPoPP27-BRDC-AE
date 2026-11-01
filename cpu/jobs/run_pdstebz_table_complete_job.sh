#!/usr/bin/env bash
set -euo pipefail

export I_MPI_ROOT="${I_MPI_ROOT:-/public/software/tools/intel/oneapi/mpi/2021.17}"  # AE: set I_MPI_ROOT to your Intel MPI
export PATH=$I_MPI_ROOT/bin:${PATH}
export LD_LIBRARY_PATH=$I_MPI_ROOT/lib:${MKLROOT:?set MKLROOT}/lib:${LD_LIBRARY_PATH:-}
export I_MPI_FABRICS=shm
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE

AE="${AE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
exe="${PDSTEBZ_EXE:-$AE/results/cpu/build_pdstebz/bench_pdstebz_parallel}"  # AE: built by run_extended_pdstebz_sweep_job.sh or: mpicc -O2 -I$MKLROOT/include cpu/bench/bench_pdstebz_parallel.c -L$MKLROOT/lib -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64 -lmkl_rt
reduced="$AE/data/inputs/reduced_dense_49152.de.bin"

mpirun -np 56 -genv I_MPI_FABRICS shm "${exe}" 16384 16384 all
mpirun -np 56 -genv I_MPI_FABRICS shm "${exe}" 49152 49152 reduced_dense "${reduced}"
