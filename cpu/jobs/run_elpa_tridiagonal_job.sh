#!/usr/bin/env bash
set -eo pipefail

export LC_ALL=C
export LANG=C

source /etc/profile >/dev/null 2>&1 || true
source "$HOME/.bashrc" >/dev/null 2>&1 || true
if command -v module >/dev/null 2>&1; then module purge >/dev/null 2>&1 || true || true; fi
if command -v module >/dev/null 2>&1; then module load gcc/11.2.0 intel/2021.3.0 python/3.8.13 tbb/2022.3 compiler-rt/2025.3.0 mkl/2025.3 mpi/2021.17 || true; fi

unset CPATH
unset C_INCLUDE_PATH
unset CPLUS_INCLUDE_PATH

export I_MPI_F90=ifort
export I_MPI_F77=ifort
export I_MPI_CC=icc
export I_MPI_CXX=g++
export MKL_INTERFACE_LAYER=LP64
export MKL_THREADING_LAYER=SEQUENTIAL
export MKL_DYNAMIC=FALSE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

AE="${AE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ROOT="$AE/cpu/bench"
PREFIX="$AE/results/deps/elpa-install-cpu-mpi"
SRC="$ROOT/bench_elpa_tridiagonal.f90"
EXE="$AE/results/cpu/bench_elpa_tridiagonal"
OUT="${1:-$AE/results/cpu/elpa_tridiag_results_${JH_JOBID:-manual}.csv}"
NP="${NP:-56}"

export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:$PREFIX/lib64/pkgconfig:${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib64:${MKLROOT}/lib:${LD_LIBRARY_PATH:-}"

echo "=== job info ==="
date
hostname
echo "OUT=$OUT"
echo "NP=$NP"
echo "PKG_CONFIG_PATH=$PKG_CONFIG_PATH"
pkg-config --cflags --libs elpa
ELPA_MODULE_DIR="$(find "$PREFIX/include" -type f -name elpa.mod -printf '%h\n' | head -1)"
echo "ELPA_MODULE_DIR=$ELPA_MODULE_DIR"

echo "=== compile ==="
mpiifort -I"$ELPA_MODULE_DIR" "$SRC" $(pkg-config --cflags --libs elpa) \
  -L"${MKLROOT}/lib" -Wl,-rpath,"${MKLROOT}/lib" \
  -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64 \
  -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl \
  -o "$EXE"

printf 'method,mode,n,ranks,nprow,npcol,nblk,reps,time_ms,checksum\n' > "$OUT"

run_case() {
  local n="$1"
  local mode="$2"
  local reps="$3"
  local limit="$4"
  local tmp="$AE/results/cpu/elpa_case_${n}_${mode}_${JH_JOBID:-manual}.tmp"

  echo "=== case n=$n mode=$mode reps=$reps timeout=${limit}s ==="
  if timeout "$limit" mpirun -np "$NP" \
      -genv I_MPI_FABRICS shm \
      -genv OMP_NUM_THREADS 1 \
      -genv MKL_NUM_THREADS 1 \
      -genv MKL_DYNAMIC FALSE \
      "$EXE" "$n" "$mode" "$reps" > "$tmp"; then
    cat "$tmp"
    awk 'NR == 2 {print}' "$tmp" >> "$OUT"
  else
    status=$?
    echo "case failed_or_timed_out status=$status n=$n mode=$mode"
    printf 'ELPA_SOLVE_TRIDIAGONAL_TIMEOUT,%s,%s,%s,0,0,64,%s,nan,nan\n' \
      "$mode" "$n" "$NP" "$reps" >> "$OUT"
  fi
}

run_case 8192 uniform 1 600
run_case 8192 clustered 1 600
run_case 16384 uniform 1 900
run_case 16384 clustered 1 900

if [ "${RUN_LARGE:-0}" = "1" ]; then
  run_case 32768 uniform 1 1800
  run_case 32768 clustered 1 1800
fi

echo "=== results ==="
cat "$OUT"
echo "=== done ==="
date
