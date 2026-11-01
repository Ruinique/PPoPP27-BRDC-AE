#!/usr/bin/env bash
export LC_ALL=C
export LANG=C
source /etc/profile >/dev/null 2>&1 || true
set -euo pipefail

ROOT="${OPENBLAS_FORK_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
JOB_TAG="${JH_JOBID:-manual}"
STAGE="$ROOT/codex_bench/deflation_recheck"
BUILD_DIR="$STAGE/build_${JOB_TAG}"
RESULT_DIR="$ROOT/codex_bench/results_supplemental/deflation_recheck_${JOB_TAG}"

mkdir -p "$BUILD_DIR" "$RESULT_DIR"
cd "$STAGE"

echo "=== BRDC deflation recheck ==="
date
hostname
pwd

module use /public/software/modules/intel || true
module use /public/software/modules/compiler || true
module load tbb/2022.3 compiler-rt/2025.3.0 mkl/2025.3 2>&1
module list 2>&1 || true
gfortran --version | head -1
echo "MKLROOT=${MKLROOT:-}"

cp dlaed0_br.f dlaed4_br.f dlaed7_br.f dlaed9_br.f dstedc_dc_br.f \
  bench_deflation_recheck.f90 openblas_get_num_nodes_stub.f90 "$BUILD_DIR/"
cd "$BUILD_DIR"

gfortran -O2 -fopenmp -std=legacy -c \
  dlaed0_br.f dlaed4_br.f dlaed7_br.f dlaed9_br.f dstedc_dc_br.f
gfortran -O2 -fopenmp -c openblas_get_num_nodes_stub.f90
gfortran -O2 -fopenmp -ffree-line-length-none \
  bench_deflation_recheck.f90 *.o \
  -L"$MKLROOT/lib" -Wl,-rpath,"$MKLROOT/lib" \
  -lmkl_rt -lpthread -lm -ldl \
  -o bench_deflation_recheck

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export MKL_INTERFACE_LAYER=LP64
export MKL_THREADING_LAYER=GNU

./bench_deflation_recheck | tee "$RESULT_DIR/deflation_recheck.csv"

echo "=== result ==="
cat "$RESULT_DIR/deflation_recheck.csv"
echo "=== done ==="
date
