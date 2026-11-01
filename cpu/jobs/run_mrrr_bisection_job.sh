#!/usr/bin/env bash
set -e

export LC_ALL=C
export LANG=C

source /etc/profile >/dev/null 2>&1 || true
source "$HOME/.bashrc" >/dev/null 2>&1 || true

set -eo pipefail

ROOT="${AE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BENCH_DIR="$ROOT/cpu/bench"
BUILD_DIR="$ROOT/results/cpu/build_mrrr_bisection"
RESULT_DIR="$ROOT/results/cpu/results_mrrr_bisection"
SRC="$BENCH_DIR/bench_mrrr_bisection.f90"

mkdir -p "$BUILD_DIR" "$RESULT_DIR"
cd "$ROOT"

echo "=== job info ==="
date
hostname
pwd
env | grep -E '^(JH|LSB|OMP|MKL|OPENBLAS)_' | sort || true

echo "=== cpu info ==="
lscpu | sed -n '1,45p'

echo "=== module setup ==="
if command -v module >/dev/null 2>&1; then module load tbb/2022.3 compiler-rt/2025.3.0 mkl/2025.3 2>&1 || true; fi
command -v gfortran
gfortran --version | head -1
echo "MKLROOT=${MKLROOT:-}"

echo "=== compile benchmark ==="
gfortran -O2 -fopenmp -std=legacy \
  "$SRC" \
  -L"$MKLROOT/lib" -Wl,-rpath,"$MKLROOT/lib" \
  -lmkl_rt -lpthread -lm -ldl \
  -o "$BUILD_DIR/bench_mrrr_bisection_mkl"

run_one() {
  local threads=$1
  local out="$RESULT_DIR/mrrr_bisection_t${threads}.csv"

  echo "=== run threads=$threads ==="
  date
  export OMP_NUM_THREADS="$threads"
  export OMP_PROC_BIND=close
  export OMP_PLACES=cores
  export OPENBLAS_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export MKL_DYNAMIC=FALSE
  export MKL_INTERFACE_LAYER=LP64
  export MKL_THREADING_LAYER=GNU
  "$BUILD_DIR/bench_mrrr_bisection_mkl" | tee "$out"
}

run_one 56

echo "=== result files ==="
ls -lh "$RESULT_DIR"
echo "=== done ==="
date
