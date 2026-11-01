#!/usr/bin/env bash
set -eo pipefail

export LC_ALL=C
export LANG=C

source /etc/profile >/dev/null 2>&1 || true
source "$HOME/.bashrc" >/dev/null 2>&1 || true
if command -v module >/dev/null 2>&1; then module purge >/dev/null 2>&1 || true || true; fi
if command -v module >/dev/null 2>&1; then module load gcc/11.2.0 tbb/2022.3 compiler-rt/2025.3.0 mkl/2025.3 mpi/2021.17 || true; fi

AE="${AE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ROOT="$AE/cpu/bench"
CACHE_DIR="${CACHE_DIR:-$AE/results/data}"
JOB_TAG="${JH_JOBID:-manual}"
BUILD_DIR="$AE/results/cpu/build_extended_mrrr_sweep_${JOB_TAG}"
RESULT_DIR="$AE/results/cpu/results_extended_mrrr_sweep_${JOB_TAG}"

mkdir -p "$BUILD_DIR" "$RESULT_DIR"
cd "$ROOT"

echo "=== extended current-cache MRRR/DSTEGR sweep ==="
date
hostname
echo "CACHE_DIR=$CACHE_DIR"

export MKL_INTERFACE_LAYER=LP64
export MKL_THREADING_LAYER=GNU
export MKL_DYNAMIC=FALSE
export OMP_NUM_THREADS=56
export MKL_NUM_THREADS=56
export OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export CACHE_DIR

gfortran -O2 -fopenmp -ffree-line-length-none \
  "$ROOT/bench_cache_dstegr_safe.f90" \
  -L"$MKLROOT/lib" -Wl,-rpath,"$MKLROOT/lib" \
  -lmkl_rt -lpthread -lm -ldl \
  -o "$BUILD_DIR/bench_cache_dstegr"

RAW="$RESULT_DIR/extended_mrrr_raw.csv"
FAIL="$RESULT_DIR/extended_mrrr_failures.csv"
echo 'rep,method,mode,n,threads,time_ms,info,m,checksum,generator' > "$RAW"
echo 'rep,method,mode,n,status' > "$FAIL"

for mode in uniform normal toeplitz clustered; do
  for n in 8192 16384 32768 49152 65536; do
    for rep in 1 2 3 4 5; do
      echo "=== DSTEGR mode=$mode n=$n rep=$rep ==="
      tmp="$RESULT_DIR/dstegr_${mode}_${n}_rep${rep}.tmp"
      set +e
      CACHE_MODE="$mode" CACHE_N="$n" timeout 1800 \
        "$BUILD_DIR/bench_cache_dstegr" > "$tmp"
      status=$?
      set -e
      if [[ "$status" -eq 0 ]]; then
        awk -v rep="$rep" 'NR == 2 {print rep "," $0}' "$tmp" >> "$RAW"
      else
        echo "$rep,DSTEGR,$mode,$n,$status" | tee -a "$FAIL"
      fi
    done
  done
done

echo "=== MRRR raw ==="
cat "$RAW"
echo "=== failures ==="
cat "$FAIL"
echo "=== done ==="
date
