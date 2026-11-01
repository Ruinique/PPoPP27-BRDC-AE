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
BUILD_DIR="$AE/results/cpu/build_extended_pdstebz_sweep_${JOB_TAG}"
RESULT_DIR="$AE/results/cpu/results_extended_pdstebz_sweep_${JOB_TAG}"

mkdir -p "$BUILD_DIR" "$RESULT_DIR"
cd "$ROOT"

echo "=== extended current-cache parallel BZ/PDSTEBZ sweep ==="
date
hostname
echo "CACHE_DIR=$CACHE_DIR"

export I_MPI_ROOT="${I_MPI_ROOT:-/public/software/tools/intel/oneapi/mpi/2021.17}"  # AE: set I_MPI_ROOT to your Intel MPI
export PATH="$I_MPI_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$I_MPI_ROOT/lib:${MKLROOT}/lib:${LD_LIBRARY_PATH:-}"
export I_MPI_FABRICS=shm
export MKL_INTERFACE_LAYER=LP64
export MKL_THREADING_LAYER=SEQUENTIAL
export MKL_DYNAMIC=FALSE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mpicc -O2 -I"$MKLROOT/include" \
  "$ROOT/bench_pdstebz_parallel.c" \
  -L"$MKLROOT/lib" -Wl,-rpath,"$MKLROOT/lib" \
  -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64 \
  -lmkl_intel_lp64 -lmkl_sequential -lmkl_core \
  -lpthread -lm -ldl \
  -o "$BUILD_DIR/bench_pdstebz_cache"

RAW="$RESULT_DIR/extended_pdstebz_raw.csv"
FAIL="$RESULT_DIR/extended_pdstebz_failures.csv"
echo 'rep,method,mode,n,nprocs,nprow,npcol,time_ms,info,m,nsplit,lwork,liwork,checksum' > "$RAW"
echo 'rep,method,mode,n,status' > "$FAIL"

for mode in uniform normal toeplitz clustered; do
  for n in 8192 16384 32768 49152 65536; do
    input="$CACHE_DIR/${mode}_${n}.de.bin"
    for rep in 1 2 3 4 5; do
      echo "=== PDSTEBZ mode=$mode n=$n rep=$rep ==="
      tmp="$RESULT_DIR/pdstebz_${mode}_${n}_rep${rep}.tmp"
      set +e
      timeout 1800 mpirun -np 56 \
        -genv I_MPI_FABRICS shm \
        -genv OMP_NUM_THREADS 1 \
        -genv MKL_NUM_THREADS 1 \
        -genv MKL_DYNAMIC FALSE \
        "$BUILD_DIR/bench_pdstebz_cache" "$n" "$n" "$mode" "$input" > "$tmp"
      status=$?
      set -e
      if [[ "$status" -eq 0 ]]; then
        awk -v rep="$rep" 'NR == 2 {print rep "," $0}' "$tmp" >> "$RAW"
      else
        echo "$rep,PDSTEBZ,$mode,$n,$status" | tee -a "$FAIL"
      fi
    done
  done
done

echo "=== PDSTEBZ raw ==="
cat "$RAW"
echo "=== failures ==="
cat "$FAIL"
echo "=== done ==="
date
