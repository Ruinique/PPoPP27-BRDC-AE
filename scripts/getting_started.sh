#!/usr/bin/env bash
# Getting started (< 30 min on the reference hardware; the CPU part takes seconds).
# Part A (CPU, any x86-64 Linux): builds the 5 BR Fortran routines and runs a BR-vs-DSTERF smoke test.
#   Needs gfortran (OpenMP) and LAPACK. Default LAPACK_LIBS=-lopenblas. With MKL: LAPACK_LIBS="-L$MKLROOT/lib -lmkl_rt".
# Part B (GPU): builds gpu/ and runs BR vs cuSOLVER Xstedc on N=4096. Needs an NVIDIA GPU and CUDA 13.2. SKIP_GPU=1 skips it.
set -euo pipefail
AE=$(cd "$(dirname "$0")/.." && pwd); mkdir -p "$AE/results"
echo "== Part A: CPU =="
"$AE/scripts/cpu/build_br.sh"
OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(nproc)} "$AE/results/build-cpu/smoke_br" 4096 | tee "$AE/results/cpu_smoke.log"
[ "${SKIP_GPU:-0}" = 1 ] && exit 0
echo "== Part B: GPU =="
cmake -S "$AE/gpu" -B "$AE/results/build-gpu" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH:-90} ${GPU_CMAKE_ARGS:-}
cmake --build "$AE/results/build-gpu" -j"$(nproc)"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
"$AE/results/build-gpu/tridiag_dc_bench" --data=normal,toeplitz,clustered --sizes=4096 \
  --iters=3 --warmup=1 --backend=2 --vectors=0 --run-cpu=1 --run-self=1 --run-cusolver=1 | tee "$AE/results/gpu_smoke.log"
