#!/usr/bin/env bash
# Fig. 6: GPU BR vs cuSOLVER cusolverDnXstedc (compz=N), eigenvalues only, from tridiagonal input.
# Adapted from scripts/gpu/original/run_h100_spectrum_tridiag_job.sh, which uses the same spectrum-prescribed
# D/E inputs as the CPU experiments. Generate them first: bash cpu/jobs/run_spectrum_de_cache_job.sh
# Scaled down: SIZES="8192 16384" (fits GPUs with < 80 GB; N=49,152 needs ~54 GiB for cuSOLVER's workspace).
set -euo pipefail
AE=$(cd "$(dirname "$0")/../.." && pwd); BIN=$AE/results/build-gpu/tridiag_dc_bench
DATA=${SPECTRUM_DE_DIR:-$AE/results/data}; O=$AE/results/fig6; mkdir -p "$O"
for mode in ${MODES:-uniform normal toeplitz clustered}; do
  for n in ${SIZES:-8192 16384 32768 49152}; do
    "$BIN" --input="$DATA/${mode}_${n}.de.bin" --input-tridiag=1 --iters=5 --warmup=2 --backend=2 --vectors=0 \
      --run-cpu=0 --run-self=1 --run-cusolver=1 | tee "$O/${mode}_${n}.log"
  done
done
