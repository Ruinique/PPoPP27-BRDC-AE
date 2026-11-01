#!/usr/bin/env bash
# Alternative Fig. 6 run with the bench's built-in generators (from original/run_ppopp_h100_dc_cusolver_large.sh). No input files needed.
set -euo pipefail
AE=$(cd "$(dirname "$0")/../.." && pwd); O=$AE/results/fig6_builtin; mkdir -p "$O"
"$AE/results/build-gpu/tridiag_dc_bench" --data=${MODES:-normal,uniform,toeplitz,clustered} --sizes=${SIZES:-16384,32768,49152} \
  --iters=5 --warmup=2 --backend=2 --vectors=0 --run-cpu=0 --run-self=1 --run-cusolver=1 | tee "$O/run.log"
