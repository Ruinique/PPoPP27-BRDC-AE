#!/usr/bin/env bash
# Scaled-down CPU check for machines without 56 cores or MKL. Not a paper experiment.
# BR vs DSTERF on random tridiagonal matrices across sizes and thread counts. Gives a mini Fig. 4-style self-speedup trend.
set -euo pipefail
AE=$(cd "$(dirname "$0")/../.." && pwd); "$AE/scripts/cpu/build_br.sh" >/dev/null
for n in ${SIZES:-4096 16384}; do for t in ${THREADS:-1 2 4 $(nproc)}; do
  echo "--- n=$n threads=$t"; OMP_NUM_THREADS=$t "$AE/results/build-cpu/smoke_br" $n
done; done | tee "$AE/results/cpu_scaled_down.log"
