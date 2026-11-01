#!/usr/bin/env bash
# Sec. 5.6: at N=65,536 cuSOLVER expected to run out of memory on 80 GB, BR completes.
# Input: cpu/jobs/run_spectrum_de_cache_job.sh (or run_uniform_65k_de_cache_job.sh) writes uniform_65536.de.bin.
set -uo pipefail
AE=$(cd "$(dirname "$0")/../.." && pwd); BIN=$AE/results/build-gpu/tridiag_dc_bench; IN=${INPUT:-${SPECTRUM_DE_DIR:-$AE/results/data}/uniform_65536.de.bin}; O=$AE/results/sec56; mkdir -p $O
$BIN --input="$IN" --input-tridiag=1 --iters=5 --warmup=2 --backend=2 --vectors=0 --run-cpu=0 --run-self=1 --run-cusolver=0 | tee $O/br.log
$BIN --input="$IN" --input-tridiag=1 --iters=1 --warmup=0 --backend=2 --vectors=0 --run-cpu=0 --run-self=0 --run-cusolver=1 > $O/cusolver.log 2>&1; echo "cusolver_exit=$?" | tee -a $O/cusolver.log
