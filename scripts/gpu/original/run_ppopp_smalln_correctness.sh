#!/bin/bash
#SBATCH --job-name=ppopp_smalln_ok
#SBATCH --partition=gpu7
#SBATCH --gpus=h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=10:00
#SBATCH --output=/home/ruinique/SyncFlowEVD/.worktrees/PPoPP-bench/smalln_correctness_%j.out
#SBATCH --error=/home/ruinique/SyncFlowEVD/.worktrees/PPoPP-bench/smalln_correctness_%j.err

set -euo pipefail
unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY
export PATH=/usr/local/cuda-13.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH:-}

cd /home/ruinique/SyncFlowEVD/.worktrees/PPoPP-bench

echo "=== git ==="
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git diff -- library/stages/tridiag_solver.cu

echo "=== correctness smoke: 4096 clustered ==="
stdbuf -oL ./build-ppopp-h100-cuda132/clients/bench/tridiag_dc_bench \
  --data=clustered \
  --sizes=4096 \
  --iters=3 --warmup=1 \
  --backend=2 --vectors=0 \
  --run-cpu=1 --run-self=1 --run-cusolver=1

echo "=== Done ==="
