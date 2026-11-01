#!/bin/bash
#SBATCH --job-name=ppopp_dc_cslv
#SBATCH --partition=gpu7
#SBATCH --gpus=h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=10:00
#SBATCH --output=/home/ruinique/SyncFlowEVD/.worktrees/PPoPP-bench/ppopp_h100_dc_cusolver_only_%j.out
#SBATCH --error=/home/ruinique/SyncFlowEVD/.worktrees/PPoPP-bench/ppopp_h100_dc_cusolver_only_%j.err

set -euo pipefail
unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY
export PATH=/usr/local/cuda-13.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH:-}
cd /home/ruinique/SyncFlowEVD/.worktrees/PPoPP-bench

echo "=== git ==="
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD

echo "=== Node: $(hostname), GPUs: ==="
nvidia-smi --query-gpu=name,pci.bus_id,memory.total,compute_cap --format=csv,noheader

echo "=== tridiag_dc_bench values-only multimode: SyncFlowEVD GPU DC vs cuSOLVER Xstedc, no CPU reference ==="
stdbuf -oL ./build-ppopp-h100-cuda132/clients/bench/tridiag_dc_bench \
  --data=normal,uniform,toeplitz,clustered \
  --sizes=16384,32768,49152 \
  --iters=5 --warmup=2 \
  --backend=2 --vectors=0 \
  --run-cpu=0 --run-self=1 --run-cusolver=1

echo "=== Done ==="
