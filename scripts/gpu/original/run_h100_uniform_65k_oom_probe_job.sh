#!/bin/bash
#SBATCH --job-name=ppopp_h100_oom65
#SBATCH --partition=gpu7
#SBATCH --gpus=h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=20:00
#SBATCH --output=/home/ruinique/SyncFlowEVD/.worktrees/PPoPP-bench/h100_oom65_%j.out
#SBATCH --error=/home/ruinique/SyncFlowEVD/.worktrees/PPoPP-bench/h100_oom65_%j.err

set -euo pipefail

unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY
export PATH=/usr/local/cuda-13.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH:-}

ROOT=/home/ruinique/SyncFlowEVD/.worktrees/PPoPP-bench
BENCH="$ROOT/build-ppopp-h100-cuda132/clients/bench/tridiag_dc_bench"
INPUT="$ROOT/spectrum-de-cache-207981/uniform_65536.de.bin"
OUT_DIR="$ROOT/spectrum-h100-oom65-${SLURM_JOB_ID:-manual}"

mkdir -p "$OUT_DIR"
cd "$ROOT"

echo "=== git ==="
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD

echo "=== node/gpu ==="
hostname
nvidia-smi --query-gpu=name,pci.bus_id,memory.total,compute_cap --format=csv,noheader

echo "=== scope ==="
echo "input=$INPUT"
echo "out_dir=$OUT_DIR"
echo "bench=$BENCH"
ls -lh "$INPUT" "$BENCH"

echo "=== BR only ==="
stdbuf -oL "$BENCH" \
  --input="$INPUT" --input-tridiag=1 \
  --iters=5 --warmup=2 \
  --backend=2 --vectors=0 \
  --run-cpu=0 --run-self=1 --run-cusolver=0 \
  | tee "$OUT_DIR/uniform_65536_br.log"
br_status=${PIPESTATUS[0]}
echo "br_exit=$br_status" | tee "$OUT_DIR/status.txt"

echo "=== cuSOLVER only, expected OOM ==="
set +e
stdbuf -oL "$BENCH" \
  --input="$INPUT" --input-tridiag=1 \
  --iters=1 --warmup=0 \
  --backend=2 --vectors=0 \
  --run-cpu=0 --run-self=0 --run-cusolver=1 \
  > "$OUT_DIR/uniform_65536_cusolver.log" 2>&1
cusolver_status=$?
set -e
cat "$OUT_DIR/uniform_65536_cusolver.log"
echo "cusolver_exit=$cusolver_status" | tee -a "$OUT_DIR/status.txt"

echo "=== done ==="
date
