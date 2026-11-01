#!/bin/bash
#SBATCH --job-name=ppopp_cusolver_toe_32_49
#SBATCH --partition=gpu7
#SBATCH --gpus=h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=45:00
#SBATCH --output=/home/ruinique/SyncFlowEVD/.worktrees/PPoPP-bench/cusolver_toe_32_49_%j.out
#SBATCH --error=/home/ruinique/SyncFlowEVD/.worktrees/PPoPP-bench/cusolver_toe_32_49_%j.err

set -euo pipefail

unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY
export PATH="$HOME/.local/bin:/usr/local/cuda-13.2/bin:$PATH"
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH:-}

ROOT=/home/ruinique/SyncFlowEVD/.worktrees/PPoPP-bench
DATA_DIR="${SPECTRUM_DE_DIR:-$ROOT/spectrum-de-cache-207981}"
BENCH="$ROOT/build-ppopp-h100-cuda132/clients/bench/tridiag_dc_bench"
OUT_DIR="$ROOT/cusolver-toeplitz-32k-49k-${SLURM_JOB_ID:-manual}"

mkdir -p "$OUT_DIR"
cd "$ROOT"

echo "=== env ==="
date
hostname
nvidia-smi --query-gpu=name,pci.bus_id,memory.total,compute_cap --format=csv,noheader
command -v nsys
nsys --version
command -v veloq
veloq --version

echo "=== git ==="
git rev-parse HEAD
git diff -- clients/bench/tridiag_dc_bench.cu | sed -n '1,120p'

time_case() {
  local n="$1"
  local input="$DATA_DIR/toeplitz_${n}.de.bin"
  local base="$OUT_DIR/toeplitz_${n}_cusolver_timing"
  echo "=== timing n=$n ==="
  ls -lh "$input"
  "$BENCH" \
    --input="$input" --input-tridiag=1 \
    --iters=7 --warmup=2 \
    --backend=2 --vectors=0 \
    --run-cpu=0 --run-self=0 --run-cusolver=1 \
    | tee "${base}.log"
}

profile_case() {
  local n="$1"
  local input="$DATA_DIR/toeplitz_${n}.de.bin"
  local base="$OUT_DIR/toeplitz_${n}_cusolver_profile"
  echo "=== profile n=$n ==="
  ls -lh "$input"
  nsys profile \
    --force-overwrite=true \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --cpuctxsw=none \
    --cuda-memory-usage=true \
    -o "$base" \
    "$BENCH" \
      --input="$input" --input-tridiag=1 \
      --iters=1 --warmup=1 \
      --backend=2 --vectors=0 \
      --run-cpu=0 --run-self=0 --run-cusolver=1 \
    | tee "${base}.log"
  echo "=== veloq summary n=$n ==="
  veloq summary "${base}.nsys-rep" | tee "${base}.veloq.summary.json"
  echo "=== veloq kernel stats n=$n ==="
  veloq stats "${base}.nsys-rep" --type kernel --sort total:desc --limit 25 \
    | tee "${base}.veloq.kernels.json" || true
  echo "=== veloq memcpy stats n=$n ==="
  veloq stats "${base}.nsys-rep" --type memcpy --sort total:desc --limit 20 \
    | tee "${base}.veloq.memcpy.json" || true
  echo "=== veloq concurrency n=$n ==="
  veloq concurrency "${base}.nsys-rep" | tee "${base}.veloq.concurrency.json" || true
}

time_case 32768
time_case 49152
profile_case 32768
profile_case 49152

echo "=== result files ==="
find "$OUT_DIR" -maxdepth 1 -type f -print -exec ls -lh {} \;
echo "=== done ==="
date
