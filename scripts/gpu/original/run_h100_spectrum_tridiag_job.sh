#!/bin/bash
#SBATCH --job-name=ppopp_spectrum_h100
#SBATCH --partition=gpu7
#SBATCH --gpus=h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=90:00
#SBATCH --output=/home/ruinique/SyncFlowEVD/.worktrees/PPoPP-bench/spectrum_h100_%j.out
#SBATCH --error=/home/ruinique/SyncFlowEVD/.worktrees/PPoPP-bench/spectrum_h100_%j.err

set -euo pipefail

unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY
export PATH=/usr/local/cuda-13.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH:-}

ROOT=/home/ruinique/SyncFlowEVD/.worktrees/PPoPP-bench
DATA_DIR="${SPECTRUM_DE_DIR:-$ROOT/spectrum-de-cache}"
OUT_DIR="$ROOT/spectrum-h100-${SLURM_JOB_ID:-manual}"
BENCH="$ROOT/build-ppopp-h100-cuda132/clients/bench/tridiag_dc_bench"

mkdir -p "$OUT_DIR"
cd "$ROOT"

echo "=== git ==="
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git diff -- clients/bench/tridiag_dc_bench.cu

echo "=== node/gpu ==="
hostname
nvidia-smi --query-gpu=name,pci.bus_id,memory.total,compute_cap --format=csv,noheader

echo "=== scope ==="
echo "data_dir=$DATA_DIR"
echo "out_dir=$OUT_DIR"
echo "bench=$BENCH"
echo "vectors=0 backend=2 run_self=1 run_cusolver=1 run_cpu=0"

if [[ ! -x "$BENCH" ]]; then
  echo "missing benchmark binary: $BENCH" >&2
  exit 2
fi

SIZES=(8192 16384 32768 49152)
MODES=(uniform normal toeplitz clustered)

summary="$OUT_DIR/spectrum_h100_summary.txt"
: > "$summary"
echo "case,mode,n,input" | tee -a "$summary"

for mode in "${MODES[@]}"; do
  for n in "${SIZES[@]}"; do
    input="$DATA_DIR/${mode}_${n}.de.bin"
    if [[ ! -f "$input" ]]; then
      echo "missing input $input" >&2
      exit 3
    fi
    echo "=== case mode=$mode n=$n input=$input ===" | tee -a "$summary"
    echo "$mode,$n,$input" >> "$summary"
    stdbuf -oL "$BENCH" \
      --input="$input" --input-tridiag=1 \
      --iters=5 --warmup=2 \
      --backend=2 --vectors=0 \
      --run-cpu=0 --run-self=1 --run-cusolver=1 \
      | tee "$OUT_DIR/${mode}_${n}.log"
  done
done

echo "=== done ==="
date
