#!/usr/bin/env bash
set -e

export LC_ALL=C
export LANG=C

source /etc/profile >/dev/null 2>&1 || true
source "$HOME/.bashrc" >/dev/null 2>&1 || true

set -euo pipefail

ROOT="${AE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BENCH_DIR="$ROOT/cpu/bench"
BUILD_DIR="$ROOT/results/cpu/build_scheduler_ablation_${JH_JOBID:-manual}"
RESULT_DIR="$ROOT/results/cpu/results_supplemental/scheduler_ablation_${JH_JOBID:-manual}"
SRC="$BENCH_DIR/bench_cpu_strong_scaling.f90"

mkdir -p "$BUILD_DIR"/br_objs "$RESULT_DIR"
cd "$ROOT"

echo "=== job info ==="
date
hostname
env | grep -E '^(JH|LSB|OMP|MKL|OPENBLAS)_' | sort || true
lscpu | sed -n '1,35p'

echo "=== scope ==="
echo "benchmark=scheduler_ablation"
echo "modes=uniform,normal,toeplitz,clustered"
echo "sizes=16384,65536"
echo "threads=56"
echo "variants=aware,outer_only"
echo "note=same stable BR code; outer_only forces level-parallel merges and disables inner-root teams"

if command -v module >/dev/null 2>&1; then module load tbb/2022.3 compiler-rt/2025.3.0 mkl/2025.3 2>&1 || true; fi
gfortran --version | head -1

echo "=== compile NUMA query helper ==="
cc -O2 -DOS_LINUX -DSMP -I"$ROOT" -c cpu/src/openblas_get_num_nodes.c \
  -o "$BUILD_DIR"/br_objs/openblas_get_num_nodes.o

echo "=== compile private BR objects ==="
for src in \
  cpu/src/dlaed0_br.f \
  cpu/src/dlaed4_br.f \
  cpu/src/dlaed7_br.f \
  cpu/src/dlaed9_br.f \
  cpu/src/dstedc_dc_br.f
do
  obj="$BUILD_DIR/br_objs/$(basename "$src" .f).o"
  gfortran -O2 -fopenmp -std=legacy -c "$src" -o "$obj"
done

echo "=== compile benchmark ==="
gfortran -O2 -fopenmp -std=legacy \
  "$SRC" \
  "$BUILD_DIR"/br_objs/*.o \
  -L"$MKLROOT/lib" -Wl,-rpath,"$MKLROOT/lib" \
  -lmkl_rt \
  -lpthread -lm -ldl \
  -o "$BUILD_DIR/bench_cpu_strong_scaling"

run_variant() {
  local variant=$1
  local out="$RESULT_DIR/${variant}_t56.csv"
  echo "=== run variant=$variant threads=56 ==="
  date
  export OMP_NUM_THREADS=56
  export OMP_PROC_BIND=close
  export OMP_PLACES=cores
  export OPENBLAS_NUM_THREADS=56
  export MKL_NUM_THREADS=1
  export MKL_DYNAMIC=FALSE
  export MKL_INTERFACE_LAYER=LP64
  export MKL_THREADING_LAYER=GNU
  if [ "$variant" = "outer_only" ]; then
    export OPENBLAS_BRDC_SCHED=outer
    export OPENBLAS_BRDC_WPROD_THREADS=1
  else
    unset OPENBLAS_BRDC_SCHED || true
    unset OPENBLAS_BRDC_WPROD_THREADS || true
  fi
  "$BUILD_DIR/bench_cpu_strong_scaling" | tee "$out"
}

run_variant aware
run_variant outer_only

echo "=== summarize ==="
if [ -x /usr/bin/python3 ]; then
  PYTHON_BIN=/usr/bin/python3
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
else
  echo "python3 not found"
  exit 1
fi

"$PYTHON_BIN" - "$RESULT_DIR" <<'PY'
import csv, os, sys

result_dir = sys.argv[1]
variants = ["aware", "outer_only"]
rows = {}

for variant in variants:
    path = os.path.join(result_dir, f"{variant}_t56.csv")
    with open(path) as f:
        reader = None
        for line in f:
            line = line.strip()
            if not line or line.startswith("omp_"):
                continue
            if line.startswith("mode,"):
                reader = line.split(",")
                continue
            if reader is None:
                continue
            parts = line.split(",")
            row = dict(zip(reader, parts))
            key = (row["mode"], int(row["n"]))
            rows.setdefault(key, {})[variant] = row

out = os.path.join(result_dir, "scheduler_ablation.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "mode", "n", "outer_only_ms", "aware_ms",
        "outer_over_aware", "aware_checksum", "outer_checksum",
    ])
    for key in sorted(rows):
        if not all(v in rows[key] for v in variants):
            continue
        aware = rows[key]["aware"]
        outer = rows[key]["outer_only"]
        aware_ms = float(aware["br_ms"])
        outer_ms = float(outer["br_ms"])
        w.writerow([
            key[0], key[1], f"{outer_ms:.4f}", f"{aware_ms:.4f}",
            f"{outer_ms/aware_ms:.4f}", aware["checksum_br"],
            outer["checksum_br"],
        ])

print(f"wrote {out}")
PY

cat "$RESULT_DIR/scheduler_ablation.csv"
echo "=== done ==="
date
