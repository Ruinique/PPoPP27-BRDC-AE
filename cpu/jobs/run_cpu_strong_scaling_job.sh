#!/usr/bin/env bash
set -e

export LC_ALL=C
export LANG=C

source /etc/profile >/dev/null 2>&1 || true
source "$HOME/.bashrc" >/dev/null 2>&1 || true

set -euo pipefail

ROOT="${AE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BENCH_DIR="$ROOT/cpu/bench"
BUILD_DIR="$ROOT/results/cpu/build_cpu_strong_scaling_${JH_JOBID:-manual}"
RESULT_DIR="$ROOT/results/cpu/results_supplemental/cpu_strong_scaling_${JH_JOBID:-manual}"
SRC="$BENCH_DIR/bench_cpu_strong_scaling.f90"

mkdir -p "$BUILD_DIR"/br_objs "$RESULT_DIR"
cd "$ROOT"

echo "=== job info ==="
date
hostname
env | grep -E '^(JH|LSB|OMP|MKL|OPENBLAS)_' | sort || true
lscpu | sed -n '1,35p'

echo "=== scope ==="
echo "benchmark=cpu_strong_scaling"
echo "modes=uniform,normal,toeplitz,clustered"
echo "sizes=16384,65536"
echo "threads=1,2,4,8,16,28,56"
echo "note=BR-only; rho adaptive; NUMA-local outer budget from OpenBLAS affinity"

if command -v module >/dev/null 2>&1; then module load tbb/2022.3 compiler-rt/2025.3.0 mkl/2025.3 2>&1 || true; fi
gfortran --version | head -1

echo "=== compile NUMA query helper ==="
cc -O2 -DOS_LINUX -DSMP -I"$ROOT" -c cpu/src/openblas_get_num_nodes.c \
  -o "$BUILD_DIR"/br_objs/openblas_get_num_nodes.o

echo "=== NUMA query helper ==="
cat > "$BUILD_DIR/check_openblas_nodes.c" <<'C'
#include <stdio.h>
int openblas_get_num_nodes(void);
int main(void) {
  printf("openblas_nodes=%d\n", openblas_get_num_nodes());
  return 0;
}
C
cc -O2 "$BUILD_DIR/check_openblas_nodes.c" \
  "$BUILD_DIR"/br_objs/openblas_get_num_nodes.o \
  -lpthread -lm -ldl \
  -o "$BUILD_DIR/check_openblas_nodes"
"$BUILD_DIR/check_openblas_nodes"

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

run_one() {
  local threads=$1
  local out="$RESULT_DIR/t${threads}.csv"
  echo "=== run threads=$threads ==="
  date
  export OMP_NUM_THREADS="$threads"
  export OMP_PROC_BIND=close
  export OMP_PLACES=cores
  export OPENBLAS_NUM_THREADS="$threads"
  export MKL_NUM_THREADS=1
  export MKL_DYNAMIC=FALSE
  export MKL_INTERFACE_LAYER=LP64
  export MKL_THREADING_LAYER=GNU
  "$BUILD_DIR/bench_cpu_strong_scaling" | tee "$out"
}

for t in 1 2 4 8 16 28 56; do
  run_one "$t"
done

echo "=== merge + efficiency ==="
if [ -x /usr/bin/python3 ]; then
  PYTHON_BIN=/usr/bin/python3
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
else
  echo "python3 not found; using awk summary fallback"
  SUMMARY="$RESULT_DIR/cpu_strong_scaling_summary.csv"
  awk -F',' '
    BEGIN {
      print "mode,n,threads,reps,br_ms,br_speedup_vs_1t,br_parallel_efficiency,checksum_br"
    }
    /^mode,/ || /^omp_/ || $0=="" { next }
    {
      t=FILENAME
      sub(/^.*\/t/, "", t); sub(/\.csv$/, "", t)
      mode=$1; n=$2; reps=$4; br=$5; cbr=$6
      key=mode SUBSEP n
      rows[key SUBSEP t]=mode SUBSEP n SUBSEP t SUBSEP reps SUBSEP br SUBSEP cbr
      if (t==1) base_br[key]=br+0
    }
    END {
      PROCINFO["sorted_in"]="@ind_str_asc"
      for (rk in rows) {
        split(rows[rk], f, SUBSEP)
        mode=f[1]; n=f[2]; t=f[3]+0; br=f[5]+0
        key=mode SUBSEP n
        if (key in base_br && base_br[key]>0) {
          brsp=base_br[key]/br; breff=brsp/t
          printf "%s,%s,%d,%s,%s,%.4f,%.4f,%s\n", mode,n,t,f[4],f[5],brsp,breff,f[6]
        }
      }
    }
  ' "$RESULT_DIR"/t*.csv > "$SUMMARY"
  echo "wrote $SUMMARY"
  ls -lh "$RESULT_DIR"
  echo "=== done ==="
  date
  exit 0
fi
"$PYTHON_BIN" - "$RESULT_DIR" <<'PY'
import csv, glob, os, sys
from collections import defaultdict

result_dir = sys.argv[1]
rows_by_key = defaultdict(dict)
threads = []

for path in sorted(glob.glob(os.path.join(result_dir, "t*.csv"))):
    t = int(os.path.basename(path).split("t")[1].split(".")[0])
    threads.append(t)
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("omp_")]
    header = None
    for ln in lines:
        if ln.startswith("mode,"):
            header = ln
            continue
        if header is None:
            continue
        parts = ln.split(",")
        mode, n = parts[0], int(parts[1])
        key = (mode, n)
        rows_by_key[key][t] = {
            "reps": int(parts[3]),
            "br_ms": float(parts[4]),
            "checksum_br": parts[5],
        }

out = os.path.join(result_dir, "cpu_strong_scaling_summary.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "mode", "n", "threads", "reps", "br_ms",
        "br_speedup_vs_1t", "br_parallel_efficiency", "checksum_br",
    ])
    for key in sorted(rows_by_key):
        mode, n = key
        base = rows_by_key[key].get(1)
        for t in sorted(rows_by_key[key]):
            r = rows_by_key[key][t]
            if base:
                br_sp = base["br_ms"] / r["br_ms"]
                br_eff = br_sp / t
            else:
                br_sp = br_eff = ""
            w.writerow([
                mode, n, t, r["reps"], f"{r['br_ms']:.4f}",
                f"{br_sp:.4f}" if br_sp != "" else "",
                f"{br_eff:.4f}" if br_eff != "" else "",
                r["checksum_br"],
            ])

print(f"wrote {out}")
PY

ls -lh "$RESULT_DIR"
echo "=== done ==="
date
