#!/usr/bin/env bash
# Build the CPU BR solver objects (5 Fortran files from the OpenBLAS fork) and the smoke test.
# Mirrors the compile lines in cpu/brdc_bench/run_*_job.sh (gfortran -O2 -fopenmp -std=legacy).
# LAPACK/BLAS provider: LAPACK_LIBS (default -lopenblas). The paper links baselines against MKL 2025.3 (-lmkl_rt).
set -euo pipefail
AE=$(cd "$(dirname "$0")/../.." && pwd)
B=${BUILD_DIR:-$AE/results/build-cpu}; mkdir -p "$B"
FC=${FC:-gfortran}; LAPACK_LIBS=${LAPACK_LIBS:--lopenblas}
for s in dlaed0_br dlaed4_br dlaed7_br dlaed9_br dstedc_dc_br; do
  $FC -O2 -fopenmp -std=legacy -c "$AE/cpu/src/$s.f" -o "$B/$s.o"
done
$FC -O2 -fopenmp -std=legacy "$AE/scripts/cpu/smoke_br.f90" "$B"/*_br.o $LAPACK_LIBS -lpthread -lm -ldl -o "$B/smoke_br" 2>/dev/null \
 || $FC -O2 -fopenmp -std=legacy "$AE/scripts/cpu/smoke_br.f90" "$B"/*_br.o ${LDFLAGS:-} $LAPACK_LIBS -lpthread -lm -ldl -o "$B/smoke_br"
echo "built $B/smoke_br"
