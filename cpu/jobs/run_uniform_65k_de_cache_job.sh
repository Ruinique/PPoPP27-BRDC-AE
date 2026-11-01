#!/usr/bin/env bash
export LC_ALL=C
export LANG=C
source /etc/profile >/dev/null 2>&1 || true
set -euo pipefail

ROOT="${AE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BENCH_DIR="$ROOT/cpu/bench"
JOB_TAG="${JH_JOBID:-manual}"
BUILD_DIR="$ROOT/results/cpu/build_uniform_65k_de_cache_${JOB_TAG}"
RESULT_DIR="${SPECTRUM_DE_DIR:-$ROOT/results/data}"
SRC="$BUILD_DIR/gen_uniform_65k_de_cache.f90"

mkdir -p "$BUILD_DIR" "$RESULT_DIR"
cd "$ROOT"

echo "=== uniform N=65536 D/E cache generation ==="
date
hostname
if command -v module >/dev/null 2>&1; then module load tbb/2022.3 compiler-rt/2025.3.0 mkl/2025.3 2>&1 || true || true; fi
gfortran --version | head -1

cat > "$SRC" <<'F90'
program gen_uniform_65k_de_cache
  use iso_fortran_env, only: int64
  implicit none
  integer, parameter :: dp = kind(1.0d0)
  integer, parameter :: n = 65536
  integer :: info
  integer(int64) :: n64
  integer :: unit
  integer :: iseed(4)
  real(dp), allocatable :: eig(:), ab(:,:), wk(:), d(:), e(:)
  character(len=512) :: path

  call get_command_argument(1, path)
  if (len_trim(path) .eq. 0) stop 2

  allocate(eig(n), ab(2,n), wk(3*n), d(n), e(n-1))
  iseed = (/ 1, 2, mod(n, 4093) + 6, 5 /)
  call dlatms(n, n, 'U', iseed, 'S', eig, 6, 1.0d0, 1.0d0, &
       1, 1, 'B', ab, 2, wk, info)
  if (info .ne. 0) stop 3
  d = ab(1,1:n)
  e = ab(2,1:n-1)

  n64 = int(n, int64)
  open(newunit=unit, file=trim(path), access='stream', form='unformatted', &
       status='replace', action='write')
  write(unit) n64
  write(unit) d
  write(unit) e
  close(unit)

  print '(a)', 'mode,n,path,generator,info'
  write(*,'(a,",",i0,",",a,",",a,",",i0)') &
       'uniform', n, trim(path), 'dlatms_symmetric_band', info
  deallocate(eig, ab, wk, d, e)
end program gen_uniform_65k_de_cache
F90

gfortran -O2 "$SRC" \
  -L"$MKLROOT/lib" -Wl,-rpath,"$MKLROOT/lib" \
  -lmkl_rt -lpthread -lm -ldl \
  -o "$BUILD_DIR/gen_uniform_65k_de_cache"

OUT="$RESULT_DIR/uniform_65536.de.bin"
"$BUILD_DIR/gen_uniform_65k_de_cache" "$OUT" | tee "$RESULT_DIR/manifest.csv"
ls -lh "$OUT"
echo "result=$OUT"
date
