#!/usr/bin/env bash
set -e

export LC_ALL=C
export LANG=C

source /etc/profile >/dev/null 2>&1 || true
source "$HOME/.bashrc" >/dev/null 2>&1 || true

set -euo pipefail

ROOT="${AE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BENCH_DIR="$ROOT/cpu/bench"
BUILD_DIR="$ROOT/results/cpu/build_dsterf_large_${JH_JOBID:-manual}"
RESULT_DIR="$ROOT/results/cpu/results_dsterf_large"
SRC="$BENCH_DIR/bench_dsterf_large.f90"

mkdir -p "$BUILD_DIR" "$RESULT_DIR"
cd "$ROOT"

echo "=== job info ==="
date
hostname
pwd
env | grep -E '^(JH|LSB|OMP|MKL|OPENBLAS)_' | sort || true
lscpu | sed -n '1,35p'

if command -v module >/dev/null 2>&1; then module load tbb/2022.3 compiler-rt/2025.3.0 mkl/2025.3 2>&1 || true; fi
echo "MKLROOT=${MKLROOT:-}"
gfortran --version | head -1

cat > "$SRC" <<'F90'
program dsterf_large_bench
  use omp_lib, only: omp_get_wtime
  implicit none

  integer, parameter :: dp = kind(1.0d0)
  integer, parameter :: nmodes = 3, nn = 11
  integer :: modes(nmodes), ns(nn)
  integer :: imode, inum, mode, n, info
  real(dp) :: elapsed, t0, checksum
  real(dp), allocatable :: d0(:), e0(:), d(:), e(:)
  character(len=20) :: mname

  external dsterf

  modes = (/ 1, 2, 3 /)
  ns = (/ 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, &
          131072, 262144, 524288 /)

  print '(a)', 'mode,n,reps,dsterf_ms,qr_storage_mb,checksum'

  do imode = 1, nmodes
     mode = modes(imode)
     do inum = 1, nn
        n = ns(inum)
        allocate(d0(n), e0(n-1), d(n), e(n-1))
        call fill_case(mode, n, d0, e0)

        d = d0
        e = e0
        t0 = omp_get_wtime()
        call dsterf(n, d, e, info)
        elapsed = omp_get_wtime() - t0
        if (info .ne. 0) then
           print *, 'DSTERF failed mode=', mode, ' n=', n, ' info=', info
           stop 1
        end if
        checksum = d(1) + d(n) + sum(d(1:n:max(1,n/16)))

        mname = mode_name(mode)
        write(*,'(a,",",i0,",",i0,",",f14.4,",",f12.2,",",es16.8)') &
             trim(mname), n, 1, 1000.0d0*elapsed, &
             (8.0d0*n + 8.0d0*(n-1)) / (1024.0d0*1024.0d0), checksum

        deallocate(d0, e0, d, e)
     end do
  end do

contains

  subroutine fill_case(mode, n, d, e)
    integer, intent(in) :: mode, n
    real(dp), intent(out) :: d(n), e(n-1)
    integer :: i

    do i = 1, n
       select case (mode)
       case (1)
          d(i) = 2.0d0 + 0.001d0*i + sin(0.37d0*i)
       case (2)
          d(i) = 2.0d0
       case (3)
          d(i) = 1.0d0 + 1.0d-12*(i - 0.5d0*(n+1))
       end select
    end do

    do i = 1, n-1
       select case (mode)
       case (1)
          e(i) = 0.25d0 + 0.03d0*cos(0.21d0*i)
       case (2)
          e(i) = 0.25d0
       case (3)
          e(i) = 1.0d-4*(1.0d0 + 0.1d0*cos(0.33d0*i))
       end select
    end do
  end subroutine fill_case

  character(len=20) function mode_name(mode)
    integer, intent(in) :: mode
    select case (mode)
    case (1)
       mode_name = 'random'
    case (2)
       mode_name = 'toeplitz'
    case (3)
       mode_name = 'clustered'
    end select
  end function mode_name

end program dsterf_large_bench
F90

echo "=== compile DSTERF benchmark ==="
gfortran -O2 -fopenmp -std=legacy \
  "$SRC" \
  -L"$MKLROOT/lib" -Wl,-rpath,"$MKLROOT/lib" \
  -lmkl_rt -lpthread -lm -ldl \
  -o "$BUILD_DIR/bench_dsterf_large"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export MKL_INTERFACE_LAYER=LP64
export MKL_THREADING_LAYER=GNU

echo "=== run DSTERF large ==="
date
"$BUILD_DIR/bench_dsterf_large" | tee "$RESULT_DIR/dsterf_large.csv"

echo "=== result files ==="
ls -lh "$RESULT_DIR"
echo "=== done ==="
date
