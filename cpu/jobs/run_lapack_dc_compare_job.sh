#!/usr/bin/env bash
set -e

export LC_ALL=C
export LANG=C

source /etc/profile >/dev/null 2>&1 || true
source "$HOME/.bashrc" >/dev/null 2>&1 || true

set -euo pipefail

ROOT="${AE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BENCH_DIR="$ROOT/cpu/bench"
BUILD_DIR="$ROOT/results/cpu/build_lapack_dc_compare_${JH_JOBID:-manual}"
RESULT_DIR="$ROOT/results/cpu/results_lapack_dc_compare_${JH_JOBID:-manual}"
SRC="$BENCH_DIR/bench_lapack_dc_compare.f90"

mkdir -p "$BUILD_DIR"/br_objs "$RESULT_DIR"
cd "$ROOT"

echo "=== job info ==="
date
hostname
pwd
env | grep -E '^(JH|LSB|OMP|MKL|OPENBLAS)_' | sort || true

echo "=== cpu info ==="
lscpu | sed -n '1,45p'

echo "=== module setup ==="
if command -v module >/dev/null 2>&1; then module load tbb/2022.3 compiler-rt/2025.3.0 mkl/2025.3 2>&1 || true || true; fi
command -v gfortran
gfortran --version | head -1
echo "MKLROOT=${MKLROOT:-}"

cat > "$SRC" <<'F90'
program lapack_dc_compare
  use omp_lib, only: omp_get_max_threads, omp_get_wtime
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none

  integer, parameter :: dp = kind(1.0d0)
  integer, parameter :: nmodes = 2, nn = 4
  integer :: modes(nmodes), ns(nn)
  integer :: imode, inum, mode, n, info, lw_br, liw_br, lw_dc, liw_dc
  integer :: ldz
  integer, allocatable :: iwork_br(:), iwork_dc(:)
  real(dp) :: t0, elapsed_br, elapsed_dc, checksum_br, checksum_dc, err
  real(dp) :: work_query(1)
  integer :: iwork_query(1)
  real(dp), allocatable :: d0(:), e0(:), d(:), e(:), ref(:), z(:,:), work_br(:), work_dc(:)
  character(len=24) :: mname
  integer(int64) :: seed

  external dstedc_dc_br
  external dstedc

  modes = (/ 1, 2 /)
  ns = (/ 4096, 8192, 16384, 32768 /)

  write(*,'(a,i0)') 'omp_max_threads=', omp_get_max_threads()
  print '(a)', 'mode,n,lapack_dc_ms,br_ms,dc_over_br,dc_lwork,dc_liwork,dc_workspace_mb,z_mb,br_lwork,br_liwork,br_workspace_mb,max_abs_err,checksum_dc,checksum_br'

  do imode = 1, nmodes
     mode = modes(imode)
     mname = mode_name(mode)
     do inum = 1, nn
        n = ns(inum)
        seed = make_seed(mode, n)
        allocate(d0(n), e0(n-1), d(n), e(n-1), ref(n), work_br(1), iwork_br(1))
        call fill_case(mode, n, seed, d0, e0)

        d = d0
        e = e0
        call dstedc_dc_br(n, d, e, work_br, -1, iwork_br, -1, info)
        if (info .ne. 0) stop 1
        lw_br = max(1, int(work_br(1)))
        liw_br = max(1, iwork_br(1))
        deallocate(work_br, iwork_br)
        allocate(work_br(lw_br), iwork_br(liw_br))

        d = d0
        e = e0
        ldz = n
        allocate(z(ldz,n))
        call dstedc('I', n, d, e, z, ldz, work_query, -1, iwork_query, -1, info)
        if (info .ne. 0) then
           print *, 'DSTEDC query failed mode=', trim(mname), ' n=', n, ' info=', info
           stop 2
        end if
        lw_dc = max(1, int(work_query(1)))
        liw_dc = max(1, iwork_query(1))
        allocate(work_dc(lw_dc), iwork_dc(liw_dc))

        d = d0
        e = e0
        t0 = omp_get_wtime()
        call dstedc('I', n, d, e, z, ldz, work_dc, lw_dc, iwork_dc, liw_dc, info)
        elapsed_dc = omp_get_wtime() - t0
        if (info .ne. 0) then
           print *, 'DSTEDC failed mode=', trim(mname), ' n=', n, ' info=', info
           stop 3
        end if
        checksum_dc = checksum(n, d)
        ref = d

        d = d0
        e = e0
        t0 = omp_get_wtime()
        call dstedc_dc_br(n, d, e, work_br, lw_br, iwork_br, liw_br, info)
        elapsed_br = omp_get_wtime() - t0
        if (info .ne. 0) then
           print *, 'BR failed mode=', trim(mname), ' n=', n, ' info=', info
           stop 4
        end if
        checksum_br = checksum(n, d)
        err = maxval(abs(d - ref))

        write(*,'(a,",",i0,",",f12.4,",",f12.4,",",f12.4,",",i0,",",i0,",",f12.2,",",f12.2,",",i0,",",i0,",",f12.2,",",es12.4,",",es16.8,",",es16.8)') &
             trim(mname), n, 1000.0d0*elapsed_dc, 1000.0d0*elapsed_br, &
             elapsed_dc/elapsed_br, lw_dc, liw_dc, &
             (8.0d0*lw_dc + 4.0d0*liw_dc)/(1024.0d0*1024.0d0), &
             8.0d0*dble(ldz)*dble(n)/(1024.0d0*1024.0d0), &
             lw_br, liw_br, &
             (8.0d0*lw_br + 4.0d0*liw_br)/(1024.0d0*1024.0d0), &
             err, checksum_dc, checksum_br

        deallocate(d0, e0, d, e, ref, z, work_br, iwork_br, work_dc, iwork_dc)
     end do
  end do

contains

  subroutine fill_case(mode, n, seed, d, e)
    integer, intent(in) :: mode, n
    integer(int64), intent(inout) :: seed
    real(dp), intent(out) :: d(n), e(n-1)
    integer :: i
    do i = 1, n
       select case (mode)
       case (1)
          d(i) = 2.0d0 * randu(seed) - 1.0d0
       case (2)
          d(i) = randn(seed)
       end select
    end do
    do i = 1, n-1
       e(i) = 0.10d0 + 0.20d0 * randu(seed)
    end do
  end subroutine fill_case

  real(dp) function checksum(n, d)
    integer, intent(in) :: n
    real(dp), intent(in) :: d(n)
    checksum = d(1) + d(n) + sum(d(1:n:max(1,n/16)))
  end function checksum

  integer(int64) function make_seed(mode, n)
    integer, intent(in) :: mode, n
    make_seed = 7640891576956012809_int64
    make_seed = ieor(make_seed, int(mode, int64) * 1000003_int64)
    make_seed = ieor(make_seed, int(n, int64) * 1099511628211_int64)
    if (make_seed .eq. 0_int64) make_seed = 88172645463393265_int64
  end function make_seed

  real(dp) function randu(seed)
    integer(int64), intent(inout) :: seed
    integer(int64) :: x
    x = seed
    x = ieor(x, shiftr(x, 12))
    x = ieor(x, shiftl(x, 25))
    x = ieor(x, shiftr(x, 27))
    seed = x
    randu = real(iand(x * 2685821657736338717_int64, huge(1_int64)), dp) / real(huge(1_int64), dp)
    randu = min(max(randu, tiny(1.0d0)), 1.0d0 - epsilon(1.0d0))
  end function randu

  real(dp) function randn(seed)
    integer(int64), intent(inout) :: seed
    real(dp) :: u1, u2
    u1 = randu(seed)
    u2 = randu(seed)
    randn = sqrt(-2.0d0 * log(u1)) * cos(6.283185307179586476925286766559d0 * u2)
  end function randn

  character(len=24) function mode_name(mode)
    integer, intent(in) :: mode
    select case (mode)
    case (1)
       mode_name = 'uniform'
    case (2)
       mode_name = 'normal'
    end select
  end function mode_name

end program lapack_dc_compare
F90

echo "=== compile private BR objects ==="
rm -f "$BUILD_DIR"/br_objs/*.o
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

echo "=== compile LAPACK DC comparison ==="
gfortran -O2 -fopenmp -std=legacy -ffree-line-length-none \
  "$SRC" \
  "$BUILD_DIR"/br_objs/*.o \
  -L"$MKLROOT/lib" -Wl,-rpath,"$MKLROOT/lib" \
  -lmkl_rt -lpthread -lm -ldl \
  -o "$BUILD_DIR/bench_lapack_dc_compare"

run_one() {
  local threads=$1
  local out="$RESULT_DIR/lapack_dc_compare_t${threads}.csv"

  echo "=== run threads=$threads ==="
  date
  export OMP_NUM_THREADS="$threads"
  export OMP_PROC_BIND=close
  export OMP_PLACES=cores
  export OPENBLAS_NUM_THREADS="$threads"
  export MKL_NUM_THREADS="$threads"
  export MKL_DYNAMIC=FALSE
  export MKL_INTERFACE_LAYER=LP64
  export MKL_THREADING_LAYER=GNU
  "$BUILD_DIR/bench_lapack_dc_compare" | tee "$out"
}

run_one 1
run_one 56

echo "=== result files ==="
find "$RESULT_DIR" -maxdepth 1 -type f -printf '%p %s\n' | sort
echo "=== done ==="
date
