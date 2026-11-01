#!/usr/bin/env bash
export LC_ALL=C
export LANG=C
source /etc/profile >/dev/null 2>&1 || true
set -euo pipefail

ROOT="${AE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BENCH_DIR="$ROOT/cpu/bench"
JOB_TAG="${JH_JOBID:-manual}"
BUILD_DIR="$ROOT/results/cpu/build_application_tridiag_deflation_${JOB_TAG}"
RESULT_DIR="$ROOT/results/cpu/results_supplemental/application_tridiag_deflation_${JOB_TAG}"
SRC="$BUILD_DIR/bench_application_tridiag_deflation.f90"

mkdir -p "$BUILD_DIR"/br_objs "$RESULT_DIR"
cd "$ROOT"

echo "=== application-shaped tridiagonal BR deflation probe ==="
date
hostname
if command -v module >/dev/null 2>&1; then module use /public/software/modules/intel || true || true; fi
if command -v module >/dev/null 2>&1; then module use /public/software/modules/compiler || true || true; fi
if command -v module >/dev/null 2>&1; then module load tbb/2022.3 compiler-rt/2025.3.0 mkl/2025.3 2>&1 || true || true; fi
gfortran --version | head -1
echo "MKLROOT=${MKLROOT:-}"

for src in \
  "$BENCH_DIR/deflation_recheck/openblas_get_num_nodes_stub.f90" \
  "$BENCH_DIR/deflation_recheck/dlaed0_br.f" \
  "$BENCH_DIR/deflation_recheck/dlaed4_br.f" \
  "$BENCH_DIR/deflation_recheck/dlaed7_br.f" \
  "$BENCH_DIR/deflation_recheck/dlaed9_br.f" \
  "$BENCH_DIR/deflation_recheck/dstedc_dc_br.f"
do
  obj="$BUILD_DIR/br_objs/$(basename "$src" .f).o"
  obj="${obj%.f90.o}.o"
  gfortran -O2 -fopenmp -std=legacy -c "$src" -o "$obj"
done

cat > "$SRC" <<'F90'
program bench_application_tridiag_deflation
  use omp_lib, only: omp_set_num_threads, omp_get_max_threads
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none

  integer, parameter :: dp = kind(1.0d0)
  integer, parameter :: ncases = 7
  integer :: cases(ncases), sizes(ncases)
  integer :: idx, icase, n, threads, info, lw, liw
  integer, allocatable :: iwork(:)
  integer(int64) :: seed
  real(dp), allocatable :: d0(:), e0(:), d(:), e(:), work(:)
  real(dp) :: qwork(1)
  real(dp) :: merges, parent_merges, root_merges, size_sum, active_sum
  real(dp) :: parent_size_sum, parent_active_sum, full_defl, max_k, root_k
  character(len=32) :: name

  external dstedc_dc_br, mkl_set_num_threads

  cases = (/ 1, 2, 3, 4, 5, 6, 5 /)
  sizes = (/ 16384, 16384, 16384, 16384, 16384, 16384, 65536 /)
  threads = 56

  call omp_set_num_threads(threads)
  call mkl_set_num_threads(1)
  write(*,'(a,i0)') 'omp_max_threads=', omp_get_max_threads()
  write(*,'(a)') 'case,n,threads,info,merges,parent_merges,root_merges,size_sum,active_sum,avg_active_ratio,avg_deflation_ratio,parent_active_ratio,full_defl_merges,max_k,root_k,checksum'
  call flush(6)

  do idx = 1, ncases
     icase = cases(idx)
     n = sizes(idx)
     allocate(d0(n), e0(n-1), d(n), e(n-1), work(1), iwork(1))
     seed = 1469598103934665603_int64 + int(icase, int64) * 1099511628211_int64 + int(n, int64)
     call fill_case(icase, n, seed, name, d0, e0)

     d = d0
     e = e0
     call dstedc_dc_br(n, d, e, qwork, -1, iwork, -1, info)
     if (info .ne. 0) stop 2
     lw = max(max(1, int(qwork(1))), 32*n)
     liw = max(max(1, iwork(1)), 8*n)
     deallocate(work, iwork)
     allocate(work(lw), iwork(liw))

     call brdc_stats_reset()
     d = d0
     e = e0
     call dstedc_dc_br(n, d, e, work, lw, iwork, liw, info)
     call brdc_stats_get(merges, parent_merges, root_merges, &
          size_sum, active_sum, parent_size_sum, parent_active_sum, &
          full_defl, max_k, root_k)
     write(*,'(a,",",i0,",",i0,",",i0,",",f0.0,",",f0.0,",",f0.0,",",f0.0,",",f0.0,",",f10.6,",",f10.6,",",f10.6,",",f0.0,",",f0.0,",",f0.0,",",es16.8)') &
          trim(name), n, threads, info, merges, parent_merges, root_merges, &
          size_sum, active_sum, ratio(active_sum, size_sum), &
          ratio(size_sum-active_sum, size_sum), &
          ratio(parent_active_sum, parent_size_sum), full_defl, max_k, root_k, &
          checksum(n, d)
     call flush(6)
     if (info .ne. 0) stop 3

     deallocate(d0, e0, d, e, work, iwork)
  end do

contains
  subroutine fill_case(icase, n, seed, name, d, e)
    integer, intent(in) :: icase, n
    integer(int64), intent(inout) :: seed
    character(len=32), intent(out) :: name
    real(dp), intent(out) :: d(n), e(n-1)
    integer :: i
    real(dp) :: x, h, width, w

    select case(icase)
    case(1)
       name = 'gauss_legendre_jacobi'
       d = 0.0d0
       do i = 1, n - 1
          e(i) = real(i, dp) / sqrt(real(4*i*i - 1, dp))
       end do
    case(2)
       name = 'gauss_hermite_jacobi'
       d = 0.0d0
       do i = 1, n - 1
          e(i) = sqrt(0.5d0 * real(i, dp))
       end do
    case(3)
       name = 'ehrenfest_birth_death'
       d = real(n - 1, dp)
       do i = 1, n - 1
          e(i) = -sqrt(real(i, dp) * real(n - i, dp))
       end do
    case(4)
       name = 'one_d_laplacian'
       d = 2.0d0
       e = -1.0d0
    case(5)
       name = 'anderson_schrodinger'
       w = 4.0d0
       do i = 1, n
          d(i) = 2.0d0 + w * (randu(seed) - 0.5d0)
       end do
       e = -1.0d0
    case(6)
       name = 'harmonic_oscillator_fd'
       width = 8.0d0
       h = 2.0d0 * width / real(n + 1, dp)
       do i = 1, n
          x = -width + h * real(i, dp)
          d(i) = 2.0d0 / (h*h) + x*x
       end do
       e = -1.0d0 / (h*h)
    end select
  end subroutine fill_case

  real(dp) function randu(seed)
    integer(int64), intent(inout) :: seed
    integer(int64) :: x
    x = seed
    x = ieor(x, shiftr(x, 12))
    x = ieor(x, shiftl(x, 25))
    x = ieor(x, shiftr(x, 27))
    seed = x
    randu = real(iand(x * 2685821657736338717_int64, huge(1_int64)), dp) / &
         real(huge(1_int64), dp)
  end function randu

  real(dp) function ratio(a, b)
    real(dp), intent(in) :: a, b
    if (b .eq. 0.0d0) then
       ratio = -1.0d0
    else
       ratio = a / b
    end if
  end function ratio

  real(dp) function checksum(n, d)
    integer, intent(in) :: n
    real(dp), intent(in) :: d(n)
    checksum = d(1) + d(n) + sum(d(1:n:max(1,n/16)))
  end function checksum
end program bench_application_tridiag_deflation

subroutine brdc_stats_reset()
  implicit none
  double precision merges, parent_merges, root_merges
  double precision size_sum, active_sum, parent_size_sum, parent_active_sum
  double precision full_defl, max_k, root_k
  common /brdc_stats/ merges, parent_merges, root_merges, size_sum, &
       active_sum, parent_size_sum, parent_active_sum, full_defl, &
       max_k, root_k
  merges = 0.0d0
  parent_merges = 0.0d0
  root_merges = 0.0d0
  size_sum = 0.0d0
  active_sum = 0.0d0
  parent_size_sum = 0.0d0
  parent_active_sum = 0.0d0
  full_defl = 0.0d0
  max_k = 0.0d0
  root_k = -1.0d0
end subroutine brdc_stats_reset

subroutine brdc_stats_record(n, k, wantq)
  implicit none
  integer, intent(in) :: n, k
  logical, intent(in) :: wantq
  double precision merges, parent_merges, root_merges
  double precision size_sum, active_sum, parent_size_sum, parent_active_sum
  double precision full_defl, max_k, root_k
  common /brdc_stats/ merges, parent_merges, root_merges, size_sum, &
       active_sum, parent_size_sum, parent_active_sum, full_defl, &
       max_k, root_k
!$omp critical(brdc_stats_update)
  merges = merges + 1.0d0
  size_sum = size_sum + dble(n)
  active_sum = active_sum + dble(k)
  if (k .eq. 0) full_defl = full_defl + 1.0d0
  if (dble(k) .gt. max_k) max_k = dble(k)
  if (wantq) then
     parent_merges = parent_merges + 1.0d0
     parent_size_sum = parent_size_sum + dble(n)
     parent_active_sum = parent_active_sum + dble(k)
  else
     root_merges = root_merges + 1.0d0
     root_k = dble(k)
  end if
!$omp end critical(brdc_stats_update)
end subroutine brdc_stats_record

subroutine brdc_stats_get(merges, parent_merges, root_merges, size_sum, &
     active_sum, parent_size_sum, parent_active_sum, full_defl, max_k, &
     root_k)
  implicit none
  double precision, intent(out) :: merges, parent_merges, root_merges
  double precision, intent(out) :: size_sum, active_sum
  double precision, intent(out) :: parent_size_sum, parent_active_sum
  double precision, intent(out) :: full_defl, max_k, root_k
  double precision c_merges, c_parent_merges, c_root_merges
  double precision c_size_sum, c_active_sum
  double precision c_parent_size_sum, c_parent_active_sum
  double precision c_full_defl, c_max_k, c_root_k
  common /brdc_stats/ c_merges, c_parent_merges, c_root_merges, &
       c_size_sum, c_active_sum, c_parent_size_sum, &
       c_parent_active_sum, c_full_defl, c_max_k, c_root_k
  merges = c_merges
  parent_merges = c_parent_merges
  root_merges = c_root_merges
  size_sum = c_size_sum
  active_sum = c_active_sum
  parent_size_sum = c_parent_size_sum
  parent_active_sum = c_parent_active_sum
  full_defl = c_full_defl
  max_k = c_max_k
  root_k = c_root_k
end subroutine brdc_stats_get
F90

gfortran -O2 -fopenmp -std=legacy -ffree-line-length-none \
  "$SRC" "$BUILD_DIR"/br_objs/*.o \
  -L"$MKLROOT/lib" -Wl,-rpath,"$MKLROOT/lib" \
  -lmkl_rt -lpthread -lm -ldl \
  -o "$BUILD_DIR/bench_application_tridiag_deflation"

export OMP_NUM_THREADS=56
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export MKL_INTERFACE_LAYER=LP64
export MKL_THREADING_LAYER=GNU

OUT="$RESULT_DIR/application_tridiag_deflation_t56.csv"
"$BUILD_DIR/bench_application_tridiag_deflation" | tee "$OUT"
echo "result=$OUT"
date
