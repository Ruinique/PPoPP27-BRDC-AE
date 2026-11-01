#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C
export LANG=C

ROOT="${AE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BENCH_DIR="$ROOT/cpu/bench"
export CACHE_DIR="${CACHE_DIR:-${SPECTRUM_DE_DIR:-$ROOT/results/data}}"
JOB_TAG="${JH_JOBID:-manual}"
BUILD_DIR="$ROOT/results/cpu/build_cache_accuracy_${JOB_TAG}"
RESULT_DIR="$ROOT/results/cpu/results_cache_accuracy_${JOB_TAG}"
SRC="$BUILD_DIR/bench_cache_accuracy_16k_65k.f90"
MKLROOT="${MKLROOT:-/public/software/tools/intel/oneapi/mkl/2025.3}"

mkdir -p "$BUILD_DIR/br_objs" "$RESULT_DIR"
cd "$ROOT"

echo "=== cache accuracy retest N=16384/65536 ==="
date
hostname
pwd
echo "CACHE_DIR=$CACHE_DIR"
if command -v module >/dev/null 2>&1; then module load tbb/2022.3 compiler-rt/2025.3.0 mkl/2025.3 2>&1 || true || true; fi
export LD_LIBRARY_PATH="$MKLROOT/lib:${LD_LIBRARY_PATH:-}"
gfortran --version | head -1
echo "MKLROOT=${MKLROOT:-}"

echo "=== compile BR objects ==="
rm -f "$BUILD_DIR"/br_objs/*.o
cc -O2 -DOS_LINUX -DSMP -I"$ROOT" -c cpu/src/openblas_get_num_nodes.c \
  -o "$BUILD_DIR/br_objs/openblas_get_num_nodes.o"
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

cat > "$SRC" <<'F90'
program bench_cache_accuracy_16k_65k
  use omp_lib, only: omp_get_wtime, omp_set_num_threads
  implicit none

  integer, parameter :: dp = kind(1.0d0)
  integer, parameter :: nm = 4, nn = 2, nt = 2
  character(len=16), dimension(nm) :: modes
  integer, dimension(nn) :: sizes
  integer, dimension(nt) :: threads_list
  character(len=256) :: cache_dir
  integer :: stat, im, inum, it, n, threads
  integer :: info_ref, info_br, lw_br, liw_br, max_idx
  integer, allocatable :: iw_br(:)
  real(dp), allocatable :: d0(:), e0(:), ref(:), out(:), work_br(:)
  real(dp) :: t_ref_ms, t_br_ms, max_abs_err, norm_abs_err
  real(dp) :: ref_at_max, br_at_max, workspace
  character(len=512) :: cache_path

  external dsterf, dlasrt, dstedc_dc_br, mkl_set_num_threads

  modes = (/ 'uniform         ', 'normal          ', 'toeplitz        ', 'clustered       ' /)
  sizes = (/ 16384, 65536 /)
  threads_list = (/ 1, 56 /)

  call get_environment_variable('CACHE_DIR', cache_dir, status=stat)
  if (stat .ne. 0 .or. len_trim(cache_dir) .eq. 0) stop 2

  call mkl_set_num_threads(1)
  print '(a)', 'mode,n,threads,dsterf_ms,br_ms,info_dsterf,info_br,' // &
       'max_abs_err,norm_abs_err,max_idx,ref_at_max,br_at_max,' // &
       'checksum_ref,checksum_br,br_workspace_mb,cache_file'
  call flush(6)

  do im = 1, nm
     do inum = 1, nn
        n = sizes(inum)
        allocate(d0(n), e0(n-1), ref(n), out(n))
        call load_case(trim(modes(im)), n, d0, e0, cache_path)
        call solve_dsterf(n, d0, e0, ref, t_ref_ms, info_ref)
        if (info_ref .ne. 0) then
           write(*,*) 'DSTERF failed mode=', trim(modes(im)), ' n=', n, ' info=', info_ref
           stop 10
        end if

        do it = 1, nt
           threads = threads_list(it)
           call omp_set_num_threads(threads)
           call br_workspace(n, d0, e0, lw_br, liw_br, info_br)
           if (info_br .ne. 0) then
              write(*,*) 'BR workspace query failed mode=', trim(modes(im)), ' n=', n, ' info=', info_br
              stop 11
           end if
           allocate(work_br(lw_br), iw_br(liw_br))
           call solve_br(n, d0, e0, work_br, lw_br, iw_br, liw_br, out, t_br_ms, info_br)
           call max_error(n, out, ref, max_abs_err, max_idx)
           norm_abs_err = max_abs_err / max(1.0d0, maxval(abs(ref)))
           ref_at_max = ref(max_idx)
           br_at_max = out(max_idx)
           workspace = workspace_mb(lw_br, liw_br)
           write(*,'(a,",",i0,",",i0,",",f12.4,",",f12.4,",",i0,",",i0,' // &
                '",",es12.4,",",es12.4,",",i0,",",es24.16,",",es24.16,' // &
                '",",es24.16,",",es24.16,",",f12.2,",",a)') &
                trim(modes(im)), n, threads, t_ref_ms, t_br_ms, info_ref, info_br, &
                max_abs_err, norm_abs_err, max_idx, ref_at_max, br_at_max, &
                checksum(n, ref), checksum(n, out), workspace, trim(cache_path)
           call flush(6)
           deallocate(work_br, iw_br)
        end do
        deallocate(d0, e0, ref, out)
     end do
  end do

contains
  function itoa(value) result(str)
    integer, intent(in) :: value
    character(len=32) :: str
    write(str, '(i0)') value
  end function itoa

  subroutine load_case(mode, n, d, e, path)
    character(len=*), intent(in) :: mode
    integer, intent(in) :: n
    real(dp), intent(out) :: d(n), e(n-1)
    character(len=*), intent(out) :: path
    integer(kind=8) :: file_n
    integer :: unit
    path = trim(cache_dir)//'/'//trim(mode)//'_'//trim(adjustl(itoa(n)))//'.de.bin'
    open(newunit=unit, file=trim(path), access='stream', form='unformatted', &
         status='old', action='read')
    read(unit) file_n
    if (file_n .ne. n) stop 3
    read(unit) d
    read(unit) e
    close(unit)
  end subroutine load_case

  subroutine solve_dsterf(n, d0, e0, out, t_ms, info)
    integer, intent(in) :: n
    real(dp), intent(in) :: d0(n), e0(n-1)
    real(dp), intent(out) :: out(n), t_ms
    integer, intent(out) :: info
    integer :: sort_info
    real(dp), allocatable :: d(:), e(:)
    real(dp) :: t0
    allocate(d(n), e(n-1))
    d = d0
    e = e0
    t0 = omp_get_wtime()
    call dsterf(n, d, e, info)
    t_ms = 1000.0d0 * (omp_get_wtime() - t0)
    if (info .eq. 0) then
       call dlasrt('I', n, d, sort_info)
       if (sort_info .ne. 0) info = sort_info
    end if
    out = d
    deallocate(d, e)
  end subroutine solve_dsterf

  subroutine br_workspace(n, d0, e0, lwork, liwork, info)
    integer, intent(in) :: n
    real(dp), intent(in) :: d0(n), e0(n-1)
    integer, intent(out) :: lwork, liwork, info
    real(dp), allocatable :: d(:), e(:), work(:)
    integer, allocatable :: iwork(:)
    allocate(d(n), e(n-1), work(1), iwork(1))
    d = d0
    e = e0
    call dstedc_dc_br(n, d, e, work, -1, iwork, -1, info)
    lwork = max(1, int(work(1)))
    liwork = max(1, iwork(1))
    deallocate(d, e, work, iwork)
  end subroutine br_workspace

  subroutine solve_br(n, d0, e0, work, lwork, iwork, liwork, out, t_ms, info)
    integer, intent(in) :: n, lwork, liwork
    real(dp), intent(in) :: d0(n), e0(n-1)
    real(dp), intent(inout) :: work(lwork)
    integer, intent(inout) :: iwork(liwork)
    real(dp), intent(out) :: out(n), t_ms
    integer, intent(out) :: info
    integer :: sort_info
    real(dp), allocatable :: d(:), e(:)
    real(dp) :: t0
    allocate(d(n), e(n-1))
    d = d0
    e = e0
    t0 = omp_get_wtime()
    call dstedc_dc_br(n, d, e, work, lwork, iwork, liwork, info)
    t_ms = 1000.0d0 * (omp_get_wtime() - t0)
    if (info .eq. 0) then
       call dlasrt('I', n, d, sort_info)
       if (sort_info .ne. 0) info = sort_info
    end if
    out = d
    deallocate(d, e)
  end subroutine solve_br

  subroutine max_error(n, x, y, err, idx)
    integer, intent(in) :: n
    real(dp), intent(in) :: x(n), y(n)
    real(dp), intent(out) :: err
    integer, intent(out) :: idx
    integer :: i
    real(dp) :: cur
    err = -1.0d0
    idx = 1
    do i = 1, n
       cur = abs(x(i) - y(i))
       if (cur .gt. err) then
          err = cur
          idx = i
       end if
    end do
  end subroutine max_error

  real(dp) function workspace_mb(lwork, liwork)
    integer, intent(in) :: lwork, liwork
    workspace_mb = (8.0d0*dble(lwork) + 4.0d0*dble(liwork)) / (1024.0d0*1024.0d0)
  end function workspace_mb

  real(dp) function checksum(n, w)
    integer, intent(in) :: n
    real(dp), intent(in) :: w(n)
    integer :: i
    checksum = 0.0d0
    do i = 1, n
       checksum = checksum + w(i) * dble(mod(i, 127) + 1)
    end do
  end function checksum
end program bench_cache_accuracy_16k_65k
F90

echo "=== compile benchmark ==="
gfortran -O2 -fopenmp -std=legacy -ffree-line-length-none \
  "$SRC" "$BUILD_DIR"/br_objs/*.o \
  -L"$MKLROOT/lib" -Wl,-rpath,"$MKLROOT/lib" \
  -lmkl_rt -lpthread -lm -ldl \
  -o "$BUILD_DIR/bench_cache_accuracy_16k_65k"

export OMP_NUM_THREADS=56
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export MKL_INTERFACE_LAYER=LP64
export MKL_THREADING_LAYER=GNU

OUT="$RESULT_DIR/cache_accuracy_16k_65k.csv"
"$BUILD_DIR/bench_cache_accuracy_16k_65k" | tee "$OUT"

echo "=== result files ==="
find "$RESULT_DIR" -maxdepth 1 -type f -print -exec ls -lh {} \;
echo "=== done ==="
date
