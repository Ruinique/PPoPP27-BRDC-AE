#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C
export LANG=C

ROOT="${AE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BENCH_DIR="$ROOT/cpu/bench"
export CACHE_DIR="${CACHE_DIR:-${SPECTRUM_DE_DIR:-$ROOT/results/data}}"
JOB_TAG="${JH_JOBID:-manual}"
BUILD_DIR="$ROOT/results/cpu/build_cache_accuracy_16k_49152_internal_cases_${JOB_TAG}"
RESULT_DIR="$ROOT/results/cpu/results_cache_accuracy_16k_49152_internal_cases_${JOB_TAG}"
SRC="$BUILD_DIR/bench_cache_accuracy_case.f90"

mkdir -p "$BUILD_DIR/br_objs_ilp64" "$RESULT_DIR"
cd "$ROOT"

echo "=== per-case cache accuracy retest N=16384/49152 with internal D&C ==="
date
hostname
pwd
echo "CACHE_DIR=$CACHE_DIR"
if command -v module >/dev/null 2>&1; then module use /public/software/modules/intel || true || true; fi
if command -v module >/dev/null 2>&1; then module use /public/software/modules/compiler || true || true; fi
if command -v module >/dev/null 2>&1; then module load tbb/2022.3 compiler-rt/2025.3.0 mkl/2025.3 2>&1 || true || true; fi
gfortran --version | head -1
echo "MKLROOT=${MKLROOT:-}"

echo "=== compile BR objects (ILP64) ==="
rm -f "$BUILD_DIR"/br_objs_ilp64/*.o
for src in \
  cpu/src/dlaed0_br.f \
  cpu/src/dlaed4_br.f \
  cpu/src/dlaed7_br.f \
  cpu/src/dlaed9_br.f \
  cpu/src/dstedc_dc_br.f
do
  obj="$BUILD_DIR/br_objs_ilp64/$(basename "$src" .f).o"
  gfortran -O2 -fopenmp -std=legacy -fdefault-integer-8 -c "$src" -o "$obj"
done

cat > "$SRC" <<'F90'
program bench_cache_accuracy_case
  use omp_lib, only: omp_get_wtime, omp_set_num_threads
  implicit none

  integer, parameter :: dp = kind(1.0d0)
  character(len=32) :: mode, arg
  character(len=256) :: cache_dir
  character(len=512) :: cache_path
  integer :: n, stat, threads
  integer :: info_ref, info_br, info_hid
  integer :: lw_br, liw_br, lw_hid, liw_hid
  integer :: max_idx_br, max_idx_hid, alloc_stat
  integer, allocatable :: iw_br(:), iw_hid(:)
  real(dp), allocatable :: d0(:), e0(:), ref(:), br(:), hidden(:)
  real(dp), allocatable :: work_br(:), work_hid(:), q(:,:), qstore(:,:)
  real(dp) :: t_ref_ms, t_br_ms, t_hid_ms
  real(dp) :: br_max_abs, br_norm_abs, hid_max_abs, hid_norm_abs

  external dsterf, dlasrt, dlaed0, dstedc_dc_br, mkl_set_num_threads

  if (command_argument_count() .ne. 2) stop 2
  call get_command_argument(1, mode)
  call get_command_argument(2, arg)
  read(arg, *) n
  threads = 56

  call get_environment_variable('CACHE_DIR', cache_dir, status=stat)
  if (stat .ne. 0 .or. len_trim(cache_dir) .eq. 0) stop 3

  allocate(d0(n), e0(n-1), ref(n), br(n), hidden(n), stat=alloc_stat)
  if (alloc_stat .ne. 0) stop 4
  call load_case(trim(mode), n, d0, e0, cache_path)

  call mkl_set_num_threads(1)
  call solve_dsterf(n, d0, e0, ref, t_ref_ms, info_ref)
  if (info_ref .ne. 0) stop 10

  call omp_set_num_threads(threads)
  call mkl_set_num_threads(threads)

  call br_workspace(n, d0, e0, lw_br, liw_br, info_br)
  if (info_br .ne. 0) stop 11
  allocate(work_br(lw_br), iw_br(liw_br), stat=alloc_stat)
  if (alloc_stat .ne. 0) stop 12
  call solve_br(n, d0, e0, work_br, lw_br, iw_br, liw_br, br, t_br_ms, info_br)
  deallocate(work_br, iw_br)
  if (info_br .ne. 0) stop 13

  call hidden_workspace(n, lw_hid, liw_hid)
  write(0,'(a,a,a,i0,a,f12.2)') 'internal query mode=', trim(mode), &
       ' n=', n, ' workspace_mb=', workspace_mb(lw_hid, liw_hid)
  allocate(work_hid(lw_hid), stat=alloc_stat)
  if (alloc_stat .ne. 0) stop 14
  allocate(iw_hid(liw_hid), stat=alloc_stat)
  if (alloc_stat .ne. 0) stop 15
  allocate(q(n,n), stat=alloc_stat)
  if (alloc_stat .ne. 0) stop 16
  allocate(qstore(n,n), stat=alloc_stat)
  if (alloc_stat .ne. 0) stop 17
  call solve_hidden(n, d0, e0, work_hid, lw_hid, iw_hid, liw_hid, &
       q, n, qstore, n, hidden, t_hid_ms, info_hid)

  call max_error(n, br, ref, br_max_abs, max_idx_br)
  call max_error(n, hidden, ref, hid_max_abs, max_idx_hid)
  br_norm_abs = br_max_abs / max(1.0d0, maxval(abs(ref)))
  hid_norm_abs = hid_max_abs / max(1.0d0, maxval(abs(ref)))

  write(*,'(a,",",i0,",",i0,",",f12.4,",",f12.4,",",f12.4,",",i0,",",i0,",",i0,' // &
       '",",es12.4,",",es12.4,",",es12.4,",",es12.4,",",i0,",",i0,' // &
       '",",es24.16,",",es24.16,",",es24.16,",",f12.2,",",f12.2,",",a)') &
       trim(mode), n, threads, t_ref_ms, t_br_ms, t_hid_ms, &
       info_ref, info_br, info_hid, br_max_abs, br_norm_abs, &
       hid_max_abs, hid_norm_abs, max_idx_br, max_idx_hid, &
       checksum(n, ref), checksum(n, br), checksum(n, hidden), &
       workspace_mb(lw_br, liw_br), workspace_mb(lw_hid, liw_hid), trim(cache_path)
  call flush(6)
  call exit(0)

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
    integer :: file_n
    integer :: unit
    path = trim(cache_dir)//'/'//trim(mode)//'_'//trim(adjustl(itoa(n)))//'.de.bin'
    open(newunit=unit, file=trim(path), access='stream', form='unformatted', &
         status='old', action='read')
    read(unit) file_n
    if (file_n .ne. n) stop 20
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

  subroutine solve_hidden(n, d0, e0, work, lwork, iwork, liwork, &
       q, ldq, qstore, ldqs, out, t_ms, info)
    integer, intent(in) :: n, lwork, liwork, ldq, ldqs
    integer, intent(out) :: info
    integer, intent(inout) :: iwork(liwork)
    real(dp), intent(in) :: d0(n), e0(n-1)
    real(dp), intent(inout) :: work(lwork), q(ldq,*), qstore(ldqs,*)
    real(dp), intent(out) :: out(n), t_ms
    real(dp), allocatable :: d(:), e(:)
    real(dp) :: t0
    allocate(d(n), e(n-1))
    d = d0
    e = e0
    t0 = omp_get_wtime()
    call hidden_dc_values(n, d, e, work, lwork, iwork, liwork, q, ldq, qstore, ldqs, info)
    t_ms = 1000.0d0 * (omp_get_wtime() - t0)
    out = d
    deallocate(d, e)
  end subroutine solve_hidden

  subroutine hidden_dc_values(n, d, e, work, lwork, iwork, liwork, &
       q, ldq, qstore, ldqs, info)
    integer, intent(in) :: n, lwork, liwork, ldq, ldqs
    integer, intent(out) :: info
    integer, intent(inout) :: iwork(liwork)
    real(dp), intent(inout) :: d(n), e(*), work(lwork)
    real(dp), intent(inout) :: q(ldq,*), qstore(ldqs,*)
    integer :: start, finish, m, smlsiz
    real(dp) :: orgnrm, eps, tiny
    integer, external :: ilaenv
    real(dp), external :: dlamch, dlanst
    external dlaed0, dlascl, dlasrt, dsterf
    info = 0
    if (n .le. 1) return
    smlsiz = ilaenv(9, 'DSTEDC', ' ', 0, 0, 0, 0)
    if (n .le. smlsiz) then
       call dsterf(n, d, e, info)
       return
    end if
    orgnrm = dlanst('M', n, d, e)
    if (orgnrm .le. 0.0d0) return
    eps = dlamch('Epsilon')
    start = 1
    do while (start .le. n)
       finish = start
       do while (finish .lt. n)
          tiny = eps*sqrt(abs(d(finish)))*sqrt(abs(d(finish+1)))
          if (abs(e(finish)) .le. tiny) exit
          finish = finish + 1
       end do
       m = finish - start + 1
       if (m .gt. 1) then
          if (m .gt. smlsiz) then
             orgnrm = dlanst('M', m, d(start), e(start))
             call dlascl('G', 0, 0, orgnrm, 1.0d0, m, 1, d(start), m, info)
             call dlascl('G', 0, 0, orgnrm, 1.0d0, m-1, 1, e(start), m-1, info)
             call dlaed0(0, n, m, d(start), e(start), q, ldq, qstore, &
                  ldqs, work, iwork, info)
             if (info .ne. 0) return
             call dlascl('G', 0, 0, 1.0d0, orgnrm, m, 1, d(start), m, info)
          else
             call dsterf(m, d(start), e(start), info)
             if (info .ne. 0) return
          end if
       end if
       start = finish + 1
    end do
    call dlasrt('I', n, d, info)
  end subroutine hidden_dc_values

  subroutine hidden_workspace(n, lwork, liwork)
    integer, intent(in) :: n
    integer, intent(out) :: lwork, liwork
    integer :: smlsiz, lgn, tstore, inner
    integer, external :: ilaenv
    smlsiz = ilaenv(9, 'DSTEDC', ' ', 0, 0, 0, 0)
    lgn = int(log(real(max(2,n), dp))/log(2.0d0)) + 2
    tstore = max(1, n * max(1, lgn))
    inner = 3*n + 3*n*max(1, lgn) + 4*n + n*n
    lwork = 1 + 4*n + n*n + tstore + inner
    liwork = 6 + 6*n + 5*n*max(1, lgn)
    if (n .le. smlsiz) then
       lwork = max(1, 2*n)
       liwork = 1
    end if
  end subroutine hidden_workspace

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
end program bench_cache_accuracy_case
F90

echo "=== compile per-case benchmark (ILP64) ==="
gfortran -O2 -fopenmp -std=legacy -ffree-line-length-none -fdefault-integer-8 \
  "$SRC" "$BUILD_DIR"/br_objs_ilp64/*.o \
  -L"$MKLROOT/lib" -Wl,-rpath,"$MKLROOT/lib" \
  -lmkl_rt -lpthread -lm -ldl \
  -o "$BUILD_DIR/bench_cache_accuracy_case"

export OMP_NUM_THREADS=56
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=56
export MKL_NUM_THREADS=56
export MKL_DYNAMIC=FALSE
export MKL_INTERFACE_LAYER=ILP64
export MKL_THREADING_LAYER=GNU

OUT="$RESULT_DIR/cache_accuracy_16k_49152_internal.csv"
FAIL="$RESULT_DIR/cache_accuracy_16k_49152_internal_failures.csv"
echo 'mode,n,threads,dsterf_ms,br_ms,internal_ms,info_dsterf,info_br,info_internal,br_max_abs_err,br_norm_abs_err,internal_max_abs_err,internal_norm_abs_err,br_max_idx,internal_max_idx,checksum_ref,checksum_br,checksum_internal,br_workspace_mb,internal_workspace_mb,cache_file' | tee "$OUT"
echo 'mode,n,status' > "$FAIL"

for mode in uniform normal toeplitz clustered; do
  for n in 16384 49152; do
    echo "=== case mode=$mode n=$n ==="
    set +e
    "$BUILD_DIR/bench_cache_accuracy_case" "$mode" "$n" | tee -a "$OUT"
    status=${PIPESTATUS[0]}
    set -e
    if [ "$status" -ne 0 ]; then
      echo "$mode,$n,$status" | tee -a "$FAIL"
    fi
  done
done

echo "=== result files ==="
find "$RESULT_DIR" -maxdepth 1 -type f -print -exec ls -lh {} \;
echo "=== failures ==="
cat "$FAIL"
echo "=== done ==="
date
