#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C
export LANG=C

ROOT="${AE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BENCH_DIR="$ROOT/cpu/bench"
JOB_TAG="${JH_JOBID:-manual}"
BUILD_DIR="$ROOT/results/cpu/build_mpfr_value_probe_${JOB_TAG}"
RESULT_DIR="$ROOT/results/cpu/results_mpfr_value_probe_${JOB_TAG}"
SRC="$BUILD_DIR/bench_mpfr_value_probe.f90"

mkdir -p "$BUILD_DIR/br_objs" "$RESULT_DIR"
cd "$ROOT"

echo "=== MPFR value probe support: glued Wilkinson solver values ==="
date
hostname
pwd
if command -v module >/dev/null 2>&1; then module use /public/software/modules/intel || true || true; fi
if command -v module >/dev/null 2>&1; then module use /public/software/modules/compiler || true || true; fi
if command -v module >/dev/null 2>&1; then module load tbb/2022.3 compiler-rt/2025.3.0 mkl/2025.3 2>&1 || true || true; fi
gfortran --version | head -1
echo "MKLROOT=${MKLROOT:-}"

echo "=== compile BR objects (LP64) ==="
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
program bench_mpfr_value_probe
  use omp_lib, only: omp_get_wtime, omp_set_num_threads
  implicit none
  integer, parameter :: dp = kind(1.0d0)
  integer, parameter :: ncases = 2, max_probe = 16
  integer, dimension(ncases), parameter :: ns = (/ 16384, 49152 /)
  integer, dimension(max_probe, ncases) :: probes
  integer, dimension(ncases) :: nprobe
  integer :: ic, n, p
  integer :: info_dsterf, info_br, info_hid, lw_br, liw_br, lw_hid, liw_hid
  integer, allocatable :: iw_br(:), iw_hid(:)
  real(dp), allocatable :: d0(:), e0(:), dsterf_w(:), br_w(:), hidden_w(:)
  real(dp), allocatable :: work_br(:), work_hid(:), q(:,:), qstore(:,:)
  real(dp) :: t_dsterf_ms, t_br_ms, t_hid_ms

  external dsterf, dlasrt, dlaed0, dstedc_dc_br, mkl_set_num_threads

  probes = 0
  nprobe = 0
  nprobe(1) = 9
  probes(1:nprobe(1),1) = (/ 16255, 16256, 16257, 16258, 16259, 16260, 16261, 16262, 16263 /)
  nprobe(2) = 11
  probes(1:nprobe(2),2) = (/ 47586, 47587, 47588, 47589, 47590, 47591, 47592, 47593, 47594, 47595, 47596 /)

  call mkl_set_num_threads(56)
  call omp_set_num_threads(56)

  print '(a)', 'mode,n,index,dsterf,br,internal,dsterf_next_gap,br_next_gap,internal_next_gap,info_dsterf,info_br,info_internal,dsterf_ms,br_ms,internal_ms'
  call flush(6)

  do ic = 1, ncases
     n = ns(ic)
     allocate(d0(n), e0(n-1), dsterf_w(n), br_w(n), hidden_w(n))
     call fill_glued_wilkinson(n, d0, e0)

     call solve_dsterf(n, d0, e0, dsterf_w, t_dsterf_ms, info_dsterf)

     call br_workspace(n, d0, e0, lw_br, liw_br, info_br)
     allocate(work_br(lw_br), iw_br(liw_br))
     call solve_br(n, d0, e0, work_br, lw_br, iw_br, liw_br, br_w, t_br_ms, info_br)
     deallocate(work_br, iw_br)

     call hidden_workspace(n, lw_hid, liw_hid)
     allocate(work_hid(lw_hid), iw_hid(liw_hid), q(n,n), qstore(n,n))
     call solve_hidden(n, d0, e0, work_hid, lw_hid, iw_hid, liw_hid, q, n, qstore, n, hidden_w, t_hid_ms, info_hid)
     deallocate(work_hid, iw_hid, q, qstore)

     do p = 1, nprobe(ic)
        call emit_index(n, probes(p, ic), dsterf_w, br_w, hidden_w, info_dsterf, info_br, info_hid, t_dsterf_ms, t_br_ms, t_hid_ms)
     end do

     deallocate(d0, e0, dsterf_w, br_w, hidden_w)
  end do

contains
  subroutine emit_index(n, k, dsterf_w, br_w, hidden_w, info_dsterf, info_br, info_hid, t_dsterf_ms, t_br_ms, t_hid_ms)
    integer, intent(in) :: n, k, info_dsterf, info_br, info_hid
    real(dp), intent(in) :: dsterf_w(n), br_w(n), hidden_w(n)
    real(dp), intent(in) :: t_dsterf_ms, t_br_ms, t_hid_ms
    real(dp) :: gd, gb, gh
    gd = 0.0d0
    gb = 0.0d0
    gh = 0.0d0
    if (k .lt. n) then
       gd = dsterf_w(k+1) - dsterf_w(k)
       gb = br_w(k+1) - br_w(k)
       gh = hidden_w(k+1) - hidden_w(k)
    end if
    write(*,'(a,",",i0,",",i0,",",es25.17,",",es25.17,",",es25.17,",",es12.4,",",es12.4,",",es12.4,",",i0,",",i0,",",i0,",",f12.4,",",f12.4,",",f12.4)') &
      'glued_wilkinson', n, k, dsterf_w(k), br_w(k), hidden_w(k), gd, gb, gh, info_dsterf, info_br, info_hid, t_dsterf_ms, t_br_ms, t_hid_ms
  end subroutine emit_index

  subroutine fill_glued_wilkinson(n, d, e)
    integer, intent(in) :: n
    real(dp), intent(out) :: d(n), e(n-1)
    integer :: i, block_size, local_index
    real(dp) :: center
    block_size = max(3, n / 25)
    if (mod(block_size, 2) .eq. 0) block_size = block_size + 1
    do i = 1, n
       local_index = mod(i - 1, block_size) + 1
       center = 0.5d0 * dble(block_size + 1)
       d(i) = abs(center - dble(local_index))
    end do
    do i = 1, n-1
       e(i) = 1.0d0
    end do
  end subroutine fill_glued_wilkinson

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

  subroutine solve_hidden(n, d0, e0, work, lwork, iwork, liwork, q, ldq, qstore, ldqs, out, t_ms, info)
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

  subroutine hidden_dc_values(n, d, e, work, lwork, iwork, liwork, q, ldq, qstore, ldqs, info)
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
             call dlaed0(0, n, m, d(start), e(start), q, ldq, qstore, ldqs, work, iwork, info)
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
    integer :: lgn, tstore, inner, smlsiz
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
end program bench_mpfr_value_probe
F90

echo "=== compile probe ==="
gfortran -O2 -fopenmp -std=legacy -ffree-line-length-none \
  "$SRC" "$BUILD_DIR"/br_objs/*.o \
  -L"$MKLROOT/lib" -Wl,-rpath,"$MKLROOT/lib" \
  -lmkl_rt -lpthread -lm -ldl \
  -o "$BUILD_DIR/bench_mpfr_value_probe"

export OMP_NUM_THREADS=56
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=56
export MKL_NUM_THREADS=56
export MKL_DYNAMIC=FALSE
export MKL_INTERFACE_LAYER=LP64
export MKL_THREADING_LAYER=GNU

OUT="$RESULT_DIR/mpfr_value_probe.csv"
"$BUILD_DIR/bench_mpfr_value_probe" | tee "$OUT"

echo "=== result files ==="
find "$RESULT_DIR" -maxdepth 1 -type f -print -exec ls -lh {} \;
echo "=== done ==="
date
