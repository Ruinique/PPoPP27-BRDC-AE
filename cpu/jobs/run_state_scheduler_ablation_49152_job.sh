#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C
export LANG=C

ROOT="${AE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BENCH_DIR="$ROOT/cpu/bench"
JOB_TAG="${JH_JOBID:-manual}"
BUILD_DIR="$ROOT/results/cpu/build_state_scheduler_ablation_49152_${JOB_TAG}"
RESULT_DIR="$ROOT/results/cpu/results_state_scheduler_ablation_49152_${JOB_TAG}"
SRC="$BUILD_DIR/bench_state_scheduler_ablation_49152.f90"
DE_CACHE="$ROOT/data/inputs/reduced_dense_49152.de.bin"

mkdir -p "$BUILD_DIR"/br_objs "$RESULT_DIR"
cd "$ROOT"

echo "=== state/scheduler ablation, reduced-dense N=49152 ==="
date
hostname
if command -v module >/dev/null 2>&1; then module use /public/software/modules/intel || true || true; fi
if command -v module >/dev/null 2>&1; then module use /public/software/modules/compiler || true || true; fi
if command -v module >/dev/null 2>&1; then module load tbb/2022.3 compiler-rt/2025.3.0 mkl/2025.3 2>&1 || true || true; fi
gfortran --version | head -1
echo "MKLROOT=${MKLROOT:-}"
echo "DE_CACHE=$DE_CACHE"

for src in \
  cpu/src/dlaed0_br.f \
  cpu/src/dlaed4_br.f \
  cpu/src/dlaed7_br.f \
  cpu/src/dlaed9_br.f \
  cpu/src/dstedc_dc_br.f
do
  obj="$BUILD_DIR/br_objs/$(basename "$src" .f).o"
  gfortran -O2 -fopenmp -std=legacy -fdefault-integer-8 -c "$src" -o "$obj"
done

cat > "$SRC" <<'F90'
program bench_state_scheduler_ablation_49152
  use omp_lib, only: omp_get_max_threads, omp_get_wtime
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none

  integer, parameter :: dp = kind(1.0d0)
  integer :: n, reps, info
  integer :: lw_br, liw_br, lw_hid, liw_hid
  integer :: run_hidden, run_dsterf
  integer, allocatable :: iw_br(:), iw_hid(:)
  real(dp) :: t_dsterf, t_hidden, t_br, err_hidden, err_br
  real(dp), allocatable :: d0(:), e0(:), d(:), e(:), ref(:)
  real(dp), allocatable :: work_br(:), work_hid(:), q(:,:), qstore(:,:)
  character(len=4096) :: path
  character(len=64) :: variant

  external dsterf, dstedc_dc_br, mkl_set_num_threads

  call get_command_argument(1, path)
  if (len_trim(path) .eq. 0) path = &
       'data/inputs/reduced_dense_49152.de.bin'
  variant = 'unknown'
  call get_environment_variable('BRDC_VARIANT', variant)
  run_hidden = get_env_int('RUN_HIDDEN', 0)
  run_dsterf = get_env_int('RUN_DSTERF', run_hidden)
  reps = 1

  call load_compact_de(trim(path), n, d0, e0)

  write(*,'(a,i0)') 'omp_max_threads=', omp_get_max_threads()
  print '(a)', 'state_variant,scheduler_variant,mode,n,threads,reps,dsterf_ms,replay_ms,br_ms,replay_workspace_mb,br_workspace_mb,replay_err,br_err,checksum_br,replay_lwork,replay_liwork,br_lwork,br_liwork'
  call flush(6)

  allocate(d(n), e(n-1), ref(n), work_br(1), iw_br(1))

  d = d0
  e = e0
  call dstedc_dc_br(n, d, e, work_br, -1, iw_br, -1, info)
  if (info .ne. 0) then
     print *, 'BR query failed n=', n, ' info=', info
     stop 2
  end if
  lw_br = max(1, int(work_br(1)))
  liw_br = max(1, iw_br(1))
  deallocate(work_br, iw_br)
  allocate(work_br(lw_br), iw_br(liw_br))

  t_dsterf = -1.0d0
  t_hidden = -1.0d0
  err_hidden = -1.0d0
  err_br = -1.0d0
  lw_hid = -1
  liw_hid = -1

  if (run_dsterf .ne. 0) then
     call mkl_set_num_threads(1)
     t_dsterf = time_dsterf(n, d0, e0, reps, ref, info)
     if (info .ne. 0) stop 3
  end if

  if (run_hidden .ne. 0) then
     call hidden_workspace(n, lw_hid, liw_hid)
     write(*,'(a,i0,a,i0,a,f12.2)') 'replay query formula lwork=', lw_hid, &
          ' liwork=', liw_hid, ' workspace_mb=', workspace_mb(lw_hid, liw_hid)
     call flush(6)
     allocate(work_hid(lw_hid), iw_hid(liw_hid), q(n,n), qstore(n,n))
     call mkl_set_num_threads(omp_get_max_threads())
     call time_hidden(n, d0, e0, reps, work_hid, lw_hid, iw_hid, liw_hid, &
          q, n, qstore, n, t_hidden, d, info)
     if (info .ne. 0) then
        print *, 'replay hidden failed n=', n, ' info=', info
        stop 4
     end if
     if (run_dsterf .ne. 0) err_hidden = maxval(abs(d - ref))
     deallocate(work_hid, iw_hid, q, qstore)
  end if

  call mkl_set_num_threads(1)
  call time_br(n, d0, e0, reps, work_br, lw_br, iw_br, liw_br, t_br, d, info)
  if (info .ne. 0) then
     print *, 'BR failed n=', n, ' info=', info
     stop 5
  end if
  if (run_dsterf .ne. 0) err_br = maxval(abs(d - ref))

  write(*,'(a,",",a,",reduced_dense,",i0,",",i0,",",i0,",",f12.4,",",f12.4,",",f12.4,",",f12.2,",",f12.2,",",es12.4,",",es12.4,",",es16.8,",",i0,",",i0,",",i0,",",i0)') &
       merge('replay-style', 'BR-state    ', run_hidden .ne. 0), &
       trim(variant), n, omp_get_max_threads(), reps, &
       1000.0d0*t_dsterf, 1000.0d0*t_hidden, 1000.0d0*t_br, &
       workspace_mb(lw_hid, liw_hid), workspace_mb(lw_br, liw_br), &
       err_hidden, err_br, checksum(n, d), lw_hid, liw_hid, lw_br, liw_br
  call flush(6)

contains

  integer function get_env_int(name, default_value)
    character(len=*), intent(in) :: name
    integer, intent(in) :: default_value
    character(len=64) :: buf
    integer :: stat, value
    get_env_int = default_value
    call get_environment_variable(name, buf, status=stat)
    if (stat .ne. 0 .or. len_trim(buf) .eq. 0) return
    read(buf, *, iostat=stat) value
    if (stat .eq. 0) get_env_int = value
  end function get_env_int

  subroutine load_compact_de(path, n, d, e)
    character(len=*), intent(in) :: path
    integer, intent(out) :: n
    real(dp), allocatable, intent(out) :: d(:), e(:)
    integer(int64) :: n64_local
    integer :: unit_local
    unit_local = 10
    open(unit=unit_local, file=trim(path), access='stream', &
         form='unformatted', status='old', action='read')
    read(unit_local) n64_local
    n = int(n64_local)
    allocate(d(n), e(n-1))
    read(unit_local) d
    read(unit_local) e
    close(unit_local)
  end subroutine load_compact_de

  subroutine hidden_workspace(n, lwork, liwork)
    integer, intent(in) :: n
    integer, intent(out) :: lwork, liwork
    integer :: lgn
    integer(int64) :: p, n64_local, lwork64, liwork64
    lgn = 0
    p = 1_int64
    n64_local = int(n, int64)
    do while (p .lt. max(1_int64, n64_local))
       p = 2_int64*p
       lgn = lgn + 1
    end do
    lwork64 = max(1_int64, 1_int64 + 3_int64*n64_local + &
         2_int64*n64_local*int(lgn, int64) + 3_int64*n64_local*n64_local)
    liwork64 = max(1_int64, 6_int64 + 6_int64*n64_local + &
         5_int64*n64_local*int(lgn, int64))
    lwork = int(lwork64)
    liwork = int(liwork64)
  end subroutine hidden_workspace

  real(dp) function workspace_mb(lwork, liwork)
    integer, intent(in) :: lwork, liwork
    if (lwork .lt. 0 .or. liwork .lt. 0) then
       workspace_mb = -1.0d0
    else
       workspace_mb = (8.0d0*dble(lwork) + 4.0d0*dble(liwork)) / &
            (1024.0d0*1024.0d0)
    end if
  end function workspace_mb

  real(dp) function time_dsterf(n, d0, e0, reps, dout, info)
    integer, intent(in) :: n, reps
    integer, intent(out) :: info
    real(dp), intent(in) :: d0(n), e0(n-1)
    real(dp), intent(out) :: dout(n)
    real(dp), allocatable :: d(:), e(:)
    integer :: r
    real(dp) :: elapsed, best, t0
    allocate(d(n), e(n-1))
    best = huge(1.0d0)
    info = 0
    do r = 1, reps
       d = d0
       e = e0
       t0 = omp_get_wtime()
       call dsterf(n, d, e, info)
       elapsed = omp_get_wtime() - t0
       if (info .ne. 0) exit
       if (elapsed .lt. best) then
          best = elapsed
          dout = d
       end if
    end do
    deallocate(d, e)
    time_dsterf = best
  end function time_dsterf

  subroutine time_hidden(n, d0, e0, reps, work, lwork, iwork, liwork, &
       q, ldq, qstore, ldqs, best, dout, info)
    integer, intent(in) :: n, reps, lwork, liwork, ldq, ldqs
    integer, intent(out) :: info
    integer, intent(inout) :: iwork(liwork)
    real(dp), intent(in) :: d0(n), e0(n-1)
    real(dp), intent(inout) :: work(lwork), q(ldq,*), qstore(ldqs,*)
    real(dp), intent(out) :: best, dout(n)
    real(dp), allocatable :: d(:), e(:)
    integer :: r
    real(dp) :: elapsed, t0
    allocate(d(n), e(n-1))
    best = huge(1.0d0)
    info = 0
    do r = 1, reps
       d = d0
       e = e0
       t0 = omp_get_wtime()
       call hidden_dc_values(n, d, e, work, lwork, iwork, liwork, &
            q, ldq, qstore, ldqs, info)
       elapsed = omp_get_wtime() - t0
       if (info .ne. 0) exit
       if (elapsed .lt. best) then
          best = elapsed
          dout = d
       end if
    end do
    deallocate(d, e)
  end subroutine time_hidden

  subroutine time_br(n, d0, e0, reps, work, lwork, iwork, liwork, best, &
       dout, info)
    integer, intent(in) :: n, reps, lwork, liwork
    integer, intent(out) :: info
    integer, intent(inout) :: iwork(liwork)
    real(dp), intent(in) :: d0(n), e0(n-1)
    real(dp), intent(inout) :: work(lwork)
    real(dp), intent(out) :: best, dout(n)
    real(dp), allocatable :: d(:), e(:)
    integer :: r
    real(dp) :: elapsed, t0
    allocate(d(n), e(n-1))
    best = huge(1.0d0)
    info = 0
    do r = 1, reps
       d = d0
       e = e0
       t0 = omp_get_wtime()
       call dstedc_dc_br(n, d, e, work, lwork, iwork, liwork, info)
       elapsed = omp_get_wtime() - t0
       if (info .ne. 0) exit
       if (elapsed .lt. best) then
          best = elapsed
          dout = d
       end if
    end do
    deallocate(d, e)
  end subroutine time_br

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
             if (info .ne. 0) then
                info = (info / (m+1) + start - 1)*(n+1) + mod(info, (m+1)) + start - 1
                return
             end if
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

  real(dp) function checksum(n, w)
    integer, intent(in) :: n
    real(dp), intent(in) :: w(n)
    integer :: i
    checksum = 0.0d0
    do i = 1, n
       checksum = checksum + w(i) * dble(mod(i, 127) + 1)
    end do
  end function checksum
end program bench_state_scheduler_ablation_49152
F90

gfortran -O2 -fopenmp -std=legacy -ffree-line-length-none -fdefault-integer-8 \
  "$SRC" "$BUILD_DIR"/br_objs/*.o \
  -L"$MKLROOT/lib" -Wl,-rpath,"$MKLROOT/lib" \
  -lmkl_rt -lpthread -lm -ldl \
  -o "$BUILD_DIR/bench_state_scheduler_ablation_49152"

export OMP_NUM_THREADS=56
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=56
export MKL_NUM_THREADS=56
export MKL_DYNAMIC=FALSE
export MKL_INTERFACE_LAYER=ILP64
export MKL_THREADING_LAYER=GNU

RAW="$RESULT_DIR/state_scheduler_ablation_49152_raw.csv"
: > "$RAW"

run_case() {
  local variant="$1"
  local run_hidden="$2"
  export BRDC_VARIANT="$variant"
  export RUN_HIDDEN="$run_hidden"
  export RUN_DSTERF="$run_hidden"
  if [ "$variant" = "task_level" ]; then
    export OPENBLAS_BRDC_SCHED=outer
    export OPENBLAS_BRDC_WPROD_THREADS=1
  else
    unset OPENBLAS_BRDC_SCHED || true
    unset OPENBLAS_BRDC_WPROD_THREADS || true
  fi
  echo "=== variant=$variant run_hidden=$run_hidden ==="
  "$BUILD_DIR/bench_state_scheduler_ablation_49152" "$DE_CACHE" | tee -a "$RAW"
}

run_case adaptive 1
run_case task_level 0

python3 - "$RAW" "$RESULT_DIR/state_scheduler_ablation_49152.csv" <<'PY'
import csv
import sys

raw, out = sys.argv[1], sys.argv[2]
rows = []
with open(raw, newline="") as f:
    for row in csv.DictReader(line for line in f if "," in line and not line.startswith("replay query")):
        if row.get("state_variant") in (None, "state_variant"):
            continue
        rows.append(row)

with open(out, "w", newline="") as f:
    fields = [
        "state_variant", "scheduler_variant", "mode", "n", "threads", "reps",
        "dsterf_ms", "replay_ms", "br_ms", "replay_workspace_mb",
        "br_workspace_mb", "replay_err", "br_err", "checksum_br",
        "replay_lwork", "replay_liwork", "br_lwork", "br_liwork",
    ]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in fields})

print(f"wrote {out}")
with open(out) as f:
    print(f.read(), end="")
PY

echo "=== done ==="
date
