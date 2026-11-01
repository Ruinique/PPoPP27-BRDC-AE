#!/usr/bin/env bash
export LC_ALL=C
export LANG=C
source /etc/profile >/dev/null 2>&1 || true
set -euo pipefail

ROOT="${AE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BENCH_DIR="$ROOT/cpu/bench"
JOB_TAG="${JH_JOBID:-manual}"
BUILD_DIR="$ROOT/results/cpu/build_application_tridiag_fast_${JOB_TAG}"
RESULT_DIR="$ROOT/results/cpu/results_supplemental/application_tridiag_fast_${JOB_TAG}"
mkdir -p "$BUILD_DIR"/br_objs "$RESULT_DIR"
cd "$ROOT"

if command -v module >/dev/null 2>&1; then module use /public/software/modules/intel || true || true; fi
if command -v module >/dev/null 2>&1; then module use /public/software/modules/compiler || true || true; fi
if command -v module >/dev/null 2>&1; then module load tbb/2022.3 compiler-rt/2025.3.0 mkl/2025.3 2>&1 || true || true; fi

BR_SRCS=(cpu/src/dlaed0_br.f cpu/src/dlaed4_br.f cpu/src/dlaed7_br.f cpu/src/dlaed9_br.f cpu/src/dstedc_dc_br.f)
rm -f "$BUILD_DIR"/br_objs/*.o
cc -O2 -DOS_LINUX -DSMP -I"$ROOT" -c cpu/src/openblas_get_num_nodes.c -o "$BUILD_DIR/br_objs/openblas_get_num_nodes.o"
for src in "${BR_SRCS[@]}"; do
  gfortran -O2 -fopenmp -std=legacy -c "$src" -o "$BUILD_DIR/br_objs/$(basename "$src" .f).o"
done

cat > "$BUILD_DIR/bench_application_tridiag_fast.f90" <<'F90'
program bench_application_tridiag_fast
  use omp_lib, only: omp_get_wtime, omp_get_max_threads
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none
  integer, parameter :: dp = kind(1.0d0)
  integer :: cases(7), sizes(7), idx, icase, n, reps, r, info, lwork, liwork
  integer, allocatable :: iwork(:)
  real(dp), allocatable :: d0(:), e0(:), d(:), e(:), ref(:), eref(:), work(:)
  real(dp) :: qwork(1), t0, elapsed, best_dsterf, best_br, err, checksum_ref, checksum_br
  character(len=32) :: name
  integer(int64) :: seed
  external dsterf, dstedc_dc_br
  cases = (/ 1, 2, 3, 4, 5, 6, 5 /)
  sizes = (/ 16384, 16384, 16384, 16384, 16384, 16384, 65536 /)
  write(*,'(a,i0)') 'omp_max_threads=', omp_get_max_threads()
  write(*,'(a)') 'case,n,reps,dsterf_ms,br_ms,dsterf_over_br,lwork,liwork,workspace_mb,max_abs_err,checksum_ref,checksum_br'
  do idx = 1, size(cases)
     icase = cases(idx); n = sizes(idx); reps = merge(3, 1, n <= 16384)
     allocate(d0(n), e0(n-1), d(n), e(n-1), ref(n), eref(n-1), work(1), iwork(1))
     seed = 1469598103934665603_int64 + int(icase, int64) * 1099511628211_int64 + int(n, int64)
     call fill_case(icase, n, seed, name, d0, e0)
     d=d0; e=e0
     call dstedc_dc_br(n, d, e, qwork, -1, iwork, -1, info)
     lwork=max(1,int(qwork(1))); liwork=max(1,iwork(1))
     deallocate(work,iwork); allocate(work(lwork),iwork(liwork))
     best_dsterf=huge(1.0d0)
     do r=1,reps
        ref=d0; eref=e0; t0=omp_get_wtime(); call dsterf(n,ref,eref,info); elapsed=omp_get_wtime()-t0
        if (info /= 0) stop 3
        if (elapsed < best_dsterf) then; best_dsterf=elapsed; checksum_ref=checksum(n,ref); end if
     end do
     best_br=huge(1.0d0)
     do r=1,reps
        d=d0; e=e0; t0=omp_get_wtime(); call dstedc_dc_br(n,d,e,work,lwork,iwork,liwork,info); elapsed=omp_get_wtime()-t0
        if (info /= 0) stop 4
        if (elapsed < best_br) then; best_br=elapsed; checksum_br=checksum(n,d); err=maxval(abs(d-ref)); end if
     end do
     write(*,'(a,",",i0,",",i0,",",f12.4,",",f12.4,",",f12.4,",",i0,",",i0,",",f12.4,",",es12.4,",",es16.8,",",es16.8)') &
       trim(name), n, reps, 1000.0d0*best_dsterf, 1000.0d0*best_br, best_dsterf/best_br, lwork, liwork, &
       (8.0d0*lwork+4.0d0*liwork)/(1024.0d0*1024.0d0), err, checksum_ref, checksum_br
     deallocate(d0,e0,d,e,ref,eref,work,iwork)
  end do
contains
  subroutine fill_case(icase,n,seed,name,d,e)
    integer,intent(in)::icase,n; integer(int64),intent(inout)::seed
    character(len=32),intent(out)::name; real(dp),intent(out)::d(n),e(n-1)
    integer::i; real(dp)::x,h,width,w
    select case(icase)
    case(1); name='gauss_legendre_jacobi'; d=0.0d0; do i=1,n-1; e(i)=real(i,dp)/sqrt(real(4*i*i-1,dp)); end do
    case(2); name='gauss_hermite_jacobi'; d=0.0d0; do i=1,n-1; e(i)=sqrt(0.5d0*real(i,dp)); end do
    case(3); name='ehrenfest_birth_death'; d=real(n-1,dp); do i=1,n-1; e(i)=-sqrt(real(i,dp)*real(n-i,dp)); end do
    case(4); name='one_d_laplacian'; d=2.0d0; e=-1.0d0
    case(5); name='anderson_schrodinger'; w=4.0d0; do i=1,n; d(i)=2.0d0+w*(randu(seed)-0.5d0); end do; e=-1.0d0
    case(6); name='harmonic_oscillator_fd'; width=8.0d0; h=2.0d0*width/real(n+1,dp); do i=1,n; x=-width+h*real(i,dp); d(i)=2.0d0/(h*h)+x*x; end do; e=-1.0d0/(h*h)
    end select
  end subroutine
  real(dp) function randu(seed)
    integer(int64),intent(inout)::seed; integer(int64)::x
    x=seed; x=ieor(x,shiftr(x,12)); x=ieor(x,shiftl(x,25)); x=ieor(x,shiftr(x,27)); seed=x
    randu=real(iand(x*2685821657736338717_int64,huge(1_int64)),dp)/real(huge(1_int64),dp)
  end function
  real(dp) function checksum(n,d)
    integer,intent(in)::n; real(dp),intent(in)::d(n); integer::stride
    stride=max(1,n/32); checksum=d(1)+d(n)+sum(d(1:n:stride))
  end function
end program
F90

gfortran -O2 -fopenmp -ffree-line-length-none "$BUILD_DIR/bench_application_tridiag_fast.f90" "$BUILD_DIR"/br_objs/*.o \
  -L"$MKLROOT/lib" -Wl,-rpath,"$MKLROOT/lib" -lmkl_rt -lpthread -lm -ldl -o "$BUILD_DIR/bench_application_tridiag_fast"

export OMP_NUM_THREADS=56 OMP_PROC_BIND=close OMP_PLACES=cores OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE MKL_INTERFACE_LAYER=LP64 MKL_THREADING_LAYER=GNU
"$BUILD_DIR/bench_application_tridiag_fast" | tee "$RESULT_DIR/application_tridiag_fast_t56.csv"
echo "result=$RESULT_DIR/application_tridiag_fast_t56.csv"
date
