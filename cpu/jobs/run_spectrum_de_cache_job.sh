#!/bin/bash
set -euo pipefail

export LC_ALL=C
export LANG=C

ROOT="${AE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BENCH_DIR="$ROOT/cpu/bench"
JOB_TAG="${JH_JOBID:-manual}"
BUILD_DIR="$ROOT/results/cpu/build_spectrum_de_cache_${JOB_TAG}"
RESULT_DIR="${SPECTRUM_DE_DIR:-$ROOT/results/data}"
SRC="$BUILD_DIR/gen_spectrum_de_cache.f90"

mkdir -p "$BUILD_DIR" "$RESULT_DIR"
cd "$ROOT"

echo "=== spectrum-prescribed D/E cache generation ==="
date
hostname
if command -v module >/dev/null 2>&1; then module load tbb/2022.3 compiler-rt/2025.3.0 mkl/2025.3 2>&1 || true || true; fi
gfortran --version | head -1

cat > "$SRC" <<'F90'
program gen_spectrum_de_cache
  use iso_fortran_env, only: int64
  implicit none
  integer, parameter :: dp = kind(1.0d0)
  integer, parameter :: nm = 4, nn = 5
  character(len=16) :: modes(nm)
  integer :: ns(nn), imode, inum, n, info
  character(len=512) :: outdir, path
  real(dp), allocatable :: d(:), e(:)

  call get_command_argument(1, outdir)
  if (len_trim(outdir) .eq. 0) stop 2
  modes = (/ 'uniform         ', 'normal          ', 'toeplitz        ', 'clustered       ' /)
  ns = (/ 8192, 16384, 32768, 49152, 65536 /)
  print '(a)', 'mode,n,path,generator,info'
  do imode = 1, nm
     do inum = 1, nn
        n = ns(inum)
        allocate(d(n), e(n-1))
        call fill_case(trim(modes(imode)), n, d, e, info)
        write(path,'(a,"/",a,"_",i0,".de.bin")') trim(outdir), trim(modes(imode)), n
        if (info .eq. 0) call write_de(trim(path), n, d, e)
        write(*,'(a,",",i0,",",a,",",a,",",i0)') trim(modes(imode)), n, trim(path), &
             generator_name(trim(modes(imode))), info
        deallocate(d, e)
     end do
  end do
contains
  subroutine write_de(path, n, d, e)
    character(len=*), intent(in) :: path
    integer, intent(in) :: n
    real(dp), intent(in) :: d(n), e(n-1)
    integer(int64) :: n64
    integer :: ios
    n64 = int(n, int64)
    open(unit=20, file=path, access='stream', form='unformatted', &
         status='replace', action='write', iostat=ios)
    if (ios .ne. 0) stop 20
    write(20) n64
    write(20) d
    write(20) e
    close(20)
  end subroutine write_de

  subroutine fill_case(mode, n, d, e, info)
    character(len=*), intent(in) :: mode
    integer, intent(in) :: n
    integer, intent(out) :: info
    real(dp), intent(out) :: d(n), e(n-1)
    integer :: i
    info = 0
    if (mode .eq. 'toeplitz') then
       do i = 1, n
          d(i) = 2.0d0
       end do
       do i = 1, n - 1
          e(i) = 0.25d0
       end do
    else if (mode .eq. 'uniform') then
       call fill_dlatms(n, 'U', 6, 1.0d0, d, e, info)
    else if (mode .eq. 'normal') then
       call fill_dlatms(n, 'N', 6, 1.0d0, d, e, info)
    else if (mode .eq. 'clustered') then
       call fill_dlatms(n, 'U', 1, 1.0d12, d, e, info)
    else
       info = -1
    end if
  end subroutine fill_case

  subroutine fill_dlatms(n, dist, mode, cond, d, e, info)
    integer, intent(in) :: n, mode
    character(len=1), intent(in) :: dist
    real(dp), intent(in) :: cond
    integer, intent(out) :: info
    real(dp), intent(out) :: d(n), e(n-1)
    integer :: iseed(4)
    real(dp), allocatable :: eig(:), ab(:,:), wk(:)
    allocate(eig(n), ab(2,n), wk(3*n))
    iseed = (/ 1, 2, mod(n, 4093) + mode, 5 /)
    call dlatms(n, n, dist, iseed, 'S', eig, mode, cond, 1.0d0, &
         1, 1, 'B', ab, 2, wk, info)
    if (info .eq. 0) then
       d = ab(1,1:n)
       e = ab(2,1:n-1)
    end if
    deallocate(eig, ab, wk)
  end subroutine fill_dlatms

  character(len=32) function generator_name(mode)
    character(len=*), intent(in) :: mode
    if (mode .eq. 'toeplitz') then
       generator_name = 'structured_toeplitz'
    else
       generator_name = 'dlatms_symmetric_band'
    end if
  end function generator_name
end program gen_spectrum_de_cache
F90

gfortran -O2 "$SRC" \
  -L"$MKLROOT/lib" -Wl,-rpath,"$MKLROOT/lib" \
  -lmkl_rt -lpthread -lm -ldl \
  -o "$BUILD_DIR/gen_spectrum_de_cache"

"$BUILD_DIR/gen_spectrum_de_cache" "$RESULT_DIR" | tee "$RESULT_DIR/manifest.csv"
ls -lh "$RESULT_DIR"
echo "=== done ==="
date
