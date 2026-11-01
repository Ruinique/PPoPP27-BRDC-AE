program bench_mrrr_bisection
  use omp_lib, only: omp_get_max_threads, omp_get_wtime
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none

  integer, parameter :: dp = kind(1.0d0)
  integer, parameter :: n = 65536
  integer :: reps, info
  real(dp), allocatable :: d0(:), e0(:), w(:)
  real(dp) :: t_dstebz, t_dstegr
  real(dp) :: chk_dstebz, chk_dstegr
  integer(int64) :: seed

  external dsterf, dstebz, dstegr, dlasrt

  reps = 1
  allocate(d0(n), e0(n-1), w(n))
  seed = make_seed(1, n)
  call fill_uniform(n, seed, d0, e0)

  write(*,'(a,i0)') 'omp_max_threads=', omp_get_max_threads()
  print '(a)', 'method,n,reps,time_ms,info,m,checksum'
  call flush(6)

  call time_dstegr(n, d0, e0, reps, t_dstegr, w, info)
  chk_dstegr = checksum(n, w)
  write(*,'(a,",",i0,",",i0,",",f12.4,",",i0,",",i0,",",es24.16)') &
       'DSTEGR', n, reps, 1000.0d0*t_dstegr, info, n, chk_dstegr
  call flush(6)

  call time_dstebz(n, d0, e0, reps, t_dstebz, w, info)
  chk_dstebz = checksum(n, w)
  write(*,'(a,",",i0,",",i0,",",f12.4,",",i0,",",i0,",",es24.16)') &
       'DSTEBZ', n, reps, 1000.0d0*t_dstebz, info, n, chk_dstebz
  call flush(6)

  deallocate(d0, e0, w)

contains

  subroutine time_dsterf(n, d0, e0, reps, best, wout, info)
    integer, intent(in) :: n, reps
    integer, intent(out) :: info
    real(dp), intent(in) :: d0(n), e0(n-1)
    real(dp), intent(out) :: best, wout(n)
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
       call dsterf(n, d, e, info)
       elapsed = omp_get_wtime() - t0
       if (info .ne. 0) exit
       if (elapsed .lt. best) then
          best = elapsed
          wout = d
       end if
    end do
    deallocate(d, e)
  end subroutine time_dsterf

  subroutine time_dstebz(n, d0, e0, reps, best, wout, info)
    integer, intent(in) :: n, reps
    integer, intent(out) :: info
    real(dp), intent(in) :: d0(n), e0(n-1)
    real(dp), intent(out) :: best, wout(n)
    integer :: r, m, nsplit, sort_info
    integer, allocatable :: iblock(:), isplit(:), iwork(:)
    real(dp), allocatable :: d(:), e(:), work(:), wtmp(:)
    real(dp) :: elapsed, t0, vl, vu, abstol
    real(dp), external :: dlamch

    allocate(d(n), e(n-1), wtmp(n), iblock(n), isplit(n), &
         work(4*n), iwork(3*n))
    best = huge(1.0d0)
    info = 0
    vl = 0.0d0
    vu = 0.0d0
    abstol = 0.0d0
    do r = 1, reps
       d = d0
       e = e0
       t0 = omp_get_wtime()
       call dstebz('A', 'E', n, vl, vu, 0, 0, abstol, d, e, m, nsplit, &
            wtmp, iblock, isplit, work, iwork, info)
       elapsed = omp_get_wtime() - t0
       if (info .ne. 0) exit
       if (m .ne. n) then
          info = -999
          exit
       end if
       call dlasrt('I', n, wtmp, sort_info)
       if (sort_info .ne. 0) then
          info = sort_info
          exit
       end if
       if (elapsed .lt. best) then
          best = elapsed
          wout = wtmp
       end if
    end do
    deallocate(d, e, wtmp, iblock, isplit, work, iwork)
  end subroutine time_dstebz

  subroutine time_dstegr(n, d0, e0, reps, best, wout, info)
    integer, intent(in) :: n, reps
    integer, intent(out) :: info
    real(dp), intent(in) :: d0(n), e0(n-1)
    real(dp), intent(out) :: best, wout(n)
    integer :: r, m, ldz, lwork, liwork, sort_info
    integer, allocatable :: isuppz(:), iwork(:)
    real(dp), allocatable :: d(:), e(:), z(:,:), work(:), wtmp(:)
    real(dp) :: elapsed, t0, vl, vu, abstol
    real(dp), external :: dlamch

    ldz = 1
    lwork = max(1, 20*n)
    liwork = max(1, 10*n)
    allocate(d(n), e(n-1), wtmp(n), z(1,1), isuppz(2*n), &
         work(lwork), iwork(liwork))
    best = huge(1.0d0)
    info = 0
    vl = 0.0d0
    vu = 0.0d0
    abstol = 0.0d0
    do r = 1, reps
       d = d0
       e = e0
       t0 = omp_get_wtime()
       call dstegr('N', 'A', n, d, e, vl, vu, 0, 0, abstol, m, wtmp, &
            z, ldz, isuppz, work, lwork, iwork, liwork, info)
       elapsed = omp_get_wtime() - t0
       if (info .ne. 0) exit
       if (m .ne. n) then
          info = -999
          exit
       end if
       call dlasrt('I', n, wtmp, sort_info)
       if (sort_info .ne. 0) then
          info = sort_info
          exit
       end if
       if (elapsed .lt. best) then
          best = elapsed
          wout = wtmp
       end if
    end do
    deallocate(d, e, wtmp, z, isuppz, work, iwork)
  end subroutine time_dstegr

  subroutine fill_uniform(n, seed, d, e)
    integer, intent(in) :: n
    integer(int64), intent(inout) :: seed
    real(dp), intent(out) :: d(n), e(n-1)
    integer :: i

    do i = 1, n
       d(i) = 2.0d0 * randu(seed) - 1.0d0
    end do
    do i = 1, n-1
       e(i) = 0.10d0 + 0.20d0 * randu(seed)
    end do
  end subroutine fill_uniform

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
    randu = real(iand(x * 2685821657736338717_int64, huge(1_int64)), dp) &
            / real(huge(1_int64), dp)
    randu = min(max(randu, tiny(1.0d0)), 1.0d0 - epsilon(1.0d0))
  end function randu

  real(dp) function checksum(n, w)
    integer, intent(in) :: n
    real(dp), intent(in) :: w(n)
    integer :: i

    checksum = 0.0d0
    do i = 1, n
       checksum = checksum + w(i) * dble(mod(i, 127) + 1)
    end do
  end function checksum

end program bench_mrrr_bisection
