program cpu_strong_scaling
  use omp_lib, only: omp_get_max_threads, omp_get_wtime
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none

  integer, parameter :: dp = kind(1.0d0)
  integer, parameter :: nmodes = 4, nn = 2
  integer :: modes(nmodes), ns(nn)
  integer :: imode, inum, mode, n, reps, r, info, lw, liw
  integer, allocatable :: iwork(:)
  real(dp) :: best_br, elapsed, t0
  real(dp) :: checksum_br
  real(dp), allocatable :: d0(:), e0(:), d(:), e(:), work(:)
  character(len=24) :: mname
  integer(int64) :: seed

  external dstedc_dc_br

  modes = (/ 1, 2, 3, 4 /)
  ns = (/ 16384, 65536 /)

  write(*,'(a,i0)') 'omp_max_threads=', omp_get_max_threads()
  print '(a)', 'mode,n,threads,reps,br_ms,checksum_br'

  do imode = 1, nmodes
     mode = modes(imode)
     mname = mode_name(mode)
     do inum = 1, nn
        n = ns(inum)
        reps = choose_reps(n)
        seed = make_seed(mode, n)

        allocate(d0(n), e0(n-1), d(n), e(n-1), work(1), iwork(1))
        call fill_case(mode, n, seed, d0, e0)

        d = d0
        e = e0
        call dstedc_dc_br(n, d, e, work, -1, iwork, -1, info)
        if (info .ne. 0) then
           print *, 'BR query failed mode=', trim(mname), ' n=', n, ' info=', info
           stop 1
        end if
        lw = max(1, int(work(1)))
        liw = max(1, iwork(1))
        deallocate(work, iwork)
        allocate(work(lw), iwork(liw))

        best_br = huge(1.0d0)
        checksum_br = 0.0d0
        do r = 1, reps
           d = d0
           e = e0
           t0 = omp_get_wtime()
           call dstedc_dc_br(n, d, e, work, lw, iwork, liw, info)
           elapsed = omp_get_wtime() - t0
           if (info .ne. 0) then
              print *, 'BR failed mode=', trim(mname), ' n=', n, ' info=', info
              stop 1
           end if
           if (elapsed .lt. best_br) then
              best_br = elapsed
              checksum_br = checksum(n, d)
           end if
        end do

        write(*,'(a,",",i0,",",i0,",",i0,",",f12.4,",",es16.8)') &
             trim(mname), n, omp_get_max_threads(), reps, &
             1000.0d0*best_br, checksum_br

        deallocate(d0, e0, d, e, work, iwork)
     end do
  end do

contains

  integer function choose_reps(n)
    integer, intent(in) :: n
    if (n .le. 16384) then
       choose_reps = 2
    else
       choose_reps = 1
    end if
  end function choose_reps

  subroutine fill_case(mode, n, seed, d, e)
    integer, intent(in) :: mode, n
    integer(int64), intent(in) :: seed
    real(dp), intent(out) :: d(n), e(n-1)
    integer :: i
    integer(int64) :: rng

    rng = seed
    select case (mode)
    case (1)
       do i = 1, n
          d(i) = 2.0d0 * randu(rng) - 1.0d0
       end do
       do i = 1, n-1
          e(i) = 0.10d0 + 0.20d0 * randu(rng)
       end do
    case (2)
       do i = 1, n
          d(i) = randn(rng)
       end do
       do i = 1, n-1
          e(i) = 0.10d0 + 0.20d0 * randu(rng)
       end do
    case (3)
       d = 2.0d0
       e = 0.25d0
    case (4)
       do i = 1, n
          d(i) = 1.0d0 + 1.0d-12*(i - 0.5d0*(n+1))
       end do
       do i = 1, n-1
          e(i) = 1.0d-4*(1.0d0 + 0.1d0*cos(0.33d0*i))
       end do
    end select
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
    randu = real(iand(x * 2685821657736338717_int64, huge(1_int64)), dp) &
            / real(huge(1_int64), dp)
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
    case (3)
       mode_name = 'toeplitz'
    case (4)
       mode_name = 'clustered'
    end select
  end function mode_name

end program cpu_strong_scaling