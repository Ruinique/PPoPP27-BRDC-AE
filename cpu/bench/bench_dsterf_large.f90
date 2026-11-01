program dsterf_large_bench
  use omp_lib, only: omp_get_wtime
  implicit none

  integer, parameter :: dp = kind(1.0d0)
  integer, parameter :: nmodes = 3, nn = 11
  integer :: modes(nmodes), ns(nn)
  integer :: imode, inum, mode, n, info
  real(dp) :: elapsed, t0, checksum
  real(dp), allocatable :: d0(:), e0(:), d(:), e(:)
  character(len=20) :: mname

  external dsterf

  modes = (/ 1, 2, 3 /)
  ns = (/ 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, &
          131072, 262144, 524288 /)

  print '(a)', 'mode,n,reps,dsterf_ms,qr_storage_mb,checksum'

  do imode = 1, nmodes
     mode = modes(imode)
     do inum = 1, nn
        n = ns(inum)
        allocate(d0(n), e0(n-1), d(n), e(n-1))
        call fill_case(mode, n, d0, e0)

        d = d0
        e = e0
        t0 = omp_get_wtime()
        call dsterf(n, d, e, info)
        elapsed = omp_get_wtime() - t0
        if (info .ne. 0) then
           print *, 'DSTERF failed mode=', mode, ' n=', n, ' info=', info
           stop 1
        end if
        checksum = d(1) + d(n) + sum(d(1:n:max(1,n/16)))

        mname = mode_name(mode)
        write(*,'(a,",",i0,",",i0,",",f14.4,",",f12.2,",",es16.8)') &
             trim(mname), n, 1, 1000.0d0*elapsed, &
             (8.0d0*n + 8.0d0*(n-1)) / (1024.0d0*1024.0d0), checksum

        deallocate(d0, e0, d, e)
     end do
  end do

contains

  subroutine fill_case(mode, n, d, e)
    integer, intent(in) :: mode, n
    real(dp), intent(out) :: d(n), e(n-1)
    integer :: i

    do i = 1, n
       select case (mode)
       case (1)
          d(i) = 2.0d0 + 0.001d0*i + sin(0.37d0*i)
       case (2)
          d(i) = 2.0d0
       case (3)
          d(i) = 1.0d0 + 1.0d-12*(i - 0.5d0*(n+1))
       end select
    end do

    do i = 1, n-1
       select case (mode)
       case (1)
          e(i) = 0.25d0 + 0.03d0*cos(0.21d0*i)
       case (2)
          e(i) = 0.25d0
       case (3)
          e(i) = 1.0d-4*(1.0d0 + 0.1d0*cos(0.33d0*i))
       end select
    end do
  end subroutine fill_case

  character(len=20) function mode_name(mode)
    integer, intent(in) :: mode
    select case (mode)
    case (1)
       mode_name = 'random'
    case (2)
       mode_name = 'toeplitz'
    case (3)
       mode_name = 'clustered'
    end select
  end function mode_name

end program dsterf_large_bench
