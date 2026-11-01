program bench_deflation_recheck
  use omp_lib, only: omp_get_max_threads
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none

  integer, parameter :: dp = kind(1.0d0)
  integer, parameter :: nmodes = 4, nn = 3
  integer :: modes(nmodes), ns(nn)
  integer :: imode, inum, mode, n, info, lw, liw
  integer, allocatable :: iwork(:)
  real(dp), allocatable :: d0(:), e0(:), d(:), e(:), work(:)
  character(len=24) :: mname
  integer(int64) :: seed
  real(dp) :: merges, parent_merges, root_merges, size_sum, active_sum
  real(dp) :: parent_size_sum, parent_active_sum, full_defl, max_k
  real(dp) :: root_k

  external dstedc_dc_br

  modes = (/ 1, 2, 3, 4 /)
  ns = (/ 1024, 16384, 65536 /)

  write(*,'(a,i0)') 'omp_max_threads=', omp_get_max_threads()
  print '(a)', 'mode,n,info,merges,parent_merges,root_merges,' // &
       'size_sum,active_sum,avg_active_ratio,avg_deflation_ratio,' // &
       'parent_active_ratio,full_defl_merges,max_k,root_k,checksum'

  do imode = 1, nmodes
     mode = modes(imode)
     mname = mode_name(mode)
     do inum = 1, nn
        n = ns(inum)
        seed = make_seed(mode, n)

        allocate(d0(n), e0(n-1), d(n), e(n-1), work(1), iwork(1))
        call fill_case(mode, n, seed, d0, e0)

        d = d0
        e = e0
        call dstedc_dc_br(n, d, e, work, -1, iwork, -1, info)
        if (info .ne. 0) stop 1
        lw = max(max(1, int(work(1))), 32*n)
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

        write(*,'(a,",",i0,",",i0,",",f0.0,",",f0.0,",",f0.0,",",f0.0,",",f0.0,",",f10.6,",",f10.6,",",f10.6,",",f0.0,",",f0.0,",",f0.0,",",es16.8)') &
             trim(mname), n, info, merges, parent_merges, root_merges, &
             size_sum, active_sum, ratio(active_sum, size_sum), &
             ratio(size_sum-active_sum, size_sum), &
             ratio(parent_active_sum, parent_size_sum), &
             full_defl, max_k, root_k, checksum(n, d)

        deallocate(d0, e0, d, e, work, iwork)
     end do
  end do

contains

  real(dp) function ratio(a, b)
    real(dp), intent(in) :: a, b
    if (b .eq. 0.0d0) then
       ratio = -1.0d0
    else
       ratio = a / b
    end if
  end function ratio

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

end program bench_deflation_recheck

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
