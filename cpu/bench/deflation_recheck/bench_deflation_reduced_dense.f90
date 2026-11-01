program bench_deflation_reduced_dense
  use omp_lib, only: omp_get_max_threads
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none

  integer, parameter :: dp = kind(1.0d0)
  integer :: n, info, lw, liw, unit_no
  integer(int64) :: n64
  integer, allocatable :: iwork(:)
  real(dp), allocatable :: d0(:), e0(:), d(:), e(:), work(:)
  real(dp) :: merges, parent_merges, root_merges, size_sum, active_sum
  real(dp) :: parent_size_sum, parent_active_sum, full_defl, max_k
  real(dp) :: root_k
  character(len=4096) :: path

  external dstedc_dc_br

  call get_command_argument(1, path)
  if (len_trim(path) .eq. 0) then
     print *, 'usage: bench_deflation_reduced_dense <compact_de_file>'
     stop 2
  end if

  unit_no = 10
  open(unit=unit_no, file=trim(path), access='stream', form='unformatted', &
       status='old', action='read')
  read(unit_no) n64
  if (n64 .le. 1_int64 .or. n64 .gt. int(huge(n), int64)) then
     print *, 'invalid n=', n64
     stop 2
  end if
  n = int(n64)

  allocate(d0(n), e0(n-1), d(n), e(n-1), work(1), iwork(1))
  read(unit_no) d0
  read(unit_no) e0
  close(unit_no)

  d = d0
  e = e0
  call dstedc_dc_br(n, d, e, work, -1, iwork, -1, info)
  if (info .ne. 0) then
     print *, 'BR query failed n=', n, ' info=', info
     stop 1
  end if
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

  write(*,'(a,i0)') 'omp_max_threads=', omp_get_max_threads()
  print '(a)', 'mode,n,info,merges,parent_merges,root_merges,' // &
       'size_sum,active_sum,avg_active_ratio,avg_deflation_ratio,' // &
       'parent_active_ratio,full_defl_merges,max_k,root_k,checksum'
  write(*,'(a,",",i0,",",i0,",",f0.0,",",f0.0,",",f0.0,",",f0.0,",",f0.0,",",f10.6,",",f10.6,",",f10.6,",",f0.0,",",f0.0,",",f0.0,",",es16.8)') &
       'reduced_dense', n, info, merges, parent_merges, root_merges, &
       size_sum, active_sum, ratio(active_sum, size_sum), &
       ratio(size_sum-active_sum, size_sum), &
       ratio(parent_active_sum, parent_size_sum), &
       full_defl, max_k, root_k, checksum(n, d)

  deallocate(d0, e0, d, e, work, iwork)

contains

  real(dp) function ratio(a, b)
    real(dp), intent(in) :: a, b
    if (b .eq. 0.0d0) then
       ratio = -1.0d0
    else
       ratio = a / b
    end if
  end function ratio

  real(dp) function checksum(n, d)
    integer, intent(in) :: n
    real(dp), intent(in) :: d(n)
    checksum = d(1) + d(n) + sum(d(1:n:max(1,n/16)))
  end function checksum

end program bench_deflation_reduced_dense

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
