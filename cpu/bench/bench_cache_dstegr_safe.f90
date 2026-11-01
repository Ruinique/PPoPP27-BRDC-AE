program bench_cache_dstegr_safe
  use omp_lib, only: omp_get_wtime
  implicit none
  integer, parameter :: dp = kind(1.0d0)
  character(len=256) :: cache_dir
  character(len=32) :: only_mode
  character(len=16), dimension(4) :: modes
  integer :: stat, i, n, threads
  external dstegr, dlasrt, mkl_set_num_threads

  modes = (/ 'uniform         ', 'normal          ', 'toeplitz        ', 'clustered       ' /)
  call get_environment_variable('CACHE_DIR', cache_dir, status=stat)
  if (stat .ne. 0) stop 2
  call get_environment_variable('CACHE_MODE', only_mode, status=stat)
  n = get_env_int('CACHE_N', 65536)
  threads = get_env_int('MKL_NUM_THREADS', 56)
  call mkl_set_num_threads(threads)

  print '(a)', 'method,mode,n,threads,time_ms,info,m,checksum,generator'
  call flush(6)
  if (stat .eq. 0 .and. len_trim(only_mode) .gt. 0) then
     call run_case(trim(only_mode), n, threads)
  else
     do i = 1, size(modes)
        call run_case(trim(modes(i)), n, threads)
     end do
  end if

contains
  integer function get_env_int(name, default_value)
    character(len=*), intent(in) :: name
    integer, intent(in) :: default_value
    character(len=64) :: buf
    integer :: s, value
    get_env_int = default_value
    call get_environment_variable(name, buf, status=s)
    if (s .ne. 0 .or. len_trim(buf) .eq. 0) return
    read(buf, *, iostat=s) value
    if (s .eq. 0) get_env_int = value
  end function get_env_int

  function itoa(value) result(str)
    integer, intent(in) :: value
    character(len=32) :: str
    write(str, '(i0)') value
  end function itoa

  subroutine load_case(mode, n, d, e)
    character(len=*), intent(in) :: mode
    integer, intent(in) :: n
    real(dp), intent(out) :: d(n), e(n-1)
    character(len=512) :: path
    integer(kind=8) :: file_n
    integer :: unit
    path = trim(cache_dir)//'/'//trim(mode)//'_'//trim(adjustl(itoa(n)))//'.de.bin'
    open(newunit=unit, file=trim(path), access='stream', form='unformatted', &
         status='old', action='read')
    read(unit) file_n
    if (file_n .ne. n) stop 3
    read(unit) d
    read(unit) e
    close(unit)
  end subroutine load_case

  subroutine run_case(mode, n, threads)
    character(len=*), intent(in) :: mode
    integer, intent(in) :: n, threads
    integer :: info, m, ldz, lwork, liwork, sort_info
    integer, allocatable :: isuppz(:), iwork(:)
    real(dp), allocatable :: d(:), e(:), w(:), z(:,:), work(:)
    real(dp) :: t0, t_ms, vl, vu, abstol

    ! DSTEGR requires E to have dimension N even though a tridiagonal cache
    ! stores only N-1 off-diagonal entries.  The final element is workspace.
    ldz = 1
    lwork = max(1, 20*n)
    liwork = max(1, 10*n)
    allocate(d(n), e(n), w(n), z(1,1), isuppz(2*n), work(lwork), iwork(liwork))
    call load_case(mode, n, d, e)
    e(n) = 0.0d0
    vl = 0.0d0
    vu = 0.0d0
    abstol = 0.0d0
    t0 = omp_get_wtime()
    call dstegr('N', 'A', n, d, e, vl, vu, 0, 0, abstol, m, w, &
         z, ldz, isuppz, work, lwork, iwork, liwork, info)
    t_ms = 1000.0d0 * (omp_get_wtime() - t0)
    if (info .eq. 0 .and. m .eq. n) call dlasrt('I', n, w, sort_info)
    write(*,'(a,",",a,",",i0,",",i0,",",f12.4,",",i0,",",i0,",",es24.16,",",a)') &
         'DSTEGR', trim(mode), n, threads, t_ms, info, m, checksum(n, w), generator_name(mode)
    call flush(6)
    deallocate(d, e, w, z, isuppz, work, iwork)
  end subroutine run_case

  character(len=32) function generator_name(mode)
    character(len=*), intent(in) :: mode
    if (trim(mode) .eq. 'toeplitz') then
       generator_name = 'structured_toeplitz'
    else
       generator_name = 'dlatms_symmetric_band'
    end if
  end function generator_name

  real(dp) function checksum(n, values)
    integer, intent(in) :: n
    real(dp), intent(in) :: values(n)
    integer :: i
    checksum = 0.0d0
    do i = 1, n
       checksum = checksum + values(i) * dble(mod(i, 127) + 1)
    end do
  end function checksum
end program bench_cache_dstegr_safe
