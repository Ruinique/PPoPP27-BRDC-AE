program bench_elpa_tridiagonal
  use iso_c_binding, only: c_int
  use iso_fortran_env, only: int64
  use elpa
  implicit none
  include 'mpif.h'

  integer, parameter :: dp = kind(1.0d0)
  integer :: mpierr, rank, nranks
  integer :: n, nblk, np_rows, np_cols, my_prow, my_pcol
  integer :: local_nrows, local_ncols, err, reps, rep
  integer :: argc
  character(len=32) :: arg, mode
  class(elpa_t), pointer :: solver
  real(dp), allocatable :: d0(:), e0(:), d(:), sd(:), q(:,:)
  real(dp) :: t0, elapsed, elapsed_max, best_ms

  call MPI_Init(mpierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, mpierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nranks, mpierr)

  argc = command_argument_count()
  if (argc < 2) then
    if (rank == 0) print *, 'usage: bench_elpa_tridiagonal N MODE [REPS]'
    call MPI_Abort(MPI_COMM_WORLD, 2, mpierr)
  end if

  call get_command_argument(1, arg)
  read(arg, *) n
  call get_command_argument(2, mode)
  reps = 1
  if (argc >= 3) then
    call get_command_argument(3, arg)
    read(arg, *) reps
  end if

  nblk = 64
  call factor_grid(nranks, np_rows, np_cols)
  my_prow = rank / np_cols
  my_pcol = mod(rank, np_cols)
  local_nrows = local_extent(n, nblk, my_prow, np_rows)
  local_ncols = local_extent(n, nblk, my_pcol, np_cols)

  allocate(d0(n), e0(n), d(n), sd(n))
  allocate(q(max(1, local_nrows), max(1, local_ncols)))
  call fill_case(trim(mode), n, d0, e0)

  if (elpa_init(20171201) /= ELPA_OK) then
    if (rank == 0) print *, 'ELPA API version not supported'
    call MPI_Abort(MPI_COMM_WORLD, 3, mpierr)
  end if

  solver => elpa_allocate(err)
  call check_elpa(err, 'elpa_allocate')
  call solver%set('na', int(n, c_int), err)
  call check_elpa(err, 'set na')
  call solver%set('nev', int(n, c_int), err)
  call check_elpa(err, 'set nev')
  call solver%set('local_nrows', int(local_nrows, c_int), err)
  call check_elpa(err, 'set local_nrows')
  call solver%set('local_ncols', int(local_ncols, c_int), err)
  call check_elpa(err, 'set local_ncols')
  call solver%set('nblk', int(nblk, c_int), err)
  call check_elpa(err, 'set nblk')
  call solver%set('mpi_comm_parent', int(MPI_COMM_WORLD, c_int), err)
  call check_elpa(err, 'set mpi_comm_parent')
  call solver%set('process_row', int(my_prow, c_int), err)
  call check_elpa(err, 'set process_row')
  call solver%set('process_col', int(my_pcol, c_int), err)
  call check_elpa(err, 'set process_col')
  err = solver%setup()
  call check_elpa(err, 'setup')

  best_ms = huge(1.0d0)
  do rep = 1, reps
    d = d0
    sd = e0
    q = 0.0d0
    call MPI_Barrier(MPI_COMM_WORLD, mpierr)
    t0 = MPI_Wtime()
    call solver%solve_tridiagonal(d, sd, q, err)
    elapsed = MPI_Wtime() - t0
    call check_elpa(err, 'solve_tridiagonal')
    call MPI_Reduce(elapsed, elapsed_max, 1, MPI_DOUBLE_PRECISION, MPI_MAX, 0, MPI_COMM_WORLD, mpierr)
    if (rank == 0) best_ms = min(best_ms, 1000.0d0 * elapsed_max)
  end do

  if (rank == 0) then
    print '(a)', 'method,mode,n,ranks,nprow,npcol,nblk,reps,time_ms,checksum'
    write(*,'(a,",",a,",",i0,",",i0,",",i0,",",i0,",",i0,",",i0,",",f12.4,",",es24.16)') &
         'ELPA_SOLVE_TRIDIAGONAL', trim(mode), n, nranks, np_rows, np_cols, &
         nblk, reps, best_ms, checksum(n, d)
    call flush(6)
  end if

  call elpa_deallocate(solver, err)
  call elpa_uninit()
  deallocate(d0, e0, d, sd, q)
  call MPI_Finalize(mpierr)

contains

  subroutine check_elpa(code, label)
    integer, intent(in) :: code
    character(len=*), intent(in) :: label
    if (code /= ELPA_OK) then
      if (rank == 0) print '(a,": ",i0)', trim(label), code
      call MPI_Abort(MPI_COMM_WORLD, code, mpierr)
    end if
  end subroutine check_elpa

  subroutine factor_grid(p, rows, cols)
    integer, intent(in) :: p
    integer, intent(out) :: rows, cols
    integer :: c

    cols = 1
    do c = int(sqrt(real(p, dp))), 1, -1
      if (mod(p, c) == 0) then
        cols = c
        exit
      end if
    end do
    rows = p / cols
  end subroutine factor_grid

  integer function local_extent(n, nblk, iproc, nprocs)
    integer, intent(in) :: n, nblk, iproc, nprocs
    integer :: first, block, owner, block_len

    local_extent = 0
    block = 0
    do first = 1, n, nblk
      owner = mod(block, nprocs)
      block_len = min(nblk, n - first + 1)
      if (owner == iproc) local_extent = local_extent + block_len
      block = block + 1
    end do
  end function local_extent

  subroutine fill_case(mode, n, d, e)
    character(len=*), intent(in) :: mode
    integer, intent(in) :: n
    real(dp), intent(out) :: d(n), e(n)
    integer :: i
    integer(int64) :: seed
    real(dp) :: center

    if (trim(mode) == 'clustered') then
      center = 0.5d0 * (dble(n) + 1.0d0)
      do i = 1, n
        d(i) = 1.0d0 + 1.0d-12 * (dble(i) - center)
      end do
      do i = 1, n - 1
        e(i) = 1.0d-4 * (1.0d0 + 0.1d0 * cos(0.33d0 * dble(i)))
      end do
    else
      seed = make_seed(n)
      do i = 1, n
        d(i) = 2.0d0 * randu(seed) - 1.0d0
      end do
      do i = 1, n - 1
        e(i) = 0.10d0 + 0.20d0 * randu(seed)
      end do
    end if
    e(n) = 0.0d0
  end subroutine fill_case

  integer(int64) function make_seed(n)
    integer, intent(in) :: n
    make_seed = 7640891576956012809_int64
    make_seed = ieor(make_seed, 1000003_int64)
    make_seed = ieor(make_seed, int(n, int64) * 1099511628211_int64)
  end function make_seed

  real(dp) function randu(seed)
    integer(int64), intent(inout) :: seed
    integer(int64) :: x

    x = seed
    x = ieor(x, shiftr(x, 12))
    x = ieor(x, shiftl(x, 25))
    x = ieor(x, shiftr(x, 27))
    seed = x
    randu = real(iand(x * 2685821657736338717_int64, huge(1_int64)), dp) / &
         real(huge(1_int64), dp)
  end function randu

  real(dp) function checksum(n, values)
    integer, intent(in) :: n
    real(dp), intent(in) :: values(n)
    integer :: i

    checksum = 0.0d0
    do i = 1, n
      checksum = checksum + values(i) * dble(mod(i, 127) + 1)
    end do
  end function checksum

end program bench_elpa_tridiagonal
