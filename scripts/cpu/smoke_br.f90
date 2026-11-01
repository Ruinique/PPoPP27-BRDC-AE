! AE getting-started smoke test (written for the artifact, not from the paper):
! random symmetric tridiagonal, eigenvalues via DSTEDC_DC_BR (BR) vs DSTERF (QR/QL).
program smoke_br
  use omp_lib, only: omp_get_wtime, omp_get_max_threads
  implicit none
  integer, parameter :: dp = kind(1.0d0)
  integer :: n, info, lw, liw, i, nargs
  character(len=32) :: arg
  real(dp), allocatable :: d0(:), e0(:), d(:), e(:), w(:), work(:)
  integer, allocatable :: iwork(:)
  real(dp) :: wq(1), t0, tbr, tst, err, scale
  integer :: iwq(1)
  external dstedc_dc_br, dsterf
  n = 4096
  nargs = command_argument_count()
  if (nargs >= 1) then
     call get_command_argument(1, arg); read(arg,*) n
  end if
  allocate(d0(n), e0(n-1), d(n), e(n-1), w(n))
  call random_seed()
  call random_number(d0); call random_number(e0)
  d0 = 2.0_dp*d0 - 1.0_dp; e0 = 2.0_dp*e0 - 1.0_dp
  d = d0; e = e0
  call dstedc_dc_br(n, d, e, wq, -1, iwq, -1, info)
  lw = max(1, int(wq(1))); liw = max(1, iwq(1))
  allocate(work(lw), iwork(liw))
  t0 = omp_get_wtime(); call dstedc_dc_br(n, d, e, work, lw, iwork, liw, info); tbr = omp_get_wtime()-t0
  if (info /= 0) then; print *, 'BR info=', info; stop 1; end if
  w = d0; e = e0
  t0 = omp_get_wtime(); call dsterf(n, w, e, info); tst = omp_get_wtime()-t0
  if (info /= 0) then; print *, 'DSTERF info=', info; stop 1; end if
  scale = max(maxval(abs(w)), tiny(1.0_dp))
  err = maxval(abs(d - w)) / scale
  print '(a,i0,a,i0)', 'n=', n, ' threads=', omp_get_max_threads()
  print '(a,i0,a,i0)', 'br_lwork=', lw, ' br_liwork=', liw
  print '(a,f10.3,a,f10.3)', 'br_ms=', 1d3*tbr, ' dsterf_ms=', 1d3*tst
  print '(a,es10.3)', 'max_rel_diff_vs_dsterf=', err
  if (err < 1.0d-10) then; print *, 'PASS'; else; print *, 'FAIL'; stop 2; end if
end program
