#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C
export LANG=C

ROOT="${AE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BENCH_DIR="$ROOT/cpu/bench"
CACHE_DIR="${SPECTRUM_DE_DIR:-$ROOT/results/data}"
JOB_TAG="${JH_JOBID:-manual}"
BUILD_DIR="$ROOT/results/cpu/build_mpfr_main_families_${JOB_TAG}"
RESULT_DIR="$ROOT/results/cpu/results_mpfr_main_families_${JOB_TAG}"
FSRC="$BUILD_DIR/solve_values.f90"
CSRC="$BUILD_DIR/mpfr_ref.c"

mkdir -p "$BUILD_DIR/br_objs" "$RESULT_DIR"
cd "$ROOT"

echo "=== MPFR main-family accuracy probe ==="
date
hostname
pwd
if command -v module >/dev/null 2>&1; then module use /public/software/modules/intel || true || true; fi
if command -v module >/dev/null 2>&1; then module use /public/software/modules/compiler || true || true; fi
if command -v module >/dev/null 2>&1; then module load tbb/2022.3 compiler-rt/2025.3.0 mkl/2025.3 2>&1 || true || true; fi
gfortran --version | head -1
cc --version | head -1
ls /usr/include/mpfr.h /usr/lib64/libmpfr.so >/dev/null

echo "=== compile BR objects (LP64) ==="
rm -f "$BUILD_DIR"/br_objs/*.o
cc -O2 -DOS_LINUX -DSMP -I"$ROOT" -c cpu/src/openblas_get_num_nodes.c \
  -o "$BUILD_DIR/br_objs/openblas_get_num_nodes.o"
for src in \
  cpu/src/dlaed0_br.f \
  cpu/src/dlaed4_br.f \
  cpu/src/dlaed7_br.f \
  cpu/src/dlaed9_br.f \
  cpu/src/dstedc_dc_br.f
do
  obj="$BUILD_DIR/br_objs/$(basename "$src" .f).o"
  gfortran -O2 -fopenmp -std=legacy -c "$src" -o "$obj"
done

cat > "$FSRC" <<'F90'
program solve_values
  use omp_lib, only: omp_get_wtime, omp_set_num_threads
  implicit none
  integer, parameter :: dp = kind(1.0d0)
  integer, parameter :: nm = 4, nn = 2, max_probe = 8
  character(len=16), dimension(nm) :: modes
  integer, dimension(nn) :: sizes
  integer, dimension(max_probe, nm, nn) :: probes
  integer, dimension(nm, nn) :: nprobe
  integer :: im, inum, n, p, k
  integer :: info_dsterf, info_br, info_hid, lw_br, liw_br, lw_hid, liw_hid
  integer, allocatable :: iw_br(:), iw_hid(:)
  real(dp), allocatable :: d0(:), e0(:), dsterf_w(:), br_w(:), hidden_w(:)
  real(dp), allocatable :: work_br(:), work_hid(:), q(:,:), qstore(:,:)
  real(dp) :: t_dsterf_ms, t_br_ms, t_hid_ms
  character(len=512) :: path

  external dsterf, dlasrt, dlaed0, dstedc_dc_br, mkl_set_num_threads

  modes = (/ 'uniform         ', 'normal          ', 'toeplitz        ', 'clustered       ' /)
  sizes = (/ 16384, 49152 /)
  probes = 0
  nprobe = 0

  ! Indices from the existing BR-vs-DSTERF/internal accuracy sweep, plus neighbors.
  call add3(probes, nprobe, 1, 1, 15273)
  call add3(probes, nprobe, 1, 1, 16244)
  call add3(probes, nprobe, 1, 2, 46880)
  call add3(probes, nprobe, 2, 1, 15838)
  call add3(probes, nprobe, 2, 2, 4456)
  call add3(probes, nprobe, 3, 1, 5545)
  call add3(probes, nprobe, 3, 2, 16557)
  call add3(probes, nprobe, 4, 1, 16384)
  call add3(probes, nprobe, 4, 2, 49152)

  call mkl_set_num_threads(56)
  call omp_set_num_threads(56)

  print '(a)', 'mode,n,index,dsterf,internal,br,info_dsterf,info_internal,info_br,dsterf_ms,internal_ms,br_ms'
  call flush(6)

  do im = 1, nm
     do inum = 1, nn
        n = sizes(inum)
        allocate(d0(n), e0(n-1), dsterf_w(n), br_w(n), hidden_w(n))
        call load_case(trim(modes(im)), n, d0, e0, path)

        call solve_dsterf(n, d0, e0, dsterf_w, t_dsterf_ms, info_dsterf)

        call br_workspace(n, d0, e0, lw_br, liw_br, info_br)
        allocate(work_br(lw_br), iw_br(liw_br))
        call solve_br(n, d0, e0, work_br, lw_br, iw_br, liw_br, br_w, t_br_ms, info_br)
        deallocate(work_br, iw_br)

        call hidden_workspace(n, lw_hid, liw_hid)
        allocate(work_hid(lw_hid), iw_hid(liw_hid), q(n,n), qstore(n,n))
        call solve_hidden(n, d0, e0, work_hid, lw_hid, iw_hid, liw_hid, q, n, qstore, n, hidden_w, t_hid_ms, info_hid)
        deallocate(work_hid, iw_hid, q, qstore)

        do p = 1, nprobe(im, inum)
           k = probes(p, im, inum)
           if (k .ge. 1 .and. k .le. n) then
              write(*,'(a,",",i0,",",i0,",",es25.17,",",es25.17,",",es25.17,",",i0,",",i0,",",i0,",",f12.4,",",f12.4,",",f12.4)') &
                   trim(modes(im)), n, k, dsterf_w(k), hidden_w(k), br_w(k), &
                   info_dsterf, info_hid, info_br, t_dsterf_ms, t_hid_ms, t_br_ms
           end if
        end do
        call flush(6)
        deallocate(d0, e0, dsterf_w, br_w, hidden_w)
     end do
  end do

contains
  subroutine add3(probes, nprobe, im, inum, center)
    integer, intent(inout) :: probes(max_probe, nm, nn), nprobe(nm, nn)
    integer, intent(in) :: im, inum, center
    call add_one(probes, nprobe, im, inum, center-1)
    call add_one(probes, nprobe, im, inum, center)
    call add_one(probes, nprobe, im, inum, center+1)
  end subroutine add3

  subroutine add_one(probes, nprobe, im, inum, k)
    integer, intent(inout) :: probes(max_probe, nm, nn), nprobe(nm, nn)
    integer, intent(in) :: im, inum, k
    integer :: j
    if (k .lt. 1) return
    do j = 1, nprobe(im, inum)
       if (probes(j, im, inum) .eq. k) return
    end do
    if (nprobe(im, inum) .ge. max_probe) stop 90
    nprobe(im, inum) = nprobe(im, inum) + 1
    probes(nprobe(im, inum), im, inum) = k
  end subroutine add_one

  function itoa(value) result(str)
    integer, intent(in) :: value
    character(len=32) :: str
    write(str, '(i0)') value
  end function itoa

  subroutine load_case(mode, n, d, e, path)
    character(len=*), intent(in) :: mode
    integer, intent(in) :: n
    real(dp), intent(out) :: d(n), e(n-1)
    character(len=*), intent(out) :: path
    integer(kind=8) :: file_n
    integer :: unit
    path = '$CACHE_DIR/'//trim(mode)//'_'//trim(adjustl(itoa(n)))//'.de.bin'
    open(newunit=unit, file=trim(path), access='stream', form='unformatted', status='old', action='read')
    read(unit) file_n
    if (file_n .ne. n) stop 3
    read(unit) d
    read(unit) e
    close(unit)
  end subroutine load_case

  subroutine solve_dsterf(n, d0, e0, out, t_ms, info)
    integer, intent(in) :: n
    real(dp), intent(in) :: d0(n), e0(n-1)
    real(dp), intent(out) :: out(n), t_ms
    integer, intent(out) :: info
    integer :: sort_info
    real(dp), allocatable :: d(:), e(:)
    real(dp) :: t0
    allocate(d(n), e(n-1))
    d = d0
    e = e0
    t0 = omp_get_wtime()
    call dsterf(n, d, e, info)
    t_ms = 1000.0d0 * (omp_get_wtime() - t0)
    if (info .eq. 0) then
       call dlasrt('I', n, d, sort_info)
       if (sort_info .ne. 0) info = sort_info
    end if
    out = d
    deallocate(d, e)
  end subroutine solve_dsterf

  subroutine br_workspace(n, d0, e0, lwork, liwork, info)
    integer, intent(in) :: n
    real(dp), intent(in) :: d0(n), e0(n-1)
    integer, intent(out) :: lwork, liwork, info
    real(dp), allocatable :: d(:), e(:), work(:)
    integer, allocatable :: iwork(:)
    allocate(d(n), e(n-1), work(1), iwork(1))
    d = d0
    e = e0
    call dstedc_dc_br(n, d, e, work, -1, iwork, -1, info)
    lwork = max(1, int(work(1)))
    liwork = max(1, iwork(1))
    deallocate(d, e, work, iwork)
  end subroutine br_workspace

  subroutine solve_br(n, d0, e0, work, lwork, iwork, liwork, out, t_ms, info)
    integer, intent(in) :: n, lwork, liwork
    real(dp), intent(in) :: d0(n), e0(n-1)
    real(dp), intent(inout) :: work(lwork)
    integer, intent(inout) :: iwork(liwork)
    real(dp), intent(out) :: out(n), t_ms
    integer, intent(out) :: info
    integer :: sort_info
    real(dp), allocatable :: d(:), e(:)
    real(dp) :: t0
    allocate(d(n), e(n-1))
    d = d0
    e = e0
    t0 = omp_get_wtime()
    call dstedc_dc_br(n, d, e, work, lwork, iwork, liwork, info)
    t_ms = 1000.0d0 * (omp_get_wtime() - t0)
    if (info .eq. 0) then
       call dlasrt('I', n, d, sort_info)
       if (sort_info .ne. 0) info = sort_info
    end if
    out = d
    deallocate(d, e)
  end subroutine solve_br

  subroutine solve_hidden(n, d0, e0, work, lwork, iwork, liwork, q, ldq, qstore, ldqs, out, t_ms, info)
    integer, intent(in) :: n, lwork, liwork, ldq, ldqs
    integer, intent(out) :: info
    integer, intent(inout) :: iwork(liwork)
    real(dp), intent(in) :: d0(n), e0(n-1)
    real(dp), intent(inout) :: work(lwork), q(ldq,*), qstore(ldqs,*)
    real(dp), intent(out) :: out(n), t_ms
    real(dp), allocatable :: d(:), e(:)
    real(dp) :: t0
    allocate(d(n), e(n-1))
    d = d0
    e = e0
    t0 = omp_get_wtime()
    call hidden_dc_values(n, d, e, work, lwork, iwork, liwork, q, ldq, qstore, ldqs, info)
    t_ms = 1000.0d0 * (omp_get_wtime() - t0)
    out = d
    deallocate(d, e)
  end subroutine solve_hidden

  subroutine hidden_dc_values(n, d, e, work, lwork, iwork, liwork, q, ldq, qstore, ldqs, info)
    integer, intent(in) :: n, lwork, liwork, ldq, ldqs
    integer, intent(out) :: info
    integer, intent(inout) :: iwork(liwork)
    real(dp), intent(inout) :: d(n), e(*), work(lwork)
    real(dp), intent(inout) :: q(ldq,*), qstore(ldqs,*)
    integer :: start, finish, m, smlsiz
    real(dp) :: orgnrm, eps, tiny
    integer, external :: ilaenv
    real(dp), external :: dlamch, dlanst
    external dlaed0, dlascl, dlasrt, dsterf
    info = 0
    if (n .le. 1) return
    smlsiz = ilaenv(9, 'DSTEDC', ' ', 0, 0, 0, 0)
    if (n .le. smlsiz) then
       call dsterf(n, d, e, info)
       return
    end if
    orgnrm = dlanst('M', n, d, e)
    if (orgnrm .le. 0.0d0) return
    eps = dlamch('Epsilon')
    start = 1
    do while (start .le. n)
       finish = start
       do while (finish .lt. n)
          tiny = eps*sqrt(abs(d(finish)))*sqrt(abs(d(finish+1)))
          if (abs(e(finish)) .le. tiny) exit
          finish = finish + 1
       end do
       m = finish - start + 1
       if (m .gt. 1) then
          if (m .gt. smlsiz) then
             orgnrm = dlanst('M', m, d(start), e(start))
             call dlascl('G', 0, 0, orgnrm, 1.0d0, m, 1, d(start), m, info)
             call dlascl('G', 0, 0, orgnrm, 1.0d0, m-1, 1, e(start), m-1, info)
             call dlaed0(0, n, m, d(start), e(start), q, ldq, qstore, ldqs, work, iwork, info)
             if (info .ne. 0) return
             call dlascl('G', 0, 0, 1.0d0, orgnrm, m, 1, d(start), m, info)
          else
             call dsterf(m, d(start), e(start), info)
             if (info .ne. 0) return
          end if
       end if
       start = finish + 1
    end do
    call dlasrt('I', n, d, info)
  end subroutine hidden_dc_values

  subroutine hidden_workspace(n, lwork, liwork)
    integer, intent(in) :: n
    integer, intent(out) :: lwork, liwork
    integer :: lgn, tstore, inner, smlsiz
    integer, external :: ilaenv
    smlsiz = ilaenv(9, 'DSTEDC', ' ', 0, 0, 0, 0)
    lgn = int(log(real(max(2,n), dp))/log(2.0d0)) + 2
    tstore = max(1, n * max(1, lgn))
    inner = 3*n + 3*n*max(1, lgn) + 4*n + n*n
    lwork = 1 + 4*n + n*n + tstore + inner
    liwork = 6 + 6*n + 5*n*max(1,lgn)
    if (n .le. smlsiz) then
       lwork = max(1, 2*n)
       liwork = 1
    end if
  end subroutine hidden_workspace
end program solve_values
F90

cat > "$CSRC" <<'C'
#include <mpfr.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct { char mode[32]; long n; long k; } Query;

typedef struct { char mode[32]; long n; double *d; double *e; } Matrix;

static Matrix load_matrix(const char *cache_dir, const char *mode, long n) {
  Matrix m;
  snprintf(m.mode, sizeof(m.mode), "%s", mode);
  m.n = n;
  m.d = (double*)malloc((size_t)n * sizeof(double));
  m.e = (double*)malloc((size_t)(n-1) * sizeof(double));
  if (!m.d || !m.e) { fprintf(stderr, "alloc failed\n"); exit(2); }
  char path[512];
  snprintf(path, sizeof(path), "%s/%s_%ld.de.bin", cache_dir, mode, n);
  FILE *f = fopen(path, "rb");
  if (!f) { perror(path); exit(3); }
  long long file_n = 0;
  if (fread(&file_n, sizeof(file_n), 1, f) != 1) { fprintf(stderr, "bad header %s\n", path); exit(4); }
  if (file_n != n) { fprintf(stderr, "n mismatch %s got %lld want %ld\n", path, file_n, n); exit(5); }
  if (fread(m.d, sizeof(double), (size_t)n, f) != (size_t)n) { fprintf(stderr, "bad d %s\n", path); exit(6); }
  if (fread(m.e, sizeof(double), (size_t)(n-1), f) != (size_t)(n-1)) { fprintf(stderr, "bad e %s\n", path); exit(7); }
  fclose(f);
  return m;
}

static long sturm_count(const Matrix *m, mpfr_t x, mpfr_prec_t prec) {
  mpfr_t p, prev, tmp;
  mpfr_init2(p, prec);
  mpfr_init2(prev, prec);
  mpfr_init2(tmp, prec);
  long count = 0;
  for (long i = 0; i < m->n; ++i) {
    mpfr_set_d(p, m->d[i], MPFR_RNDN);
    mpfr_sub(p, p, x, MPFR_RNDN);
    if (i > 0) {
      if (mpfr_zero_p(prev)) {
        mpfr_set_si(prev, -1, MPFR_RNDN);
        mpfr_mul_2si(prev, prev, -((long)prec / 2), MPFR_RNDN);
      }
      mpfr_set_d(tmp, m->e[i-1], MPFR_RNDN);
      mpfr_mul(tmp, tmp, tmp, MPFR_RNDN);
      mpfr_div(tmp, tmp, prev, MPFR_RNDN);
      mpfr_sub(p, p, tmp, MPFR_RNDN);
    }
    if (mpfr_sgn(p) < 0) count++;
    mpfr_set(prev, p, MPFR_RNDN);
  }
  mpfr_clear(p); mpfr_clear(prev); mpfr_clear(tmp);
  return count;
}

static void gershgorin_bounds(const Matrix *m, double *lo, double *hi) {
  double l = m->d[0], h = m->d[0];
  for (long i = 0; i < m->n; ++i) {
    double r = 0.0;
    if (i > 0) r += fabs(m->e[i-1]);
    if (i + 1 < m->n) r += fabs(m->e[i]);
    double a = m->d[i] - r;
    double b = m->d[i] + r;
    if (a < l) l = a;
    if (b > h) h = b;
  }
  *lo = l - 1.0;
  *hi = h + 1.0;
}

static void bisect_index(const Matrix *m, long k, mpfr_prec_t prec, int iters, mpfr_t ref, mpfr_t width) {
  double dlo, dhi;
  gershgorin_bounds(m, &dlo, &dhi);
  mpfr_t lo, hi, mid;
  mpfr_init2(lo, prec); mpfr_init2(hi, prec); mpfr_init2(mid, prec);
  mpfr_set_d(lo, dlo, MPFR_RNDN);
  mpfr_set_d(hi, dhi, MPFR_RNDN);
  for (int it = 0; it < iters; ++it) {
    mpfr_add(mid, lo, hi, MPFR_RNDN);
    mpfr_div_ui(mid, mid, 2, MPFR_RNDN);
    long c = sturm_count(m, mid, prec);
    if (c >= k) mpfr_set(hi, mid, MPFR_RNDN);
    else mpfr_set(lo, mid, MPFR_RNDN);
  }
  mpfr_add(ref, lo, hi, MPFR_RNDN);
  mpfr_div_ui(ref, ref, 2, MPFR_RNDN);
  mpfr_sub(width, hi, lo, MPFR_RNDN);
  mpfr_clear(lo); mpfr_clear(hi); mpfr_clear(mid);
}

int main(int argc, char **argv) {
  if (argc != 4) {
    fprintf(stderr, "usage: %s CACHE_DIR queries.csv refs.csv\n", argv[0]);
    return 2;
  }
  const char *cache_dir = argv[1];
  FILE *in = fopen(argv[2], "r");
  if (!in) { perror(argv[2]); return 3; }
  FILE *out = fopen(argv[3], "w");
  if (!out) { perror(argv[3]); return 4; }
  fprintf(out, "mode,n,index,mpfr_ref,interval_width\n");
  char line[256];
  if (!fgets(line, sizeof(line), in)) return 5;
  Matrix current = {"", 0, NULL, NULL};
  mpfr_prec_t prec = 256;
  int iters = 260;
  while (fgets(line, sizeof(line), in)) {
    Query q;
    if (sscanf(line, "%31[^,],%ld,%ld", q.mode, &q.n, &q.k) != 3) continue;
    if (q.k < 1 || q.k > q.n) continue;
    if (!current.d || current.n != q.n || strcmp(current.mode, q.mode) != 0) {
      free(current.d); free(current.e);
      current = load_matrix(cache_dir, q.mode, q.n);
    }
    mpfr_t ref, width;
    mpfr_init2(ref, prec); mpfr_init2(width, prec);
    bisect_index(&current, q.k, prec, iters, ref, width);
    fprintf(out, "%s,%ld,%ld,", q.mode, q.n, q.k);
    mpfr_fprintf(out, "%.80Rg,%.6Re\n", ref, width);
    fflush(out);
    mpfr_clear(ref); mpfr_clear(width);
  }
  free(current.d); free(current.e);
  fclose(in); fclose(out);
  return 0;
}
C

echo "=== compile solvers and MPFR reference ==="
gfortran -O2 -fopenmp -std=legacy -ffree-line-length-none \
  "$FSRC" "$BUILD_DIR"/br_objs/*.o \
  -L"$MKLROOT/lib" -Wl,-rpath,"$MKLROOT/lib" \
  -lmkl_rt -lpthread -lm -ldl \
  -o "$BUILD_DIR/solve_values"
cc -O2 "$CSRC" -lmpfr -lgmp -lm -o "$BUILD_DIR/mpfr_ref"

export OMP_NUM_THREADS=56
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=56
export MKL_NUM_THREADS=56
export MKL_DYNAMIC=FALSE
export MKL_INTERFACE_LAYER=LP64
export MKL_THREADING_LAYER=GNU

VALUES="$RESULT_DIR/solver_values.csv"
QUERIES="$RESULT_DIR/mpfr_queries.csv"
REFS="$RESULT_DIR/mpfr_refs.csv"
JOINED="$RESULT_DIR/mpfr_joined.csv"
"$BUILD_DIR/solve_values" | tee "$VALUES"
awk -F, 'NR==1{print "mode,n,index"; next} /^uniform|^normal|^toeplitz|^clustered/{print $1","$2","$3}' "$VALUES" > "$QUERIES"
"$BUILD_DIR/mpfr_ref" "$CACHE_DIR" "$QUERIES" "$REFS"
python3 - <<PY
from decimal import Decimal, getcontext
from pathlib import Path
import csv
getcontext().prec = 90
values = Path('$VALUES')
refs = Path('$REFS')
joined = Path('$JOINED')
refmap = {}
with refs.open() as f:
    for r in csv.DictReader(f):
        refmap[(r['mode'], int(r['n']), int(r['index']))] = Decimal(r['mpfr_ref'])
rows = []
with values.open() as f:
    for r in csv.DictReader(f):
        if r['mode'] not in ['uniform','normal','toeplitz','clustered']:
            continue
        key = (r['mode'], int(r['n']), int(r['index']))
        ref = refmap[key]
        scale = max(Decimal(1), abs(ref))
        out = dict(r)
        out['mpfr_ref'] = format(ref, '.40E')
        for m in ['dsterf','internal','br']:
            v = Decimal(r[m])
            err = abs(v - ref)
            out[f'{m}_abs_err_mpfr'] = format(err, '.18E')
            out[f'{m}_norm_err_mpfr'] = format(err / scale, '.18E')
        rows.append(out)
with joined.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print('mode,n,qr_norm,internal_norm,br_norm,qr_idx,internal_idx,br_idx')
for mode in ['uniform','normal','toeplitz','clustered']:
    for n in [16384,49152]:
        sub = [r for r in rows if r['mode']==mode and int(r['n'])==n]
        vals = {}
        for m in ['dsterf','internal','br']:
            best = max(sub, key=lambda r: Decimal(r[f'{m}_norm_err_mpfr']))
            vals[m] = (best[f'{m}_norm_err_mpfr'], best['index'])
        print(f"{mode},{n},{vals['dsterf'][0]},{vals['internal'][0]},{vals['br'][0]},{vals['dsterf'][1]},{vals['internal'][1]},{vals['br'][1]}")
PY

echo "=== result files ==="
find "$RESULT_DIR" -maxdepth 1 -type f -print -exec ls -lh {} \;
echo "=== done ==="
date
