#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mkl_blacs.h"
#include "mkl_scalapack.h"
#include "mpi.h"

static uint64_t make_seed(int mode, int n) {
  uint64_t seed = 7640891576956012809ULL;
  seed ^= (uint64_t)mode * 1000003ULL;
  seed ^= (uint64_t)n * 1099511628211ULL;
  return seed ? seed : 88172645463393265ULL;
}

static double randu(uint64_t *seed) {
  uint64_t x = *seed;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  *seed = x;
  x *= 2685821657736338717ULL;
  double u = (double)(x & 0x7fffffffffffffffULL) /
             (double)0x7fffffffffffffffULL;
  if (u <= 0.0) u = 1e-300;
  if (u >= 1.0) u = 1.0 - 2.2204460492503131e-16;
  return u;
}

static void fill_uniform(int n, uint64_t seed0, double *d, double *e) {
  uint64_t seed = seed0;
  for (int i = 0; i < n; ++i) d[i] = 2.0 * randu(&seed) - 1.0;
  for (int i = 0; i < n - 1; ++i) e[i] = 0.10 + 0.20 * randu(&seed);
}

static double randn(uint64_t *seed) {
  double u1 = randu(seed);
  double u2 = randu(seed);
  return sqrt(-2.0 * log(u1)) * cos(6.283185307179586476925286766559 * u2);
}

static int read_reduced_dense(const char *path, int n, double *d, double *e) {
  FILE *fp = fopen(path, "rb");
  if (!fp) return 1;
  int64_t file_n = 0;
  if (fread(&file_n, sizeof(file_n), 1, fp) != 1) {
    fclose(fp);
    return 2;
  }
  if (file_n != (int64_t)n) {
    fclose(fp);
    return 3;
  }
  if (fread(d, sizeof(double), (size_t)n, fp) != (size_t)n) {
    fclose(fp);
    return 4;
  }
  if (fread(e, sizeof(double), (size_t)(n - 1), fp) != (size_t)(n - 1)) {
    fclose(fp);
    return 5;
  }
  fclose(fp);
  return 0;
}

static int fill_case(const char *mode, int n, double *d, double *e,
                     const char *input_path) {
  /*
   * When an input path is supplied, always read the exact cached D/E arrays.
   * The generator fallback is retained for the intentionally pseudo-random
   * scaling probe.
   */
  if (input_path && input_path[0] != '\0') {
    return read_reduced_dense(input_path, n, d, e);
  } else if (strcmp(mode, "normal") == 0) {
    uint64_t seed = make_seed(2, n);
    for (int i = 0; i < n; ++i) d[i] = randn(&seed);
    for (int i = 0; i < n - 1; ++i) e[i] = 0.10 + 0.20 * randu(&seed);
  } else if (strcmp(mode, "toeplitz") == 0) {
    for (int i = 0; i < n; ++i) d[i] = 2.0;
    for (int i = 0; i < n - 1; ++i) e[i] = 0.25;
  } else if (strcmp(mode, "clustered") == 0) {
    double center = 0.5 * ((double)n + 1.0);
    for (int i = 0; i < n; ++i) d[i] = 1.0 + 1.0e-12 * (((double)i + 1.0) - center);
    for (int i = 0; i < n - 1; ++i) e[i] = 1.0e-4 * (1.0 + 0.1 * cos(0.33 * ((double)i + 1.0)));
  } else {
    fill_uniform(n, make_seed(1, n), d, e);
  }
  return 0;
}

static void choose_grid(int p, int *pr, int *pc) {
  *pr = 1;
  for (int r = 1; r <= p; ++r) {
    if (p % r == 0 && r * r <= p) *pr = r;
  }
  *pc = p / *pr;
}

static double checksum(MKL_INT n, const double *w) {
  double sum = 0.0;
  for (MKL_INT i = 0; i < n; ++i) {
    sum += w[i] * (double)(((i + 1) % 127) + 1);
  }
  return sum;
}

static int run_one(MKL_INT ictxt, MKL_INT iam, MKL_INT nprocs, MKL_INT n,
                   MKL_INT nprow, MKL_INT npcol, MKL_INT lwork,
                   MKL_INT liwork, const char *mode,
                   const char *input_path) {
  double *d = (double *)malloc((size_t)n * sizeof(double));
  double *e = (double *)malloc((size_t)(n > 1 ? n - 1 : 1) * sizeof(double));
  double *w = (double *)malloc((size_t)n * sizeof(double));
  MKL_INT *iblock = (MKL_INT *)malloc((size_t)n * sizeof(MKL_INT));
  MKL_INT *isplit = (MKL_INT *)malloc((size_t)n * sizeof(MKL_INT));
  double *work = (double *)malloc((size_t)lwork * sizeof(double));
  MKL_INT *iwork = (MKL_INT *)malloc((size_t)liwork * sizeof(MKL_INT));
  if (!d || !e || !w || !iblock || !isplit || !work || !iwork) {
    if (iam == 0) fprintf(stderr, "allocation failed for n=%d\n", (int)n);
    free(d);
    free(e);
    free(w);
    free(iblock);
    free(isplit);
    free(work);
    free(iwork);
    return 12;
  }

  int fill_info = fill_case(mode, (int)n, d, e, input_path);
  if (fill_info != 0) {
    if (iam == 0) {
      fprintf(stderr, "fill_case failed mode=%s n=%d code=%d path=%s\n",
              mode, (int)n, fill_info, input_path ? input_path : "");
    }
    free(d);
    free(e);
    free(w);
    free(iblock);
    free(isplit);
    free(work);
    free(iwork);
    return 13;
  }

  double vl = 0.0, vu = 0.0, abstol = 0.0;
  MKL_INT il = 1, iu = n, m = 0, nsplit = 0, info = 0;

  MPI_Barrier(MPI_COMM_WORLD);
  double t0 = MPI_Wtime();
  PDSTEBZ(&ictxt, "A", "E", &n, &vl, &vu, &il, &iu, &abstol, d, e, &m,
          &nsplit, w, iblock, isplit, work, &lwork, iwork, &liwork, &info);
  MPI_Barrier(MPI_COMM_WORLD);
  double t1 = MPI_Wtime();

  if (iam == 0) {
    double chk = (info == 0 && m == n) ? checksum(m, w) : 0.0;
    printf("PDSTEBZ,%s,%d,%d,%d,%d,%.4f,%d,%d,%d,%d,%d,%.16e\n", mode,
           (int)n, (int)nprocs, (int)nprow, (int)npcol,
           1000.0 * (t1 - t0), (int)info, (int)m, (int)nsplit,
           (int)lwork, (int)liwork, chk);
    fflush(stdout);
  }

  free(d);
  free(e);
  free(w);
  free(iblock);
  free(isplit);
  free(work);
  free(iwork);
  return info == 0 ? 0 : 20;
}

int main(int argc, char **argv) {
  MKL_INT first_n = 4096, last_n = 65536;
  const char *mode = "uniform";
  const char *input_path = "";
  if (argc > 1) first_n = (MKL_INT)atoi(argv[1]);
  if (argc > 2) last_n = (MKL_INT)atoi(argv[2]);
  if (argc > 3) mode = argv[3];
  if (argc > 4) input_path = argv[4];

  MKL_INT iam = 0, nprocs = 1;
  MKL_INT ictxt = 0, nprow = 1, npcol = 1, myrow = 0, mycol = 0;
  MKL_INT minus_one = -1, zero = 0;

  blacs_pinfo(&iam, &nprocs);
  choose_grid((int)nprocs, (int *)&nprow, (int *)&npcol);
  blacs_get(&minus_one, &zero, &ictxt);
  blacs_gridinit(&ictxt, "R", &nprow, &npcol);
  blacs_gridinfo(&ictxt, &nprow, &npcol, &myrow, &mycol);

  if (iam == 0) {
    printf("method,mode,n,nprocs,nprow,npcol,time_ms,info,m,nsplit,lwork,liwork,checksum\n");
    fflush(stdout);
  }

  const char *all_modes[] = {"uniform", "normal", "toeplitz", "clustered"};
  int mode_count = strcmp(mode, "all") == 0 ? 4 : 1;
  for (int imode = 0; imode < mode_count; ++imode) {
    const char *active_mode = mode_count == 4 ? all_modes[imode] : mode;
    for (MKL_INT n = first_n; n <= last_n; n *= 2) {
      MKL_INT lwork = 8 * n + 4 * nprocs + 1024;
      MKL_INT liwork = 6 * n + 4 * nprocs + 1024;
      int rc = run_one(ictxt, iam, nprocs, n, nprow, npcol, lwork, liwork,
                       active_mode, input_path);
      if (rc != 0) {
        blacs_gridexit(&ictxt);
        blacs_exit(&zero);
        return rc;
      }
    }
  }

  blacs_gridexit(&ictxt);
  blacs_exit(&zero);
  return 0;
}
