#ifndef EVD_TYPES_H
#define EVD_TYPES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum evd_fill_mode {
    EVD_FILL_MODE_LOWER = 0,
    EVD_FILL_MODE_UPPER = 1
} evd_fill_mode;

typedef enum evd_stage_id {
    EVD_STAGE_REDUCTION_TO_BAND = 0,
    EVD_STAGE_BULGE_CHASING = 1,
    EVD_STAGE_TRIDIAG_SOLVER = 2,
    EVD_STAGE_BACK_TRANSFORM = 3,
    EVD_STAGE_COUNT = 4
} evd_stage_id;

typedef enum evd_error_source {
    EVD_ERROR_SRC_NONE = 0,
    EVD_ERROR_SRC_CUDA,
    EVD_ERROR_SRC_CUBLAS,
    EVD_ERROR_SRC_CUSOLVER
} evd_error_source;

typedef enum evd_tridiag_solver_backend {
    EVD_TRIG_SOLVER_BACKEND_AUTO = 0,
    EVD_TRIG_SOLVER_BACKEND_CPU = 1,
    EVD_TRIG_SOLVER_BACKEND_GPU = 2
} evd_tridiag_solver_backend;

typedef enum evd_tridiag_sort_backend {
    EVD_TRIDIAG_SORT_BACKEND_AUTO = 0,
    EVD_TRIDIAG_SORT_BACKEND_LEGACY = 1,
    EVD_TRIDIAG_SORT_BACKEND_CUB = 2
} evd_tridiag_sort_backend;

typedef struct evd_stage_timing {
    evd_stage_id stage;
    const char *name;
    double milliseconds;
    int available;
} evd_stage_timing;

typedef struct evd_syev2stage_config {
    int64_t n;
    int64_t band_size;
    int64_t bulge_block_size;
    evd_fill_mode uplo;
    int compute_eigenvectors;
    int enable_timing;
    evd_tridiag_solver_backend tridiag_solver_backend;
    evd_tridiag_sort_backend tridiag_sort_backend;
    int enable_tridiag_diagnostics;
} evd_syev2stage_config;

#ifdef __cplusplus
} // extern "C"
#endif

#endif // EVD_TYPES_H
