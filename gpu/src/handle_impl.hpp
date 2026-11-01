#ifndef EVD_LIBRARY_INTERNAL_HANDLE_IMPL_HPP
#define EVD_LIBRARY_INTERNAL_HANDLE_IMPL_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "evd/handle.h"
#include "evd/types.h"

struct evd_handle_struct {
    struct TridiagMergeLevelDiagnostics {
        int level;
        int merge_count;
        int observed_merges;
        int size_min;
        int size_max;
        int size_sum;
        int dd_before_min;
        int dd_before_max;
        int dd_before_sum;
        int dd_after_min;
        int dd_after_max;
        int dd_after_sum;
        int secular_iter_count;
        int secular_iter_sum;
        int secular_iter_max;
        int secular_iter_failed;
        int zero_deflated_total;
        int close_deflated_total;
        int deflated_total;
        int vector_task_count;
        int vector_batch_groups;
        int vector_batched_tasks;
        int vector_single_tasks;
        int vector_deflated_tasks;
        int vector_no_deflation_tasks;
        int vector_max_size;
        int vector_max_dd;
        std::uint64_t vector_mnk_sum;
        std::uint64_t split_left_mnk_sum;
        std::uint64_t split_right_mnk_sum;
        int split_left_max_m;
        int split_left_max_n;
        int split_left_max_k;
        int split_right_max_m;
        int split_right_max_n;
        int split_right_max_k;
        float values_solve_ms;
        float values_active_roots_ms;
        float values_deflated_boundary_ms;
        float values_copyd_ms;
        float values_secular_solve_ms;
        float values_rescale_ms;
        float values_boundary_prepare_ms;
        float values_boundary_update_ms;
        float rescale_ms;
        float copyc_ms;
        float reshufflec_ms;
        float vector_prepare_ms;
        float gemm_ms;
        float update_ms;
        float global_copyc_ms;
        float global_reshufflec_ms;
        float deflated_gather_ms;
        float split_make_indices_ms;
        float split_left_pack_ms;
        float split_left_gemm_ms;
        float split_right_pack_ms;
        float split_right_gemm_ms;
        float merge_update_ms;
        float update_all_vectors_ms;
    };

    struct TridiagRunDiagnostics {
        int enabled;
        int available;
        int total_merges;
        int residual_failed_merges;
        std::vector<TridiagMergeLevelDiagnostics> levels;
    };

    struct BackTransformDiagnostics {
        int available;
        double copy_signed_ms;
        double bc_back_ms;
        double sbr_back_ms;
    };

    cublasHandle_t cublas_handle;
    cusolverDnHandle_t cusolver_handle;
    cudaStream_t stream;
    void *workspace;
    size_t workspace_size;
    int timing_enabled;
    int last_error_code;
    evd_error_source last_error_source;
    const char *last_error_site;
    cudaEvent_t stage_start_events[static_cast<int>(EVD_STAGE_COUNT)];
    cudaEvent_t stage_stop_events[static_cast<int>(EVD_STAGE_COUNT)];
    evd_stage_timing last_stage_timings[static_cast<int>(EVD_STAGE_COUNT)];
    TridiagRunDiagnostics last_tridiag_diagnostics;
    BackTransformDiagnostics last_back_transform_diagnostics;
};

namespace evd::internal {

inline const char *stage_name(evd_stage_id stage) {
    switch (stage) {
    case EVD_STAGE_REDUCTION_TO_BAND:
        return "reduction_to_band";
    case EVD_STAGE_BULGE_CHASING:
        return "bulge_chasing";
    case EVD_STAGE_TRIDIAG_SOLVER:
        return "tridiag_solver";
    case EVD_STAGE_BACK_TRANSFORM:
        return "back_transform";
    case EVD_STAGE_COUNT:
    default:
        return "unknown";
    }
}

inline void reset_stage_timings(evd_handle handle) {
    if (!handle) {
        return;
    }

    for (int i = 0; i < static_cast<int>(EVD_STAGE_COUNT); ++i) {
        handle->last_stage_timings[i].stage = static_cast<evd_stage_id>(i);
        handle->last_stage_timings[i].name =
            stage_name(static_cast<evd_stage_id>(i));
        handle->last_stage_timings[i].milliseconds = 0.0;
        handle->last_stage_timings[i].available = 0;
    }

    handle->last_back_transform_diagnostics.available = 0;
    handle->last_back_transform_diagnostics.copy_signed_ms = 0.0;
    handle->last_back_transform_diagnostics.bc_back_ms = 0.0;
    handle->last_back_transform_diagnostics.sbr_back_ms = 0.0;
}

inline void reset_tridiag_diagnostics(evd_handle handle,
                                      const bool enabled = false) {
    if (!handle) {
        return;
    }

    handle->last_tridiag_diagnostics.enabled = enabled ? 1 : 0;
    handle->last_tridiag_diagnostics.available = 0;
    handle->last_tridiag_diagnostics.total_merges = 0;
    handle->last_tridiag_diagnostics.residual_failed_merges = 0;
    handle->last_tridiag_diagnostics.levels.clear();
}

inline void reset_last_error(evd_handle handle) {
    if (!handle) {
        return;
    }

    handle->last_error_code = 0;
    handle->last_error_source = EVD_ERROR_SRC_NONE;
    handle->last_error_site = nullptr;
}

inline void set_last_error(evd_handle handle, int code, evd_error_source source,
                           const char *site) {
    if (!handle) {
        return;
    }

    handle->last_error_code = code;
    handle->last_error_source = source;
    handle->last_error_site = site;
}

} // namespace evd::internal

#define EVD_RETURN_IF_CUDA_ERROR(handle, site, ...)                            \
    do {                                                                       \
        const cudaError_t evd_cuda_status__ = (__VA_ARGS__);                   \
        if (evd_cuda_status__ != cudaSuccess) {                                \
            ::evd::internal::set_last_error(                                   \
                (handle), static_cast<int>(evd_cuda_status__),                 \
                EVD_ERROR_SRC_CUDA, site);                                     \
            return EVD_STATUS_CUDA_ERROR;                                      \
        }                                                                      \
    } while (false)

#define EVD_RETURN_IF_CUBLAS_ERROR(handle, site, ...)                          \
    do {                                                                       \
        const cublasStatus_t evd_cublas_status__ = (__VA_ARGS__);              \
        if (evd_cublas_status__ != CUBLAS_STATUS_SUCCESS) {                    \
            ::evd::internal::set_last_error(                                   \
                (handle), static_cast<int>(evd_cublas_status__),               \
                EVD_ERROR_SRC_CUBLAS, site);                                   \
            return EVD_STATUS_CUDA_ERROR;                                      \
        }                                                                      \
    } while (false)

#define EVD_RETURN_IF_CUSOLVER_ERROR(handle, site, ...)                        \
    do {                                                                       \
        const cusolverStatus_t evd_cusolver_status__ = (__VA_ARGS__);          \
        if (evd_cusolver_status__ != CUSOLVER_STATUS_SUCCESS) {                \
            ::evd::internal::set_last_error(                                   \
                (handle), static_cast<int>(evd_cusolver_status__),             \
                EVD_ERROR_SRC_CUSOLVER, site);                                 \
            return EVD_STATUS_CUDA_ERROR;                                      \
        }                                                                      \
    } while (false)

#endif // EVD_LIBRARY_INTERNAL_HANDLE_IMPL_HPP
