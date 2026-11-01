#ifndef EVD_LIBRARY_INTERNAL_TRIDIAG_SOLVER_DC_CUH
#define EVD_LIBRARY_INTERNAL_TRIDIAG_SOLVER_DC_CUH

#include <cstddef>
#include <cstdint>

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

#include "evd/status.h"
#include "evd/types.h"
#include "workspace_align.hpp"

namespace evd::internal {

inline constexpr int kTridiagSolverExtractBlockSize = 256;
inline constexpr int kTridiagSolverLeafBlockSize = 32;
inline constexpr int kTridiagSolverMaxLeafSize = 32;
inline constexpr int kTridiagSolverMinDcSize = 17;

inline constexpr std::size_t align_tridiag_workspace(std::size_t value) {
    const std::size_t remainder = value % kDeviceWorkspaceAlignment;
    return remainder == 0 ? value
                          : value + kDeviceWorkspaceAlignment - remainder;
}

inline constexpr std::size_t get_splits_size(const int n) {
    return 13UL * static_cast<std::size_t>(n);
}
inline constexpr std::size_t get_tmpz_size(const int n) {
    return 10UL * static_cast<std::size_t>(n);
}
inline constexpr std::size_t get_boundary_rows_size(const int n) {
    return 2UL * static_cast<std::size_t>(n);
}

inline bool use_gpu_tridiag_solver(const evd_syev2stage_config &config) {
    if (config.tridiag_solver_backend == EVD_TRIG_SOLVER_BACKEND_CPU) {
        return false;
    }
    if (config.tridiag_solver_backend == EVD_TRIG_SOLVER_BACKEND_GPU) {
        return true;
    }
    // AUTO: use GPU for sizes where the DC algorithm is active.
    return config.n >= kTridiagSolverMinDcSize;
}

template <typename T>
__host__ __device__ inline T *ptr_msz(const int n, T *splits) {
    return splits + 0 * n;
}

template <typename T>
__host__ __device__ inline T *ptr_mps(const int n, T *splits) {
    return splits + 1 * n;
}

template <typename T>
__host__ __device__ inline T *ptr_bsz(const int n, T *splits) {
    return splits + 2 * n;
}

template <typename T>
__host__ __device__ inline T *ptr_bps(const int n, T *splits) {
    return splits + 3 * n;
}

template <typename T>
__host__ __device__ inline T *ptr_em(const int n, T *splits) {
    return splits + 4 * n;
}

template <typename T>
__host__ __device__ inline T *ptr_nsz(const int n, T *splits) {
    return splits + 5 * n;
}

template <typename T>
__host__ __device__ inline T *ptr_nps(const int n, T *splits) {
    return splits + 6 * n;
}

template <typename T>
__host__ __device__ inline T *ptr_ndd(const int n, T *splits) {
    return splits + 7 * n;
}

template <typename T>
__host__ __device__ inline T *ptr_mask(const int n, T *splits) {
    return splits + 8 * n;
}

template <typename T>
__host__ __device__ inline T *ptr_dcount(const int n, T *splits) {
    return splits + 9 * n;
}

template <typename T>
__host__ __device__ inline T *ptr_map(const int n, T *splits) {
    return splits + 10 * n;
}

template <typename T>
__host__ __device__ inline T *ptr_cand(const int n, T *splits) {
    return splits + 11 * n;
}

template <typename T>
__host__ __device__ inline T *ptr_dbg(const int n, T *splits) {
    return splits + 12 * n;
}

template <typename T>
__host__ __device__ inline T *ptr_z(const int n, T *tmpz) {
    return tmpz + 0 * n;
}

template <typename T>
__host__ __device__ inline T *ptr_evs(const int n, T *tmpz) {
    return tmpz + 1 * n;
}

template <typename T>
__host__ __device__ inline T *ptr_cc(const int n, T *tmpz) {
    return tmpz + 2 * n;
}

template <typename T>
__host__ __device__ inline T *ptr_ss(const int n, T *tmpz) {
    return tmpz + 3 * n;
}

template <typename T>
__host__ __device__ inline T *ptr_tolsD(const int n, T *tmpz) {
    return tmpz + 4 * n;
}

template <typename T>
__host__ __device__ inline T *ptr_tolsZ(const int n, T *tmpz) {
    return tmpz + 5 * n;
}

template <typename T>
__host__ __device__ inline T *ptr_md(const int n, T *tmpz) {
    return tmpz + 6 * n;
}

template <typename T>
__host__ __device__ inline T *ptr_cd(const int n, T *tmpz) {
    return tmpz + 7 * n;
}

template <typename T>
__host__ __device__ inline T *ptr_cz(const int n, T *tmpz) {
    return tmpz + 8 * n;
}

template <typename T>
__host__ __device__ inline T *ptr_r1p(const int n, T *tmpz) {
    return tmpz + 9 * n;
}

template <typename T>
__host__ __device__ inline T *ptr_first_rows(const int n, T *boundary_rows) {
    (void)n;
    return boundary_rows;
}

template <typename T>
__host__ __device__ inline T *ptr_last_rows(const int n, T *boundary_rows) {
    return boundary_rows + n;
}

inline constexpr std::size_t get_tempgemm_size(const int n) {
    const std::size_t size = static_cast<std::size_t>(n);
    return size * size;
}

inline constexpr std::size_t get_values_only_tempgemm_scratch_size(
    const int n) {
    const std::size_t size = static_cast<std::size_t>(n);
    return 16UL * size;
}

inline constexpr int kValuesOnlyEtmpdColumns = 4096;

inline constexpr std::size_t get_values_only_etmpd_scratch_size(
    const int n) {
    const std::size_t size = static_cast<std::size_t>(n);
    const std::size_t columns =
        static_cast<std::size_t>(n < kValuesOnlyEtmpdColumns
                                     ? n
                                     : kValuesOnlyEtmpdColumns);
    return size * columns;
}

template <typename T>
__host__ __device__ inline T *ptr_vecs(const int n, T *tempgemm) {
    (void)n;
    return tempgemm;
}

template <typename T>
__host__ __device__ inline T *ptr_etmpd(const int n, T *tempgemm) {
    const std::size_t size = static_cast<std::size_t>(n);
    return tempgemm + size * size;
}

inline int stedc_num_levels(const int n) {
    if (n <= 16) {
        return 0;
    }

    int levels = 0;
    int remaining = n - 1;
    while (remaining > 0) {
        ++levels;
        remaining >>= 1;
    }
    return levels - 4;
}

struct TridiagSolverWorkspaceLayout {
    std::size_t diag_offset{0};
    std::size_t diag_bytes{0};
    std::size_t subdiag_offset{0};
    std::size_t subdiag_bytes{0};
    std::size_t dc_diag_offset{0};
    std::size_t dc_diag_bytes{0};
    std::size_t dc_subdiag_offset{0};
    std::size_t dc_subdiag_bytes{0};
    std::size_t tempgemm_offset{0};
    std::size_t tempgemm_bytes{0};
    std::size_t etmpd_offset{0};
    std::size_t etmpd_bytes{0};
    std::size_t split_gemm_offset{0};
    std::size_t split_gemm_bytes{0};
    std::size_t temp_eigs_offset{0};
    std::size_t temp_eigs_bytes{0};
    std::size_t splits_offset{0};
    std::size_t splits_bytes{0};
    std::size_t tmpz_offset{0};
    std::size_t tmpz_bytes{0};
    std::size_t eigenvectors_offset{0};
    std::size_t eigenvectors_bytes{0};
    std::size_t boundary_rows_offset{0};
    std::size_t boundary_rows_bytes{0};
    std::size_t total{0};
};

struct TridiagSolverWorkspaceView {
    double *d_diag{nullptr};
    double *d_subdiag{nullptr};
    double *d_dc_diag{nullptr};
    double *d_dc_subdiag{nullptr};
    double *d_tempgemm{nullptr};
    double *d_etmpd{nullptr};
    double *d_split_gemm_scratch{nullptr};
    double *d_temp_eigs{nullptr};
    int *d_splits{nullptr};
    double *d_tmpz{nullptr};
    double *d_tridiag_eigenvectors{nullptr};
    double *d_boundary_rows{nullptr};
};

template <typename T>
inline T *workspace_ptr_at(void *base, const std::size_t offset) {
    if (base == nullptr) {
        return nullptr;
    }

    return reinterpret_cast<T *>(static_cast<unsigned char *>(base) + offset);
}

inline evd_status
make_tridiag_solver_workspace_layout(const evd_syev2stage_config &config,
                                     TridiagSolverWorkspaceLayout *layout,
                                     const bool external_eigenvectors = false) {
    if (layout == nullptr) {
        return EVD_STATUS_INVALID_POINTER;
    }

    if (config.n <= 0) {
        return EVD_STATUS_INVALID_VALUE;
    }

    const std::size_t n = static_cast<std::size_t>(config.n);
    std::size_t cursor = 0;

    cursor = align_tridiag_workspace(cursor);
    layout->diag_offset = cursor;
    layout->diag_bytes = n * sizeof(double);
    cursor += layout->diag_bytes;

    cursor = align_tridiag_workspace(cursor);
    layout->subdiag_offset = cursor;
    layout->subdiag_bytes = (n > 1 ? (n - 1) : 1) * sizeof(double);
    cursor += layout->subdiag_bytes;

    layout->dc_diag_offset = cursor;
    layout->dc_diag_bytes = 0;
    layout->dc_subdiag_offset = cursor;
    layout->dc_subdiag_bytes = 0;
    layout->tempgemm_offset = cursor;
    layout->tempgemm_bytes = 0;
    layout->etmpd_offset = cursor;
    layout->etmpd_bytes = 0;
    layout->split_gemm_offset = cursor;
    layout->split_gemm_bytes = 0;
    layout->temp_eigs_offset = cursor;
    layout->temp_eigs_bytes = 0;
    layout->splits_offset = cursor;
    layout->splits_bytes = 0;
    layout->tmpz_offset = cursor;
    layout->tmpz_bytes = 0;
    if (use_gpu_tridiag_solver(config)) {
        cursor = align_tridiag_workspace(cursor);
        layout->dc_diag_offset = cursor;
        layout->dc_diag_bytes = n * sizeof(double);
        cursor += layout->dc_diag_bytes;

        cursor = align_tridiag_workspace(cursor);
        layout->dc_subdiag_offset = cursor;
        layout->dc_subdiag_bytes = (n > 1 ? (n - 1) : 1) * sizeof(double);
        cursor += layout->dc_subdiag_bytes;

        cursor = align_tridiag_workspace(cursor);
        layout->tempgemm_offset = cursor;
        layout->tempgemm_bytes =
            (config.compute_eigenvectors
                 ? get_tempgemm_size(config.n)
                 : get_values_only_tempgemm_scratch_size(config.n)) *
            sizeof(double);
        cursor += layout->tempgemm_bytes;

        cursor = align_tridiag_workspace(cursor);
        layout->etmpd_offset = cursor;
        layout->etmpd_bytes =
            (config.compute_eigenvectors
                 ? get_tempgemm_size(config.n)
                 : get_values_only_etmpd_scratch_size(config.n)) *
            sizeof(double);
        cursor += layout->etmpd_bytes;

        if (config.compute_eigenvectors) {
            cursor = align_tridiag_workspace(cursor);
            layout->split_gemm_offset = cursor;
            layout->split_gemm_bytes = n * n * sizeof(double);
            cursor += layout->split_gemm_bytes;
        }

        cursor = align_tridiag_workspace(cursor);
        layout->temp_eigs_offset = cursor;
        layout->temp_eigs_bytes = n * sizeof(double);
        cursor += layout->temp_eigs_bytes;

        cursor = align_tridiag_workspace(cursor);
        layout->splits_offset = cursor;
        layout->splits_bytes = get_splits_size(config.n) * sizeof(int);
        cursor += layout->splits_bytes;

        cursor = align_tridiag_workspace(cursor);
        layout->boundary_rows_offset = cursor;
        layout->boundary_rows_bytes =
            get_boundary_rows_size(config.n) * sizeof(double);
        cursor += layout->boundary_rows_bytes;

        cursor = align_tridiag_workspace(cursor);
        layout->tmpz_offset = cursor;
        layout->tmpz_bytes = get_tmpz_size(config.n) * sizeof(double);
        cursor += layout->tmpz_bytes;
    }

    layout->eigenvectors_offset = cursor;
    layout->eigenvectors_bytes = 0;
    if (config.compute_eigenvectors && !external_eigenvectors) {
        cursor = align_tridiag_workspace(cursor);
        layout->eigenvectors_offset = cursor;
        layout->eigenvectors_bytes = n * n * sizeof(double);
        cursor += layout->eigenvectors_bytes;
    }

    layout->total = align_tridiag_workspace(cursor);
    return EVD_STATUS_SUCCESS;
}

inline evd_status make_tridiag_solver_workspace_view(
    const evd_syev2stage_config &config, void *workspace,
    const std::size_t workspace_size, TridiagSolverWorkspaceView *view,
    TridiagSolverWorkspaceLayout *layout_out = nullptr,
    double *external_eigenvectors = nullptr) {
    if (view == nullptr) {
        return EVD_STATUS_INVALID_POINTER;
    }

    TridiagSolverWorkspaceLayout layout{};
    const evd_status status =
        make_tridiag_solver_workspace_layout(
            config, &layout, external_eigenvectors != nullptr);
    if (status != EVD_STATUS_SUCCESS) {
        return status;
    }

    if (workspace_size < layout.total) {
        return EVD_STATUS_INVALID_WORKSPACE;
    }

    if (layout.total > 0 && workspace == nullptr) {
        return EVD_STATUS_INVALID_WORKSPACE;
    }

    view->d_diag = workspace_ptr_at<double>(workspace, layout.diag_offset);
    view->d_subdiag =
        workspace_ptr_at<double>(workspace, layout.subdiag_offset);
    view->d_dc_diag =
        layout.dc_diag_bytes == 0
            ? nullptr
            : workspace_ptr_at<double>(workspace, layout.dc_diag_offset);
    view->d_dc_subdiag =
        layout.dc_subdiag_bytes == 0
            ? nullptr
            : workspace_ptr_at<double>(workspace, layout.dc_subdiag_offset);
    view->d_tempgemm =
        layout.tempgemm_bytes == 0
            ? nullptr
            : workspace_ptr_at<double>(workspace, layout.tempgemm_offset);
    view->d_etmpd =
        layout.etmpd_bytes == 0
            ? nullptr
            : workspace_ptr_at<double>(workspace, layout.etmpd_offset);
    view->d_split_gemm_scratch =
        layout.split_gemm_bytes == 0
            ? nullptr
            : workspace_ptr_at<double>(workspace, layout.split_gemm_offset);
    view->d_temp_eigs =
        layout.temp_eigs_bytes == 0
            ? nullptr
            : workspace_ptr_at<double>(workspace, layout.temp_eigs_offset);
    view->d_splits =
        layout.splits_bytes == 0
            ? nullptr
            : workspace_ptr_at<int>(workspace, layout.splits_offset);
    view->d_tmpz =
        layout.tmpz_bytes == 0
            ? nullptr
            : workspace_ptr_at<double>(workspace, layout.tmpz_offset);
    view->d_tridiag_eigenvectors =
        external_eigenvectors != nullptr
            ? external_eigenvectors
            : (layout.eigenvectors_bytes == 0
                   ? nullptr
                   : workspace_ptr_at<double>(workspace,
                                              layout.eigenvectors_offset));
    view->d_boundary_rows =
        layout.boundary_rows_bytes == 0
            ? nullptr
            : workspace_ptr_at<double>(workspace, layout.boundary_rows_offset);

    if (layout_out != nullptr) {
        *layout_out = layout;
    }

    return EVD_STATUS_SUCCESS;
}

} // namespace evd::internal

#endif // EVD_LIBRARY_INTERNAL_TRIDIAG_SOLVER_DC_CUH
