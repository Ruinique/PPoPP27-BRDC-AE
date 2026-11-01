#include "handle_impl.hpp"
#include "back_transform.hpp"
#include "syev2stage_pipeline.hpp"
#include "tridiag_solver_dc.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_select.cuh>
#include <limits>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>

#include <nvtx3/nvToolsExt.h>

#include "tridiag_solver_cpu.hpp"
#include "tridiag_solver_device.cuh"
#include "tridiag_solver_extract.cuh"
#include "tridiag_solver_leaf.cuh"
#include "tridiag_solver_merge_kernels.cuh"
#include "tridiag_solver_secular.cuh"

namespace evd::internal {

namespace {

constexpr int kValuesOnlyFullColumnMaxN = 4096;
constexpr int kValuesOnlyPackedSeqMaxDd = kSecularSolveCubMinDd;
constexpr int kValuesOnlyPackedSeqThreads = 128;
constexpr int kValuesOnlyBlockSolveThreads = 128;
constexpr int kValuesOnlyBlockSolve128MaxDd = 8192;

struct MergeVectorsTask {
    int start{0};
    int size{0};
    int dd{0};
    int split{0};
    int left_count{0};
    int right_count{0};

    bool no_deflation() const { return dd == size; }
};

constexpr int kSplitMergeVectorsMinSize = 512;

int split_merge_vectors_min_size(const int n) {
    (void)n;
    return kSplitMergeVectorsMinSize;
}

bool use_split_merge_vectors_task(const int n, const MergeVectorsTask &task) {
    const int left_rows = task.split - task.start;
    const int right_rows = task.size - left_rows;
    return task.size >= split_merge_vectors_min_size(n) && left_rows > 0 &&
           right_rows > 0 && task.left_count > 0 && task.right_count > 0 &&
           task.left_count + task.right_count == task.dd;
}

bool all_merge_vectors_tasks_use_split(
    const int n, const std::vector<MergeVectorsTask> &tasks) {
    if (tasks.empty()) {
        return false;
    }
    for (const MergeVectorsTask &task : tasks) {
        if (!use_split_merge_vectors_task(n, task)) {
            return false;
        }
    }
    return true;
}

struct MergeVectorsDispatchStats {
    int task_count{0};
    int batch_groups{0};
    int batched_tasks{0};
    int single_tasks{0};
    int deflated_tasks{0};
    int no_deflation_tasks{0};
    int max_size{0};
    int max_dd{0};
    std::uint64_t mnk_sum{0};
    std::uint64_t split_left_mnk_sum{0};
    std::uint64_t split_right_mnk_sum{0};
    int split_left_max_m{0};
    int split_left_max_n{0};
    int split_left_max_k{0};
    int split_right_max_m{0};
    int split_right_max_n{0};
    int split_right_max_k{0};
};

struct ActiveRootSelection {
    int *indices{nullptr};
    int count{0};
};

struct MergeLevelTimings {
    float values_solve_ms{0.0f};
    float values_active_roots_ms{0.0f};
    float values_deflated_boundary_ms{0.0f};
    float values_copyd_ms{0.0f};
    float values_secular_solve_ms{0.0f};
    float values_rescale_ms{0.0f};
    float values_boundary_prepare_ms{0.0f};
    float values_boundary_update_ms{0.0f};
    float rescale_ms{0.0f};
    float copyc_ms{0.0f};
    float reshufflec_ms{0.0f};
    float vector_prepare_ms{0.0f};
    float gemm_ms{0.0f};
    float update_ms{0.0f};
    float global_copyc_ms{0.0f};
    float global_reshufflec_ms{0.0f};
    float deflated_gather_ms{0.0f};
    float split_make_indices_ms{0.0f};
    float split_left_pack_ms{0.0f};
    float split_left_gemm_ms{0.0f};
    float split_right_pack_ms{0.0f};
    float split_right_gemm_ms{0.0f};
    float merge_update_ms{0.0f};
    float update_all_vectors_ms{0.0f};
};

enum class MergeTimingSlot {
    ValuesSolve = 0,
    ValuesActiveRoots,
    ValuesDeflatedBoundary,
    ValuesCopyD,
    ValuesSecularSolve,
    ValuesRescale,
    ValuesBoundaryPrepare,
    ValuesBoundaryUpdate,
    Rescale,
    GlobalCopyC,
    GlobalReshuffleC,
    DeflatedGather,
    VectorPrepare,
    Gemm,
    MergeUpdate,
    UpdateAllVectors,
    Count,
};

struct SplitMergeVectorsTimings {
    float make_indices_ms{0.0f};
    float left_pack_ms{0.0f};
    float left_gemm_ms{0.0f};
    float right_pack_ms{0.0f};
    float right_gemm_ms{0.0f};
};

struct MergeLevelTimer {
    static constexpr int kCount = static_cast<int>(MergeTimingSlot::Count);

    bool enabled{false};
    cudaEvent_t start[kCount]{};
    cudaEvent_t stop[kCount]{};
    bool recorded[kCount]{};
    float accumulated_ms[kCount]{};

    evd_status init(evd_handle handle, const bool enable) {
        enabled = enable;
        if (!enabled) {
            return EVD_STATUS_SUCCESS;
        }
        for (int i = 0; i < kCount; ++i) {
            EVD_RETURN_IF_CUDA_ERROR(
                handle, "tridiag_solver.merge_timing_create_start",
                cudaEventCreateWithFlags(&start[i], cudaEventDefault));
            EVD_RETURN_IF_CUDA_ERROR(
                handle, "tridiag_solver.merge_timing_create_stop",
                cudaEventCreateWithFlags(&stop[i], cudaEventDefault));
        }
        return EVD_STATUS_SUCCESS;
    }

    void reset_level() {
        for (int i = 0; i < kCount; ++i) {
            recorded[i] = false;
            accumulated_ms[i] = 0.0f;
        }
    }

    evd_status begin(evd_handle handle, const MergeTimingSlot slot) {
        if (!enabled) {
            return EVD_STATUS_SUCCESS;
        }
        const int id = static_cast<int>(slot);
        recorded[id] = true;
        EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.merge_timing_begin",
                                 cudaEventRecord(start[id], handle->stream));
        return EVD_STATUS_SUCCESS;
    }

    evd_status end(evd_handle handle, const MergeTimingSlot slot) {
        if (!enabled) {
            return EVD_STATUS_SUCCESS;
        }
        const int id = static_cast<int>(slot);
        EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.merge_timing_end",
                                 cudaEventRecord(stop[id], handle->stream));
        return EVD_STATUS_SUCCESS;
    }

    evd_status add(evd_handle handle, const MergeTimingSlot slot) {
        if (!enabled) {
            return EVD_STATUS_SUCCESS;
        }
        const int id = static_cast<int>(slot);
        if (!recorded[id]) {
            return EVD_STATUS_SUCCESS;
        }
        EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.merge_timing_sync",
                                 cudaEventSynchronize(stop[id]));
        float milliseconds = 0.0f;
        EVD_RETURN_IF_CUDA_ERROR(
            handle, "tridiag_solver.merge_timing_elapsed",
            cudaEventElapsedTime(&milliseconds, start[id], stop[id]));
        accumulated_ms[id] += milliseconds;
        recorded[id] = false;
        return EVD_STATUS_SUCCESS;
    }

    evd_status elapsed(evd_handle handle, const MergeTimingSlot slot,
                       float *milliseconds) {
        if (milliseconds == nullptr) {
            return EVD_STATUS_INVALID_POINTER;
        }
        *milliseconds = 0.0f;
        if (!enabled) {
            return EVD_STATUS_SUCCESS;
        }
        const int id = static_cast<int>(slot);
        if (!recorded[id]) {
            *milliseconds = accumulated_ms[id];
            return EVD_STATUS_SUCCESS;
        }
        EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.merge_timing_sync",
                                 cudaEventSynchronize(stop[id]));
        float last_ms = 0.0f;
        EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.merge_timing_elapsed",
                                 cudaEventElapsedTime(&last_ms, start[id],
                                                      stop[id]));
        *milliseconds = accumulated_ms[id] + last_ms;
        return EVD_STATUS_SUCCESS;
    }

    evd_status collect(evd_handle handle, MergeLevelTimings *timings) {
        if (timings == nullptr) {
            return EVD_STATUS_INVALID_POINTER;
        }
        *timings = {};
        if (!enabled) {
            return EVD_STATUS_SUCCESS;
        }
        evd_status status = elapsed(handle, MergeTimingSlot::ValuesSolve,
                                    &timings->values_solve_ms);
        if (status != EVD_STATUS_SUCCESS) {
            return status;
        }
        status = elapsed(handle, MergeTimingSlot::ValuesActiveRoots,
                         &timings->values_active_roots_ms);
        if (status != EVD_STATUS_SUCCESS) {
            return status;
        }
        status = elapsed(handle, MergeTimingSlot::ValuesDeflatedBoundary,
                         &timings->values_deflated_boundary_ms);
        if (status != EVD_STATUS_SUCCESS) {
            return status;
        }
        status = elapsed(handle, MergeTimingSlot::ValuesCopyD,
                         &timings->values_copyd_ms);
        if (status != EVD_STATUS_SUCCESS) {
            return status;
        }
        status = elapsed(handle, MergeTimingSlot::ValuesSecularSolve,
                         &timings->values_secular_solve_ms);
        if (status != EVD_STATUS_SUCCESS) {
            return status;
        }
        status = elapsed(handle, MergeTimingSlot::ValuesRescale,
                         &timings->values_rescale_ms);
        if (status != EVD_STATUS_SUCCESS) {
            return status;
        }
        status = elapsed(handle, MergeTimingSlot::ValuesBoundaryPrepare,
                         &timings->values_boundary_prepare_ms);
        if (status != EVD_STATUS_SUCCESS) {
            return status;
        }
        status = elapsed(handle, MergeTimingSlot::ValuesBoundaryUpdate,
                         &timings->values_boundary_update_ms);
        if (status != EVD_STATUS_SUCCESS) {
            return status;
        }
        status =
            elapsed(handle, MergeTimingSlot::Rescale, &timings->rescale_ms);
        if (status != EVD_STATUS_SUCCESS) {
            return status;
        }
        status = elapsed(handle, MergeTimingSlot::GlobalCopyC,
                         &timings->global_copyc_ms);
        if (status != EVD_STATUS_SUCCESS) {
            return status;
        }
        status = elapsed(handle, MergeTimingSlot::GlobalReshuffleC,
                         &timings->global_reshufflec_ms);
        if (status != EVD_STATUS_SUCCESS) {
            return status;
        }
        status = elapsed(handle, MergeTimingSlot::DeflatedGather,
                         &timings->deflated_gather_ms);
        if (status != EVD_STATUS_SUCCESS) {
            return status;
        }
        status = elapsed(handle, MergeTimingSlot::VectorPrepare,
                         &timings->vector_prepare_ms);
        if (status != EVD_STATUS_SUCCESS) {
            return status;
        }
        status = elapsed(handle, MergeTimingSlot::Gemm, &timings->gemm_ms);
        if (status != EVD_STATUS_SUCCESS) {
            return status;
        }
        status = elapsed(handle, MergeTimingSlot::MergeUpdate,
                         &timings->merge_update_ms);
        if (status != EVD_STATUS_SUCCESS) {
            return status;
        }
        status = elapsed(handle, MergeTimingSlot::UpdateAllVectors,
                         &timings->update_all_vectors_ms);
        if (status != EVD_STATUS_SUCCESS) {
            return status;
        }
        timings->copyc_ms = timings->global_copyc_ms;
        timings->reshufflec_ms =
            timings->global_reshufflec_ms + timings->deflated_gather_ms;
        timings->update_ms =
            timings->merge_update_ms + timings->update_all_vectors_ms;
        return EVD_STATUS_SUCCESS;
    }

    ~MergeLevelTimer() {
        for (int i = 0; i < kCount; ++i) {
            if (start[i] != nullptr) {
                (void)cudaEventDestroy(start[i]);
            }
            if (stop[i] != nullptr) {
                (void)cudaEventDestroy(stop[i]);
            }
        }
    }
};

struct CudaElapsedTimer {
    bool enabled{false};
    cudaEvent_t start{nullptr};
    cudaEvent_t stop{nullptr};

    evd_status init(evd_handle handle, const bool enable) {
        enabled = enable;
        if (!enabled) {
            return EVD_STATUS_SUCCESS;
        }
        EVD_RETURN_IF_CUDA_ERROR(
            handle, "tridiag_solver.cuda_elapsed_create_start",
            cudaEventCreateWithFlags(&start, cudaEventDefault));
        EVD_RETURN_IF_CUDA_ERROR(
            handle, "tridiag_solver.cuda_elapsed_create_stop",
            cudaEventCreateWithFlags(&stop, cudaEventDefault));
        return EVD_STATUS_SUCCESS;
    }

    evd_status begin(evd_handle handle) {
        if (!enabled) {
            return EVD_STATUS_SUCCESS;
        }
        EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.cuda_elapsed_begin",
                                 cudaEventRecord(start, handle->stream));
        return EVD_STATUS_SUCCESS;
    }

    evd_status end(evd_handle handle, float *milliseconds) {
        if (!enabled) {
            return EVD_STATUS_SUCCESS;
        }
        if (milliseconds == nullptr) {
            return EVD_STATUS_INVALID_POINTER;
        }
        EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.cuda_elapsed_end",
                                 cudaEventRecord(stop, handle->stream));
        EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.cuda_elapsed_sync",
                                 cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.cuda_elapsed_elapsed",
                                 cudaEventElapsedTime(&elapsed, start, stop));
        *milliseconds += elapsed;
        return EVD_STATUS_SUCCESS;
    }

    ~CudaElapsedTimer() {
        if (start != nullptr) {
            (void)cudaEventDestroy(start);
        }
        if (stop != nullptr) {
            (void)cudaEventDestroy(stop);
        }
    }
};

std::size_t align_up(const std::size_t value, const std::size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

template <typename T>
T *take_scratch(char *base, std::size_t *offset, const std::size_t count) {
    *offset = align_up(*offset, alignof(T));
    T *ptr = reinterpret_cast<T *>(base + *offset);
    *offset += sizeof(T) * count;
    return ptr;
}

std::size_t find_strided_batch_end(const std::vector<MergeVectorsTask> &tasks,
                                   const std::size_t begin) {
    const MergeVectorsTask &task = tasks[begin];
    std::size_t end = begin + 1;
    while (end < tasks.size()) {
        const MergeVectorsTask &next = tasks[end];
        const MergeVectorsTask &prev = tasks[end - 1];
        if (next.size != task.size || next.dd != task.dd ||
            next.start != prev.start + task.size) {
            break;
        }
        ++end;
    }
    return end;
}

MergeVectorsDispatchStats
summarize_merge_vectors_dispatch(const int n,
                                 const std::vector<MergeVectorsTask> &tasks) {
    MergeVectorsDispatchStats stats{};
    stats.task_count = static_cast<int>(tasks.size());
    for (const MergeVectorsTask &task : tasks) {
        stats.deflated_tasks += task.no_deflation() ? 0 : 1;
        stats.no_deflation_tasks += task.no_deflation() ? 1 : 0;
        stats.max_size = std::max(stats.max_size, task.size);
        stats.max_dd = std::max(stats.max_dd, task.dd);
        if (use_split_merge_vectors_task(n, task)) {
            const int left_rows = task.split - task.start;
            const int right_rows = task.size - left_rows;
            const std::uint64_t left_mnk =
                static_cast<std::uint64_t>(left_rows) *
                static_cast<std::uint64_t>(task.left_count) *
                static_cast<std::uint64_t>(task.dd);
            const std::uint64_t right_mnk =
                static_cast<std::uint64_t>(right_rows) *
                static_cast<std::uint64_t>(task.right_count) *
                static_cast<std::uint64_t>(task.dd);
            stats.split_left_mnk_sum += left_mnk;
            stats.split_right_mnk_sum += right_mnk;
            stats.mnk_sum += left_mnk + right_mnk;
            if (left_mnk >
                static_cast<std::uint64_t>(stats.split_left_max_m) *
                    static_cast<std::uint64_t>(stats.split_left_max_n) *
                    static_cast<std::uint64_t>(stats.split_left_max_k)) {
                stats.split_left_max_m = left_rows;
                stats.split_left_max_n = task.dd;
                stats.split_left_max_k = task.left_count;
            }
            if (right_mnk >
                static_cast<std::uint64_t>(stats.split_right_max_m) *
                    static_cast<std::uint64_t>(stats.split_right_max_n) *
                    static_cast<std::uint64_t>(stats.split_right_max_k)) {
                stats.split_right_max_m = right_rows;
                stats.split_right_max_n = task.dd;
                stats.split_right_max_k = task.right_count;
            }
        } else {
            stats.mnk_sum += static_cast<std::uint64_t>(task.size) *
                             static_cast<std::uint64_t>(task.dd) *
                             static_cast<std::uint64_t>(task.dd);
        }
    }

    for (std::size_t i = 0; i < tasks.size();) {
        const std::size_t end = find_strided_batch_end(tasks, i);
        const int batch_count = static_cast<int>(end - i);
        if (batch_count > 1) {
            ++stats.batch_groups;
            stats.batched_tasks += batch_count;
        } else {
            ++stats.single_tasks;
        }
        i = end;
    }
    return stats;
}

__global__ void make_merge_vector_task_candidates_kernel(
    const int n, int *splits, MergeVectorsTask *candidates, int *flags) {
    const int gid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (gid >= n) {
        return;
    }

    int *nps = ptr_nps(n, splits);
    int *nsz = ptr_nsz(n, splits);
    int *ndd = ptr_ndd(n, splits);
    int *em = ptr_em(n, splits);
    int *bsz = ptr_bsz(n, splits);
    int *bps = ptr_bps(n, splits);
    int *map = ptr_map(n, splits);
    const int start = nps[gid];
    const int size = nsz[gid];
    const int dd = ndd[gid];
    const bool is_task = gid == start && size > 0 && dd > 0;
    int split = start + size;
    int left_count = dd;
    int right_count = 0;
    if (is_task) {
        const int merge = em[gid];
        split = (merge >= 0) ? bps[2 * merge + 1] : start + size;
        if (split <= start || split >= start + size) {
            const int left_size = (merge >= 0) ? bsz[2 * merge] : size;
            split = start + left_size;
        }
        left_count = 0;
        right_count = 0;
        for (int i = 0; i < dd; ++i) {
            const int source_col = map[start + i];
            if (source_col < split) {
                ++left_count;
            } else {
                ++right_count;
            }
        }
    }
    candidates[gid] = {start, size, dd, split, left_count, right_count};
    flags[gid] = is_task ? 1 : 0;
}

__global__ void make_active_root_candidates_kernel(const int n, int *splits,
                                                   int *candidates,
                                                   int *flags) {
    const int gid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (gid >= n) {
        return;
    }

    int *nps = ptr_nps(n, splits);
    int *ndd = ptr_ndd(n, splits);
    const int start = nps[gid];
    const int dd = ndd[gid];
    const int local = gid - start;
    candidates[gid] = gid;
    flags[gid] = (local >= 0 && local < dd) ? 1 : 0;
}

evd_status build_active_root_indices_cub(evd_handle handle, const int n,
                                         double *boundary_scratch,
                                         int *splits,
                                         ActiveRootSelection *selection) {
    if (handle == nullptr || boundary_scratch == nullptr || splits == nullptr ||
        selection == nullptr) {
        return EVD_STATUS_INVALID_POINTER;
    }

    constexpr std::size_t boundary_scratch_rows = 2;
    char *scratch = reinterpret_cast<char *>(
        boundary_scratch + boundary_scratch_rows * static_cast<std::size_t>(n));
    const std::size_t scratch_bytes =
        (get_values_only_tempgemm_scratch_size(n) -
         boundary_scratch_rows * static_cast<std::size_t>(n)) *
        sizeof(double);
    std::size_t offset = 0;
    int *candidates =
        take_scratch<int>(scratch, &offset, static_cast<std::size_t>(n));
    int *selected =
        take_scratch<int>(scratch, &offset, static_cast<std::size_t>(n));
    int *flags =
        take_scratch<int>(scratch, &offset, static_cast<std::size_t>(n));
    int *selected_count = take_scratch<int>(scratch, &offset, 1);
    offset = align_up(offset, 256);
    if (offset > scratch_bytes) {
        return EVD_STATUS_INVALID_WORKSPACE;
    }

    std::size_t select_temp_bytes = 0;
    cudaError_t cub_status = cub::DeviceSelect::Flagged(
        nullptr, select_temp_bytes, candidates, flags, selected, selected_count,
        n, handle->stream);
    if (cub_status != cudaSuccess) {
        EVD_RETURN_IF_CUDA_ERROR(handle,
                                 "tridiag_solver.active_roots_cub_query",
                                 cub_status);
    }
    if (offset + select_temp_bytes > scratch_bytes) {
        return EVD_STATUS_INVALID_WORKSPACE;
    }
    void *select_temp_storage = scratch + offset;

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    make_active_root_candidates_kernel<<<blocks, threads, 0, handle->stream>>>(
        n, splits, candidates, flags);
    EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.active_roots_make",
                             cudaGetLastError());

    cub_status = cub::DeviceSelect::Flagged(
        select_temp_storage, select_temp_bytes, candidates, flags, selected,
        selected_count, n, handle->stream);
    if (cub_status != cudaSuccess) {
        EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.active_roots_cub",
                                 cub_status);
    }

    int count = 0;
    EVD_RETURN_IF_CUDA_ERROR(
        handle, "tridiag_solver.active_roots_copy_count",
        cudaMemcpyAsync(&count, selected_count, sizeof(int),
                        cudaMemcpyDeviceToHost, handle->stream));
    EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.active_roots_sync",
                             cudaStreamSynchronize(handle->stream));
    if (count < 0 || count > n) {
        return EVD_STATUS_INTERNAL_ERROR;
    }

    selection->indices = selected;
    selection->count = count;
    return EVD_STATUS_SUCCESS;
}

evd_status
build_merge_vectors_tasks_cub(evd_handle handle, const int n,
                              const int max_tasks,
                              const TridiagSolverWorkspaceView &view,
                              std::vector<MergeVectorsTask> *tasks_out) {
    if (handle == nullptr || view.d_tempgemm == nullptr ||
        view.d_splits == nullptr || tasks_out == nullptr) {
        return EVD_STATUS_INVALID_POINTER;
    }
    if (max_tasks < 0) {
        return EVD_STATUS_INVALID_VALUE;
    }

    tasks_out->clear();
    char *scratch = reinterpret_cast<char *>(view.d_tempgemm);
    const std::size_t scratch_bytes = get_tempgemm_size(n) * sizeof(double);
    std::size_t offset = 0;
    MergeVectorsTask *candidates = take_scratch<MergeVectorsTask>(
        scratch, &offset, static_cast<std::size_t>(n));
    MergeVectorsTask *selected = take_scratch<MergeVectorsTask>(
        scratch, &offset, static_cast<std::size_t>(n));
    int *flags =
        take_scratch<int>(scratch, &offset, static_cast<std::size_t>(n));
    int *selected_count = take_scratch<int>(scratch, &offset, 1);
    offset = align_up(offset, 256);
    if (offset > scratch_bytes) {
        return EVD_STATUS_INVALID_WORKSPACE;
    }

    std::size_t select_temp_bytes = 0;
    cudaError_t cub_status = cub::DeviceSelect::Flagged(
        nullptr, select_temp_bytes, candidates, flags, selected, selected_count,
        n, handle->stream);
    if (cub_status != cudaSuccess) {
        EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.merge_tasks_cub_query",
                                 cub_status);
    }
    if (offset + select_temp_bytes > scratch_bytes) {
        return EVD_STATUS_INVALID_WORKSPACE;
    }
    void *select_temp_storage = scratch + offset;

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    make_merge_vector_task_candidates_kernel<<<blocks, threads, 0,
                                               handle->stream>>>(
        n, view.d_splits, candidates, flags);
    EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.merge_tasks_make",
                             cudaGetLastError());

    cub_status = cub::DeviceSelect::Flagged(
        select_temp_storage, select_temp_bytes, candidates, flags, selected,
        selected_count, n, handle->stream);
    if (cub_status != cudaSuccess) {
        EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.merge_tasks_cub",
                                 cub_status);
    }

    int count = 0;
    tasks_out->resize(static_cast<std::size_t>(max_tasks));
    EVD_RETURN_IF_CUDA_ERROR(
        handle, "tridiag_solver.merge_tasks_copy_count",
        cudaMemcpyAsync(&count, selected_count, sizeof(int),
                        cudaMemcpyDeviceToHost, handle->stream));
    if (max_tasks > 0) {
        EVD_RETURN_IF_CUDA_ERROR(
            handle, "tridiag_solver.merge_tasks_copy_tasks",
            cudaMemcpyAsync(tasks_out->data(), selected,
                            sizeof(MergeVectorsTask) *
                                static_cast<std::size_t>(max_tasks),
                            cudaMemcpyDeviceToHost, handle->stream));
    }
    EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.merge_tasks_sync",
                             cudaStreamSynchronize(handle->stream));
    if (count < 0 || count > max_tasks) {
        return EVD_STATUS_INTERNAL_ERROR;
    }
    tasks_out->resize(static_cast<std::size_t>(count));
    return EVD_STATUS_SUCCESS;
}

evd_status
launch_merge_vectors_gemm_task(evd_handle handle, const int n,
                               const TridiagSolverWorkspaceView &view,
                               const MergeVectorsTask &task);

__global__ void make_merge_vector_side_indices_kernel(
    const int n, const int start, const int dd, const int split, const int *map,
    int *left_indices, int *right_indices) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    int left = 0;
    int right = 0;
    for (int i = 0; i < dd; ++i) {
        const int source_col = map[start + i];
        if (source_col < split) {
            left_indices[left++] = i;
        } else {
            right_indices[right++] = i;
        }
    }
}

__global__ void pack_merge_vector_side_kernel(
    const int n, const int start, const int row_start, const int rows,
    const int dd, const int side_count, const int *indices, const double *V,
    const int ldv, const int *map, const bool use_mapped_source_columns,
    const double *etmpd, double *packed_a, double *packed_b) {
    const std::size_t total_a =
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(side_count);
    const std::size_t total_b =
        static_cast<std::size_t>(side_count) * static_cast<std::size_t>(dd);
    const std::size_t total = total_a + total_b;
    const std::size_t gid =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= total) {
        return;
    }

    if (gid < total_a) {
        const int row = static_cast<int>(gid % static_cast<std::size_t>(rows));
        const int side_col =
            static_cast<int>(gid / static_cast<std::size_t>(rows));
        const int active_col = indices[side_col];
        const int source_col = use_mapped_source_columns
                                   ? map[start + active_col]
                                   : start + active_col;
        packed_a[gid] = V[dense_index(row_start + row, source_col, ldv)];
        return;
    }

    const std::size_t b_gid = gid - total_a;
    const int side_row =
        static_cast<int>(b_gid % static_cast<std::size_t>(side_count));
    const int col =
        static_cast<int>(b_gid / static_cast<std::size_t>(side_count));
    const int active_row = indices[side_row];
    packed_b[b_gid] = etmpd[dense_index(active_row, start + col, n)];
}

__global__ void gather_deflated_merge_vector_columns_kernel(
    const int n, const int start, const int size, const int dd, const int *map,
    const double *V, const int ldv, double *vecs) {
    const int local_col = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    if (local_col < dd || local_col >= size) {
        return;
    }

    const int dst_col = start + local_col;
    const int src_col = map[dst_col];
    for (int row = start + tid; row < start + size; row += blockDim.x) {
        vecs[dense_index(row, dst_col, n)] = V[dense_index(row, src_col, ldv)];
    }
}

__global__ void scatter_deflated_merge_vector_columns_kernel(
    const int n, const int start, const int size, const int dd,
    const double *vecs, double *V, const int ldv) {
    const int local_col = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    if (local_col < dd || local_col >= size) {
        return;
    }

    const int col = start + local_col;
    for (int row = start + tid; row < start + size; row += blockDim.x) {
        V[dense_index(row, col, ldv)] = vecs[dense_index(row, col, n)];
    }
}

evd_status
gather_deflated_merge_vector_columns(evd_handle handle, const int n,
                                     const TridiagSolverWorkspaceView &view,
                                     const MergeVectorsTask &task) {
    if (task.dd >= task.size) {
        return EVD_STATUS_SUCCESS;
    }
    int *map = ptr_map(n, view.d_splits);
    gather_deflated_merge_vector_columns_kernel<<<task.size, 256, 0,
                                                  handle->stream>>>(
        n, task.start, task.size, task.dd, map, view.d_tridiag_eigenvectors, n,
        tempgemm_vecs(n, view));
    EVD_RETURN_IF_CUDA_ERROR(handle,
                             "tridiag_solver.merge_vectors.gather_deflated",
                             cudaGetLastError());
    return EVD_STATUS_SUCCESS;
}

evd_status
scatter_deflated_merge_vector_columns(evd_handle handle, const int n,
                                      const TridiagSolverWorkspaceView &view,
                                      const MergeVectorsTask &task) {
    if (task.dd >= task.size) {
        return EVD_STATUS_SUCCESS;
    }
    scatter_deflated_merge_vector_columns_kernel<<<task.size, 256, 0,
                                                   handle->stream>>>(
        n, task.start, task.size, task.dd, tempgemm_vecs(n, view),
        view.d_tridiag_eigenvectors, n);
    EVD_RETURN_IF_CUDA_ERROR(handle,
                             "tridiag_solver.merge_vectors.scatter_deflated",
                             cudaGetLastError());
    return EVD_STATUS_SUCCESS;
}

evd_status launch_merge_vectors_split_side(
    evd_handle handle, const int n, const int start, const int row_start,
    const int rows, const int dd, const int side_count, const int *indices,
    const int *map, const bool use_mapped_source_columns, const double *V,
    const double *etmpd, double *output, const int ld_output, double *scratch,
    const std::size_t scratch_doubles, const bool is_left_side,
    SplitMergeVectorsTimings *split_timings) {
    if (side_count <= 0 || rows <= 0) {
        return EVD_STATUS_SUCCESS;
    }
    const std::size_t a_elems =
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(side_count);
    const std::size_t b_elems =
        static_cast<std::size_t>(side_count) * static_cast<std::size_t>(dd);
    if (a_elems + b_elems > scratch_doubles) {
        return EVD_STATUS_INVALID_WORKSPACE;
    }

    double *packed_a = scratch;
    double *packed_b = scratch + a_elems;
    const std::size_t pack_total = a_elems + b_elems;
    const int threads = 256;
    const int blocks = static_cast<int>((pack_total + threads - 1) / threads);
    CudaElapsedTimer pack_timer;
    evd_status timer_status = pack_timer.init(handle, split_timings != nullptr);
    if (timer_status != EVD_STATUS_SUCCESS) {
        return timer_status;
    }
    timer_status = pack_timer.begin(handle);
    if (timer_status != EVD_STATUS_SUCCESS) {
        return timer_status;
    }
    pack_merge_vector_side_kernel<<<blocks, threads, 0, handle->stream>>>(
        n, start, row_start, rows, dd, side_count, indices, V, n, map,
        use_mapped_source_columns, etmpd, packed_a, packed_b);
    EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.merge_vectors_pack_side",
                             cudaGetLastError());
    timer_status = pack_timer.end(
        handle, split_timings == nullptr
                    ? nullptr
                    : (is_left_side ? &split_timings->left_pack_ms
                                    : &split_timings->right_pack_ms));
    if (timer_status != EVD_STATUS_SUCCESS) {
        return timer_status;
    }

    const double alpha = 1.0;
    const double beta = 0.0;
    CudaElapsedTimer gemm_timer;
    timer_status = gemm_timer.init(handle, split_timings != nullptr);
    if (timer_status != EVD_STATUS_SUCCESS) {
        return timer_status;
    }
    timer_status = gemm_timer.begin(handle);
    if (timer_status != EVD_STATUS_SUCCESS) {
        return timer_status;
    }
    const cublasStatus_t status = cublasDgemm_64(
        handle->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
        static_cast<int64_t>(rows), static_cast<int64_t>(dd),
        static_cast<int64_t>(side_count), &alpha, packed_a,
        static_cast<int64_t>(rows), packed_b, static_cast<int64_t>(side_count),
        &beta, output + dense_index(row_start, start, ld_output),
        static_cast<int64_t>(ld_output));
    EVD_RETURN_IF_CUBLAS_ERROR(
        handle, "tridiag_solver.merge_vectors.split_gemm", status);
    timer_status = gemm_timer.end(
        handle, split_timings == nullptr
                    ? nullptr
                    : (is_left_side ? &split_timings->left_gemm_ms
                                    : &split_timings->right_gemm_ms));
    if (timer_status != EVD_STATUS_SUCCESS) {
        return timer_status;
    }
    return EVD_STATUS_SUCCESS;
}

evd_status launch_merge_vectors_split_task(
    evd_handle handle, const int n, const TridiagSolverWorkspaceView &view,
    const MergeVectorsTask &task, const bool use_mapped_source_columns,
    const bool direct_write_active_vectors,
    SplitMergeVectorsTimings *split_timings) {
    if (view.d_split_gemm_scratch == nullptr) {
        return launch_merge_vectors_gemm_task(handle, n, view, task);
    }
    const int left_rows = task.split - task.start;
    const int right_rows = task.size - left_rows;
    if (left_rows <= 0 || right_rows <= 0 || task.left_count <= 0 ||
        task.right_count <= 0 ||
        task.left_count + task.right_count != task.dd) {
        return launch_merge_vectors_gemm_task(handle, n, view, task);
    }

    char *base = reinterpret_cast<char *>(view.d_split_gemm_scratch);
    std::size_t offset = 0;
    int *left_indices =
        take_scratch<int>(base, &offset, static_cast<std::size_t>(task.dd));
    int *right_indices =
        take_scratch<int>(base, &offset, static_cast<std::size_t>(task.dd));
    // Keep packed GEMM operands cacheline-aligned. The split scratch starts
    // after int index buffers, so alignof(double) can leave packed_a only
    // 8-byte aligned and make cuBLAS choose weaker H100 DGEMM kernels.
    offset = align_up(offset, 128);
    double *scratch = reinterpret_cast<double *>(base + offset);
    const std::size_t scratch_bytes = static_cast<std::size_t>(n) *
                                      static_cast<std::size_t>(n) *
                                      sizeof(double);
    if (offset > scratch_bytes) {
        return EVD_STATUS_INVALID_WORKSPACE;
    }
    const std::size_t scratch_doubles =
        (scratch_bytes - offset) / sizeof(double);

    int *map = ptr_map(n, view.d_splits);
    CudaElapsedTimer indices_timer;
    evd_status timer_status =
        indices_timer.init(handle, split_timings != nullptr);
    if (timer_status != EVD_STATUS_SUCCESS) {
        return timer_status;
    }
    timer_status = indices_timer.begin(handle);
    if (timer_status != EVD_STATUS_SUCCESS) {
        return timer_status;
    }
    make_merge_vector_side_indices_kernel<<<1, 1, 0, handle->stream>>>(
        n, task.start, task.dd, task.split, map, left_indices, right_indices);
    EVD_RETURN_IF_CUDA_ERROR(handle,
                             "tridiag_solver.merge_vectors.make_side_indices",
                             cudaGetLastError());
    timer_status = indices_timer.end(
        handle,
        split_timings == nullptr ? nullptr : &split_timings->make_indices_ms);
    if (timer_status != EVD_STATUS_SUCCESS) {
        return timer_status;
    }

    const double *V = view.d_tridiag_eigenvectors;
    const double *etmpd = tempgemm_etmpd(n, view);
    double *output = direct_write_active_vectors ? view.d_tridiag_eigenvectors
                                                 : tempgemm_vecs(n, view);
    const int ld_output = n;

    evd_status status = launch_merge_vectors_split_side(
        handle, n, task.start, task.start, left_rows, task.dd, task.left_count,
        left_indices, map, use_mapped_source_columns, V, etmpd, output,
        ld_output, scratch, scratch_doubles, true, split_timings);
    if (status != EVD_STATUS_SUCCESS) {
        return status;
    }
    status = launch_merge_vectors_split_side(
        handle, n, task.start, task.split, right_rows, task.dd,
        task.right_count, right_indices, map, use_mapped_source_columns, V,
        etmpd, output, ld_output, scratch, scratch_doubles, false,
        split_timings);
    if (status != EVD_STATUS_SUCCESS) {
        return status;
    }
    return EVD_STATUS_SUCCESS;
}

evd_status
launch_merge_vectors_gemm_task(evd_handle handle, const int n,
                               const TridiagSolverWorkspaceView &view,
                               const MergeVectorsTask &task) {
    const double alpha = 1.0;
    const double beta = 0.0;
    const double *a =
        view.d_tridiag_eigenvectors + dense_index(task.start, task.start, n);
    const double *b = tempgemm_etmpd(n, view) + dense_index(0, task.start, n);
    double *c = tempgemm_vecs(n, view) + dense_index(task.start, task.start, n);
    const cublasStatus_t status = cublasDgemm_64(
        handle->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
        static_cast<int64_t>(task.size), static_cast<int64_t>(task.dd),
        static_cast<int64_t>(task.dd), &alpha, a, static_cast<int64_t>(n), b,
        static_cast<int64_t>(n), &beta, c, static_cast<int64_t>(n));
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr,
                     "merge_vectors.gemm failed: cublasStatus=%d n=%d start=%d "
                     "size=%d dd=%d lda=%d\n",
                     static_cast<int>(status), n, task.start, task.size,
                     task.dd, n);
        EVD_RETURN_IF_CUBLAS_ERROR(handle, "tridiag_solver.merge_vectors.gemm",
                                   status);
    }
    return EVD_STATUS_SUCCESS;
}

evd_status launch_merge_vectors_strided_batched(
    evd_handle handle, const int n, const TridiagSolverWorkspaceView &view,
    const std::vector<MergeVectorsTask> &tasks, const std::size_t begin,
    const int batch_count) {
    const MergeVectorsTask &task = tasks[begin];
    const double alpha = 1.0;
    const double beta = 0.0;
    const double *a =
        view.d_tridiag_eigenvectors + dense_index(task.start, task.start, n);
    const double *b = tempgemm_etmpd(n, view) + dense_index(0, task.start, n);
    double *c = tempgemm_vecs(n, view) + dense_index(task.start, task.start, n);
    const long long stride_a =
        static_cast<long long>(task.size) * static_cast<long long>(n + 1);
    const long long stride_b =
        static_cast<long long>(task.size) * static_cast<long long>(n);
    const long long stride_c = stride_a;
    const cublasStatus_t batched_status = cublasDgemmStridedBatched_64(
        handle->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
        static_cast<int64_t>(task.size), static_cast<int64_t>(task.dd),
        static_cast<int64_t>(task.dd), &alpha, a, static_cast<int64_t>(n),
        stride_a, b, static_cast<int64_t>(n), stride_b, &beta, c,
        static_cast<int64_t>(n), stride_c, static_cast<int64_t>(batch_count));
    if (batched_status == CUBLAS_STATUS_SUCCESS) {
        return EVD_STATUS_SUCCESS;
    }

    // cuBLAS 12.x can return INTERNAL_ERROR for large-stride batched DGEMM
    // even though the equivalent individual GEMMs are valid.
    for (int batch = 0; batch < batch_count; ++batch) {
        const evd_status status = launch_merge_vectors_gemm_task(
            handle, n, view, tasks[begin + static_cast<std::size_t>(batch)]);
        if (status != EVD_STATUS_SUCCESS) {
            return status;
        }
    }
    return EVD_STATUS_SUCCESS;
}

evd_status
launch_merge_vectors_cublas(evd_handle handle, const int n,
                            const TridiagSolverWorkspaceView &view,
                            const std::vector<MergeVectorsTask> &tasks,
                            const bool use_mapped_source_columns,
                            const bool direct_write_active_vectors,
                            SplitMergeVectorsTimings *split_timings) {
    if (tasks.empty()) {
        return EVD_STATUS_SUCCESS;
    }

    for (std::size_t i = 0; i < tasks.size();) {
        if (use_split_merge_vectors_task(n, tasks[i])) {
            const evd_status status = launch_merge_vectors_split_task(
                handle, n, view, tasks[i], use_mapped_source_columns,
                direct_write_active_vectors, split_timings);
            if (status != EVD_STATUS_SUCCESS) {
                return status;
            }
            ++i;
            continue;
        }

        const std::size_t j = find_strided_batch_end(tasks, i);
        const int batch_count = static_cast<int>(j - i);
        if (batch_count > 1) {
            const evd_status status = launch_merge_vectors_strided_batched(
                handle, n, view, tasks, i, batch_count);
            if (status != EVD_STATUS_SUCCESS) {
                return status;
            }
            i = j;
            continue;
        }

        const evd_status status =
            launch_merge_vectors_gemm_task(handle, n, view, tasks[i]);
        if (status != EVD_STATUS_SUCCESS) {
            return status;
        }
        ++i;
    }

    return EVD_STATUS_SUCCESS;
}

struct MergeDdSummary {
    int observed_merges{0};
    int size_min{0};
    int size_max{0};
    int size_sum{0};
    int dd_min{0};
    int dd_max{0};
    int dd_sum{0};
};

struct SecularIterationSummary {
    int count{0};
    int sum{0};
    int max{0};
    int failed{0};
};

MergeDdSummary summarize_merge_dd(const std::vector<int> &nps,
                                  const std::vector<int> &nsz,
                                  const std::vector<int> &ndd) {
    MergeDdSummary summary{};
    int size_min = INT_MAX;
    int dd_min = INT_MAX;
    std::unordered_set<int> seen_starts;
    seen_starts.reserve(nps.size());
    for (std::size_t i = 0; i < nps.size() && i < nsz.size() && i < ndd.size();
         ++i) {
        if (!seen_starts.insert(nps[i]).second) {
            continue;
        }

        const int size = nsz[i];
        const int dd = ndd[i];
        size_min = std::min(size_min, size);
        summary.size_max = std::max(summary.size_max, size);
        summary.size_sum += size;
        dd_min = std::min(dd_min, dd);
        summary.dd_max = std::max(summary.dd_max, dd);
        summary.dd_sum += dd;
        ++summary.observed_merges;
    }

    summary.size_min = summary.observed_merges == 0 ? 0 : size_min;
    summary.dd_min = summary.observed_merges == 0 ? 0 : dd_min;
    return summary;
}

void append_merge_level_diagnostics(
    evd_handle handle, const int level, const int merge_count,
    const std::vector<int> &nps, const std::vector<int> &nsz,
    const std::vector<int> &dd_before, const std::vector<int> &dd_after,
    const std::vector<int> *dbg_after,
    const MergeVectorsDispatchStats *vector_stats,
    const MergeLevelTimings *timings) {
    if (handle == nullptr || handle->last_tridiag_diagnostics.enabled == 0) {
        return;
    }

    const MergeDdSummary before = summarize_merge_dd(nps, nsz, dd_before);
    const MergeDdSummary after = summarize_merge_dd(nps, nsz, dd_after);

    evd_handle_struct::TridiagMergeLevelDiagnostics diag{};
    diag.level = level;
    diag.merge_count = merge_count;
    diag.observed_merges = after.observed_merges;
    diag.size_min = after.size_min;
    diag.size_max = after.size_max;
    diag.size_sum = after.size_sum;
    diag.dd_before_min = before.dd_min;
    diag.dd_before_max = before.dd_max;
    diag.dd_before_sum = before.dd_sum;
    diag.dd_after_min = after.dd_min;
    diag.dd_after_max = after.dd_max;
    diag.dd_after_sum = after.dd_sum;
    if (dbg_after != nullptr) {
        SecularIterationSummary iter_summary{};
        const std::size_t entry_count =
            std::min({dbg_after->size(), nps.size(), dd_after.size()});
        for (std::size_t idx = 0; idx < entry_count; ++idx) {
            const int local = static_cast<int>(idx) - nps[idx];
            if (local < 0 || local >= dd_after[idx]) {
                continue;
            }
            const int value = (*dbg_after)[idx];
            if (value == 0) {
                continue;
            }
            const int iterations = std::abs(value);
            ++iter_summary.count;
            iter_summary.sum += iterations;
            iter_summary.max = std::max(iter_summary.max, iterations);
            if (value < 0) {
                ++iter_summary.failed;
            }
        }
        diag.secular_iter_count = iter_summary.count;
        diag.secular_iter_sum = iter_summary.sum;
        diag.secular_iter_max = iter_summary.max;
        diag.secular_iter_failed = iter_summary.failed;
    }
    diag.zero_deflated_total = after.size_sum - before.dd_sum;
    diag.close_deflated_total = before.dd_sum - after.dd_sum;
    diag.deflated_total = after.size_sum - after.dd_sum;
    if (vector_stats != nullptr) {
        diag.vector_task_count = vector_stats->task_count;
        diag.vector_batch_groups = vector_stats->batch_groups;
        diag.vector_batched_tasks = vector_stats->batched_tasks;
        diag.vector_single_tasks = vector_stats->single_tasks;
        diag.vector_deflated_tasks = vector_stats->deflated_tasks;
        diag.vector_no_deflation_tasks = vector_stats->no_deflation_tasks;
        diag.vector_max_size = vector_stats->max_size;
        diag.vector_max_dd = vector_stats->max_dd;
        diag.vector_mnk_sum = vector_stats->mnk_sum;
        diag.split_left_mnk_sum = vector_stats->split_left_mnk_sum;
        diag.split_right_mnk_sum = vector_stats->split_right_mnk_sum;
        diag.split_left_max_m = vector_stats->split_left_max_m;
        diag.split_left_max_n = vector_stats->split_left_max_n;
        diag.split_left_max_k = vector_stats->split_left_max_k;
        diag.split_right_max_m = vector_stats->split_right_max_m;
        diag.split_right_max_n = vector_stats->split_right_max_n;
        diag.split_right_max_k = vector_stats->split_right_max_k;
    }
    if (timings != nullptr) {
        diag.values_solve_ms = timings->values_solve_ms;
        diag.values_active_roots_ms = timings->values_active_roots_ms;
        diag.values_deflated_boundary_ms =
            timings->values_deflated_boundary_ms;
        diag.values_copyd_ms = timings->values_copyd_ms;
        diag.values_secular_solve_ms = timings->values_secular_solve_ms;
        diag.values_rescale_ms = timings->values_rescale_ms;
        diag.values_boundary_prepare_ms =
            timings->values_boundary_prepare_ms;
        diag.values_boundary_update_ms = timings->values_boundary_update_ms;
        diag.rescale_ms = timings->rescale_ms;
        diag.copyc_ms = timings->copyc_ms;
        diag.reshufflec_ms = timings->reshufflec_ms;
        diag.vector_prepare_ms = timings->vector_prepare_ms;
        diag.gemm_ms = timings->gemm_ms;
        diag.update_ms = timings->update_ms;
        diag.global_copyc_ms = timings->global_copyc_ms;
        diag.global_reshufflec_ms = timings->global_reshufflec_ms;
        diag.deflated_gather_ms = timings->deflated_gather_ms;
        diag.split_make_indices_ms = timings->split_make_indices_ms;
        diag.split_left_pack_ms = timings->split_left_pack_ms;
        diag.split_left_gemm_ms = timings->split_left_gemm_ms;
        diag.split_right_pack_ms = timings->split_right_pack_ms;
        diag.split_right_gemm_ms = timings->split_right_gemm_ms;
        diag.merge_update_ms = timings->merge_update_ms;
        diag.update_all_vectors_ms = timings->update_all_vectors_ms;
    }
    handle->last_tridiag_diagnostics.levels.push_back(diag);
}

__global__ void init_sort_indices_kernel(const int n, int *indices) {
    const int gid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (gid < n) {
        indices[gid] = gid;
    }
}

__global__ void gather_sorted_vector_columns_kernel(
    const int n, const int *permutation, const double *source,
    const int ld_source, double *destination, const int ld_destination) {
    const int gid = static_cast<int>(blockIdx.x);
    if (gid >= n) {
        return;
    }

    const int source_col = permutation[gid];
    const double *src_ptr = source + static_cast<std::size_t>(source_col) *
                                         static_cast<std::size_t>(ld_source);
    double *dst_ptr =
        destination + static_cast<std::size_t>(gid) *
                          static_cast<std::size_t>(ld_destination);
    for (int row = threadIdx.x; row < n; row += blockDim.x) {
        dst_ptr[row] = src_ptr[row];
    }
}

__global__ void copy_device_tridiagonal_kernel(const int n,
                                               const double *source_diag,
                                               const double *source_subdiag,
                                               double *dest_diag,
                                               double *dest_subdiag) {
    const int gid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (gid < n) {
        dest_diag[gid] = source_diag[gid];
    }
    if (gid + 1 < n) {
        dest_subdiag[gid] = device_abs(source_subdiag[gid]);
    }
}

} // namespace

inline evd_status sort_eigenpairs_inplace(
    evd_handle handle, const int n, const TridiagSolverWorkspaceView &view,
    const evd_tridiag_sort_backend sort_backend, bool compute_vectors) {
    if (view.d_tempgemm == nullptr || view.d_etmpd == nullptr ||
        view.d_dc_diag == nullptr) {
        return EVD_STATUS_INVALID_POINTER;
    }
    double *sorted_vectors = tempgemm_vecs(n, view);
    double *sorted_values = tempgemm_etmpd(n, view);
    double *vec_in = (compute_vectors && view.d_tridiag_eigenvectors != nullptr)
                         ? view.d_tridiag_eigenvectors
                         : nullptr;
    double *vec_out =
        (compute_vectors && view.d_tridiag_eigenvectors != nullptr)
            ? sorted_vectors
            : nullptr;
    const int ldvec =
        (compute_vectors && view.d_tridiag_eigenvectors != nullptr) ? n : 1;

    const bool use_cub_sort = sort_backend != EVD_TRIDIAG_SORT_BACKEND_LEGACY;
    if (!use_cub_sort) {
        stedc_sort_pairs_kernel<<<n, 256, 0, handle->stream>>>(
            n, view.d_dc_diag, vec_in, ldvec, sorted_values, vec_out, ldvec);
        EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.sort_pairs",
                                 cudaGetLastError());
    } else {
        constexpr int kSortThreads = 256;
        const int sort_blocks = (n + kSortThreads - 1) / kSortThreads;
        if (view.d_temp_eigs == nullptr) {
            return EVD_STATUS_INVALID_POINTER;
        }

        // d_temp_eigs is sized as n doubles, which is exactly enough for two
        // n-element int buffers used by CUB SortPairs.
        int *index_in = reinterpret_cast<int *>(view.d_temp_eigs);
        int *index_out = index_in + n;
        init_sort_indices_kernel<<<sort_blocks, kSortThreads, 0,
                                   handle->stream>>>(n, index_in);
        EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.init_sort_indices",
                                 cudaGetLastError());

        void *sort_temp_storage = sorted_vectors;
        std::size_t sort_temp_storage_bytes = 0;
        cudaError_t cub_status = cub::DeviceRadixSort::SortPairs(
            nullptr, sort_temp_storage_bytes, view.d_dc_diag, sorted_values,
            index_in, index_out, n, 0, static_cast<int>(sizeof(double) * 8),
            handle->stream);
        if (cub_status != cudaSuccess) {
            EVD_RETURN_IF_CUDA_ERROR(
                handle, "tridiag_solver.sort_pairs_cub_query", cub_status);
        }

        const std::size_t available_sort_temp_bytes =
            (compute_vectors ? static_cast<std::size_t>(n) *
                                   static_cast<std::size_t>(n)
                             : get_values_only_tempgemm_scratch_size(n)) *
            sizeof(double);
        if (sort_temp_storage_bytes > available_sort_temp_bytes) {
            return EVD_STATUS_INVALID_WORKSPACE;
        }

        cub_status = cub::DeviceRadixSort::SortPairs(
            sort_temp_storage, sort_temp_storage_bytes, view.d_dc_diag,
            sorted_values, index_in, index_out, n, 0,
            static_cast<int>(sizeof(double) * 8), handle->stream);
        if (cub_status != cudaSuccess) {
            EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.sort_pairs_cub",
                                     cub_status);
        }

        if (compute_vectors && vec_in != nullptr && vec_out != nullptr) {
            gather_sorted_vector_columns_kernel<<<n, 256, 0, handle->stream>>>(
                n, index_out, vec_in, ldvec, vec_out, ldvec);
            EVD_RETURN_IF_CUDA_ERROR(handle,
                                     "tridiag_solver.gather_sorted_vectors",
                                     cudaGetLastError());
        }
    }

    EVD_RETURN_IF_CUDA_ERROR(
        handle, "tridiag_solver.copy_sorted_values",
        cudaMemcpyAsync(view.d_dc_diag, sorted_values, sizeof(double) * n,
                        cudaMemcpyDeviceToDevice, handle->stream));
    if (compute_vectors && view.d_tridiag_eigenvectors != nullptr) {
        stedc_copyc_kernel<<<n, 256, 0, handle->stream>>>(
            n, sorted_vectors, n, view.d_tridiag_eigenvectors, n);
        EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.copy_sorted_vectors",
                                 cudaGetLastError());
    }
    return EVD_STATUS_SUCCESS;
}

evd_status run_stedc_leaf_stage(evd_handle handle, const int n,
                                const TridiagSolverWorkspaceLayout &layout,
                                const TridiagSolverWorkspaceView &view,
                                bool compute_vectors) {
    if (layout.splits_bytes == 0 || view.d_dc_diag == nullptr ||
        view.d_dc_subdiag == nullptr || view.d_splits == nullptr ||
        view.d_tmpz == nullptr) {
        return EVD_STATUS_SUCCESS;
    }
    // V is required only for compute_vectors; boundary rows replace it
    // otherwise.
    if (compute_vectors && view.d_tridiag_eigenvectors == nullptr) {
        return EVD_STATUS_SUCCESS;
    }
    if (!compute_vectors && view.d_boundary_rows == nullptr &&
        view.d_tridiag_eigenvectors == nullptr) {
        return EVD_STATUS_SUCCESS;
    }

    const int levels = stedc_num_levels(n);
    const int blks = 1 << levels;
    stedc_divide_kernel<<<1, 1, 0, handle->stream>>>(
        blks, n, view.d_dc_diag, view.d_dc_subdiag, view.d_splits);
    EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.divide",
                             cudaGetLastError());

    stedc_leaf_solve_kernel<<<blks, kTridiagSolverLeafBlockSize, 0,
                              handle->stream>>>(
        blks, n, view.d_dc_diag, view.d_dc_subdiag, view.d_splits, view.d_tmpz,
        view.d_tridiag_eigenvectors, n, compute_vectors, view.d_boundary_rows);
    EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.leaf_solve",
                             cudaGetLastError());

    return EVD_STATUS_SUCCESS;
}

evd_status run_stedc_merge_levels(evd_handle handle, const int n,
                                  const TridiagSolverWorkspaceLayout &layout,
                                  const TridiagSolverWorkspaceView &view,
                                  int *summary_out, bool *all_blocks_taken_over,
                                  bool compute_vectors,
                                  bool enable_diagnostics) {
    // Values-only reuses a narrow etmpd slab across root chunks; the full
    // vector path keeps the column-major n-by-n secular-vector workspace.
    if (all_blocks_taken_over != nullptr) {
        *all_blocks_taken_over = false;
    }
    if (layout.splits_bytes == 0 || view.d_splits == nullptr ||
        view.d_tmpz == nullptr || view.d_dc_diag == nullptr ||
        view.d_dc_subdiag == nullptr || view.d_tempgemm == nullptr) {
        return EVD_STATUS_SUCCESS;
    }
    if (compute_vectors && view.d_etmpd == nullptr) {
        return EVD_STATUS_SUCCESS;
    }
    if (!compute_vectors && layout.etmpd_bytes > 0 &&
        view.d_etmpd == nullptr) {
        return EVD_STATUS_SUCCESS;
    }
    if (compute_vectors && view.d_tridiag_eigenvectors == nullptr) {
        return EVD_STATUS_SUCCESS;
    }
    if (!compute_vectors && view.d_boundary_rows == nullptr &&
        view.d_tridiag_eigenvectors == nullptr) {
        return EVD_STATUS_SUCCESS;
    }

    const int levels = stedc_num_levels(n);
    const double eps = std::numeric_limits<double>::epsilon();
    const double safemin = std::numeric_limits<double>::min();
    const double ssfmin = std::sqrt(safemin) / (eps * eps);
    const double ssfmax = std::sqrt(1.0 / safemin) / 3.0;
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    int total_merges = 0;
    std::vector<int> nps_host;
    std::vector<int> nsz_host;
    std::vector<int> dd_before_host;
    std::vector<int> dd_after_host;
    std::vector<int> dbg_host;
    if (enable_diagnostics) {
        nps_host.resize(static_cast<std::size_t>(n));
        nsz_host.resize(static_cast<std::size_t>(n));
        dd_before_host.resize(static_cast<std::size_t>(n));
        dd_after_host.resize(static_cast<std::size_t>(n));
        dbg_host.resize(static_cast<std::size_t>(n));
    }
    MergeLevelTimer merge_timer;
    {
        const evd_status timing_status =
            merge_timer.init(handle, enable_diagnostics);
        if (timing_status != EVD_STATUS_SUCCESS) {
            return timing_status;
        }
    }
    double *current_boundary_rows = view.d_boundary_rows;
    double *boundary_pingpong = tempgemm_vecs(n, view);
    for (int k = 0; k < levels; ++k) {
        merge_timer.reset_level();
        const int merge_count = 1 << (levels - k - 1);
        const int level_max_dd_upper =
            (n + merge_count - 1) / merge_count;
        total_merges += merge_count;
        const int split_block_count =
            (2 * merge_count + threads - 1) / threads;
        stedc_update_splits_copy_blocks_kernel<<<split_block_count, threads, 0,
                                                 handle->stream>>>(
            levels, k, n, view.d_splits);
        EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.update_splits_copy",
                                 cudaGetLastError());
        const int merge_block_count = (merge_count + threads - 1) / threads;
        stedc_update_splits_merge_blocks_kernel<<<merge_block_count, threads, 0,
                                                  handle->stream>>>(
            levels, k, n, view.d_splits);
        EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.update_splits",
                                 cudaGetLastError());
        stedc_update_splits_rows_kernel<<<blocks, threads, 0, handle->stream>>>(
            n, view.d_splits);
        EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.update_splits_rows",
                                 cudaGetLastError());

        if (compute_vectors) {
            stedc_merge_prepare_deflatezero_kernel<<<merge_count, 1, 0,
                                                     handle->stream>>>(
                merge_count, n, view.d_dc_diag, view.d_dc_subdiag,
                view.d_tridiag_eigenvectors, n, view.d_tmpz, view.d_splits,
                eps);
        } else {
            stedc_merge_prepare_deflatezero_boundary_kernel<256>
                <<<merge_count, 256, 0, handle->stream>>>(
                    merge_count, n, view.d_dc_diag, view.d_dc_subdiag,
                    current_boundary_rows, view.d_tmpz, view.d_splits, eps);
        }
        EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.merge_prepare_zero",
                                 cudaGetLastError());

        stedc_merge_sortd_kernel<<<blocks, threads, 0, handle->stream>>>(
            n, view.d_dc_diag, view.d_tmpz, view.d_splits);
        EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.merge_sortd",
                                 cudaGetLastError());
        if (enable_diagnostics) {
            EVD_RETURN_IF_CUDA_ERROR(
                handle, "tridiag_solver.diag_copy_nps",
                cudaMemcpyAsync(nps_host.data(), ptr_nps(n, view.d_splits),
                                sizeof(int) * static_cast<std::size_t>(n),
                                cudaMemcpyDeviceToHost, handle->stream));
            EVD_RETURN_IF_CUDA_ERROR(
                handle, "tridiag_solver.diag_copy_dd_before",
                cudaMemcpyAsync(dd_before_host.data(),
                                ptr_ndd(n, view.d_splits),
                                sizeof(int) * static_cast<std::size_t>(n),
                                cudaMemcpyDeviceToHost, handle->stream));
            EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.diag_sync_before",
                                     cudaStreamSynchronize(handle->stream));
        }

        stedc_merge_setcandflags_kernel<<<blocks, threads, 0, handle->stream>>>(
            n, view.d_tmpz, view.d_splits);
        EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.merge_setcand",
                                 cudaGetLastError());

        stedc_merge_deflatecount_kernel<<<blocks, threads, 0, handle->stream>>>(
            n, view.d_tmpz, view.d_splits);
        EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.merge_deflatecount",
                                 cudaGetLastError());

        stedc_merge_deflateapply_kernel<<<blocks, threads, 0, handle->stream>>>(
            n, view.d_tmpz, view.d_splits);
        EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.merge_deflateapply",
                                 cudaGetLastError());

        if (compute_vectors) {
            stedc_merge_rotate_kernel<<<n, 1, 0, handle->stream>>>(
                n, view.d_tmpz, view.d_splits, view.d_tridiag_eigenvectors, n);
            EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.merge_rotate",
                                     cudaGetLastError());
        } else {
            double *rotated_boundary_rows =
                (current_boundary_rows == boundary_pingpong) ? view.d_boundary_rows
                                                             : boundary_pingpong;
            stedc_project_boundary_rows_kernel<<<blocks, threads, 0,
                                                 handle->stream>>>(
                n, view.d_splits, current_boundary_rows, rotated_boundary_rows);
            EVD_RETURN_IF_CUDA_ERROR(
                handle, "tridiag_solver.project_boundary_rows",
                cudaGetLastError());
            stedc_rotate_boundary_rows_kernel<<<blocks, threads, 0,
                                                handle->stream>>>(
                n, view.d_tmpz, view.d_splits, rotated_boundary_rows);
            EVD_RETURN_IF_CUDA_ERROR(
                handle, "tridiag_solver.rotate_boundary_rows",
                cudaGetLastError());
            current_boundary_rows = rotated_boundary_rows;
        }

        stedc_merge_solvedz_kernel<<<blocks, threads, 0, handle->stream>>>(
            n, view.d_dc_diag, view.d_tmpz, view.d_splits);
        EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.merge_solvedz",
                                 cudaGetLastError());
        double *vecs = tempgemm_vecs(n, view);
        double *etmpd = tempgemm_etmpd(n, view);
        std::vector<MergeVectorsTask> vector_tasks;
        MergeVectorsDispatchStats vector_stats{};
        int level_max_dd = 0;
        SplitMergeVectorsTimings split_timings{};

        if (enable_diagnostics || compute_vectors) {
            nvtxRangePushA(enable_diagnostics ? "tridiag.hostSyncForMergeTasks"
                                              : "tridiag.cubBuildMergeTasks");
        }
        if (enable_diagnostics) {
            EVD_RETURN_IF_CUDA_ERROR(
                handle, "tridiag_solver.diag_copy_nps_after",
                cudaMemcpyAsync(nps_host.data(), ptr_nps(n, view.d_splits),
                                sizeof(int) * static_cast<std::size_t>(n),
                                cudaMemcpyDeviceToHost, handle->stream));
            EVD_RETURN_IF_CUDA_ERROR(
                handle, "tridiag_solver.diag_copy_dd_after",
                cudaMemcpyAsync(dd_after_host.data(), ptr_ndd(n, view.d_splits),
                                sizeof(int) * static_cast<std::size_t>(n),
                                cudaMemcpyDeviceToHost, handle->stream));
            EVD_RETURN_IF_CUDA_ERROR(
                handle, "tridiag_solver.diag_copy_nsz_after",
                cudaMemcpyAsync(nsz_host.data(), ptr_nsz(n, view.d_splits),
                                sizeof(int) * static_cast<std::size_t>(n),
                                cudaMemcpyDeviceToHost, handle->stream));
            EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.diag_sync_after",
                                     cudaStreamSynchronize(handle->stream));
            if (compute_vectors) {
                const evd_status task_status = build_merge_vectors_tasks_cub(
                    handle, n, merge_count, view, &vector_tasks);
                if (task_status != EVD_STATUS_SUCCESS) {
                    nvtxRangePop();
                    return task_status;
                }
                vector_stats =
                    summarize_merge_vectors_dispatch(n, vector_tasks);
                level_max_dd = vector_stats.max_dd;
            }
        } else if (compute_vectors) {
            const evd_status task_status = build_merge_vectors_tasks_cub(
                handle, n, merge_count, view, &vector_tasks);
            if (task_status != EVD_STATUS_SUCCESS) {
                nvtxRangePop();
                return task_status;
            }
            for (const MergeVectorsTask &task : vector_tasks) {
                level_max_dd = std::max(level_max_dd, task.dd);
            }
        }
        if (enable_diagnostics || compute_vectors) {
            nvtxRangePop();
        }

        EVD_RETURN_IF_CUDA_ERROR(
            handle, "tridiag_solver.copy_sorted_z",
            cudaMemcpyAsync(ptr_z(n, view.d_tmpz), ptr_cz(n, view.d_tmpz),
                            sizeof(double) * n, cudaMemcpyDeviceToDevice,
                            handle->stream));

        const bool use_cub_values_solve =
            (compute_vectors ? level_max_dd : level_max_dd_upper) >=
            kSecularSolveCubMinDd;
        if (compute_vectors) {
            {
                const evd_status timing_status =
                    merge_timer.begin(handle, MergeTimingSlot::ValuesSolve);
                if (timing_status != EVD_STATUS_SUCCESS) {
                    return timing_status;
                }
            }
            stedc_merge_copyd_kernel<<<n, 256, 0, handle->stream>>>(
                n, view.d_dc_diag, view.d_tmpz, view.d_splits, etmpd);
            EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.merge_copyd",
                                     cudaGetLastError());

            if (use_cub_values_solve) {
                stedc_merge_values_solve_cub_kernel<kSecularSolveCubBlockSize>
                    <<<n, kSecularSolveCubBlockSize, 0, handle->stream>>>(
                        n, view.d_tmpz, etmpd, view.d_splits, eps, ssfmin,
                        ssfmax);
            } else {
                const int solve_blocks =
                    (n + kSecularSolveBlockSize - 1) / kSecularSolveBlockSize;
                stedc_merge_values_solve_kernel<<<
                    solve_blocks, kSecularSolveBlockSize, 0, handle->stream>>>(
                    n, view.d_tmpz, etmpd, view.d_splits, eps, ssfmin, ssfmax);
            }
            EVD_RETURN_IF_CUDA_ERROR(handle,
                                     "tridiag_solver.merge_values_solve",
                                     cudaGetLastError());
            {
                const evd_status timing_status =
                    merge_timer.end(handle, MergeTimingSlot::ValuesSolve);
                if (timing_status != EVD_STATUS_SUCCESS) {
                    return timing_status;
                }
            }
            {
                const evd_status timing_status =
                    merge_timer.begin(handle, MergeTimingSlot::Rescale);
                if (timing_status != EVD_STATUS_SUCCESS) {
                    return timing_status;
                }
            }
            stedc_merge_rescale_cub_kernel<256><<<n, 256, 0, handle->stream>>>(
                n, view.d_dc_diag, view.d_tmpz, view.d_splits, etmpd);
            EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.merge_rescale",
                                     cudaGetLastError());
            {
                const evd_status timing_status =
                    merge_timer.end(handle, MergeTimingSlot::Rescale);
                if (timing_status != EVD_STATUS_SUCCESS) {
                    return timing_status;
                }
            }
        } else {
            double *active_scratch = vecs;
            double *next_boundary_rows =
                (current_boundary_rows == boundary_pingpong) ? view.d_boundary_rows
                                                             : boundary_pingpong;
            const int slab_columns = static_cast<int>(
                layout.etmpd_bytes / (sizeof(double) * static_cast<std::size_t>(n)));
            ActiveRootSelection active_roots{};
            const bool use_full_column_small_n =
                n <= kValuesOnlyFullColumnMaxN && slab_columns >= n;
            {
                const evd_status timing_status =
                    merge_timer.begin(handle, MergeTimingSlot::ValuesSolve);
                if (timing_status != EVD_STATUS_SUCCESS) {
                    return timing_status;
                }
            }
            auto time_begin = [&](const MergeTimingSlot slot) -> evd_status {
                return merge_timer.begin(handle, slot);
            };
            auto time_end = [&](const MergeTimingSlot slot) -> evd_status {
                evd_status timing_status = merge_timer.end(handle, slot);
                if (timing_status != EVD_STATUS_SUCCESS) {
                    return timing_status;
                }
                return merge_timer.add(handle, slot);
            };
            if (use_full_column_small_n) {
                {
                    const evd_status timing_status =
                        time_begin(MergeTimingSlot::ValuesCopyD);
                    if (timing_status != EVD_STATUS_SUCCESS) {
                        return timing_status;
                    }
                }
                stedc_merge_copyd_kernel<<<n, 256, 0, handle->stream>>>(
                    n, view.d_dc_diag, view.d_tmpz, view.d_splits, etmpd);
                EVD_RETURN_IF_CUDA_ERROR(handle,
                                         "tridiag_solver.merge_copyd_small_n",
                                         cudaGetLastError());
                {
                    const evd_status timing_status =
                        time_end(MergeTimingSlot::ValuesCopyD);
                    if (timing_status != EVD_STATUS_SUCCESS) {
                        return timing_status;
                    }
                }
                {
                    const evd_status timing_status =
                        time_begin(MergeTimingSlot::ValuesSecularSolve);
                    if (timing_status != EVD_STATUS_SUCCESS) {
                        return timing_status;
                    }
                }
                if (use_cub_values_solve) {
                    stedc_merge_values_solve_chunk_adaptive_cub_kernel<
                        kValuesOnlyBlockSolveThreads>
                        <<<n, kValuesOnlyBlockSolveThreads, 0,
                           handle->stream>>>(n, 0, n, nullptr, view.d_tmpz, etmpd,
                                             view.d_splits, eps, ssfmin,
                                             ssfmax);
                } else {
                    const int solve_blocks =
                        (n + kValuesOnlyPackedSeqThreads - 1) /
                        kValuesOnlyPackedSeqThreads;
                    stedc_merge_values_solve_chunk_seq_kernel<<<
                        solve_blocks, kValuesOnlyPackedSeqThreads, 0,
                        handle->stream>>>(n, 0, n, nullptr, view.d_tmpz, etmpd,
                                          view.d_splits, eps, ssfmin, ssfmax);
                }
                EVD_RETURN_IF_CUDA_ERROR(
                    handle, "tridiag_solver.merge_values_solve_small_n",
                    cudaGetLastError());
                {
                    const evd_status timing_status =
                        time_end(MergeTimingSlot::ValuesSecularSolve);
                    if (timing_status != EVD_STATUS_SUCCESS) {
                        return timing_status;
                    }
                }
                {
                    const evd_status timing_status =
                        time_begin(MergeTimingSlot::ValuesRescale);
                    if (timing_status != EVD_STATUS_SUCCESS) {
                        return timing_status;
                    }
                }
                stedc_merge_rescale_cub_kernel<256>
                    <<<n, 256, 0, handle->stream>>>(
                        n, view.d_dc_diag, view.d_tmpz, view.d_splits, etmpd);
                EVD_RETURN_IF_CUDA_ERROR(
                    handle, "tridiag_solver.merge_rescale_small_n",
                    cudaGetLastError());
                {
                    const evd_status timing_status =
                        time_end(MergeTimingSlot::ValuesRescale);
                    if (timing_status != EVD_STATUS_SUCCESS) {
                        return timing_status;
                    }
                }
                {
                    const evd_status timing_status =
                        time_begin(MergeTimingSlot::ValuesBoundaryPrepare);
                    if (timing_status != EVD_STATUS_SUCCESS) {
                        return timing_status;
                    }
                }
                stedc_merge_vectors_prepare_kernel<<<
                    n, kSecularSolveCubBlockSize, 0, handle->stream>>>(
                    n, view.d_tmpz, etmpd, view.d_splits);
                EVD_RETURN_IF_CUDA_ERROR(
                    handle, "tridiag_solver.boundary_prepare_etmpd_small_n",
                    cudaGetLastError());
                {
                    const evd_status timing_status =
                        time_end(MergeTimingSlot::ValuesBoundaryPrepare);
                    if (timing_status != EVD_STATUS_SUCCESS) {
                        return timing_status;
                    }
                }
                {
                    const evd_status timing_status =
                        time_begin(MergeTimingSlot::ValuesBoundaryUpdate);
                    if (timing_status != EVD_STATUS_SUCCESS) {
                        return timing_status;
                    }
                }
                stedc_update_boundary_rows_cub_kernel<
                    kSecularSolveCubBlockSize>
                    <<<n, kSecularSolveCubBlockSize, 0, handle->stream>>>(
                        n, etmpd, view.d_splits, current_boundary_rows,
                        next_boundary_rows);
                EVD_RETURN_IF_CUDA_ERROR(
                    handle, "tridiag_solver.update_boundary_rows_small_n",
                    cudaGetLastError());
                {
                    const evd_status timing_status =
                        time_end(MergeTimingSlot::ValuesBoundaryUpdate);
                    if (timing_status != EVD_STATUS_SUCCESS) {
                        return timing_status;
                    }
                }
            } else {
                {
                    const evd_status timing_status =
                        time_begin(MergeTimingSlot::ValuesActiveRoots);
                    if (timing_status != EVD_STATUS_SUCCESS) {
                        return timing_status;
                    }
                }
                const evd_status active_status = build_active_root_indices_cub(
                    handle, n, active_scratch, view.d_splits, &active_roots);
                if (active_status != EVD_STATUS_SUCCESS) {
                    return active_status;
                }
                {
                    const evd_status timing_status =
                        time_end(MergeTimingSlot::ValuesActiveRoots);
                    if (timing_status != EVD_STATUS_SUCCESS) {
                        return timing_status;
                    }
                }
                {
                    const evd_status timing_status =
                        time_begin(MergeTimingSlot::ValuesDeflatedBoundary);
                    if (timing_status != EVD_STATUS_SUCCESS) {
                        return timing_status;
                    }
                }
                stedc_update_deflated_boundary_rows_kernel<<<blocks, threads, 0,
                                                             handle->stream>>>(
                    n, view.d_splits, current_boundary_rows,
                    next_boundary_rows);
                EVD_RETURN_IF_CUDA_ERROR(
                    handle, "tridiag_solver.update_deflated_boundary_rows",
                    cudaGetLastError());
                {
                    const evd_status timing_status =
                        time_end(MergeTimingSlot::ValuesDeflatedBoundary);
                    if (timing_status != EVD_STATUS_SUCCESS) {
                        return timing_status;
                    }
                }
                for (int chunk_start = 0; chunk_start < active_roots.count;
                     chunk_start += slab_columns) {
                    const int chunk_size =
                        std::min(slab_columns, active_roots.count - chunk_start);
                    {
                        const evd_status timing_status =
                            time_begin(MergeTimingSlot::ValuesCopyD);
                        if (timing_status != EVD_STATUS_SUCCESS) {
                            return timing_status;
                        }
                    }
                    stedc_merge_copyd_chunk_kernel<<<chunk_size, 256, 0,
                                                     handle->stream>>>(
                        n, chunk_start, chunk_size, active_roots.indices,
                        view.d_dc_diag, view.d_tmpz, view.d_splits, etmpd);
                    EVD_RETURN_IF_CUDA_ERROR(
                        handle, "tridiag_solver.merge_copyd_chunk",
                        cudaGetLastError());
                    {
                        const evd_status timing_status =
                            time_end(MergeTimingSlot::ValuesCopyD);
                        if (timing_status != EVD_STATUS_SUCCESS) {
                            return timing_status;
                        }
                    }
                    {
                        const evd_status timing_status =
                            time_begin(MergeTimingSlot::ValuesSecularSolve);
                        if (timing_status != EVD_STATUS_SUCCESS) {
                            return timing_status;
                        }
                    }
                    if (level_max_dd_upper < kValuesOnlyPackedSeqMaxDd) {
                        const int solve_blocks =
                            (chunk_size + kValuesOnlyPackedSeqThreads - 1) /
                            kValuesOnlyPackedSeqThreads;
                        stedc_merge_values_solve_chunk_seq_kernel<<<
                            solve_blocks, kValuesOnlyPackedSeqThreads, 0,
                            handle->stream>>>(
                            n, chunk_start, chunk_size, active_roots.indices,
                            view.d_tmpz, etmpd, view.d_splits, eps, ssfmin,
                            ssfmax);
                        EVD_RETURN_IF_CUDA_ERROR(
                            handle,
                            "tridiag_solver.merge_values_solve_chunk_seq",
                            cudaGetLastError());
                    } else {
                        const int fallback_blocks =
                            (chunk_size + kValuesOnlyPackedSeqThreads - 1) /
                            kValuesOnlyPackedSeqThreads;
                        stedc_merge_values_solve_chunk_fallback_seq_kernel<<<
                            fallback_blocks, kValuesOnlyPackedSeqThreads, 0,
                            handle->stream>>>(
                            n, chunk_start, chunk_size, active_roots.indices,
                            view.d_tmpz, etmpd, view.d_splits, eps, ssfmin,
                            ssfmax);
                        EVD_RETURN_IF_CUDA_ERROR(
                            handle,
                            "tridiag_solver.merge_values_solve_chunk_fallback",
                            cudaGetLastError());
                        if (level_max_dd_upper <=
                            kValuesOnlyBlockSolve128MaxDd) {
                            stedc_merge_values_solve_chunk_ext_cub_kernel<
                                kValuesOnlyBlockSolveThreads>
                                <<<chunk_size, kValuesOnlyBlockSolveThreads, 0,
                                   handle->stream>>>(
                                    n, chunk_start, chunk_size,
                                    active_roots.indices, view.d_tmpz, etmpd,
                                    view.d_splits, eps, ssfmin, ssfmax);
                            EVD_RETURN_IF_CUDA_ERROR(
                                handle,
                                "tridiag_solver.merge_values_solve_chunk_ext",
                                cudaGetLastError());
                            stedc_merge_values_solve_chunk_cub_kernel<
                                kValuesOnlyBlockSolveThreads>
                                <<<chunk_size, kValuesOnlyBlockSolveThreads, 0,
                                   handle->stream>>>(
                                    n, chunk_start, chunk_size,
                                    active_roots.indices, view.d_tmpz, etmpd,
                                    view.d_splits, eps, ssfmin, ssfmax);
                        } else {
                            stedc_merge_values_solve_chunk_ext_cub_kernel<
                                kSecularSolveCubBlockSize>
                                <<<chunk_size, kSecularSolveCubBlockSize, 0,
                                   handle->stream>>>(
                                    n, chunk_start, chunk_size,
                                    active_roots.indices, view.d_tmpz, etmpd,
                                    view.d_splits, eps, ssfmin, ssfmax);
                            EVD_RETURN_IF_CUDA_ERROR(
                                handle,
                                "tridiag_solver.merge_values_solve_chunk_ext",
                                cudaGetLastError());
                            stedc_merge_values_solve_chunk_cub_kernel<
                                kSecularSolveCubBlockSize>
                                <<<chunk_size, kSecularSolveCubBlockSize, 0,
                                   handle->stream>>>(
                                    n, chunk_start, chunk_size,
                                    active_roots.indices, view.d_tmpz, etmpd,
                                    view.d_splits, eps, ssfmin, ssfmax);
                        }
                        EVD_RETURN_IF_CUDA_ERROR(
                            handle,
                            "tridiag_solver.merge_values_solve_chunk",
                            cudaGetLastError());
                    }
                    {
                        const evd_status timing_status =
                            time_end(MergeTimingSlot::ValuesSecularSolve);
                        if (timing_status != EVD_STATUS_SUCCESS) {
                            return timing_status;
                        }
                    }
                    {
                        const evd_status timing_status =
                            time_begin(MergeTimingSlot::ValuesRescale);
                        if (timing_status != EVD_STATUS_SUCCESS) {
                            return timing_status;
                        }
                    }
                    stedc_merge_rescale_chunk_cub_kernel<256>
                        <<<chunk_size, 256, 0, handle->stream>>>(
                            n, chunk_start, chunk_size, active_roots.indices,
                            view.d_dc_diag, view.d_tmpz, view.d_splits, etmpd);
                    EVD_RETURN_IF_CUDA_ERROR(
                        handle, "tridiag_solver.merge_rescale_chunk",
                        cudaGetLastError());
                    {
                        const evd_status timing_status =
                            time_end(MergeTimingSlot::ValuesRescale);
                        if (timing_status != EVD_STATUS_SUCCESS) {
                            return timing_status;
                        }
                    }
                    {
                        const evd_status timing_status =
                            time_begin(MergeTimingSlot::ValuesBoundaryPrepare);
                        if (timing_status != EVD_STATUS_SUCCESS) {
                            return timing_status;
                        }
                    }
                    stedc_merge_vectors_prepare_chunk_kernel<
                        kSecularSolveCubBlockSize>
                        <<<chunk_size, kSecularSolveCubBlockSize, 0,
                           handle->stream>>>(
                            n, chunk_start, chunk_size, view.d_tmpz, etmpd,
                            view.d_splits, active_roots.indices);
                    EVD_RETURN_IF_CUDA_ERROR(
                        handle, "tridiag_solver.boundary_prepare_etmpd_chunk",
                        cudaGetLastError());
                    {
                        const evd_status timing_status =
                            time_end(MergeTimingSlot::ValuesBoundaryPrepare);
                        if (timing_status != EVD_STATUS_SUCCESS) {
                            return timing_status;
                        }
                    }
                    {
                        const evd_status timing_status =
                            time_begin(MergeTimingSlot::ValuesBoundaryUpdate);
                        if (timing_status != EVD_STATUS_SUCCESS) {
                            return timing_status;
                        }
                    }
                    stedc_update_boundary_rows_chunk_cub_kernel<
                        kSecularSolveCubBlockSize>
                        <<<chunk_size, kSecularSolveCubBlockSize, 0,
                           handle->stream>>>(
                            n, chunk_start, chunk_size, active_roots.indices,
                            etmpd, view.d_splits, current_boundary_rows,
                            next_boundary_rows);
                    EVD_RETURN_IF_CUDA_ERROR(
                        handle, "tridiag_solver.update_boundary_rows_chunk",
                        cudaGetLastError());
                    {
                        const evd_status timing_status =
                            time_end(MergeTimingSlot::ValuesBoundaryUpdate);
                        if (timing_status != EVD_STATUS_SUCCESS) {
                            return timing_status;
                        }
                    }
                }
            }
            {
                const evd_status timing_status =
                    merge_timer.end(handle, MergeTimingSlot::ValuesSolve);
                if (timing_status != EVD_STATUS_SUCCESS) {
                    return timing_status;
                }
            }
            current_boundary_rows = next_boundary_rows;
        }

        const bool use_boundary_path = !compute_vectors;

        if (use_boundary_path) {
            nvtxRangePushA("tridiag.mergeUpdateValues");
            stedc_merge_update_values_kernel<<<blocks, threads, 0,
                                               handle->stream>>>(
                n, view.d_dc_diag, view.d_dc_subdiag, view.d_tmpz,
                view.d_splits);
            EVD_RETURN_IF_CUDA_ERROR(handle,
                                     "tridiag_solver.merge_update_values",
                                     cudaGetLastError());
            nvtxRangePop();
        } else {
            // Full vector path: copy, reshuffle, GEMM, and update.
            const bool use_split_no_global_reshuffle =
                all_merge_vectors_tasks_use_split(n, vector_tasks);
            const bool direct_write_active_vectors =
                use_split_no_global_reshuffle;
            nvtxRangePushA("tridiag.copyC");
            if (!use_split_no_global_reshuffle) {
                {
                    const evd_status timing_status =
                        merge_timer.begin(handle, MergeTimingSlot::GlobalCopyC);
                    if (timing_status != EVD_STATUS_SUCCESS) {
                        nvtxRangePop();
                        return timing_status;
                    }
                }
                stedc_copyc_kernel<<<n, 256, 0, handle->stream>>>(
                    n, view.d_tridiag_eigenvectors, n, vecs, n);
                EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.copyc",
                                         cudaGetLastError());
                {
                    const evd_status timing_status =
                        merge_timer.end(handle, MergeTimingSlot::GlobalCopyC);
                    if (timing_status != EVD_STATUS_SUCCESS) {
                        nvtxRangePop();
                        return timing_status;
                    }
                }
            }
            nvtxRangePop();

            nvtxRangePushA("tridiag.reshuffleC");
            if (use_split_no_global_reshuffle) {
                {
                    const evd_status timing_status = merge_timer.begin(
                        handle, MergeTimingSlot::DeflatedGather);
                    if (timing_status != EVD_STATUS_SUCCESS) {
                        nvtxRangePop();
                        return timing_status;
                    }
                }
                for (const MergeVectorsTask &task : vector_tasks) {
                    const evd_status gather_status =
                        gather_deflated_merge_vector_columns(handle, n, view,
                                                             task);
                    if (gather_status != EVD_STATUS_SUCCESS) {
                        nvtxRangePop();
                        return gather_status;
                    }
                }
                {
                    const evd_status timing_status = merge_timer.end(
                        handle, MergeTimingSlot::DeflatedGather);
                    if (timing_status != EVD_STATUS_SUCCESS) {
                        nvtxRangePop();
                        return timing_status;
                    }
                }
            } else {
                {
                    const evd_status timing_status = merge_timer.begin(
                        handle, MergeTimingSlot::GlobalReshuffleC);
                    if (timing_status != EVD_STATUS_SUCCESS) {
                        nvtxRangePop();
                        return timing_status;
                    }
                }
                stedc_reshufflec_kernel<<<n, 256, 0, handle->stream>>>(
                    n, vecs, n, view.d_tridiag_eigenvectors, n, view.d_splits);
                EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.reshufflec",
                                         cudaGetLastError());
                {
                    const evd_status timing_status = merge_timer.end(
                        handle, MergeTimingSlot::GlobalReshuffleC);
                    if (timing_status != EVD_STATUS_SUCCESS) {
                        nvtxRangePop();
                        return timing_status;
                    }
                }
            }
            nvtxRangePop();

            nvtxRangePushA("tridiag.mergeVectors");
            {
                const evd_status timing_status =
                    merge_timer.begin(handle, MergeTimingSlot::VectorPrepare);
                if (timing_status != EVD_STATUS_SUCCESS) {
                    nvtxRangePop();
                    return timing_status;
                }
            }
            stedc_merge_vectors_prepare_kernel<<<n, 256, 0, handle->stream>>>(
                n, view.d_tmpz, etmpd, view.d_splits);
            EVD_RETURN_IF_CUDA_ERROR(handle,
                                     "tridiag_solver.merge_vectors_prepare",
                                     cudaGetLastError());
            {
                const evd_status timing_status =
                    merge_timer.end(handle, MergeTimingSlot::VectorPrepare);
                if (timing_status != EVD_STATUS_SUCCESS) {
                    nvtxRangePop();
                    return timing_status;
                }
            }

            {
                const evd_status timing_status =
                    merge_timer.begin(handle, MergeTimingSlot::Gemm);
                if (timing_status != EVD_STATUS_SUCCESS) {
                    nvtxRangePop();
                    return timing_status;
                }
            }
            SplitMergeVectorsTimings *split_timings_ptr =
                merge_timer.enabled ? &split_timings : nullptr;
            const evd_status dispatch_status = launch_merge_vectors_cublas(
                handle, n, view, vector_tasks, use_split_no_global_reshuffle,
                direct_write_active_vectors, split_timings_ptr);
            if (dispatch_status != EVD_STATUS_SUCCESS) {
                nvtxRangePop();
                return dispatch_status;
            }
            {
                const evd_status timing_status =
                    merge_timer.end(handle, MergeTimingSlot::Gemm);
                if (timing_status != EVD_STATUS_SUCCESS) {
                    nvtxRangePop();
                    return timing_status;
                }
            }
            nvtxRangePop();

            nvtxRangePushA("tridiag.mergeUpdate");
            {
                const evd_status timing_status = merge_timer.begin(
                    handle, use_split_no_global_reshuffle
                                ? MergeTimingSlot::UpdateAllVectors
                                : MergeTimingSlot::MergeUpdate);
                if (timing_status != EVD_STATUS_SUCCESS) {
                    nvtxRangePop();
                    return timing_status;
                }
            }
            if (direct_write_active_vectors) {
                for (const MergeVectorsTask &task : vector_tasks) {
                    const evd_status scatter_status =
                        scatter_deflated_merge_vector_columns(handle, n, view,
                                                              task);
                    if (scatter_status != EVD_STATUS_SUCCESS) {
                        nvtxRangePop();
                        return scatter_status;
                    }
                }
                stedc_merge_update_values_kernel<<<blocks, threads, 0,
                                                   handle->stream>>>(
                    n, view.d_dc_diag, view.d_dc_subdiag, view.d_tmpz,
                    view.d_splits);
            } else if (use_split_no_global_reshuffle) {
                stedc_merge_update_all_vectors_kernel<<<blocks, threads, 0,
                                                        handle->stream>>>(
                    n, view.d_dc_diag, view.d_dc_subdiag,
                    view.d_tridiag_eigenvectors, n, view.d_tmpz,
                    view.d_tempgemm, view.d_splits);
            } else {
                stedc_merge_update_kernel<<<blocks, threads, 0,
                                            handle->stream>>>(
                    n, view.d_dc_diag, view.d_dc_subdiag,
                    view.d_tridiag_eigenvectors, n, view.d_tmpz,
                    view.d_tempgemm, view.d_splits);
            }
            EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.merge_update",
                                     cudaGetLastError());
            {
                const evd_status timing_status = merge_timer.end(
                    handle, use_split_no_global_reshuffle
                                ? MergeTimingSlot::UpdateAllVectors
                                : MergeTimingSlot::MergeUpdate);
                if (timing_status != EVD_STATUS_SUCCESS) {
                    nvtxRangePop();
                    return timing_status;
                }
            }
            nvtxRangePop();
        }

        if (enable_diagnostics) {
            EVD_RETURN_IF_CUDA_ERROR(
                handle, "tridiag_solver.diag_copy_dbg_after",
                cudaMemcpyAsync(dbg_host.data(), ptr_dbg(n, view.d_splits),
                                sizeof(int) * static_cast<std::size_t>(n),
                                cudaMemcpyDeviceToHost, handle->stream));
            EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.diag_dbg_sync_after",
                                     cudaStreamSynchronize(handle->stream));
            MergeLevelTimings timings{};
            const evd_status timing_status =
                merge_timer.collect(handle, &timings);
            if (timing_status != EVD_STATUS_SUCCESS) {
                return timing_status;
            }
            timings.split_make_indices_ms = split_timings.make_indices_ms;
            timings.split_left_pack_ms = split_timings.left_pack_ms;
            timings.split_left_gemm_ms = split_timings.left_gemm_ms;
            timings.split_right_pack_ms = split_timings.right_pack_ms;
            timings.split_right_gemm_ms = split_timings.right_gemm_ms;
            append_merge_level_diagnostics(
                handle, k, merge_count, nps_host, nsz_host, dd_before_host,
                dd_after_host, &dbg_host,
                compute_vectors ? &vector_stats : nullptr,
                &timings);
        }
    }

    int residual_failed_merges = 0;
    if (enable_diagnostics) {
        std::vector<int> dbg_local(static_cast<std::size_t>(n), 0);
        std::vector<int> nps_local(static_cast<std::size_t>(n), 0);
        EVD_RETURN_IF_CUDA_ERROR(
            handle, "tridiag_solver.copy_dbg_raw",
            cudaMemcpyAsync(dbg_local.data(), ptr_dbg(n, view.d_splits),
                            sizeof(int) * static_cast<std::size_t>(n),
                            cudaMemcpyDeviceToHost, handle->stream));
        EVD_RETURN_IF_CUDA_ERROR(
            handle, "tridiag_solver.copy_nps_raw",
            cudaMemcpyAsync(nps_local.data(), ptr_nps(n, view.d_splits),
                            sizeof(int) * static_cast<std::size_t>(n),
                            cudaMemcpyDeviceToHost, handle->stream));
        EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.copy_dbg_sync",
                                 cudaStreamSynchronize(handle->stream));

        std::unordered_set<int> failed_merge_starts;
        for (int i = 0; i < n; ++i) {
            if (dbg_local[i] < 0) {
                failed_merge_starts.insert(nps_local[i]);
            }
        }
        residual_failed_merges = static_cast<int>(failed_merge_starts.size());
        handle->last_tridiag_diagnostics.available = 1;
        handle->last_tridiag_diagnostics.total_merges = total_merges;
        handle->last_tridiag_diagnostics.residual_failed_merges =
            residual_failed_merges;
    }

    int summary_local[kSecularDbgSummaryCount] = {0};
    summary_local[kSecularDbgAllTakenOver] = 1;
    summary_local[kSecularDbgTotalMerges] = total_merges;
    summary_local[kSecularDbgResidualFailedMerges] = residual_failed_merges;
    if (summary_out != nullptr) {
        for (int i = 0; i < kSecularDbgSummaryCount; ++i) {
            summary_out[i] = summary_local[i];
        }
    }
    if (view.d_splits != nullptr) {
        EVD_RETURN_IF_CUDA_ERROR(
            handle, "tridiag_solver.copy_dbg_summary",
            cudaMemcpyAsync(ptr_dbg(n, view.d_splits), summary_local,
                            sizeof(int) * kSecularDbgSummaryCount,
                            cudaMemcpyHostToDevice, handle->stream));
    }
    if (all_blocks_taken_over != nullptr) {
        *all_blocks_taken_over = true;
    }
    return EVD_STATUS_SUCCESS;
}

evd_status tridiag_solver_workspace_size(const evd_syev2stage_config &config,
                                         size_t *bytes) {
    if (bytes == nullptr) {
        return EVD_STATUS_INVALID_POINTER;
    }

    TridiagSolverWorkspaceLayout layout{};
    const evd_status status =
        make_tridiag_solver_workspace_layout(config, &layout);
    if (status != EVD_STATUS_SUCCESS) {
        return status;
    }

    *bytes = layout.total;
    return EVD_STATUS_SUCCESS;
}

evd_status run_prepared_gpu_tridiag_solver(
    evd_handle handle, const evd_syev2stage_config &config, double *d_W,
    const TridiagSolverWorkspaceLayout &layout,
    const TridiagSolverWorkspaceView &view) {
    const int n = static_cast<int>(config.n);

    if (view.d_tempgemm != nullptr) {
        EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.clear_tempgemm",
                                 cudaMemsetAsync(view.d_tempgemm, 0,
                                                 layout.tempgemm_bytes,
                                                 handle->stream));
    }
    if (view.d_tridiag_eigenvectors != nullptr) {
        nvtxRangePushA("tridiag.clearInternalVectors");
        EVD_RETURN_IF_CUDA_ERROR(
            handle, "tridiag_solver.clear_internal_vectors",
            cudaMemsetAsync(view.d_tridiag_eigenvectors, 0,
                            layout.eigenvectors_bytes, handle->stream));
        nvtxRangePop();
    }

    if (layout.tmpz_bytes > 0) {
        EVD_RETURN_IF_CUDA_ERROR(
            handle, "tridiag_solver.clear_tmpz",
            cudaMemsetAsync(view.d_tmpz, 0, layout.tmpz_bytes, handle->stream));
    }

    if (layout.splits_bytes > 0) {
        const int levs = stedc_num_levels(n);
        init_stedc_splits_kernel<<<1, 1, 0, handle->stream>>>(levs, 1 << levs,
                                                              n, view.d_splits);
        EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.init_splits",
                                 cudaGetLastError());

        const bool compute_vectors = config.compute_eigenvectors != 0;
        const bool enable_diagnostics = config.enable_tridiag_diagnostics != 0;
        nvtxRangePushA("tridiag.leafStage");
        const evd_status leaf_status =
            run_stedc_leaf_stage(handle, n, layout, view, compute_vectors);
        nvtxRangePop();
        if (leaf_status != EVD_STATUS_SUCCESS) {
            return leaf_status;
        }

        // Extract boundary rows from leaf eigenvectors when V exists.
        // When !compute_vectors, the leaf kernel writes boundary rows directly
        // and this extraction is a no-op (V may be null in that case).
        if (view.d_boundary_rows != nullptr &&
            view.d_tridiag_eigenvectors != nullptr && compute_vectors) {
            const int blks = 1 << stedc_num_levels(n);
            nvtxRangePushA("tridiag.extractBoundaryRows");
            stedc_leaf_extract_boundary_rows_kernel<<<
                blks, kTridiagSolverMaxLeafSize, 0, handle->stream>>>(
                blks, n, view.d_tridiag_eigenvectors, n, view.d_splits,
                view.d_boundary_rows);
            EVD_RETURN_IF_CUDA_ERROR(handle,
                                     "tridiag_solver.extract_boundary_rows",
                                     cudaGetLastError());
            nvtxRangePop();
        }

        int dbg_summary[kSecularDbgSummaryCount] = {0};
        bool all_blocks_taken_over = false;
        const evd_status merge_status = run_stedc_merge_levels(
            handle, n, layout, view, dbg_summary, &all_blocks_taken_over,
            compute_vectors, enable_diagnostics);
        if (merge_status != EVD_STATUS_SUCCESS) {
            return merge_status;
        }

        const evd_status sort_status = sort_eigenpairs_inplace(
            handle, n, view, config.tridiag_sort_backend, compute_vectors);
        if (sort_status != EVD_STATUS_SUCCESS) {
            return sort_status;
        }
        EVD_RETURN_IF_CUDA_ERROR(
            handle, "tridiag_solver.copy_merge_result",
            cudaMemcpyAsync(d_W, view.d_dc_diag, sizeof(double) * n,
                            cudaMemcpyDeviceToDevice, handle->stream));
        return EVD_STATUS_SUCCESS;
    }

    set_last_error(handle, 0, EVD_ERROR_SRC_NONE,
                   "tridiag_solver.dense_fallback_disabled");
    return EVD_STATUS_INTERNAL_ERROR;
}

evd_status tridiag_solver_run(evd_handle handle, const Syev2StageState &state,
                              void *workspace, size_t workspace_size) {
    if (handle == nullptr) {
        return EVD_STATUS_INVALID_HANDLE;
    }

    if (state.config == nullptr || state.d_A == nullptr ||
        state.d_W == nullptr) {
        return EVD_STATUS_INVALID_POINTER;
    }

    if (state.config->n <= 0 || state.lda < state.config->n ||
        state.config->uplo != EVD_FILL_MODE_LOWER) {
        return EVD_STATUS_INVALID_VALUE;
    }

    reset_tridiag_diagnostics(handle,
                              state.config->enable_tridiag_diagnostics != 0);

    TridiagSolverWorkspaceLayout layout{};
    TridiagSolverWorkspaceView view{};
    const bool use_external_eigenvectors =
        state.config->compute_eigenvectors != 0 && state.d_Q != nullptr;
    const evd_status view_status = make_tridiag_solver_workspace_view(
        *state.config, workspace, workspace_size, &view, &layout,
        use_external_eigenvectors ? state.d_Q : nullptr);
    if (view_status != EVD_STATUS_SUCCESS) {
        return view_status;
    }

    if (state.artifacts != nullptr && state.config->compute_eigenvectors != 0) {
        state.artifacts->d_tridiag_eigenvectors = view.d_tridiag_eigenvectors;
        state.artifacts->ld_tridiag_eigenvectors = state.config->n;
    }

    const unsigned int grid_x = static_cast<unsigned int>(
        (state.config->n + kTridiagSolverExtractBlockSize - 1) /
        kTridiagSolverExtractBlockSize);
    extract_lower_tridiagonal_kernel<<<grid_x, kTridiagSolverExtractBlockSize,
                                       0, handle->stream>>>(
        state.config->n, state.d_A, state.lda, view.d_diag, view.d_subdiag,
        view.d_dc_diag, view.d_dc_subdiag);
    EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.extract_tridiagonal",
                             cudaGetLastError());

    if (state.artifacts != nullptr &&
        state.config->compute_eigenvectors != 0 &&
        state.artifacts->d_tridiag_signs != nullptr) {
        const evd_status signs_status = make_tridiag_signs_from_subdiag(
            handle, state.config->n, view.d_subdiag,
            state.artifacts->d_tridiag_signs);
        if (signs_status != EVD_STATUS_SUCCESS) {
            return signs_status;
        }
    }

    if (!use_gpu_tridiag_solver(*state.config)) {
        return run_tridiag_solver_cpu(handle, state, view);
    }

    return run_prepared_gpu_tridiag_solver(handle, *state.config, state.d_W,
                                           layout, view);
}

evd_status tridiag_solver_run_from_device_tridiagonal(
    evd_handle handle, const evd_syev2stage_config &config,
    const double *d_diag, const double *d_subdiag, double *d_W, void *workspace,
    size_t workspace_size) {
    if (handle == nullptr) {
        return EVD_STATUS_INVALID_HANDLE;
    }

    if (d_diag == nullptr || d_W == nullptr ||
        (config.n > 1 && d_subdiag == nullptr)) {
        return EVD_STATUS_INVALID_POINTER;
    }

    if (config.n <= 0 || config.uplo != EVD_FILL_MODE_LOWER ||
        !use_gpu_tridiag_solver(config)) {
        return EVD_STATUS_INVALID_VALUE;
    }

    reset_tridiag_diagnostics(handle, config.enable_tridiag_diagnostics != 0);

    TridiagSolverWorkspaceLayout layout{};
    TridiagSolverWorkspaceView view{};
    const evd_status view_status = make_tridiag_solver_workspace_view(
        config, workspace, workspace_size, &view, &layout);
    if (view_status != EVD_STATUS_SUCCESS) {
        return view_status;
    }

    if (view.d_dc_diag == nullptr || view.d_dc_subdiag == nullptr) {
        return EVD_STATUS_INVALID_POINTER;
    }

    const int n = static_cast<int>(config.n);
    constexpr int kCopyThreads = 256;
    const int copy_blocks = (n + kCopyThreads - 1) / kCopyThreads;
    copy_device_tridiagonal_kernel<<<copy_blocks, kCopyThreads, 0,
                                     handle->stream>>>(
        n, d_diag, d_subdiag, view.d_dc_diag, view.d_dc_subdiag);
    EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.copy_device_tridiagonal",
                             cudaGetLastError());

    return run_prepared_gpu_tridiag_solver(handle, config, d_W, layout, view);
}

} // namespace evd::internal
