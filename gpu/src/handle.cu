#include "handle_impl.hpp"

#include <algorithm>
#include <new>

namespace {

void destroy_stage_events(evd_handle handle) {
    if (handle == nullptr) {
        return;
    }

    for (int i = 0; i < static_cast<int>(EVD_STAGE_COUNT); ++i) {
        if (handle->stage_start_events[i] != nullptr) {
            (void)cudaEventDestroy(handle->stage_start_events[i]);
            handle->stage_start_events[i] = nullptr;
        }

        if (handle->stage_stop_events[i] != nullptr) {
            (void)cudaEventDestroy(handle->stage_stop_events[i]);
            handle->stage_stop_events[i] = nullptr;
        }
    }
}

evd_status create_stage_events(evd_handle handle) {
    if (handle == nullptr) {
        return EVD_STATUS_INVALID_HANDLE;
    }

    for (int i = 0; i < static_cast<int>(EVD_STAGE_COUNT); ++i) {
        if (cudaEventCreate(&handle->stage_start_events[i]) != cudaSuccess) {
            destroy_stage_events(handle);
            return EVD_STATUS_CUDA_ERROR;
        }

        if (cudaEventCreate(&handle->stage_stop_events[i]) != cudaSuccess) {
            destroy_stage_events(handle);
            return EVD_STATUS_CUDA_ERROR;
        }
    }

    return EVD_STATUS_SUCCESS;
}

evd_status destroy_handle_contents(evd_handle handle) {
    if (handle == nullptr) {
        return EVD_STATUS_INVALID_HANDLE;
    }

    destroy_stage_events(handle);

    if (handle->cublas_handle != nullptr) {
        cublasDestroy(handle->cublas_handle);
        handle->cublas_handle = nullptr;
    }

    if (handle->cusolver_handle != nullptr) {
        cusolverDnDestroy(handle->cusolver_handle);
        handle->cusolver_handle = nullptr;
    }

    if (handle->stream != nullptr) {
        cudaStreamDestroy(handle->stream);
        handle->stream = nullptr;
    }

    handle->workspace = nullptr;
    handle->workspace_size = 0;
    evd::internal::reset_last_error(handle);
    return EVD_STATUS_SUCCESS;
}

} // namespace

extern "C" evd_status evd_create_handle(evd_handle *handle) {
    if (handle == nullptr) {
        return EVD_STATUS_INVALID_POINTER;
    }

    *handle = nullptr;

    evd_handle impl = new (std::nothrow) evd_handle_struct{};
    if (impl == nullptr) {
        return EVD_STATUS_ALLOC_FAILED;
    }

    impl->workspace = nullptr;
    impl->workspace_size = 0;
    impl->timing_enabled = 1;
    evd::internal::reset_last_error(impl);
    evd::internal::reset_stage_timings(impl);
    evd::internal::reset_tridiag_diagnostics(impl);

    if (cudaStreamCreateWithFlags(&impl->stream, cudaStreamNonBlocking) !=
        cudaSuccess) {
        delete impl;
        return EVD_STATUS_CUDA_ERROR;
    }

    if (cublasCreate(&impl->cublas_handle) != CUBLAS_STATUS_SUCCESS) {
        destroy_handle_contents(impl);
        delete impl;
        return EVD_STATUS_INTERNAL_ERROR;
    }

    if (cublasSetStream(impl->cublas_handle, impl->stream) !=
        CUBLAS_STATUS_SUCCESS) {
        destroy_handle_contents(impl);
        delete impl;
        return EVD_STATUS_INTERNAL_ERROR;
    }

    if (cusolverDnCreate(&impl->cusolver_handle) != CUSOLVER_STATUS_SUCCESS) {
        destroy_handle_contents(impl);
        delete impl;
        return EVD_STATUS_INTERNAL_ERROR;
    }

    if (cusolverDnSetStream(impl->cusolver_handle, impl->stream) !=
        CUSOLVER_STATUS_SUCCESS) {
        destroy_handle_contents(impl);
        delete impl;
        return EVD_STATUS_INTERNAL_ERROR;
    }

    const evd_status event_status = create_stage_events(impl);
    if (event_status != EVD_STATUS_SUCCESS) {
        destroy_handle_contents(impl);
        delete impl;
        return event_status;
    }

    *handle = impl;
    return EVD_STATUS_SUCCESS;
}

extern "C" evd_status evd_destroy_handle(evd_handle handle) {
    if (handle == nullptr) {
        return EVD_STATUS_INVALID_HANDLE;
    }

    destroy_handle_contents(handle);
    delete handle;
    return EVD_STATUS_SUCCESS;
}

extern "C" evd_status evd_get_last_error(evd_handle handle, int *code,
                                         evd_error_source *source,
                                         const char **site) {
    if (handle == nullptr) {
        return EVD_STATUS_INVALID_HANDLE;
    }

    if (code != nullptr) {
        *code = handle->last_error_code;
    }

    if (source != nullptr) {
        *source = handle->last_error_source;
    }

    if (site != nullptr) {
        *site = handle->last_error_site;
    }

    return EVD_STATUS_SUCCESS;
}

extern "C" evd_status evd_get_last_stage_timings(evd_handle handle,
                                                 evd_stage_timing *timings,
                                                 int max_count,
                                                 int *actual_count) {
    if (handle == nullptr) {
        return EVD_STATUS_INVALID_HANDLE;
    }

    if (max_count < 0) {
        return EVD_STATUS_INVALID_VALUE;
    }

    if (timings == nullptr && max_count > 0) {
        return EVD_STATUS_INVALID_POINTER;
    }

    if (actual_count != nullptr) {
        *actual_count = static_cast<int>(EVD_STAGE_COUNT);
    }

    const int count = std::min(max_count, static_cast<int>(EVD_STAGE_COUNT));
    for (int i = 0; i < count; ++i) {
        timings[i] = handle->last_stage_timings[i];
    }

    return EVD_STATUS_SUCCESS;
}

extern "C" const char *evd_status_to_string(evd_status status) {
    switch (status) {
    case EVD_STATUS_SUCCESS:
        return "success";
    case EVD_STATUS_INVALID_HANDLE:
        return "invalid_handle";
    case EVD_STATUS_INVALID_POINTER:
        return "invalid_pointer";
    case EVD_STATUS_INVALID_VALUE:
        return "invalid_value";
    case EVD_STATUS_INVALID_WORKSPACE:
        return "invalid_workspace";
    case EVD_STATUS_ALLOC_FAILED:
        return "alloc_failed";
    case EVD_STATUS_CUDA_ERROR:
        return "cuda_error";
    case EVD_STATUS_INTERNAL_ERROR:
        return "internal_error";
    case EVD_STATUS_NOT_IMPLEMENTED:
        return "not_implemented";
    default:
        return "unknown_status";
    }
}
