#include "tridiag_solver_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>

#include <cuda_runtime.h>

#include "handle_impl.hpp"

extern "C" {
void dstedc_(const char *compz, const int *n, double *d, double *e,
             double *z, const int *ldz, double *work, const int *lwork,
             int *iwork, const int *liwork, int *info);
}

namespace evd::internal {

namespace {

template <typename T> class PinnedHostBuffer {
  public:
    PinnedHostBuffer() = default;
    PinnedHostBuffer(const PinnedHostBuffer &) = delete;
    PinnedHostBuffer &operator=(const PinnedHostBuffer &) = delete;

    ~PinnedHostBuffer() {
        if (data_ != nullptr) {
            (void)cudaFreeHost(data_);
        }
    }

    T *get() { return data_; }
    const T *get() const { return data_; }

    evd_status allocate(evd_handle handle, size_t count, const char *site) {
        if (count == 0) {
            return EVD_STATUS_SUCCESS;
        }

        void *allocation = nullptr;
        const cudaError_t cuda_status =
            cudaMallocHost(&allocation, count * sizeof(T));
        if (cuda_status != cudaSuccess) {
            set_last_error(handle, static_cast<int>(cuda_status),
                           EVD_ERROR_SRC_CUDA, site);
            return EVD_STATUS_ALLOC_FAILED;
        }

        data_ = static_cast<T *>(allocation);
        return EVD_STATUS_SUCCESS;
    }

  private:
    T *data_{nullptr};
};

} // namespace

evd_status run_tridiag_solver_cpu(evd_handle handle,
                                  const Syev2StageState &state,
                                  const TridiagSolverWorkspaceView &view) {
    const int n = static_cast<int>(state.config->n);
    const size_t diagonal_pitch =
        static_cast<size_t>(state.lda + 1) * sizeof(double);

    PinnedHostBuffer<double> h_d;
    evd_status status = h_d.allocate(handle, static_cast<size_t>(n),
                                     "tridiag_solver.cpu_alloc_d");
    if (status != EVD_STATUS_SUCCESS) {
        return status;
    }

    PinnedHostBuffer<double> h_e;
    status = h_e.allocate(handle, static_cast<size_t>(std::max(1, n - 1)),
                          "tridiag_solver.cpu_alloc_e");
    if (status != EVD_STATUS_SUCCESS) {
        return status;
    }

    EVD_RETURN_IF_CUDA_ERROR(
        handle, "tridiag_solver.cpu_copy_diagonal",
        cudaMemcpy2DAsync(h_d.get(), sizeof(double), state.d_A, diagonal_pitch,
                          sizeof(double), static_cast<size_t>(n),
                          cudaMemcpyDeviceToHost, handle->stream));
    if (n > 1) {
        EVD_RETURN_IF_CUDA_ERROR(
            handle, "tridiag_solver.cpu_copy_subdiagonal",
            cudaMemcpy2DAsync(h_e.get(), sizeof(double), state.d_A + 1,
                              diagonal_pitch, sizeof(double),
                              static_cast<size_t>(n - 1),
                              cudaMemcpyDeviceToHost, handle->stream));
    }

    EVD_RETURN_IF_CUDA_ERROR(handle, "tridiag_solver.cpu_sync",
                             cudaStreamSynchronize(handle->stream));

    for (int i = 0; i < n - 1; ++i) {
        h_e.get()[i] = std::abs(h_e.get()[i]);
    }

    char compz = state.config->compute_eigenvectors ? 'I' : 'N';
    double z_dummy = 0.0;
    double *z_ptr = &z_dummy;
    int ldz = 1;
    PinnedHostBuffer<double> h_z;

    if (state.config->compute_eigenvectors) {
        status = h_z.allocate(handle,
                              static_cast<size_t>(n) * static_cast<size_t>(n),
                              "tridiag_solver.cpu_alloc_z");
        if (status != EVD_STATUS_SUCCESS) {
            return status;
        }
        z_ptr = h_z.get();
        ldz = n;
    }

    int info = 0;
    int lwork_query = -1;
    int liwork_query = -1;
    double work_query = 0.0;
    int iwork_query = 0;

    dstedc_(&compz, &n, h_d.get(), h_e.get(), z_ptr, &ldz, &work_query,
            &lwork_query, &iwork_query, &liwork_query, &info);
    if (info != 0) {
        set_last_error(handle, info, EVD_ERROR_SRC_NONE,
                       "tridiag_solver.dstedc_query");
        return EVD_STATUS_INTERNAL_ERROR;
    }

    const int lwork = std::max(1, static_cast<int>(work_query));
    const int liwork = std::max(1, iwork_query);

    PinnedHostBuffer<double> work_host;
    status = work_host.allocate(handle, static_cast<size_t>(lwork),
                                "tridiag_solver.cpu_alloc_work");
    if (status != EVD_STATUS_SUCCESS) {
        return status;
    }

    PinnedHostBuffer<int> iwork_host;
    status = iwork_host.allocate(handle, static_cast<size_t>(liwork),
                                 "tridiag_solver.cpu_alloc_iwork");
    if (status != EVD_STATUS_SUCCESS) {
        return status;
    }

    dstedc_(&compz, &n, h_d.get(), h_e.get(), z_ptr, &ldz, work_host.get(),
            &lwork, iwork_host.get(), &liwork, &info);
    if (info != 0) {
        set_last_error(handle, info, EVD_ERROR_SRC_NONE,
                       "tridiag_solver.dstedc_compute");
        return EVD_STATUS_INTERNAL_ERROR;
    }

    EVD_RETURN_IF_CUDA_ERROR(
        handle, "tridiag_solver.cpu_copy_eigenvalues",
        cudaMemcpyAsync(state.d_W, h_d.get(), sizeof(double) * n,
                        cudaMemcpyHostToDevice, handle->stream));

    if (state.config->compute_eigenvectors && view.d_tridiag_eigenvectors != nullptr) {
        EVD_RETURN_IF_CUDA_ERROR(
            handle, "tridiag_solver.cpu_copy_eigenvectors",
            cudaMemcpyAsync(view.d_tridiag_eigenvectors, h_z.get(),
                            sizeof(double) * n * n,
                            cudaMemcpyHostToDevice, handle->stream));
    }

    return EVD_STATUS_SUCCESS;
}

} // namespace evd::internal
