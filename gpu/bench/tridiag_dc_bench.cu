#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "evd/handle.h"
#include "evd/status.h"
#include "evd/types.h"
#include "handle_impl.hpp"
#include "syev2stage_pipeline.hpp"
#include "tridiag_solver_dc.cuh"

extern "C" {
void dstedc_(const char *compz, const int *n, double *d, double *e, double *z,
             const int *ldz, double *work, const int *lwork, int *iwork,
             const int *liwork, int *info);
}

namespace {

struct HandleDeleter {
    void operator()(evd_handle handle) const {
        if (handle != nullptr) {
            (void)evd_destroy_handle(handle);
        }
    }
};

using HandleOwner =
    std::unique_ptr<std::remove_pointer_t<evd_handle>, HandleDeleter>;

struct CusolverHandleDeleter {
    void operator()(cusolverDnHandle_t handle) const {
        if (handle != nullptr) {
            (void)cusolverDnDestroy(handle);
        }
    }
};

using CusolverHandleOwner =
    std::unique_ptr<std::remove_pointer_t<cusolverDnHandle_t>,
                    CusolverHandleDeleter>;

struct CusolverParamsDeleter {
    void operator()(cusolverDnParams_t params) const {
        if (params != nullptr) {
            (void)cusolverDnDestroyParams(params);
        }
    }
};

using CusolverParamsOwner =
    std::unique_ptr<std::remove_pointer_t<cusolverDnParams_t>,
                    CusolverParamsDeleter>;

struct CpuWorkspace {
    int lwork{0};
    int liwork{0};
    std::vector<double> work;
    std::vector<int> iwork;
};

struct DeviceDoubleDeleter {
    void operator()(double *ptr) const {
        if (ptr != nullptr) {
            (void)cudaFree(ptr);
        }
    }
};

using DeviceDoubleOwner = std::unique_ptr<double, DeviceDoubleDeleter>;

struct GpuRunSummary {
    double eig_relative_error{0.0};
    double eigenpair_residual{0.0};
    double eigenvector_norm_max_error{0.0};
    double eigenvector_sample_orthogonality{0.0};
    double eigenvector_check_ms{0.0};
    evd_handle_struct::TridiagRunDiagnostics tridiag_diagnostics{};
};

struct BenchResult {
    double median_ms{0.0};
    double min_ms{0.0};
    double max_ms{0.0};
};

struct TridiagonalData {
    int64_t n{0};
    std::vector<double> diag;
    std::vector<double> subdiag;
};

struct PreparedInput {
    TridiagonalData tridiag;
    double prep_ms{0.0};
    std::string source;
};

double bytes_to_mib(std::size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

[[noreturn]] void throw_runtime(const std::string &message) {
    throw std::runtime_error(message);
}

void check_cuda(cudaError_t status, const char *what) {
    if (status != cudaSuccess) {
        std::ostringstream oss;
        oss << what << ": " << cudaGetErrorString(status);
        throw_runtime(oss.str());
    }
}

void check_evd(evd_status status, evd_handle handle, const char *what) {
    if (status == EVD_STATUS_SUCCESS) {
        return;
    }
    int code = 0;
    evd_error_source source = EVD_ERROR_SRC_NONE;
    const char *site = nullptr;
    if (handle != nullptr) {
        (void)evd_get_last_error(handle, &code, &source, &site);
    }
    std::ostringstream oss;
    oss << what << ": evd_status=" << static_cast<int>(status)
        << " last_error_code=" << code << " source=" << static_cast<int>(source)
        << " site=" << (site != nullptr ? site : "<null>");
    throw_runtime(oss.str());
}

void check_cusolver(cusolverStatus_t status, const char *what) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::ostringstream oss;
        oss << what << ": cusolverStatus=" << static_cast<int>(status);
        throw_runtime(oss.str());
    }
}

HandleOwner make_handle() {
    evd_handle raw = nullptr;
    check_evd(evd_create_handle(&raw), nullptr, "evd_create_handle");
    return HandleOwner(raw);
}

CusolverHandleOwner make_cusolver_handle() {
    cusolverDnHandle_t handle = nullptr;
    check_cusolver(cusolverDnCreate(&handle), "cusolverDnCreate");
    return CusolverHandleOwner(handle);
}

CusolverParamsOwner make_cusolver_params() {
    cusolverDnParams_t params = nullptr;
    check_cusolver(cusolverDnCreateParams(&params), "cusolverDnCreateParams");
    return CusolverParamsOwner(params);
}

std::vector<int64_t> parse_sizes(int argc, char **argv) {
    std::vector<int64_t> sizes = {512, 1024, 2048, 4096};
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg.rfind("--sizes=", 0) == 0) {
            sizes.clear();
            std::stringstream ss(arg.substr(8));
            std::string token;
            while (std::getline(ss, token, ',')) {
                if (!token.empty()) {
                    sizes.push_back(std::stoll(token));
                }
            }
        }
    }
    return sizes;
}

int parse_int_flag(int argc, char **argv, const char *name, int default_value) {
    const std::string prefix = std::string(name) + "=";
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg.rfind(prefix, 0) == 0) {
            return std::stoi(arg.substr(prefix.size()));
        }
    }
    return default_value;
}

std::string parse_string_flag(int argc, char **argv, const char *name,
                              const char *default_value = "") {
    const std::string prefix = std::string(name) + "=";
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg.rfind(prefix, 0) == 0) {
            return arg.substr(prefix.size());
        }
    }
    return default_value;
}

std::vector<std::string> split_csv(const std::string &text) {
    std::vector<std::string> values;
    std::stringstream ss(text);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            values.push_back(token);
        }
    }
    return values;
}

void make_bench_tridiagonal_data(int64_t n, uint32_t seed,
                                 std::vector<double> *diag_out,
                                 std::vector<double> *subdiag_out) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> diag_noise(-0.5, 0.5);
    std::uniform_real_distribution<double> offdiag_dist(0.05, 0.95);

    std::vector<double> diag(static_cast<std::size_t>(n));
    std::vector<double> subdiag(
        static_cast<std::size_t>(std::max<int64_t>(1, n - 1)));
    for (int64_t i = 0; i < n; ++i) {
        diag[static_cast<std::size_t>(i)] = static_cast<double>(n) +
                                            0.25 * static_cast<double>(i) +
                                            diag_noise(rng);
    }
    for (int64_t i = 0; i + 1 < n; ++i) {
        subdiag[static_cast<std::size_t>(i)] = offdiag_dist(rng);
    }

    if (diag_out != nullptr) {
        *diag_out = diag;
    }
    if (subdiag_out != nullptr) {
        *subdiag_out = std::vector<double>(
            subdiag.begin(), subdiag.begin() + std::max<int64_t>(0, n - 1));
    }
}

void make_paper_tridiagonal_data(const std::string &mode, int64_t n,
                                 std::vector<double> *diag_out,
                                 std::vector<double> *subdiag_out) {
    std::vector<double> diag(static_cast<std::size_t>(n));
    std::vector<double> subdiag(
        static_cast<std::size_t>(std::max<int64_t>(0, n - 1)));

    const uint64_t seed_base = static_cast<uint64_t>(n) * 0x9E3779B97F4A7C15ULL;
    const uint64_t mode_hash =
        (mode == "normal") ? 0xD1B54A32D192ED03ULL : 0x94D049BB133111EBULL;
    std::mt19937_64 rng(seed_base ^ mode_hash);
    std::normal_distribution<double> normal_diag(0.0, 1.0);
    std::normal_distribution<double> normal_offdiag(0.0, 0.20);
    std::uniform_real_distribution<double> uniform_diag(-1.0, 1.0);
    std::uniform_real_distribution<double> uniform_offdiag(0.10, 0.30);

    for (int64_t i = 0; i < n; ++i) {
        if (mode == "normal") {
            diag[static_cast<std::size_t>(i)] = normal_diag(rng);
        } else if (mode == "uniform") {
            diag[static_cast<std::size_t>(i)] = uniform_diag(rng);
        } else if (mode == "toeplitz") {
            diag[static_cast<std::size_t>(i)] = 2.0;
        } else if (mode == "clustered") {
            const double one_based_i = static_cast<double>(i + 1);
            diag[static_cast<std::size_t>(i)] =
                1.0 + 1.0e-12 * (one_based_i - 0.5 * (static_cast<double>(n) + 1.0));
        } else {
            throw_runtime("unsupported --data-mode entry: " + mode);
        }
    }

    for (int64_t i = 0; i + 1 < n; ++i) {
        if (mode == "normal") {
            subdiag[static_cast<std::size_t>(i)] = std::abs(normal_offdiag(rng));
        } else if (mode == "uniform") {
            subdiag[static_cast<std::size_t>(i)] = uniform_offdiag(rng);
        } else if (mode == "toeplitz") {
            subdiag[static_cast<std::size_t>(i)] = 0.25;
        } else if (mode == "clustered") {
            const double one_based_i = static_cast<double>(i + 1);
            subdiag[static_cast<std::size_t>(i)] =
                1.0e-4 * (1.0 + 0.1 * std::cos(0.33 * one_based_i));
        }
    }

    if (diag_out != nullptr) {
        *diag_out = diag;
    }
    if (subdiag_out != nullptr) {
        *subdiag_out = subdiag;
    }
}

void make_test_tridiagonal_data(int64_t n, uint32_t seed,
                                std::vector<double> *diag_out,
                                std::vector<double> *subdiag_out) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    std::vector<double> diag(static_cast<std::size_t>(n));
    std::vector<double> subdiag(
        static_cast<std::size_t>(std::max<int64_t>(0, n - 1)));
    for (int64_t col = 0; col < n; ++col) {
        for (int64_t row = 0; row < n; ++row) {
            const double value = distribution(generator);
            if (row == col) {
                diag[static_cast<std::size_t>(col)] =
                    value + static_cast<double>(n);
            }
            if (row == col + 1) {
                subdiag[static_cast<std::size_t>(col)] = value;
            }
        }
    }

    if (diag_out != nullptr) {
        *diag_out = diag;
    }
    if (subdiag_out != nullptr) {
        *subdiag_out = subdiag;
    }
}

std::vector<double> read_matrix_file(const std::string &path, int64_t *n_out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw_runtime("failed to open input file: " + path);
    }

    int64_t n = 0;
    in.read(reinterpret_cast<char *>(&n), sizeof(n));
    if (!in || n <= 0) {
        throw_runtime("invalid matrix header in file: " + path);
    }

    std::vector<double> matrix(static_cast<std::size_t>(n * n), 0.0);
    in.read(reinterpret_cast<char *>(matrix.data()),
            static_cast<std::streamsize>(sizeof(double) * matrix.size()));
    if (!in) {
        throw_runtime("failed to read matrix payload from file: " + path);
    }

    if (n_out != nullptr) {
        *n_out = n;
    }
    return matrix;
}

std::uint64_t file_size_bytes(std::ifstream *in) {
    in->seekg(0, std::ios::end);
    const std::streamoff size = in->tellg();
    in->seekg(0, std::ios::beg);
    if (size < 0) {
        throw_runtime("failed to query input file size");
    }
    return static_cast<std::uint64_t>(size);
}

TridiagonalData extract_tridiagonal_data(const std::vector<double> &matrix,
                                         int64_t n) {
    TridiagonalData data;
    data.n = n;
    data.diag.resize(static_cast<std::size_t>(n));
    data.subdiag.resize(static_cast<std::size_t>(std::max<int64_t>(0, n - 1)));
    for (int64_t i = 0; i < n; ++i) {
        data.diag[static_cast<std::size_t>(i)] =
            matrix[static_cast<std::size_t>(i + i * n)];
        if (i + 1 < n) {
            data.subdiag[static_cast<std::size_t>(i)] =
                matrix[static_cast<std::size_t>((i + 1) + i * n)];
        }
    }
    return data;
}

TridiagonalData read_tridiagonal_file(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw_runtime("failed to open tridiagonal input file: " + path);
    }

    const std::uint64_t total_bytes = file_size_bytes(&in);
    int64_t n = 0;
    in.read(reinterpret_cast<char *>(&n), sizeof(n));
    if (!in || n <= 0) {
        throw_runtime("invalid tridiagonal header in file: " + path);
    }

    const std::uint64_t compact_payload =
        static_cast<std::uint64_t>(2 * n - 1) * sizeof(double);
    const std::uint64_t dense_payload =
        static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(n) *
        sizeof(double);
    const std::uint64_t payload_bytes = total_bytes - sizeof(n);

    if (payload_bytes == compact_payload) {
        TridiagonalData data;
        data.n = n;
        data.diag.resize(static_cast<std::size_t>(n));
        data.subdiag.resize(static_cast<std::size_t>(std::max<int64_t>(0, n - 1)));
        in.read(reinterpret_cast<char *>(data.diag.data()),
                static_cast<std::streamsize>(sizeof(double) * data.diag.size()));
        if (!data.subdiag.empty()) {
            in.read(reinterpret_cast<char *>(data.subdiag.data()),
                    static_cast<std::streamsize>(sizeof(double) * data.subdiag.size()));
        }
        if (!in) {
            throw_runtime("failed to read compact tridiagonal payload: " + path);
        }
        return data;
    }

    if (payload_bytes == dense_payload) {
        std::vector<double> matrix(static_cast<std::size_t>(n * n), 0.0);
        in.read(reinterpret_cast<char *>(matrix.data()),
                static_cast<std::streamsize>(sizeof(double) * matrix.size()));
        if (!in) {
            throw_runtime("failed to read dense tridiagonal payload: " + path);
        }
        return extract_tridiagonal_data(matrix, n);
    }

    std::ostringstream oss;
    oss << "unsupported tridiagonal file payload bytes=" << payload_bytes
        << " for n=" << n << " in " << path;
    throw_runtime(oss.str());
}

void write_tridiagonal_file(const std::string &path,
                            const TridiagonalData &data) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw_runtime("failed to open tridiagonal output file: " + path);
    }

    out.write(reinterpret_cast<const char *>(&data.n), sizeof(data.n));
    out.write(reinterpret_cast<const char *>(data.diag.data()),
              static_cast<std::streamsize>(sizeof(double) * data.diag.size()));
    if (!data.subdiag.empty()) {
        out.write(reinterpret_cast<const char *>(data.subdiag.data()),
                  static_cast<std::streamsize>(sizeof(double) * data.subdiag.size()));
    }
    if (!out) {
        throw_runtime("failed to write tridiagonal output file: " + path);
    }
}

PreparedInput
prepare_tridiagonal_from_dense_matrix(const std::vector<double> &host_matrix,
                                      int64_t n, int64_t band_size,
                                      int64_t bulge_block_size) {
    PreparedInput prepared{};
    prepared.source = "dense_bin";

    auto handle = make_handle();
    evd_syev2stage_config config{};
    config.n = n;
    config.band_size = band_size;
    config.bulge_block_size = bulge_block_size;
    config.uplo = EVD_FILL_MODE_LOWER;
    config.compute_eigenvectors = 0;
    config.enable_timing = 0;
    config.tridiag_solver_backend = EVD_TRIG_SOLVER_BACKEND_AUTO;
    config.tridiag_sort_backend = EVD_TRIDIAG_SORT_BACKEND_CUB;
    config.enable_tridiag_diagnostics = 0;

    size_t reduction_bytes = 0;
    size_t bulge_bytes = 0;
    check_evd(evd::internal::reduction_to_band_workspace_size(config,
                                                              &reduction_bytes),
              handle.get(), "reduction_to_band_workspace_size");
    check_evd(evd::internal::bulge_chasing_workspace_size(config, &bulge_bytes),
              handle.get(), "bulge_chasing_workspace_size");

    const size_t workspace_bytes = std::max(reduction_bytes, bulge_bytes);
    double *d_matrix = nullptr;
    double *d_w = nullptr;
    void *workspace = nullptr;
    check_cuda(
        cudaMalloc(&d_matrix, sizeof(double) * static_cast<std::size_t>(n * n)),
        "cudaMalloc dense d_matrix");
    check_cuda(cudaMalloc(&d_w, sizeof(double) * static_cast<std::size_t>(n)),
               "cudaMalloc dense d_w");
    check_cuda(cudaMalloc(&workspace, workspace_bytes),
               "cudaMalloc dense workspace");
    check_cuda(cudaMemcpy(d_matrix, host_matrix.data(),
                          sizeof(double) * static_cast<std::size_t>(n * n),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy dense H2D");

    evd::internal::Syev2StageState state{};
    state.config = &config;
    state.d_A = d_matrix;
    state.lda = n;
    state.d_W = d_w;
    state.d_Q = nullptr;

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate dense prep start");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate dense prep stop");
    check_cuda(cudaEventRecord(start, handle->stream),
               "cudaEventRecord dense prep start");
    check_evd(evd::internal::reduction_to_band_run(handle.get(), state,
                                                   workspace, workspace_bytes),
              handle.get(), "reduction_to_band_run");
    check_evd(evd::internal::bulge_chasing_run(handle.get(), state, workspace,
                                               workspace_bytes),
              handle.get(), "bulge_chasing_run");
    check_cuda(cudaEventRecord(stop, handle->stream),
               "cudaEventRecord dense prep stop");
    check_cuda(cudaEventSynchronize(stop),
               "cudaEventSynchronize dense prep stop");
    float prep_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&prep_ms, start, stop),
               "cudaEventElapsedTime dense prep");

    std::vector<double> tridiag_matrix(static_cast<std::size_t>(n * n), 0.0);
    check_cuda(cudaMemcpy(tridiag_matrix.data(), d_matrix,
                          sizeof(double) * static_cast<std::size_t>(n * n),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy dense D2H");

    prepared.tridiag = extract_tridiagonal_data(tridiag_matrix, n);
    prepared.prep_ms = static_cast<double>(prep_ms);

    check_cuda(cudaEventDestroy(start), "cudaEventDestroy dense prep start");
    check_cuda(cudaEventDestroy(stop), "cudaEventDestroy dense prep stop");
    (void)cudaFree(workspace);
    (void)cudaFree(d_w);
    (void)cudaFree(d_matrix);
    return prepared;
}

CpuWorkspace make_cpu_workspace(int n, bool compute_vectors,
                                const std::vector<double> &base_diag,
                                const std::vector<double> &base_subdiag) {
    CpuWorkspace ws;
    std::vector<double> diag = base_diag;
    std::vector<double> subdiag = base_subdiag;
    char compz = compute_vectors ? 'I' : 'N';
    double z_dummy = 0.0;
    double work_query = 0.0;
    int iwork_query = 0;
    int info = 0;
    int lwork_query = -1;
    int liwork_query = -1;
    int ldz = compute_vectors ? n : 1;
    std::vector<double> z_buffer(
        compute_vectors ? static_cast<std::size_t>(n * n) : 1, 0.0);
    double *z_ptr = compute_vectors ? z_buffer.data() : &z_dummy;
    dstedc_(&compz, &n, diag.data(), subdiag.empty() ? nullptr : subdiag.data(),
            z_ptr, &ldz, &work_query, &lwork_query, &iwork_query, &liwork_query,
            &info);
    if (info != 0) {
        throw_runtime("dstedc_ workspace query failed");
    }
    ws.lwork = std::max(1, static_cast<int>(work_query));
    ws.liwork = std::max(1, iwork_query);
    ws.work.resize(static_cast<std::size_t>(ws.lwork));
    ws.iwork.resize(static_cast<std::size_t>(ws.liwork));
    return ws;
}

BenchResult summarize(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    BenchResult out{};
    out.min_ms = samples.front();
    out.max_ms = samples.back();
    out.median_ms = samples[samples.size() / 2];
    return out;
}

void print_tridiag_diagnostics(
    const evd_handle_struct::TridiagRunDiagnostics &diag, const char *label) {
    if (!diag.available) {
        return;
    }

    std::cout << "\n  [" << label
              << " tridiag diagnostics: total_merges=" << diag.total_merges
              << " residual_failed_merges=" << diag.residual_failed_merges
              << "]";
    for (const auto &level : diag.levels) {
        const double before_avg =
            level.observed_merges > 0
                ? static_cast<double>(level.dd_before_sum) /
                      static_cast<double>(level.observed_merges)
                : 0.0;
        const double after_avg =
            level.observed_merges > 0
                ? static_cast<double>(level.dd_after_sum) /
                      static_cast<double>(level.observed_merges)
                : 0.0;
        const double size_avg =
            level.observed_merges > 0
                ? static_cast<double>(level.size_sum) /
                      static_cast<double>(level.observed_merges)
                : 0.0;
        const double zero_deflation_pct =
            level.size_sum > 0
                ? 100.0 * static_cast<double>(level.zero_deflated_total) /
                      static_cast<double>(level.size_sum)
                : 0.0;
        const double close_deflation_pct =
            level.size_sum > 0
                ? 100.0 * static_cast<double>(level.close_deflated_total) /
                      static_cast<double>(level.size_sum)
                : 0.0;
        const double deflation_pct =
            level.size_sum > 0
                ? 100.0 * static_cast<double>(level.deflated_total) /
                      static_cast<double>(level.size_sum)
                : 0.0;
        const double active_ratio =
            level.size_sum > 0 ? static_cast<double>(level.dd_after_sum) /
                                     static_cast<double>(level.size_sum)
                               : 0.0;
        const double secular_iter_avg =
            level.secular_iter_count > 0
                ? static_cast<double>(level.secular_iter_sum) /
                      static_cast<double>(level.secular_iter_count)
                : 0.0;
        const double vector_gflop_est =
            2.0e-9 * static_cast<double>(level.vector_mnk_sum);
        const double split_left_gflop_est =
            2.0e-9 * static_cast<double>(level.split_left_mnk_sum);
        const double split_right_gflop_est =
            2.0e-9 * static_cast<double>(level.split_right_mnk_sum);
        const bool has_timing =
            level.values_solve_ms != 0.0f || level.rescale_ms != 0.0f ||
            level.values_active_roots_ms != 0.0f ||
            level.values_deflated_boundary_ms != 0.0f ||
            level.values_copyd_ms != 0.0f ||
            level.values_secular_solve_ms != 0.0f ||
            level.values_rescale_ms != 0.0f ||
            level.values_boundary_prepare_ms != 0.0f ||
            level.values_boundary_update_ms != 0.0f ||
            level.copyc_ms != 0.0f || level.reshufflec_ms != 0.0f ||
            level.vector_prepare_ms != 0.0f || level.gemm_ms != 0.0f ||
            level.update_ms != 0.0f || level.global_copyc_ms != 0.0f ||
            level.global_reshufflec_ms != 0.0f ||
            level.deflated_gather_ms != 0.0f ||
            level.split_make_indices_ms != 0.0f ||
            level.split_left_pack_ms != 0.0f ||
            level.split_left_gemm_ms != 0.0f ||
            level.split_right_pack_ms != 0.0f ||
            level.split_right_gemm_ms != 0.0f ||
            level.merge_update_ms != 0.0f ||
            level.update_all_vectors_ms != 0.0f;
        std::cout
            << "\n    level=" << level.level << " merges=" << level.merge_count
            << " observed=" << level.observed_merges
            << " size[min/avg/max]=" << level.size_min << "/" << std::fixed
            << std::setprecision(1) << size_avg << "/" << level.size_max
            << " dd_before[min/avg/max]=" << level.dd_before_min << "/"
            << before_avg << "/" << level.dd_before_max
            << " dd_after[min/avg/max]=" << level.dd_after_min << "/"
            << after_avg << "/" << level.dd_after_max
            << " deflation[zero/close/total]=" << level.zero_deflated_total
            << "/" << level.close_deflated_total << "/" << level.deflated_total
            << " (" << zero_deflation_pct << "%/" << close_deflation_pct << "%/"
            << deflation_pct << "%)"
            << " active_ratio=" << std::setprecision(3) << active_ratio
            << " secular_iter[count/avg/max/fail]="
            << level.secular_iter_count << "/" << secular_iter_avg << "/"
            << level.secular_iter_max << "/" << level.secular_iter_failed
            << std::setprecision(1)
            << " vector_tasks=" << level.vector_task_count
            << " batch_groups=" << level.vector_batch_groups
            << " batched_tasks=" << level.vector_batched_tasks
            << " single_tasks=" << level.vector_single_tasks
            << " deflated_tasks=" << level.vector_deflated_tasks
            << " no_deflation_tasks=" << level.vector_no_deflation_tasks
            << " max(size,dd)=" << level.vector_max_size << ","
            << level.vector_max_dd << " gemm_mnk_sum=" << level.vector_mnk_sum
            << " gemm_gflop_est=" << vector_gflop_est
            << " split_mnk[left/right]=" << level.split_left_mnk_sum << "/"
            << level.split_right_mnk_sum
            << " split_shape_max[left/right MxNxK]=" << level.split_left_max_m
            << "x" << level.split_left_max_n << "x" << level.split_left_max_k
            << "/" << level.split_right_max_m << "x" << level.split_right_max_n
            << "x" << level.split_right_max_k;
        if (has_timing) {
            const double gemm_tflops =
                level.gemm_ms > 0.0f ? vector_gflop_est / level.gemm_ms : 0.0;
            const double left_tflops =
                level.split_left_gemm_ms > 0.0f
                    ? split_left_gflop_est / level.split_left_gemm_ms
                    : 0.0;
            const double right_tflops =
                level.split_right_gemm_ms > 0.0f
                    ? split_right_gflop_est / level.split_right_gemm_ms
                    : 0.0;
            std::cout << " timing_ms[val/rescale/copy/reshuf/prep/gemm/update]="
                      << std::fixed << std::setprecision(3)
                      << level.values_solve_ms << "/" << level.rescale_ms << "/"
                      << level.copyc_ms << "/" << level.reshufflec_ms << "/"
                      << level.vector_prepare_ms << "/" << level.gemm_ms << "/"
                      << level.update_ms
                      << " values_detail_ms[active_roots/defl_boundary/copyd/"
                         "solve/rescale/boundary_prepare/boundary_update]="
                      << level.values_active_roots_ms << "/"
                      << level.values_deflated_boundary_ms << "/"
                      << level.values_copyd_ms << "/"
                      << level.values_secular_solve_ms << "/"
                      << level.values_rescale_ms << "/"
                      << level.values_boundary_prepare_ms << "/"
                      << level.values_boundary_update_ms
                      << " detail_ms[global_copy/global_reshuf/defl_gather/"
                         "make_idx/left_pack/left_gemm/right_pack/"
                         "right_gemm/merge_update/update_all]="
                      << level.global_copyc_ms << "/"
                      << level.global_reshufflec_ms << "/"
                      << level.deflated_gather_ms << "/"
                      << level.split_make_indices_ms << "/"
                      << level.split_left_pack_ms << "/"
                      << level.split_left_gemm_ms << "/"
                      << level.split_right_pack_ms << "/"
                      << level.split_right_gemm_ms << "/"
                      << level.merge_update_ms << "/"
                      << level.update_all_vectors_ms
                      << " gemm_tflops[total/left/right]=" << gemm_tflops << "/"
                      << left_tflops << "/" << right_tflops;
        }
    }
    std::cout << '\n';
}

double relative_error(const std::vector<double> &left,
                      const std::vector<double> &right) {
    double num = 0.0;
    double den = 0.0;
    for (std::size_t i = 0; i < left.size(); ++i) {
        const double delta = left[i] - right[i];
        num += delta * delta;
        den += right[i] * right[i];
    }
    return std::sqrt(num / std::max(den, 1.0e-30));
}

std::vector<double> absolutize_subdiag(const std::vector<double> &subdiag) {
    std::vector<double> result = subdiag;
    for (double &value : result) {
        value = std::abs(value);
    }
    return result;
}

double tridiagonal_frobenius_norm(const std::vector<double> &diag,
                                  const std::vector<double> &subdiag) {
    double norm2 = 0.0;
    for (const double value : diag) {
        norm2 += value * value;
    }
    for (const double value : subdiag) {
        norm2 += 2.0 * value * value;
    }
    return std::sqrt(norm2);
}

DeviceDoubleOwner make_device_double_buffer(std::size_t count,
                                            const char *what) {
    if (count == 0) {
        return DeviceDoubleOwner(nullptr);
    }
    double *ptr = nullptr;
    check_cuda(cudaMalloc(&ptr, sizeof(double) * count), what);
    return DeviceDoubleOwner(ptr);
}

constexpr int kVectorCheckBlockSize = 256;

__global__ void vector_residual_norm_kernel(std::int64_t n, const double *diag,
                                            const double *subdiag,
                                            const double *eigs, const double *q,
                                            double *residual2_cols,
                                            double *norm_err_cols) {
    __shared__ double residual_s[kVectorCheckBlockSize];
    __shared__ double norm_s[kVectorCheckBlockSize];

    const std::int64_t col = static_cast<std::int64_t>(blockIdx.x);
    const int tid = threadIdx.x;
    double residual2 = 0.0;
    double norm2 = 0.0;
    if (col < n) {
        const double lambda = eigs[col];
        const std::int64_t col_offset = col * n;
        for (std::int64_t row = tid; row < n; row += blockDim.x) {
            const std::int64_t idx = col_offset + row;
            const double qv = q[idx];
            double tq = diag[row] * qv;
            if (row > 0) {
                tq += fabs(subdiag[row - 1]) * q[idx - 1];
            }
            if (row + 1 < n) {
                tq += fabs(subdiag[row]) * q[idx + 1];
            }
            const double diff = tq - lambda * qv;
            residual2 += diff * diff;
            norm2 += qv * qv;
        }
    }

    residual_s[tid] = residual2;
    norm_s[tid] = norm2;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            residual_s[tid] += residual_s[tid + stride];
            norm_s[tid] += norm_s[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0 && col < n) {
        residual2_cols[col] = residual_s[0];
        norm_err_cols[col] = fabs(norm_s[0] - 1.0);
    }
}

__global__ void vector_sample_orthogonality_kernel(std::int64_t n,
                                                   const double *q, int offset,
                                                   double *orth_vals) {
    __shared__ double dot_s[kVectorCheckBlockSize];

    const std::int64_t col = static_cast<std::int64_t>(blockIdx.x);
    const std::int64_t other = col + static_cast<std::int64_t>(offset);
    const int tid = threadIdx.x;
    double dot = 0.0;
    if (other < n) {
        const std::int64_t col_offset = col * n;
        const std::int64_t other_offset = other * n;
        for (std::int64_t row = tid; row < n; row += blockDim.x) {
            dot += q[col_offset + row] * q[other_offset + row];
        }
    }

    dot_s[tid] = dot;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            dot_s[tid] += dot_s[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0 && other < n) {
        orth_vals[col] = fabs(dot_s[0]);
    }
}

BenchResult bench_cpu_dstedc(const std::vector<double> &base_diag,
                             const std::vector<double> &base_subdiag,
                             bool compute_vectors, int warmup, int iters,
                             std::vector<double> *last_eigs_out) {
    const int n = static_cast<int>(base_diag.size());
    CpuWorkspace ws =
        make_cpu_workspace(n, compute_vectors, base_diag, base_subdiag);
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(iters));
    std::vector<double> eigs(static_cast<std::size_t>(n));
    std::vector<double> z(compute_vectors ? static_cast<std::size_t>(n * n) : 1,
                          0.0);
    char compz = compute_vectors ? 'I' : 'N';
    const int ldz = compute_vectors ? n : 1;

    for (int rep = 0; rep < warmup + iters; ++rep) {
        std::vector<double> diag = base_diag;
        std::vector<double> subdiag = base_subdiag;
        int info = 0;
        auto t0 = std::chrono::steady_clock::now();
        dstedc_(&compz, &n, diag.data(),
                subdiag.empty() ? nullptr : subdiag.data(), z.data(), &ldz,
                ws.work.data(), &ws.lwork, ws.iwork.data(), &ws.liwork, &info);
        auto t1 = std::chrono::steady_clock::now();
        if (info != 0) {
            throw_runtime("dstedc_ compute failed");
        }
        eigs = diag;
        if (rep >= warmup) {
            const double ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            times.push_back(ms);
        }
    }

    if (last_eigs_out != nullptr) {
        *last_eigs_out = eigs;
    }
    return summarize(std::move(times));
}

struct GpuBenchContext {
    HandleOwner handle;
    evd_syev2stage_config config{};
    std::size_t workspace_bytes{0};
    double *d_diag_base{nullptr};
    double *d_subdiag_base{nullptr};
    double *d_w{nullptr};
    void *workspace{nullptr};
};

struct CusolverBenchContext {
    CusolverHandleOwner handle;
    CusolverParamsOwner params;
    const double *h_diag_base{nullptr};
    const double *h_subdiag_base{nullptr};
    std::vector<double> h_z_base;
    double *d_diag{nullptr};
    double *d_subdiag{nullptr};
    double *d_z{nullptr};
    int *d_info{nullptr};
    void *d_work{nullptr};
    void *h_work{nullptr};
    size_t d_work_bytes{0};
    size_t h_work_bytes{0};
    int64_t n{0};
    cusolverEigComp_t compz{CUSOLVER_EIG_COMP_I};
};

GpuBenchContext make_gpu_context(const std::vector<double> &diag,
                                 const std::vector<double> &subdiag, int64_t n,
                                 bool compute_vectors, int backend,
                                 bool enable_tridiag_diagnostics) {
    GpuBenchContext ctx{};
    ctx.handle = make_handle();
    ctx.config.n = n;
    ctx.config.band_size = 1;
    ctx.config.bulge_block_size = 1;
    ctx.config.uplo = EVD_FILL_MODE_LOWER;
    ctx.config.compute_eigenvectors = compute_vectors ? 1 : 0;
    ctx.config.tridiag_solver_backend =
        static_cast<evd_tridiag_solver_backend>(backend);
    ctx.config.enable_timing = 0;
    ctx.config.enable_tridiag_diagnostics = enable_tridiag_diagnostics ? 1 : 0;

    check_evd(evd::internal::tridiag_solver_workspace_size(
                  ctx.config, &ctx.workspace_bytes),
              ctx.handle.get(), "tridiag_solver_workspace_size");
    check_cuda(cudaMalloc(&ctx.d_diag_base,
                          sizeof(double) * static_cast<std::size_t>(n)),
               "cudaMalloc d_diag_base");
    check_cuda(cudaMalloc(&ctx.d_subdiag_base,
                          sizeof(double) * static_cast<std::size_t>(
                                               std::max<int64_t>(1, n - 1))),
               "cudaMalloc d_subdiag_base");
    check_cuda(
        cudaMalloc(&ctx.d_w, sizeof(double) * static_cast<std::size_t>(n)),
        "cudaMalloc d_w");
    check_cuda(cudaMalloc(&ctx.workspace, ctx.workspace_bytes),
               "cudaMalloc workspace");
    check_cuda(cudaMemcpy(ctx.d_diag_base, diag.data(),
                          sizeof(double) * static_cast<std::size_t>(n),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy H2D diag_base");
    if (n > 1) {
        check_cuda(cudaMemcpy(ctx.d_subdiag_base, subdiag.data(),
                              sizeof(double) * static_cast<std::size_t>(n - 1),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy H2D subdiag_base");
    }
    return ctx;
}

void destroy_gpu_context(GpuBenchContext *ctx) {
    if (ctx == nullptr) {
        return;
    }
    if (ctx->workspace != nullptr) {
        (void)cudaFree(ctx->workspace);
    }
    if (ctx->d_w != nullptr) {
        (void)cudaFree(ctx->d_w);
    }
    if (ctx->d_subdiag_base != nullptr) {
        (void)cudaFree(ctx->d_subdiag_base);
    }
    if (ctx->d_diag_base != nullptr) {
        (void)cudaFree(ctx->d_diag_base);
    }
}

std::vector<double> copy_gpu_eigenvalues(const GpuBenchContext &ctx) {
    std::vector<double> eigs(static_cast<std::size_t>(ctx.config.n));
    check_cuda(
        cudaMemcpy(eigs.data(), ctx.d_w,
                   sizeof(double) * static_cast<std::size_t>(ctx.config.n),
                   cudaMemcpyDeviceToHost),
        "cudaMemcpy final gpu eigs");
    return eigs;
}

void fill_gpu_vector_checks_device(const std::vector<double> &diag,
                                   const std::vector<double> &subdiag,
                                   const GpuBenchContext &ctx,
                                   GpuRunSummary *summary) {
    if (summary == nullptr || ctx.config.n <= 0 ||
        ctx.config.compute_eigenvectors == 0) {
        return;
    }

    const auto check_start = std::chrono::steady_clock::now();
    evd::internal::TridiagSolverWorkspaceLayout layout{};
    check_evd(evd::internal::make_tridiag_solver_workspace_layout(ctx.config,
                                                                  &layout),
              ctx.handle.get(), "make_tridiag_solver_workspace_layout");
    if (layout.eigenvectors_bytes == 0) {
        return;
    }

    const auto *workspace = static_cast<const unsigned char *>(ctx.workspace);
    const auto *d_q = reinterpret_cast<const double *>(
        workspace + layout.eigenvectors_offset);
    const std::int64_t n = ctx.config.n;
    const std::size_t n_size = static_cast<std::size_t>(n);
    const cudaStream_t stream = ctx.handle->stream;
    DeviceDoubleOwner d_residual2 =
        make_device_double_buffer(n_size, "cudaMalloc vector check residual");
    DeviceDoubleOwner d_norm_err =
        make_device_double_buffer(n_size, "cudaMalloc vector check norm");

    vector_residual_norm_kernel<<<static_cast<unsigned int>(n),
                                  kVectorCheckBlockSize, 0, stream>>>(
        n, ctx.d_diag_base, ctx.d_subdiag_base, ctx.d_w, d_q, d_residual2.get(),
        d_norm_err.get());
    check_cuda(cudaGetLastError(), "launch vector_residual_norm_kernel");

    std::vector<double> residual2_cols(n_size);
    std::vector<double> norm_err_cols(n_size);
    check_cuda(cudaMemcpyAsync(residual2_cols.data(), d_residual2.get(),
                               sizeof(double) * n_size, cudaMemcpyDeviceToHost,
                               stream),
               "cudaMemcpyAsync vector check residual");
    check_cuda(cudaMemcpyAsync(norm_err_cols.data(), d_norm_err.get(),
                               sizeof(double) * n_size, cudaMemcpyDeviceToHost,
                               stream),
               "cudaMemcpyAsync vector check norm");

    const int offsets[] = {1, 7, 31, 127, 511};
    std::vector<std::size_t> orth_bases;
    std::vector<int> active_offsets;
    std::size_t orth_count = 0;
    for (const int offset : offsets) {
        if (offset < n) {
            orth_bases.push_back(orth_count);
            active_offsets.push_back(offset);
            orth_count += static_cast<std::size_t>(n - offset);
        }
    }

    DeviceDoubleOwner d_orth_vals =
        make_device_double_buffer(orth_count, "cudaMalloc vector check orth");
    for (std::size_t i = 0; i < active_offsets.size(); ++i) {
        const int offset = active_offsets[i];
        vector_sample_orthogonality_kernel<<<
            static_cast<unsigned int>(n - offset), kVectorCheckBlockSize, 0,
            stream>>>(n, d_q, offset, d_orth_vals.get() + orth_bases[i]);
        check_cuda(cudaGetLastError(),
                   "launch vector_sample_orthogonality_kernel");
    }

    std::vector<double> orth_vals(orth_count);
    if (orth_count > 0) {
        check_cuda(cudaMemcpyAsync(orth_vals.data(), d_orth_vals.get(),
                                   sizeof(double) * orth_count,
                                   cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync vector check orth");
    }
    check_cuda(cudaStreamSynchronize(stream),
               "cudaStreamSynchronize vector check");

    double residual2 = 0.0;
    for (const double value : residual2_cols) {
        residual2 += value;
    }
    double norm_max = 0.0;
    for (const double value : norm_err_cols) {
        norm_max = std::max(norm_max, value);
    }
    double orth_max = 0.0;
    for (const double value : orth_vals) {
        orth_max = std::max(orth_max, value);
    }

    const std::vector<double> abs_subdiag = absolutize_subdiag(subdiag);
    summary->eigenpair_residual =
        std::sqrt(residual2) /
        std::max(tridiagonal_frobenius_norm(diag, abs_subdiag), 1.0e-16);
    summary->eigenvector_norm_max_error = norm_max;
    summary->eigenvector_sample_orthogonality = orth_max;
    const auto check_stop = std::chrono::steady_clock::now();
    summary->eigenvector_check_ms =
        std::chrono::duration<double, std::milli>(check_stop - check_start)
            .count();
}

CusolverBenchContext make_cusolver_context(const std::vector<double> &diag,
                                           const std::vector<double> &subdiag,
                                           int64_t n, bool compute_vectors) {
    CusolverBenchContext ctx{};
    ctx.handle = make_cusolver_handle();
    ctx.params = make_cusolver_params();
    ctx.h_diag_base = diag.data();
    ctx.h_subdiag_base = subdiag.data();
    ctx.n = n;
    ctx.compz = compute_vectors ? CUSOLVER_EIG_COMP_I : CUSOLVER_EIG_COMP_N;

    check_cuda(
        cudaMalloc(&ctx.d_diag, sizeof(double) * static_cast<std::size_t>(n)),
        "cusolver cudaMalloc d_diag");
    check_cuda(cudaMalloc(&ctx.d_subdiag,
                          sizeof(double) * static_cast<std::size_t>(
                                               std::max<int64_t>(1, n - 1))),
               "cusolver cudaMalloc d_subdiag");
    check_cuda(cudaMalloc(&ctx.d_info, sizeof(int)),
               "cusolver cudaMalloc d_info");

    if (compute_vectors) {
        ctx.h_z_base.assign(static_cast<std::size_t>(n * n), 0.0);
        for (int64_t i = 0; i < n; ++i) {
            ctx.h_z_base[static_cast<std::size_t>(i + i * n)] = 1.0;
        }
        check_cuda(cudaMalloc(&ctx.d_z,
                              sizeof(double) * static_cast<std::size_t>(n * n)),
                   "cusolver cudaMalloc d_z");
    }

    check_cusolver(cusolverDnXstedc_bufferSize(
                       ctx.handle.get(), ctx.params.get(), ctx.compz, n,
                       CUDA_R_64F, ctx.d_diag, ctx.d_subdiag, CUDA_R_64F,
                       ctx.d_z, n, CUDA_R_64F, &ctx.d_work_bytes,
                       &ctx.h_work_bytes),
                   "cusolverDnXstedc_bufferSize");

    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes),
               "cudaMemGetInfo before cusolver workspace alloc");
    std::cout << "cusolver workspace query"
              << " n=" << n
              << " compz=" << (ctx.compz == CUSOLVER_EIG_COMP_N ? "N" : "I")
              << " d_work_mib=" << std::fixed << std::setprecision(3)
              << bytes_to_mib(ctx.d_work_bytes)
              << " h_work_mib=" << bytes_to_mib(ctx.h_work_bytes)
              << " free_device_mib=" << bytes_to_mib(free_bytes)
              << " total_device_mib=" << bytes_to_mib(total_bytes) << '\n';

    if (ctx.d_work_bytes > 0) {
        check_cuda(cudaMalloc(&ctx.d_work, ctx.d_work_bytes),
                   "cusolver cudaMalloc d_work");
    }
    if (ctx.h_work_bytes > 0) {
        ctx.h_work = std::malloc(ctx.h_work_bytes);
        if (ctx.h_work == nullptr) {
            throw_runtime("cusolver malloc h_work failed");
        }
    }
    return ctx;
}

void destroy_cusolver_context(CusolverBenchContext *ctx) {
    if (ctx == nullptr) {
        return;
    }
    if (ctx->d_work != nullptr) {
        (void)cudaFree(ctx->d_work);
    }
    if (ctx->h_work != nullptr) {
        std::free(ctx->h_work);
    }
    if (ctx->d_info != nullptr) {
        (void)cudaFree(ctx->d_info);
    }
    if (ctx->d_z != nullptr) {
        (void)cudaFree(ctx->d_z);
    }
    if (ctx->d_subdiag != nullptr) {
        (void)cudaFree(ctx->d_subdiag);
    }
    if (ctx->d_diag != nullptr) {
        (void)cudaFree(ctx->d_diag);
    }
}

void restore_cusolver_inputs(const CusolverBenchContext &ctx) {
    check_cuda(cudaMemcpy(ctx.d_diag, ctx.h_diag_base,
                          sizeof(double) * static_cast<std::size_t>(ctx.n),
                          cudaMemcpyHostToDevice),
               "cusolver cudaMemcpy d_diag H2D");
    if (ctx.n > 1) {
        check_cuda(
            cudaMemcpy(ctx.d_subdiag, ctx.h_subdiag_base,
                       sizeof(double) * static_cast<std::size_t>(ctx.n - 1),
                       cudaMemcpyHostToDevice),
            "cusolver cudaMemcpy d_subdiag H2D");
    }
    if (ctx.compz != CUSOLVER_EIG_COMP_N) {
        check_cuda(
            cudaMemcpy(ctx.d_z, ctx.h_z_base.data(),
                       sizeof(double) * static_cast<std::size_t>(ctx.n * ctx.n),
                       cudaMemcpyHostToDevice),
            "cusolver cudaMemcpy d_z H2D");
    }
}

BenchResult bench_gpu_tridiag_solver(GpuBenchContext *ctx, int warmup,
                                     int iters, GpuRunSummary *summary_out) {
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(iters));
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");

    for (int rep = 0; rep < warmup + iters; ++rep) {
        check_cuda(cudaEventRecord(start, ctx->handle->stream),
                   "cudaEventRecord start");
        const evd_status solver_status =
            evd::internal::tridiag_solver_run_from_device_tridiagonal(
                ctx->handle.get(), ctx->config, ctx->d_diag_base,
                ctx->d_subdiag_base, ctx->d_w, ctx->workspace,
                ctx->workspace_bytes);
        if (solver_status != EVD_STATUS_SUCCESS) {
            check_evd(solver_status, ctx->handle.get(),
                      "tridiag_solver_run failed");
        }
        check_cuda(cudaEventRecord(stop, ctx->handle->stream),
                   "cudaEventRecord stop");
        check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
        float ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&ms, start, stop),
                   "cudaEventElapsedTime");
        if (rep >= warmup) {
            times.push_back(static_cast<double>(ms));
        }
    }

    check_cuda(cudaEventDestroy(start), "cudaEventDestroy start");
    check_cuda(cudaEventDestroy(stop), "cudaEventDestroy stop");
    if (summary_out != nullptr) {
        summary_out->tridiag_diagnostics =
            ctx->handle->last_tridiag_diagnostics;
    }
    return summarize(std::move(times));
}

BenchResult bench_cusolver_tridiag_solver(CusolverBenchContext *ctx, int warmup,
                                          int iters,
                                          std::vector<double> *last_eigs_out) {
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(iters));
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check_cuda(cudaEventCreate(&start), "cusolver cudaEventCreate start");
    check_cuda(cudaEventCreate(&stop), "cusolver cudaEventCreate stop");

    for (int rep = 0; rep < warmup + iters; ++rep) {
        restore_cusolver_inputs(*ctx);
        check_cuda(cudaEventRecord(start, 0), "cusolver cudaEventRecord start");
        check_cusolver(
            cusolverDnXstedc(ctx->handle.get(), ctx->params.get(), ctx->compz,
                             ctx->n, CUDA_R_64F, ctx->d_diag, ctx->d_subdiag,
                             CUDA_R_64F, ctx->d_z, ctx->n, CUDA_R_64F,
                             ctx->d_work, ctx->d_work_bytes, ctx->h_work,
                             ctx->h_work_bytes, ctx->d_info),
            "cusolverDnXstedc");
        check_cuda(cudaEventRecord(stop, 0), "cusolver cudaEventRecord stop");
        check_cuda(cudaEventSynchronize(stop), "cusolver cudaEventSynchronize");

        int info = 0;
        check_cuda(
            cudaMemcpy(&info, ctx->d_info, sizeof(int), cudaMemcpyDeviceToHost),
            "cusolver cudaMemcpy d_info D2H");
        if (info != 0) {
            std::ostringstream oss;
            oss << "cusolverDnXstedc returned info=" << info;
            throw_runtime(oss.str());
        }

        float ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&ms, start, stop),
                   "cusolver cudaEventElapsedTime");
        if (rep >= warmup) {
            times.push_back(static_cast<double>(ms));
        }
    }

    std::vector<double> eigs(static_cast<std::size_t>(ctx->n));
    check_cuda(cudaMemcpy(eigs.data(), ctx->d_diag,
                          sizeof(double) * static_cast<std::size_t>(ctx->n),
                          cudaMemcpyDeviceToHost),
               "cusolver cudaMemcpy eigs D2H");

    check_cuda(cudaEventDestroy(start), "cusolver cudaEventDestroy start");
    check_cuda(cudaEventDestroy(stop), "cusolver cudaEventDestroy stop");
    if (last_eigs_out != nullptr) {
        *last_eigs_out = eigs;
    }
    return summarize(std::move(times));
}

} // namespace

int main(int argc, char **argv) {
    try {
        const auto sizes = parse_sizes(argc, argv);
        const int warmup = parse_int_flag(argc, argv, "--warmup", 2);
        const int iters = parse_int_flag(argc, argv, "--iters", 10);
        const bool compute_vectors =
            parse_int_flag(argc, argv, "--vectors", 1) != 0;
        const int backend = parse_int_flag(argc, argv, "--backend", 0);
        const bool run_cpu = parse_int_flag(argc, argv, "--run-cpu", 1) != 0;
        const bool run_self = parse_int_flag(argc, argv, "--run-self", 1) != 0;
        const bool run_cusolver =
            parse_int_flag(argc, argv, "--run-cusolver", 1) != 0;
        const bool enable_tridiag_diagnostics =
            parse_int_flag(argc, argv, "--tridiag-diagnostics", 0) != 0;
        const bool check_vectors =
            parse_int_flag(argc, argv, "--check-vectors", 0) != 0;
        const std::string data_mode =
            parse_string_flag(argc, argv, "--data", "bench");
        std::vector<std::string> data_modes = split_csv(data_mode);
        if (data_modes.empty()) {
            data_modes.push_back("bench");
        }
        const std::string input_path = parse_string_flag(argc, argv, "--input");
        const std::string output_tridiag_path =
            parse_string_flag(argc, argv, "--output-tridiag");
        const bool input_is_tridiag =
            parse_int_flag(argc, argv, "--input-tridiag", 0) != 0;
        const int64_t band_size =
            static_cast<int64_t>(parse_int_flag(argc, argv, "--band-size", 32));
        const int64_t bulge_block_size = static_cast<int64_t>(
            parse_int_flag(argc, argv, "--bulge-block-size", 256));

        if (!run_cpu && !run_self && !run_cusolver && output_tridiag_path.empty()) {
            throw_runtime("at least one of --run-cpu, --run-self, "
                          "--run-cusolver, or --output-tridiag must be enabled");
        }

        std::cout << std::left << std::setw(12) << "mode" << std::setw(8)
                  << "n" << std::setw(12)
                  << "prep_ms";
        if (run_cpu) {
            std::cout << std::setw(14) << "cpu_ms";
        }
        if (run_self) {
            std::cout << std::setw(14) << "self_ms";
        }
        if (run_cpu && run_self) {
            std::cout << std::setw(16) << "self_err";
        }
        if (run_cusolver) {
            std::cout << std::setw(14) << "cslv_ms";
        }
        if (run_self && run_cusolver) {
            std::cout << std::setw(12) << "cslv/self";
        }
        if (run_cpu && run_cusolver) {
            std::cout << std::setw(16) << "cslv_err";
        }
        std::cout << '\n';

        if (!input_path.empty()) {
            PreparedInput prepared{};
            if (input_is_tridiag) {
                prepared.tridiag = read_tridiagonal_file(input_path);
                prepared.prep_ms = 0.0;
                prepared.source = "tridiag_bin";
            } else {
                int64_t n = 0;
                const std::vector<double> host_matrix =
                    read_matrix_file(input_path, &n);
                prepared = prepare_tridiagonal_from_dense_matrix(
                    host_matrix, n, band_size, bulge_block_size);
            }
            if (!output_tridiag_path.empty()) {
                write_tridiagonal_file(output_tridiag_path, prepared.tridiag);
                std::cout << "wrote_tridiag=" << output_tridiag_path
                          << " n=" << prepared.tridiag.n
                          << " bytes="
                          << (sizeof(int64_t) +
                              sizeof(double) * (2 * prepared.tridiag.n - 1))
                          << '\n';
            }

            GpuRunSummary gpu_summary{};
            BenchResult cpu{};
            BenchResult gpu_time{};
            BenchResult cusolver_time{};
            std::vector<double> cpu_eigs;
            double cusolver_eig_rel_err = 0.0;

            if (run_cpu) {
                cpu = bench_cpu_dstedc(
                    prepared.tridiag.diag, prepared.tridiag.subdiag,
                    compute_vectors, warmup, iters, &cpu_eigs);
            }

            std::cout << "input=" << input_path << " source=" << prepared.source
                      << " vectors=" << (compute_vectors ? 1 : 0)
                      << " backend=" << backend
                      << " run_cpu=" << (run_cpu ? 1 : 0)
                      << " run_self=" << (run_self ? 1 : 0)
                      << " run_cusolver=" << (run_cusolver ? 1 : 0) << '\n';

            if (run_self) {
                GpuBenchContext gpu = make_gpu_context(
                    prepared.tridiag.diag, prepared.tridiag.subdiag,
                    prepared.tridiag.n, compute_vectors, backend,
                    enable_tridiag_diagnostics);
                gpu_time =
                    bench_gpu_tridiag_solver(&gpu, warmup, iters, &gpu_summary);

                if (run_cpu) {
                    std::vector<double> gpu_eigs = copy_gpu_eigenvalues(gpu);
                    gpu_summary.eig_relative_error =
                        relative_error(gpu_eigs, cpu_eigs);
                }
                if (check_vectors && compute_vectors) {
                    fill_gpu_vector_checks_device(prepared.tridiag.diag,
                                                  prepared.tridiag.subdiag, gpu,
                                                  &gpu_summary);
                }
                destroy_gpu_context(&gpu);
            }

            if (run_cusolver) {
                CusolverBenchContext cusolver = make_cusolver_context(
                    prepared.tridiag.diag, prepared.tridiag.subdiag,
                    prepared.tridiag.n, compute_vectors);
                std::vector<double> cusolver_eigs;
                cusolver_time = bench_cusolver_tridiag_solver(
                    &cusolver, warmup, iters, &cusolver_eigs);
                if (run_cpu) {
                    cusolver_eig_rel_err =
                        relative_error(cusolver_eigs, cpu_eigs);
                }
                destroy_cusolver_context(&cusolver);
            }

            std::cout << std::left << std::setw(8) << prepared.tridiag.n
                      << std::setw(12) << std::fixed << std::setprecision(3)
                      << prepared.prep_ms;
            if (run_cpu) {
                std::cout << std::setw(14) << cpu.median_ms;
            }
            if (run_self) {
                std::cout << std::setw(14) << gpu_time.median_ms;
            }
            if (run_cpu && run_self) {
                std::cout << std::setw(16) << std::scientific
                          << std::setprecision(3)
                          << gpu_summary.eig_relative_error;
            }
            if (run_cusolver) {
                std::cout << std::fixed << std::setprecision(3) << std::setw(14)
                          << cusolver_time.median_ms;
            }
            if (run_self && run_cusolver) {
                std::cout << std::setw(12) << std::fixed << std::setprecision(2)
                          << (gpu_time.median_ms > 0.0
                                  ? cusolver_time.median_ms / gpu_time.median_ms
                                  : 0.0);
            }
            if (run_cpu && run_cusolver) {
                std::cout << std::setw(16) << std::scientific
                          << std::setprecision(3) << cusolver_eig_rel_err;
            }
            std::cout << '\n';
            if (check_vectors && compute_vectors && run_self) {
                std::cout << "self vector check: residual=" << std::scientific
                          << std::setprecision(3)
                          << gpu_summary.eigenpair_residual << " norm_max="
                          << gpu_summary.eigenvector_norm_max_error
                          << " sample_orth="
                          << gpu_summary.eigenvector_sample_orthogonality
                          << " check_ms=" << std::fixed << std::setprecision(3)
                          << gpu_summary.eigenvector_check_ms << '\n';
            }
            if (enable_tridiag_diagnostics && run_self) {
                print_tridiag_diagnostics(gpu_summary.tridiag_diagnostics,
                                          "self_stage");
            }
            return 0;
        }

        for (const std::string &mode : data_modes) {
        for (std::size_t idx = 0; idx < sizes.size(); ++idx) {
            const int64_t n = sizes[idx];
            std::vector<double> diag;
            std::vector<double> subdiag;
            if (mode == "test") {
                make_test_tridiagonal_data(n, static_cast<uint32_t>(1300 + n),
                                           &diag, &subdiag);
            } else if (mode == "bench") {
                make_bench_tridiagonal_data(n, static_cast<uint32_t>(1234 + idx),
                                            &diag, &subdiag);
            } else if (mode == "normal" || mode == "uniform" ||
                       mode == "toeplitz" || mode == "clustered") {
                make_paper_tridiagonal_data(mode, n, &diag, &subdiag);
            } else {
                throw_runtime("unsupported --data entry, expected bench, test, "
                              "normal, uniform, toeplitz, or clustered");
            }

            GpuRunSummary gpu_summary{};
            BenchResult cpu{};
            BenchResult gpu_time{};
            BenchResult cusolver_time{};
            std::vector<double> cpu_eigs;
            double cusolver_eig_rel_err = 0.0;

            if (run_cpu) {
                cpu = bench_cpu_dstedc(diag, subdiag, compute_vectors, warmup,
                                       iters, &cpu_eigs);
            }

            if (run_self) {
                GpuBenchContext gpu =
                    make_gpu_context(diag, subdiag, n, compute_vectors, backend,
                                     enable_tridiag_diagnostics);
                gpu_time =
                    bench_gpu_tridiag_solver(&gpu, warmup, iters, &gpu_summary);
                if (run_cpu) {
                    std::vector<double> gpu_eigs = copy_gpu_eigenvalues(gpu);
                    gpu_summary.eig_relative_error =
                        relative_error(gpu_eigs, cpu_eigs);
                }
                if (check_vectors && compute_vectors) {
                    fill_gpu_vector_checks_device(diag, subdiag, gpu,
                                                  &gpu_summary);
                }
                destroy_gpu_context(&gpu);
            }

            if (run_cusolver) {
                CusolverBenchContext cusolver =
                    make_cusolver_context(diag, subdiag, n, compute_vectors);
                std::vector<double> cusolver_eigs;
                cusolver_time = bench_cusolver_tridiag_solver(
                    &cusolver, warmup, iters, &cusolver_eigs);
                if (run_cpu) {
                    cusolver_eig_rel_err =
                        relative_error(cusolver_eigs, cpu_eigs);
                }
                destroy_cusolver_context(&cusolver);
            }

            std::cout << std::left << std::setw(12) << mode << std::setw(8)
                      << n << std::setw(12)
                      << std::fixed << std::setprecision(3) << 0.0;
            if (run_cpu) {
                std::cout << std::setw(14) << std::fixed << std::setprecision(3)
                          << cpu.median_ms;
            }
            if (run_self) {
                std::cout << std::setw(14) << std::fixed << std::setprecision(3)
                          << gpu_time.median_ms;
            }
            if (run_cpu && run_self) {
                std::cout << std::setw(16) << std::scientific
                          << std::setprecision(3)
                          << gpu_summary.eig_relative_error;
            }
            if (run_cusolver) {
                std::cout << std::setw(14) << std::fixed << std::setprecision(3)
                          << cusolver_time.median_ms;
            }
            if (run_self && run_cusolver) {
                std::cout << std::setw(12) << std::fixed << std::setprecision(2)
                          << (gpu_time.median_ms > 0.0
                                  ? cusolver_time.median_ms / gpu_time.median_ms
                                  : 0.0);
            }
            if (run_cpu && run_cusolver) {
                std::cout << std::setw(16) << std::scientific
                          << std::setprecision(3) << cusolver_eig_rel_err;
            }
            std::cout << '\n';
            if (check_vectors && compute_vectors && run_self) {
                std::cout << "self vector check: residual=" << std::scientific
                          << std::setprecision(3)
                          << gpu_summary.eigenpair_residual << " norm_max="
                          << gpu_summary.eigenvector_norm_max_error
                          << " sample_orth="
                          << gpu_summary.eigenvector_sample_orthogonality
                          << " check_ms=" << std::fixed << std::setprecision(3)
                          << gpu_summary.eigenvector_check_ms << '\n';
            }
            if (enable_tridiag_diagnostics && run_self) {
                print_tridiag_diagnostics(gpu_summary.tridiag_diagnostics,
                                          "self_stage");
            }
        }
        }
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
