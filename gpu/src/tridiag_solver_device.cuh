#ifndef EVD_LIBRARY_INTERNAL_TRIDIAG_SOLVER_DEVICE_CUH
#define EVD_LIBRARY_INTERNAL_TRIDIAG_SOLVER_DEVICE_CUH

#include "tridiag_solver_dc.cuh"

namespace evd::internal {

namespace {


__device__ inline double device_abs(const double value) {
    return value >= 0.0 ? value : -value;
}

__device__ inline void device_lartg(const double f, const double g, double *c,
                                    double *s, double *r) {
    if (g == 0.0) {
        *c = 1.0;
        *s = 0.0;
        *r = f;
        return;
    }
    if (f == 0.0) {
        *c = 0.0;
        *s = 1.0;
        *r = g;
        return;
    }
    const double scale = device_abs(f) + device_abs(g);
    const double fs = f / scale;
    const double gs = g / scale;
    const double norm = scale * sqrt(fs * fs + gs * gs);
    *c = f / norm;
    *s = g / norm;
    *r = norm;
}

__device__ void sort_small_values(double *values, int *perm, const int n) {
    for (int i = 0; i < n; ++i) {
        perm[i] = i;
    }
    for (int i = 1; i < n; ++i) {
        const double value = values[i];
        const int p = perm[i];
        int j = i - 1;
        while (j >= 0 && values[j] > value) {
            values[j + 1] = values[j];
            perm[j + 1] = perm[j];
            --j;
        }
        values[j + 1] = value;
        perm[j + 1] = p;
    }
}

inline double *tempgemm_vecs(const int n,
                             const TridiagSolverWorkspaceView &view) {
    return view.d_tempgemm == nullptr ? nullptr : ptr_vecs(n, view.d_tempgemm);
}

inline double *tempgemm_etmpd(const int n,
                              const TridiagSolverWorkspaceView &view) {
    (void)n;
    return view.d_etmpd;
}

__host__ __device__ inline std::size_t dense_index(const int row,
                                                   const int col,
                                                   const int ld) {
    return static_cast<std::size_t>(row) +
           static_cast<std::size_t>(col) * static_cast<std::size_t>(ld);
}

} // namespace

} // namespace evd::internal

#endif
