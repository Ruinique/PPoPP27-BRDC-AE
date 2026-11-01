#ifndef EVD_LIBRARY_INTERNAL_TRIDIAG_SOLVER_LEAF_CUH
#define EVD_LIBRARY_INTERNAL_TRIDIAG_SOLVER_LEAF_CUH

#include "tridiag_solver_dc.cuh"
#include "tridiag_solver_device.cuh"

namespace evd::internal {

namespace {

__global__ void stedc_leaf_solve_kernel(const int blks, const int n, double *D,
                                        double *E, int *splits, double *tmpz,
                                        double *V, const int ldv,
                                        const bool compute_vectors,
                                        double *boundary_rows = nullptr) {
    const int block = static_cast<int>(blockIdx.x);
    if (block >= blks) {
        return;
    }

    __shared__ double
        matrix[kTridiagSolverMaxLeafSize * kTridiagSolverMaxLeafSize];
    __shared__ double
        eigvec[kTridiagSolverMaxLeafSize * kTridiagSolverMaxLeafSize];
    __shared__ double
        sorted_vec[kTridiagSolverMaxLeafSize * kTridiagSolverMaxLeafSize];
    __shared__ double eigenvalues[kTridiagSolverMaxLeafSize];
    __shared__ int perm[kTridiagSolverMaxLeafSize];

    int *msz = ptr_msz(n, splits);
    int *mps = ptr_mps(n, splits);
    const int size = msz[block];
    const int start = mps[block];

    if (size <= 0 || size > kTridiagSolverMaxLeafSize) {
        return;
    }

    if (threadIdx.x == 0) {
        for (int row = 0; row < size; ++row) {
            for (int col = 0; col < size; ++col) {
                matrix[row * kTridiagSolverMaxLeafSize + col] = 0.0;
                eigvec[row * kTridiagSolverMaxLeafSize + col] =
                    row == col ? 1.0 : 0.0;
            }
        }
        for (int row = 0; row < size; ++row) {
            matrix[row * kTridiagSolverMaxLeafSize + row] = D[start + row];
            if (row + 1 < size) {
                const double offdiag = E[start + row];
                matrix[row * kTridiagSolverMaxLeafSize + (row + 1)] = offdiag;
                matrix[(row + 1) * kTridiagSolverMaxLeafSize + row] = offdiag;
            }
        }

        const double tolerance = 1.0e-12;
        const int max_sweeps = 50 * size * size;
        for (int sweep = 0; sweep < max_sweeps; ++sweep) {
            double max_offdiag = 0.0;
            for (int p = 0; p < size; ++p) {
                for (int q = p + 1; q < size; ++q) {
                    const double value =
                        device_abs(matrix[p * kTridiagSolverMaxLeafSize + q]);
                    if (value > max_offdiag) {
                        max_offdiag = value;
                    }
                }
            }
            if (max_offdiag < tolerance) {
                break;
            }

            for (int p = 0; p < size; ++p) {
                for (int q = p + 1; q < size; ++q) {
                    double &apq = matrix[p * kTridiagSolverMaxLeafSize + q];
                    if (device_abs(apq) < tolerance) {
                        continue;
                    }

                    double &app = matrix[p * kTridiagSolverMaxLeafSize + p];
                    double &aqq = matrix[q * kTridiagSolverMaxLeafSize + q];
                    const double tau = (aqq - app) / (2.0 * apq);
                    const double root = sqrt(1.0 + tau * tau);
                    const double t =
                        tau >= 0.0 ? 1.0 / (tau + root) : -1.0 / (-tau + root);
                    const double c = 1.0 / sqrt(1.0 + t * t);
                    const double s = t * c;
                    const double old_apq = apq;
                    app -= t * old_apq;
                    aqq += t * old_apq;
                    apq = 0.0;
                    matrix[q * kTridiagSolverMaxLeafSize + p] = 0.0;

                    for (int r = 0; r < size; ++r) {
                        if (r == p || r == q) {
                            continue;
                        }

                        const double arp =
                            matrix[r * kTridiagSolverMaxLeafSize + p];
                        const double arq =
                            matrix[r * kTridiagSolverMaxLeafSize + q];
                        const double new_rp = c * arp - s * arq;
                        const double new_rq = s * arp + c * arq;
                        matrix[r * kTridiagSolverMaxLeafSize + p] = new_rp;
                        matrix[p * kTridiagSolverMaxLeafSize + r] = new_rp;
                        matrix[r * kTridiagSolverMaxLeafSize + q] = new_rq;
                        matrix[q * kTridiagSolverMaxLeafSize + r] = new_rq;
                    }

                    for (int r = 0; r < size; ++r) {
                        const double vrp =
                            eigvec[r * kTridiagSolverMaxLeafSize + p];
                        const double vrq =
                            eigvec[r * kTridiagSolverMaxLeafSize + q];
                        eigvec[r * kTridiagSolverMaxLeafSize + p] =
                            c * vrp - s * vrq;
                        eigvec[r * kTridiagSolverMaxLeafSize + q] =
                            s * vrp + c * vrq;
                    }
                }
            }
        }

        double *leaf_values = ptr_md(n, tmpz) + start;
        for (int i = 0; i < size; ++i) {
            eigenvalues[i] = matrix[i * kTridiagSolverMaxLeafSize + i];
        }
        sort_small_values(eigenvalues, perm, size);
        for (int col = 0; col < size; ++col) {
            for (int row = 0; row < size; ++row) {
                sorted_vec[row * kTridiagSolverMaxLeafSize + col] =
                    eigvec[row * kTridiagSolverMaxLeafSize + perm[col]];
            }
        }

        for (int i = 0; i < size; ++i) {
            D[start + i] = eigenvalues[i];
            leaf_values[i] = eigenvalues[i];
            if (i + 1 < size) {
                E[start + i] = 0.0;
            }
        }

        if (V != nullptr) {
            for (int col = 0; col < size; ++col) {
                for (int row = 0; row < size; ++row) {
                    V[dense_index(start + row, start + col, ldv)] =
                        sorted_vec[row * kTridiagSolverMaxLeafSize + col];
                }
            }
        }
        if (!compute_vectors && boundary_rows != nullptr) {
            double *first_rows = ptr_first_rows(n, boundary_rows);
            double *last_rows = ptr_last_rows(n, boundary_rows);
            for (int col = 0; col < size; ++col) {
                first_rows[start + col] =
                    sorted_vec[0 * kTridiagSolverMaxLeafSize + col];
                last_rows[start + col] =
                    sorted_vec[(size - 1) * kTridiagSolverMaxLeafSize + col];
            }
        }
    }
}

__global__ void stedc_leaf_extract_boundary_rows_kernel(
    const int blks, const int n, double *V, const int ldv, int *splits,
    double *boundary_rows) {
    const int block = static_cast<int>(blockIdx.x);
    if (block >= blks) {
        return;
    }

    int *msz = ptr_msz(n, splits);
    int *mps = ptr_mps(n, splits);
    const int size = msz[block];
    const int start = mps[block];

    if (size <= 0 || size > kTridiagSolverMaxLeafSize) {
        return;
    }

    double *first_rows = ptr_first_rows(n, boundary_rows);
    double *last_rows = ptr_last_rows(n, boundary_rows);

    const int col = static_cast<int>(threadIdx.x);
    if (col < size) {
        first_rows[start + col] = V[dense_index(start, start + col, ldv)];
        last_rows[start + col] =
            V[dense_index(start + size - 1, start + col, ldv)];
    }
}

} // namespace

} // namespace evd::internal

#endif
