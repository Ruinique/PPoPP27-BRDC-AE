#ifndef EVD_LIBRARY_INTERNAL_TRIDIAG_SOLVER_SECULAR_CUH
#define EVD_LIBRARY_INTERNAL_TRIDIAG_SOLVER_SECULAR_CUH

#include "tridiag_solver_device.cuh"

#include <cub/block/block_reduce.cuh>

namespace evd::internal {

namespace {

constexpr int kSecularMaxIterations = 50;

constexpr int kSecularSolveBlockSize = 32;
constexpr int kSecularSolveCubBlockSize = 256;
constexpr int kSecularSolveCubMinDd = 512;

constexpr int kSecularDbgAllTakenOver = 0;
constexpr int kSecularDbgTotalMerges = 1;
constexpr int kSecularDbgResidualFailedMerges = 3;
constexpr int kSecularDbgSummaryCount = 5;

__device__ inline int secular_solve_info(const bool converged,
                                         const int iterations) {
    return converged ? iterations : -iterations;
}

struct SecularEvalSums {
    double gx;
    double gdx;
    double hx;
    double hdx;
    double er;
};

template <int BlockThreads>
__device__ inline SecularEvalSums
block_eval_device(const int type, const int k, const int dd, double *D,
                  const double *z, const double p, const double cor,
                  const bool modify, const double ssfmin) {
    int gout = 0;
    int hout = 0;
    if (type == 0) {
        gout = k + 1;
        hout = k;
    } else if (type == 1) {
        if (modify && threadIdx.x == 0) {
            D[k] -= cor;
        }
        gout = k;
        hout = k;
    } else {
        if (modify && threadIdx.x == 0) {
            D[k] -= cor;
            D[k + 1] -= cor;
        }
        gout = k;
        hout = k + 1;
    }
    __syncthreads();

    SecularEvalSums local{0.0, 0.0, 0.0, 0.0, 0.0};
    for (int i = threadIdx.x; i < gout; i += BlockThreads) {
        double tmp = D[i] - cor;
        if (modify) {
            D[i] = tmp;
        }
        const double zi = z[i];
        if (device_abs(tmp) < ssfmin) {
            tmp = tmp >= 0.0 ? ssfmin : -ssfmin;
        }
        tmp = zi / tmp;
        const double term = zi * tmp;
        local.gx += term;
        local.gdx += tmp * tmp;
        local.er += device_abs(term);
    }

    for (int i = dd - 1 - threadIdx.x; i > hout; i -= BlockThreads) {
        double tmp = D[i] - cor;
        if (modify) {
            D[i] = tmp;
        }
        const double zi = z[i];
        if (device_abs(tmp) < ssfmin) {
            tmp = tmp >= 0.0 ? ssfmin : -ssfmin;
        }
        tmp = zi / tmp;
        const double term = zi * tmp;
        local.hx += term;
        local.hdx += tmp * tmp;
        local.er += device_abs(term);
    }

    using BlockReduce = cub::BlockReduce<double, BlockThreads>;
    __shared__ typename BlockReduce::TempStorage reduce_storage;
    __shared__ SecularEvalSums shared_sum;
    const double gx = BlockReduce(reduce_storage).Sum(local.gx);
    __syncthreads();
    const double gdx = BlockReduce(reduce_storage).Sum(local.gdx);
    __syncthreads();
    const double hx = BlockReduce(reduce_storage).Sum(local.hx);
    __syncthreads();
    const double hdx = BlockReduce(reduce_storage).Sum(local.hdx);
    __syncthreads();
    const double er = BlockReduce(reduce_storage).Sum(local.er);
    __syncthreads();
    if (threadIdx.x == 0) {
        shared_sum = {gx, gdx, hx, hdx, er};
    }
    __syncthreads();
    return shared_sum;
}

template <int BlockThreads>
__device__ inline double
block_eval_value_only_device(const int type, const int k, const int dd,
                             const double *D, const double *z, const double p,
                             const double cor, const double ssfmin) {
    int gout = 0;
    int hout = 0;
    if (type == 0) {
        gout = k + 1;
        hout = k;
    } else if (type == 1) {
        gout = k;
        hout = k;
    } else {
        gout = k;
        hout = k + 1;
    }

    double local = 0.0;
    for (int i = threadIdx.x; i < gout; i += BlockThreads) {
        double tmp = D[i] - cor;
        const double zi = z[i];
        if (device_abs(tmp) < ssfmin) {
            tmp = tmp >= 0.0 ? ssfmin : -ssfmin;
        }
        local += zi * (zi / tmp);
    }
    for (int i = dd - 1 - threadIdx.x; i > hout; i -= BlockThreads) {
        double tmp = D[i] - cor;
        const double zi = z[i];
        if (device_abs(tmp) < ssfmin) {
            tmp = tmp >= 0.0 ? ssfmin : -ssfmin;
        }
        local += zi * (zi / tmp);
    }

    using BlockReduce = cub::BlockReduce<double, BlockThreads>;
    __shared__ typename BlockReduce::TempStorage reduce_storage;
    __shared__ double shared_sum;
    const double sum = BlockReduce(reduce_storage).Sum(local);
    if (threadIdx.x == 0) {
        shared_sum = p + sum;
    }
    __syncthreads();
    return shared_sum;
}

template <int BlockThreads>
__device__ inline void
block_eval_values(const int type, const int k, const int dd, double *D,
                  const double *z, const double p, const double cor, double *fx,
                  double *fdx, double *gx, double *gdx, double *hx, double *hdx,
                  double *er, const bool modify, const double ssfmin) {
    const SecularEvalSums sums = block_eval_device<BlockThreads>(
        type, k, dd, D, z, p, cor, modify, ssfmin);
    *gx = sums.gx;
    *gdx = sums.gdx;
    *hx = sums.hx;
    *hdx = sums.hdx;
    *er = sums.er;
    *fx = p + sums.gx + sums.hx;
    *fdx = sums.gdx + sums.hdx;
}

template <int BlockThreads>
__device__ inline void block_eval_shifted_values(
    const int type, const int k, const int dd, double *D, const double *z,
    const double p, const double origin, const double cor, double *fx,
    double *fdx, double *gx, double *gdx, double *hx, double *hdx, double *er,
    const bool modify, const double ssfmin) {
    int gout = 0;
    int hout = 0;
    if (type == 0) {
        gout = k + 1;
        hout = k;
    } else if (type == 1) {
        if (modify && threadIdx.x == 0) {
            D[k] = (D[k] - origin) - cor;
        }
        gout = k;
        hout = k;
    } else {
        if (modify && threadIdx.x == 0) {
            D[k] = (D[k] - origin) - cor;
            D[k + 1] = (D[k + 1] - origin) - cor;
        }
        gout = k;
        hout = k + 1;
    }
    __syncthreads();

    SecularEvalSums local{0.0, 0.0, 0.0, 0.0, 0.0};
    for (int i = threadIdx.x; i < gout; i += BlockThreads) {
        double tmp = (D[i] - origin) - cor;
        if (modify) {
            D[i] = tmp;
        }
        const double zi = z[i];
        if (device_abs(tmp) < ssfmin) {
            tmp = tmp >= 0.0 ? ssfmin : -ssfmin;
        }
        tmp = zi / tmp;
        const double term = zi * tmp;
        local.gx += term;
        local.gdx += tmp * tmp;
        local.er += device_abs(term);
    }

    for (int i = dd - 1 - threadIdx.x; i > hout; i -= BlockThreads) {
        double tmp = (D[i] - origin) - cor;
        if (modify) {
            D[i] = tmp;
        }
        const double zi = z[i];
        if (device_abs(tmp) < ssfmin) {
            tmp = tmp >= 0.0 ? ssfmin : -ssfmin;
        }
        tmp = zi / tmp;
        const double term = zi * tmp;
        local.hx += term;
        local.hdx += tmp * tmp;
        local.er += device_abs(term);
    }

    using BlockReduce = cub::BlockReduce<double, BlockThreads>;
    __shared__ typename BlockReduce::TempStorage reduce_storage;
    __shared__ SecularEvalSums shared_sum;
    const double reduced_gx = BlockReduce(reduce_storage).Sum(local.gx);
    __syncthreads();
    const double reduced_gdx = BlockReduce(reduce_storage).Sum(local.gdx);
    __syncthreads();
    const double reduced_hx = BlockReduce(reduce_storage).Sum(local.hx);
    __syncthreads();
    const double reduced_hdx = BlockReduce(reduce_storage).Sum(local.hdx);
    __syncthreads();
    const double reduced_er = BlockReduce(reduce_storage).Sum(local.er);
    __syncthreads();
    if (threadIdx.x == 0) {
        shared_sum = {reduced_gx, reduced_gdx, reduced_hx, reduced_hdx,
                      reduced_er};
    }
    __syncthreads();
    *gx = shared_sum.gx;
    *gdx = shared_sum.gdx;
    *hx = shared_sum.hx;
    *hdx = shared_sum.hdx;
    *er = shared_sum.er;
    *fx = p + shared_sum.gx + shared_sum.hx;
    *fdx = shared_sum.gdx + shared_sum.hdx;
}

__device__ inline void
seq_eval_device(const int type, const int k, const int dd, double *D,
                const double *z, const double p, const double cor, double *fx,
                double *fdx, double *gx, double *gdx, double *hx, double *hdx,
                double *er, const bool modify, const double eps,
                const double ssfmin, const double ssfmax) {
    (void)eps;
    (void)ssfmax;
    int gout = 0;
    int hout = 0;
    if (type == 0) {
        gout = k + 1;
        hout = k;
    } else if (type == 1) {
        if (modify) {
            D[k] -= cor;
        }
        gout = k;
        hout = k;
    } else {
        if (modify) {
            D[k] -= cor;
            D[k + 1] -= cor;
        }
        gout = k;
        hout = k + 1;
    }

    *gx = 0.0;
    *gdx = 0.0;
    *er = 0.0;
    for (int i = 0; i < gout; ++i) {
        double tmp = D[i] - cor;
        if (modify) {
            D[i] = tmp;
        }
        const double zi = z[i];
        if (device_abs(tmp) < ssfmin) {
            tmp = tmp >= 0.0 ? ssfmin : -ssfmin;
        }
        tmp = zi / tmp;
        const double term = zi * tmp;
        *gx += term;
        *gdx += tmp * tmp;
        *er += device_abs(term);
    }

    *hx = 0.0;
    *hdx = 0.0;
    for (int i = dd - 1; i > hout; --i) {
        double tmp = D[i] - cor;
        if (modify) {
            D[i] = tmp;
        }
        const double zi = z[i];
        if (device_abs(tmp) < ssfmin) {
            tmp = tmp >= 0.0 ? ssfmin : -ssfmin;
        }
        tmp = zi / tmp;
        const double term = zi * tmp;
        *hx += term;
        *hdx += tmp * tmp;
        *er += device_abs(term);
    }

    *fx = p + *gx + *hx;
    *fdx = *gdx + *hdx;
}

__device__ inline void seq_eval_shifted_device(
    const int type, const int k, const int dd, double *D, const double *z,
    const double p, const double origin, const double cor, double *fx,
    double *fdx, double *gx, double *gdx, double *hx, double *hdx, double *er,
    const bool modify, const double eps, const double ssfmin,
    const double ssfmax) {
    (void)eps;
    (void)ssfmax;
    int gout = 0;
    int hout = 0;
    if (type == 0) {
        gout = k + 1;
        hout = k;
    } else if (type == 1) {
        if (modify) {
            D[k] = (D[k] - origin) - cor;
        }
        gout = k;
        hout = k;
    } else {
        if (modify) {
            D[k] = (D[k] - origin) - cor;
            D[k + 1] = (D[k + 1] - origin) - cor;
        }
        gout = k;
        hout = k + 1;
    }

    *gx = 0.0;
    *gdx = 0.0;
    *er = 0.0;
    for (int i = 0; i < gout; ++i) {
        double tmp = (D[i] - origin) - cor;
        if (modify) {
            D[i] = tmp;
        }
        const double zi = z[i];
        if (device_abs(tmp) < ssfmin) {
            tmp = tmp >= 0.0 ? ssfmin : -ssfmin;
        }
        tmp = zi / tmp;
        const double term = zi * tmp;
        *gx += term;
        *gdx += tmp * tmp;
        *er += device_abs(term);
    }

    *hx = 0.0;
    *hdx = 0.0;
    for (int i = dd - 1; i > hout; --i) {
        double tmp = (D[i] - origin) - cor;
        if (modify) {
            D[i] = tmp;
        }
        const double zi = z[i];
        if (device_abs(tmp) < ssfmin) {
            tmp = tmp >= 0.0 ? ssfmin : -ssfmin;
        }
        tmp = zi / tmp;
        const double term = zi * tmp;
        *hx += term;
        *hdx += tmp * tmp;
        *er += device_abs(term);
    }

    *fx = p + *gx + *hx;
    *fdx = *gdx + *hdx;
}

__device__ inline int seq_solve_device(const int dd, double *D, const double *z,
                                       const double p, const int k, double *ev,
                                       const double tol, const double eps,
                                       const double ssfmin,
                                       const double ssfmax) {
    bool converged = false;
    bool up = false;
    bool fixed = false;
    double lowb = 0.0;
    double uppb = 0.0;
    double aa = 0.0;
    double bb = 0.0;
    double cc = 0.0;
    double x = 0.0;
    double er = 0.0;
    double fx = 0.0;
    double fdx = 0.0;
    double gx = 0.0;
    double gdx = 0.0;
    double hx = 0.0;
    double hdx = 0.0;
    double oldfx = 0.0;
    double tau = 0.0;
    double eta = 0.0;
    const double dk = D[k];
    const double dk1 = D[k + 1];
    double ddk = 0.0;
    double ddk1 = 0.0;
    int kk = 0;
    int iterations = 1;
    const double pinv = 1.0 / p;

    x = (dk + dk1) / 2.0;
    tau = dk1 - dk;

    seq_eval_device(2, k, dd, D, z, pinv, x, &cc, &fdx, &gx, &gdx, &hx, &hdx,
                    &er, false, eps, ssfmin, ssfmax);
    gdx = z[k] * z[k];
    hdx = z[k + 1] * z[k + 1];
    fx = cc + 2.0 * (hdx - gdx) / tau;
    if (fx > 0.0) {
        lowb = 0.0;
        uppb = tau / 2.0;
        up = true;
        kk = k;
        aa = cc * tau + gdx + hdx;
        bb = gdx * tau;
        eta = sqrt(device_abs(aa * aa - 4.0 * bb * cc));
        tau = aa > 0.0 ? (2.0 * bb) / (aa + eta) : (aa - eta) / (2.0 * cc);
        x = dk + tau;
    } else {
        lowb = -tau / 2.0;
        uppb = 0.0;
        up = false;
        kk = k + 1;
        aa = cc * tau - gdx - hdx;
        bb = hdx * tau;
        eta = sqrt(device_abs(aa * aa + 4.0 * bb * cc));
        tau = aa < 0.0 ? (2.0 * bb) / (aa - eta) : -(aa + eta) / (2.0 * cc);
        x = dk1 + tau;
    }

    seq_eval_shifted_device(1, kk, dd, D, z, pinv, up ? dk : dk1, tau, &fx,
                            &fdx, &gx, &gdx, &hx, &hdx, &er, true, eps,
                            ssfmin, ssfmax);
    bb = z[kk];
    aa = bb / D[kk];
    fdx += aa * aa;
    bb *= aa;
    fx += bb;
    er += 8.0 * (hx - gx) + 2.0 * pinv + 3.0 * device_abs(bb) +
          device_abs(tau) * fdx;
    if (device_abs(fx) <= tol * er) {
        converged = true;
    } else {
        lowb = (fx <= 0.0) ? fmax(lowb, tau) : lowb;
        uppb = (fx > 0.0) ? fmin(uppb, tau) : uppb;

        ddk = D[k];
        ddk1 = D[k + 1];
        if (up) {
            cc = fx - ddk1 * fdx - (dk - dk1) * z[k] * z[k] / ddk / ddk;
        } else {
            cc =
                fx - ddk * fdx - (dk1 - dk) * z[k + 1] * z[k + 1] / ddk1 / ddk1;
        }
        aa = (ddk + ddk1) * fx - ddk * ddk1 * fdx;
        bb = ddk * ddk1 * fx;
        if (cc == 0.0) {
            if (aa == 0.0) {
                aa = up ? z[k] * z[k] + ddk1 * ddk1 * (gdx + hdx)
                        : z[k + 1] * z[k + 1] + ddk * ddk * (gdx + hdx);
            }
            eta = bb / aa;
        } else {
            eta = sqrt(device_abs(aa * aa - 4.0 * bb * cc));
            eta = aa <= 0.0 ? (aa - eta) / (2.0 * cc) : (2.0 * bb) / (aa + eta);
        }

        if (fx * eta >= 0.0) {
            eta = -fx / fdx;
        }
        if (tau + eta > uppb || tau + eta < lowb) {
            eta = fx < 0.0 ? (uppb - tau) / 2.0 : (lowb - tau) / 2.0;
        }

        tau += eta;
        x = (up ? dk : dk1) + tau;

        oldfx = fx;
        seq_eval_device(1, kk, dd, D, z, pinv, eta, &fx, &fdx, &gx, &gdx, &hx,
                        &hdx, &er, true, eps, ssfmin, ssfmax);
        bb = z[kk];
        aa = bb / D[kk];
        fdx += aa * aa;
        bb *= aa;
        fx += bb;
        er += 8.0 * (hx - gx) + 2.0 * pinv + 3.0 * device_abs(bb) +
              device_abs(tau) * fdx;
        iterations = 2;
        cc = up ? -1.0 : 1.0;
        fixed = (cc * fx) > (device_abs(oldfx) / 10.0);

        for (int iter = 1; iter < kSecularMaxIterations; ++iter) {
            if (device_abs(fx) <= tol * er) {
                converged = true;
                break;
            }

            lowb = (fx <= 0.0) ? fmax(lowb, tau) : lowb;
            uppb = (fx > 0.0) ? fmin(uppb, tau) : uppb;

            ddk = D[k];
            ddk1 = D[k + 1];
            if (fixed) {
                if (up) {
                    cc = fx - ddk1 * fdx - (dk - dk1) * z[k] * z[k] / ddk / ddk;
                } else {
                    cc = fx - ddk * fdx -
                         (dk1 - dk) * z[k + 1] * z[k + 1] / ddk1 / ddk1;
                }
            } else {
                if (up) {
                    gdx += aa * aa;
                } else {
                    hdx += aa * aa;
                }
                cc = fx - ddk * gdx - ddk1 * hdx;
            }
            aa = (ddk + ddk1) * fx - ddk * ddk1 * fdx;
            bb = ddk * ddk1 * fx;
            if (cc == 0.0) {
                if (aa == 0.0) {
                    if (fixed) {
                        aa = up ? z[k] * z[k] + ddk1 * ddk1 * (gdx + hdx)
                                : z[k + 1] * z[k + 1] + ddk * ddk * (gdx + hdx);
                    } else {
                        aa = ddk * ddk * gdx + ddk1 * ddk1 * hdx;
                    }
                }
                eta = bb / aa;
            } else {
                eta = sqrt(device_abs(aa * aa - 4.0 * bb * cc));
                eta = aa <= 0.0 ? (aa - eta) / (2.0 * cc)
                                : (2.0 * bb) / (aa + eta);
            }

            if (fx * eta >= 0.0) {
                eta = -fx / fdx;
            }
            if (tau + eta > uppb || tau + eta < lowb) {
            eta = fx < 0.0 ? (uppb - tau) / 2.0 : (lowb - tau) / 2.0;
        }

            tau += eta;
            x = (up ? dk : dk1) + tau;

            oldfx = fx;
            seq_eval_device(1, kk, dd, D, z, pinv, eta, &fx, &fdx, &gx, &gdx,
                            &hx, &hdx, &er, true, eps, ssfmin, ssfmax);
            bb = z[kk];
            aa = bb / D[kk];
            fdx += aa * aa;
            bb *= aa;
            fx += bb;
            er += 8.0 * (hx - gx) + 2.0 * pinv + 3.0 * device_abs(bb) +
                  device_abs(tau) * fdx;
            iterations = iter + 2;

            if (fx * oldfx > 0.0 && device_abs(fx) > device_abs(oldfx) / 10.0) {
                fixed = !fixed;
            }
        }
    }

    *ev = x;
    return secular_solve_info(converged, iterations);
}

template <int BlockThreads, bool Up>
__device__ inline int block_solve_direction_device(
    const int dd, double *D, const double *z, const double p, const int k,
    double *ev, const double tol, const double ssfmin, const double dk,
    const double dk1, const double cc0, const double gdx0,
    const double hdx0) {
    bool converged = false;
    bool fixed = false;
    double cc = cc0;
    double gx = 0.0;
    double gdx = gdx0;
    double hx = 0.0;
    double hdx = hdx0;
    double fx = 0.0;
    double fdx = 0.0;
    double er = 0.0;
    double aa = 0.0;
    double bb = 0.0;
    double eta = 0.0;
    double tau = dk1 - dk;
    double lowb = Up ? 0.0 : -tau / 2.0;
    double uppb = Up ? tau / 2.0 : 0.0;
    const int kk = Up ? k : k + 1;
    const double origin = Up ? dk : dk1;
    const double pinv = 1.0 / p;
    int iterations = 1;

    if (Up) {
        aa = cc * tau + gdx + hdx;
        bb = gdx * tau;
        eta = sqrt(device_abs(aa * aa - 4.0 * bb * cc));
        tau = aa > 0.0 ? (2.0 * bb) / (aa + eta) : (aa - eta) / (2.0 * cc);
    } else {
        aa = cc * tau - gdx - hdx;
        bb = hdx * tau;
        eta = sqrt(device_abs(aa * aa + 4.0 * bb * cc));
        tau = aa < 0.0 ? (2.0 * bb) / (aa - eta) : -(aa + eta) / (2.0 * cc);
    }
    double x = origin + tau;

    block_eval_shifted_values<BlockThreads>(
        1, kk, dd, D, z, pinv, origin, tau, &fx, &fdx, &gx, &gdx, &hx, &hdx,
        &er, true, ssfmin);
    bb = z[kk];
    aa = bb / D[kk];
    fdx += aa * aa;
    bb *= aa;
    fx += bb;
    er += 8.0 * (hx - gx) + 2.0 * pinv + 3.0 * device_abs(bb) +
          device_abs(tau) * fdx;
    if (device_abs(fx) <= tol * er) {
        converged = true;
    } else {
        lowb = (fx <= 0.0) ? fmax(lowb, tau) : lowb;
        uppb = (fx > 0.0) ? fmin(uppb, tau) : uppb;

        double ddk = D[k];
        double ddk1 = D[k + 1];
        if (Up) {
            cc = fx - ddk1 * fdx - (dk - dk1) * z[k] * z[k] / ddk / ddk;
        } else {
            cc = fx - ddk * fdx -
                 (dk1 - dk) * z[k + 1] * z[k + 1] / ddk1 / ddk1;
        }
        aa = (ddk + ddk1) * fx - ddk * ddk1 * fdx;
        bb = ddk * ddk1 * fx;
        if (cc == 0.0) {
            if (aa == 0.0) {
                aa = Up ? z[k] * z[k] + ddk1 * ddk1 * (gdx + hdx)
                        : z[k + 1] * z[k + 1] + ddk * ddk * (gdx + hdx);
            }
            eta = bb / aa;
        } else {
            eta = sqrt(device_abs(aa * aa - 4.0 * bb * cc));
            eta = aa <= 0.0 ? (aa - eta) / (2.0 * cc) : (2.0 * bb) / (aa + eta);
        }

        if (fx * eta >= 0.0) {
            eta = -fx / fdx;
        }
        if (tau + eta > uppb || tau + eta < lowb) {
            eta = fx < 0.0 ? (uppb - tau) / 2.0 : (lowb - tau) / 2.0;
        }

        tau += eta;
        x = origin + tau;

        double oldfx = fx;
        block_eval_values<BlockThreads>(1, kk, dd, D, z, pinv, eta, &fx, &fdx,
                                        &gx, &gdx, &hx, &hdx, &er, true,
                                        ssfmin);
        bb = z[kk];
        aa = bb / D[kk];
        fdx += aa * aa;
        bb *= aa;
        fx += bb;
        er += 8.0 * (hx - gx) + 2.0 * pinv + 3.0 * device_abs(bb) +
              device_abs(tau) * fdx;
        iterations = 2;
        fixed = ((Up ? -1.0 : 1.0) * fx) > (device_abs(oldfx) / 10.0);

        for (int iter = 1; iter < kSecularMaxIterations; ++iter) {
            if (device_abs(fx) <= tol * er) {
                converged = true;
                break;
            }

            lowb = (fx <= 0.0) ? fmax(lowb, tau) : lowb;
            uppb = (fx > 0.0) ? fmin(uppb, tau) : uppb;

            ddk = D[k];
            ddk1 = D[k + 1];
            if (fixed) {
                if (Up) {
                    cc = fx - ddk1 * fdx - (dk - dk1) * z[k] * z[k] / ddk / ddk;
                } else {
                    cc = fx - ddk * fdx -
                         (dk1 - dk) * z[k + 1] * z[k + 1] / ddk1 / ddk1;
                }
            } else {
                if (Up) {
                    gdx += aa * aa;
                } else {
                    hdx += aa * aa;
                }
                cc = fx - ddk * gdx - ddk1 * hdx;
            }
            aa = (ddk + ddk1) * fx - ddk * ddk1 * fdx;
            bb = ddk * ddk1 * fx;
            if (cc == 0.0) {
                if (aa == 0.0) {
                    if (fixed) {
                        aa = Up ? z[k] * z[k] + ddk1 * ddk1 * (gdx + hdx)
                                : z[k + 1] * z[k + 1] + ddk * ddk * (gdx + hdx);
                    } else {
                        aa = ddk * ddk * gdx + ddk1 * ddk1 * hdx;
                    }
                }
                eta = bb / aa;
            } else {
                eta = sqrt(device_abs(aa * aa - 4.0 * bb * cc));
                eta = aa <= 0.0 ? (aa - eta) / (2.0 * cc)
                                : (2.0 * bb) / (aa + eta);
            }

            if (fx * eta >= 0.0) {
                eta = -fx / fdx;
            }
            if (tau + eta > uppb || tau + eta < lowb) {
                eta = fx < 0.0 ? (uppb - tau) / 2.0 : (lowb - tau) / 2.0;
            }

            tau += eta;
            x = origin + tau;

            oldfx = fx;
            block_eval_values<BlockThreads>(1, kk, dd, D, z, pinv, eta, &fx,
                                            &fdx, &gx, &gdx, &hx, &hdx, &er,
                                            true, ssfmin);
            bb = z[kk];
            aa = bb / D[kk];
            fdx += aa * aa;
            bb *= aa;
            fx += bb;
            er += 8.0 * (hx - gx) + 2.0 * pinv + 3.0 * device_abs(bb) +
                  device_abs(tau) * fdx;
            iterations = iter + 2;

            if (fx * oldfx > 0.0 && device_abs(fx) > device_abs(oldfx) / 10.0) {
                fixed = !fixed;
            }
        }
    }

    if (threadIdx.x == 0) {
        *ev = x;
    }
    return secular_solve_info(converged, iterations);
}

template <int BlockThreads>
__device__ inline int
block_solve_device(const int dd, double *D, const double *z, const double p,
                   const int k, double *ev, const double tol, const double eps,
                   const double ssfmin, const double ssfmax) {
    (void)eps;
    (void)ssfmax;
    const double dk = D[k];
    const double dk1 = D[k + 1];
    const double pinv = 1.0 / p;
    const double midpoint = (dk + dk1) / 2.0;
    const double width = dk1 - dk;
    const double cc = block_eval_value_only_device<BlockThreads>(
        2, k, dd, D, z, pinv, midpoint, ssfmin);
    const double gdx = z[k] * z[k];
    const double hdx = z[k + 1] * z[k + 1];
    const double fx = cc + 2.0 * (hdx - gdx) / width;
    if (fx > 0.0) {
        return block_solve_direction_device<BlockThreads, true>(
            dd, D, z, p, k, ev, tol, ssfmin, dk, dk1, cc, gdx, hdx);
    }
    return block_solve_direction_device<BlockThreads, false>(
        dd, D, z, p, k, ev, tol, ssfmin, dk, dk1, cc, gdx, hdx);
}

template <int BlockThreads>
__device__ inline int
block_solve_ext_device(const int dd, double *D, const double *z, const double p,
                       double *ev, const double tol, const double eps,
                       const double ssfmin, const double ssfmax) {
    (void)eps;
    (void)ssfmax;
    bool converged = false;
    double lowb = 0.0;
    double uppb = 0.0;
    double aa = 0.0;
    double bb = 0.0;
    double cc = 0.0;
    double x = 0.0;
    double er = 0.0;
    double fx = 0.0;
    double fdx = 0.0;
    double gx = 0.0;
    double gdx = 0.0;
    double hx = 0.0;
    double hdx = 0.0;
    double tau = 0.0;
    double eta = 0.0;
    const int k = dd - 1;
    const int km1 = dd - 2;
    const double dk = D[k];
    const double dkm1 = D[km1];
    double ddk = 0.0;
    double ddkm1 = 0.0;
    int iterations = 1;
    const double pinv = 1.0 / p;

    x = dk + p / 2.0;

    cc = block_eval_value_only_device<BlockThreads>(2, km1, dd, D, z, pinv,
                                                    x, ssfmin);
    gdx = z[km1] * z[km1];
    hdx = z[k] * z[k];
    fx = cc + gdx / (dkm1 - x) - 2.0 * hdx * pinv;
    if (fx > 0.0) {
        lowb = 0.0;
        uppb = p / 2.0;
        tau = dk - dkm1;
        aa = -cc * tau + gdx + hdx;
        bb = hdx * tau;
        eta = sqrt(aa * aa + 4.0 * bb * cc);
        tau = aa < 0.0 ? (2.0 * bb) / (eta - aa) : (aa + eta) / (2.0 * cc);
    } else {
        lowb = p / 2.0;
        uppb = p;
        eta = gdx / (dk - dkm1 + p) + hdx / p;
        if (cc <= eta) {
            tau = p;
        } else {
            tau = dk - dkm1;
            aa = -cc * tau + gdx + hdx;
            bb = hdx * tau;
            eta = sqrt(aa * aa + 4.0 * bb * cc);
            tau = aa < 0.0 ? (2.0 * bb) / (eta - aa)
                            : (aa + eta) / (2.0 * cc);
        }
    }
    x = dk + tau;

    block_eval_shifted_values<BlockThreads>(0, km1, dd, D, z, pinv, dk, tau,
                                            &fx, &fdx, &gx, &gdx, &hx, &hdx,
                                            &er, true, ssfmin);
    er += device_abs(tau) * (hdx + gdx) - 8.0 * (hx + gx) - hx + pinv;
    if (device_abs(fx) <= tol * er) {
        converged = true;
    } else {
        lowb = (fx <= 0.0) ? fmax(lowb, tau) : lowb;
        uppb = (fx > 0.0) ? fmin(uppb, tau) : uppb;

        ddk = D[k];
        ddkm1 = D[km1];
        cc = device_abs(fx - ddkm1 * gdx - ddk * hdx);
        aa = (ddk + ddkm1) * fx - ddk * ddkm1 * (gdx + hdx);
        bb = ddk * ddkm1 * fx;
        if (cc == 0.0) {
            eta = uppb - tau;
        } else {
            eta = sqrt(device_abs(aa * aa - 4.0 * bb * cc));
            eta = aa >= 0.0 ? (aa + eta) / (2.0 * cc)
                             : (2.0 * bb) / (aa - eta);
        }

        if (fx * eta > 0.0) {
            eta = -fx / (gdx + hdx);
        }
        if (tau + eta > uppb || tau + eta < lowb) {
            eta = fx < 0.0 ? (uppb - tau) / 2.0 : (lowb - tau) / 2.0;
        }

        tau += eta;
        x = dk + tau;

        block_eval_values<BlockThreads>(0, km1, dd, D, z, pinv, eta, &fx,
                                        &fdx, &gx, &gdx, &hx, &hdx, &er, true,
                                        ssfmin);
        er += device_abs(tau) * (hdx + gdx) - 8.0 * (hx + gx) - hx + pinv;
        iterations = 2;

        for (int iter = 1; iter < kSecularMaxIterations; ++iter) {
            if (device_abs(fx) <= tol * er) {
                converged = true;
                break;
            }

            lowb = (fx <= 0.0) ? fmax(lowb, tau) : lowb;
            uppb = (fx > 0.0) ? fmin(uppb, tau) : uppb;

            ddk = D[k];
            ddkm1 = D[km1];
            cc = fx - ddkm1 * gdx - ddk * hdx;
            aa = (ddk + ddkm1) * fx - ddk * ddkm1 * (gdx + hdx);
            bb = ddk * ddkm1 * fx;
            eta = sqrt(device_abs(aa * aa - 4.0 * bb * cc));
            eta = aa >= 0.0 ? (aa + eta) / (2.0 * cc)
                             : (2.0 * bb) / (aa - eta);

            if (fx * eta > 0.0) {
                eta = -fx / (gdx + hdx);
            }
            if (tau + eta > uppb || tau + eta < lowb) {
                eta = fx < 0.0 ? (uppb - tau) / 2.0 : (lowb - tau) / 2.0;
            }

            tau += eta;
            x = dk + tau;

            block_eval_values<BlockThreads>(0, km1, dd, D, z, pinv, eta, &fx,
                                            &fdx, &gx, &gdx, &hx, &hdx, &er,
                                            true, ssfmin);
            er += device_abs(tau) * (hdx + gdx) - 8.0 * (hx + gx) - hx + pinv;
            iterations = iter + 2;
        }
    }

    if (threadIdx.x == 0) {
        *ev = x;
    }
    return secular_solve_info(converged, iterations);
}

__device__ inline int
seq_solve_ext_device(const int dd, double *D, const double *z, const double p,
                     double *ev, const double tol, const double eps,
                     const double ssfmin, const double ssfmax) {
    bool converged = false;
    double lowb = 0.0;
    double uppb = 0.0;
    double aa = 0.0;
    double bb = 0.0;
    double cc = 0.0;
    double x = 0.0;
    double er = 0.0;
    double fx = 0.0;
    double fdx = 0.0;
    double gx = 0.0;
    double gdx = 0.0;
    double hx = 0.0;
    double hdx = 0.0;
    double tau = 0.0;
    double eta = 0.0;
    const int k = dd - 1;
    const int km1 = dd - 2;
    const double dk = D[k];
    const double dkm1 = D[km1];
    double ddk = 0.0;
    double ddkm1 = 0.0;
    int iterations = 1;
    const double pinv = 1.0 / p;

    x = dk + p / 2.0;

    seq_eval_device(2, km1, dd, D, z, pinv, x, &cc, &fdx, &gx, &gdx, &hx, &hdx,
                    &er, false, eps, ssfmin, ssfmax);
    gdx = z[km1] * z[km1];
    hdx = z[k] * z[k];
    fx = cc + gdx / (dkm1 - x) - 2.0 * hdx * pinv;
    if (fx > 0.0) {
        lowb = 0.0;
        uppb = p / 2.0;
        tau = dk - dkm1;
        aa = -cc * tau + gdx + hdx;
        bb = hdx * tau;
        eta = sqrt(aa * aa + 4.0 * bb * cc);
        tau = aa < 0.0 ? (2.0 * bb) / (eta - aa) : (aa + eta) / (2.0 * cc);
    } else {
        lowb = p / 2.0;
        uppb = p;
        eta = gdx / (dk - dkm1 + p) + hdx / p;
        if (cc <= eta) {
            tau = p;
        } else {
            tau = dk - dkm1;
            aa = -cc * tau + gdx + hdx;
            bb = hdx * tau;
            eta = sqrt(aa * aa + 4.0 * bb * cc);
            tau = aa < 0.0 ? (2.0 * bb) / (eta - aa) : (aa + eta) / (2.0 * cc);
        }
    }
    x = dk + tau;

    seq_eval_shifted_device(0, km1, dd, D, z, pinv, dk, tau, &fx, &fdx, &gx,
                            &gdx, &hx, &hdx, &er, true, eps, ssfmin, ssfmax);
    er += device_abs(tau) * (hdx + gdx) - 8.0 * (hx + gx) - hx + pinv;
    if (device_abs(fx) <= tol * er) {
        converged = true;
    } else {
        lowb = (fx <= 0.0) ? fmax(lowb, tau) : lowb;
        uppb = (fx > 0.0) ? fmin(uppb, tau) : uppb;

        ddk = D[k];
        ddkm1 = D[km1];
        cc = device_abs(fx - ddkm1 * gdx - ddk * hdx);
        aa = (ddk + ddkm1) * fx - ddk * ddkm1 * (gdx + hdx);
        bb = ddk * ddkm1 * fx;
        if (cc == 0.0) {
            eta = uppb - tau;
        } else {
            eta = sqrt(device_abs(aa * aa - 4.0 * bb * cc));
            eta = aa >= 0.0 ? (aa + eta) / (2.0 * cc) : (2.0 * bb) / (aa - eta);
        }

        if (fx * eta > 0.0) {
            eta = -fx / (gdx + hdx);
        }
        if (tau + eta > uppb || tau + eta < lowb) {
            eta = fx < 0.0 ? (uppb - tau) / 2.0 : (lowb - tau) / 2.0;
        }

        tau += eta;
        x = dk + tau;

        seq_eval_device(0, km1, dd, D, z, pinv, eta, &fx, &fdx, &gx, &gdx, &hx,
                        &hdx, &er, true, eps, ssfmin, ssfmax);
        er += device_abs(tau) * (hdx + gdx) - 8.0 * (hx + gx) - hx + pinv;
        iterations = 2;

        for (int iter = 1; iter < kSecularMaxIterations; ++iter) {
            if (device_abs(fx) <= tol * er) {
                converged = true;
                break;
            }

            lowb = (fx <= 0.0) ? fmax(lowb, tau) : lowb;
            uppb = (fx > 0.0) ? fmin(uppb, tau) : uppb;

            ddk = D[k];
            ddkm1 = D[km1];
            cc = fx - ddkm1 * gdx - ddk * hdx;
            aa = (ddk + ddkm1) * fx - ddk * ddkm1 * (gdx + hdx);
            bb = ddk * ddkm1 * fx;
            eta = sqrt(device_abs(aa * aa - 4.0 * bb * cc));
            eta = aa >= 0.0 ? (aa + eta) / (2.0 * cc) : (2.0 * bb) / (aa - eta);

            if (fx * eta > 0.0) {
                eta = -fx / (gdx + hdx);
            }
            if (tau + eta > uppb || tau + eta < lowb) {
            eta = fx < 0.0 ? (uppb - tau) / 2.0 : (lowb - tau) / 2.0;
        }

            tau += eta;
            x = dk + tau;

            seq_eval_device(0, km1, dd, D, z, pinv, eta, &fx, &fdx, &gx, &gdx,
                            &hx, &hdx, &er, true, eps, ssfmin, ssfmax);
            er += device_abs(tau) * (hdx + gdx) - 8.0 * (hx + gx) - hx + pinv;
            iterations = iter + 2;
        }
    }

    *ev = x;
    return secular_solve_info(converged, iterations);
}

__global__ void stedc_merge_values_solve_kernel(const int n, double *tmpz,
                                                double *temp_deltas,
                                                int *splits, const double eps,
                                                const double ssfmin,
                                                const double ssfmax) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) {
        return;
    }

    int *ndd = ptr_ndd(n, splits);
    int *nps = ptr_nps(n, splits);
    int *dbg = ptr_dbg(n, splits);
    double *z = ptr_z(n, tmpz);
    double *r1p = ptr_r1p(n, tmpz);
    double *evs = ptr_evs(n, tmpz);

    const int start = nps[i];
    const int dd = ndd[i];
    const int local = i - start;
    dbg[i] = 0;
    if (local < 0 || local >= dd || temp_deltas == nullptr) {
        return;
    }

    const double p = r1p[i];
    double *etmpd = temp_deltas + dense_index(0, i, n);
    const double rho = device_abs(p);
    int info = 0;
    if (dd == 1) {
        evs[i] = etmpd[0] + rho * z[start] * z[start];
        etmpd[0] -= evs[i];
        info = isfinite(evs[i]) ? 0 : 1;
    } else if (rho == 0.0) {
        evs[i] = etmpd[local];
        for (int j = 0; j < dd; ++j) {
            etmpd[j] -= evs[i];
        }
    } else {
        const double tol = eps;
        info = (local == dd - 1)
                   ? seq_solve_ext_device(dd, etmpd, z + start, rho, evs + i,
                                          tol, eps, ssfmin, ssfmax)
                   : seq_solve_device(dd, etmpd, z + start, rho, local, evs + i,
                                      tol, eps, ssfmin, ssfmax);
    }

    if (p < 0.0) {
        evs[i] *= -1.0;
    }
    dbg[i] = info;
}

template <int BlockThreads>
__global__ void stedc_merge_values_solve_cub_kernel(
    const int n, double *tmpz, double *temp_deltas, int *splits,
    const double eps, const double ssfmin, const double ssfmax) {
    const int i = static_cast<int>(blockIdx.x);
    if (i >= n) {
        return;
    }

    int *ndd = ptr_ndd(n, splits);
    int *nps = ptr_nps(n, splits);
    int *dbg = ptr_dbg(n, splits);
    double *z = ptr_z(n, tmpz);
    double *r1p = ptr_r1p(n, tmpz);
    double *evs = ptr_evs(n, tmpz);

    const int start = nps[i];
    const int dd = ndd[i];
    const int local = i - start;
    if (threadIdx.x == 0) {
        dbg[i] = 0;
    }
    if (local < 0 || local >= dd || temp_deltas == nullptr) {
        return;
    }

    const double p = r1p[i];
    double *etmpd = temp_deltas + dense_index(0, i, n);
    const double rho = device_abs(p);
    int info = 0;
    if (dd == 1) {
        if (threadIdx.x == 0) {
            evs[i] = etmpd[0] + rho * z[start] * z[start];
            etmpd[0] -= evs[i];
            info = isfinite(evs[i]) ? 0 : 1;
        }
    } else if (rho == 0.0) {
        if (threadIdx.x == 0) {
            evs[i] = etmpd[local];
        }
        __syncthreads();
        for (int j = threadIdx.x; j < dd; j += BlockThreads) {
            etmpd[j] -= evs[i];
        }
    } else {
        const double tol = eps;
        if (dd < kSecularSolveCubMinDd || local == dd - 1) {
            if (threadIdx.x == 0) {
                info =
                    (local == dd - 1)
                        ? seq_solve_ext_device(dd, etmpd, z + start, rho,
                                               evs + i, tol, eps, ssfmin,
                                               ssfmax)
                        : seq_solve_device(dd, etmpd, z + start, rho, local,
                                           evs + i, tol, eps, ssfmin, ssfmax);
            }
        } else {
            info = block_solve_device<BlockThreads>(dd, etmpd, z + start, rho,
                                                    local, evs + i, tol, eps,
                                                    ssfmin, ssfmax);
        }
    }

    if (threadIdx.x == 0) {
        if (p < 0.0) {
            evs[i] *= -1.0;
        }
        dbg[i] = info;
    }
}

template <int BlockThreads>
__global__ void stedc_merge_values_solve_chunk_cub_kernel(
    const int n, const int chunk_start, const int chunk_size,
    const int *active_indices, double *tmpz, double *temp_deltas, int *splits,
    const double eps, const double ssfmin, const double ssfmax) {
    const int chunk_col = static_cast<int>(blockIdx.x);
    if (chunk_col >= chunk_size) {
        return;
    }
    const int i = active_indices != nullptr ? active_indices[chunk_start + chunk_col]
                                            : chunk_start + chunk_col;
    if (i >= n) {
        return;
    }

    int *ndd = ptr_ndd(n, splits);
    int *nps = ptr_nps(n, splits);
    int *dbg = ptr_dbg(n, splits);
    double *z = ptr_z(n, tmpz);
    double *r1p = ptr_r1p(n, tmpz);
    double *evs = ptr_evs(n, tmpz);

    const int start = nps[i];
    const int dd = ndd[i];
    const int local = i - start;
    if (local < 0 || local >= dd || dd < kSecularSolveCubMinDd ||
        local == dd - 1) {
        return;
    }
    if (threadIdx.x == 0) {
        dbg[i] = 0;
    }

    const double p = r1p[i];
    double *etmpd =
        temp_deltas +
        static_cast<std::size_t>(chunk_col) * static_cast<std::size_t>(n);
    const double rho = device_abs(p);
    int info = 0;
    if (dd == 1) {
        if (threadIdx.x == 0) {
            evs[i] = etmpd[0] + rho * z[start] * z[start];
            etmpd[0] -= evs[i];
            info = isfinite(evs[i]) ? 0 : 1;
        }
    } else if (rho == 0.0) {
        if (threadIdx.x == 0) {
            evs[i] = etmpd[local];
        }
        __syncthreads();
        for (int j = threadIdx.x; j < dd; j += BlockThreads) {
            etmpd[j] -= evs[i];
        }
    } else {
        const double tol = eps;
        info = block_solve_device<BlockThreads>(dd, etmpd, z + start, rho,
                                                local, evs + i, tol, eps,
                                                ssfmin, ssfmax);
    }

    if (threadIdx.x == 0) {
        if (p < 0.0) {
            evs[i] *= -1.0;
        }
        dbg[i] = info;
    }
}

template <int BlockThreads>
__global__ void stedc_merge_values_solve_chunk_ext_cub_kernel(
    const int n, const int chunk_start, const int chunk_size,
    const int *active_indices, double *tmpz, double *temp_deltas, int *splits,
    const double eps, const double ssfmin, const double ssfmax) {
    const int chunk_col = static_cast<int>(blockIdx.x);
    if (chunk_col >= chunk_size) {
        return;
    }
    const int i = active_indices != nullptr ? active_indices[chunk_start + chunk_col]
                                            : chunk_start + chunk_col;
    if (i >= n) {
        return;
    }

    int *ndd = ptr_ndd(n, splits);
    int *nps = ptr_nps(n, splits);
    int *dbg = ptr_dbg(n, splits);
    double *z = ptr_z(n, tmpz);
    double *r1p = ptr_r1p(n, tmpz);
    double *evs = ptr_evs(n, tmpz);

    const int start = nps[i];
    const int dd = ndd[i];
    const int local = i - start;
    if (local < 0 || local >= dd || dd < kSecularSolveCubMinDd ||
        local != dd - 1) {
        return;
    }

    const double p = r1p[i];
    double *etmpd =
        temp_deltas +
        static_cast<std::size_t>(chunk_col) * static_cast<std::size_t>(n);
    const double rho = device_abs(p);
    int info = 0;
    if (rho == 0.0) {
        if (threadIdx.x == 0) {
            evs[i] = etmpd[local];
        }
        __syncthreads();
        for (int j = threadIdx.x; j < dd; j += BlockThreads) {
            etmpd[j] -= evs[i];
        }
    } else {
        const double tol = eps;
        info = block_solve_ext_device<BlockThreads>(dd, etmpd, z + start, rho,
                                                    evs + i, tol, eps, ssfmin,
                                                    ssfmax);
    }

    if (threadIdx.x == 0) {
        if (p < 0.0) {
            evs[i] *= -1.0;
        }
        dbg[i] = info;
    }
}

__global__ void stedc_merge_values_solve_chunk_seq_kernel(
    const int n, const int chunk_start, const int chunk_size,
    const int *active_indices, double *tmpz, double *temp_deltas, int *splits,
    const double eps, const double ssfmin, const double ssfmax) {
    const int chunk_col =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (chunk_col >= chunk_size) {
        return;
    }
    const int i = active_indices != nullptr ? active_indices[chunk_start + chunk_col]
                                            : chunk_start + chunk_col;
    if (i >= n) {
        return;
    }

    int *ndd = ptr_ndd(n, splits);
    int *nps = ptr_nps(n, splits);
    int *dbg = ptr_dbg(n, splits);
    double *z = ptr_z(n, tmpz);
    double *r1p = ptr_r1p(n, tmpz);
    double *evs = ptr_evs(n, tmpz);

    const int start = nps[i];
    const int dd = ndd[i];
    const int local = i - start;
    dbg[i] = 0;
    if (local < 0 || local >= dd) {
        return;
    }

    const double p = r1p[i];
    double *etmpd =
        temp_deltas +
        static_cast<std::size_t>(chunk_col) * static_cast<std::size_t>(n);
    const double rho = device_abs(p);
    int info = 0;
    if (dd == 1) {
        evs[i] = etmpd[0] + rho * z[start] * z[start];
        etmpd[0] -= evs[i];
        info = isfinite(evs[i]) ? 0 : 1;
    } else if (rho == 0.0) {
        evs[i] = etmpd[local];
        for (int j = 0; j < dd; ++j) {
            etmpd[j] -= evs[i];
        }
    } else {
        const double tol = eps;
        info = (local == dd - 1)
                   ? seq_solve_ext_device(dd, etmpd, z + start, rho, evs + i,
                                          tol, eps, ssfmin, ssfmax)
                   : seq_solve_device(dd, etmpd, z + start, rho, local, evs + i,
                                      tol, eps, ssfmin, ssfmax);
    }

    if (p < 0.0) {
        evs[i] *= -1.0;
    }
    dbg[i] = info;
}

__global__ void stedc_merge_values_solve_chunk_fallback_seq_kernel(
    const int n, const int chunk_start, const int chunk_size,
    const int *active_indices, double *tmpz, double *temp_deltas, int *splits,
    const double eps, const double ssfmin, const double ssfmax) {
    const int chunk_col =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (chunk_col >= chunk_size) {
        return;
    }
    const int i = active_indices != nullptr ? active_indices[chunk_start + chunk_col]
                                            : chunk_start + chunk_col;
    if (i >= n) {
        return;
    }

    int *ndd = ptr_ndd(n, splits);
    int *nps = ptr_nps(n, splits);
    int *dbg = ptr_dbg(n, splits);
    double *z = ptr_z(n, tmpz);
    double *r1p = ptr_r1p(n, tmpz);
    double *evs = ptr_evs(n, tmpz);

    const int start = nps[i];
    const int dd = ndd[i];
    const int local = i - start;
    if (local < 0 || local >= dd || dd >= kSecularSolveCubMinDd) {
        return;
    }

    const double p = r1p[i];
    double *etmpd =
        temp_deltas +
        static_cast<std::size_t>(chunk_col) * static_cast<std::size_t>(n);
    const double rho = device_abs(p);
    int info = 0;
    if (dd == 1) {
        evs[i] = etmpd[0] + rho * z[start] * z[start];
        etmpd[0] -= evs[i];
        info = isfinite(evs[i]) ? 0 : 1;
    } else if (rho == 0.0) {
        evs[i] = etmpd[local];
        for (int j = 0; j < dd; ++j) {
            etmpd[j] -= evs[i];
        }
    } else {
        const double tol = eps;
        info = (local == dd - 1)
                   ? seq_solve_ext_device(dd, etmpd, z + start, rho, evs + i,
                                          tol, eps, ssfmin, ssfmax)
                   : seq_solve_device(dd, etmpd, z + start, rho, local, evs + i,
                                      tol, eps, ssfmin, ssfmax);
    }

    if (p < 0.0) {
        evs[i] *= -1.0;
    }
    dbg[i] = info;
}

template <int BlockThreads>
__global__ void stedc_merge_values_solve_chunk_adaptive_cub_kernel(
    const int n, const int chunk_start, const int chunk_size,
    const int *active_indices, double *tmpz, double *temp_deltas, int *splits,
    const double eps, const double ssfmin, const double ssfmax) {
    const int chunk_col = static_cast<int>(blockIdx.x);
    if (chunk_col >= chunk_size) {
        return;
    }
    const int i = active_indices != nullptr ? active_indices[chunk_start + chunk_col]
                                            : chunk_start + chunk_col;
    if (i >= n) {
        return;
    }

    int *ndd = ptr_ndd(n, splits);
    int *nps = ptr_nps(n, splits);
    int *dbg = ptr_dbg(n, splits);
    double *z = ptr_z(n, tmpz);
    double *r1p = ptr_r1p(n, tmpz);
    double *evs = ptr_evs(n, tmpz);

    const int start = nps[i];
    const int dd = ndd[i];
    const int local = i - start;
    if (local < 0 || local >= dd) {
        return;
    }
    if (threadIdx.x == 0) {
        dbg[i] = 0;
    }

    const double p = r1p[i];
    double *etmpd =
        temp_deltas +
        static_cast<std::size_t>(chunk_col) * static_cast<std::size_t>(n);
    const double rho = device_abs(p);
    int info = 0;

    if (dd < kSecularSolveCubMinDd) {
        if (threadIdx.x == 0) {
            if (dd == 1) {
                evs[i] = etmpd[0] + rho * z[start] * z[start];
                etmpd[0] -= evs[i];
                info = isfinite(evs[i]) ? 0 : 1;
            } else if (rho == 0.0) {
                evs[i] = etmpd[local];
                for (int j = 0; j < dd; ++j) {
                    etmpd[j] -= evs[i];
                }
            } else {
                const double tol = eps;
                info = (local == dd - 1)
                           ? seq_solve_ext_device(dd, etmpd, z + start, rho,
                                                  evs + i, tol, eps, ssfmin,
                                                  ssfmax)
                           : seq_solve_device(dd, etmpd, z + start, rho, local,
                                              evs + i, tol, eps, ssfmin,
                                              ssfmax);
            }
            if (p < 0.0) {
                evs[i] *= -1.0;
            }
            dbg[i] = info;
        }
        return;
    }

    if (dd == 1) {
        if (threadIdx.x == 0) {
            evs[i] = etmpd[0] + rho * z[start] * z[start];
            etmpd[0] -= evs[i];
            info = isfinite(evs[i]) ? 0 : 1;
        }
    } else if (rho == 0.0) {
        if (threadIdx.x == 0) {
            evs[i] = etmpd[local];
        }
        __syncthreads();
        for (int j = threadIdx.x; j < dd; j += BlockThreads) {
            etmpd[j] -= evs[i];
        }
    } else {
        const double tol = eps;
        info = (local == dd - 1)
                   ? block_solve_ext_device<BlockThreads>(
                         dd, etmpd, z + start, rho, evs + i, tol, eps, ssfmin,
                         ssfmax)
                   : block_solve_device<BlockThreads>(
                         dd, etmpd, z + start, rho, local, evs + i, tol, eps,
                         ssfmin, ssfmax);
    }

    if (threadIdx.x == 0) {
        if (p < 0.0) {
            evs[i] *= -1.0;
        }
        dbg[i] = info;
    }
}

} // namespace

} // namespace evd::internal

#endif
