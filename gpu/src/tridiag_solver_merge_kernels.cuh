#ifndef EVD_LIBRARY_INTERNAL_TRIDIAG_SOLVER_MERGE_KERNELS_CUH
#define EVD_LIBRARY_INTERNAL_TRIDIAG_SOLVER_MERGE_KERNELS_CUH

#include "tridiag_solver_dc.cuh"
#include "tridiag_solver_device.cuh"

#include <cub/block/block_reduce.cuh>

#ifndef EVD_TRIDIAG_DEFLATION_TOL_SCALE
#define EVD_TRIDIAG_DEFLATION_TOL_SCALE 8.0
#endif

namespace evd::internal {

namespace {

struct DeviceMax {
    __device__ __forceinline__ double operator()(const double a,
                                                 const double b) const {
        return a > b ? a : b;
    }
};

struct DeviceMultiply {
    __device__ __forceinline__ double operator()(const double a,
                                                 const double b) const {
        return a * b;
    }
};

__global__ void stedc_update_splits_copy_blocks_kernel(const int levs,
                                                       const int k,
                                                       const int n,
                                                       int *splits) {
    int *msz = ptr_msz(n, splits);
    int *mps = ptr_mps(n, splits);
    int *bsz = ptr_bsz(n, splits);
    int *bps = ptr_bps(n, splits);

    const int n_merges = 1 << (levs - k - 1);
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < 2 * n_merges) {
        bsz[i] = msz[i];
        bps[i] = mps[i];
    }
}

__global__ void stedc_update_splits_merge_blocks_kernel(const int levs,
                                                        const int k,
                                                        const int n,
                                                        int *splits) {
    int *msz = ptr_msz(n, splits);
    int *mps = ptr_mps(n, splits);
    int *bsz = ptr_bsz(n, splits);
    int *bps = ptr_bps(n, splits);

    const int n_merges = 1 << (levs - k - 1);
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n_merges) {
        msz[i] = bsz[2 * i] + bsz[2 * i + 1];
        mps[i] = bps[2 * i];
    }
}

__global__ void stedc_update_splits_rows_kernel(const int n, int *splits) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) {
        return;
    }

    int *msz = ptr_msz(n, splits);
    int *mps = ptr_mps(n, splits);
    int *em = ptr_em(n, splits);
    int *nsz = ptr_nsz(n, splits);
    int *nps = ptr_nps(n, splits);
    int *map = ptr_map(n, splits);
    int *dcount = ptr_dcount(n, splits);
    int *mask = ptr_mask(n, splits);
    int *cand = ptr_cand(n, splits);

    const int merge = em[i] / 2;
    nsz[i] = msz[merge];
    nps[i] = mps[merge];
    map[i] = i;
    dcount[i] = 0;
    mask[i] = 1;
    cand[i] = 0;
    em[i] = merge;
}

__global__ void stedc_merge_prepare_deflatezero_kernel(
    const int merge_count, const int n, const double *D, const double *E,
    const double *V, const int ldv, double *tmpz, int *splits,
    const double eps) {
    const int merge = static_cast<int>(blockIdx.x);
    if (merge >= merge_count || threadIdx.x != 0) {
        return;
    }

    int *bsz = ptr_bsz(n, splits);
    int *bps = ptr_bps(n, splits);
    int *mask = ptr_mask(n, splits);
    double *z = ptr_z(n, tmpz);
    double *r1p = ptr_r1p(n, tmpz);
    double *tolsD = ptr_tolsD(n, tmpz);
    double *tolsZ = ptr_tolsZ(n, tmpz);

    const int size_left = bsz[2 * merge];
    const int size_right = bsz[2 * merge + 1];
    const int start = bps[2 * merge];
    const int split = bps[2 * merge + 1];
    const int size = size_left + size_right;

    const double rho = 2.0 * E[split - 1];
    double maxd = 0.0;
    double maxz = 0.0;
    for (int i = 0; i < size_left; ++i) {
        const double val =
            V[dense_index(split - 1, start + i, ldv)] / sqrt(2.0);
        z[start + i] = val;
        maxz = maxz > device_abs(val) ? maxz : device_abs(val);
    }
    for (int i = 0; i < size_right; ++i) {
        const double val = V[dense_index(split, split + i, ldv)] / sqrt(2.0);
        z[split + i] = val;
        maxz = maxz > device_abs(val) ? maxz : device_abs(val);
    }
    for (int i = 0; i < size; ++i) {
        maxd =
            maxd > device_abs(D[start + i]) ? maxd : device_abs(D[start + i]);
    }
    const double tol =
        EVD_TRIDIAG_DEFLATION_TOL_SCALE * eps * (maxd > maxz ? maxd : maxz);

    for (int i = 0; i < size; ++i) {
        r1p[start + i] = rho;
        tolsD[start + i] = tol;
        tolsZ[start + i] = tol;
        mask[start + i] = device_abs(rho * z[start + i]) > tol ? 1 : 0;
    }
}

__global__ void stedc_merge_sortd_kernel(const int n, const double *D,
                                         double *tmpz, int *splits) {
    const int gid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (gid >= n) {
        return;
    }

    int *mask = ptr_mask(n, splits);
    int *ndd = ptr_ndd(n, splits);
    int *map = ptr_map(n, splits);
    int *nps = ptr_nps(n, splits);
    int *nsz = ptr_nsz(n, splits);
    double *md = ptr_md(n, tmpz);

    const int start = nps[gid];
    const int size = nsz[gid];
    const int active = mask[gid];
    const double value = D[gid];

    int lt = 0;
    int eq = 0;
    int active_count = 0;
    for (int i = 0; i < size; ++i) {
        const int idx = start + i;
        const int other_active = mask[idx];
        active_count += other_active;
        if ((active < other_active) ||
            (active == other_active && D[idx] < value)) {
            ++lt;
        }
        if (active == other_active && D[idx] == value && idx < gid) {
            ++eq;
        }
    }

    const int pos = start + lt + eq;
    md[pos] = value;
    map[pos] = gid;
    ndd[pos] = active_count;
}

__global__ void stedc_merge_setcandflags_kernel(const int n, double *tmpz,
                                                int *splits) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) {
        return;
    }

    constexpr int F_BCAND = 1;
    constexpr int F_TCAND = 2;

    int *nps = ptr_nps(n, splits);
    int *ndd = ptr_ndd(n, splits);
    int *cand = ptr_cand(n, splits);
    double *md = ptr_md(n, tmpz);
    double *tolsD = ptr_tolsD(n, tmpz);

    const int next = (i + 1) < n ? (i + 1) : i;
    const int prev = i > 0 ? (i - 1) : 0;
    const int start = nps[i];
    const int next_start = nps[next];
    const int prev_start = nps[prev];
    const int active = ndd[i];
    const double tol = tolsD[i];

    const int bcandidate = (i - start < active - 1) && (start == next_start) &&
                           (device_abs(md[i] - md[next]) <= tol) &&
                           (i != n - 1);
    const int tcandidate = (i - start < active) && (start == prev_start) &&
                           (device_abs(md[i] - md[prev]) <= tol) && (i > 0);
    cand[i] = (bcandidate ? F_BCAND : 0) | (tcandidate ? F_TCAND : 0);
}

__global__ void stedc_merge_deflatecount_kernel(const int n, double *tmpz,
                                                int *splits) {
    const int base = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (base >= n) {
        return;
    }

    constexpr int F_BCAND = 1;
    constexpr int F_TCAND = 2;

    int *dcount = ptr_dcount(n, splits);
    int *cand = ptr_cand(n, splits);
    double *md = ptr_md(n, tmpz);
    double *tolsD = ptr_tolsD(n, tmpz);

    dcount[base] = 0;
    const int prev = base > 0 ? (base - 1) : 0;
    const int cand_prev = base > 0 ? cand[prev] : 0;
    if (!(cand[base] & F_BCAND) || (base > 0 && (cand_prev & F_BCAND))) {
        return;
    }

    double base_value = md[base];
    double tol = tolsD[base];
    int cluster_base = base;
    int top = base + 2;
    while (top < n && (cand[top] & F_TCAND)) {
        const double top_value = md[top];
        if ((top_value - base_value) > tol) {
            dcount[cluster_base] = top - cluster_base - 1;
            cluster_base = top;
            base_value = top_value;
            tol = tolsD[top];
        }
        ++top;
    }
    dcount[cluster_base] = top - cluster_base - 1;
}

__global__ void stedc_merge_deflateapply_kernel(const int n, double *tmpz,
                                                int *splits) {
    const int base = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (base >= n) {
        return;
    }

    int *mask = ptr_mask(n, splits);
    int *map = ptr_map(n, splits);
    int *dcount = ptr_dcount(n, splits);
    double *z = ptr_z(n, tmpz);
    double *cc = ptr_cc(n, tmpz);
    double *ss = ptr_ss(n, tmpz);

    const int cnt = dcount[base];
    if (cnt <= 0) {
        return;
    }

    const int base_idx = map[base];
    double base_val = z[base_idx];
    for (int j = 0; j < cnt; ++j) {
        const int top = base + j + 1;
        const int idx = map[top];
        double c = 0.0;
        double s = 0.0;
        double r = 0.0;
        device_lartg(base_val, z[idx], &c, &s, &r);
        base_val = r;
        mask[idx] = 0;
        z[idx] = 0.0;
        cc[idx] = c;
        ss[idx] = s;
    }
    z[base_idx] = base_val;
}

__global__ void stedc_merge_rotate_kernel(const int n, double *tmpz,
                                          int *splits, double *V,
                                          const int ldv) {
    const int base = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (base >= n) {
        return;
    }

    int *map = ptr_map(n, splits);
    int *dcount = ptr_dcount(n, splits);
    double *cc = ptr_cc(n, tmpz);
    double *ss = ptr_ss(n, tmpz);

    const int cnt = dcount[base];
    if (cnt <= 0) {
        return;
    }

    const int base_idx = map[base];
    for (int j = 1; j <= cnt; ++j) {
        const int top_idx = map[base + j];
        const double c = cc[top_idx];
        const double s = ss[top_idx];
        for (int row = 0; row < n; ++row) {
            const std::size_t base_offset = dense_index(row, base_idx, ldv);
            const std::size_t top_offset = dense_index(row, top_idx, ldv);
            const double vb = V[base_offset];
            const double vt = V[top_offset];
            V[base_offset] = vb * c - vt * s;
            V[top_offset] = vb * s + vt * c;
        }
    }
}

__global__ void stedc_project_boundary_rows_kernel(
    const int n, int *splits, const double *boundary_rows,
    double *projected_boundary_rows) {
    const int gid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (gid >= n) {
        return;
    }

    int *bps = ptr_bps(n, splits);
    int *em = ptr_em(n, splits);
    const double *old_first = ptr_first_rows(n, boundary_rows);
    const double *old_last = ptr_last_rows(n, boundary_rows);
    double *proj_first = ptr_first_rows(n, projected_boundary_rows);
    double *proj_last = ptr_last_rows(n, projected_boundary_rows);

    const int merge = em[gid];
    const int split = bps[2 * merge + 1];
    if (gid < split) {
        proj_first[gid] = old_first[gid];
        proj_last[gid] = 0.0;
    } else {
        proj_first[gid] = 0.0;
        proj_last[gid] = old_last[gid];
    }
}

__global__ void stedc_rotate_boundary_rows_kernel(
    const int n, double *tmpz, int *splits, double *boundary_rows) {
    const int base = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (base >= n) {
        return;
    }

    int *map = ptr_map(n, splits);
    int *dcount = ptr_dcount(n, splits);
    double *cc = ptr_cc(n, tmpz);
    double *ss = ptr_ss(n, tmpz);
    double *first_rows = ptr_first_rows(n, boundary_rows);
    double *last_rows = ptr_last_rows(n, boundary_rows);

    const int cnt = dcount[base];
    if (cnt <= 0) {
        return;
    }

    const int base_idx = map[base];
    for (int j = 1; j <= cnt; ++j) {
        const int top_idx = map[base + j];
        const double c = cc[top_idx];
        const double s = ss[top_idx];

        const double fb = first_rows[base_idx];
        const double ft = first_rows[top_idx];
        first_rows[base_idx] = fb * c - ft * s;
        first_rows[top_idx] = fb * s + ft * c;

        const double lb = last_rows[base_idx];
        const double lt = last_rows[top_idx];
        last_rows[base_idx] = lb * c - lt * s;
        last_rows[top_idx] = lb * s + lt * c;
    }
}

__global__ void stedc_merge_solvedz_kernel(const int n, const double *D,
                                           double *tmpz, int *splits) {
    const int gid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (gid >= n) {
        return;
    }

    int *mask = ptr_mask(n, splits);
    int *map = ptr_map(n, splits);
    int *ndd = ptr_ndd(n, splits);
    int *nps = ptr_nps(n, splits);
    int *nsz = ptr_nsz(n, splits);
    double *z = ptr_z(n, tmpz);
    double *cd = ptr_cd(n, tmpz);
    double *cz = ptr_cz(n, tmpz);
    double *evs = ptr_evs(n, tmpz);
    double *r1p = ptr_r1p(n, tmpz);

    const int start = nps[gid];
    const int size = nsz[gid];
    const int active = mask[gid];
    const double rho = r1p[gid];
    const double sign = rho < 0.0 ? -1.0 : 1.0;
    const double signed_value = sign * D[gid];
    const double z_value = z[gid];

    int lt = 0;
    int eq = 0;
    int active_count = 0;
    for (int i = 0; i < size; ++i) {
        const int idx = start + i;
        const int other_active = mask[idx];
        const double other_signed = sign * D[idx];
        active_count += other_active;
        if ((active < other_active) ||
            (active == other_active && other_signed < signed_value)) {
            ++lt;
        }
        if (active == other_active && other_signed == signed_value &&
            idx < gid) {
            ++eq;
        }
    }

    const int pos = start + lt + eq;
    cd[pos] = signed_value;
    cz[pos] = z_value;
    evs[pos] = D[gid];
    map[pos] = gid;
    ndd[pos] = active_count;
}

__global__ void stedc_copyc_kernel(const int n, const double *src,
                                   const int ldsrc, double *dst,
                                   const int lddst) {
    const int gid = static_cast<int>(blockIdx.x);
    if (gid >= n) {
        return;
    }

    const double *src_col = src + dense_index(0, gid, ldsrc);
    double *dst_col = dst + dense_index(0, gid, lddst);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        dst_col[i] = src_col[i];
    }
}

__global__ void stedc_reshufflec_kernel(const int n, const double *src,
                                        const int ldsrc, double *dst,
                                        const int lddst, int *splits) {
    const int gid = static_cast<int>(blockIdx.x);
    if (gid >= n) {
        return;
    }

    int *map = ptr_map(n, splits);
    const int src_col = map[gid];
    const double *src_ptr = src + dense_index(0, src_col, ldsrc);
    double *dst_ptr = dst + dense_index(0, gid, lddst);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        dst_ptr[i] = src_ptr[i];
    }
}

__global__ void stedc_merge_vectors_prepare_kernel(const int n, double *tmpz,
                                                   double *etmpd,
                                                   int *splits) {
    const int eid = static_cast<int>(blockIdx.x);
    if (eid >= n) {
        return;
    }

    const int tid = static_cast<int>(threadIdx.x);
    const int dim = static_cast<int>(blockDim.x);
    int *nps = ptr_nps(n, splits);
    int *ndd = ptr_ndd(n, splits);
    double *z = ptr_z(n, tmpz);

    const int start = nps[eid];
    const int dd = ndd[eid];
    if (eid - start >= dd) {
        return;
    }
    const std::size_t col_offset = dense_index(0, eid, n);

    __shared__ double scratch[256];
    double nrm = 0.0;
    for (int i = tid; i < dd; i += dim) {
        const double denom = etmpd[col_offset + static_cast<std::size_t>(i)];
        const double value = denom != 0.0 ? z[start + i] / denom : 0.0;
        etmpd[col_offset + static_cast<std::size_t>(i)] = value;
        nrm += value * value;
    }
    scratch[tid] = nrm;
    for (int r = dim / 2; r > 0; r /= 2) {
        __syncthreads();
        if (tid < r) {
            scratch[tid] += scratch[tid + r];
        }
    }
    __syncthreads();
    nrm = sqrt(scratch[0]);
    __syncthreads();

    for (int i = tid; i < dd; i += dim) {
        const std::size_t offset = col_offset + static_cast<std::size_t>(i);
        etmpd[offset] = nrm != 0.0 ? etmpd[offset] / nrm : 0.0;
    }
}

__global__ void stedc_merge_vectors_kernel(const int n, const double *V,
                                           const int ldv, double *tmpz,
                                           double *tempgemm, int *splits) {
    (void)tmpz;

    constexpr int kTileRows = 32;
    constexpr int kTileCols = 32;
    constexpr int kTileK = 32;
    constexpr int kBlockRows = 16;
    constexpr int kBlockCols = 16;

    const int row_tile = static_cast<int>(blockIdx.x) * kTileRows;
    const int col_tile = static_cast<int>(blockIdx.y) * kTileCols;
    if (row_tile >= n || col_tile >= n) {
        return;
    }

    int *nsz = ptr_nsz(n, splits);
    int *nps = ptr_nps(n, splits);
    int *ndd = ptr_ndd(n, splits);
    double *vecs = ptr_vecs(n, tempgemm);
    double *etmpd = ptr_etmpd(n, tempgemm);

    const int start = nps[col_tile];
    const int size = nsz[col_tile];
    const int dd = ndd[col_tile];
    if (col_tile - start >= dd || row_tile < start ||
        row_tile >= start + size) {
        return;
    }

    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int row0 = row_tile + tx;
    const int row1 = row0 + kBlockRows;
    const int col0 = col_tile + ty;
    const int col1 = col0 + kBlockCols;
    const bool valid_row0 = row0 < start + size;
    const bool valid_row1 = row1 < start + size;
    const bool valid_col0 = col0 < start + dd;
    const bool valid_col1 = col1 < start + dd;

    __shared__ double tile_v[kTileRows][kTileK + 1];
    __shared__ double tile_w[kTileK][kTileCols + 1];

    double acc00 = 0.0;
    double acc01 = 0.0;
    double acc10 = 0.0;
    double acc11 = 0.0;

    for (int k_base = 0; k_base < dd; k_base += kTileK) {
        const int local_k0 = ty;
        const int local_k1 = ty + kBlockCols;
        const int global_k0 = start + k_base + local_k0;
        const int global_k1 = start + k_base + local_k1;

        tile_v[tx][local_k0] = (valid_row0 && k_base + local_k0 < dd)
                                   ? V[dense_index(row0, global_k0, ldv)]
                                   : 0.0;
        tile_v[tx + kBlockRows][local_k0] =
            (valid_row1 && k_base + local_k0 < dd)
                ? V[dense_index(row1, global_k0, ldv)]
                : 0.0;
        tile_v[tx][local_k1] = (valid_row0 && k_base + local_k1 < dd)
                                   ? V[dense_index(row0, global_k1, ldv)]
                                   : 0.0;
        tile_v[tx + kBlockRows][local_k1] =
            (valid_row1 && k_base + local_k1 < dd)
                ? V[dense_index(row1, global_k1, ldv)]
                : 0.0;

        tile_w[tx][ty] = (k_base + tx < dd && valid_col0)
                             ? etmpd[dense_index(k_base + tx, col0, n)]
                             : 0.0;
        tile_w[tx][ty + kBlockCols] =
            (k_base + tx < dd && valid_col1)
                ? etmpd[dense_index(k_base + tx, col1, n)]
                : 0.0;
        tile_w[tx + kBlockRows][ty] =
            (k_base + tx + kBlockRows < dd && valid_col0)
                ? etmpd[dense_index(k_base + tx + kBlockRows, col0, n)]
                : 0.0;
        tile_w[tx + kBlockRows][ty + kBlockCols] =
            (k_base + tx + kBlockRows < dd && valid_col1)
                ? etmpd[dense_index(k_base + tx + kBlockRows, col1, n)]
                : 0.0;
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < kTileK; ++kk) {
            const double v0 = tile_v[tx][kk];
            const double v1 = tile_v[tx + kBlockRows][kk];
            const double w0 = tile_w[kk][ty];
            const double w1 = tile_w[kk][ty + kBlockCols];
            acc00 += v0 * w0;
            acc01 += v0 * w1;
            acc10 += v1 * w0;
            acc11 += v1 * w1;
        }
        __syncthreads();
    }

    if (valid_row0 && valid_col0) {
        vecs[dense_index(row0, col0, n)] = acc00;
    }
    if (valid_row0 && valid_col1) {
        vecs[dense_index(row0, col1, n)] = acc01;
    }
    if (valid_row1 && valid_col0) {
        vecs[dense_index(row1, col0, n)] = acc10;
    }
    if (valid_row1 && valid_col1) {
        vecs[dense_index(row1, col1, n)] = acc11;
    }
}

__global__ void stedc_merge_vectors_single_kernel(const int n, const int start,
                                                  const int size, const int dd,
                                                  const double *V,
                                                  const int ldv,
                                                  double *tempgemm) {
    constexpr int kTileRows = 32;
    constexpr int kTileCols = 32;
    constexpr int kTileK = 32;
    constexpr int kBlockRows = 16;
    constexpr int kBlockCols = 16;

    const int row_tile = start + static_cast<int>(blockIdx.x) * kTileRows;
    const int col_tile = start + static_cast<int>(blockIdx.y) * kTileCols;
    if (row_tile >= start + size || col_tile >= start + dd) {
        return;
    }

    double *vecs = ptr_vecs(n, tempgemm);
    double *etmpd = ptr_etmpd(n, tempgemm);

    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int row0 = row_tile + tx;
    const int row1 = row0 + kBlockRows;
    const int col0 = col_tile + ty;
    const int col1 = col0 + kBlockCols;
    const bool valid_row0 = row0 < start + size;
    const bool valid_row1 = row1 < start + size;
    const bool valid_col0 = col0 < start + dd;
    const bool valid_col1 = col1 < start + dd;

    __shared__ double tile_v[kTileRows][kTileK + 1];
    __shared__ double tile_w[kTileK][kTileCols + 1];

    double acc00 = 0.0;
    double acc01 = 0.0;
    double acc10 = 0.0;
    double acc11 = 0.0;

    for (int k_base = 0; k_base < dd; k_base += kTileK) {
        const int local_k0 = ty;
        const int local_k1 = ty + kBlockCols;
        const int global_k0 = start + k_base + local_k0;
        const int global_k1 = start + k_base + local_k1;

        tile_v[tx][local_k0] = (valid_row0 && k_base + local_k0 < dd)
                                   ? V[dense_index(row0, global_k0, ldv)]
                                   : 0.0;
        tile_v[tx + kBlockRows][local_k0] =
            (valid_row1 && k_base + local_k0 < dd)
                ? V[dense_index(row1, global_k0, ldv)]
                : 0.0;
        tile_v[tx][local_k1] = (valid_row0 && k_base + local_k1 < dd)
                                   ? V[dense_index(row0, global_k1, ldv)]
                                   : 0.0;
        tile_v[tx + kBlockRows][local_k1] =
            (valid_row1 && k_base + local_k1 < dd)
                ? V[dense_index(row1, global_k1, ldv)]
                : 0.0;

        tile_w[tx][ty] = (k_base + tx < dd && valid_col0)
                             ? etmpd[dense_index(k_base + tx, col0, n)]
                             : 0.0;
        tile_w[tx][ty + kBlockCols] =
            (k_base + tx < dd && valid_col1)
                ? etmpd[dense_index(k_base + tx, col1, n)]
                : 0.0;
        tile_w[tx + kBlockRows][ty] =
            (k_base + tx + kBlockRows < dd && valid_col0)
                ? etmpd[dense_index(k_base + tx + kBlockRows, col0, n)]
                : 0.0;
        tile_w[tx + kBlockRows][ty + kBlockCols] =
            (k_base + tx + kBlockRows < dd && valid_col1)
                ? etmpd[dense_index(k_base + tx + kBlockRows, col1, n)]
                : 0.0;
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < kTileK; ++kk) {
            const double v0 = tile_v[tx][kk];
            const double v1 = tile_v[tx + kBlockRows][kk];
            const double w0 = tile_w[kk][ty];
            const double w1 = tile_w[kk][ty + kBlockCols];
            acc00 += v0 * w0;
            acc01 += v0 * w1;
            acc10 += v1 * w0;
            acc11 += v1 * w1;
        }
        __syncthreads();
    }

    if (valid_row0 && valid_col0) {
        vecs[dense_index(row0, col0, n)] = acc00;
    }
    if (valid_row0 && valid_col1) {
        vecs[dense_index(row0, col1, n)] = acc01;
    }
    if (valid_row1 && valid_col0) {
        vecs[dense_index(row1, col0, n)] = acc10;
    }
    if (valid_row1 && valid_col1) {
        vecs[dense_index(row1, col1, n)] = acc11;
    }
}

__global__ void stedc_merge_update_kernel(const int n, double *D, double *E,
                                          double *V, const int ldv,
                                          double *tmpz, double *tempgemm,
                                          int *splits) {
    const int gid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (gid >= n) {
        return;
    }

    int *ndd = ptr_ndd(n, splits);
    int *nps = ptr_nps(n, splits);
    int *nsz = ptr_nsz(n, splits);
    double *evs = ptr_evs(n, tmpz);
    double *vecs = ptr_vecs(n, tempgemm);

    const int start = nps[gid];
    const int size = nsz[gid];
    const int dd = ndd[gid];
    const int local = gid - start;
    D[gid] = evs[gid];
    if (local < dd) {
        for (int row = start; row < start + size; ++row) {
            V[dense_index(row, gid, ldv)] = vecs[dense_index(row, gid, n)];
        }
    }
    if (gid < start + size - 1) {
        E[gid] = 0.0;
    }
}

__global__ void
stedc_merge_update_all_vectors_kernel(const int n, double *D, double *E,
                                      double *V, const int ldv, double *tmpz,
                                      double *tempgemm, int *splits) {
    const int gid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (gid >= n) {
        return;
    }

    int *nps = ptr_nps(n, splits);
    int *nsz = ptr_nsz(n, splits);
    double *evs = ptr_evs(n, tmpz);
    double *vecs = ptr_vecs(n, tempgemm);

    const int start = nps[gid];
    const int size = nsz[gid];
    const int local = gid - start;
    D[gid] = evs[gid];
    if (local >= 0 && local < size) {
        for (int row = start; row < start + size; ++row) {
            V[dense_index(row, gid, ldv)] = vecs[dense_index(row, gid, n)];
        }
    }
    if (gid < start + size - 1) {
        E[gid] = 0.0;
    }
}

__global__ void stedc_merge_update_values_kernel(const int n, double *D,
                                                 double *E, double *tmpz,
                                                 int *splits) {
    const int gid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (gid >= n) {
        return;
    }

    int *nps = ptr_nps(n, splits);
    int *nsz = ptr_nsz(n, splits);
    double *evs = ptr_evs(n, tmpz);

    const int start = nps[gid];
    const int size = nsz[gid];
    D[gid] = evs[gid];
    if (gid < start + size - 1) {
        E[gid] = 0.0;
    }
}

__global__ void stedc_sort_pairs_kernel(const int n, const double *din,
                                        const double *vin, const int ldvin,
                                        double *dout, double *vout,
                                        const int ldvout) {
    const int gid = static_cast<int>(blockIdx.x);
    if (gid >= n) {
        return;
    }

    const double value = din[gid];
    int lt = 0;
    int eq = 0;
    for (int x = 0; x < n; ++x) {
        const double other = din[x];
        lt += other < value;
        eq += (other == value && x < gid);
    }
    const int pos = lt + eq;
    if (threadIdx.x == 0) {
        dout[pos] = value;
    }
    if (vin != nullptr && vout != nullptr) {
        const double *src_col = vin + dense_index(0, gid, ldvin);
        double *dst_col = vout + dense_index(0, pos, ldvout);
        for (int row = threadIdx.x; row < n; row += blockDim.x) {
            dst_col[row] = src_col[row];
        }
    }
}

__global__ void stedc_merge_copyd_kernel(const int n, double *D, double *tmpz,
                                         int *splits, double *temp_deltas) {
    const int eid = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    if (eid >= n) {
        return;
    }

    int *ndd = ptr_ndd(n, splits);
    int *nps = ptr_nps(n, splits);
    double *evs = ptr_evs(n, tmpz);
    double *cd = ptr_cd(n, tmpz);

    const int dd = ndd[eid];
    const int start = nps[eid];
    const std::size_t col_offset = dense_index(0, eid, n);
    for (int i = tid; i < dd; i += blockDim.x) {
        temp_deltas[col_offset + static_cast<std::size_t>(i)] = cd[start + i];
    }
}

__global__ void stedc_merge_copyd_chunk_kernel(const int n,
                                               const int chunk_start,
                                               const int chunk_size,
                                               const int *active_indices,
                                               double *D, double *tmpz,
                                               int *splits,
                                               double *temp_deltas) {
    const int chunk_col = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    if (chunk_col >= chunk_size) {
        return;
    }
    const int eid = active_indices != nullptr ? active_indices[chunk_start + chunk_col]
                                              : chunk_start + chunk_col;
    if (eid >= n) {
        return;
    }

    int *ndd = ptr_ndd(n, splits);
    int *nps = ptr_nps(n, splits);
    double *cd = ptr_cd(n, tmpz);

    const int dd = ndd[eid];
    const int start = nps[eid];
    const int local = eid - start;
    if (local < 0 || local >= dd) {
        return;
    }
    const std::size_t col_offset =
        static_cast<std::size_t>(chunk_col) * static_cast<std::size_t>(n);
    for (int i = tid; i < dd; i += blockDim.x) {
        temp_deltas[col_offset + static_cast<std::size_t>(i)] = cd[start + i];
    }
}

__global__ void stedc_merge_rescale_kernel(const int n, const double *D,
                                           double *tmpz, int *splits,
                                           double *temp_deltas) {
    const int eid = static_cast<int>(blockIdx.x);
    if (eid >= n || threadIdx.x != 0) {
        return;
    }

    int *nps = ptr_nps(n, splits);
    int *ndd = ptr_ndd(n, splits);
    double *cz = ptr_cz(n, tmpz);

    const int start = nps[eid];
    const int dd = ndd[eid];
    const int local = eid - start;
    if (local < 0 || local >= dd) {
        return;
    }

    double val = 1.0;
    const std::size_t col_offset = dense_index(0, eid, n);
    for (int j = 0; j < dd; ++j) {
        // etmpd/temp_deltas uses column-major storage: col * n + row.
        const double delta =
            temp_deltas[col_offset + static_cast<std::size_t>(j)];
        if (local == j) {
            val *= delta;
        } else {
            const double denom = D[start + local] - D[start + j];
            val *= denom != 0.0 ? delta / denom : 1.0;
        }
    }
    val = sqrt(device_abs(val));
    cz[eid] = cz[eid] < 0.0 ? -val : val;
}

template <int BlockThreads>
__global__ void stedc_merge_rescale_cub_kernel(const int n, const double *D,
                                               double *tmpz, int *splits,
                                               double *temp_deltas) {
    const int eid = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    if (eid >= n) {
        return;
    }

    int *nps = ptr_nps(n, splits);
    int *ndd = ptr_ndd(n, splits);
    double *cz = ptr_cz(n, tmpz);

    const int start = nps[eid];
    const int dd = ndd[eid];
    const int local = eid - start;
    if (local < 0 || local >= dd) {
        return;
    }

    double partial = 1.0;
    const std::size_t col_offset = dense_index(0, eid, n);
    for (int j = tid; j < dd; j += BlockThreads) {
        const double delta =
            temp_deltas[col_offset + static_cast<std::size_t>(j)];
        if (local == j) {
            partial *= delta;
        } else {
            const double denom = D[start + local] - D[start + j];
            partial *= denom != 0.0 ? delta / denom : 1.0;
        }
    }

    using BlockReduce = cub::BlockReduce<double, BlockThreads>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    const double reduced =
        BlockReduce(temp_storage).Reduce(partial, DeviceMultiply{});
    if (tid == 0) {
        const double val = sqrt(device_abs(reduced));
        cz[eid] = cz[eid] < 0.0 ? -val : val;
    }
}

template <int BlockThreads>
__global__ void stedc_merge_rescale_chunk_cub_kernel(
    const int n, const int chunk_start, const int chunk_size,
    const int *active_indices, const double *D, double *tmpz, int *splits,
    double *temp_deltas) {
    const int chunk_col = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    if (chunk_col >= chunk_size) {
        return;
    }
    const int eid = active_indices != nullptr ? active_indices[chunk_start + chunk_col]
                                              : chunk_start + chunk_col;
    if (eid >= n) {
        return;
    }

    int *nps = ptr_nps(n, splits);
    int *ndd = ptr_ndd(n, splits);
    double *cz = ptr_cz(n, tmpz);

    const int start = nps[eid];
    const int dd = ndd[eid];
    const int local = eid - start;
    if (local < 0 || local >= dd) {
        return;
    }

    double partial = 1.0;
    const std::size_t col_offset =
        static_cast<std::size_t>(chunk_col) * static_cast<std::size_t>(n);
    for (int j = tid; j < dd; j += BlockThreads) {
        const double delta =
            temp_deltas[col_offset + static_cast<std::size_t>(j)];
        if (local == j) {
            partial *= delta;
        } else {
            const double denom = D[start + local] - D[start + j];
            partial *= denom != 0.0 ? delta / denom : 1.0;
        }
    }

    using BlockReduce = cub::BlockReduce<double, BlockThreads>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    const double reduced =
        BlockReduce(temp_storage).Reduce(partial, DeviceMultiply{});
    if (tid == 0) {
        const double val = sqrt(device_abs(reduced));
        cz[eid] = cz[eid] < 0.0 ? -val : val;
    }
}

// Values-only variant of stedc_merge_prepare_deflatezero_kernel.
// Constructs z from boundary rows instead of the dense V matrix.
// Boundary rows must have been initialized at the leaf stage and
// updated after each previous merge level.
template <int BlockThreads>
__global__ void stedc_merge_prepare_deflatezero_boundary_kernel(
    const int merge_count, const int n, const double *D, const double *E,
    const double *boundary_rows, double *tmpz, int *splits, const double eps) {
    const int merge = static_cast<int>(blockIdx.x);
    if (merge >= merge_count) {
        return;
    }
    const int tid = static_cast<int>(threadIdx.x);

    int *bsz = ptr_bsz(n, splits);
    int *bps = ptr_bps(n, splits);
    int *mask = ptr_mask(n, splits);
    double *z = ptr_z(n, tmpz);
    double *r1p = ptr_r1p(n, tmpz);
    double *tolsD = ptr_tolsD(n, tmpz);
    double *tolsZ = ptr_tolsZ(n, tmpz);

    const double *first_rows = ptr_first_rows(n, boundary_rows);
    const double *last_rows = ptr_last_rows(n, boundary_rows);

    const int size_left = bsz[2 * merge];
    const int size_right = bsz[2 * merge + 1];
    const int start = bps[2 * merge];
    const int split = bps[2 * merge + 1];
    const int size = size_left + size_right;
    // Contiguous merge layout: left z occupies [start, split), right z [split, start+size).
    // split == start + size_left for every merge in the STEDC split tree.

    const double rho = 2.0 * E[split - 1];

    double local_maxz = 0.0;
    double local_maxd = 0.0;
    for (int k = tid; k < size; k += BlockThreads) {
        const double zval = (k < size_left)
                                ? last_rows[start + k] / sqrt(2.0)
                                : first_rows[split + (k - size_left)] / sqrt(2.0);
        z[start + k] = zval;
        local_maxz = local_maxz > device_abs(zval) ? local_maxz : device_abs(zval);
        local_maxd = local_maxd > device_abs(D[start + k])
                         ? local_maxd
                         : device_abs(D[start + k]);
    }

    using BlockReduce = cub::BlockReduce<double, BlockThreads>;
    __shared__ typename BlockReduce::TempStorage tmp_z;
    __shared__ typename BlockReduce::TempStorage tmp_d;
    __shared__ double s_tol;
    const double maxz = BlockReduce(tmp_z).Reduce(local_maxz, DeviceMax{});
    const double maxd = BlockReduce(tmp_d).Reduce(local_maxd, DeviceMax{});
    if (tid == 0) {
        s_tol = EVD_TRIDIAG_DEFLATION_TOL_SCALE * eps *
                (maxd > maxz ? maxd : maxz);
    }
    __syncthreads();
    const double tol = s_tol;

    for (int k = tid; k < size; k += BlockThreads) {
        r1p[start + k] = rho;
        tolsD[start + k] = tol;
        tolsZ[start + k] = tol;
        mask[start + k] = device_abs(rho * z[start + k]) > tol ? 1 : 0;
    }
}

__global__ void stedc_update_boundary_rows_kernel(const int n,
                                                  const double *etmpd_colmajor,
                                                  int *splits,
                                                  double *boundary_rows,
                                                  double *scratch) {
    const int eid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (eid >= n) {
        return;
    }

    int *nps = ptr_nps(n, splits);
    int *ndd = ptr_ndd(n, splits);
    int *map = ptr_map(n, splits);

    const int start = nps[eid];
    const int dd = ndd[eid];
    const int local = eid - start;

    const double *old_first = ptr_first_rows(n, boundary_rows);
    const double *old_last = ptr_last_rows(n, boundary_rows);
    double *new_first = ptr_first_rows(n, scratch);
    double *new_last = ptr_last_rows(n, scratch);

    double acc_first = 0.0;
    double acc_last = 0.0;

    if (local < dd) {
        for (int i = 0; i < dd; ++i) {
            const int source_col = map[start + i];
            const double et = etmpd_colmajor[dense_index(i, eid, n)];
            acc_first += old_first[source_col] * et;
            acc_last += old_last[source_col] * et;
        }
    } else {
        const int source_col = map[eid];
        acc_first = old_first[source_col];
        acc_last = old_last[source_col];
    }

    new_first[eid] = acc_first;
    new_last[eid] = acc_last;
}

template <int BlockThreads>
__global__ void
stedc_update_boundary_rows_cub_kernel(const int n,
                                      const double *etmpd_colmajor, int *splits,
                                      double *boundary_rows, double *scratch) {
    const int eid = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    if (eid >= n) {
        return;
    }

    int *nps = ptr_nps(n, splits);
    int *ndd = ptr_ndd(n, splits);
    int *map = ptr_map(n, splits);

    const int start = nps[eid];
    const int dd = ndd[eid];
    const int local = eid - start;

    const double *old_first = ptr_first_rows(n, boundary_rows);
    const double *old_last = ptr_last_rows(n, boundary_rows);
    double *new_first = ptr_first_rows(n, scratch);
    double *new_last = ptr_last_rows(n, scratch);

    if (local >= 0 && local < dd) {
        double acc_first = 0.0;
        double acc_last = 0.0;
        for (int i = tid; i < dd; i += BlockThreads) {
            const int source_col = map[start + i];
            const double et = etmpd_colmajor[dense_index(i, eid, n)];
            acc_first += old_first[source_col] * et;
            acc_last += old_last[source_col] * et;
        }

        using BlockReduce = cub::BlockReduce<double, BlockThreads>;
        __shared__ typename BlockReduce::TempStorage first_storage;
        __shared__ typename BlockReduce::TempStorage last_storage;
        const double reduced_first = BlockReduce(first_storage).Sum(acc_first);
        const double reduced_last = BlockReduce(last_storage).Sum(acc_last);
        if (tid == 0) {
            new_first[eid] = reduced_first;
            new_last[eid] = reduced_last;
        }
    } else if (tid == 0) {
        const int source_col = map[eid];
        new_first[eid] = old_first[source_col];
        new_last[eid] = old_last[source_col];
    }
}

template <int BlockThreads>
__global__ void stedc_merge_vectors_prepare_chunk_kernel(
    const int n, const int chunk_start, const int chunk_size, double *tmpz,
    double *etmpd, int *splits, const int *active_indices) {
    const int chunk_col = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int dim = static_cast<int>(blockDim.x);
    if (chunk_col >= chunk_size) {
        return;
    }
    const int eid = active_indices != nullptr ? active_indices[chunk_start + chunk_col]
                                              : chunk_start + chunk_col;
    if (eid >= n) {
        return;
    }

    int *nps = ptr_nps(n, splits);
    int *ndd = ptr_ndd(n, splits);
    double *z = ptr_z(n, tmpz);

    const int start = nps[eid];
    const int dd = ndd[eid];
    if (eid - start >= dd) {
        return;
    }
    const std::size_t col_offset =
        static_cast<std::size_t>(chunk_col) * static_cast<std::size_t>(n);

    __shared__ double scratch[BlockThreads];
    double nrm = 0.0;
    for (int i = tid; i < dd; i += dim) {
        const double denom = etmpd[col_offset + static_cast<std::size_t>(i)];
        const double value = denom != 0.0 ? z[start + i] / denom : 0.0;
        etmpd[col_offset + static_cast<std::size_t>(i)] = value;
        nrm += value * value;
    }
    scratch[tid] = nrm;
    for (int r = dim / 2; r > 0; r /= 2) {
        __syncthreads();
        if (tid < r) {
            scratch[tid] += scratch[tid + r];
        }
    }
    __syncthreads();
    nrm = sqrt(scratch[0]);
    __syncthreads();

    for (int i = tid; i < dd; i += dim) {
        const std::size_t offset = col_offset + static_cast<std::size_t>(i);
        etmpd[offset] = nrm != 0.0 ? etmpd[offset] / nrm : 0.0;
    }
}

template <int BlockThreads>
__global__ void stedc_update_boundary_rows_chunk_cub_kernel(
    const int n, const int chunk_start, const int chunk_size,
    const int *active_indices, const double *etmpd_colmajor, int *splits,
    double *boundary_rows, double *scratch) {
    const int chunk_col = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    if (chunk_col >= chunk_size) {
        return;
    }
    const int eid = active_indices != nullptr ? active_indices[chunk_start + chunk_col]
                                              : chunk_start + chunk_col;
    if (eid >= n) {
        return;
    }

    int *nps = ptr_nps(n, splits);
    int *ndd = ptr_ndd(n, splits);
    int *map = ptr_map(n, splits);

    const int start = nps[eid];
    const int dd = ndd[eid];
    const int local = eid - start;

    const double *old_first = ptr_first_rows(n, boundary_rows);
    const double *old_last = ptr_last_rows(n, boundary_rows);
    double *new_first = ptr_first_rows(n, scratch);
    double *new_last = ptr_last_rows(n, scratch);

    if (local >= 0 && local < dd) {
        double acc_first = 0.0;
        double acc_last = 0.0;
        const std::size_t col_offset =
            static_cast<std::size_t>(chunk_col) * static_cast<std::size_t>(n);
        for (int i = tid; i < dd; i += BlockThreads) {
            const int source_col = map[start + i];
            const double et =
                etmpd_colmajor[col_offset + static_cast<std::size_t>(i)];
            acc_first += old_first[source_col] * et;
            acc_last += old_last[source_col] * et;
        }

        using BlockReduce = cub::BlockReduce<double, BlockThreads>;
        __shared__ typename BlockReduce::TempStorage first_storage;
        __shared__ typename BlockReduce::TempStorage last_storage;
        const double reduced_first = BlockReduce(first_storage).Sum(acc_first);
        const double reduced_last = BlockReduce(last_storage).Sum(acc_last);
        if (tid == 0) {
            new_first[eid] = reduced_first;
            new_last[eid] = reduced_last;
        }
    } else if (tid == 0) {
        const int source_col = map[eid];
        new_first[eid] = old_first[source_col];
        new_last[eid] = old_last[source_col];
    }
}

__global__ void stedc_update_deflated_boundary_rows_kernel(
    const int n, int *splits, double *boundary_rows, double *scratch) {
    const int eid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (eid >= n) {
        return;
    }

    int *nps = ptr_nps(n, splits);
    int *ndd = ptr_ndd(n, splits);
    int *em = ptr_em(n, splits);
    int *map = ptr_map(n, splits);

    const int start = nps[eid];
    const int dd = ndd[eid];
    const int local = eid - start;
    if (local >= 0 && local < dd) {
        return;
    }

    const int merge = em[eid];
    if (merge < 0) {
        return;
    }
    const int source_col = map[eid];

    const double *old_first = ptr_first_rows(n, boundary_rows);
    const double *old_last = ptr_last_rows(n, boundary_rows);
    double *new_first = ptr_first_rows(n, scratch);
    double *new_last = ptr_last_rows(n, scratch);
    new_first[eid] = old_first[source_col];
    new_last[eid] = old_last[source_col];
}

__global__ void stedc_copy_boundary_rows_kernel(const int n, const double *src,
                                                double *dst) {
    const int gid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = 2 * n;
    if (gid >= total) {
        return;
    }
    dst[gid] = src[gid];
}

} // namespace

} // namespace evd::internal

#endif
