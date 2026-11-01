#ifndef EVD_LIBRARY_INTERNAL_TRIDIAG_SOLVER_EXTRACT_CUH
#define EVD_LIBRARY_INTERNAL_TRIDIAG_SOLVER_EXTRACT_CUH

#include <cstdint>
#include "tridiag_solver_device.cuh"

namespace evd::internal {

namespace {


__global__ void extract_lower_tridiagonal_kernel(const std::int64_t n,
                                                 const double *A,
                                                 const std::int64_t lda,
                                                 double *D, double *E,
                                                 double *Dc, double *Ec) {
    const std::int64_t idx =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx < n) {
        const double value = A[idx + idx * lda];
        D[idx] = value;
        if (Dc != nullptr) {
            Dc[idx] = value;
        }
    }

    if (idx + 1 < n) {
        const double value = device_abs(A[idx + 1 + idx * lda]);
        E[idx] = value;
        if (Ec != nullptr) {
            Ec[idx] = value;
        }
    }
}

__global__ void init_stedc_splits_kernel(const int levs, const int blks,
                                         const int n, int *splits) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    int *msz = ptr_msz(n, splits);
    int *mps = ptr_mps(n, splits);
    int *bsz = ptr_bsz(n, splits);
    int *bps = ptr_bps(n, splits);
    int *em = ptr_em(n, splits);
    int *nsz = ptr_nsz(n, splits);
    int *nps = ptr_nps(n, splits);
    int *ndd = ptr_ndd(n, splits);
    int *mask = ptr_mask(n, splits);
    int *dcount = ptr_dcount(n, splits);
    int *map = ptr_map(n, splits);
    int *cand = ptr_cand(n, splits);
    int *dbg = ptr_dbg(n, splits);

    for (int i = 0; i < n; ++i) {
        msz[i] = 0;
        mps[i] = 0;
        bsz[i] = 0;
        bps[i] = 0;
        em[i] = 0;
        nsz[i] = 0;
        nps[i] = 0;
        ndd[i] = 0;
        mask[i] = 1;
        dcount[i] = 0;
        map[i] = i;
        cand[i] = 0;
        dbg[i] = 0;
    }

    msz[0] = n;
    for (int level = 0; level < levs; ++level) {
        for (int node = (1 << level); node > 0; --node) {
            const int parent = node - 1;
            const int size = msz[parent];
            msz[node * 2 - 1] = size / 2 + (size & 1);
            msz[node * 2 - 2] = size / 2;
        }
    }

    int position = 0;
    mps[0] = 0;
    for (int block = 1; block < blks; ++block) {
        position += msz[block - 1];
        mps[block] = position;
    }

    for (int block = 0; block < blks; ++block) {
        const int size = msz[block];
        const int start = mps[block];
        bsz[block] = size;
        bps[block] = start;
        for (int i = 0; i < size; ++i) {
            em[start + i] = block;
            nsz[start + i] = size;
            nps[start + i] = start;
            map[start + i] = start + i;
        }
    }
}

__global__ void stedc_divide_kernel(const int blks, const int n, double *D,
                                    const double *E, int *splits) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    int *mps = ptr_mps(n, splits);
    for (int block = 1; block < blks; ++block) {
        const int split = mps[block];
        const double coupling = E[split - 1];
        D[split - 1] -= coupling;
        D[split] -= coupling;
    }
}

} // namespace

} // namespace evd::internal

#endif
