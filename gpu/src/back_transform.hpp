#ifndef EVD_LIBRARY_INTERNAL_BACK_TRANSFORM_HPP
#define EVD_LIBRARY_INTERNAL_BACK_TRANSFORM_HPP

#include <stdint.h>

#include "evd/handle.h"
#include "evd/status.h"
#include "sbr_history_layout.hpp"

namespace evd::internal {

bool back_transform_band_size_supported(int64_t band_size);

evd_status make_tridiag_signs_from_subdiag(evd_handle handle, int64_t n,
                                           const double *subdiag,
                                           double *signs);

evd_status apply_bulge_reflectors_reverse(
    evd_handle handle, int64_t n, int64_t band_size, const double *reflectors,
    int64_t ld_reflectors, double *q, int64_t ldq);

evd_status apply_sbr_reflectors_reverse(
    evd_handle handle, int64_t n, int64_t band_size, double *sbr_w,
    const double *sbr_y, const SbrHistoryLayout &sbr_layout, double *q,
    int64_t ldq, double *apply_work, double *reduce_work);

} // namespace evd::internal

#endif // EVD_LIBRARY_INTERNAL_BACK_TRANSFORM_HPP
