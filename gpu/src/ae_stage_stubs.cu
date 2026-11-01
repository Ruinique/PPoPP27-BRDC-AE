// AE trim: the paper's experiments start from tridiagonal input (Sec. 5.6:
// "both paths starting from tridiagonal matrices ... excluding dense reduction
// and back transformation"). The dense-reduction and back-transform stages of
// the full SyncFlowEVD library are therefore not shipped; these stubs keep the
// unchanged paper sources linking and report EVD_STATUS_NOT_IMPLEMENTED if the
// dense-input (--input=<dense .bin>) or eigenvector paths are requested.
#include "back_transform.hpp"
#include "syev2stage_pipeline.hpp"
namespace evd { namespace internal {
evd_status reduction_to_band_workspace_size(const evd_syev2stage_config &, size_t *b) { if (b) *b = 0; return EVD_STATUS_NOT_IMPLEMENTED; }
evd_status reduction_to_band_run(evd_handle, const Syev2StageState &, void *, size_t) { return EVD_STATUS_NOT_IMPLEMENTED; }
evd_status bulge_chasing_workspace_size(const evd_syev2stage_config &, size_t *b) { if (b) *b = 0; return EVD_STATUS_NOT_IMPLEMENTED; }
evd_status bulge_chasing_run(evd_handle, const Syev2StageState &, void *, size_t) { return EVD_STATUS_NOT_IMPLEMENTED; }
evd_status back_transform_workspace_size(const evd_syev2stage_config &, size_t *b) { if (b) *b = 0; return EVD_STATUS_NOT_IMPLEMENTED; }
evd_status back_transform_run(evd_handle, const Syev2StageState &, void *, size_t) { return EVD_STATUS_NOT_IMPLEMENTED; }
bool back_transform_band_size_supported(int64_t) { return false; }
evd_status make_tridiag_signs_from_subdiag(evd_handle, int64_t, const double *, double *) { return EVD_STATUS_NOT_IMPLEMENTED; }
}}
