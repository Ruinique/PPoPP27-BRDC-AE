#ifndef EVD_LIBRARY_INTERNAL_SYEV2STAGE_PIPELINE_HPP
#define EVD_LIBRARY_INTERNAL_SYEV2STAGE_PIPELINE_HPP

#include <cstddef>
#include <stdint.h>

#include "evd/handle.h"
#include "evd/status.h"
#include "evd/types.h"
#include "sbr_history_layout.hpp"
#include "workspace_align.hpp"

namespace evd::internal {

struct Syev2StageArtifacts {
    double *d_bulge_reflectors{nullptr};
    int64_t ld_bulge_reflectors{0};
    double *d_tridiag_eigenvectors{nullptr};
    int64_t ld_tridiag_eigenvectors{0};
    double *d_tridiag_signs{nullptr};
    double *d_sbr_w{nullptr};
    double *d_sbr_y{nullptr};
    SbrHistoryLayout sbr_history_layout{};
};

struct Syev2StageState {
    const evd_syev2stage_config *config;
    double *d_A;
    int64_t lda;
    double *d_W;
    double *d_Q;
    Syev2StageArtifacts *artifacts{nullptr};
};

struct StageWorkspaceSlot {
    size_t offset;
    size_t bytes;
};

struct Syev2StageWorkspaceLayout {
    StageWorkspaceSlot reduction_to_band;
    StageWorkspaceSlot tridiag_signs;
    StageWorkspaceSlot sbr_history;
    StageWorkspaceSlot bulge_chasing;
    StageWorkspaceSlot tridiag_solver;
    StageWorkspaceSlot back_transform;
    size_t total;
};

size_t align_up(size_t value, size_t alignment = kDeviceWorkspaceAlignment);

evd_status
compute_syev2stage_workspace_layout(const evd_syev2stage_config &config,
                                    Syev2StageWorkspaceLayout *layout);

evd_status run_syev2stage_pipeline(evd_handle handle,
                                   const Syev2StageState &state);

evd_status reduction_to_band_workspace_size(const evd_syev2stage_config &config,
                                            size_t *bytes);
evd_status reduction_to_band_run(evd_handle handle,
                                 const Syev2StageState &state, void *workspace,
                                 size_t workspace_size);

evd_status bulge_chasing_workspace_size(const evd_syev2stage_config &config,
                                        size_t *bytes);
evd_status bulge_chasing_run(evd_handle handle, const Syev2StageState &state,
                             void *workspace, size_t workspace_size);

evd_status tridiag_solver_workspace_size(const evd_syev2stage_config &config,
                                         size_t *bytes);
evd_status tridiag_solver_run(evd_handle handle, const Syev2StageState &state,
                              void *workspace, size_t workspace_size);
evd_status tridiag_solver_run_from_device_tridiagonal(
    evd_handle handle, const evd_syev2stage_config &config,
    const double *d_diag, const double *d_subdiag, double *d_W,
    void *workspace, size_t workspace_size);

evd_status back_transform_workspace_size(const evd_syev2stage_config &config,
                                         size_t *bytes);
evd_status back_transform_run(evd_handle handle, const Syev2StageState &state,
                              void *workspace, size_t workspace_size);

} // namespace evd::internal

#endif // EVD_LIBRARY_INTERNAL_SYEV2STAGE_PIPELINE_HPP
