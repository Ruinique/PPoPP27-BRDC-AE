#ifndef EVD_EVD_H
#define EVD_EVD_H

#include <stddef.h>

#include "evd/handle.h"
#include "evd/types.h"

#ifdef __cplusplus
extern "C" {
#endif

evd_status
evd_get_syev2stage_workspace_size(evd_handle handle,
                                  const evd_syev2stage_config *config,
                                  size_t *workspace_size_in_bytes);

evd_status evd_set_workspace(evd_handle handle, void *workspace,
                             size_t workspace_size_in_bytes);

evd_status evd_syev2stage(evd_handle handle,
                          const evd_syev2stage_config *config, double *d_A,
                          int64_t lda, double *d_W, double *d_Q);

evd_status evd_get_last_stage_timings(evd_handle handle,
                                      evd_stage_timing *timings, int max_count,
                                      int *actual_count);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // EVD_EVD_H
