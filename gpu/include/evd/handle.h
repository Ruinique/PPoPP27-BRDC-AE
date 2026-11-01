#ifndef EVD_HANDLE_H
#define EVD_HANDLE_H

#include "evd/status.h"
#include "evd/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct evd_handle_struct *evd_handle;

evd_status evd_create_handle(evd_handle *handle);
evd_status evd_destroy_handle(evd_handle handle);
evd_status evd_get_last_error(evd_handle handle, int *code,
                              evd_error_source *source, const char **site);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // EVD_HANDLE_H
