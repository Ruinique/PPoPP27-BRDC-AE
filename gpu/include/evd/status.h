#ifndef EVD_STATUS_H
#define EVD_STATUS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum evd_status {
    EVD_STATUS_SUCCESS = 0,
    EVD_STATUS_INVALID_HANDLE,
    EVD_STATUS_INVALID_POINTER,
    EVD_STATUS_INVALID_VALUE,
    EVD_STATUS_INVALID_WORKSPACE,
    EVD_STATUS_ALLOC_FAILED,
    EVD_STATUS_CUDA_ERROR,
    EVD_STATUS_INTERNAL_ERROR,
    EVD_STATUS_NOT_IMPLEMENTED
} evd_status;

const char *evd_status_to_string(evd_status status);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // EVD_STATUS_H
