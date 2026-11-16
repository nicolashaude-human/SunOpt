#pragma once

#ifdef _WIN32
#define SUNOPT_CUDA_EXPORT __declspec(dllexport)
#else
#define SUNOPT_CUDA_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SunoptCudaContext SunoptCudaContext;

SUNOPT_CUDA_EXPORT int sunopt_cuda_create(
    int device_id,
    SunoptCudaContext** out_context
);

SUNOPT_CUDA_EXPORT int sunopt_cuda_destroy(
    SunoptCudaContext* context
);

SUNOPT_CUDA_EXPORT int sunopt_cuda_device_id(
    const SunoptCudaContext* context,
    int* out_device_id
);

SUNOPT_CUDA_EXPORT int sunopt_cuda_stream_synchronize(
    SunoptCudaContext* context
);

SUNOPT_CUDA_EXPORT const char* sunopt_cuda_error_string(
    int error_code
);

#ifdef __cplusplus
}
#endif
