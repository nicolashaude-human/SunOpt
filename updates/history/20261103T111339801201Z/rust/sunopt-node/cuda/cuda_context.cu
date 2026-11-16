#include "cuda_context.h"

#include <cuda_runtime.h>

struct SunoptCudaContext {
    int device_id;
    cudaStream_t stream;
};

extern "C" int sunopt_cuda_create(
    int device_id,
    SunoptCudaContext** out_context
) {
    if (out_context == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    *out_context = nullptr;

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    SunoptCudaContext* context = new SunoptCudaContext{};
    context->device_id = device_id;
    context->stream = nullptr;

    err = cudaStreamCreateWithFlags(
        &context->stream,
        cudaStreamNonBlocking
    );

    if (err != cudaSuccess) {
        delete context;
        return static_cast<int>(err);
    }

    *out_context = context;
    return static_cast<int>(cudaSuccess);
}

extern "C" int sunopt_cuda_destroy(
    SunoptCudaContext* context
) {
    if (context == nullptr) {
        return static_cast<int>(cudaSuccess);
    }

    cudaError_t err = cudaSetDevice(context->device_id);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    err = cudaStreamDestroy(context->stream);
    delete context;

    return static_cast<int>(err);
}

extern "C" int sunopt_cuda_device_id(
    const SunoptCudaContext* context,
    int* out_device_id
) {
    if (context == nullptr || out_device_id == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    *out_device_id = context->device_id;
    return static_cast<int>(cudaSuccess);
}

extern "C" int sunopt_cuda_stream_synchronize(
    SunoptCudaContext* context
) {
    if (context == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    cudaError_t err = cudaSetDevice(context->device_id);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    return static_cast<int>(cudaStreamSynchronize(context->stream));
}

extern "C" const char* sunopt_cuda_error_string(
    int error_code
) {
    return cudaGetErrorString(static_cast<cudaError_t>(error_code));
}
