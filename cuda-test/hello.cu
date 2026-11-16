// C:\NextJS\SunOpt\sunopt\cuda-test\hello.cu

#include <cstdio>
#include <cuda_runtime.h>

__global__ void helloKernel()
{
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

int main()
{
    cudaDeviceProp prop{};

    cudaError_t err = cudaGetDeviceProperties(&prop, 0);

    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("GPU: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);

    helloKernel << <1, 4 >> > ();

    err = cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
        printf("Kernel error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("CUDA test successful.\n");

    return 0;
}