#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// ============================================================================
// CUDA error handling
// ============================================================================

#define CUDA_CHECK(call)                                                     \
    do                                                                       \
    {                                                                        \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess)                                              \
        {                                                                    \
            std::fprintf(                                                    \
                stderr,                                                      \
                "CUDA error at %s:%d: %s\n",                                 \
                __FILE__,                                                    \
                __LINE__,                                                    \
                cudaGetErrorString(err));                                    \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)


// ============================================================================
// SunOpt-like state
//
// This is intentionally simple:
// - storage thermal energy
// - PV energy available
// - thermal demand
// - standing loss
//
// Each element is independent and can therefore be processed massively
// in parallel.
// ============================================================================

struct State
{
    float storageEnergy;
    float pvEnergy;
    float demandEnergy;
    float storageLoss;
};


// ============================================================================
// Calculation shared conceptually by CPU and GPU
// ============================================================================

__host__ __device__
inline float simulateState(
    float storageEnergy,
    float pvEnergy,
    float demandEnergy,
    float storageLoss)
{
    // Standing/storage loss
    float energy = storageEnergy - storageLoss;

    if (energy < 0.0f)
    {
        energy = 0.0f;
    }

    // PV charges the storage
    energy += pvEnergy;

    // Demand drains the storage
    energy -= demandEnergy;

    // Negative storage means grid import.
    float gridImport = 0.0f;

    if (energy < 0.0f)
    {
        gridImport = -energy;
        energy = 0.0f;
    }

    // Some extra numerical work to make this less trivial than
    // a single addition/subtraction benchmark.
    //
    // This loosely mimics the kind of repeated thermodynamic arithmetic
    // SunOpt performs.
    float temperatureFactor =
        0.85f +
        0.15f * std::sqrt(energy + 1.0f);

    float thermalCorrection =
        std::log1p(temperatureFactor) *
        0.015f;

    gridImport += thermalCorrection;

    return gridImport;
}


// ============================================================================
// CPU implementation
// ============================================================================

void runCpu(
    const State* states,
    float* output,
    std::size_t count)
{
    for (std::size_t i = 0; i < count; ++i)
    {
        const State& s = states[i];

        output[i] = simulateState(
            s.storageEnergy,
            s.pvEnergy,
            s.demandEnergy,
            s.storageLoss);
    }
}


// ============================================================================
// CUDA kernel
// ============================================================================

__global__
void runGpuKernel(
    const State* states,
    float* output,
    std::size_t count)
{
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) *
        blockDim.x +
        threadIdx.x;

    if (i >= count)
    {
        return;
    }

    const State s = states[i];

    output[i] = simulateState(
        s.storageEnergy,
        s.pvEnergy,
        s.demandEnergy,
        s.storageLoss);
}


// ============================================================================
// Benchmark
// ============================================================================

void benchmark(std::size_t count)
{
    std::printf("\n");
    std::printf("============================================================\n");
    std::printf("Cases: %zu\n", count);
    std::printf("============================================================\n");

    // ------------------------------------------------------------------------
    // Host data
    // ------------------------------------------------------------------------

    std::vector<State> states(count);
    std::vector<float> cpuOutput(count);
    std::vector<float> gpuOutput(count);

    for (std::size_t i = 0; i < count; ++i)
    {
        // Deterministic data rather than random input.
        //
        // Enough variation to prevent every thread doing exactly
        // the same arithmetic.

        states[i].storageEnergy =
            5.0f +
            static_cast<float>(i % 500) * 0.05f;

        states[i].pvEnergy =
            static_cast<float>(i % 200) * 0.02f;

        states[i].demandEnergy =
            1.0f +
            static_cast<float>(i % 100) * 0.025f;

        states[i].storageLoss =
            0.05f +
            static_cast<float>(i % 20) * 0.002f;
    }


    // ------------------------------------------------------------------------
    // CPU benchmark
    // ------------------------------------------------------------------------

    auto cpuStart = std::chrono::high_resolution_clock::now();

    runCpu(
        states.data(),
        cpuOutput.data(),
        count);

    auto cpuEnd = std::chrono::high_resolution_clock::now();

    const double cpuMs =
        std::chrono::duration<double, std::milli>(
            cpuEnd - cpuStart)
        .count();


    // ------------------------------------------------------------------------
    // Allocate GPU memory
    // ------------------------------------------------------------------------

    State* deviceStates = nullptr;
    float* deviceOutput = nullptr;

    const std::size_t stateBytes =
        count * sizeof(State);

    const std::size_t outputBytes =
        count * sizeof(float);

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(&deviceStates),
            stateBytes));

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(&deviceOutput),
            outputBytes));


    // ------------------------------------------------------------------------
    // GPU total timer
    //
    // Includes:
    // host -> device
    // kernel
    // device -> host
    // ------------------------------------------------------------------------

    auto gpuTotalStart =
        std::chrono::high_resolution_clock::now();


    CUDA_CHECK(
        cudaMemcpy(
            deviceStates,
            states.data(),
            stateBytes,
            cudaMemcpyHostToDevice));


    // ------------------------------------------------------------------------
    // Kernel timing using CUDA events
    // ------------------------------------------------------------------------

    cudaEvent_t kernelStart;
    cudaEvent_t kernelEnd;

    CUDA_CHECK(cudaEventCreate(&kernelStart));
    CUDA_CHECK(cudaEventCreate(&kernelEnd));


    constexpr int threadsPerBlock = 256;

    const int blocks =
        static_cast<int>(
            (count + threadsPerBlock - 1) /
            threadsPerBlock);


    CUDA_CHECK(
        cudaEventRecord(kernelStart));


    runGpuKernel << <blocks, threadsPerBlock >> > (
        deviceStates,
        deviceOutput,
        count);


    CUDA_CHECK(
        cudaGetLastError());


    CUDA_CHECK(
        cudaEventRecord(kernelEnd));

    CUDA_CHECK(
        cudaEventSynchronize(kernelEnd));


    float gpuKernelMs = 0.0f;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &gpuKernelMs,
            kernelStart,
            kernelEnd));


    // ------------------------------------------------------------------------
    // Copy result back
    // ------------------------------------------------------------------------

    CUDA_CHECK(
        cudaMemcpy(
            gpuOutput.data(),
            deviceOutput,
            outputBytes,
            cudaMemcpyDeviceToHost));


    auto gpuTotalEnd =
        std::chrono::high_resolution_clock::now();


    const double gpuTotalMs =
        std::chrono::duration<double, std::milli>(
            gpuTotalEnd - gpuTotalStart)
        .count();


    // ------------------------------------------------------------------------
    // Verify CPU/GPU parity
    // ------------------------------------------------------------------------

    double maxDifference = 0.0;

    for (std::size_t i = 0; i < count; ++i)
    {
        const double difference =
            std::fabs(
                static_cast<double>(cpuOutput[i]) -
                static_cast<double>(gpuOutput[i]));

        if (difference > maxDifference)
        {
            maxDifference = difference;
        }
    }


    // ------------------------------------------------------------------------
    // Results
    // ------------------------------------------------------------------------

    const double kernelSpeedup =
        cpuMs /
        static_cast<double>(gpuKernelMs);

    const double totalSpeedup =
        cpuMs /
        gpuTotalMs;


    std::printf(
        "CPU time:       %10.3f ms\n",
        cpuMs);

    std::printf(
        "GPU kernel:     %10.3f ms\n",
        gpuKernelMs);

    std::printf(
        "GPU total:      %10.3f ms\n",
        gpuTotalMs);

    std::printf(
        "Kernel speedup: %10.2fx\n",
        kernelSpeedup);

    std::printf(
        "Total speedup:  %10.2fx\n",
        totalSpeedup);

    std::printf(
        "Max difference: %.9f\n",
        maxDifference);


    // ------------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------------

    CUDA_CHECK(cudaEventDestroy(kernelStart));
    CUDA_CHECK(cudaEventDestroy(kernelEnd));

    CUDA_CHECK(cudaFree(deviceStates));
    CUDA_CHECK(cudaFree(deviceOutput));
}


// ============================================================================
// main
// ============================================================================

int main()
{
    int deviceCount = 0;

    CUDA_CHECK(
        cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        std::printf("No CUDA device found.\n");
        return EXIT_FAILURE;
    }


    cudaDeviceProp prop{};

    CUDA_CHECK(
        cudaGetDeviceProperties(
            &prop,
            0));


    std::printf("CUDA SunOpt benchmark\n");
    std::printf("---------------------\n");

    std::printf(
        "GPU: %s\n",
        prop.name);

    std::printf(
        "Compute capability: %d.%d\n",
        prop.major,
        prop.minor);

    std::printf(
        "SM count: %d\n",
        prop.multiProcessorCount);

    std::printf(
        "Global memory: %.2f GiB\n",
        static_cast<double>(prop.totalGlobalMem) /
        (1024.0 * 1024.0 * 1024.0));


    // Warm up the CUDA runtime before timing.
    CUDA_CHECK(cudaFree(nullptr));


    benchmark(10'000);
    benchmark(100'000);
    benchmark(1'000'000);
    benchmark(10'000'000);


    return EXIT_SUCCESS;
}