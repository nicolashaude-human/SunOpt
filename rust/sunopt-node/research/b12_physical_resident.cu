// rust/sunopt-node/research/b12_physical_resident.cu
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr std::uint32_t kExpectedVersion = 1;
constexpr int F_START = 0;
constexpr int F_AFTER = 1;
constexpr int F_DIRECT = 2;
constexpr int F_HP_THERMAL = 17;
constexpr int F_LOSS = 18;
constexpr int F_END = 19;
constexpr int F_UNMET = 23;

constexpr double T_MIN = 60.0;
constexpr double T_MAX = 75.0;
constexpr double T_AMBIENT = 20.0;
constexpr double RHO = 1.0;
constexpr double CP = 4.186;
constexpr double KJ_PER_KWH = 3600.0;
constexpr double LAMBDA = 0.030;
constexpr double INSULATION = 0.10;
constexpr double HEIGHT = 1.70;
constexpr double PI = 3.141592653589793238462643383279502884;

struct Header {
    std::uint32_t version = 0;
    std::uint32_t steps = 0;
    std::uint32_t tanks = 0;
    std::uint32_t fields = 0;
    double dt = 0.0;
    std::vector<double> volumes;
    std::size_t data_offset = 0;
};

struct StepResult {
    double after_demand;
    double heat_loss;
    double unmet;
    double end;
};

struct ErrorStats {
    std::uint64_t mismatches = 0;
    double max_error = 0.0;
    std::size_t max_job = 0;
};

void cuda_check(cudaError_t err, const char* op) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string(op) + " failed: " +
            cudaGetErrorString(err) +
            " (CUDA error " +
            std::to_string(static_cast<int>(err)) + ")"
        );
    }
}

template <typename T>
T read_scalar(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    if (offset + sizeof(T) > bytes.size()) {
        throw std::runtime_error("Reference file ended while reading header.");
    }
    T value{};
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    return value;
}

std::vector<std::uint8_t> read_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("Unable to open " + path);
    }
    const auto size = input.tellg();
    if (size <= 0) {
        throw std::runtime_error("Reference file is empty.");
    }
    input.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    if (!input.read(reinterpret_cast<char*>(bytes.data()), size)) {
        throw std::runtime_error("Unable to read " + path);
    }
    return bytes;
}

Header parse_header(const std::vector<std::uint8_t>& bytes) {
    if (bytes.size() < 40) {
        throw std::runtime_error("Reference file is too small.");
    }

    const std::string magic(
        reinterpret_cast<const char*>(bytes.data()),
        15
    );

    if (magic != "SUNOPT_B12_PHY1") {
        throw std::runtime_error("Unexpected B12 reference magic: " + magic);
    }

    Header h;
    h.version = read_scalar<std::uint32_t>(bytes, 16);
    h.steps = read_scalar<std::uint32_t>(bytes, 20);
    h.tanks = read_scalar<std::uint32_t>(bytes, 24);
    h.fields = read_scalar<std::uint32_t>(bytes, 28);
    h.dt = read_scalar<double>(bytes, 32);

    if (
        h.version != kExpectedVersion ||
        h.steps == 0 ||
        h.tanks == 0 ||
        h.fields <= F_UNMET
    ) {
        throw std::runtime_error("B12 reference header is invalid.");
    }

    h.volumes.resize(h.tanks);

    for (std::uint32_t i = 0; i < h.tanks; ++i) {
        h.volumes[i] =
            read_scalar<double>(
                bytes,
                40 + static_cast<std::size_t>(i) * sizeof(double)
            );
    }

    h.data_offset =
        40 + static_cast<std::size_t>(h.tanks) * sizeof(double);

    const std::size_t payload =
        static_cast<std::size_t>(h.steps) *
        static_cast<std::size_t>(h.tanks) *
        static_cast<std::size_t>(h.fields) *
        sizeof(double);

    if (h.data_offset + payload != bytes.size()) {
        throw std::runtime_error("B12 reference size does not match header.");
    }

    return h;
}

__host__ __device__ inline double dmin(double a, double b) {
    return a < b ? a : b;
}

__host__ __device__ inline double dmax(double a, double b) {
    return a > b ? a : b;
}

__host__ __device__ inline double clampd(double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

__host__ __device__ inline double capacity(double volume) {
    return volume * RHO * CP * (T_MAX - T_MIN) / KJ_PER_KWH;
}

__host__ __device__ inline double energy_per_kelvin(double volume) {
    return volume * RHO * CP / KJ_PER_KWH;
}

__host__ __device__ inline double conductance(double volume) {
    const double volume_m3 = volume / 1000.0;
    const double r =
        sqrt(volume_m3 / (PI * HEIGHT));
    const double r_out =
        r + INSULATION;

    const double side =
        (2.0 * PI * LAMBDA * HEIGHT) /
        log(r_out / r);

    // Production source defines endArea as BOTH top and bottom.
    const double end_area =
        2.0 * PI * r * r;

    const double ends =
        LAMBDA * end_area / INSULATION;

    return side + ends;
}

__host__ __device__ inline StepResult storage_step(
    double volume,
    double g,
    double dt,
    double current,
    double input,
    double output
) {
    const double cap = capacity(volume);
    const double e_per_k = energy_per_kelvin(volume);

    double stored =
        clampd(current, 0.0, cap);

    const double temperature =
        clampd(
            T_MIN + stored / e_per_k,
            T_MIN,
            T_MAX
        );

    const double requested_loss =
        g *
        dmax(0.0, temperature - T_AMBIENT) *
        dt /
        1000.0;

    const double loss =
        dmin(stored, requested_loss);

    stored -= loss;

    const double discharged =
        dmin(output, stored);

    stored -= discharged;

    const double unmet =
        dmax(0.0, output - discharged);

    const double free_capacity =
        dmax(0.0, cap - stored);

    const double charged =
        dmin(input, free_capacity);

    const double after_demand =
        stored;

    stored += charged;

    StepResult r{};
    r.after_demand = clampd(after_demand, 0.0, cap);
    r.heat_loss = loss;
    r.unmet = unmet;
    r.end = clampd(stored, 0.0, cap);
    return r;
}

__host__ __device__ inline double field(
    const double* ref,
    std::size_t steps,
    std::size_t fields,
    std::size_t tank,
    std::size_t step,
    int field_index
) {
    const std::size_t job =
        tank * steps + step;

    return ref[
        job * fields +
        static_cast<std::size_t>(field_index)
    ];
}

__host__ __device__ inline double demand_from_reference(
    const double* ref,
    std::size_t steps,
    std::size_t fields,
    std::size_t tank,
    std::size_t step
) {
    const double start =
        field(ref, steps, fields, tank, step, F_START);

    const double loss =
        field(ref, steps, fields, tank, step, F_LOSS);

    const double after =
        field(ref, steps, fields, tank, step, F_AFTER);

    const double direct =
        field(ref, steps, fields, tank, step, F_DIRECT);

    return dmax(
        0.0,
        (start - loss - after) + direct
    );
}

__global__ void local_kernel(
    const double* ref,
    std::size_t steps,
    std::size_t tanks,
    std::size_t fields,
    double dt,
    const double* volumes,
    double* out_after,
    double* out_loss,
    double* out_unmet,
    double* out_end
) {
    const std::size_t job =
        static_cast<std::size_t>(blockIdx.x) *
        blockDim.x +
        threadIdx.x;

    const std::size_t jobs =
        steps * tanks;

    if (job >= jobs) {
        return;
    }

    const std::size_t tank =
        job / steps;

    const std::size_t step =
        job - tank * steps;

    const double volume =
        volumes[tank];

    const double start =
        field(ref, steps, fields, tank, step, F_START);

    const double hp =
        field(ref, steps, fields, tank, step, F_HP_THERMAL);

    const double demand =
        demand_from_reference(
            ref,
            steps,
            fields,
            tank,
            step
        );

    /*
     * Production semantics:
     *   preview loss + DHW,
     *   direct HP supply first,
     *   only residual HP heat charges storage.
     */
    const StepResult preview =
        storage_step(
            volume,
            conductance(volume),
            dt,
            start,
            0.0,
            demand
        );

    const double direct_supply =
        dmin(preview.unmet, hp);

    const double thermal_for_storage =
        dmax(0.0, hp - direct_supply);

    const double final_unmet =
        dmax(0.0, preview.unmet - direct_supply);

    const StepResult result =
        storage_step(
            volume,
            conductance(volume),
            dt,
            start,
            thermal_for_storage,
            demand
        );

    out_after[job] = result.after_demand;
    out_loss[job] = result.heat_loss;
    out_unmet[job] = final_unmet;
    out_end[job] = result.end;
}

__global__ void chain_kernel(
    const double* ref,
    std::size_t steps,
    std::size_t tanks,
    std::size_t fields,
    double dt,
    const double* volumes,
    double* out_start,
    double* out_after,
    double* out_loss,
    double* out_unmet,
    double* out_end
) {
    const std::size_t tank =
        static_cast<std::size_t>(threadIdx.x);

    if (tank >= tanks) {
        return;
    }

    const double volume =
        volumes[tank];

    const double g =
        conductance(volume);

    double stored =
        field(
            ref,
            steps,
            fields,
            tank,
            0,
            F_START
        );

    for (std::size_t step = 0; step < steps; ++step) {
        const std::size_t job =
            tank * steps + step;

        out_start[job] =
            stored;

        const double hp =
            field(
                ref,
                steps,
                fields,
                tank,
                step,
                F_HP_THERMAL
            );

        const double demand =
            demand_from_reference(
                ref,
                steps,
                fields,
                tank,
                step
            );

        const StepResult preview =
            storage_step(
                volume,
                g,
                dt,
                stored,
                0.0,
                demand
            );

        const double direct_supply =
            dmin(preview.unmet, hp);

        const double thermal_for_storage =
            dmax(0.0, hp - direct_supply);

        const double final_unmet =
            dmax(0.0, preview.unmet - direct_supply);

        const StepResult result =
            storage_step(
                volume,
                g,
                dt,
                stored,
                thermal_for_storage,
                demand
            );

        out_after[job] = result.after_demand;
        out_loss[job] = result.heat_loss;
        out_unmet[job] = final_unmet;
        out_end[job] = result.end;

        stored =
            result.end;
    }
}

ErrorStats compare_field(
    const std::vector<double>& computed,
    const double* ref,
    const Header& h,
    int ref_field,
    double tolerance
) {
    ErrorStats stats{};

    const std::size_t jobs =
        static_cast<std::size_t>(h.steps) *
        static_cast<std::size_t>(h.tanks);

    for (std::size_t job = 0; job < jobs; ++job) {
        const double expected =
            ref[
                job * h.fields +
                static_cast<std::size_t>(ref_field)
            ];

        const double error =
            std::abs(computed[job] - expected);

        if (error > stats.max_error) {
            stats.max_error = error;
            stats.max_job = job;
        }

        if (error > tolerance) {
            ++stats.mismatches;
        }
    }

    return stats;
}

std::uint64_t fnv1a(
    const std::vector<double>& values
) {
    std::uint64_t hash =
        1469598103934665603ULL;

    for (double value : values) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));

        for (int byte = 0; byte < 8; ++byte) {
            hash ^=
                (bits >> (byte * 8)) &
                0xffULL;

            hash *=
                1099511628211ULL;
        }
    }

    return hash;
}

void emit(
    const char* name,
    const ErrorStats& s
) {
    std::cout
        << "\"" << name << "Mismatches\":"
        << s.mismatches
        << ",\"" << name << "MaxAbsError\":"
        << std::setprecision(17)
        << s.max_error
        << ",\"" << name << "MaxErrorJob\":"
        << s.max_job;
}
}

int main(int argc, char** argv) {
    try {
        const std::string path =
            argc >= 2
                ? argv[1]
                : "blackwell-physical-reference.bin";

        const int device =
            argc >= 3
                ? std::stoi(argv[2])
                : 0;

        const double tolerance =
            argc >= 4
                ? std::stod(argv[3])
                : 1.0e-9;

        const auto bytes =
            read_file(path);

        const Header h =
            parse_header(bytes);

        const double* host_ref =
            reinterpret_cast<const double*>(
                bytes.data() + h.data_offset
            );

        const std::size_t jobs =
            static_cast<std::size_t>(h.steps) *
            static_cast<std::size_t>(h.tanks);

        const std::size_t ref_doubles =
            jobs *
            static_cast<std::size_t>(h.fields);

        const std::size_t ref_bytes =
            ref_doubles * sizeof(double);

        const std::size_t job_bytes =
            jobs * sizeof(double);

        cuda_check(
            cudaSetDevice(device),
            "cudaSetDevice"
        );

        double* d_ref = nullptr;
        double* d_volumes = nullptr;
        double* d_start = nullptr;
        double* d_after = nullptr;
        double* d_loss = nullptr;
        double* d_unmet = nullptr;
        double* d_end = nullptr;

        const auto h2d_start =
            std::chrono::steady_clock::now();

        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ref), ref_bytes), "cudaMalloc(ref)");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_volumes), h.volumes.size() * sizeof(double)), "cudaMalloc(volumes)");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_start), job_bytes), "cudaMalloc(start)");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_after), job_bytes), "cudaMalloc(after)");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_loss), job_bytes), "cudaMalloc(loss)");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_unmet), job_bytes), "cudaMalloc(unmet)");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_end), job_bytes), "cudaMalloc(end)");

        cuda_check(cudaMemcpy(d_ref, host_ref, ref_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(ref H2D)");
        cuda_check(cudaMemcpy(d_volumes, h.volumes.data(), h.volumes.size() * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(volumes H2D)");

        const auto h2d_stop =
            std::chrono::steady_clock::now();

        const double h2d_ms =
            std::chrono::duration<double, std::milli>(
                h2d_stop - h2d_start
            ).count();

        cudaEvent_t start_event = nullptr;
        cudaEvent_t stop_event = nullptr;

        cuda_check(cudaEventCreate(&start_event), "cudaEventCreate(start)");
        cuda_check(cudaEventCreate(&stop_event), "cudaEventCreate(stop)");

        const int threads = 256;
        const int blocks =
            static_cast<int>(
                (jobs + threads - 1) / threads
            );

        cuda_check(cudaEventRecord(start_event), "cudaEventRecord(local start)");

        local_kernel<<<blocks, threads>>>(
            d_ref,
            h.steps,
            h.tanks,
            h.fields,
            h.dt,
            d_volumes,
            d_after,
            d_loss,
            d_unmet,
            d_end
        );

        cuda_check(cudaGetLastError(), "local_kernel launch");
        cuda_check(cudaEventRecord(stop_event), "cudaEventRecord(local stop)");
        cuda_check(cudaEventSynchronize(stop_event), "cudaEventSynchronize(local)");

        float local_ms_f = 0.0f;
        cuda_check(cudaEventElapsedTime(&local_ms_f, start_event, stop_event), "cudaEventElapsedTime(local)");

        std::vector<double> local_after(jobs);
        std::vector<double> local_loss(jobs);
        std::vector<double> local_unmet(jobs);
        std::vector<double> local_end(jobs);

        cuda_check(cudaMemcpy(local_after.data(), d_after, job_bytes, cudaMemcpyDeviceToHost), "local after D2H");
        cuda_check(cudaMemcpy(local_loss.data(), d_loss, job_bytes, cudaMemcpyDeviceToHost), "local loss D2H");
        cuda_check(cudaMemcpy(local_unmet.data(), d_unmet, job_bytes, cudaMemcpyDeviceToHost), "local unmet D2H");
        cuda_check(cudaMemcpy(local_end.data(), d_end, job_bytes, cudaMemcpyDeviceToHost), "local end D2H");

        const ErrorStats local_after_stats =
            compare_field(local_after, host_ref, h, F_AFTER, tolerance);
        const ErrorStats local_loss_stats =
            compare_field(local_loss, host_ref, h, F_LOSS, tolerance);
        const ErrorStats local_unmet_stats =
            compare_field(local_unmet, host_ref, h, F_UNMET, tolerance);
        const ErrorStats local_end_stats =
            compare_field(local_end, host_ref, h, F_END, tolerance);

        cuda_check(cudaEventRecord(start_event), "cudaEventRecord(chain start)");

        chain_kernel<<<1, 32>>>(
            d_ref,
            h.steps,
            h.tanks,
            h.fields,
            h.dt,
            d_volumes,
            d_start,
            d_after,
            d_loss,
            d_unmet,
            d_end
        );

        cuda_check(cudaGetLastError(), "chain_kernel launch");
        cuda_check(cudaEventRecord(stop_event), "cudaEventRecord(chain stop)");
        cuda_check(cudaEventSynchronize(stop_event), "cudaEventSynchronize(chain)");

        float chain_ms_f = 0.0f;
        cuda_check(cudaEventElapsedTime(&chain_ms_f, start_event, stop_event), "cudaEventElapsedTime(chain)");

        const auto d2h_start =
            std::chrono::steady_clock::now();

        std::vector<double> chain_start(jobs);
        std::vector<double> chain_after(jobs);
        std::vector<double> chain_loss(jobs);
        std::vector<double> chain_unmet(jobs);
        std::vector<double> chain_end(jobs);

        cuda_check(cudaMemcpy(chain_start.data(), d_start, job_bytes, cudaMemcpyDeviceToHost), "chain start D2H");
        cuda_check(cudaMemcpy(chain_after.data(), d_after, job_bytes, cudaMemcpyDeviceToHost), "chain after D2H");
        cuda_check(cudaMemcpy(chain_loss.data(), d_loss, job_bytes, cudaMemcpyDeviceToHost), "chain loss D2H");
        cuda_check(cudaMemcpy(chain_unmet.data(), d_unmet, job_bytes, cudaMemcpyDeviceToHost), "chain unmet D2H");
        cuda_check(cudaMemcpy(chain_end.data(), d_end, job_bytes, cudaMemcpyDeviceToHost), "chain end D2H");

        const auto d2h_stop =
            std::chrono::steady_clock::now();

        const double d2h_ms =
            std::chrono::duration<double, std::milli>(
                d2h_stop - d2h_start
            ).count();

        const ErrorStats chain_start_stats =
            compare_field(chain_start, host_ref, h, F_START, tolerance);
        const ErrorStats chain_after_stats =
            compare_field(chain_after, host_ref, h, F_AFTER, tolerance);
        const ErrorStats chain_loss_stats =
            compare_field(chain_loss, host_ref, h, F_LOSS, tolerance);
        const ErrorStats chain_unmet_stats =
            compare_field(chain_unmet, host_ref, h, F_UNMET, tolerance);
        const ErrorStats chain_end_stats =
            compare_field(chain_end, host_ref, h, F_END, tolerance);

        const std::uint64_t checksum =
            fnv1a(chain_end);

        std::cout
            << "SUNOPT_B12_PHYSICAL_RESIDENT_JSON={"
            << "\"referencePath\":\"" << path << "\","
            << "\"deviceId\":" << device << ","
            << "\"tolerance\":" << std::setprecision(17) << tolerance << ","
            << "\"stepCount\":" << h.steps << ","
            << "\"tankCount\":" << h.tanks << ","
            << "\"fieldCount\":" << h.fields << ","
            << "\"causalJobs\":" << jobs << ","
            << "\"inputH2DMs\":" << h2d_ms << ","
            << "\"localKernelMs\":" << static_cast<double>(local_ms_f) << ","
            << "\"chainKernelMs\":" << static_cast<double>(chain_ms_f) << ","
            << "\"chainResultD2HMs\":" << d2h_ms << ",";

        emit("localAfterDemand", local_after_stats); std::cout << ",";
        emit("localLoss", local_loss_stats); std::cout << ",";
        emit("localUnmet", local_unmet_stats); std::cout << ",";
        emit("localEnd", local_end_stats); std::cout << ",";
        emit("chainStart", chain_start_stats); std::cout << ",";
        emit("chainAfterDemand", chain_after_stats); std::cout << ",";
        emit("chainLoss", chain_loss_stats); std::cout << ",";
        emit("chainUnmet", chain_unmet_stats); std::cout << ",";
        emit("chainEnd", chain_end_stats);

        std::cout
            << ",\"chainEndChecksumHex\":\""
            << std::hex
            << std::setw(16)
            << std::setfill('0')
            << checksum
            << std::dec
            << "\"}"
            << std::endl;

        std::cout << std::endl;
        std::cout << "--- B12 PHYSICAL RESIDENT PARITY ---" << std::endl;
        std::cout << "steps=" << h.steps << std::endl;
        std::cout << "tanks=" << h.tanks << std::endl;
        std::cout << "jobs=" << jobs << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "inputH2DMs=" << h2d_ms << std::endl;
        std::cout << "localKernelMs=" << static_cast<double>(local_ms_f) << std::endl;
        std::cout << "chainKernelMs=" << static_cast<double>(chain_ms_f) << std::endl;
        std::cout << "chainResultD2HMs=" << d2h_ms << std::endl;
        std::cout << std::scientific;
        std::cout << "localAfterDemandMismatches=" << local_after_stats.mismatches << " maxError=" << local_after_stats.max_error << std::endl;
        std::cout << "localLossMismatches=" << local_loss_stats.mismatches << " maxError=" << local_loss_stats.max_error << std::endl;
        std::cout << "localUnmetMismatches=" << local_unmet_stats.mismatches << " maxError=" << local_unmet_stats.max_error << std::endl;
        std::cout << "localEndMismatches=" << local_end_stats.mismatches << " maxError=" << local_end_stats.max_error << std::endl;
        std::cout << "chainStartMismatches=" << chain_start_stats.mismatches << " maxError=" << chain_start_stats.max_error << std::endl;
        std::cout << "chainAfterDemandMismatches=" << chain_after_stats.mismatches << " maxError=" << chain_after_stats.max_error << std::endl;
        std::cout << "chainLossMismatches=" << chain_loss_stats.mismatches << " maxError=" << chain_loss_stats.max_error << std::endl;
        std::cout << "chainUnmetMismatches=" << chain_unmet_stats.mismatches << " maxError=" << chain_unmet_stats.max_error << std::endl;
        std::cout << "chainEndMismatches=" << chain_end_stats.mismatches << " maxError=" << chain_end_stats.max_error << std::endl;
        std::cout << "chainEndChecksumHex="
                  << std::hex << std::setw(16) << std::setfill('0') << checksum
                  << std::dec << std::endl;

        cudaEventDestroy(stop_event);
        cudaEventDestroy(start_event);
        cudaFree(d_end);
        cudaFree(d_unmet);
        cudaFree(d_loss);
        cudaFree(d_after);
        cudaFree(d_start);
        cudaFree(d_volumes);
        cudaFree(d_ref);

        return 0;

    } catch (const std::exception& error) {
        std::cerr
            << "B12_ERROR="
            << error.what()
            << std::endl;
        return 1;
    }
}

