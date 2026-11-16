extern "C" int sunopt_cuda_run_causal_step_graph(
    void* graph,
    size_t current_index,
    size_t horizon_steps,
    const SunoptCausalDynamic* host_dynamic_parameters,
    size_t job_count,
    double* host_cut_level_results
) {
    auto* step_graph =
        static_cast<SunoptCausalStepGraph*>(graph);

    if (
        step_graph == nullptr ||
        step_graph->context == nullptr ||
        host_dynamic_parameters == nullptr ||
        host_cut_level_results == nullptr ||
        job_count != step_graph->job_count
    ) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    SunoptCudaContext* context = step_graph->context;

    cudaError_t err = cudaSetDevice(context->device_id);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    const size_t result_bytes =
        job_count * sizeof(double);

    err = reserve_mapped_cut_buffer(
        context,
        result_bytes
    );
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    /*
     * The graph node was created by stream capture.  Preserve the
     * exact CUDA-owned func/grid/block/sharedMem metadata captured for
     * that node and replace only its argument values.
     */
    cudaKernelNodeParams params{};

    err = cudaGraphKernelNodeGetParams(
        step_graph->kernel_node,
        &params
    );
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    SunoptCausalDynamicBatch dynamic_batch{};
    for (size_t i = 0; i < job_count; ++i) {
        dynamic_batch.values[i] =
            host_dynamic_parameters[i];
    }

    const size_t one_array_bytes =
        context->causal_length * sizeof(double);

    unsigned char* base =
        static_cast<unsigned char*>(
            context->input_buffer
        );

    const double* export_power =
        reinterpret_cast<const double*>(base);

    const double* demand_energy =
        reinterpret_cast<const double*>(
            base + one_array_bytes
        );

    const auto* device_parameters =
        static_cast<const SunoptCausalParameters*>(
            context->causal_static_parameter_buffer
        );

    int use_dynamic_parameters = 1;
    size_t current_index_value = current_index;
    size_t horizon_steps_value = horizon_steps;
    SunoptCausalResult* results = nullptr;
    double* cut_level_results =
        context->causal_mapped_cut_device;
    size_t job_count_value = job_count;

    void* kernel_args[] = {
        &export_power,
        &demand_energy,
        &device_parameters,
        &dynamic_batch,
        &use_dynamic_parameters,
        &current_index_value,
        &horizon_steps_value,
        &results,
        &cut_level_results,
        &job_count_value,
    };

    params.kernelParams = kernel_args;
    params.extra = nullptr;

    err = cudaGraphExecKernelNodeSetParams(
        step_graph->graph_exec,
        step_graph->kernel_node,
        &params
    );
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    err = cudaGraphLaunch(
        step_graph->graph_exec,
        context->stream
    );
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    err = cudaStreamSynchronize(
        context->stream
    );
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    std::memcpy(
        host_cut_level_results,
        context->causal_mapped_cut_host,
        result_bytes
    );

    context->profile.causal_calls += 1;
    context->profile.causal_jobs +=
        static_cast<uint64_t>(job_count);

    return static_cast<int>(cudaSuccess);
}
