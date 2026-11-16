extern "C" int sunopt_cuda_create_causal_step_graph(
    SunoptCudaContext* context,
    size_t job_count,
    void** out_graph
) {
    if (
        context == nullptr ||
        out_graph == nullptr ||
        job_count == 0 ||
        job_count > 32 ||
        context->causal_length == 0 ||
        context->causal_static_parameter_buffer == nullptr ||
        context->causal_static_job_count != job_count
    ) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    *out_graph = nullptr;

    cudaError_t err =
        cudaSetDevice(context->device_id);

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    const size_t result_bytes =
        job_count * sizeof(double);

    err =
        reserve_mapped_cut_buffer(
            context,
            result_bytes
        );

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    auto* step_graph =
        new (std::nothrow)
        SunoptCausalStepGraph();

    if (step_graph == nullptr) {
        return static_cast<int>(
            cudaErrorMemoryAllocation
        );
    }

    step_graph->context = context;
    step_graph->job_count = job_count;

    const size_t one_array_bytes =
        context->causal_length *
        sizeof(double);

    unsigned char* base =
        static_cast<unsigned char*>(
            context->input_buffer
        );

    const double* export_power =
        reinterpret_cast<const double*>(
            base
        );

    const double* demand_energy =
        reinterpret_cast<const double*>(
            base + one_array_bytes
        );

    const auto* device_parameters =
        static_cast<
            const SunoptCausalParameters*
        >(
            context
                ->causal_static_parameter_buffer
        );

    SunoptCausalDynamicBatch dynamic_batch{};

    constexpr unsigned int THREADS =
        32;

    const unsigned int blocks =
        static_cast<unsigned int>(
            job_count
        );

    /*
     * Capture exactly one production FP64 causal kernel.
     * Host/controller feedback remains between every annual step.
     */
    err =
        cudaStreamBeginCapture(
            context->stream,
            cudaStreamCaptureModeThreadLocal
        );

    if (err == cudaSuccess) {
        causal_batch_kernel<<<
            blocks,
            THREADS,
            0,
            context->stream
        >>>(
            export_power,
            demand_energy,
            device_parameters,
            dynamic_batch,
            1,
            0,
            1,
            nullptr,
            context->causal_mapped_cut_device,
            job_count
        );

        err =
            cudaGetLastError();
    }

    if (err == cudaSuccess) {
        err =
            cudaStreamEndCapture(
                context->stream,
                &step_graph->graph
            );
    } else {
        cudaGraph_t discarded =
            nullptr;

        cudaStreamEndCapture(
            context->stream,
            &discarded
        );

        if (discarded != nullptr) {
            cudaGraphDestroy(discarded);
        }
    }

    if (
        err == cudaSuccess &&
        step_graph->graph == nullptr
    ) {
        err = cudaErrorInvalidValue;
    }

    size_t node_count = 0;

    if (err == cudaSuccess) {
        err =
            cudaGraphGetNodes(
                step_graph->graph,
                nullptr,
                &node_count
            );
    }

    if (
        err == cudaSuccess &&
        node_count != 1
    ) {
        err = cudaErrorInvalidValue;
    }

    if (err == cudaSuccess) {
        size_t requested = 1;

        err =
            cudaGraphGetNodes(
                step_graph->graph,
                &step_graph->kernel_node,
                &requested
            );

        if (
            err == cudaSuccess &&
            (
                requested != 1 ||
                step_graph->kernel_node ==
                    nullptr
            )
        ) {
            err = cudaErrorInvalidValue;
        }
    }

    if (err == cudaSuccess) {
        cudaGraphNodeType node_type;

        err =
            cudaGraphNodeGetType(
                step_graph->kernel_node,
                &node_type
            );

        if (
            err == cudaSuccess &&
            node_type !=
                cudaGraphNodeTypeKernel
        ) {
            err = cudaErrorInvalidValue;
        }
    }

    if (err == cudaSuccess) {
        err =
            cudaGraphInstantiate(
                &step_graph->graph_exec,
                step_graph->graph,
                nullptr,
                nullptr,
                0
            );
    }

    if (err != cudaSuccess) {
        if (
            step_graph->graph_exec !=
            nullptr
        ) {
            cudaGraphExecDestroy(
                step_graph->graph_exec
            );
        }

        if (
            step_graph->graph !=
            nullptr
        ) {
            cudaGraphDestroy(
                step_graph->graph
            );
        }

        delete step_graph;

        return static_cast<int>(err);
    }

    *out_graph =
        static_cast<void*>(
            step_graph
        );

    return static_cast<int>(
        cudaSuccess
    );
}
