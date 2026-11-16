use crate::cuda_context::CudaSimulationContext;
use napi::bindgen_prelude::*;
use napi_derive::napi;

/// Public N-API configuration.
///
/// napi-rs exposes snake_case Rust fields as camelCase JavaScript properties:
/// - cpu_workers -> cpuWorkers
/// - gpu_workers -> gpuWorkers
#[napi(object)]
pub struct AccelerationOptions {
    pub cpu_workers: u32,
    pub gpu_workers: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AccelerationConfig {
    pub cpu_workers: usize,
    pub gpu_workers: usize,
}

impl AccelerationConfig {
    pub fn from_options(options: AccelerationOptions) -> Result<Self> {
        let cpu_workers = options.cpu_workers as usize;
        let gpu_workers = options.gpu_workers as usize;

        if cpu_workers == 0 && gpu_workers == 0 {
            return Err(Error::from_reason(
                "At least one CPU worker or GPU worker must be enabled.",
            ));
        }

        Ok(Self {
            cpu_workers,
            gpu_workers,
        })
    }

    pub fn total_workers(&self) -> usize {
        self.cpu_workers + self.gpu_workers
    }
}

/// One persistent GPU worker.
///
/// Each worker owns one long-lived CUDA simulation context. The context owns
/// a non-blocking CUDA stream and will later also own persistent VRAM buffers.
///
/// A GPU worker is NOT a CUDA hardware thread.
#[derive(Debug)]
pub struct GpuWorker {
    id: usize,
    context: CudaSimulationContext,
}

impl GpuWorker {
    fn new(id: usize, device_id: i32) -> Result<Self> {
        Ok(Self {
            id,
            context: CudaSimulationContext::new(device_id)?,
        })
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn device_id(&self) -> Result<i32> {
        self.context.device_id()
    }

    pub fn synchronize(&mut self) -> Result<()> {
        self.context.synchronize()
    }

    pub fn context(&self) -> &CudaSimulationContext {
        &self.context
    }

    pub fn context_mut(&mut self) -> &mut CudaSimulationContext {
        &mut self.context
    }
}

/// Central SunOpt acceleration controller.
///
/// CPU workers remain represented by their configured worker count for now.
/// GPU workers are already real persistent CUDA contexts.
///
/// Current GPU lifecycle:
///
/// AccelerationController::new
///   -> create N GPU workers
///   -> create one CudaSimulationContext per GPU worker
///   -> keep CUDA stream alive
///   -> run future simulation jobs
///   -> controller drop
///   -> CudaSimulationContext::drop
///   -> cudaStreamDestroy
///
/// The next step is to add persistent device buffers/workspaces to each
/// CudaSimulationContext.
#[derive(Debug)]
pub struct AccelerationController {
    config: AccelerationConfig,
    gpu_workers: Vec<GpuWorker>,
}

impl AccelerationController {
    /// Create a controller using CUDA device 0 for every GPU worker.
    ///
    /// A later multi-GPU extension can add explicit device assignment without
    /// changing the meaning of `gpu_workers`.
    pub fn new(config: AccelerationConfig) -> Result<Self> {
        Self::new_on_device(config, 0)
    }

    pub fn new_on_device(
        config: AccelerationConfig,
        device_id: i32,
    ) -> Result<Self> {
        let mut gpu_workers =
            Vec::with_capacity(config.gpu_workers);

        // Fail closed: if any CUDA worker cannot be created, previously
        // created workers are dropped automatically and their CUDA streams
        // are destroyed.
        for id in 0..config.gpu_workers {
            gpu_workers.push(
                GpuWorker::new(id, device_id)?
            );
        }

        Ok(Self {
            config,
            gpu_workers,
        })
    }

    pub fn config(&self) -> AccelerationConfig {
        self.config
    }

    pub fn cpu_workers(&self) -> usize {
        self.config.cpu_workers
    }

    pub fn gpu_worker_count(&self) -> usize {
        self.gpu_workers.len()
    }

    pub fn total_workers(&self) -> usize {
        self.cpu_workers() + self.gpu_worker_count()
    }

    pub fn gpu_workers(&self) -> &[GpuWorker] {
        &self.gpu_workers
    }

    pub fn gpu_workers_mut(&mut self) -> &mut [GpuWorker] {
        &mut self.gpu_workers
    }

    /// Synchronize all persistent GPU worker streams.
    ///
    /// Primarily useful for lifecycle tests and later scheduler barriers.
    pub fn synchronize_gpu_workers(&mut self) -> Result<()> {
        for worker in &mut self.gpu_workers {
            worker.synchronize()?;
        }

        Ok(())
    }
}

#[napi(object)]
pub struct AccelerationStatus {
    pub cpu_workers: u32,
    pub gpu_workers: u32,
    pub total_workers: u32,
}

/// N-API smoke-test/status export.
///
/// This now creates real CUDA contexts when gpuWorkers > 0.
///
/// JavaScript:
///
/// testAccelerationConfig({
///   cpuWorkers: 6,
///   gpuWorkers: 2
/// })
#[napi]
pub fn test_acceleration_config(
    options: AccelerationOptions,
) -> Result<AccelerationStatus> {
    let config = AccelerationConfig::from_options(options)?;
    let mut controller = AccelerationController::new(config)?;

    controller.synchronize_gpu_workers()?;

    Ok(AccelerationStatus {
        cpu_workers: controller.cpu_workers() as u32,
        gpu_workers: controller.gpu_worker_count() as u32,
        total_workers: controller.total_workers() as u32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(
        cpu_workers: u32,
        gpu_workers: u32,
    ) -> AccelerationConfig {
        AccelerationConfig::from_options(
            AccelerationOptions {
                cpu_workers,
                gpu_workers,
            },
        )
        .unwrap()
    }

    #[test]
    fn accepts_cpu_only_without_cuda_contexts() {
        let controller =
            AccelerationController::new(config(4, 0)).unwrap();

        assert_eq!(controller.cpu_workers(), 4);
        assert_eq!(controller.gpu_worker_count(), 0);
        assert_eq!(controller.total_workers(), 4);
        assert!(controller.gpu_workers().is_empty());
    }

    #[test]
    fn rejects_zero_workers() {
        let result =
            AccelerationConfig::from_options(
                AccelerationOptions {
                    cpu_workers: 0,
                    gpu_workers: 0,
                },
            );

        assert!(result.is_err());
    }

    /// Requires a working CUDA device 0.
    #[test]
    fn creates_real_persistent_gpu_worker() {
        let mut controller =
            AccelerationController::new(config(0, 1)).unwrap();

        assert_eq!(controller.gpu_worker_count(), 1);
        assert_eq!(controller.gpu_workers()[0].id(), 0);
        assert_eq!(
            controller.gpu_workers()[0].device_id().unwrap(),
            0
        );

        controller
            .synchronize_gpu_workers()
            .unwrap();
    }

    /// Requires a working CUDA device 0.
    #[test]
    fn creates_requested_number_of_gpu_contexts() {
        let mut controller =
            AccelerationController::new(config(2, 2)).unwrap();

        assert_eq!(controller.cpu_workers(), 2);
        assert_eq!(controller.gpu_worker_count(), 2);
        assert_eq!(controller.total_workers(), 4);

        assert_eq!(controller.gpu_workers()[0].id(), 0);
        assert_eq!(controller.gpu_workers()[1].id(), 1);

        controller
            .synchronize_gpu_workers()
            .unwrap();
    }
}
