use crate::cuda_context::CudaSimulationContext;
use napi::bindgen_prelude::*;
use napi_derive::napi;

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

    pub fn reserve(
        &mut self,
        input_bytes: usize,
        workspace_bytes: usize,
        result_bytes: usize,
    ) -> Result<()> {
        self.context.reserve(
            input_bytes,
            workspace_bytes,
            result_bytes,
        )
    }

    pub fn context(&self) -> &CudaSimulationContext {
        &self.context
    }

    pub fn context_mut(&mut self) -> &mut CudaSimulationContext {
        &mut self.context
    }
}

#[derive(Debug)]
pub struct AccelerationController {
    config: AccelerationConfig,
    gpu_workers: Vec<GpuWorker>,
}

impl AccelerationController {
    pub fn new(config: AccelerationConfig) -> Result<Self> {
        Self::new_on_device(config, 0)
    }

    pub fn new_on_device(
        config: AccelerationConfig,
        device_id: i32,
    ) -> Result<Self> {
        let mut gpu_workers =
            Vec::with_capacity(config.gpu_workers);

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

    pub fn reserve_gpu_worker_buffers(
        &mut self,
        input_bytes: usize,
        workspace_bytes: usize,
        result_bytes: usize,
    ) -> Result<()> {
        for worker in &mut self.gpu_workers {
            worker.reserve(
                input_bytes,
                workspace_bytes,
                result_bytes,
            )?;
        }

        Ok(())
    }

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

    #[test]
    fn creates_and_reserves_persistent_gpu_buffers() {
        let mut controller =
            AccelerationController::new(config(0, 1)).unwrap();

        controller
            .reserve_gpu_worker_buffers(
                1024,
                2048,
                512,
            )
            .unwrap();

        let caps = controller.gpu_workers()[0]
            .context()
            .capacities()
            .unwrap();

        assert!(caps.input_bytes >= 1024);
        assert!(caps.workspace_bytes >= 2048);
        assert!(caps.result_bytes >= 512);

        controller
            .synchronize_gpu_workers()
            .unwrap();
    }
}
