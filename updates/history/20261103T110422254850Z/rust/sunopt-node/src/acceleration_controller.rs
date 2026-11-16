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

/// Logical persistent GPU worker slot.
///
/// This is deliberately backend-agnostic for now. The next CUDA package
/// will attach one persistent `CudaSimulationContext` to each slot.
///
/// A GPU worker is NOT a CUDA hardware thread. It is a long-lived host-side
/// worker/context that owns GPU-resident simulation state and may later map
/// to a CUDA stream/context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuWorkerSlot {
    id: usize,
}

impl GpuWorkerSlot {
    fn new(id: usize) -> Self {
        Self { id }
    }

    pub fn id(&self) -> usize {
        self.id
    }
}

/// Central SunOpt acceleration controller.
///
/// Current responsibilities:
/// - validate CPU/GPU worker counts;
/// - expose worker counts to the native controller;
/// - pre-create the requested number of logical persistent GPU worker slots.
///
/// Next step:
/// - replace/extend each `GpuWorkerSlot` with a persistent
///   `CudaSimulationContext`;
/// - allocate GPU buffers once per context;
/// - keep simulation data resident in VRAM through hot search phases;
/// - preserve the existing CPU/Cranelift path as fallback.
#[derive(Debug)]
pub struct AccelerationController {
    config: AccelerationConfig,
    gpu_workers: Vec<GpuWorkerSlot>,
}

impl AccelerationController {
    pub fn new(config: AccelerationConfig) -> Self {
        let gpu_workers = (0..config.gpu_workers)
            .map(GpuWorkerSlot::new)
            .collect();

        Self {
            config,
            gpu_workers,
        }
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

    pub fn gpu_workers(&self) -> &[GpuWorkerSlot] {
        &self.gpu_workers
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
/// Keep this export while the real scheduler is being connected. It verifies
/// that JavaScript -> N-API -> Rust worker configuration is correct.
#[napi]
pub fn test_acceleration_config(
    options: AccelerationOptions,
) -> Result<AccelerationStatus> {
    let config = AccelerationConfig::from_options(options)?;
    let controller = AccelerationController::new(config);

    Ok(AccelerationStatus {
        cpu_workers: controller.cpu_workers() as u32,
        gpu_workers: controller.gpu_worker_count() as u32,
        total_workers: controller.total_workers() as u32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn controller(cpu_workers: u32, gpu_workers: u32) -> AccelerationController {
        let config = AccelerationConfig::from_options(AccelerationOptions {
            cpu_workers,
            gpu_workers,
        })
        .unwrap();

        AccelerationController::new(config)
    }

    #[test]
    fn accepts_cpu_and_gpu_workers() {
        let controller = controller(6, 2);

        assert_eq!(controller.cpu_workers(), 6);
        assert_eq!(controller.gpu_worker_count(), 2);
        assert_eq!(controller.total_workers(), 8);
    }

    #[test]
    fn creates_requested_gpu_worker_slots() {
        let controller = controller(4, 3);

        assert_eq!(controller.gpu_workers().len(), 3);
        assert_eq!(controller.gpu_workers()[0].id(), 0);
        assert_eq!(controller.gpu_workers()[1].id(), 1);
        assert_eq!(controller.gpu_workers()[2].id(), 2);
    }

    #[test]
    fn accepts_gpu_only() {
        let controller = controller(0, 1);

        assert_eq!(controller.cpu_workers(), 0);
        assert_eq!(controller.gpu_worker_count(), 1);
        assert_eq!(controller.total_workers(), 1);
    }

    #[test]
    fn accepts_cpu_only() {
        let controller = controller(4, 0);

        assert_eq!(controller.cpu_workers(), 4);
        assert_eq!(controller.gpu_worker_count(), 0);
        assert_eq!(controller.total_workers(), 4);
        assert!(controller.gpu_workers().is_empty());
    }

    #[test]
    fn rejects_zero_workers() {
        let result = AccelerationConfig::from_options(AccelerationOptions {
            cpu_workers: 0,
            gpu_workers: 0,
        });

        assert!(result.is_err());
    }
}
