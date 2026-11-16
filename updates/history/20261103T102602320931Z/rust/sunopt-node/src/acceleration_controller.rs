use napi::bindgen_prelude::*;
use napi_derive::napi;

/// Public N-API configuration.
/// napi-rs exposes snake_case Rust fields as camelCase JavaScript properties:
/// cpu_workers -> cpuWorkers
/// gpu_workers -> gpuWorkers
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

/// Central acceleration controller configuration.
///
/// `gpu_workers` means persistent GPU worker contexts / CUDA streams,
/// not CUDA hardware threads.
#[derive(Debug)]
pub struct AccelerationController {
    config: AccelerationConfig,
}

impl AccelerationController {
    pub fn new(config: AccelerationConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> AccelerationConfig {
        self.config
    }

    pub fn cpu_workers(&self) -> usize {
        self.config.cpu_workers
    }

    pub fn gpu_workers(&self) -> usize {
        self.config.gpu_workers
    }

    pub fn total_workers(&self) -> usize {
        self.config.total_workers()
    }
}

#[napi(object)]
pub struct AccelerationStatus {
    pub cpu_workers: u32,
    pub gpu_workers: u32,
    pub total_workers: u32,
}

/// Temporary N-API smoke-test export.
/// Keep this until the real controller is wired into the simulation path.
#[napi]
pub fn test_acceleration_config(
    options: AccelerationOptions,
) -> Result<AccelerationStatus> {
    let config = AccelerationConfig::from_options(options)?;
    let controller = AccelerationController::new(config);

    Ok(AccelerationStatus {
        cpu_workers: controller.cpu_workers() as u32,
        gpu_workers: controller.gpu_workers() as u32,
        total_workers: controller.total_workers() as u32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_cpu_and_gpu_workers() {
        let config = AccelerationConfig::from_options(AccelerationOptions {
            cpu_workers: 6,
            gpu_workers: 2,
        })
        .unwrap();

        assert_eq!(config.cpu_workers, 6);
        assert_eq!(config.gpu_workers, 2);
        assert_eq!(config.total_workers(), 8);
    }

    #[test]
    fn accepts_gpu_only() {
        let config = AccelerationConfig::from_options(AccelerationOptions {
            cpu_workers: 0,
            gpu_workers: 1,
        })
        .unwrap();

        assert_eq!(config.total_workers(), 1);
    }

    #[test]
    fn accepts_cpu_only() {
        let config = AccelerationConfig::from_options(AccelerationOptions {
            cpu_workers: 4,
            gpu_workers: 0,
        })
        .unwrap();

        assert_eq!(config.total_workers(), 4);
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
