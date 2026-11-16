use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::ffi::{c_char, c_int, CStr};
use std::ptr::NonNull;

#[repr(C)]
struct SunoptCudaContextOpaque {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn sunopt_cuda_create(
        device_id: c_int,
        out_context: *mut *mut SunoptCudaContextOpaque,
    ) -> c_int;

    fn sunopt_cuda_destroy(
        context: *mut SunoptCudaContextOpaque,
    ) -> c_int;

    fn sunopt_cuda_device_id(
        context: *const SunoptCudaContextOpaque,
        out_device_id: *mut c_int,
    ) -> c_int;

    fn sunopt_cuda_stream_synchronize(
        context: *mut SunoptCudaContextOpaque,
    ) -> c_int;

    fn sunopt_cuda_error_string(
        error_code: c_int,
    ) -> *const c_char;
}

fn cuda_error(operation: &str, code: c_int) -> Error {
    let message = unsafe {
        let ptr = sunopt_cuda_error_string(code);

        if ptr.is_null() {
            format!("CUDA error {code}")
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    Error::from_reason(format!(
        "{operation} failed: {message} (CUDA error {code})"
    ))
}

/// Persistent CUDA simulation context.
///
/// The CUDA stream is created once and remains alive until this Rust value
/// is dropped. Future packages will add persistent device buffers/workspace
/// to this same context.
#[derive(Debug)]
pub struct CudaSimulationContext {
    raw: NonNull<SunoptCudaContextOpaque>,
}

unsafe impl Send for CudaSimulationContext {}

impl CudaSimulationContext {
    pub fn new(device_id: i32) -> Result<Self> {
        let mut raw = std::ptr::null_mut();

        let code = unsafe {
            sunopt_cuda_create(device_id as c_int, &mut raw)
        };

        if code != 0 {
            return Err(cuda_error("sunopt_cuda_create", code));
        }

        let raw = NonNull::new(raw).ok_or_else(|| {
            Error::from_reason(
                "sunopt_cuda_create returned success with a null context",
            )
        })?;

        Ok(Self { raw })
    }

    pub fn device_id(&self) -> Result<i32> {
        let mut device_id: c_int = -1;

        let code = unsafe {
            sunopt_cuda_device_id(self.raw.as_ptr(), &mut device_id)
        };

        if code != 0 {
            return Err(cuda_error("sunopt_cuda_device_id", code));
        }

        Ok(device_id as i32)
    }

    pub fn synchronize(&mut self) -> Result<()> {
        let code = unsafe {
            sunopt_cuda_stream_synchronize(self.raw.as_ptr())
        };

        if code != 0 {
            return Err(cuda_error(
                "sunopt_cuda_stream_synchronize",
                code,
            ));
        }

        Ok(())
    }
}

impl Drop for CudaSimulationContext {
    fn drop(&mut self) {
        let _ = unsafe { sunopt_cuda_destroy(self.raw.as_ptr()) };
    }
}

#[napi(object)]
pub struct CudaContextStatus {
    pub device_id: i32,
    pub stream_ready: bool,
}

/// End-to-end smoke test for the persistent CUDA context.
/// JavaScript name: testCudaContext
#[napi]
pub fn test_cuda_context(
    device_id: Option<i32>,
) -> Result<CudaContextStatus> {
    let mut context = CudaSimulationContext::new(device_id.unwrap_or(0))?;
    context.synchronize()?;

    Ok(CudaContextStatus {
        device_id: context.device_id()?,
        stream_ready: true,
    })
}
