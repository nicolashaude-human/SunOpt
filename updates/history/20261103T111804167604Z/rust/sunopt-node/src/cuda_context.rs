use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::ffi::{c_char, c_int, c_void, CStr};
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

    fn sunopt_cuda_reserve(
        context: *mut SunoptCudaContextOpaque,
        input_bytes: usize,
        workspace_bytes: usize,
        result_bytes: usize,
    ) -> c_int;

    fn sunopt_cuda_capacities(
        context: *const SunoptCudaContextOpaque,
        out_input_bytes: *mut usize,
        out_workspace_bytes: *mut usize,
        out_result_bytes: *mut usize,
    ) -> c_int;

    fn sunopt_cuda_upload_input(
        context: *mut SunoptCudaContextOpaque,
        host_data: *const c_void,
        bytes: usize,
    ) -> c_int;

    fn sunopt_cuda_download_result(
        context: *mut SunoptCudaContextOpaque,
        host_data: *mut c_void,
        bytes: usize,
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
            CStr::from_ptr(ptr)
                .to_string_lossy()
                .into_owned()
        }
    };

    Error::from_reason(format!(
        "{operation} failed: {message} (CUDA error {code})"
    ))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaBufferCapacities {
    pub input_bytes: usize,
    pub workspace_bytes: usize,
    pub result_bytes: usize,
}

/// Persistent CUDA simulation context.
///
/// The CUDA stream and device allocations are long-lived. `reserve()` only
/// grows buffers when necessary; it never shrinks them during the context
/// lifetime. This is the basis for SunOpt's GPU-resident execution model.
#[derive(Debug)]
pub struct CudaSimulationContext {
    raw: NonNull<SunoptCudaContextOpaque>,
}

unsafe impl Send for CudaSimulationContext {}

impl CudaSimulationContext {
    pub fn new(device_id: i32) -> Result<Self> {
        let mut raw = std::ptr::null_mut();

        let code = unsafe {
            sunopt_cuda_create(
                device_id as c_int,
                &mut raw,
            )
        };

        if code != 0 {
            return Err(cuda_error(
                "sunopt_cuda_create",
                code,
            ));
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
            sunopt_cuda_device_id(
                self.raw.as_ptr(),
                &mut device_id,
            )
        };

        if code != 0 {
            return Err(cuda_error(
                "sunopt_cuda_device_id",
                code,
            ));
        }

        Ok(device_id as i32)
    }

    pub fn synchronize(&mut self) -> Result<()> {
        let code = unsafe {
            sunopt_cuda_stream_synchronize(
                self.raw.as_ptr(),
            )
        };

        if code != 0 {
            return Err(cuda_error(
                "sunopt_cuda_stream_synchronize",
                code,
            ));
        }

        Ok(())
    }

    pub fn reserve(
        &mut self,
        input_bytes: usize,
        workspace_bytes: usize,
        result_bytes: usize,
    ) -> Result<()> {
        let code = unsafe {
            sunopt_cuda_reserve(
                self.raw.as_ptr(),
                input_bytes,
                workspace_bytes,
                result_bytes,
            )
        };

        if code != 0 {
            return Err(cuda_error(
                "sunopt_cuda_reserve",
                code,
            ));
        }

        Ok(())
    }

    pub fn capacities(
        &self,
    ) -> Result<CudaBufferCapacities> {
        let mut input_bytes = 0usize;
        let mut workspace_bytes = 0usize;
        let mut result_bytes = 0usize;

        let code = unsafe {
            sunopt_cuda_capacities(
                self.raw.as_ptr(),
                &mut input_bytes,
                &mut workspace_bytes,
                &mut result_bytes,
            )
        };

        if code != 0 {
            return Err(cuda_error(
                "sunopt_cuda_capacities",
                code,
            ));
        }

        Ok(CudaBufferCapacities {
            input_bytes,
            workspace_bytes,
            result_bytes,
        })
    }

    pub fn upload_input(
        &mut self,
        data: &[u8],
    ) -> Result<()> {
        let code = unsafe {
            sunopt_cuda_upload_input(
                self.raw.as_ptr(),
                data.as_ptr().cast::<c_void>(),
                data.len(),
            )
        };

        if code != 0 {
            return Err(cuda_error(
                "sunopt_cuda_upload_input",
                code,
            ));
        }

        Ok(())
    }

    pub fn download_result(
        &mut self,
        output: &mut [u8],
    ) -> Result<()> {
        let code = unsafe {
            sunopt_cuda_download_result(
                self.raw.as_ptr(),
                output.as_mut_ptr().cast::<c_void>(),
                output.len(),
            )
        };

        if code != 0 {
            return Err(cuda_error(
                "sunopt_cuda_download_result",
                code,
            ));
        }

        Ok(())
    }
}

impl Drop for CudaSimulationContext {
    fn drop(&mut self) {
        let _ = unsafe {
            sunopt_cuda_destroy(
                self.raw.as_ptr(),
            )
        };
    }
}

#[napi(object)]
pub struct CudaContextStatus {
    pub device_id: i32,
    pub stream_ready: bool,
}

#[napi(object)]
pub struct CudaBufferStatus {
    pub device_id: i32,
    pub input_bytes: u32,
    pub workspace_bytes: u32,
    pub result_bytes: u32,
}

/// End-to-end smoke test for the persistent CUDA stream.
#[napi]
pub fn test_cuda_context(
    device_id: Option<i32>,
) -> Result<CudaContextStatus> {
    let mut context =
        CudaSimulationContext::new(device_id.unwrap_or(0))?;

    context.synchronize()?;

    Ok(CudaContextStatus {
        device_id: context.device_id()?,
        stream_ready: true,
    })
}

/// Reserve real persistent VRAM buffers and report the resulting capacities.
///
/// This intentionally does not run a simulation yet. It validates the
/// allocation/reuse layer independently before causalSearch is ported.
#[napi]
pub fn test_cuda_buffers(
    input_bytes: u32,
    workspace_bytes: u32,
    result_bytes: u32,
    device_id: Option<i32>,
) -> Result<CudaBufferStatus> {
    let mut context =
        CudaSimulationContext::new(device_id.unwrap_or(0))?;

    context.reserve(
        input_bytes as usize,
        workspace_bytes as usize,
        result_bytes as usize,
    )?;

    context.synchronize()?;

    let capacities = context.capacities()?;

    Ok(CudaBufferStatus {
        device_id: context.device_id()?,
        input_bytes: capacities.input_bytes as u32,
        workspace_bytes: capacities.workspace_bytes as u32,
        result_bytes: capacities.result_bytes as u32,
    })
}
