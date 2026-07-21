use crate::layers::HiddenAct;
use chalice::{Device, Result, Tensor};
use std::sync::Once;

static INIT: Once = Once::new();
static mut CUBLASLT: Option<CublasLtWrapper> = None;

pub fn get_cublas_lt_wrapper() -> Option<&'static CublasLtWrapper> {
    unsafe {
        INIT.call_once(|| {
            let enable_cublaslt = std::env::var("TEI_ENABLE_EXPERIMENTAL_CUBLASLT")
                .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
                .unwrap_or(false);
            if !enable_cublaslt {
                CUBLASLT = None;
                return;
            }

            #[cfg(not(feature = "cuda"))]
            {
                CUBLASLT = None;
            }

            #[cfg(feature = "cuda")]
            {
                // Check if we can call the driver
                // Then check if we can create a device
                // Then check that the device is CUDA
                use chalice::cuda_backend::cudarc::driver;
                CUBLASLT = driver::result::init()
                    .ok()
                    .and_then(|_| Device::cuda_if_available(0).ok())
                    .and_then(|device| match device {
                        Device::Cuda(_) => Some(CublasLtWrapper {}),
                        _ => None,
                    });
            }
        });
        #[allow(static_mut_refs)]
        CUBLASLT.as_ref()
    }
}

#[derive(Debug, Clone)]
pub struct CublasLtWrapper {}

impl CublasLtWrapper {
    #[allow(clippy::too_many_arguments)]
    pub fn matmul(
        &self,
        a: &Tensor,
        b: &Tensor,
        out: Option<&Tensor>,
        alpha: Option<f32>,
        beta: Option<f32>,
        bias: Option<&Tensor>,
        act: Option<HiddenAct>,
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            let mut result = b.matmul_with_epilogue(&a.t()?, bias, None)?;
            if let Some(alpha) = alpha {
                result = (result * alpha as f64)?;
            }
            if let Some(out) = out {
                result = if let Some(beta) = beta {
                    (result + (out * beta as f64)?)?
                } else {
                    (result + out)?
                };
            }

            if Some(HiddenAct::Swiglu) == act {
                result = chalice_nn::ops::swiglu(&result)?;
            } else if let Some(act) = &act {
                result = act.forward(&result)?;
            }
            Ok(result)
        }
        #[cfg(not(feature = "cuda"))]
        {
            chalice::bail!("`cuda` feature is not enabled")
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn batch_matmul(
        &self,
        a: &Tensor,
        b: &Tensor,
        out: Option<&Tensor>,
        alpha: Option<f32>,
        beta: Option<f32>,
        bias: Option<&Tensor>,
        act: Option<HiddenAct>,
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            let mut result = b.matmul_with_epilogue(&a.t()?, bias, None)?;
            if let Some(alpha) = alpha {
                result = (result * alpha as f64)?;
            }
            if let Some(out) = out {
                result = if let Some(beta) = beta {
                    (result + (out * beta as f64)?)?
                } else {
                    (result + out)?
                };
            }

            if Some(HiddenAct::Swiglu) == act {
                result = chalice_nn::ops::swiglu(&result)?;
            } else if let Some(act) = &act {
                result = act.forward(&result)?;
            }
            Ok(result)
        }
        #[cfg(not(feature = "cuda"))]
        {
            chalice::bail!("`cuda` feature is not enabled")
        }
    }
}
