//! Runtime management for Python integration
//! Fixes the nested runtime issue that causes panics

use std::sync::{Arc, Once};
use tokio::runtime::Runtime;
use pyo3::prelude::*;

static INIT: Once = Once::new();
static mut GLOBAL_RUNTIME: Option<Arc<Runtime>> = None;

/// Global runtime manager for Python integration
pub struct RuntimeManager;

impl RuntimeManager {
    /// Initialize the global runtime (called once)
    pub fn initialize() -> PyResult<()> {
        unsafe {
            INIT.call_once(|| {
                match Runtime::new() {
                    Ok(runtime) => {
                        GLOBAL_RUNTIME = Some(Arc::new(runtime));
                        tracing::info!("✅ Global Tokio runtime initialized for Python integration");
                    }
                    Err(e) => {
                        tracing::error!("❌ Failed to create global runtime: {}", e);
                    }
                }
            });
        }
        
        if unsafe { GLOBAL_RUNTIME.is_none() } {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Failed to initialize global runtime"
            ));
        }
        
        Ok(())
    }
    
    /// Get the global runtime
    pub fn get_runtime() -> PyResult<Arc<Runtime>> {
        unsafe {
            GLOBAL_RUNTIME.clone().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err(
                    "Global runtime not initialized. Call RuntimeManager::initialize() first."
                )
            })
        }
    }
    
    /// Execute an async block on the global runtime
    pub fn block_on<F, R>(future: F) -> PyResult<R>
    where
        F: std::future::Future<Output = R>,
    {
        let runtime = Self::get_runtime()?;
        Ok(runtime.block_on(future))
    }
}

