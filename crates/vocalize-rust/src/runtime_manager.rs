//! Runtime management for Python integration
//! Fixes the nested runtime issue that causes panics

use std::sync::{Arc, Mutex, Once};
use tokio::runtime::Runtime;
use vocalize_core::TtsEngine;
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

/// Lazy TTS engine that initializes on first use
#[derive(Debug)]
pub struct LazyTtsEngine {
    engine: Arc<Mutex<Option<Arc<TtsEngine>>>>,
}

impl LazyTtsEngine {
    pub fn new() -> Self {
        Self {
            engine: Arc::new(Mutex::new(None)),
        }
    }
    
    pub fn get_or_init(&self) -> PyResult<Arc<TtsEngine>> {
        let mut engine_guard = self.engine.lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Failed to acquire engine lock: {}", e)
            ))?;
        
        if engine_guard.is_none() {
            tracing::info!("🔄 Initializing TTS engine...");
            
            let engine_result = RuntimeManager::block_on(async {
                TtsEngine::new().await
            })?; // This gives us Result<TtsEngine, VocalizeError>
            let engine = engine_result.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Failed to create TTS engine: {}", e)
            ))?;
            
            *engine_guard = Some(Arc::new(engine));
            tracing::info!("✅ TTS engine initialized successfully");
        }
        
        Ok(engine_guard.as_ref().unwrap().clone())
    }
    
    /// Check if the engine is initialized
    pub fn is_initialized(&self) -> bool {
        if let Ok(guard) = self.engine.lock() {
            guard.is_some()
        } else {
            false
        }
    }
}

impl Default for LazyTtsEngine {
    fn default() -> Self {
        Self::new()
    }
}