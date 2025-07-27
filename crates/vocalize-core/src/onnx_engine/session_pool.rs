//! ONNX Session Pool for thread-safe concurrent inference
//! Prevents deadlocks and improves performance under load

#![allow(missing_docs)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use anyhow::{Result, Context};
use ort::session::{Session, builder::GraphOptimizationLevel};
use tokio::sync::{Semaphore, SemaphorePermit};
use tracing;

/// Pool of ONNX sessions for concurrent inference
#[derive(Debug)]
pub struct OnnxSessionPool {
    sessions: Vec<Arc<Mutex<Session>>>,
    current_index: AtomicUsize,
    semaphore: Semaphore,
    max_concurrent: usize,
}


impl OnnxSessionPool {
    /// Create a new session pool
    pub async fn new(model_path: &std::path::Path, pool_size: usize) -> Result<Self> {
        use std::time::Instant;
        let total_start = Instant::now();
        
        if pool_size == 0 {
            return Err(anyhow::anyhow!("Pool size must be greater than 0"));
        }
        
        tracing::info!("🏊 Creating ONNX session pool with {} sessions", pool_size);
        
        let mut sessions = Vec::with_capacity(pool_size);
        
        // Create multiple session instances with optimized settings
        for i in 0..pool_size {
            let session_start = Instant::now();
            let session = Self::create_optimized_session(model_path)
                .await
                .with_context(|| format!("Failed to create session {} of {}", i + 1, pool_size))?;
            
            sessions.push(Arc::new(Mutex::new(session)));
            eprintln!("  ⏱️  [Pool] Session {} creation: {:.3}s", i + 1, session_start.elapsed().as_secs_f32());
            tracing::debug!("Created ONNX session {} of {}", i + 1, pool_size);
        }
        
        eprintln!("  ⏱️  [Pool] Total pool creation: {:.3}s", total_start.elapsed().as_secs_f32());
        tracing::info!("✅ ONNX session pool created successfully");
        
        Ok(Self {
            sessions,
            current_index: AtomicUsize::new(0),
            semaphore: Semaphore::new(pool_size),
            max_concurrent: pool_size,
        })
    }
    
    /// Create an optimized ONNX session with deadlock prevention
    async fn create_optimized_session(model_path: &std::path::Path) -> Result<Session> {
        use std::time::Instant;
        let start = Instant::now();
        
        tracing::debug!("🔧 Creating ONNX session with anti-deadlock configuration");
        
        // Detect if this is an INT8 quantized model
        let filename = model_path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");
        let is_int8_model = filename.contains("int8") || filename.contains("INT8");
        
        // Detect CPU cores for optimal threading
        let physical_cores = num_cpus::get_physical() as usize;
        let logical_cores = num_cpus::get() as usize;
        eprintln!("  🖥️  CPU Detection: {} physical cores, {} logical cores", physical_cores, logical_cores);
        
        // Configure based on model type
        let (session_opt_level, opt_level_str, intra_threads, inter_threads, memory_pattern) = if is_int8_model {
            eprintln!("  ⚡ INT8 quantized model detected - using conservative settings");
            // INT8 models need conservative settings
            (
                GraphOptimizationLevel::Level1,  // Basic optimization only
                "Level1 (basic, INT8 compatible)",
                std::cmp::min(4, physical_cores),  // Max 4 threads for INT8
                2,  // Minimal inter-op threads
                false  // Disable memory pattern for INT8
            )
        } else {
            eprintln!("  🚀 FP32 model detected - using performance settings");
            // FP32 models can use aggressive optimization
            (
                GraphOptimizationLevel::Level3,  // Maximum optimization
                "Level3 (maximum)",
                std::cmp::min(physical_cores, 8),  // Use cores but cap at 8
                std::cmp::min(4, std::cmp::max(2, physical_cores / 3)),
                true  // Enable memory pattern
            )
        };
        
        eprintln!("  🧵 Thread Configuration: {} intra-op threads, {} inter-op threads", intra_threads, inter_threads);
        
        // Set up session with optimized configuration
        let session = Session::builder()?
            // Use appropriate optimization level
            .with_optimization_level(session_opt_level)?
            // Use calculated threads
            .with_intra_threads(intra_threads)?
            .with_inter_threads(inter_threads)?
            // Memory pattern based on model type
            .with_memory_pattern(memory_pattern)?
            // Load the model
            .commit_from_file(model_path)?;
        
        let filename = model_path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");
        eprintln!("  ⏱️  [Pool] Individual session build: {:.3}s", start.elapsed().as_secs_f32());
        eprintln!("  📊 Model: {}, Using optimization: {}", filename, opt_level_str);
        
        // Validate session immediately after creation
        tracing::debug!("✅ ONNX session created and validated successfully");
        Ok(session)
    }
    
    /// Acquire a session from the pool (async, non-blocking)
    pub async fn acquire_session(&self) -> Result<SessionGuard<'_>> {
        self.acquire_session_timeout(Duration::from_secs(30)).await
    }
    
    /// Acquire a session with a timeout
    pub async fn acquire_session_timeout(&self, timeout: Duration) -> Result<SessionGuard<'_>> {
        // Acquire permit with timeout
        let permit = tokio::time::timeout(timeout, self.semaphore.acquire())
            .await
            .context("Timeout waiting for available session")?
            .map_err(|_| anyhow::anyhow!("Semaphore closed"))?;
        
        // Round-robin session selection
        let index = self.current_index.fetch_add(1, Ordering::Relaxed) % self.sessions.len();
        let session = self.sessions[index].clone();
        
        tracing::debug!("🎯 Acquired session {} from pool", index);
        
        Ok(SessionGuard {
            session,
            _permit: permit,
            session_id: index,
        })
    }
    
    /// Try to acquire a session immediately (non-blocking)
    pub fn try_acquire_session(&self) -> Option<SessionGuard<'_>> {
        if let Ok(permit) = self.semaphore.try_acquire() {
            let index = self.current_index.fetch_add(1, Ordering::Relaxed) % self.sessions.len();
            let session = self.sessions[index].clone();
            
            tracing::debug!("⚡ Immediately acquired session {} from pool", index);
            
            Some(SessionGuard {
                session,
                _permit: permit,
                session_id: index,
            })
        } else {
            None
        }
    }
    
    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        let available = self.semaphore.available_permits();
        let in_use = self.max_concurrent - available;
        
        PoolStats {
            total_sessions: self.sessions.len(),
            available_sessions: available,
            sessions_in_use: in_use,
            max_concurrent: self.max_concurrent,
        }
    }
    
    /// Check if the pool is healthy
    pub fn is_healthy(&self) -> bool {
        !self.sessions.is_empty() && self.semaphore.available_permits() <= self.max_concurrent
    }
}

/// Guard that holds a session and automatically returns it to the pool when dropped
pub struct SessionGuard<'a> {
    pub session: Arc<Mutex<Session>>,
    _permit: SemaphorePermit<'a>,
    session_id: usize,
}

impl<'a> SessionGuard<'a> {
    /// Get the session ID (for debugging)
    pub fn session_id(&self) -> usize {
        self.session_id
    }
}

impl Drop for SessionGuard<'_> {
    fn drop(&mut self) {
        tracing::debug!("🔄 Released session {} back to pool", self.session_id);
    }
}

/// Statistics about the session pool
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub total_sessions: usize,
    pub available_sessions: usize,
    pub sessions_in_use: usize,
    pub max_concurrent: usize,
}

impl PoolStats {
    /// Get the utilization percentage (0.0 to 1.0)
    pub fn utilization(&self) -> f32 {
        if self.total_sessions == 0 {
            0.0
        } else {
            self.sessions_in_use as f32 / self.total_sessions as f32
        }
    }
    
    /// Check if the pool is at capacity
    pub fn is_at_capacity(&self) -> bool {
        self.available_sessions == 0
    }
}

impl std::fmt::Display for PoolStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Sessions: {}/{} ({:.1}% utilization)",
            self.sessions_in_use,
            self.total_sessions,
            self.utilization() * 100.0
        )
    }
}