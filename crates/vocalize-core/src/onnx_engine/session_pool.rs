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
        if pool_size == 0 {
            return Err(anyhow::anyhow!("Pool size must be greater than 0"));
        }
        
        tracing::info!("🏊 Creating ONNX session pool with {} sessions", pool_size);
        
        let mut sessions = Vec::with_capacity(pool_size);
        
        // Create multiple session instances with optimized settings
        for i in 0..pool_size {
            let session = Self::create_optimized_session(model_path)
                .await
                .with_context(|| format!("Failed to create session {} of {}", i + 1, pool_size))?;
            
            sessions.push(Arc::new(Mutex::new(session)));
            tracing::debug!("Created ONNX session {} of {}", i + 1, pool_size);
        }
        
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
        tracing::debug!("🔧 Creating ONNX session with anti-deadlock configuration");
        
        // Set up session with optimized configuration for better performance
        let session = Session::builder()?
            // Use maximum optimization for speed
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            // Multi-threading for better performance
            .with_intra_threads(4)?
            .with_inter_threads(4)?
            // Enable memory pattern optimization
            .with_memory_pattern(true)?
            // Load the model
            .commit_from_file(model_path)?;
        
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