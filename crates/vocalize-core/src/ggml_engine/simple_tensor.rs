//! Simple tensor wrapper for mock GGML implementation
//! Replaces Candle to avoid cross-compilation issues

use anyhow::Result;

/// Simple tensor wrapper for our mock implementation
#[derive(Debug, Clone)]
pub struct SimpleTensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl SimpleTensor {
    /// Create a new tensor from a vector and shape
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Result<Self> {
        // Validate shape matches data
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            return Err(anyhow::anyhow!(
                "Data size {} doesn't match shape {:?} (expected {})",
                data.len(), shape, expected_size
            ));
        }
        
        Ok(Self {
            data,
            shape: shape.to_vec(),
        })
    }
    
    /// Create a new tensor with a single value
    pub fn new(value: &[f32]) -> Result<Self> {
        Ok(Self {
            data: value.to_vec(),
            shape: vec![value.len()],
        })
    }
    
    /// Flatten all dimensions
    pub fn flatten_all(&self) -> Result<&Vec<f32>> {
        Ok(&self.data)
    }
    
    /// Convert to vector of specific type (for compatibility)
    pub fn to_vec1<T: From<f32>>(&self) -> Result<Vec<f32>> {
        Ok(self.data.clone())
    }
    
    /// Index select operation (simplified)
    pub fn index_select(&self, _indices: &SimpleTensor, _dim: usize) -> Result<SimpleTensor> {
        // Mock implementation - just return a copy
        Ok(self.clone())
    }
}

/// Device enum for CPU/GPU selection
#[derive(Debug, Clone, Copy)]
pub enum Device {
    Cpu,
}

/// Dummy DType enum (not used in mock implementation)
#[derive(Debug, Clone, Copy)]
pub enum DType {
    F32,
    F16,
    I32,
}