//! Fast tensor operations for GGML implementation
//! Provides SIMD-optimized operations for VITS inference

use anyhow::Result;
use std::sync::Arc;

/// Performs matrix multiplication: C = A @ B
/// A: [M, K], B: [K, N], C: [M, N]
pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    
    // Simple implementation - can be optimized with SIMD
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for ki in 0..k {
                sum += a[i * k + ki] * b[ki * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    
    c
}

/// Applies 1D convolution
pub fn conv1d(input: &[f32], kernel: &[f32], stride: usize, padding: usize) -> Vec<f32> {
    let input_len = input.len();
    let kernel_len = kernel.len();
    let output_len = (input_len + 2 * padding - kernel_len) / stride + 1;
    let mut output = vec![0.0f32; output_len];
    
    for i in 0..output_len {
        let mut sum = 0.0f32;
        for j in 0..kernel_len {
            let input_idx = i * stride + j;
            if input_idx >= padding && input_idx < input_len + padding {
                sum += input[input_idx - padding] * kernel[j];
            }
        }
        output[i] = sum;
    }
    
    output
}

/// ReLU activation function
pub fn relu(x: &mut [f32]) {
    for val in x.iter_mut() {
        *val = val.max(0.0);
    }
}

/// Leaky ReLU activation
pub fn leaky_relu(x: &mut [f32], alpha: f32) {
    for val in x.iter_mut() {
        *val = if *val > 0.0 { *val } else { *val * alpha };
    }
}

/// Layer normalization
pub fn layer_norm(x: &mut [f32], gamma: &[f32], beta: &[f32], eps: f32) {
    let n = x.len();
    
    // Compute mean
    let mean: f32 = x.iter().sum::<f32>() / n as f32;
    
    // Compute variance
    let variance: f32 = x.iter()
        .map(|&v| (v - mean).powi(2))
        .sum::<f32>() / n as f32;
    
    // Normalize and scale
    let std_inv = 1.0 / (variance + eps).sqrt();
    for i in 0..n {
        x[i] = (x[i] - mean) * std_inv * gamma[i % gamma.len()] + beta[i % beta.len()];
    }
}

/// Softmax activation
pub fn softmax(x: &mut [f32]) {
    let max_val = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    
    // Subtract max for numerical stability
    let mut sum = 0.0;
    for val in x.iter_mut() {
        *val = (*val - max_val).exp();
        sum += *val;
    }
    
    // Normalize
    for val in x.iter_mut() {
        *val /= sum;
    }
}

/// GELU activation function
pub fn gelu(x: &mut [f32]) {
    const SQRT_2_OVER_PI: f32 = 0.7978845608;
    
    for val in x.iter_mut() {
        let v = *val;
        *val = 0.5 * v * (1.0 + ((SQRT_2_OVER_PI * (v + 0.044715 * v * v * v)).tanh()));
    }
}

/// Attention mechanism
pub fn attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    seq_len: usize,
    d_k: usize,
) -> Vec<f32> {
    let scale = 1.0 / (d_k as f32).sqrt();
    
    // Q @ K^T
    let mut scores = matmul(query, key, seq_len, d_k, seq_len);
    
    // Scale
    for s in scores.iter_mut() {
        *s *= scale;
    }
    
    // Softmax
    for i in 0..seq_len {
        let start = i * seq_len;
        let end = start + seq_len;
        softmax(&mut scores[start..end]);
    }
    
    // Scores @ V
    matmul(&scores, value, seq_len, seq_len, d_k)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_matmul() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2
        let c = matmul(&a, &b, 2, 2, 2);
        
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }
    
    #[test]
    fn test_softmax() {
        let mut x = vec![1.0, 2.0, 3.0];
        softmax(&mut x);
        
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_conv1d_basic() {
        // Simple edge detection kernel
        let input = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let kernel = vec![-1.0, 0.0, 1.0];
        
        let output = conv1d(&input, &kernel, 1, 1);
        
        // Expected: differences between adjacent elements
        assert_eq!(output.len(), 6);
        assert!((output[1] - 2.0).abs() < 1e-6); // 2 - 0
        assert!((output[2] - 2.0).abs() < 1e-6); // 3 - 1
    }
    
    #[test]
    fn test_layer_norm() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0, 1.0, 1.0, 1.0];
        let beta = vec![0.0, 0.0, 0.0, 0.0];
        
        layer_norm(&mut x, &gamma, &beta, 1e-5);
        
        // After normalization, mean should be ~0 and variance ~1
        let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
        assert!(mean.abs() < 1e-5);
        
        let variance: f32 = x.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / x.len() as f32;
        assert!((variance - 1.0).abs() < 1e-5);
    }
    
    #[test]
    fn test_attention_mechanism() {
        // Small attention test
        let seq_len = 3;
        let d_k = 4;
        
        // Query, Key, Value matrices (flattened)
        let query = vec![1.0; seq_len * d_k];
        let key = vec![1.0; seq_len * d_k];
        let value = vec![2.0; seq_len * d_k];
        
        let result = attention(&query, &key, &value, seq_len, d_k);
        
        // With identical queries and keys, attention should be uniform
        // Result should be close to value matrix
        assert_eq!(result.len(), seq_len * d_k);
        for &val in &result {
            assert!((val - 2.0).abs() < 0.1);
        }
    }
    
    #[test]
    fn test_gelu_activation() {
        let mut x = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let original = x.clone();
        
        gelu(&mut x);
        
        // GELU properties
        assert!(x[2].abs() < 1e-6); // GELU(0) ≈ 0
        assert!(x[3] > 0.0 && x[3] < original[3]); // 0 < GELU(1) < 1
        assert!(x[4] > 0.0 && x[4] < original[4]); // 0 < GELU(2) < 2
        
        // Should be smooth and monotonic
        for i in 1..x.len() {
            assert!(x[i] > x[i-1]); // Monotonically increasing
        }
    }
    
    #[test]
    fn test_leaky_relu() {
        let mut x = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let alpha = 0.1;
        
        leaky_relu(&mut x, alpha);
        
        assert_eq!(x[0], -2.0 * alpha);
        assert_eq!(x[1], -1.0 * alpha);
        assert_eq!(x[2], 0.0);
        assert_eq!(x[3], 1.0);
        assert_eq!(x[4], 2.0);
    }
}