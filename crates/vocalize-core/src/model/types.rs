// Model types and enums for TTS models

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Supported TTS model identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelId {
    /// Kokoro TTS - Default model (82MB, Apache 2.0)
    Kokoro,
    /// Chatterbox - Premium model (150MB, Apache 2.0)
    Chatterbox,
    /// Dia - Premium model (1.6GB, Apache 2.0)
    Dia,
}

impl ModelId {
    /// Get the default model (Kokoro TTS)
    pub fn default() -> Self {
        Self::Kokoro
    }
    
    /// Get model name as string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Kokoro => "kokoro",
            Self::Chatterbox => "chatterbox", 
            Self::Dia => "dia",
        }
    }
}

/// Model configuration and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model identifier
    pub id: ModelId,
    /// Human-readable model name
    pub name: String,
    /// Model description
    pub description: String,
    /// Model size in megabytes
    pub size_mb: u64,
    /// Software license
    pub license: String,
    /// Audio sample rate
    pub sample_rate: u32,
    /// HuggingFace repository ID
    pub repo_id: String,
    /// Required model files
    pub files: Vec<String>,
}

impl ModelInfo {
    /// Get Kokoro TTS model info (default) - 2025 optimized version
    pub fn kokoro() -> Self {
        Self {
            id: ModelId::Kokoro,
            name: "Kokoro TTS".to_string(),
            description: "2025 optimized neural TTS model (82M parameters)".to_string(),
            size_mb: 410,  // Combined model + voices size
            license: "Apache 2.0".to_string(),
            sample_rate: 24000,
            repo_id: "direct_download".to_string(),
            files: vec![
                "kokoro-v1.0.onnx".to_string(),  // 2025 working model file
                "voices-v1.0.bin".to_string(),   // Unified voice data
            ],
        }
    }
    
    /// Get Chatterbox model info (premium)
    pub fn chatterbox() -> Self {
        Self {
            id: ModelId::Chatterbox,
            name: "Chatterbox TTS".to_string(),
            description: "Fast neural TTS model (150MB)".to_string(),
            size_mb: 150,
            license: "Apache 2.0".to_string(),
            sample_rate: 22050,
            repo_id: "facebook/chatterbox-en".to_string(),
            files: vec![
                "model.onnx".to_string(),
                "tokenizer.json".to_string(),
            ],
        }
    }
    
    /// Get Dia model info (premium, high-quality)
    pub fn dia() -> Self {
        Self {
            id: ModelId::Dia,
            name: "Dia TTS".to_string(),
            description: "Premium neural TTS model (1.6GB)".to_string(),
            size_mb: 1600,
            license: "Apache 2.0".to_string(),
            sample_rate: 48000,
            repo_id: "microsoft/dia-en-large".to_string(),
            files: vec![
                "pytorch_model.bin".to_string(),
                "config.json".to_string(),
                "tokenizer.json".to_string(),
            ],
        }
    }
}

/// Model configuration for ONNX runtime
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Path to the ONNX model file
    pub model_path: PathBuf,
    /// Audio sample rate
    pub sample_rate: u32,
    /// Maximum text length for synthesis
    pub max_length: usize,
}

impl ModelConfig {
    /// Create a new model configuration
    pub fn new(model_path: PathBuf, sample_rate: u32) -> Self {
        Self {
            model_path,
            sample_rate,
            max_length: 1000, // Default max text length
        }
    }
}