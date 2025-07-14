//! Model management for neural TTS with ONNX Runtime
//! Implements Kokoro TTS and other premium models

/// Smart model discovery system
pub mod discovery;
/// Model manager for downloading and loading TTS models
pub mod manager;
/// Model types and enums
pub mod types;

pub use discovery::{ModelDiscovery, KokoroModelFiles, ModelManifest};
pub use manager::ModelManager;
pub use types::{ModelId, ModelInfo, ModelConfig};