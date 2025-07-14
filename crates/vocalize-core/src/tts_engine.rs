//! TTS engine implementation with model management system.
//!
//! This module provides a production-ready TTS engine that uses the ModelRegistry
//! system for managing different TTS models. The engine supports auto-installation
//! of default models and provides a clean interface for synthesis.

use crate::error::{VocalizeError, VocalizeResult};
use crate::voice_manager::Voice;
use crate::models::ModelRegistry;
use std::sync::Arc;
use std::path::PathBuf;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Audio data type - 32-bit floating point samples
pub type AudioData = Vec<f32>;

/// TTS engine configuration
#[derive(Debug, Clone)]
pub struct TtsConfig {
    /// Directory for model cache storage
    pub model_cache_dir: PathBuf,
    /// Device to use for inference (CPU/GPU)
    pub device: TtsDevice,
    /// Maximum text length to process
    pub max_text_length: usize,
    /// Default sample rate
    pub sample_rate: u32,
    /// Enable auto-installation of default model
    pub auto_install_default: bool,
    /// Default model ID to use
    pub default_model_id: String,
}

impl Default for TtsConfig {
    fn default() -> Self {
        let home_dir = get_home_dir();
        let cache_dir = home_dir.join(".vocalize");
        
        Self {
            model_cache_dir: cache_dir,
            device: TtsDevice::Cpu,
            max_text_length: crate::MAX_TEXT_LENGTH,
            sample_rate: crate::DEFAULT_SAMPLE_RATE,
            auto_install_default: true,
            default_model_id: "kokoro".to_string(),
        }
    }
}

/// Device type for TTS inference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtsDevice {
    /// Use CPU for inference
    Cpu,
    /// Use GPU for inference (if available)
    Gpu,
    /// Automatically select the best device
    Auto,
}

/// TTS synthesis parameters
#[derive(Debug, Clone)]
pub struct SynthesisParams {
    /// Voice to use for synthesis
    pub voice: Voice,
    /// Speed multiplier (0.1 to 3.0)
    pub speed: f32,
    /// Pitch adjustment (-1.0 to 1.0)
    pub pitch: f32,
    /// Enable streaming synthesis
    pub streaming: bool,
    /// Chunk size for streaming (in samples)
    pub chunk_size: usize,
}

impl SynthesisParams {
    /// Create new synthesis parameters with a voice
    #[must_use]
    pub fn new(voice: Voice) -> Self {
        Self {
            speed: voice.speed,
            pitch: voice.pitch,
            voice,
            streaming: false,
            chunk_size: 1024,
        }
    }

    /// Set speed multiplier
    ///
    /// # Errors
    ///
    /// Returns an error if speed is not in valid range (0.1 to 3.0)
    pub fn with_speed(mut self, speed: f32) -> VocalizeResult<Self> {
        if !(0.1..=3.0).contains(&speed) {
            return Err(VocalizeError::invalid_input(format!(
                "Speed must be between 0.1 and 3.0, got {speed}"
            )));
        }
        self.speed = speed;
        Ok(self)
    }

    /// Set pitch adjustment
    ///
    /// # Errors
    ///
    /// Returns an error if pitch is not in valid range (-1.0 to 1.0)
    pub fn with_pitch(mut self, pitch: f32) -> VocalizeResult<Self> {
        if !(-1.0..=1.0).contains(&pitch) {
            return Err(VocalizeError::invalid_input(format!(
                "Pitch must be between -1.0 and 1.0, got {pitch}"
            )));
        }
        self.pitch = pitch;
        Ok(self)
    }

    /// Enable streaming synthesis
    #[must_use]
    pub fn with_streaming(mut self, chunk_size: usize) -> Self {
        self.streaming = true;
        self.chunk_size = chunk_size;
        self
    }

    /// Validate synthesis parameters
    pub fn validate(&self) -> VocalizeResult<()> {
        self.voice.validate()?;

        if !(0.1..=3.0).contains(&self.speed) {
            return Err(VocalizeError::invalid_input(format!(
                "Speed must be between 0.1 and 3.0, got {}",
                self.speed
            )));
        }

        if !(-1.0..=1.0).contains(&self.pitch) {
            return Err(VocalizeError::invalid_input(format!(
                "Pitch must be between -1.0 and 1.0, got {}",
                self.pitch
            )));
        }

        if self.chunk_size == 0 {
            return Err(VocalizeError::invalid_input(
                "Chunk size must be greater than 0".to_string(),
            ));
        }

        Ok(())
    }
}

/// High-performance TTS engine with model management
#[derive(Debug)]
pub struct TtsEngine {
    config: TtsConfig,
    model_registry: Arc<RwLock<ModelRegistry>>,
    initialized: Arc<RwLock<bool>>,
}

impl TtsEngine {
    /// Create a new TTS engine with default configuration
    /// 
    /// # Errors
    /// 
    /// Returns an error if the model registry cannot be created or if
    /// initialization fails.
    pub async fn new() -> VocalizeResult<Self> {
        Self::with_config(TtsConfig::default()).await
    }


    /// Create a new TTS engine with custom configuration
    /// 
    /// # Errors
    /// 
    /// Returns an error if the model registry cannot be created or if
    /// initialization fails.
    pub async fn with_config(config: TtsConfig) -> VocalizeResult<Self> {
        info!("Creating TTS engine with config: {:?}", config);

        let registry = ModelRegistry::new(&config.model_cache_dir)?;
        
        let engine = Self {
            config,
            model_registry: Arc::new(RwLock::new(registry)),
            initialized: Arc::new(RwLock::new(false)),
        };

        engine.initialize().await?;
        Ok(engine)
    }

    /// Initialize the TTS engine and ensure a model is available
    async fn initialize(&self) -> VocalizeResult<()> {
        let mut initialized = self.initialized.write().await;
        if *initialized {
            debug!("TTS engine already initialized");
            return Ok(());
        }

        info!("Initializing TTS engine...");
        
        // Check if we have any models installed
        let mut registry = self.model_registry.write().await;
        
        if !registry.has_any_model() && self.config.auto_install_default {
            info!("No TTS models installed. Installing default model: {}", self.config.default_model_id);
            registry.install_model(&self.config.default_model_id).await?;
        }
        
        // If we still have no models, return an error
        if !registry.has_any_model() {
            return Err(VocalizeError::model(
                "No TTS models available. Please install a model first.".to_string()
            ));
        }
        
        // Load a default model if none is active
        if registry.get_active_model().is_err() {
            let model_id = {
                let installed_models = registry.get_installed_models();
                installed_models.first().map(|m| m.id.clone())
            };
            if let Some(model_id) = model_id {
                info!("Loading model: {}", model_id);
                registry.load_model(&model_id)?;
            }
        }
        
        *initialized = true;
        info!("TTS engine initialized successfully");
        
        Ok(())
    }

    /// Check if the engine is initialized
    pub async fn is_initialized(&self) -> bool {
        *self.initialized.read().await
    }

    /// Synthesize text to audio
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The engine is not initialized
    /// - The text is empty or too long
    /// - The synthesis parameters are invalid
    /// - No model is available
    /// - The synthesis process fails
    pub async fn synthesize(&self, text: &str, params: &SynthesisParams) -> VocalizeResult<AudioData> {
        self.validate_input(text, params).await?;

        debug!("Synthesizing text: {} characters", text.len());
        
        let mut registry = self.model_registry.write().await;
        
        // Ensure we have an active model
        if registry.get_active_model().is_err() {
            // Try to auto-install default model if enabled
            if self.config.auto_install_default {
                warn!("No active model found. Installing default model: {}", self.config.default_model_id);
                registry.install_model(&self.config.default_model_id).await?;
                registry.load_model(&self.config.default_model_id)?;
            } else {
                return Err(VocalizeError::synthesis("No TTS model available"));
            }
        }
        
        let model = registry.get_active_model()?;
        let audio = model.synthesize(text, &params.voice.id, params)?;

        info!("Successfully synthesized {} samples", audio.len());
        Ok(audio)
    }

    /// Synthesize text to audio with streaming
    ///
    /// # Errors
    ///
    /// Returns an error if the synthesis fails or parameters are invalid
    pub async fn synthesize_streaming(
        &self,
        text: &str,
        params: &SynthesisParams,
    ) -> VocalizeResult<Vec<AudioData>> {
        self.validate_input(text, params).await?;

        if !params.streaming {
            // If streaming is not enabled, return single chunk
            let audio = self.synthesize(text, params).await?;
            return Ok(vec![audio]);
        }

        debug!("Streaming synthesis for {} characters", text.len());

        // Split text into chunks for streaming
        let words: Vec<&str> = text.split_whitespace().collect();
        let chunk_size = (words.len() / 4).max(1); // Divide into ~4 chunks
        
        let mut chunks = Vec::new();
        for word_chunk in words.chunks(chunk_size) {
            let chunk_text = word_chunk.join(" ");
            if !chunk_text.is_empty() {
                let audio = self.synthesize(&chunk_text, params).await?;
                chunks.push(audio);
            }
        }

        info!("Generated {} audio chunks", chunks.len());
        Ok(chunks)
    }

    /// Install a model by ID
    /// 
    /// # Errors
    /// 
    /// Returns an error if the model ID is not found or installation fails.
    pub async fn install_model(&self, model_id: &str) -> VocalizeResult<()> {
        let mut registry = self.model_registry.write().await;
        registry.install_model(model_id).await
    }
    
    /// Remove an installed model
    /// 
    /// # Errors
    /// 
    /// Returns an error if the model is not installed or removal fails.
    pub async fn remove_model(&self, model_id: &str) -> VocalizeResult<()> {
        let mut registry = self.model_registry.write().await;
        registry.remove_model(model_id)
    }
    
    /// Set the active model
    /// 
    /// # Errors
    /// 
    /// Returns an error if the model is not installed or loading fails.
    pub async fn set_active_model(&self, model_id: &str) -> VocalizeResult<()> {
        let mut registry = self.model_registry.write().await;
        registry.load_model(model_id)?;
        registry.set_default_model(model_id)
    }
    
    /// List all available models that can be installed
    pub async fn list_available_models(&self) -> Vec<crate::models::ModelInfo> {
        ModelRegistry::get_available_models()
    }
    
    /// List installed models
    pub async fn list_installed_models(&self) -> Vec<crate::models::ModelInfo> {
        let registry = self.model_registry.read().await;
        registry.get_installed_models().into_iter().cloned().collect()
    }

    /// Validate input parameters
    async fn validate_input(&self, text: &str, params: &SynthesisParams) -> VocalizeResult<()> {
        if !self.is_initialized().await {
            return Err(VocalizeError::synthesis("TTS engine not initialized"));
        }

        if text.is_empty() {
            return Err(VocalizeError::invalid_input("Text cannot be empty"));
        }

        if text.len() > self.config.max_text_length {
            return Err(VocalizeError::invalid_input(format!(
                "Text length {} exceeds maximum of {}",
                text.len(),
                self.config.max_text_length
            )));
        }

        params.validate()?;

        Ok(())
    }

    /// Get engine configuration
    #[must_use]
    pub fn get_config(&self) -> &TtsConfig {
        &self.config
    }

    /// Get engine statistics
    #[must_use]
    pub async fn get_stats(&self) -> TtsStats {
        let registry = self.model_registry.read().await;
        let installed_models = registry.get_installed_models();
        
        TtsStats {
            initialized: self.is_initialized().await,
            device: self.config.device,
            sample_rate: self.config.sample_rate,
            max_text_length: self.config.max_text_length,
            installed_model_count: installed_models.len(),
            active_model: registry.active_model.clone(),
        }
    }

    /// Preload models for faster synthesis
    pub async fn preload_models(&self) -> VocalizeResult<()> {
        if !self.is_initialized().await {
            self.initialize().await?;
        }
        
        info!("Models preloaded successfully");
        Ok(())
    }

    /// Clear model cache to free memory
    pub async fn clear_cache(&self) -> VocalizeResult<()> {
        debug!("Clearing model cache");
        
        let mut registry = self.model_registry.write().await;
        
        // Unload all models
        for model in registry.loaded_models.values_mut() {
            model.unload();
        }
        registry.loaded_models.clear();
        registry.active_model = None;
        
        let mut initialized = self.initialized.write().await;
        *initialized = false;
        
        info!("Model cache cleared");
        Ok(())
    }
}

/// TTS engine statistics
#[derive(Debug, Clone)]
pub struct TtsStats {
    /// Whether the engine is initialized
    pub initialized: bool,
    /// Device being used for inference
    pub device: TtsDevice,
    /// Current sample rate
    pub sample_rate: u32,
    /// Maximum text length
    pub max_text_length: usize,
    /// Number of installed models
    pub installed_model_count: usize,
    /// Currently active model ID
    pub active_model: Option<String>,
}

impl Default for TtsStats {
    fn default() -> Self {
        Self {
            initialized: false,
            device: TtsDevice::Cpu,
            sample_rate: crate::DEFAULT_SAMPLE_RATE,
            max_text_length: crate::MAX_TEXT_LENGTH,
            installed_model_count: 0,
            active_model: None,
        }
    }
}

// Cross-platform home directory detection using dirs crate

fn get_home_dir() -> PathBuf {
    #[cfg(test)]
    {
        PathBuf::from("/tmp")
    }
    #[cfg(not(test))]
    {
        // Use standard cross-platform home directory detection
        if let Some(home) = std::env::var_os("HOME") {
            PathBuf::from(home)
        } else if let Some(userprofile) = std::env::var_os("USERPROFILE") {
            PathBuf::from(userprofile)
        } else if let Some(homepath) = std::env::var_os("HOMEPATH") {
            if let Some(homedrive) = std::env::var_os("HOMEDRIVE") {
                PathBuf::from(homedrive).join(homepath)
            } else {
                PathBuf::from(homepath)
            }
        } else {
            // Last resort fallback
            PathBuf::from(".")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voice_manager::Voice;
    use tempfile::TempDir;

    fn create_test_config(temp_dir: &TempDir) -> TtsConfig {
        TtsConfig {
            model_cache_dir: temp_dir.path().to_path_buf(),
            auto_install_default: false, // Disable auto-install for most tests
            ..TtsConfig::default()
        }
    }

    #[test]
    fn test_tts_config_default() {
        let config = TtsConfig::default();
        assert_eq!(config.device, TtsDevice::Cpu);
        assert_eq!(config.max_text_length, crate::MAX_TEXT_LENGTH);
        assert_eq!(config.sample_rate, crate::DEFAULT_SAMPLE_RATE);
        assert!(config.auto_install_default);
        assert_eq!(config.default_model_id, "kokoro");
    }

    #[test]
    fn test_tts_device() {
        assert_eq!(TtsDevice::Cpu, TtsDevice::Cpu);
        assert_ne!(TtsDevice::Cpu, TtsDevice::Gpu);
    }

    #[test]
    fn test_synthesis_params_new() {
        let voice = Voice::default();
        let params = SynthesisParams::new(voice.clone());
        
        assert_eq!(params.voice, voice);
        assert_eq!(params.speed, voice.speed);
        assert_eq!(params.pitch, voice.pitch);
        assert!(!params.streaming);
        assert_eq!(params.chunk_size, 1024);
    }

    #[test]
    fn test_synthesis_params_with_speed_valid() {
        let voice = Voice::default();
        let params = SynthesisParams::new(voice)
            .with_speed(1.5)
            .expect("Valid speed should work");
        
        assert_eq!(params.speed, 1.5);
    }

    #[test]
    fn test_synthesis_params_with_speed_invalid() {
        let voice = Voice::default();
        let params = SynthesisParams::new(voice);
        
        assert!(params.clone().with_speed(0.05).is_err());
        assert!(params.with_speed(5.0).is_err());
    }

    #[test]
    fn test_synthesis_params_with_pitch_valid() {
        let voice = Voice::default();
        let params = SynthesisParams::new(voice)
            .with_pitch(0.5)
            .expect("Valid pitch should work");
        
        assert_eq!(params.pitch, 0.5);
    }

    #[test]
    fn test_synthesis_params_with_pitch_invalid() {
        let voice = Voice::default();
        let params = SynthesisParams::new(voice);
        
        assert!(params.clone().with_pitch(-1.5).is_err());
        assert!(params.with_pitch(2.0).is_err());
    }

    #[test]
    fn test_synthesis_params_with_streaming() {
        let voice = Voice::default();
        let params = SynthesisParams::new(voice)
            .with_streaming(2048);
        
        assert!(params.streaming);
        assert_eq!(params.chunk_size, 2048);
    }

    #[test]
    fn test_synthesis_params_validation() {
        let voice = Voice::default();
        let params = SynthesisParams::new(voice);
        assert!(params.validate().is_ok());
        
        // Invalid speed
        let mut params = SynthesisParams::new(Voice::default());
        params.speed = 0.05;
        assert!(params.validate().is_err());
        
        // Invalid pitch
        let mut params = SynthesisParams::new(Voice::default());
        params.pitch = 2.0;
        assert!(params.validate().is_err());
        
        // Invalid chunk size
        let mut params = SynthesisParams::new(Voice::default());
        params.chunk_size = 0;
        assert!(params.validate().is_err());
    }

    #[tokio::test]
    async fn test_tts_engine_creation_no_auto_install() {
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config(&temp_dir);
        
        let result = TtsEngine::with_config(config).await;
        assert!(result.is_err()); // Should fail because no models and auto-install disabled
    }

    #[tokio::test]
    async fn test_tts_engine_creation_with_auto_install() {
        let temp_dir = TempDir::new().unwrap();
        let config = TtsConfig {
            model_cache_dir: temp_dir.path().to_path_buf(),
            auto_install_default: true,
            ..TtsConfig::default()
        };
        
        let engine = TtsEngine::with_config(config).await.unwrap();
        assert!(engine.is_initialized().await);
        
        // Should have installed and loaded default model
        let stats = engine.get_stats().await;
        assert_eq!(stats.installed_model_count, 1);
        assert!(stats.active_model.is_some());
    }

    #[tokio::test]
    async fn test_tts_engine_synthesis_with_mock_model() {
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config(&temp_dir);
        
        // Test with mock model - this will fail initially (expected for TDD)
        let result = TtsEngine::with_config(config).await;
        assert!(result.is_err()); // Should fail because no models and auto-install disabled
        
        // Test with auto-install enabled
        let config = TtsConfig {
            model_cache_dir: temp_dir.path().to_path_buf(),
            auto_install_default: true,
            default_model_id: "kokoro".to_string(),
            ..TtsConfig::default()
        };
        
        let engine = TtsEngine::with_config(config).await.unwrap();
        assert!(engine.is_initialized().await);
        
        // Test synthesis with the installed model
        let voice = Voice::default();
        let params = SynthesisParams::new(voice);
        let result = engine.synthesize("Hello world", &params).await;
        assert!(result.is_ok());
        
        let audio = result.unwrap();
        assert!(!audio.is_empty());
        assert!(audio.iter().all(|&sample| sample.abs() <= 1.0));
    }

    #[tokio::test]
    async fn test_tts_engine_model_management() {
        let temp_dir = TempDir::new().unwrap();
        let config = TtsConfig {
            model_cache_dir: temp_dir.path().to_path_buf(),
            auto_install_default: false,
            ..TtsConfig::default()
        };
        
        // Create engine without any models initially
        let result = TtsEngine::with_config(config).await;
        assert!(result.is_err()); // Should fail because no models
        
        // Create engine with auto-install for testing model management
        let config = TtsConfig {
            model_cache_dir: temp_dir.path().to_path_buf(),
            auto_install_default: true,
            default_model_id: "kokoro".to_string(),
            ..TtsConfig::default()
        };
        
        let engine = TtsEngine::with_config(config).await.unwrap();
        assert!(engine.is_initialized().await);
        
        // Test listing available models
        let available = engine.list_available_models().await;
        assert!(!available.is_empty());
        assert!(available.iter().any(|m| m.id == "kokoro"));
        
        // Test listing installed models
        let installed = engine.list_installed_models().await;
        assert!(!installed.is_empty());
        assert_eq!(installed.len(), 1);
        assert_eq!(installed[0].id, "kokoro");
        
        // Test setting active model
        let result = engine.set_active_model("kokoro").await;
        assert!(result.is_ok());
        
        // Test installing another model (should fail because kokoro is the only one)
        let result = engine.install_model("kokoro").await;
        assert!(result.is_ok()); // Should succeed (already installed)
        
        // Test removing model
        let result = engine.remove_model("kokoro").await;
        assert!(result.is_ok());
        
        // Verify model is removed
        let installed = engine.list_installed_models().await;
        assert!(installed.is_empty());
    }

    #[tokio::test]
    async fn test_tts_engine_validation() {
        let temp_dir = TempDir::new().unwrap();
        let config = TtsConfig {
            model_cache_dir: temp_dir.path().to_path_buf(),
            auto_install_default: true,
            ..TtsConfig::default()
        };
        
        let engine = TtsEngine::with_config(config).await.unwrap();
        let voice = Voice::default();
        let params = SynthesisParams::new(voice);
        
        // Test empty text validation
        let result = engine.synthesize("", &params).await;
        assert!(result.is_err());
        
        // Test too long text validation
        let long_text = "a".repeat(engine.config.max_text_length + 1);
        let result = engine.synthesize(&long_text, &params).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_tts_engine_get_stats() {
        let temp_dir = TempDir::new().unwrap();
        let config = TtsConfig {
            model_cache_dir: temp_dir.path().to_path_buf(),
            auto_install_default: true,
            ..TtsConfig::default()
        };
        
        let engine = TtsEngine::with_config(config).await.unwrap();
        let stats = engine.get_stats().await;
        
        assert!(stats.initialized);
        assert_eq!(stats.device, TtsDevice::Cpu);
        assert_eq!(stats.sample_rate, crate::DEFAULT_SAMPLE_RATE);
        assert_eq!(stats.max_text_length, crate::MAX_TEXT_LENGTH);
        assert!(stats.installed_model_count > 0);
    }

    #[tokio::test]
    async fn test_tts_engine_preload_models() {
        let temp_dir = TempDir::new().unwrap();
        let config = TtsConfig {
            model_cache_dir: temp_dir.path().to_path_buf(),
            auto_install_default: true,
            ..TtsConfig::default()
        };
        
        let engine = TtsEngine::with_config(config).await.unwrap();
        let result = engine.preload_models().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_tts_engine_clear_cache() {
        let temp_dir = TempDir::new().unwrap();
        let config = TtsConfig {
            model_cache_dir: temp_dir.path().to_path_buf(),
            auto_install_default: true,
            ..TtsConfig::default()
        };
        
        let engine = TtsEngine::with_config(config).await.unwrap();
        assert!(engine.is_initialized().await);
        
        let result = engine.clear_cache().await;
        assert!(result.is_ok());
        assert!(!engine.is_initialized().await);
    }

    #[tokio::test]
    async fn test_tts_engine_list_models() {
        let temp_dir = TempDir::new().unwrap();
        let config = TtsConfig {
            model_cache_dir: temp_dir.path().to_path_buf(),
            auto_install_default: true,
            default_model_id: "kokoro".to_string(),
            ..TtsConfig::default()
        };
        
        // Create engine with auto-install to test listing
        let engine = TtsEngine::with_config(config).await.unwrap();
        
        // Test listing available models
        let available = engine.list_available_models().await;
        assert!(!available.is_empty());
        assert!(available.iter().any(|m| m.id == "kokoro"));
        
        // Test listing installed models
        let installed = engine.list_installed_models().await;
        assert!(!installed.is_empty());
        assert_eq!(installed.len(), 1);
        assert_eq!(installed[0].id, "kokoro");
        assert!(installed[0].installed);
        
        // Test with engine without models initially
        let temp_dir2 = TempDir::new().unwrap();
        let config2 = TtsConfig {
            model_cache_dir: temp_dir2.path().to_path_buf(),
            auto_install_default: false,
            ..TtsConfig::default()
        };
        
        // This should fail since we don't have models and auto-install is disabled
        let result = TtsEngine::with_config(config2).await;
        assert!(result.is_err());
    }
}