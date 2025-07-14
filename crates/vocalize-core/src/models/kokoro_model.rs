//! Kokoro TTS model implementation using ONNX Runtime
//!
//! This module provides a production-ready implementation of the Kokoro TTS model
//! using ONNX Runtime for high-performance neural speech synthesis.

use crate::models::TtsModel;
use crate::onnx_engine::OnnxTtsEngine;
use crate::model::ModelId;
use crate::{VocalizeResult, VocalizeError, SynthesisParams, AudioData};
use std::sync::{Arc, Mutex};
use std::path::PathBuf;

/// Kokoro TTS model implementation using ONNX Runtime
#[derive(Debug)]
pub struct KokoroModel {
    /// Model identifier
    id: String,
    /// Model name
    name: String,
    /// Whether the model is currently loaded
    loaded: bool,
    /// ONNX engine for inference
    onnx_engine: Option<Arc<Mutex<OnnxTtsEngine>>>,
    /// Cache directory for model files
    cache_dir: PathBuf,
}

impl KokoroModel {
    /// Create a new Kokoro model instance
    pub fn new(cache_dir: PathBuf) -> Self {
        Self {
            id: "kokoro".to_string(),
            name: "Kokoro TTS".to_string(),
            loaded: false,
            onnx_engine: None,
            cache_dir,
        }
    }
    
    /// Get the path to the cached model files
    fn get_model_paths(&self) -> VocalizeResult<(PathBuf, PathBuf)> {
        let model_dir = self.cache_dir
            .join("models--direct_download")
            .join("local");
            
        let model_file = model_dir.join("kokoro-v1.0.onnx");
        let voices_file = model_dir.join("voices-v1.0.bin");
        
        if !model_file.exists() {
            return Err(VocalizeError::synthesis(
                "Kokoro model file not found. Please download it first using: vocalize models download kokoro"
            ));
        }
        
        if !voices_file.exists() {
            return Err(VocalizeError::synthesis(
                "Kokoro voices file not found. Please download it first using: vocalize models download kokoro"
            ));
        }
        
        Ok((model_file, voices_file))
    }
}

impl TtsModel for KokoroModel {
    fn model_id(&self) -> &str {
        &self.id
    }
    
    fn model_name(&self) -> &str {
        &self.name
    }
    
    fn is_loaded(&self) -> bool {
        self.loaded && self.onnx_engine.is_some()
    }
    
    fn load(&mut self) -> VocalizeResult<()> {
        if self.loaded {
            return Ok(());
        }
        
        tracing::info!("Loading Kokoro TTS model from cache");
        
        // Validate model files exist
        let (_model_file, _voices_file) = self.get_model_paths()?;
        
        // Create a runtime for async operations
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| VocalizeError::synthesis(&format!("Failed to create async runtime: {}", e)))?;
        
        // Create and initialize ONNX engine asynchronously
        let mut onnx_engine = rt.block_on(async {
            OnnxTtsEngine::new(self.cache_dir.clone()).await
        }).map_err(|e| VocalizeError::synthesis(&format!("Failed to create ONNX engine: {}", e)))?;
        
        // Load the Kokoro model
        rt.block_on(async {
            onnx_engine.load_model(ModelId::Kokoro).await
        }).map_err(|e| VocalizeError::synthesis(&format!("Failed to load Kokoro model: {}", e)))?;
        
        // Store the loaded engine
        self.onnx_engine = Some(Arc::new(Mutex::new(onnx_engine)));
        self.loaded = true;
        
        tracing::info!("Successfully loaded Kokoro TTS model");
        Ok(())
    }
    
    fn unload(&mut self) {
        if self.loaded {
            tracing::info!("Unloading Kokoro TTS model");
            self.onnx_engine = None;
            self.loaded = false;
        }
    }
    
    fn synthesize(&self, text: &str, voice_id: &str, _params: &SynthesisParams) -> VocalizeResult<AudioData> {
        if !self.is_loaded() {
            return Err(VocalizeError::synthesis("Kokoro model is not loaded"));
        }
        
        let onnx_engine = self.onnx_engine.as_ref()
            .ok_or_else(|| VocalizeError::synthesis("ONNX engine not available"))?;
        
        // Use the existing ONNX engine for synthesis
        let audio_data = {
            let mut engine = onnx_engine.lock()
                .map_err(|e| VocalizeError::synthesis(&format!("Failed to acquire engine lock: {}", e)))?;
            
            // Create a runtime for async operation
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| VocalizeError::synthesis(&format!("Failed to create async runtime: {}", e)))?;
            
            rt.block_on(async {
                engine.synthesize(text, ModelId::Kokoro, Some(voice_id)).await
            })?
        };
        
        tracing::debug!("Kokoro synthesis completed: {} samples generated", audio_data.len());
        Ok(audio_data)
    }
    
    fn supported_voices(&self) -> Vec<String> {
        // Return the standard Kokoro voices based on research
        vec![
            // American Female voices
            "af_heart".to_string(),
            "af_alloy".to_string(), 
            "af_aoede".to_string(),
            "af_bella".to_string(),
            "af_jessica".to_string(),
            "af_kore".to_string(),
            "af_nicole".to_string(),
            "af_nova".to_string(),
            "af_river".to_string(),
            "af_sarah".to_string(),
            "af_sky".to_string(),
            
            // American Male voices
            "am_adam".to_string(),
            "am_echo".to_string(),
            "am_eric".to_string(),
            "am_fenrir".to_string(),
            "am_liam".to_string(),
            "am_michael".to_string(),
            "am_onyx".to_string(),
            "am_puck".to_string(),
            "am_santa".to_string(),
            
            // British Female voices
            "bf_alice".to_string(),
            "bf_emma".to_string(),
            "bf_isabella".to_string(),
            "bf_lily".to_string(),
            
            // British Male voices
            "bm_daniel".to_string(),
            "bm_fable".to_string(),
            "bm_george".to_string(),
            "bm_lewis".to_string(),
        ]
    }
}