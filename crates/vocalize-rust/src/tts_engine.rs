//! Python bindings for TTS engine

use pyo3::prelude::*;
use std::collections::HashMap;
use vocalize_core::SynthesisParams;

use crate::error::IntoPyResult;
use crate::voice_manager::PyVoice;
use crate::runtime_manager::{RuntimeManager, LazyTtsEngine};

/// Python wrapper for SynthesisParams
#[pyclass(name = "SynthesisParams")]
#[derive(Debug, Clone)]
pub struct PySynthesisParams {
    inner: SynthesisParams,
}

impl PySynthesisParams {
    pub fn new(params: SynthesisParams) -> Self {
        Self { inner: params }
    }

    pub fn inner(&self) -> &SynthesisParams {
        &self.inner
    }

    pub fn into_inner(self) -> SynthesisParams {
        self.inner
    }
}

#[pymethods]
impl PySynthesisParams {
    #[new]
    fn py_new(voice: PyVoice) -> Self {
        let params = SynthesisParams::new(voice.into_inner());
        Self::new(params)
    }

    #[getter]
    fn voice(&self) -> PyVoice {
        PyVoice::new(self.inner.voice.clone())
    }

    #[getter]
    fn speed(&self) -> f32 {
        self.inner.speed
    }

    #[getter]
    fn pitch(&self) -> f32 {
        self.inner.pitch
    }

    #[getter]
    fn streaming(&self) -> bool {
        self.inner.streaming
    }

    #[getter]
    fn chunk_size(&self) -> usize {
        self.inner.chunk_size
    }

    fn with_speed(&self, speed: f32) -> PyResult<PySynthesisParams> {
        let params = self.inner.clone().with_speed(speed).into_py_result()?;
        Ok(Self::new(params))
    }

    fn with_pitch(&self, pitch: f32) -> PyResult<PySynthesisParams> {
        let params = self.inner.clone().with_pitch(pitch).into_py_result()?;
        Ok(Self::new(params))
    }

    fn with_streaming(&self, chunk_size: usize) -> PySynthesisParams {
        let params = self.inner.clone().with_streaming(chunk_size);
        Self::new(params)
    }

    fn without_streaming(&self) -> PySynthesisParams {
        let mut params = self.inner.clone();
        params.streaming = false;
        params.chunk_size = 0;
        Self::new(params)
    }

    fn __repr__(&self) -> String {
        format!(
            "SynthesisParams(voice='{}', speed={}, pitch={}, streaming={})",
            self.inner.voice.id,
            self.inner.speed,
            self.inner.pitch,
            self.inner.streaming
        )
    }

    fn to_dict(&self) -> HashMap<String, String> {
        let mut dict = HashMap::new();
        dict.insert("voice_id".to_string(), self.inner.voice.id.clone());
        dict.insert("speed".to_string(), self.inner.speed.to_string());
        dict.insert("pitch".to_string(), self.inner.pitch.to_string());
        dict.insert("streaming".to_string(), self.inner.streaming.to_string());
        dict.insert("chunk_size".to_string(), self.inner.chunk_size.to_string());
        dict
    }
}

/// Python wrapper for TtsEngine
#[pyclass(name = "TtsEngine")]
#[derive(Debug)]
pub struct PyTtsEngine {
    lazy_engine: LazyTtsEngine,
}

impl PyTtsEngine {
    pub fn new() -> Self {
        Self {
            lazy_engine: LazyTtsEngine::new(),
        }
    }
}

#[pymethods]
impl PyTtsEngine {
    #[new]
    fn py_new() -> PyResult<Self> {
        // Initialize global runtime if not already done
        RuntimeManager::initialize()?;
        
        // Create lazy engine (doesn't initialize TTS engine yet)
        Ok(PyTtsEngine::new())
    }
    
    /// Initialize the TTS engine (lazy initialization)
    fn initialize(&self) -> PyResult<()> {
        self.lazy_engine.get_or_init()?;
        Ok(())
    }

    /// Synthesize text to audio (using real TTS engine)
    fn synthesize_sync(&self, text: String, params: &PySynthesisParams) -> PyResult<Vec<f32>> {
        // Get the TTS engine (initialize if needed)
        let engine = self.lazy_engine.get_or_init()?;
        
        // Convert Python params to Rust params
        let rust_params = params.inner();
        
        // Perform synthesis using the global runtime
        let audio = RuntimeManager::block_on(async {
            engine.synthesize(&text, rust_params).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
            format!("Synthesis failed: {}", e)
        ))?;
        
        audio.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
            format!("Audio synthesis failed: {}", e)
        ))
    }

    /// Check if the engine is ready
    fn is_ready(&self) -> bool {
        self.lazy_engine.is_initialized()
    }
    
    /// Get engine statistics
    fn get_stats(&self) -> PyResult<HashMap<String, String>> {
        let engine = self.lazy_engine.get_or_init()?;
        
        let stats = RuntimeManager::block_on(async {
            engine.get_stats().await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
            format!("Failed to get stats: {}", e)
        ))?;
        
        let mut result = HashMap::new();
        result.insert("initialized".to_string(), stats.initialized.to_string());
        result.insert("device".to_string(), format!("{:?}", stats.device));
        result.insert("sample_rate".to_string(), stats.sample_rate.to_string());
        result.insert("installed_models".to_string(), stats.installed_model_count.to_string());
        result.insert("active_model".to_string(), 
                     stats.active_model.unwrap_or_else(|| "None".to_string()));
        
        Ok(result)
    }

    fn __repr__(&self) -> String {
        "TtsEngine()".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voice_manager::{PyVoice, PyGender, PyVoiceStyle};
    use vocalize_core::{Voice, Gender, VoiceStyle};
    
    // Helper function to create a test voice since Voice::default() is no longer available
    fn create_test_voice() -> PyVoice {
        let voice = Voice::new(
            "af_alloy".to_string(),
            "Alloy".to_string(),
            "en-US".to_string(),
            Gender::Male,
            VoiceStyle::Natural,
        );
        PyVoice::new(voice)
    }
    use crate::voice_manager::{PyGender, PyVoiceStyle};

    #[test]
    fn test_py_synthesis_params_creation() {
        let voice = PyVoice::py_new(
            "test".to_string(),
            "Test".to_string(),
            "en-US".to_string(),
            PyGender::Female,
            PyVoiceStyle::Natural,
        );
        
        let params = PySynthesisParams::py_new(voice);
        assert_eq!(params.voice().id(), "test");
        assert_eq!(params.speed(), None);
        assert_eq!(params.pitch(), None);
        assert_eq!(params.streaming_chunk_size(), None);
    }

    #[test]
    fn test_py_synthesis_params_with_speed() {
        let voice = PyVoice::default();
        let params = PySynthesisParams::py_new(voice);
        
        // Valid speed
        let with_speed = params.with_speed(1.5);
        assert!(with_speed.is_ok());
        assert_eq!(with_speed.unwrap().speed(), Some(1.5));
        
        // Invalid speed
        let invalid_speed = params.with_speed(0.05);
        assert!(invalid_speed.is_err());
    }

    #[test]
    fn test_py_synthesis_params_with_pitch() {
        let voice = PyVoice::default();
        let params = PySynthesisParams::py_new(voice);
        
        // Valid pitch
        let with_pitch = params.with_pitch(0.5);
        assert!(with_pitch.is_ok());
        assert_eq!(with_pitch.unwrap().pitch(), Some(0.5));
        
        // Invalid pitch
        let invalid_pitch = params.with_pitch(-1.5);
        assert!(invalid_pitch.is_err());
    }

    #[test]
    fn test_py_synthesis_params_streaming() {
        let voice = PyVoice::default();
        let params = PySynthesisParams::py_new(voice);
        
        let with_streaming = params.with_streaming(1024);
        assert_eq!(with_streaming.streaming_chunk_size(), Some(1024));
        
        let without_streaming = with_streaming.without_streaming();
        assert_eq!(without_streaming.streaming_chunk_size(), None);
    }

    #[test]
    fn test_py_synthesis_params_to_dict() {
        let voice = create_test_voice();
        let params = PySynthesisParams::py_new(voice)
            .with_speed(1.2).unwrap()
            .with_pitch(0.1).unwrap()
            .with_streaming(512);
        
        let dict = params.to_dict();
        assert_eq!(dict.get("voice_id"), Some(&"af_alloy".to_string()));
        assert_eq!(dict.get("speed"), Some(&"1.2".to_string()));
        assert_eq!(dict.get("pitch"), Some(&"0.1".to_string()));
        assert_eq!(dict.get("streaming_chunk_size"), Some(&"512".to_string()));
    }

    #[test]
    fn test_py_synthesis_params_repr() {
        let voice = create_test_voice();
        let params = PySynthesisParams::py_new(voice);
        let repr = params.__repr__();
        
        assert!(repr.contains("SynthesisParams"));
        assert!(repr.contains("af_alloy"));
    }

    #[test]
    fn test_py_tts_engine_creation() {
        let engine = PyTtsEngine::py_new();
        assert!(engine.is_ok());
        assert_eq!(engine.unwrap().__repr__(), "TtsEngine()");
    }

    #[test]
    fn test_py_tts_engine_synthesize() {
        let engine = PyTtsEngine::py_new().unwrap();
        let voice = create_test_voice();
        let params = PySynthesisParams::py_new(voice);
        
        let result = engine.synthesize_sync("Hello".to_string(), &params);
        assert!(result.is_ok());
        
        let audio = result.unwrap();
        assert!(!audio.is_empty());
    }
}