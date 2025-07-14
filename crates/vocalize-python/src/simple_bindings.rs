//! Simple Python bindings for core TTS functionality

use pyo3::prelude::*;
use vocalize_core::{TtsEngine, SynthesisParams, VoiceManager, AudioWriter, AudioFormat, EncodingSettings};

/// Simple TTS synthesis function
#[pyfunction]
fn synthesize_text(text: String, voice_id: Option<String>, speed: Option<f32>, pitch: Option<f32>) -> PyResult<Vec<f32>> {
    // Create TTS engine with default config
    let config = vocalize_core::TtsConfig::default();
    let engine = TtsEngine::new_blocking(config)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to create TTS engine: {}", e)))?;
    
    // Get voice
    let voice_manager = VoiceManager::new();
    let voice_id = voice_id.unwrap_or_else(|| "af_bella".to_string());
    let voice = voice_manager.get_voice(&voice_id)
        .unwrap_or_else(|_| voice_manager.get_default_voice());
    
    // Create synthesis parameters
    let mut params = SynthesisParams::new(voice);
    if let Some(s) = speed {
        params = params.with_speed(s)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid speed: {}", e)))?;
    }
    if let Some(p) = pitch {
        params = params.with_pitch(p)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid pitch: {}", e)))?;
    }
    
    // Synthesize audio
    match engine.synthesize_blocking(&text, &params) {
        Ok(audio_data) => Ok(audio_data),
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("TTS synthesis failed: {}", e))),
    }
}

/// Get list of available voices
#[pyfunction]
fn list_voices() -> PyResult<Vec<(String, String, String, String)>> {
    let voice_manager = VoiceManager::new();
    let voices = voice_manager.get_available_voices();
    
    let mut result = Vec::new();
    for voice in voices {
        result.push((
            voice.id.clone(),
            voice.name.clone(),
            voice.gender.to_string(),
            voice.language.clone(),
        ));
    }
    
    Ok(result)
}

/// Save audio data to a file
#[pyfunction] 
fn save_audio(audio_data: Vec<f32>, output_path: String, format: Option<String>) -> PyResult<()> {
    let format_str = format.unwrap_or_else(|| "wav".to_string());
    let audio_format = match format_str.as_str() {
        "wav" => AudioFormat::Wav,
        "mp3" => AudioFormat::Mp3,
        "flac" => AudioFormat::Flac,
        "ogg" => AudioFormat::Ogg,
        _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Unsupported format: {}", format_str))),
    };
    
    let writer = AudioWriter::new();
    let settings = EncodingSettings::default();
    
    // Convert to async and run
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        writer.write_file(&audio_data, &output_path, audio_format, Some(settings)).await
    }).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to save audio: {}", e)))?;
    
    Ok(())
}

/// Python module for simple TTS functionality
#[pymodule]
fn _core(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(synthesize_text, m)?)?;
    m.add_function(wrap_pyfunction!(list_voices, m)?)?;
    m.add_function(wrap_pyfunction!(save_audio, m)?)?;
    
    // Add constants
    m.add("DEFAULT_SAMPLE_RATE", vocalize_core::DEFAULT_SAMPLE_RATE)?;
    m.add("DEFAULT_CHANNELS", vocalize_core::DEFAULT_CHANNELS)?;
    m.add("VERSION", vocalize_core::VERSION)?;
    
    Ok(())
}