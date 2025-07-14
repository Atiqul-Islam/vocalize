//! Integration tests for vocalize-core crate

use vocalize_core::{
    AudioDevice, AudioFormat, AudioWriter, TtsEngine, Voice, VoiceManager, VoiceStyle, Gender,
    EncodingSettings, SynthesisParams, AudioConfig, PlaybackState,
};
use std::time::Duration;
use tempfile::NamedTempFile;

#[tokio::test]
async fn test_full_tts_pipeline() {
    // Create TTS engine
    let engine = TtsEngine::new().await.expect("Should create TTS engine");
    
    // Create voice
    let voice = Voice::default();
    let params = SynthesisParams::new(voice);
    
    // Synthesize audio
    let audio = engine.synthesize("Hello, world!", &params).await
        .expect("Should synthesize audio");
    
    assert!(!audio.is_empty());
    assert!(audio.iter().all(|&s| s.abs() <= 1.0));
}

#[tokio::test]
async fn test_voice_manager_integration() {
    let manager = VoiceManager::new();
    
    // Test getting available voices
    let voices = manager.get_available_voices();
    assert!(!voices.is_empty());
    
    // Test getting voice by ID
    let voice = manager.get_voice("af_bella").expect("Should find bella voice");
    assert_eq!(voice.id, "af_bella");
    assert_eq!(voice.gender, Gender::Female);
    
    // Test filtering by language
    let en_voices = manager.get_voices_by_language("en-US");
    assert!(!en_voices.is_empty());
    assert!(en_voices.iter().all(|v| v.supports_language("en-US")));
    
    // Test filtering by gender
    let female_voices = manager.get_voices_by_gender(Gender::Female);
    assert!(!female_voices.is_empty());
    assert!(female_voices.iter().all(|v| v.gender == Gender::Female));
}

#[tokio::test]
async fn test_audio_writer_integration() {
    let writer = AudioWriter::new();
    let audio_data = vec![0.1, 0.2, -0.1, -0.2, 0.0]; // Simple test audio
    let settings = EncodingSettings::new(24000, 1);
    
    // Test WAV writing
    let temp_file = NamedTempFile::with_suffix(".wav").expect("Should create temp file");
    let result = writer.write_file(&audio_data, temp_file.path(), AudioFormat::Wav, Some(settings.clone())).await;
    assert!(result.is_ok());
    
    // Verify file was created and has content
    let metadata = std::fs::metadata(temp_file.path()).expect("File should exist");
    assert!(metadata.len() > 44); // WAV header is 44 bytes
    
    // Test auto-detection
    let temp_file2 = NamedTempFile::with_suffix(".wav").expect("Should create temp file");
    let result2 = writer.write_file_auto(&audio_data, temp_file2.path(), Some(settings)).await;
    assert!(result2.is_ok());
}

#[test]
fn test_voice_configuration() {
    // Test voice creation and validation
    let voice = Voice::new(
        "test_voice".to_string(),
        "Test Voice".to_string(),
        "en-US".to_string(),
        Gender::Male,
        VoiceStyle::Professional,
    )
    .with_description("Test voice for integration testing".to_string())
    .with_sample_rate(48000);
    
    assert!(voice.validate().is_ok());
    assert_eq!(voice.sample_rate, 48000);
    assert!(voice.supports_language("en-US"));
    assert!(voice.supports_language("en"));
    assert!(!voice.supports_language("es"));
}

#[test]
fn test_synthesis_params_configuration() {
    let voice = Voice::default();
    let params = SynthesisParams::new(voice)
        .with_speed(1.5).expect("Valid speed")
        .with_pitch(0.2).expect("Valid pitch")
        .with_streaming(2048);
    
    assert!(params.validate().is_ok());
    assert_eq!(params.speed, 1.5);
    assert_eq!(params.pitch, 0.2);
    assert!(params.streaming);
    assert_eq!(params.chunk_size, 2048);
}

#[test]
fn test_encoding_settings_configuration() {
    let settings = EncodingSettings::new(48000, 2)
        .with_bit_depth(24)
        .with_quality(0.8)
        .with_variable_bitrate();
    
    assert!(settings.validate().is_ok());
    assert_eq!(settings.sample_rate, 48000);
    assert_eq!(settings.channels, 2);
    assert_eq!(settings.bit_depth, 24);
    assert_eq!(settings.quality, Some(0.8));
    assert!(settings.variable_bitrate);
}

#[test]
fn test_audio_format_detection() {
    // Test format detection from extensions
    assert_eq!(AudioFormat::from_extension("wav").unwrap(), AudioFormat::Wav);
    assert_eq!(AudioFormat::from_extension("MP3").unwrap(), AudioFormat::Mp3);
    assert_eq!(AudioFormat::from_extension("flac").unwrap(), AudioFormat::Flac);
    assert_eq!(AudioFormat::from_extension("OGG").unwrap(), AudioFormat::Ogg);
    
    // Test format detection from paths
    assert_eq!(AudioFormat::from_path("test.wav").unwrap(), AudioFormat::Wav);
    assert_eq!(AudioFormat::from_path("/path/to/audio.mp3").unwrap(), AudioFormat::Mp3);
    
    // Test format properties
    assert!(!AudioFormat::Wav.is_lossy());
    assert!(AudioFormat::Mp3.is_lossy());
    assert!(!AudioFormat::Flac.is_lossy());
    assert!(AudioFormat::Ogg.is_lossy());
}

#[tokio::test]
async fn test_tts_engine_streaming() {
    let engine = TtsEngine::new().await.expect("Should create TTS engine");
    let voice = Voice::default();
    let params = SynthesisParams::new(voice).with_streaming(512);
    
    // Test streaming synthesis
    let chunks = engine.synthesize_streaming("This is a longer text that should be split into multiple chunks for streaming synthesis", &params).await
        .expect("Should synthesize streaming audio");
    
    assert!(chunks.len() > 1, "Should produce multiple chunks");
    
    for chunk in &chunks {
        assert!(!chunk.is_empty(), "Each chunk should have audio data");
        assert!(chunk.iter().all(|&s| s.abs() <= 1.0), "All samples should be in valid range");
    }
}

#[tokio::test]
async fn test_tts_engine_different_voices() {
    let engine = TtsEngine::new().await.expect("Should create TTS engine");
    
    // Test different voice configurations
    let voices = vec![
        Voice::new("male".to_string(), "Male Voice".to_string(), "en-US".to_string(), Gender::Male, VoiceStyle::Professional),
        Voice::new("female".to_string(), "Female Voice".to_string(), "en-US".to_string(), Gender::Female, VoiceStyle::Natural),
        Voice::new("neutral".to_string(), "Neutral Voice".to_string(), "en-US".to_string(), Gender::Neutral, VoiceStyle::Calm),
    ];
    
    for voice in voices {
        let params = SynthesisParams::new(voice);
        let audio = engine.synthesize("Test audio with different voice", &params).await
            .expect("Should synthesize with any voice");
        
        assert!(!audio.is_empty());
        assert!(audio.iter().all(|&s| s.abs() <= 1.0));
    }
}

#[tokio::test]
async fn test_error_handling() {
    let engine = TtsEngine::new().await.expect("Should create TTS engine");
    let voice = Voice::default();
    
    // Test empty text error
    let params = SynthesisParams::new(voice.clone());
    let result = engine.synthesize("", &params).await;
    assert!(result.is_err());
    
    // Test invalid speed
    let invalid_params = SynthesisParams::new(voice.clone()).with_speed(5.0);
    assert!(invalid_params.is_err());
    
    // Test invalid pitch
    let invalid_params = SynthesisParams::new(voice).with_pitch(2.0);
    assert!(invalid_params.is_err());
}

#[test]
fn test_audio_config() {
    let config = AudioConfig::default();
    assert_eq!(config.device_id, None);
    assert_eq!(config.sample_rate, vocalize_core::DEFAULT_SAMPLE_RATE);
    assert_eq!(config.channels, vocalize_core::DEFAULT_CHANNELS);
    assert_eq!(config.buffer_size, 1024);
    assert_eq!(config.latency, Duration::from_millis(50));
}

#[test]
fn test_playback_state() {
    assert_eq!(PlaybackState::Stopped.to_string(), "Stopped");
    assert_eq!(PlaybackState::Playing.to_string(), "Playing");
    assert_eq!(PlaybackState::Paused.to_string(), "Paused");
    assert_eq!(PlaybackState::Error.to_string(), "Error");
    
    assert_eq!(PlaybackState::Stopped, PlaybackState::Stopped);
    assert_ne!(PlaybackState::Playing, PlaybackState::Stopped);
}

#[tokio::test]
async fn test_tts_engine_stats() {
    let engine = TtsEngine::new().await.expect("Should create TTS engine");
    let stats = engine.get_stats().await;
    
    assert!(stats.initialized);
    assert_eq!(stats.sample_rate, vocalize_core::DEFAULT_SAMPLE_RATE);
    assert_eq!(stats.max_text_length, vocalize_core::MAX_TEXT_LENGTH);
}

#[tokio::test]
async fn test_tts_engine_cache_management() {
    let engine = TtsEngine::new().await.expect("Should create TTS engine");
    assert!(engine.is_initialized().await);
    
    // Test preloading
    let result = engine.preload_models().await;
    assert!(result.is_ok());
    
    // Test cache clearing
    let result = engine.clear_cache().await;
    assert!(result.is_ok());
    assert!(!engine.is_initialized().await);
}

#[test]
fn test_constants() {
    assert_eq!(vocalize_core::DEFAULT_SAMPLE_RATE, 24_000);
    assert_eq!(vocalize_core::DEFAULT_CHANNELS, 1);
    assert_eq!(vocalize_core::MAX_TEXT_LENGTH, 100_000);
    assert!(!vocalize_core::VERSION.is_empty());
}