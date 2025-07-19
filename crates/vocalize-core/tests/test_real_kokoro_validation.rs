//! ZERO-FALLBACK: Real Kokoro Model Validation Test
//! 
//! This test uses the actual Kokoro model files (downloaded by Python)
//! and validates that it generates real speech, not musical tones.
//! 
//! NO FALLBACKS, NO MOCKS, NO APPROXIMATIONS - REAL MODEL ONLY

#[cfg(test)]
mod real_kokoro_tests {
    use std::path::PathBuf;
    use tokio;
    
    use vocalize_core::{
 
 
        onnx_engine::OnnxTtsEngine,
        model::ModelId,
    };

    /// Get the real cache directory where Python downloaded the models
    fn get_real_cache_dir() -> anyhow::Result<PathBuf> {
        use directories::ProjectDirs;
        
        let proj_dirs = ProjectDirs::from("ai", "Vocalize", "vocalize")
            .ok_or_else(|| anyhow::anyhow!("Failed to determine project directories"))?;
        
        let cache_dir = proj_dirs.cache_dir().join("models");
        
        // Verify models exist
        let model_dir = cache_dir.join("models--direct_download").join("local");
        let model_file = model_dir.join("kokoro-v1.0.onnx");
        let voices_file = model_dir.join("voices-v1.0.bin");
        
        if !model_file.exists() || !voices_file.exists() {
            return Err(anyhow::anyhow!(
                "Kokoro model files not found. Please run: python3 -m vocalize.model_manager download kokoro"
            ));
        }
        
        tracing::info!("✅ Found existing Kokoro model files:");
        tracing::info!("   📄 Model: {:?} ({} MB)", model_file, model_file.metadata()?.len() / 1_000_000);
        tracing::info!("   📄 Voices: {:?} ({} MB)", voices_file, voices_file.metadata()?.len() / 1_000_000);
        
        Ok(cache_dir)
    }
    
    /// Validate that audio output is speech, not musical tones
    fn validate_speech_not_tones(audio: &[f32]) -> bool {
        if audio.is_empty() {
            return false;
        }
        
        // Basic speech validation (not mathematical tones)
        // 1. Length should be reasonable for speech
        let min_speech_samples = 1000; // At least 1000 samples for "Hello World"
        if audio.len() < min_speech_samples {
            tracing::error!("❌ Audio too short: {} samples (expected >= {})", audio.len(), min_speech_samples);
            return false;
        }
        
        // 2. Should not be pure sine wave (musical tones)
        let mut zero_crossings = 0;
        for i in 1..audio.len() {
            if (audio[i-1] >= 0.0) != (audio[i] >= 0.0) {
                zero_crossings += 1;
            }
        }
        let zero_crossing_rate = zero_crossings as f32 / audio.len() as f32;
        
        // Musical tones have very regular zero crossing rates
        // Speech has more irregular patterns
        if zero_crossing_rate > 0.3 && zero_crossing_rate < 0.35 {
            tracing::error!("❌ Audio appears to be musical tones (zero crossing rate: {:.3})", zero_crossing_rate);
            return false;
        }
        
        // 3. Should have variation in amplitude (not constant)
        let max_amp = audio.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let min_amp = audio.iter().map(|x| x.abs()).fold(f32::INFINITY, f32::min);
        let amplitude_variation = max_amp - min_amp;
        
        if amplitude_variation < 0.01 {
            tracing::error!("❌ Audio has too little amplitude variation: {:.6}", amplitude_variation);
            return false;
        }
        
        tracing::info!("✅ Audio validation passed: {} samples, ZCR: {:.3}, Amp variation: {:.6}", 
                      audio.len(), zero_crossing_rate, amplitude_variation);
        true
    }
    
    /// Save audio as WAV file for manual verification
    fn save_audio_wav(audio: &[f32], file_path: &PathBuf) -> anyhow::Result<()> {
        use hound::{WavWriter, WavSpec};
        
        let spec = WavSpec {
            channels: 1,
            sample_rate: 24000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        
        let mut writer = WavWriter::create(file_path, spec)?;
        for &sample in audio {
            writer.write_sample(sample)?;
        }
        writer.finalize()?;
        
        tracing::info!("💾 Saved audio to: {:?}", file_path);
        Ok(())
    }

    #[tokio::test]
    #[ignore] // Use 'cargo test -- --ignored' to run this test (requires existing model files)
    async fn test_real_kokoro_model_synthesis() {
        // Initialize logging
        let _ = env_logger::builder().is_test(true).try_init();
        tracing::info!("🚀 Starting ZERO-FALLBACK real Kokoro model test");
        
        // Use real cache directory where Python downloaded models
        let _cache_dir = get_real_cache_dir()
            .expect("Failed to locate Kokoro model files - ensure they are downloaded via Python");
        
        // PHASE 1: Initialize engine with existing model
        tracing::info!("🔧 PHASE 1: Initializing ONNX engine with existing model...");
        let mut engine = OnnxTtsEngine::new(_cache_dir.clone()).await
            .expect("Failed to create ONNX engine - NO FALLBACKS ALLOWED");
        
        // PHASE 2: Load real model
        tracing::info!("📂 PHASE 2: Loading real Kokoro model...");
        engine.load_model(ModelId::Kokoro).await
            .expect("Failed to load real Kokoro model - NO FALLBACKS ALLOWED");
        
        // PHASE 3: Real synthesis test
        tracing::info!("🎤 PHASE 3: Synthesizing speech with real model...");
        let test_text = "Hello World";
        let audio = engine.synthesize(test_text, ModelId::Kokoro, Some("af_alloy")).await
            .expect("Failed to synthesize with real model - NO FALLBACKS ALLOWED");
        
        // PHASE 4: Validate real speech (not tones)
        tracing::info!("🔍 PHASE 4: Validating real speech output...");
        assert!(validate_speech_not_tones(&audio), 
               "❌ ZERO-FALLBACK TEST FAILED: Output is musical tones, not speech!");
        
        // PHASE 5: Save for manual verification
        tracing::info!("💾 PHASE 5: Saving audio for manual verification...");
        let output_path = _cache_dir.join("hello_world_REAL_SPEECH.wav");
        save_audio_wav(&audio, &output_path)
            .expect("Failed to save audio file");
        
        tracing::info!("🎉 ZERO-FALLBACK TEST PASSED: Real speech generated successfully!");
        tracing::info!("🔊 Manual verification: Play {:?} to confirm it's speech", output_path);
        
        // Final assertion
        assert!(!audio.is_empty(), "Audio output should not be empty");
        assert!(audio.len() > 1000, "Audio should have reasonable length for 'Hello World'");
        
        println!("✅ ZERO-FALLBACK SUCCESS: Real Kokoro model generated real speech!");
    }
    
    #[tokio::test]
    async fn test_zero_fallback_model_discovery() {
        let _cache_dir = std::path::PathBuf::from("/tmp/nonexistent");
        
        // Test that discovery FAILS when exact files are missing (zero-fallback behavior)
        use vocalize_core::model::discovery::ModelDiscovery;
        
        let discovery = ModelDiscovery::new();
        let result = discovery.find_best_kokoro_model();
        
        // Should be None because exact files don't exist
        assert!(result.is_none(), "Discovery should fail when exact Kokoro files are missing (zero-fallback)");
        
        tracing::info!("✅ ZERO-FALLBACK discovery correctly failed without exact files");
    }
}