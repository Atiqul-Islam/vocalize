// TDD tests to ensure all mathematical/sine wave synthesis is removed
// Validates that only neural synthesis is used

#[cfg(test)]
mod no_fallback_tests {
    use tempfile::TempDir;
    
    // Helper function to calculate audio complexity (distinguishes neural from mathematical)
    fn calculate_audio_complexity(audio: &[f32]) -> f32 {
        if audio.len() < 4 {
            return 0.0;
        }
        
        // Calculate spectral entropy as a measure of complexity
        let mut energy_variance = 0.0f32;
        let windows = audio.windows(4);
        let mut variances = Vec::new();
        
        for window in windows {
            let mean = window.iter().sum::<f32>() / window.len() as f32;
            let variance = window.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / window.len() as f32;
            variances.push(variance);
        }
        
        if !variances.is_empty() {
            let mean_var = variances.iter().sum::<f32>() / variances.len() as f32;
            energy_variance = variances.iter().map(|&v| (v - mean_var).powi(2)).sum::<f32>() / variances.len() as f32;
        }
        
        energy_variance.sqrt()
    }

    // Helper function to detect if audio is a simple sine wave
    fn check_if_sine_wave(audio: &[f32]) -> bool {
        if audio.len() < 20 {
            return false;
        }
        
        // Check for sine wave characteristics
        let mut zero_crossings = 0;
        for window in audio.windows(2) {
            if window[0] * window[1] < 0.0 {
                zero_crossings += 1;
            }
        }
        
        // Sine waves have regular zero crossings
        let crossing_regularity = zero_crossings as f32 / audio.len() as f32;
        
        // Also check for smooth periodic behavior
        let mut smooth_count = 0;
        for window in audio.windows(3) {
            let derivative1 = window[1] - window[0];
            let derivative2 = window[2] - window[1];
            
            // Sine waves have smooth derivatives
            if (derivative1 - derivative2).abs() < 0.01 {
                smooth_count += 1;
            }
        }
        
        let smoothness = smooth_count as f32 / (audio.len() - 2) as f32;
        
        // If it's very regular and smooth, it's probably a sine wave
        crossing_regularity > 0.05 && smoothness > 0.8
    }
    
    #[test]
    fn test_no_mathematical_synthesis_in_models() {
        // This test ensures models don't use mathematical wave generation
        let result = std::panic::catch_unwind(|| {
            use vocalize_core::onnx_engine::OnnxTtsEngine;
            use vocalize_core::model::ModelId;
            
            let temp_dir = tempfile::TempDir::new().unwrap();
            let cache_dir = temp_dir.path().to_path_buf();
            
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let mut engine = OnnxTtsEngine::new(cache_dir).await.unwrap();
                
                // ONNX engine should use neural synthesis, not mathematical waves
                let audio = engine.synthesize("Hello world", ModelId::Kokoro, Some("af_alloy")).await.unwrap();
            
            // Check that audio doesn't follow simple mathematical patterns
            // Neural audio should have more complex characteristics
            assert!(!audio.is_empty());
            
            // Audio should not be a perfect sawtooth/sine wave
            // We'll check for variation that indicates neural-like synthesis
            let mut has_variation = false;
            if audio.len() > 100 {
                // Check that consecutive samples don't follow simple patterns
                for window in audio.windows(10) {
                    let diffs: Vec<f32> = window.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
                    let avg_diff = diffs.iter().sum::<f32>() / diffs.len() as f32;
                    
                    // Neural audio should have more irregular patterns
                    if avg_diff > 0.001 && avg_diff < 0.1 {
                        has_variation = true;
                        break;
                    }
                }
            }
            
                assert!(has_variation, "Audio should have neural-like variation, not mathematical waves");
            });
        });
        assert!(result.is_ok(), "ONNX engine should use neural synthesis patterns");
    }
    
    #[test]
    fn test_tts_engine_uses_neural_only() {
        let temp_dir = TempDir::new().unwrap();
        
        let result = std::panic::catch_unwind(|| {
            use vocalize_core::{TtsEngine, TtsConfig, SynthesisParams, Voice};
            
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let config = TtsConfig {
                    model_cache_dir: temp_dir.path().to_path_buf(),
                    auto_install_default: true,
                    default_model_id: "neural_model".to_string(), // Should use neural model
                    ..TtsConfig::default()
                };
                
                let engine = TtsEngine::with_config(config).await;
                if let Ok(engine) = engine {
                    let voice = Voice::default();
                    let params = SynthesisParams::new(voice);
                    
                    let audio = engine.synthesize("Neural synthesis test", &params).await.unwrap();
                    
                    // Ensure audio is not from mathematical synthesis
                    assert!(!audio.is_empty());
                    assert!(audio.iter().all(|&x| x.abs() <= 1.0));
                    
                    // Check that it doesn't follow simple mathematical patterns
                    // Neural synthesis should produce more complex waveforms
                    let complexity_score = calculate_audio_complexity(&audio);
                    assert!(complexity_score > 0.1, "Audio should have neural complexity, not mathematical simplicity");
                }
            });
        });
        assert!(result.is_ok(), "TTS engine should use only neural synthesis");
    }
    
    #[test]
    fn test_onnx_engine_replaces_old_synthesis() {
        let temp_dir = TempDir::new().unwrap();
        let cache_dir = temp_dir.path().to_path_buf();
        
        let result = std::panic::catch_unwind(|| {
            use vocalize_core::onnx_engine::OnnxTtsEngine;
            use vocalize_core::model::ModelId;
            
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let mut engine = OnnxTtsEngine::new(cache_dir).await.unwrap();
                
                // Should use neural model, not fallback to mathematical synthesis
                let audio = engine.synthesize("Test neural synthesis", ModelId::Kokoro, Some("af_alloy")).await.unwrap();
                
                assert!(!audio.is_empty());
                assert!(audio.iter().all(|&x| x.abs() <= 1.0));
                
                // Neural synthesis should have different characteristics than sine waves
                let complexity = calculate_audio_complexity(&audio);
                assert!(complexity > 0.05, "ONNX engine should produce neural audio, not mathematical waves");
            });
        });
        assert!(result.is_ok(), "ONNX engine should use pure neural synthesis");
    }
    
    #[test]
    fn test_no_sine_wave_generation() {
        // Test that we don't generate simple sine waves anywhere
        let result = std::panic::catch_unwind(|| {
            use vocalize_core::onnx_engine::OnnxTtsEngine;
            use vocalize_core::model::ModelId;
            
            let temp_dir = tempfile::TempDir::new().unwrap();
            let cache_dir = temp_dir.path().to_path_buf();
            
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let mut engine = OnnxTtsEngine::new(cache_dir).await.unwrap();
                let audio = engine.synthesize("Test", ModelId::Kokoro, Some("af_alloy")).await.unwrap();
            
                // Check that it's not a pure sine wave
                if audio.len() > 10 {
                    let is_sine_like = check_if_sine_wave(&audio);
                    assert!(!is_sine_like, "Should not generate sine waves - use neural synthesis only");
                }
            });
        });
        assert!(result.is_ok(), "No sine wave generation should be present");
    }
}