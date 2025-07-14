// TDD tests for ONNX-based TTS Engine implementation
// Replaces the existing TTS engine with neural synthesis

#[cfg(test)]
mod onnx_tts_engine_tests {
    use tempfile::TempDir;
    
    #[test]
    fn test_onnx_tts_engine_creation() {
        let temp_dir = TempDir::new().unwrap();
        let cache_dir = temp_dir.path().to_path_buf();
        
        // This will fail until we implement OnnxTtsEngine
        let result = std::panic::catch_unwind(|| {
            use vocalize_core::onnx_engine::OnnxTtsEngine;
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let _engine = OnnxTtsEngine::new(cache_dir).await;
            });
        });
        assert!(result.is_ok(), "OnnxTtsEngine should be creatable");
    }
    
    #[test]
    fn test_onnx_tts_synthesis() {
        let temp_dir = TempDir::new().unwrap();
        let cache_dir = temp_dir.path().to_path_buf();
        
        let result = std::panic::catch_unwind(|| {
            use vocalize_core::onnx_engine::OnnxTtsEngine;
            use vocalize_core::model::ModelId;
            
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let mut engine = OnnxTtsEngine::new(cache_dir).await.unwrap();
                
                // Should synthesize using neural model
                let audio = engine.synthesize("Hello world", ModelId::Kokoro, Some("af_alloy")).await;
                assert!(audio.is_ok());
                
                let audio_data = audio.unwrap();
                assert!(!audio_data.is_empty());
                assert!(audio_data.len() > 1000); // Should produce reasonable audio length
            });
        });
        assert!(result.is_ok(), "OnnxTtsEngine should synthesize audio");
    }
    
    #[test]
    fn test_onnx_text_preprocessing() {
        let temp_dir = TempDir::new().unwrap();
        let cache_dir = temp_dir.path().to_path_buf();
        
        let result = std::panic::catch_unwind(|| {
            use vocalize_core::onnx_engine::OnnxTtsEngine;
            
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let engine = OnnxTtsEngine::new(cache_dir).await.unwrap();
                
                // Should preprocess text (normalize, tokenize)
                let processed = engine.preprocess_text("Hello, World! How are you?");
                assert!(!processed.is_empty());
                assert_ne!(processed, "Hello, World! How are you?"); // Should be different after processing
            });
        });
        assert!(result.is_ok(), "OnnxTtsEngine should preprocess text");
    }
    
    #[test]
    fn test_onnx_model_loading() {
        let temp_dir = TempDir::new().unwrap();
        let cache_dir = temp_dir.path().to_path_buf();
        
        let result = std::panic::catch_unwind(|| {
            use vocalize_core::onnx_engine::OnnxTtsEngine;
            use vocalize_core::model::ModelId;
            
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let mut engine = OnnxTtsEngine::new(cache_dir).await.unwrap();
                
                // Should load and switch models
                let load_result = engine.load_model(ModelId::Kokoro).await;
                assert!(load_result.is_ok());
                
                // Should track current model
                assert_eq!(engine.current_model(), Some(ModelId::Kokoro));
            });
        });
        assert!(result.is_ok(), "OnnxTtsEngine should load models");
    }
    
    #[test]
    fn test_onnx_audio_postprocessing() {
        let temp_dir = TempDir::new().unwrap();
        let cache_dir = temp_dir.path().to_path_buf();
        
        let result = std::panic::catch_unwind(|| {
            use vocalize_core::onnx_engine::OnnxTtsEngine;
            
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let engine = OnnxTtsEngine::new(cache_dir).await.unwrap();
                
                // Should postprocess raw model output
                let raw_output = vec![0.1f32, 0.2, -0.1, 0.5, -0.3];
                let processed = engine.postprocess_audio(&raw_output);
                
                assert_eq!(processed.len(), raw_output.len());
                // Should normalize audio to proper range
                assert!(processed.iter().all(|&x| x >= -1.0 && x <= 1.0));
            });
        });
        assert!(result.is_ok(), "OnnxTtsEngine should postprocess audio");
    }
}