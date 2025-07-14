// Test to ensure ONNX Runtime and dependencies are properly configured

#[cfg(test)]
mod dependency_tests {
    #[test]
    fn test_onnx_runtime_available() {
        // This will fail until we add `ort` dependency
        let result = std::panic::catch_unwind(|| {
            // Try to use ort crate - test that we can create a session builder
            use ort::session::builder::SessionBuilder;
            let _builder = SessionBuilder::new();
        });
        assert!(result.is_ok(), "ONNX Runtime (ort) should be available");
    }
    
    #[test]
    fn test_hf_hub_available() {
        // This will fail until we add `hf-hub` dependency
        let result = std::panic::catch_unwind(|| {
            // Try to use hf-hub crate
            let _api = hf_hub::api::sync::Api::new();
        });
        assert!(result.is_ok(), "HuggingFace Hub should be available");
    }
    
    #[test]
    fn test_tokenizers_available() {
        // This will fail until we add `tokenizers` dependency
        let result = std::panic::catch_unwind(|| {
            // Try to create a simple tokenizer
            use tokenizers::models::bpe::BPE;
            let _model = BPE::default();
        });
        assert!(result.is_ok(), "Tokenizers should be available");
    }
    
    #[test]
    fn test_reqwest_available() {
        // Test reqwest for downloads
        let result = std::panic::catch_unwind(|| {
            let _client = reqwest::Client::new();
        });
        assert!(result.is_ok(), "Reqwest should be available");
    }
    
    #[test]
    fn test_sha2_available() {
        // Test sha2 for model verification
        let result = std::panic::catch_unwind(|| {
            use sha2::Digest;
            let _hasher = sha2::Sha256::new();
        });
        assert!(result.is_ok(), "SHA2 should be available");
    }
}