//! Model management system for TTS engines
//!
//! This module provides a pluggable architecture for managing different TTS models.
//! Models can be installed, removed, and switched via CLI commands.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};
use crate::error::{VocalizeError, VocalizeResult};
use crate::{SynthesisParams, AudioData};

pub mod kokoro_model;

/// Information about a TTS model
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelInfo {
    /// Unique identifier for the model
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Model version
    pub version: String,
    /// Model size in bytes
    pub size: usize,
    /// Download URL for the model
    pub download_url: String,
    /// License type (e.g., "MIT", "Apache-2.0")
    pub license: String,
    /// Whether the model is currently installed
    pub installed: bool,
    /// Local installation path
    pub install_path: PathBuf,
    /// Supported languages
    pub supported_languages: Vec<String>,
    /// Supported voice IDs
    pub supported_voices: Vec<String>,
}

/// Trait that all TTS models must implement
pub trait TtsModel: Send + Sync + std::fmt::Debug {
    /// Get the model's unique identifier
    fn model_id(&self) -> &str;
    
    /// Get the model's human-readable name
    fn model_name(&self) -> &str;
    
    /// Check if the model is currently loaded in memory
    fn is_loaded(&self) -> bool;
    
    /// Load the model into memory for synthesis
    /// 
    /// # Errors
    /// 
    /// Returns an error if the model files cannot be loaded or if there's
    /// insufficient memory.
    fn load(&mut self) -> VocalizeResult<()>;
    
    /// Unload the model from memory to free resources
    fn unload(&mut self);
    
    /// Synthesize text to audio using this model
    /// 
    /// # Arguments
    /// 
    /// * `text` - The text to synthesize
    /// * `voice_id` - The voice ID to use for synthesis
    /// * `params` - Synthesis parameters (speed, pitch, etc.)
    /// 
    /// # Errors
    /// 
    /// Returns an error if the model is not loaded, the text is invalid,
    /// or synthesis fails.
    fn synthesize(&self, text: &str, voice_id: &str, params: &SynthesisParams) -> VocalizeResult<AudioData>;
    
    /// Get the list of voice IDs supported by this model
    fn supported_voices(&self) -> Vec<String>;
}

/// Registry for managing installed and available TTS models
#[derive(Debug)]
pub struct ModelRegistry {
    /// Currently installed models
    installed_models: HashMap<String, ModelInfo>,
    /// Currently loaded models in memory
    pub loaded_models: HashMap<String, Box<dyn TtsModel>>,
    /// The currently active model ID
    pub active_model: Option<String>,
    /// Path to the registry file
    registry_path: PathBuf,
    /// Base directory for model storage
    cache_dir: PathBuf,
}

impl ModelRegistry {
    /// Create a new model registry
    /// 
    /// # Arguments
    /// 
    /// * `cache_dir` - Directory where models will be stored
    pub fn new(cache_dir: &Path) -> VocalizeResult<Self> {
        let registry_path = cache_dir.join("models.json");
        let cache_dir = cache_dir.to_path_buf();
        
        // Ensure cache directory exists
        std::fs::create_dir_all(&cache_dir)?;
        
        let mut registry = Self {
            installed_models: HashMap::new(),
            loaded_models: HashMap::new(),
            active_model: None,
            registry_path,
            cache_dir,
        };
        
        // Load existing registry if it exists
        registry.load_registry()?;
        
        // Auto-detect cached Kokoro model from Python downloads
        registry.detect_cached_kokoro_model()?;
        
        Ok(registry)
    }

    
    /// Get the list of all available models that can be installed
    pub fn get_available_models() -> Vec<ModelInfo> {
        vec![
            ModelInfo {
                id: "kokoro".to_string(),
                name: "Kokoro TTS".to_string(),
                version: "v1.0".to_string(),
                size: 410_000_000, // ~410MB (310MB model + 26MB voices)
                download_url: "direct_download".to_string(), // Managed by Python
                license: "Apache 2.0".to_string(),
                installed: false,
                install_path: PathBuf::new(),
                supported_languages: vec![
                    "en-US".to_string(), 
                    "en-GB".to_string(),
                    "ja-JP".to_string(),
                    "zh-CN".to_string()
                ],
                supported_voices: vec![
                    "af_heart".to_string(),
                    "af_alloy".to_string(),
                    "af_bella".to_string(),
                    "af_sarah".to_string(),
                    "am_adam".to_string(),
                    "am_echo".to_string(),
                    "bf_alice".to_string(),
                    "bm_daniel".to_string(),
                ],
            },
        ]
    }
    
    /// Check if any models are installed
    pub fn has_any_model(&self) -> bool {
        !self.installed_models.is_empty()
    }
    
    /// Get the currently active model
    /// 
    /// # Errors
    /// 
    /// Returns an error if no model is active or if the active model
    /// is not loaded.
    pub fn get_active_model(&mut self) -> VocalizeResult<&mut Box<dyn TtsModel>> {
        let active_id = self.active_model.as_ref()
            .ok_or_else(|| VocalizeError::synthesis("No active TTS model"))?;
            
        self.loaded_models.get_mut(active_id)
            .ok_or_else(|| VocalizeError::synthesis("Active model not loaded"))
    }
    
    /// Load the registry from disk
    fn load_registry(&mut self) -> VocalizeResult<()> {
        if self.registry_path.exists() {
            let content = std::fs::read_to_string(&self.registry_path)?;
            let installed: HashMap<String, ModelInfo> = serde_json::from_str(&content)
                .unwrap_or_default();
            self.installed_models = installed;
        }
        Ok(())
    }
    
    /// Save the registry to disk
    fn save_registry(&self) -> VocalizeResult<()> {
        let content = serde_json::to_string_pretty(&self.installed_models)?;
        std::fs::write(&self.registry_path, content)?;
        Ok(())
    }
    
    /// Install a model by downloading it from the specified URL
    /// 
    /// # Errors
    /// 
    /// Returns an error if the model ID is not found in available models,
    /// if the download fails, or if the installation process fails.
    pub async fn install_model(&mut self, model_id: &str) -> VocalizeResult<()> {
        let available_models = Self::get_available_models();
        let model_info = available_models
            .into_iter()
            .find(|m| m.id == model_id)
            .ok_or_else(|| VocalizeError::model_not_found(model_id))?;
        
        let install_path = self.cache_dir.join("models").join(model_id);
        std::fs::create_dir_all(&install_path)?;
        
        // Download model (placeholder implementation)
        self.download_model(&model_info.download_url, &install_path).await?;
        
        // Update registry
        let mut installed_info = model_info;
        installed_info.installed = true;
        installed_info.install_path = install_path;
        self.installed_models.insert(model_id.to_string(), installed_info);
        
        self.save_registry()?;
        
        tracing::info!("Model '{}' installed successfully", model_id);
        Ok(())
    }
    
    /// Remove an installed model
    /// 
    /// # Errors
    /// 
    /// Returns an error if the model is not installed or if the removal fails.
    pub fn remove_model(&mut self, model_id: &str) -> VocalizeResult<()> {
        let model_info = self.installed_models
            .remove(model_id)
            .ok_or_else(|| VocalizeError::model_not_found(model_id))?;
        
        // Remove from loaded models if it's currently loaded
        self.loaded_models.remove(model_id);
        
        // Clear active model if this was the active one
        if self.active_model.as_ref() == Some(&model_id.to_string()) {
            self.active_model = None;
        }
        
        // Remove files from disk
        if model_info.install_path.exists() {
            std::fs::remove_dir_all(&model_info.install_path)?;
        }
        
        self.save_registry()?;
        
        tracing::info!("Model '{}' removed successfully", model_id);
        Ok(())
    }
    
    /// Set the default/active model
    /// 
    /// # Errors
    /// 
    /// Returns an error if the model is not installed.
    pub fn set_default_model(&mut self, model_id: &str) -> VocalizeResult<()> {
        if !self.installed_models.contains_key(model_id) {
            return Err(VocalizeError::model_not_found(model_id));
        }
        
        self.active_model = Some(model_id.to_string());
        tracing::info!("Set active model to '{}'", model_id);
        Ok(())
    }
    
    /// Load a model into memory for synthesis
    /// 
    /// # Errors
    /// 
    /// Returns an error if the model is not installed or fails to load.
    pub fn load_model(&mut self, model_id: &str) -> VocalizeResult<()> {
        let _model_info = self.installed_models
            .get(model_id)
            .ok_or_else(|| VocalizeError::model_not_found(model_id))?;
        
        if self.loaded_models.contains_key(model_id) {
            tracing::debug!("Model '{}' already loaded", model_id);
            return Ok(());
        }
        
        // Create the appropriate model instance based on model ID
        let mut model: Box<dyn TtsModel> = match model_id {
            "kokoro" => {
                use crate::models::kokoro_model::KokoroModel;
                Box::new(KokoroModel::new(self.cache_dir.clone()))
            },
            _ => return Err(VocalizeError::model(format!("Unknown model type: {}", model_id))),
        };
        
        // Load the model
        model.load()?;
        
        // Add to loaded models
        self.loaded_models.insert(model_id.to_string(), model);
        
        // Set as active if no active model
        if self.active_model.is_none() {
            self.active_model = Some(model_id.to_string());
        }
        
        tracing::info!("Model '{}' loaded successfully", model_id);
        Ok(())
    }
    
    /// Get list of installed models
    pub fn get_installed_models(&self) -> Vec<&ModelInfo> {
        self.installed_models.values().collect()
    }
    
    /// Check if a model is installed
    pub fn is_model_installed(&self, model_id: &str) -> bool {
        self.installed_models.contains_key(model_id)
    }
    
    /// Check if a model is loaded
    pub fn is_model_loaded(&self, model_id: &str) -> bool {
        self.loaded_models.contains_key(model_id)
    }
    
    /// Download model from URL (handled by Python model manager)
    async fn download_model(&self, _url: &str, _install_path: &std::path::Path) -> VocalizeResult<()> {
        // Model downloads are handled by the Python model manager using huggingface_hub
        // This ensures reliable downloads with proper authentication and caching
        return Err(VocalizeError::network(
            "Model download is handled by Python. Use 'vocalize models download kokoro' from the CLI, \
             or call the Python model manager directly. \
             The Rust component only loads models from the Python-managed cache."
        ));
    }
    
    /// Enhanced Kokoro model detection with smart discovery
    fn detect_cached_kokoro_model(&mut self) -> VocalizeResult<()> {
        // Check if Kokoro is already registered
        if self.installed_models.contains_key("kokoro") {
            return Ok(());
        }
        
        // Use smart model discovery
        let discovery = crate::model::ModelDiscovery::new();
        
        if let Some(kokoro_files) = discovery.find_best_kokoro_model() {
            tracing::info!("🎯 Smart discovery found Kokoro model: {:?}", kokoro_files.model_file);
            
            // Validate model files
            if !kokoro_files.is_complete() {
                tracing::warn!("⚠️ Kokoro model installation appears incomplete");
            }
            
            // Create manifest if none exists
            let manifest = kokoro_files.manifest.clone()
                .unwrap_or_else(|| discovery.create_manifest_for_model(&kokoro_files));
            
            // Calculate model size
            let total_size = kokoro_files.total_size();
            
            // Detect available voices
            let supported_voices = self.detect_available_voices(&kokoro_files);
            
            let kokoro_info = ModelInfo {
                id: "kokoro".to_string(),
                name: manifest.description.clone().unwrap_or_else(|| "Kokoro TTS".to_string()),
                version: manifest.version.clone(),
                size: total_size as usize,
                download_url: "auto-detected".to_string(),
                license: manifest.license.clone(),
                installed: true,
                install_path: kokoro_files.base_directory().to_path_buf(),
                supported_languages: vec![
                    "en-US".to_string(), 
                    "en-GB".to_string(),
                    "ja-JP".to_string(),
                    "zh-CN".to_string()
                ],
                supported_voices,
            };
            
            // Register the model
            self.installed_models.insert("kokoro".to_string(), kokoro_info);
            
            // Save registry to persist the detection
            self.save_registry()?;
            
            tracing::info!("✅ Successfully registered Kokoro model (size: {} MB)", 
                          total_size / 1_000_000);
            
            // Save a manifest file for future reference
            self.save_model_manifest(&kokoro_files, &manifest)?;
            
        } else {
            tracing::warn!("⚠️ No Kokoro model found in standard cache locations");
            tracing::info!("💡 To install Kokoro: run 'vocalize models download kokoro'");
        }
        
        Ok(())
    }
    
    /// Detect available voices for a Kokoro model
    fn detect_available_voices(&self, kokoro_files: &crate::model::KokoroModelFiles) -> Vec<String> {
        let mut voices = Vec::new();
        
        // Default Kokoro voices (always available as fallback)
        let default_voices = vec![
            "af_heart", "af_alloy", "af_bella", "af_sarah",
            "am_adam", "am_echo", "bf_alice", "bm_daniel"
        ];
        
        if let Some(voices_file) = &kokoro_files.voices_file {
            // Try to read and parse voices file to get actual available voices
            if let Ok(voice_data) = std::fs::read(voices_file) {
                // For now, assume all default voices are available if voices file exists
                // In a full implementation, we would parse the voices file format
                tracing::info!("📢 Found voices file with {} bytes", voice_data.len());
                voices.extend(default_voices.iter().map(|s| s.to_string()));
            }
        } else {
            // No voices file - use default voices with generated embeddings
            tracing::info!("📢 No voices file found, using default voices with fallback embeddings");
            voices.extend(default_voices.iter().map(|s| s.to_string()));
        }
        
        voices
    }
    
    /// Save model manifest for future reference
    fn save_model_manifest(&self, kokoro_files: &crate::model::KokoroModelFiles, manifest: &crate::model::ModelManifest) -> VocalizeResult<()> {
        let manifest_path = kokoro_files.base_directory().join(".vocalize_manifest.json");
        
        let manifest_content = serde_json::to_string_pretty(manifest)
            .map_err(|e| VocalizeError::file(format!("Failed to serialize manifest: {}", e)))?;
        
        std::fs::write(&manifest_path, manifest_content)
            .map_err(|e| VocalizeError::file(format!("Failed to write manifest: {}", e)))?;
        
        tracing::debug!("💾 Saved model manifest: {:?}", manifest_path);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_model_registry_creation() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ModelRegistry::new(temp_dir.path()).unwrap();
        
        assert!(registry.installed_models.is_empty());
        assert!(registry.loaded_models.is_empty());
        assert_eq!(registry.active_model, None);
        assert!(registry.registry_path.parent().unwrap().exists());
        assert!(!registry.has_any_model());
    }
    
    
    #[test]
    fn test_model_info_serialization() {
        let model = ModelInfo {
            id: "test".to_string(),
            name: "Test Model".to_string(),
            version: "v1".to_string(),
            size: 1000,
            download_url: "http://example.com".to_string(),
            license: "MIT".to_string(),
            installed: true,
            install_path: PathBuf::from("/test/path"),
            supported_languages: vec!["en".to_string()],
            supported_voices: vec!["voice1".to_string()],
        };
        
        let json = serde_json::to_string(&model).unwrap();
        let deserialized: ModelInfo = serde_json::from_str(&json).unwrap();
        
        assert_eq!(model, deserialized);
    }
    
    #[test]
    fn test_registry_persistence() {
        let temp_dir = TempDir::new().unwrap();
        let _registry_path = temp_dir.path().join("models.json");
        
        // Create a model info and save it
        {
            let mut registry = ModelRegistry::new(temp_dir.path()).unwrap();
            let model = ModelInfo {
                id: "test".to_string(),
                name: "Test".to_string(),
                version: "v1".to_string(),
                size: 1000,
                download_url: "http://example.com".to_string(),
                license: "MIT".to_string(),
                installed: true,
                install_path: temp_dir.path().join("test"),
                supported_languages: vec!["en".to_string()],
                supported_voices: vec!["voice1".to_string()],
            };
            
            registry.installed_models.insert("test".to_string(), model);
            registry.save_registry().unwrap();
        }
        
        // Load a new registry and verify persistence
        {
            let registry = ModelRegistry::new(temp_dir.path()).unwrap();
            assert!(registry.installed_models.contains_key("test"));
            assert!(registry.has_any_model());
        }
    }
    
    
    #[tokio::test]
    async fn test_model_installation_invalid_model() {
        let temp_dir = TempDir::new().unwrap();
        let mut registry = ModelRegistry::new(temp_dir.path()).unwrap();
        
        let result = registry.install_model("nonexistent").await;
        assert!(result.is_err());
        assert!(!registry.is_model_installed("nonexistent"));
    }
    
    #[test]
    fn test_model_removal() {
        let temp_dir = TempDir::new().unwrap();
        let mut registry = ModelRegistry::new(temp_dir.path()).unwrap();
        
        // First install a model (simulate installation)
        let model_path = temp_dir.path().join("models").join("test_model");
        std::fs::create_dir_all(&model_path).unwrap();
        std::fs::write(model_path.join("test.txt"), "test").unwrap();
        
        let model_info = ModelInfo {
            id: "test_model".to_string(),
            name: "Test Model".to_string(),
            version: "v1".to_string(),
            size: 1000,
            download_url: "http://example.com".to_string(),
            license: "MIT".to_string(),
            installed: true,
            install_path: model_path.clone(),
            supported_languages: vec!["en".to_string()],
            supported_voices: vec!["voice1".to_string()],
        };
        
        registry.installed_models.insert("test_model".to_string(), model_info);
        registry.active_model = Some("test_model".to_string());
        
        // Test removal
        let result = registry.remove_model("test_model");
        assert!(result.is_ok());
        
        // Verify model is removed
        assert!(!registry.is_model_installed("test_model"));
        assert_eq!(registry.active_model, None);
        assert!(!model_path.exists());
    }
    
    #[test]
    fn test_set_default_model() {
        let temp_dir = TempDir::new().unwrap();
        let mut registry = ModelRegistry::new(temp_dir.path()).unwrap();
        
        // Add a model to installed models
        let model_info = ModelInfo {
            id: "test_model".to_string(),
            name: "Test Model".to_string(),
            version: "v1".to_string(),
            size: 1000,
            download_url: "http://example.com".to_string(),
            license: "MIT".to_string(),
            installed: true,
            install_path: PathBuf::from("/test/path"),
            supported_languages: vec!["en".to_string()],
            supported_voices: vec!["voice1".to_string()],
        };
        
        registry.installed_models.insert("test_model".to_string(), model_info);
        
        // Test setting default model
        let result = registry.set_default_model("test_model");
        assert!(result.is_ok());
        assert_eq!(registry.active_model, Some("test_model".to_string()));
        
        // Test setting non-existent model as default
        let result = registry.set_default_model("nonexistent");
        assert!(result.is_err());
    }
    
    #[test]
    fn test_model_loading_mock() {
        let temp_dir = TempDir::new().unwrap();
        let mut registry = ModelRegistry::new(temp_dir.path()).unwrap();
        
        // Add mock model to installed models
        let model_info = ModelInfo {
            id: "mock".to_string(),
            name: "Mock Model".to_string(),
            version: "v1".to_string(),
            size: 1000,
            download_url: "http://example.com".to_string(),
            license: "MIT".to_string(),
            installed: true,
            install_path: temp_dir.path().to_path_buf(),
            supported_languages: vec!["en".to_string()],
            supported_voices: vec!["voice1".to_string()],
        };
        
        registry.installed_models.insert("mock".to_string(), model_info);
        
        // Test loading model
        let result = registry.load_model("mock");
        assert!(result.is_ok());
        
        // Verify model is loaded
        assert!(registry.is_model_loaded("mock"));
        assert_eq!(registry.active_model, Some("mock".to_string()));
        
        // Test loading already loaded model
        let result = registry.load_model("mock");
        assert!(result.is_ok()); // Should not error
    }
    
    #[test]
    fn test_get_installed_models() {
        let temp_dir = TempDir::new().unwrap();
        let mut registry = ModelRegistry::new(temp_dir.path()).unwrap();
        
        // Initially no models
        assert!(registry.get_installed_models().is_empty());
        
        // Add some models
        let model1 = ModelInfo {
            id: "model1".to_string(),
            name: "Model 1".to_string(),
            version: "v1".to_string(),
            size: 1000,
            download_url: "http://example.com".to_string(),
            license: "MIT".to_string(),
            installed: true,
            install_path: PathBuf::from("/test/path1"),
            supported_languages: vec!["en".to_string()],
            supported_voices: vec!["voice1".to_string()],
        };
        
        let model2 = ModelInfo {
            id: "model2".to_string(),
            name: "Model 2".to_string(),
            version: "v1".to_string(),
            size: 2000,
            download_url: "http://example.com".to_string(),
            license: "Apache-2.0".to_string(),
            installed: true,
            install_path: PathBuf::from("/test/path2"),
            supported_languages: vec!["en".to_string()],
            supported_voices: vec!["voice2".to_string()],
        };
        
        registry.installed_models.insert("model1".to_string(), model1);
        registry.installed_models.insert("model2".to_string(), model2);
        
        let installed = registry.get_installed_models();
        assert_eq!(installed.len(), 2);
        
        let ids: Vec<&str> = installed.iter().map(|m| m.id.as_str()).collect();
        assert!(ids.contains(&"model1"));
        assert!(ids.contains(&"model2"));
    }
    
    #[test]
    fn test_model_status_checks() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ModelRegistry::new(temp_dir.path()).unwrap();
        
        // Test non-existent model
        assert!(!registry.is_model_installed("nonexistent"));
        assert!(!registry.is_model_loaded("nonexistent"));
    }
    
    #[tokio::test]
    async fn test_get_active_model_none() {
        let temp_dir = TempDir::new().unwrap();
        let mut registry = ModelRegistry::new(temp_dir.path()).unwrap();
        
        let result = registry.get_active_model();
        assert!(result.is_err());
        
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("No active TTS model"));
    }
    
    #[tokio::test]
    async fn test_get_active_model_not_loaded() {
        let temp_dir = TempDir::new().unwrap();
        let mut registry = ModelRegistry::new(temp_dir.path()).unwrap();
        
        // Set active model but don't load it
        registry.active_model = Some("nonexistent".to_string());
        
        let result = registry.get_active_model();
        assert!(result.is_err());
        
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Active model not loaded"));
    }
}