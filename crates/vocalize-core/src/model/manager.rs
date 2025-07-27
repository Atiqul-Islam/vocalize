// ModelManager implementation for neural TTS models
// Loads models from Python-managed cache only (no direct downloads)

use std::path::PathBuf;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::RwLock;
use anyhow::{Result, Context};
use ort::session::Session;
use directories::ProjectDirs;

use super::types::{ModelId, ModelInfo};

/// Manages ONNX model loading from Python-managed cache
#[derive(Debug)]
pub struct ModelManager {
    /// Directory for caching downloaded models
    pub cache_dir: PathBuf,
    loaded_models: Arc<RwLock<HashMap<ModelId, Arc<Mutex<Session>>>>>,
}

impl ModelManager {
    /// Create a new ModelManager with specified cache directory
    pub fn new(cache_dir: PathBuf) -> Self {
        // Ensure cache directory exists
        if let Err(e) = std::fs::create_dir_all(&cache_dir) {
            tracing::warn!("Failed to create cache directory: {}", e);
        }
        
        Self {
            cache_dir,
            loaded_models: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Create a new ModelManager with cross-platform cache directory
    pub fn new_with_default_cache() -> Result<Self> {
        let proj_dirs = ProjectDirs::from("ai", "Vocalize", "vocalize")
            .ok_or_else(|| anyhow::anyhow!("Failed to determine project directories"))?;
        
        let cache_dir = proj_dirs.cache_dir().join("models");
        
        tracing::info!("Using cross-platform cache directory: {:?}", cache_dir);
        
        Ok(Self::new(cache_dir))
    }
    
    /// Get the default model info (Kokoro TTS)
    pub fn get_default_model(&self) -> ModelInfo {
        ModelInfo::kokoro()
    }
    
    /// Check if a model is cached locally by Python model manager
    pub fn is_model_cached(&self, model_id: ModelId) -> bool {
        let model_info = self.get_model_info(model_id);
        let repo_cache_name = model_info.repo_id.replace("/", "--");
        
        tracing::info!("DEBUG: Checking if model {:?} is cached", model_id);
        tracing::info!("DEBUG: Model info: name={}, repo_id={}", model_info.name, model_info.repo_id);
        tracing::info!("DEBUG: Required files: {:?}", model_info.files);
        tracing::info!("DEBUG: Cache dir: {:?}", self.cache_dir);
        
        // First check in local directory (no symlinks, cross-platform compatible)
        let model_local_dir = self.cache_dir.join(format!("models--{}", &repo_cache_name)).join("local");
        tracing::info!("DEBUG: Checking local dir: {:?}", model_local_dir);
        tracing::info!("DEBUG: Local dir exists: {}", model_local_dir.exists());
        
        if model_local_dir.exists() {
            for file in &model_info.files {
                let file_path = model_local_dir.join(file);
                let exists = file_path.exists();
                let is_file = file_path.is_file();
                tracing::info!("DEBUG: File {:?} - exists: {}, is_file: {}", file_path, exists, is_file);
            }
            
            let all_files_exist = model_info.files.iter().all(|file| {
                let file_path = model_local_dir.join(file);
                file_path.exists() && file_path.is_file()
            });
            tracing::info!("DEBUG: All files exist in local dir: {}", all_files_exist);
            if all_files_exist {
                return true;
            }
        }
        
        // Fallback: look in legacy snapshots directory (symlink structure)
        let model_cache_dir = self.cache_dir.join(format!("models--{}", &repo_cache_name));
        tracing::info!("DEBUG: Checking snapshots dir: {:?}", model_cache_dir.join("snapshots"));
        
        if let Ok(entries) = std::fs::read_dir(model_cache_dir.join("snapshots")) {
            for entry in entries.flatten() {
                let snapshot_dir = entry.path();
                tracing::info!("DEBUG: Found snapshot dir: {:?}", snapshot_dir);
                if snapshot_dir.is_dir() {
                    let all_files_exist = model_info.files.iter().all(|file| {
                        let file_path = snapshot_dir.join(file);
                        let exists = file_path.exists();
                        tracing::info!("DEBUG: Snapshot file {:?} exists: {}", file_path, exists);
                        exists
                    });
                    if all_files_exist {
                        tracing::info!("DEBUG: All files found in snapshot dir: {:?}", snapshot_dir);
                        return true;
                    }
                }
            }
        }
        
        tracing::warn!("DEBUG: Model {:?} NOT FOUND in cache", model_id);
        false
    }
    
    /// Get the path to a cached model file
    fn get_model_file_path(&self, model_id: ModelId, filename: &str) -> Option<PathBuf> {
        let model_info = self.get_model_info(model_id);
        let repo_cache_name = model_info.repo_id.replace("/", "--");
        
        // First check in local directory (no symlinks, cross-platform compatible)
        let model_local_dir = self.cache_dir.join(format!("models--{}", &repo_cache_name)).join("local");
        let local_file_path = model_local_dir.join(filename);
        if local_file_path.exists() && local_file_path.is_file() {
            return Some(local_file_path);
        }
        
        // Fallback: look in legacy snapshots directory (symlink structure)
        let model_cache_dir = self.cache_dir.join(format!("models--{}", &repo_cache_name));
        if let Ok(entries) = std::fs::read_dir(model_cache_dir.join("snapshots")) {
            for entry in entries.flatten() {
                let snapshot_dir = entry.path();
                if snapshot_dir.is_dir() {
                    let file_path = snapshot_dir.join(filename);
                    if file_path.exists() {
                        return Some(file_path);
                    }
                }
            }
        }
        
        None
    }
    
    /// Load a model into ONNX Runtime session from Python-managed cache
    pub async fn load_model(&self, model_id: ModelId) -> Result<Arc<Mutex<Session>>> {
        // Check if already loaded
        {
            let loaded = self.loaded_models.read().await;
            if let Some(session) = loaded.get(&model_id) {
                return Ok(session.clone());
            }
        }
        
        // Ensure model is available in Python-managed cache
        if !self.is_model_cached(model_id) {
            let model_info = self.get_model_info(model_id);
            // Debug: show what paths we're checking
            let repo_cache_name = model_info.repo_id.replace("/", "--");
            let model_local_dir = self.cache_dir.join(format!("models--{}", &repo_cache_name)).join("local");
            let model_cache_dir = self.cache_dir.join(format!("models--{}", &repo_cache_name));
            
            tracing::error!("Model {} not found in cache", model_info.name);
            tracing::error!("Checked local dir: {:?}", model_local_dir);
            tracing::error!("Checked cache dir: {:?}", model_cache_dir);
            tracing::error!("Required files: {:?}", model_info.files);
            
            return Err(anyhow::anyhow!(
                "Model {} ({}) not found in cache. Please download it first using Python: vocalize models download {}",
                model_info.name, model_info.repo_id, model_id.as_str()
            ));
        }
        
        let model_info = self.get_model_info(model_id);
        
        // Find the ONNX model file (first .onnx file in the files list)
        let onnx_filename = model_info.files.iter()
            .find(|f| f.ends_with(".onnx"))
            .ok_or_else(|| anyhow::anyhow!("No ONNX file found for model {}", model_info.name))?;
            
        let onnx_file = self.get_model_file_path(model_id, onnx_filename)
            .ok_or_else(|| anyhow::anyhow!("ONNX file {} not found in cache", onnx_filename))?;
        
        // Create ONNX session using 2025 optimization settings and threading configuration
        tracing::info!("Loading model {} from Python-managed cache: {:?}", model_info.name, onnx_file);
        // Detect if this is an INT8 quantized model
        let is_int8_model = onnx_filename.contains("int8") || onnx_filename.contains("INT8");
        let physical_cores = num_cpus::get_physical();
        
        // Configure based on model type
        let (opt_level, intra_threads, inter_threads, memory_pattern) = if is_int8_model {
            tracing::info!("INT8 quantized model detected - using conservative settings");
            (
                ort::session::builder::GraphOptimizationLevel::Level1,
                std::cmp::min(4, physical_cores),
                2,
                false
            )
        } else {
            tracing::info!("FP32 model detected - using performance settings");
            (
                ort::session::builder::GraphOptimizationLevel::Level3,
                std::cmp::min(physical_cores, 8),
                std::cmp::min(4, std::cmp::max(2, physical_cores / 3)),
                true
            )
        };
        
        let session = Session::builder()?
            .with_optimization_level(opt_level)?
            .with_intra_threads(intra_threads)?
            .with_inter_threads(inter_threads)?
            .with_memory_pattern(memory_pattern)?
            .commit_from_file(&onnx_file)
            .context(format!("Failed to load ONNX model from {:?}", onnx_file))?;
            
        let session = Arc::new(Mutex::new(session));
        
        // Cache the loaded session
        {
            let mut loaded = self.loaded_models.write().await;
            loaded.insert(model_id, session.clone());
        }
        
        tracing::info!("Successfully loaded model: {}", model_info.name);
        Ok(session)
    }
    
    /// Get model information for a given model ID
    fn get_model_info(&self, model_id: ModelId) -> ModelInfo {
        match model_id {
            ModelId::Kokoro => ModelInfo::kokoro(),
            ModelId::Chatterbox => ModelInfo::chatterbox(),
            ModelId::Dia => ModelInfo::dia(),
        }
    }
    
    /// Get the path to the model's ONNX file for session pool initialization
    /// Checks for quantized variants based on passed flags
    pub async fn get_model_path(&self, model_id: ModelId, use_quantized: bool) -> Result<PathBuf> {
        // Ensure model is available in cache
        if !self.is_model_cached(model_id) {
            let model_info = self.get_model_info(model_id);
            return Err(anyhow::anyhow!(
                "Model {} ({}) not found in cache. Please download it first using Python: vocalize models download {}",
                model_info.name, model_info.repo_id, model_id.as_str()
            ));
        }
        
        
        tracing::debug!("Optimization flags - Quantization: {}", use_quantized);
        
        // Determine which model variant to use
        // Priority: quantized > original
        let variants = if use_quantized {
            vec!["kokoro-v1.0-int8.onnx", "kokoro-v1.0.onnx"]
        } else {
            vec!["kokoro-v1.0.onnx"]
        };
        
        // Try each variant in order
        for variant in variants {
            if let Some(path) = self.get_model_file_path(model_id, variant) {
                tracing::info!("🎯 Using model variant: {} for {:?}", variant, model_id);
                return Ok(path);
            }
        }
        
        // Fallback to original model search
        let model_info = self.get_model_info(model_id);
        let onnx_filename = model_info.files.iter()
            .find(|f| f.ends_with(".onnx"))
            .ok_or_else(|| anyhow::anyhow!("No ONNX file found for model {}", model_info.name))?;
            
        let onnx_file = self.get_model_file_path(model_id, onnx_filename)
            .ok_or_else(|| anyhow::anyhow!("ONNX file {} not found in cache", onnx_filename))?;
        
        tracing::debug!("Found model path for {:?}: {:?}", model_id, onnx_file);
        Ok(onnx_file)
    }
    
    /// List all available models
    pub fn list_available_models(&self) -> Vec<ModelInfo> {
        vec![
            ModelInfo::kokoro(),
            ModelInfo::chatterbox(),
            ModelInfo::dia(),
        ]
    }
}