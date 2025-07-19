//! Smart model discovery system
//! Automatically finds and validates TTS models from various sources

#![allow(missing_docs)]

use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};
use anyhow::Result;
use directories::ProjectDirs;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelManifest {
    pub model_file: String,
    pub voices_file: Option<String>,
    pub tokenizer_file: Option<String>,
    pub config_file: Option<String>,
    pub version: String,
    pub checksum: Option<String>,
    pub model_type: String,
    pub license: String,
    pub description: Option<String>,
}

pub struct ModelDiscovery {
    cache_dirs: Vec<PathBuf>,
}

impl ModelDiscovery {
    pub fn new() -> Self {
        let mut cache_dirs = Vec::new();
        
        // Standard cache directories using cross-platform paths
        if let Some(proj_dirs) = ProjectDirs::from("ai", "Vocalize", "vocalize") {
            // Use cross-platform cache directory
            cache_dirs.push(proj_dirs.cache_dir().join("models"));
        }
        
        // Also check legacy paths for backward compatibility
        if let Ok(home) = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")) {
            let home_path = PathBuf::from(home);
            // HuggingFace cache
            cache_dirs.push(home_path.join(".cache/huggingface/hub"));
            // Legacy vocalize cache
            cache_dirs.push(home_path.join(".vocalize/models"));
        }
        
        // System cache directories
        cache_dirs.push(PathBuf::from("/var/cache/vocalize"));
        cache_dirs.push(PathBuf::from("./models")); // Local models directory
        cache_dirs.push(PathBuf::from("./.vocalize/models")); // Project-local cache
        
        // Environment variable override
        if let Ok(custom_cache) = std::env::var("VOCALIZE_MODEL_CACHE") {
            cache_dirs.insert(0, PathBuf::from(custom_cache));
        }
        
        Self { cache_dirs }
    }
    
    /// Find all available Kokoro model installations
    pub fn find_kokoro_models(&self) -> Vec<KokoroModelFiles> {
        let mut found_models = Vec::new();
        
        for cache_dir in &self.cache_dirs {
            if let Ok(models) = self.scan_directory_for_kokoro(cache_dir) {
                found_models.extend(models);
            }
        }
        
        // Remove duplicates based on model file path
        found_models.sort_by(|a, b| a.model_file.cmp(&b.model_file));
        found_models.dedup_by(|a, b| a.model_file == b.model_file);
        
        found_models
    }
    
    /// Find the EXACT Kokoro model (ZERO-FALLBACK: require exact files)
    pub fn find_best_kokoro_model(&self) -> Option<KokoroModelFiles> {
        // ZERO-FALLBACK: Only accept models with ALL required files
        for cache_dir in &self.cache_dirs {
            if let Some(exact_model) = self.find_exact_kokoro_model(cache_dir) {
                return Some(exact_model);
            }
        }
        None
    }
    
    /// ZERO-FALLBACK: Find EXACT Kokoro model with ALL required files
    fn find_exact_kokoro_model(&self, cache_dir: &Path) -> Option<KokoroModelFiles> {
        // Check multiple possible locations for Kokoro model
        let possible_paths = [
            cache_dir.join("models--hexgrad--Kokoro-82M").join("local"),
            cache_dir.join("models--direct_download").join("local"), // Python-managed location
        ];
        
        let mut exact_kokoro_path = None;
        for path in &possible_paths {
            if path.exists() {
                exact_kokoro_path = Some(path.clone());
                tracing::debug!("Found Kokoro model directory: {:?}", path);
                break;
            }
        }
        
        let exact_kokoro_path = exact_kokoro_path?;
        
        // REQUIRE EXACT FILES (no patterns, no approximations)
        let required_files = [
            ("kokoro-v1.0.onnx", "model_file"),
            ("voices-v1.0.bin", "voices_file"), 
            ("tokenizer.json", "tokenizer_file"),
        ];
        
        let mut model_file = None;
        let mut voices_file = None;
        let mut tokenizer_file = None;
        
        // Check ALL required files exist with EXACT names
        for (exact_filename, file_type) in &required_files {
            let exact_path = exact_kokoro_path.join(exact_filename);
            if !exact_path.exists() || !exact_path.is_file() {
                tracing::error!("❌ ZERO-FALLBACK: Missing required file '{}' in {:?}", exact_filename, exact_kokoro_path);
                return None;
            }
            
            match *file_type {
                "model_file" => model_file = Some(exact_path),
                "voices_file" => voices_file = Some(exact_path),
                "tokenizer_file" => tokenizer_file = Some(exact_path),
                _ => {}
            }
        }
        
        // REQUIRE ALL files to be present
        let model_file = model_file?;
        
        // Validate ONNX file is actually valid
        if !self.is_valid_onnx_file(&model_file) {
            tracing::error!("❌ ZERO-FALLBACK: Invalid ONNX file: {:?}", model_file);
            return None;
        }
        
        tracing::info!("✅ ZERO-FALLBACK: Found EXACT Kokoro model with ALL required files: {:?}", exact_kokoro_path);
        
        Some(KokoroModelFiles {
            model_file,
            voices_file,
            tokenizer_file,
            manifest: None, // Manifest is optional in zero-fallback mode
        })
    }
    
    fn scan_directory_for_kokoro(&self, base_dir: &Path) -> Result<Vec<KokoroModelFiles>> {
        let mut found_models = Vec::new();
        
        if !base_dir.exists() {
            return Ok(found_models);
        }
        
        // Search patterns for different cache structures
        let search_patterns = [
            // HuggingFace cache patterns
            "models--*kokoro*/snapshots/*/",
            "models--onnx-community--Kokoro*/snapshots/*/",
            "models--direct_download/local/",
            // Direct model directories
            "kokoro*/",
            "Kokoro*/",
            "*/kokoro*/",
            "*/Kokoro*/",
        ];
        
        for pattern in &search_patterns {
            if let Ok(matches) = self.glob_search(base_dir, pattern) {
                for match_dir in matches {
                    if let Some(model_files) = self.analyze_kokoro_directory(&match_dir) {
                        found_models.push(model_files);
                    }
                }
            }
        }
        
        // Also search the base directory directly
        if let Some(model_files) = self.analyze_kokoro_directory(base_dir) {
            found_models.push(model_files);
        }
        
        Ok(found_models)
    }
    
    fn analyze_kokoro_directory(&self, dir: &Path) -> Option<KokoroModelFiles> {
        if !dir.exists() || !dir.is_dir() {
            return None;
        }
        
        tracing::debug!("🔍 Analyzing directory for Kokoro model: {:?}", dir);
        
        // Priority order for model files
        let model_patterns = [
            "kokoro-v1.0.onnx",
            "model_quantized.onnx",
            "kokoro.onnx",
            "model.onnx",
            "*.onnx"
        ];
        
        let voices_patterns = [
            "voices-v1.0.bin",
            "voices.bin", 
            "embeddings.bin",
            "voice_embeddings.bin"
        ];
        
        let tokenizer_patterns = [
            "tokenizer.json",
            "vocab.json",
            "tokenizer_config.json"
        ];
        
        // Find model file
        let model_file = self.find_file_by_patterns(dir, &model_patterns)?;
        
        // Validate it's actually an ONNX file
        if !self.is_valid_onnx_file(&model_file) {
            tracing::debug!("❌ Invalid ONNX file: {:?}", model_file);
            return None;
        }
        
        // Find additional files
        let voices_file = self.find_file_by_patterns(dir, &voices_patterns);
        let tokenizer_file = self.find_file_by_patterns(dir, &tokenizer_patterns);
        let manifest = self.load_manifest(dir);
        
        tracing::info!("✅ Found Kokoro model: {:?}", model_file);
        if voices_file.is_some() {
            tracing::info!("   📢 Voices file: {:?}", voices_file.as_ref().unwrap());
        }
        if tokenizer_file.is_some() {
            tracing::info!("   🔤 Tokenizer file: {:?}", tokenizer_file.as_ref().unwrap());
        }
        
        Some(KokoroModelFiles {
            model_file,
            voices_file,
            tokenizer_file,
            manifest,
        })
    }
    
    fn find_file_by_patterns(&self, dir: &Path, patterns: &[&str]) -> Option<PathBuf> {
        for pattern in patterns {
            if pattern.contains('*') {
                // Glob pattern
                if let Ok(matches) = self.glob_search(dir, pattern) {
                    if let Some(first_match) = matches.into_iter().next() {
                        return Some(first_match);
                    }
                }
            } else {
                // Exact filename
                let path = dir.join(pattern);
                if path.exists() && path.is_file() {
                    return Some(path);
                }
            }
        }
        None
    }
    
    fn glob_search(&self, base_dir: &Path, pattern: &str) -> Result<Vec<PathBuf>> {
        let full_pattern = base_dir.join(pattern);
        let pattern_str = full_pattern.to_string_lossy();
        
        let mut matches = Vec::new();
        
        if let Ok(entries) = glob::glob(&pattern_str) {
            for entry in entries {
                if let Ok(path) = entry {
                    matches.push(path);
                }
            }
        }
        
        Ok(matches)
    }
    
    fn is_valid_onnx_file(&self, path: &Path) -> bool {
        // Basic ONNX file validation
        if !path.exists() || !path.is_file() {
            return false;
        }
        
        // Check file extension
        if let Some(ext) = path.extension() {
            if ext != "onnx" {
                return false;
            }
        } else {
            return false;
        }
        
        // Check file size (should be reasonable for a TTS model)
        if let Ok(metadata) = std::fs::metadata(path) {
            let size = metadata.len();
            // Expect at least 1MB and at most 2GB
            if size < 1_000_000 || size > 2_000_000_000 {
                tracing::warn!("ONNX file size suspicious: {} bytes", size);
                return false;
            }
        }
        
        // Try to read the first few bytes to check ONNX magic
        if let Ok(mut file) = std::fs::File::open(path) {
            use std::io::Read;
            let mut buffer = [0u8; 16];
            if file.read_exact(&mut buffer).is_ok() {
                // ONNX files typically start with protobuf headers
                // This is a simple heuristic check
                return buffer[0] != 0 && buffer.iter().any(|&b| b > 0);
            }
        }
        
        true // If we can't validate, assume it's valid
    }
    
    fn load_manifest(&self, dir: &Path) -> Option<ModelManifest> {
        let manifest_files = [
            "manifest.json",
            "model_manifest.json", 
            "config.json",
            ".vocalize_manifest.json"
        ];
        
        for manifest_file in &manifest_files {
            let manifest_path = dir.join(manifest_file);
            if manifest_path.exists() {
                if let Ok(content) = std::fs::read_to_string(&manifest_path) {
                    if let Ok(manifest) = serde_json::from_str::<ModelManifest>(&content) {
                        tracing::debug!("📄 Loaded manifest: {:?}", manifest_path);
                        return Some(manifest);
                    }
                }
            }
        }
        
        None
    }
    
    /// Create a manifest for a discovered model
    pub fn create_manifest_for_model(&self, model_files: &KokoroModelFiles) -> ModelManifest {
        ModelManifest {
            model_file: model_files.model_file.file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            voices_file: model_files.voices_file.as_ref()
                .and_then(|p| p.file_name())
                .map(|n| n.to_string_lossy().to_string()),
            tokenizer_file: model_files.tokenizer_file.as_ref()
                .and_then(|p| p.file_name())
                .map(|n| n.to_string_lossy().to_string()),
            config_file: None,
            version: "auto-detected".to_string(),
            checksum: None,
            model_type: "kokoro".to_string(),
            license: "Apache 2.0".to_string(),
            description: Some("Auto-detected Kokoro TTS model".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct KokoroModelFiles {
    pub model_file: PathBuf,
    pub voices_file: Option<PathBuf>,
    pub tokenizer_file: Option<PathBuf>,
    pub manifest: Option<ModelManifest>,
}

impl KokoroModelFiles {
    /// Get the base directory containing the model files
    pub fn base_directory(&self) -> &Path {
        self.model_file.parent().unwrap_or(Path::new("."))
    }
    
    /// Calculate total size of all model files
    pub fn total_size(&self) -> u64 {
        let mut total = 0;
        
        if let Ok(meta) = std::fs::metadata(&self.model_file) {
            total += meta.len();
        }
        
        if let Some(voices_file) = &self.voices_file {
            if let Ok(meta) = std::fs::metadata(voices_file) {
                total += meta.len();
            }
        }
        
        if let Some(tokenizer_file) = &self.tokenizer_file {
            if let Ok(meta) = std::fs::metadata(tokenizer_file) {
                total += meta.len();
            }
        }
        
        total
    }
    
    /// Check if this model installation is complete
    pub fn is_complete(&self) -> bool {
        self.model_file.exists() && 
        (self.voices_file.is_none() || self.voices_file.as_ref().unwrap().exists())
    }
}

impl Default for ModelDiscovery {
    fn default() -> Self {
        Self::new()
    }
}