//! ONNX-based neural TTS engine implementation
//! Replaces mathematical synthesis with real neural models

pub mod session_pool;

use std::path::PathBuf;
use anyhow::{Result, Context};
use unicode_normalization::UnicodeNormalization;

use crate::model::{ModelManager, ModelId};
use crate::{VocalizeResult, VocalizeError};
use session_pool::OnnxSessionPool;

/// ONNX-based neural TTS engine
#[derive(Debug)]
pub struct OnnxTtsEngine {
    model_manager: ModelManager,
    session_pool: Option<OnnxSessionPool>,
    current_model: Option<ModelId>,
    // Removed tokenizer - text processing handled by Python layer
}

impl OnnxTtsEngine {
    /// Ensure ONNX Runtime library is available for the current platform
    async fn ensure_onnx_runtime_available() -> Result<String> {
        // Determine the platform and appropriate library name
        let (platform, lib_name) = if cfg!(target_os = "windows") {
            ("win-x64", "onnxruntime.dll")
        } else if cfg!(target_os = "macos") {
            ("osx-x64", "libonnxruntime.dylib")  // or osx-arm64 for Apple Silicon
        } else {
            ("linux-x64", "libonnxruntime.so")
        };
        
        let version = "1.22.0";
        let ort_dir = PathBuf::from("onnxruntime");
        let lib_path = ort_dir.join(lib_name);
        
        // Check if library already exists
        if lib_path.exists() {
            tracing::info!("ONNX Runtime library already exists: {:?}", lib_path);
            return Ok(lib_path.to_string_lossy().to_string());
        }
        
        // Create directory
        std::fs::create_dir_all(&ort_dir)
            .context("Failed to create onnxruntime directory")?;
        
        // Download the platform-specific ONNX Runtime
        tracing::info!("Downloading ONNX Runtime {} for platform: {}", version, platform);
        
        let archive_name = if cfg!(target_os = "windows") {
            format!("onnxruntime-{}-{}.zip", platform, version)
        } else {
            format!("onnxruntime-{}-{}.tgz", platform, version)
        };
        
        let download_url = format!(
            "https://github.com/microsoft/onnxruntime/releases/download/v{}/{}",
            version, archive_name
        );
        
        tracing::info!("Downloading from: {}", download_url);
        
        // Download and extract in a blocking task
        let lib_path_clone = lib_path.clone();
        let ort_dir_clone = ort_dir.clone();
        let result = tokio::task::spawn_blocking(move || {
            Self::download_and_extract_ort(&download_url, &archive_name, &ort_dir_clone, &lib_path_clone, lib_name)
        }).await.context("Failed to spawn download task")?;
        
        result?;
        
        if lib_path.exists() {
            tracing::info!("Successfully downloaded ONNX Runtime library: {:?}", lib_path);
            Ok(lib_path.to_string_lossy().to_string())
        } else {
            Err(anyhow::anyhow!("Downloaded ONNX Runtime but library file not found: {:?}", lib_path))
        }
    }
    
    /// Download and extract ONNX Runtime for the current platform
    fn download_and_extract_ort(
        download_url: &str,
        archive_name: &str,
        ort_dir: &PathBuf,
        lib_path: &PathBuf,
        lib_name: &str,
    ) -> Result<()> {
        use std::fs::File;
        use std::io::Write;
        
        // Download the archive
        let archive_path = ort_dir.join(archive_name);
        tracing::info!("Downloading to: {:?}", archive_path);
        
        let response = reqwest::blocking::get(download_url)
            .context("Failed to download ONNX Runtime")?;
        
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Failed to download ONNX Runtime: HTTP {}", response.status()));
        }
        
        let bytes = response.bytes().context("Failed to read download response")?;
        let mut file = File::create(&archive_path).context("Failed to create archive file")?;
        file.write_all(&bytes).context("Failed to write archive file")?;
        
        // Extract the archive
        tracing::info!("Extracting archive: {:?}", archive_path);
        
        if cfg!(target_os = "windows") {
            // Extract ZIP file
            let file = File::open(&archive_path).context("Failed to open ZIP archive")?;
            let mut archive = zip::ZipArchive::new(file).context("Failed to read ZIP archive")?;
            
            for i in 0..archive.len() {
                let mut file = archive.by_index(i).context("Failed to get ZIP entry")?;
                let outpath = ort_dir.join(file.mangled_name());
                
                if file.name().ends_with('/') {
                    std::fs::create_dir_all(&outpath).context("Failed to create directory")?;
                } else {
                    if let Some(p) = outpath.parent() {
                        if !p.exists() {
                            std::fs::create_dir_all(&p).context("Failed to create parent directory")?;
                        }
                    }
                    let mut outfile = File::create(&outpath).context("Failed to create output file")?;
                    std::io::copy(&mut file, &mut outfile).context("Failed to extract file")?;
                }
            }
        } else {
            // Extract TAR.GZ file
            let tar_gz = File::open(&archive_path).context("Failed to open TAR.GZ archive")?;
            let tar = flate2::read::GzDecoder::new(tar_gz);
            let mut archive = tar::Archive::new(tar);
            archive.unpack(&ort_dir).context("Failed to extract TAR.GZ archive")?;
        }
        
        // Find the extracted library file
        for entry in std::fs::read_dir(&ort_dir).context("Failed to read ort directory")? {
            let entry = entry.context("Failed to read directory entry")?;
            let path = entry.path();
            
            if path.is_dir() {
                // Look for the library in the lib subdirectory
                let lib_dir = path.join("lib");
                if lib_dir.exists() {
                    let source_lib = lib_dir.join(lib_name);
                    if source_lib.exists() {
                        std::fs::copy(&source_lib, &lib_path)
                            .context("Failed to copy library file")?;
                        tracing::info!("Copied library from {:?} to {:?}", source_lib, lib_path);
                        break;
                    }
                }
            }
        }
        
        // Clean up archive
        let _ = std::fs::remove_file(&archive_path);
        
        Ok(())
    }
    /// Create a new ONNX TTS engine
    pub async fn new(cache_dir: PathBuf) -> Result<Self> {
        // Set environment variables to prevent ONNX Runtime threading deadlocks
        // These MUST be set before any ort initialization
        std::env::set_var("OMP_NUM_THREADS", "1");
        std::env::set_var("MKL_NUM_THREADS", "1");
        std::env::set_var("OPENBLAS_NUM_THREADS", "1");
        std::env::set_var("BLIS_NUM_THREADS", "1");
        
        // 2025 ONNX Fix: Enable float16 optimization to prevent noise output
        std::env::set_var("ORT_ENABLE_FP16", "1");
        std::env::set_var("ORT_DISABLE_ALL_OPTIMIZATIONS", "0");
        
        tracing::info!("ONNX Engine: Set threading and float16 optimization environment variables");
        
        // Initialize ONNX Runtime with load-dynamic feature
        // This MUST be called before any ort usage when using load-dynamic
        tracing::info!("ONNX Engine: Initializing ONNX Runtime...");
        
        // With load-dynamic feature, we need to ensure ONNX Runtime library is available
        // Set up cross-platform ONNX Runtime library path
        tracing::info!("ONNX Engine: Setting up cross-platform ONNX Runtime...");
        
        // Check if system library is already available
        let system_lib_available = if cfg!(target_os = "windows") {
            std::env::var("ORT_DYLIB_PATH").is_ok() || std::path::Path::new("onnxruntime.dll").exists()
        } else if cfg!(target_os = "macos") {
            std::env::var("ORT_DYLIB_PATH").is_ok() || std::path::Path::new("libonnxruntime.dylib").exists()
        } else {
            std::env::var("ORT_DYLIB_PATH").is_ok() || std::path::Path::new("libonnxruntime.so").exists()
        };
        
        if !system_lib_available {
            tracing::info!("System ONNX Runtime not found, downloading platform-specific version...");
            let ort_path = Self::ensure_onnx_runtime_available().await?;
            
            // Set ORT_DYLIB_PATH environment variable for load-dynamic feature
            std::env::set_var("ORT_DYLIB_PATH", &ort_path);
            tracing::info!("Set ORT_DYLIB_PATH to: {}", ort_path);
        } else {
            tracing::info!("Using system-provided ONNX Runtime library");
        }
        
        // Initialize ONNX Runtime with load-dynamic feature
        match ort::init().commit() {
            Ok(_) => {
                tracing::info!("ONNX Engine: Successfully initialized ONNX Runtime with load-dynamic and optimizations");
            }
            Err(e) => {
                return Err(anyhow::anyhow!(
                    "Failed to initialize ONNX Runtime with load-dynamic feature. Error: {}. \
                     Make sure libonnxruntime is available or ORT_DYLIB_PATH is set correctly.", e
                ));
            }
        }
        
        let model_manager = ModelManager::new(cache_dir);
        
        Ok(Self {
            model_manager,
            session_pool: None,
            current_model: None,
        })
    }
    
    /// Load a specific model for synthesis
    pub async fn load_model(&mut self, model_id: ModelId) -> Result<()> {
        tracing::info!("🔄 ONNX Engine: Loading model {:?}", model_id);
        
        // 2025 Fix: Always reload model to prevent tensor shape issues
        self.session_pool = None;
        self.current_model = None;
        
        // Get model path from ModelManager
        tracing::debug!("📂 Getting model path from ModelManager...");
        let model_path = self.model_manager.get_model_path(model_id).await
            .context(format!("Failed to get model path for {:?}", model_id))?;
        
        // Create session pool with multiple sessions for concurrent access
        tracing::info!("🏊 Creating session pool for model...");
        let pool_size = std::thread::available_parallelism()
            .map(|p| (p.get() / 2).max(1).min(4)) // Use half of CPU cores, max 4
            .unwrap_or(2); // Fallback to 2 sessions
        
        let session_pool = OnnxSessionPool::new(&model_path, pool_size).await
            .context("Failed to create ONNX session pool")?;
        
        tracing::info!("✅ ONNX Engine: Session pool created with {} sessions", pool_size);
        
        // Model info available if needed for future enhancements
        let _model_info = match model_id {
            ModelId::Kokoro => crate::model::ModelInfo::kokoro(),
            ModelId::Chatterbox => crate::model::ModelInfo::chatterbox(),
            ModelId::Dia => crate::model::ModelInfo::dia(),
        };
        
        // Text processing is now handled by Python layer using ttstokenizer
        // This engine only handles neural inference with pre-processed token IDs
        tracing::info!("Model loaded - text processing delegated to Python layer");
        
        self.session_pool = Some(session_pool);
        self.current_model = Some(model_id);
        
        tracing::info!("✅ Successfully loaded neural model: {:?}", model_id);
        Ok(())
    }
    
    /// Get the currently loaded model
    pub fn current_model(&self) -> Option<ModelId> {
        self.current_model
    }
    
    /// Debug model inputs and requirements - 2025 Fix for tensor shape issues
    pub fn debug_model_inputs(&self) -> Result<()> {
        tracing::debug!("=== MODEL DEBUG INFO ===");
        tracing::debug!("Current model loaded: {:?}", self.current_model);
        
        if let Some(pool) = &self.session_pool {
            let stats = pool.stats();
            tracing::debug!("Session pool stats: {}", stats);
            tracing::debug!("Pool health: {}", pool.is_healthy());
        } else {
            tracing::debug!("No session pool loaded");
        }
        
        Ok(())
    }
    
    /// Get session pool statistics
    pub fn get_pool_stats(&self) -> Option<session_pool::PoolStats> {
        self.session_pool.as_ref().map(|pool| pool.stats())
    }

    /// Synthesize text to audio using neural model (DEPRECATED - use synthesize_from_tokens)
    /// This method is kept for backward compatibility but delegates to Python for text processing
    pub async fn synthesize(&mut self, _text: &str, _model_id: ModelId, _voice_id: Option<&str>) -> Result<Vec<f32>> {
        return Err(anyhow::anyhow!(
            "Direct text synthesis deprecated. Use Python phoneme processor first:\n\
             1. KokoroPhonemeProcessor.process_text(text) -> tokens\n\
             2. OnnxTtsEngine.synthesize_from_tokens(tokens)\n\
             This ensures proper 2025 Kokoro phoneme-based processing."
        ));
    }
    
    /// Preprocess text for TTS (normalize, clean) - Fixed for Kokoro TTS
    pub fn preprocess_text(&self, text: &str) -> String {
        // Unicode normalization (NFC is better for TTS than NFD)
        let normalized: String = text.nfc().collect();
        
        // 2025 Fix: Preserve proper linguistic features for Kokoro TTS
        // Keep capitalization, punctuation, and natural language structure
        let cleaned = normalized
            .chars()
            .filter(|c| {
                // Keep letters, numbers, spaces, and important punctuation
                c.is_alphabetic() || c.is_numeric() || c.is_whitespace() || 
                matches!(*c, '.' | ',' | '!' | '?' | ':' | ';' | '-' | '\'' | '"')
            })
            .collect::<String>()
            .trim()
            .to_string();
        
        // 2025 Fix: NO startoftext/endoftext tokens - Kokoro uses padding tokens
        // Return clean text without special tokens - padding will be handled in tokenization
        if cleaned.is_empty() {
            "Hello world".to_string() // Fallback for empty input
        } else {
            cleaned
        }
    }
    
    /// Synthesize audio from pre-processed token IDs (from Python phoneme processor)
    pub async fn synthesize_from_tokens(
        &mut self, 
        input_ids: Vec<i64>, 
        style_vector: Vec<f32>, 
        speed: f32,
        model_id: ModelId
    ) -> Result<Vec<f32>> {
        tracing::debug!("ONNX Engine: Starting synthesis from {} pre-processed tokens", input_ids.len());
        
        // Ensure correct model is loaded
        if self.current_model != Some(model_id) {
            tracing::debug!("ONNX Engine: Loading model {:?}...", model_id);
            self.load_model(model_id).await.context("Failed to load model in synthesize")?;
        }
        
        // Validate input constraints
        if input_ids.len() > 512 {
            return Err(anyhow::anyhow!("Token sequence too long: {} tokens (max 512)", input_ids.len()));
        }
        
        if style_vector.len() != 256 {
            return Err(anyhow::anyhow!("Style vector must be 256 dimensions, got {}", style_vector.len()));
        }
        
        // Validate style vector for neural network stability
        if !self.validate_style_vector(&style_vector) {
            return Err(anyhow::anyhow!("Invalid style vector detected - contains values that would cause model instability"));
        }
        
        // Perform ONNX inference with timeout protection
        tracing::info!("🔒 Starting synthesis with 30-second timeout protection");
        match tokio::time::timeout(
            std::time::Duration::from_secs(30),
            self.perform_inference_with_tokens(input_ids, style_vector, speed)
        ).await {
            Ok(result) => result,
            Err(_) => {
                tracing::error!("❌ Synthesis timeout after 30 seconds - model may be stuck");
                Err(anyhow::anyhow!("Synthesis timeout: Model inference hung for >30 seconds. This usually indicates invalid input data or model corruption."))
            }
        }
    }
    
    /// Validate style vector to prevent neural network instability
    fn validate_style_vector(&self, style_vector: &[f32]) -> bool {
        // Check for NaN/Inf values (immediate model corruption)
        if style_vector.iter().any(|&x| !x.is_finite()) {
            tracing::error!("❌ Style vector contains NaN/Inf values");
            return false;
        }
        
        // Check for extreme values (gradient explosion risk)
        if style_vector.iter().any(|&x| x.abs() > 10.0) {
            tracing::error!("❌ Style vector contains extreme values (max: {})", 
                           style_vector.iter().map(|&x| x.abs()).fold(0.0f32, f32::max));
            return false;
        }
        
        // Check for all zeros (failed loading indicator)
        if style_vector.iter().all(|&x| x.abs() < 0.001) {
            tracing::error!("❌ Style vector appears to be all zeros");
            return false;
        }
        
        // Check for high variance (random values indicator)
        let mean = style_vector.iter().sum::<f32>() / style_vector.len() as f32;
        let variance = style_vector.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / style_vector.len() as f32;
        
        if mean.abs() < 0.01 && variance > 0.8 {
            tracing::error!("❌ Style vector appears to be random values (mean: {:.3}, variance: {:.3})", mean, variance);
            return false;
        }
        
        tracing::debug!("✅ Style vector validation passed (mean: {:.3}, variance: {:.3}, range: [{:.3}, {:.3}])", 
                       mean, variance,
                       style_vector.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
                       style_vector.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
        true
    }
    
    
    // Removed adaptive tensor function - simplified approach for immediate fix

    /// Perform ONNX inference with pre-processed token IDs
    async fn perform_inference_with_tokens(
        &self, 
        input_ids: Vec<i64>, 
        style_vector: Vec<f32>, 
        speed: f32
    ) -> Result<Vec<f32>> {
        // Acquire session from pool
        tracing::info!("🔄 Acquiring ONNX session from pool...");
        let session_guard = self.session_pool.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No session pool available"))?
            .acquire_session().await
            .context("Failed to acquire session from pool")?;
        
        let tokens_count = input_ids.len();
        tracing::info!("Creating tensors: {} tokens, {} style values, speed: {}", 
                      tokens_count, style_vector.len(), speed);
        
        // Run inference with ONNX Runtime
        tracing::info!("🚀 ONNX Engine: Running inference...");
        let audio_data: Vec<f32> = {
            // Create inputs with actual data
            let mut attempt_inputs: std::collections::HashMap<String, ort::value::Value> = std::collections::HashMap::new();
            
            // Create tokens tensor with pre-processed tokens
            let tokens_tensor = ort::value::Tensor::from_array(([1, input_ids.len()], input_ids))
                .context("Failed to create tokens tensor")?;
            attempt_inputs.insert("tokens".to_string(), tokens_tensor.into());
            
            // Create style tensor with pre-processed style vector
            let style_tensor = ort::value::Tensor::from_array(([1, style_vector.len()], style_vector))
                .context("Failed to create style tensor")?;
            attempt_inputs.insert("style".to_string(), style_tensor.into());
            
            // Create speed tensor
            let speed_tensor = ort::value::Tensor::from_array(([1], vec![speed]))
                .context("Failed to create speed tensor")?;
            attempt_inputs.insert("speed".to_string(), speed_tensor.into());
            
            // Lock the mutex to get mutable access to the session
            let mut session = session_guard.session.lock()
                .map_err(|e| anyhow::anyhow!("Failed to acquire session lock: {}", e))?;
            let outputs = session.run(attempt_inputs)
                .map_err(|e| anyhow::anyhow!("ONNX inference failed: {}", e))?;
            
            // Extract audio data using ort 2.0.0-rc.10 API
            if let Some(output) = outputs.get("audio") {
                let (_, data) = output.try_extract_tensor::<f32>()
                    .context("Failed to extract audio data from 'audio' output")?;
                data.to_vec()
            } else if let Some(output) = outputs.get("output") {
                let (_, data) = output.try_extract_tensor::<f32>()
                    .context("Failed to extract audio data from 'output' output")?;
                data.to_vec()
            } else if let Some((_, output)) = outputs.iter().next() {
                let (_, data) = output.try_extract_tensor::<f32>()
                    .context("Failed to extract audio data from first output")?;
                data.to_vec()
            } else {
                return Err(anyhow::anyhow!("No audio output found in model"));
            }
        };
        
        tracing::info!("✅ Generated {} audio samples from {} tokens at 24kHz", audio_data.len(), tokens_count);
        Ok(audio_data)
    }
    
    
    /// Postprocess raw model output
    pub fn postprocess_audio(&self, raw_audio: &[f32]) -> Vec<f32> {
        // Normalize audio to [-1.0, 1.0] range
        let max_val = raw_audio.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        
        if max_val > 0.0 {
            raw_audio.iter().map(|&x| (x / max_val).clamp(-1.0, 1.0)).collect()
        } else {
            raw_audio.to_vec()
        }
    }
    
    fn load_voice_embedding(&self, model_id: &str, voice_id: &str) -> VocalizeResult<Vec<f32>> {
        // Construct voice file path based on model cache structure
        let cache_dir = self.model_manager.cache_dir.clone();
        
        // 2025 Fix: Use correct path structure that matches ModelRegistry
        let model_cache = match model_id {
            "kokoro" => cache_dir.join("models--direct_download").join("local"),
            _ => return Err(VocalizeError::SynthesisError {
                message: format!("Unsupported model for voice loading: {}", model_id)
            }),
        };
        
        // 2025 Fix: Look for voice files in the correct location
        // Try multiple possible voice file locations
        let voice_file_locations = vec![
            model_cache.join("voices").join(format!("{}.bin", voice_id)),
            model_cache.join(format!("voice_{}.bin", voice_id)),
            model_cache.join("voices-v1.0.bin"), // Single voices file
        ];
        
        let mut voice_file = None;
        for location in voice_file_locations {
            if location.exists() {
                voice_file = Some(location);
                break;
            }
        }
        
        let voice_file = voice_file.ok_or_else(|| VocalizeError::SynthesisError {
            message: format!(
                "Voice file not found for '{}' in model cache. Tried locations: {:?}. \
                 Run 'vocalize models download {}' to get voices.", 
                voice_id, 
                model_cache.join("voices").join(format!("{}.bin", voice_id)),
                model_id
            )
        })?;
        
        tracing::debug!("Loading voice embedding from: {:?}", voice_file);
        
        // 2025 Fix: Enhanced voice embedding loading with fallback support
        let voice_embedding = if voice_file.file_name().unwrap_or_default() == "voices-v1.0.bin" {
            // Single voices file containing multiple embeddings
            self.load_voice_from_combined_file(&voice_file, voice_id)?
        } else {
            // Individual voice file
            self.load_voice_from_individual_file(&voice_file, voice_id)?
        };
        
        tracing::info!("✅ Loaded voice embedding '{}': {} floats, range [{:.3}, {:.3}]", 
                      voice_id, voice_embedding.len(),
                      voice_embedding.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
                      voice_embedding.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
        
        Ok(voice_embedding)
    }
    
    /// Load voice embedding from a combined voices file
    fn load_voice_from_combined_file(&self, voice_file: &std::path::Path, voice_id: &str) -> VocalizeResult<Vec<f32>> {
        // Load and parse combined voices file
        let voice_data = std::fs::read(voice_file)
            .map_err(|e| VocalizeError::SynthesisError {
                message: format!("Failed to read combined voices file: {}", e)
            })?;
        
        // Parse binary format: [4 bytes: voice_count][voice_entries][voice_data]
        // Each entry: [32 bytes: voice_id][4 bytes: offset][4 bytes: size]
        
        if voice_data.len() < 4 {
            return Err(VocalizeError::SynthesisError {
                message: "Combined voices file too small to contain header".to_string()
            });
        }
        
        // Read voice count from first 4 bytes (little-endian)
        let voice_count = u32::from_le_bytes([voice_data[0], voice_data[1], voice_data[2], voice_data[3]]) as usize;
        
        tracing::debug!("Combined voices file contains {} voices", voice_count);
        
        // Calculate expected header size: 4 + (voice_count * 40)
        let header_size = 4 + (voice_count * 40);
        if voice_data.len() < header_size {
            return Err(VocalizeError::SynthesisError {
                message: format!("Combined voices file too small for {} voice entries", voice_count)
            });
        }
        
        // Search for the requested voice_id
        for i in 0..voice_count {
            let entry_offset = 4 + (i * 40);
            
            // Read voice_id (32 bytes, null-terminated string)
            let voice_id_bytes = &voice_data[entry_offset..entry_offset + 32];
            let voice_id_str = std::str::from_utf8(voice_id_bytes)
                .unwrap_or("")
                .trim_end_matches('\0');
            
            if voice_id_str == voice_id {
                // Found matching voice, read offset and size
                let offset_bytes = &voice_data[entry_offset + 32..entry_offset + 36];
                let size_bytes = &voice_data[entry_offset + 36..entry_offset + 40];
                
                let data_offset = u32::from_le_bytes([offset_bytes[0], offset_bytes[1], offset_bytes[2], offset_bytes[3]]) as usize;
                let data_size = u32::from_le_bytes([size_bytes[0], size_bytes[1], size_bytes[2], size_bytes[3]]) as usize;
                
                tracing::debug!("Found voice '{}' at offset {} with size {}", voice_id, data_offset, data_size);
                
                // Validate bounds
                if data_offset + data_size > voice_data.len() {
                    return Err(VocalizeError::SynthesisError {
                        message: format!("Voice data for '{}' extends beyond file bounds", voice_id)
                    });
                }
                
                // Extract voice data and convert to f32 array
                let voice_bytes = &voice_data[data_offset..data_offset + data_size];
                
                if voice_bytes.len() % 4 != 0 {
                    return Err(VocalizeError::SynthesisError {
                        message: format!("Voice data size {} not divisible by 4", voice_bytes.len())
                    });
                }
                
                let mut voice_embedding = Vec::with_capacity(voice_bytes.len() / 4);
                for chunk in voice_bytes.chunks_exact(4) {
                    let float_val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    if !float_val.is_finite() {
                        return Err(VocalizeError::SynthesisError {
                            message: format!("Invalid float value in voice '{}': {}", voice_id, float_val)
                        });
                    }
                    voice_embedding.push(float_val);
                }
                
                // For Kokoro, we expect 256-dimensional style vectors
                if voice_embedding.len() >= 256 {
                    voice_embedding.truncate(256);
                }
                
                tracing::info!("✅ Loaded voice '{}' from combined file: {} floats", voice_id, voice_embedding.len());
                return Ok(voice_embedding);
            }
        }
        
        // Voice not found in combined file
        Err(VocalizeError::SynthesisError {
            message: format!("Voice '{}' not found in combined voices file", voice_id)
        })
    }
    
    /// Load voice embedding from an individual voice file
    fn load_voice_from_individual_file(&self, voice_file: &std::path::Path, _voice_id: &str) -> VocalizeResult<Vec<f32>> {
        // Load binary voice embedding with validation
        let voice_data = std::fs::read(voice_file)
            .map_err(|e| VocalizeError::SynthesisError {
                message: format!("Failed to read voice file: {}", e)
            })?;
        
        // 2025 Fix: Validate voice file integrity
        if voice_data.len() % 4 != 0 {
            return Err(VocalizeError::SynthesisError {
                message: format!("Invalid voice file format: size {} not divisible by 4", voice_data.len())
            });
        }
        
        if voice_data.is_empty() {
            return Err(VocalizeError::SynthesisError {
                message: "Voice file is empty".to_string()
            });
        }
        
        // Convert bytes to f32 array (little-endian)
        let float_count = voice_data.len() / 4;
        let mut voice_embedding = Vec::with_capacity(float_count);
        
        for chunk in voice_data.chunks_exact(4) {
            let float_val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            // 2025 Fix: Validate float values are reasonable (not NaN/infinite)
            if !float_val.is_finite() {
                return Err(VocalizeError::SynthesisError {
                    message: format!("Invalid float value in voice embedding: {}", float_val)
                });
            }
            voice_embedding.push(float_val);
        }
        
        // 2025 Fix: Kokoro voice embeddings are actually (510, 256) = 130,560 floats
        // We need the first 256 values for the style vector
        let expected_total_size = 510 * 256; // 130,560
        let style_embedding_size = 256;
        
        if voice_embedding.len() == expected_total_size {
            // Extract the style vector (first 256 values)
            voice_embedding.truncate(style_embedding_size);
            tracing::info!("✅ Extracted style vector from voice embedding: {} floats", voice_embedding.len());
        } else if voice_embedding.len() == style_embedding_size {
            // Already correct size
            tracing::info!("✅ Voice embedding already correct size: {} floats", voice_embedding.len());
        } else {
            tracing::warn!("Unexpected voice embedding size {} (expected {} or {})", 
                          voice_embedding.len(), expected_total_size, style_embedding_size);
        }
        
        Ok(voice_embedding)
    }
    
}