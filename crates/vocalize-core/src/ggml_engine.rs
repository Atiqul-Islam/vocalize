//! GGML-based neural TTS engine implementation
//! Fast inference with memory-mapped models

pub mod gguf_format;
pub mod phoneme_processor;
pub mod simple_tensor;
pub mod tensor_ops;
pub mod vits_model;

use std::path::PathBuf;
use anyhow::{Result, Context};
use memmap2::{Mmap, MmapOptions};
use std::fs::File;
use std::collections::HashMap;
use parking_lot::RwLock;
use std::sync::Arc;
use byteorder::{ByteOrder, LittleEndian};
use simple_tensor::{Device, SimpleTensor};
use vits_model::VITSModel;

use crate::model::{ModelManager, ModelId};
use crate::{VocalizeResult, VocalizeError};
use gguf_format::{GGUFFile, TensorInfo};
use phoneme_processor::PhonemeProcessor;

/// GGML-based neural TTS engine
#[derive(Debug)]
pub struct GGMLTtsEngine {
    model_manager: ModelManager,
    loaded_model: Option<LoadedModel>,
    phoneme_processor: Arc<PhonemeProcessor>,
    device: Device,
    models: HashMap<ModelId, LoadedModel>,
    vits_model: VITSModel,
}

#[derive(Debug)]
struct LoadedModel {
    model_id: ModelId,
    model_data: Arc<Mmap>,
    metadata: HashMap<String, String>,
    tensors: HashMap<String, TensorInfo>,
    weights: Arc<RwLock<HashMap<String, SimpleTensor>>>,
}

impl GGMLTtsEngine {
    /// Create a new GGML TTS engine
    pub async fn new(cache_dir: PathBuf) -> Result<Self> {
        use std::time::Instant;
        let total_start = Instant::now();
        
        tracing::info!("GGML Engine: Initializing with cache dir: {:?}", cache_dir);
        
        // Initialize device (CPU only for now)
        let device = Device::Cpu;
        tracing::info!("GGML Engine: Using device: {:?}", device);
        
        // Initialize phoneme processor
        let phoneme_start = Instant::now();
        let phoneme_processor = Arc::new(PhonemeProcessor::new()?);
        eprintln!("  ⏱️  [GGML] Phoneme processor init: {:.3}s", phoneme_start.elapsed().as_secs_f32());
        
        let model_manager = ModelManager::new(cache_dir);
        
        eprintln!("  ⏱️  [GGML] Total engine new(): {:.3}s", total_start.elapsed().as_secs_f32());
        
        Ok(Self {
            model_manager,
            loaded_model: None,
            phoneme_processor,
            device,
            models: HashMap::new(),
            vits_model: VITSModel::new(),
        })
    }
    
    /// Create a new GGML TTS engine with cross-platform cache directory
    pub async fn new_with_default_cache() -> Result<Self> {
        use directories::ProjectDirs;
        
        let proj_dirs = ProjectDirs::from("ai", "Vocalize", "vocalize")
            .ok_or_else(|| anyhow::anyhow!("Failed to determine project directories"))?;
        
        let cache_dir = proj_dirs.cache_dir().join("models");
        
        tracing::info!("GGML Engine: Using cross-platform cache directory: {:?}", cache_dir);
        
        Self::new(cache_dir).await
    }
    
    /// Load a specific model for synthesis
    pub async fn load_model(&mut self, model_id: ModelId, model_path: String) -> Result<()> {
        use std::time::Instant;
        let total_start = Instant::now();
        
        tracing::info!("🔄 GGML Engine: Loading model {:?} from {}", model_id, model_path);
        
        // Memory map the GGUF file
        let mmap_start = Instant::now();
        let file = File::open(&model_path)
            .with_context(|| format!("Failed to open model file: {}", model_path))?;
        
        let model_data = Arc::new(unsafe {
            MmapOptions::new()
                .map(&file)
                .with_context(|| format!("Failed to memory map model file: {}", model_path))?
        });
        eprintln!("  ⏱️  [GGML] Memory mapping: {:.3}s", mmap_start.elapsed().as_secs_f32());
        
        // Parse GGUF file
        let parse_start = Instant::now();
        let gguf = GGUFFile::parse(&model_data)?;
        eprintln!("  ⏱️  [GGML] GGUF parsing: {:.3}s", parse_start.elapsed().as_secs_f32());
        
        // Extract metadata
        let mut metadata = HashMap::new();
        for (key, value) in &gguf.metadata {
            metadata.insert(key.clone(), value.to_string());
        }
        
        tracing::info!("✅ GGML Engine: Model loaded with {} tensors", gguf.tensors.len());
        
        // Create loaded model
        let loaded_model = LoadedModel {
            model_id,
            model_data,
            metadata,
            tensors: gguf.tensors,
            weights: Arc::new(RwLock::new(HashMap::new())),
        };
        
        // Store loaded model
        self.loaded_model = Some(loaded_model);
        // Note: We're only using loaded_model for now, not models HashMap
        
        eprintln!("  ⏱️  [GGML] Total load_model: {:.3}s", total_start.elapsed().as_secs_f32());
        
        Ok(())
    }
    
    /// Get the currently loaded model
    pub fn current_model(&self) -> Option<ModelId> {
        self.loaded_model.as_ref().map(|m| m.model_id)
    }
    
    /// Process text to phonemes using fast Rust implementation
    pub fn process_text(&self, text: &str) -> Result<Vec<i64>> {
        self.phoneme_processor.process_text(text)
    }
    
    /// Synthesize audio from phoneme tokens
    pub async fn synthesize_from_tokens(
        &mut self, 
        input_ids: Vec<i64>, 
        style_vector: Vec<f32>, 
        speed: f32,
        model_id: ModelId,
        model_path: String
    ) -> Result<Vec<f32>> {
        use std::time::Instant;
        let total_start = Instant::now();
        
        tracing::debug!("GGML Engine: Starting synthesis from {} tokens", input_ids.len());
        
        // Ensure correct model is loaded
        if self.current_model() != Some(model_id) {
            tracing::debug!("GGML Engine: Loading model {:?}", model_id);
            self.load_model(model_id, model_path).await?;
        }
        
        let model = self.loaded_model.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No model loaded"))?;
        
        // Validate inputs
        if input_ids.len() > 512 {
            return Err(anyhow::anyhow!("Token sequence too long: {} tokens (max 512)", input_ids.len()));
        }
        
        if style_vector.len() != 256 {
            return Err(anyhow::anyhow!("Style vector must be 256 dimensions, got {}", style_vector.len()));
        }
        
        // Perform inference
        let inference_start = Instant::now();
        let audio = self.perform_inference(
            &model,
            &input_ids,
            &style_vector,
            speed
        ).await?;
        eprintln!("  ⏱️  [GGML] Inference: {:.3}s", inference_start.elapsed().as_secs_f32());
        
        eprintln!("  ⏱️  [GGML] Total synthesize_from_tokens: {:.3}s", total_start.elapsed().as_secs_f32());
        
        Ok(audio)
    }
    
    /// Perform VITS inference
    async fn perform_inference(
        &self,
        model: &LoadedModel,
        input_ids: &[i64],
        style_vector: &[f32],
        speed: f32,
    ) -> Result<Vec<f32>> {
        use std::time::Instant;
        
        // Convert inputs to tensors
        let tensor_start = Instant::now();
        
        // Create input tensors
        let tokens = SimpleTensor::from_vec(
            input_ids.iter().map(|&x| x as f32).collect::<Vec<_>>(),
            &[1, input_ids.len()]
        )?;
        
        let style = SimpleTensor::from_vec(
            style_vector.to_vec(),
            &[1, style_vector.len()]
        )?;
        
        let speed_tensor = SimpleTensor::new(&[speed])?;
        
        eprintln!("  ⏱️  [GGML] Tensor creation: {:.3}s", tensor_start.elapsed().as_secs_f32());
        
        // Load model weights on demand
        let weights_start = Instant::now();
        self.ensure_weights_loaded(model)?;
        let weights = model.weights.read();
        eprintln!("  ⏱️  [GGML] Weight loading: {:.3}s", weights_start.elapsed().as_secs_f32());
        
        // Run VITS forward pass
        let forward_start = Instant::now();
        
        // Use the real VITS model
        let audio_vec = self.vits_model.synthesize(
            input_ids,
            style_vector,
            &*weights
        )?;
        
        eprintln!("  ⏱️  [GGML] Forward pass: {:.3}s", forward_start.elapsed().as_secs_f32());
        
        // Apply speed adjustment
        let audio_vec = if (speed - 1.0).abs() > 0.01 {
            self.adjust_speed(audio_vec, speed)
        } else {
            audio_vec
        };
        
        Ok(audio_vec)
    }
    
    /// Ensure model weights are loaded into memory
    fn ensure_weights_loaded(&self, model: &LoadedModel) -> Result<()> {
        let mut weights = model.weights.write();
        
        if !weights.is_empty() {
            return Ok(()); // Already loaded
        }
        
        tracing::debug!("Loading model weights from GGUF...");
        
        // Load each tensor from the memory-mapped file
        for (name, info) in &model.tensors {
            let tensor = self.load_tensor_from_gguf(
                &model.model_data,
                info
            )?;
            weights.insert(name.clone(), tensor);
        }
        
        tracing::debug!("Loaded {} weight tensors", weights.len());
        Ok(())
    }
    
    /// Load a single tensor from GGUF file
    fn load_tensor_from_gguf(
        &self,
        data: &[u8],
        info: &TensorInfo,
    ) -> Result<SimpleTensor> {
        use gguf_format::GGMLType;
        
        // Get tensor data slice
        let tensor_data = &data[info.offset..info.offset + info.size];
        
        match info.dtype {
            GGMLType::F32 => {
                let mut values = vec![0.0f32; tensor_data.len() / 4];
                byteorder::LittleEndian::read_f32_into(tensor_data, &mut values);
                SimpleTensor::from_vec(values, &info.shape)?
            }
            GGMLType::F16 => {
                // Convert f16 to f32
                let mut values = Vec::with_capacity(tensor_data.len() / 2);
                for chunk in tensor_data.chunks_exact(2) {
                    let f16_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    let f16_val = half::f16::from_bits(f16_bits);
                    values.push(f16_val.to_f32());
                }
                SimpleTensor::from_vec(values, &info.shape)?
            }
            GGMLType::Q8_0 => {
                // Dequantize Q8_0
                let values = self.dequantize_q8_0(tensor_data, info.numel())?;
                SimpleTensor::from_vec(values, &info.shape)?
            }
            GGMLType::Q4_0 => {
                // Dequantize Q4_0
                let values = self.dequantize_q4_0(tensor_data, info.numel())?;
                SimpleTensor::from_vec(values, &info.shape)?
            }
            _ => Err(anyhow::anyhow!("Unsupported tensor type: {:?}", info.dtype))
        }
    }
    
    /// Dequantize Q8_0 format
    /// Q8_0: blocks of 32 elements quantized to 8-bit with a 16-bit scale factor
    fn dequantize_q8_0(&self, data: &[u8], numel: usize) -> Result<Vec<f32>> {
        let mut result = Vec::with_capacity(numel);
        let block_size = 32;
        let bytes_per_block = 2 + block_size; // 2 bytes for scale + 32 bytes for values
        
        // Process complete blocks
        for chunk in data.chunks_exact(bytes_per_block) {
            // Read scale (f16)
            let scale_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            let scale = half::f16::from_bits(scale_bits).to_f32();
            
            // Dequantize values
            for &byte in &chunk[2..] {
                let val = byte as i8;
                result.push(val as f32 * scale);
            }
        }
        
        // Handle remainder if any
        let remainder = data.len() % bytes_per_block;
        if remainder > 2 {
            let last_chunk = &data[data.len() - remainder..];
            let scale_bits = u16::from_le_bytes([last_chunk[0], last_chunk[1]]);
            let scale = half::f16::from_bits(scale_bits).to_f32();
            
            for &byte in &last_chunk[2..] {
                let val = byte as i8;
                result.push(val as f32 * scale);
                if result.len() >= numel {
                    break;
                }
            }
        }
        
        // Ensure exact size
        result.truncate(numel);
        Ok(result)
    }
    
    /// Dequantize Q4_0 format
    /// Q4_0: blocks of 32 elements quantized to 4-bit with a 16-bit scale factor
    fn dequantize_q4_0(&self, data: &[u8], numel: usize) -> Result<Vec<f32>> {
        let mut result = Vec::with_capacity(numel);
        let block_size = 32;
        let bytes_per_block = 2 + block_size / 2; // 2 bytes for scale + 16 bytes for values (2 values per byte)
        
        // Process complete blocks
        for chunk in data.chunks_exact(bytes_per_block) {
            // Read scale (f16)
            let scale_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            let scale = half::f16::from_bits(scale_bits).to_f32();
            
            // Dequantize values (2 4-bit values per byte)
            for &byte in &chunk[2..] {
                // Low 4 bits
                let low = (byte & 0x0F) as i8 - 8; // Center around 0
                result.push(low as f32 * scale);
                
                if result.len() >= numel {
                    break;
                }
                
                // High 4 bits
                let high = ((byte >> 4) & 0x0F) as i8 - 8; // Center around 0
                result.push(high as f32 * scale);
                
                if result.len() >= numel {
                    break;
                }
            }
        }
        
        // Ensure exact size
        result.truncate(numel);
        Ok(result)
    }
    
    fn adjust_speed(&self, audio: Vec<f32>, speed: f32) -> Vec<f32> {
        // Simple speed adjustment by resampling
        // In production: Use proper resampling algorithm
        if speed == 1.0 {
            return audio;
        }
        
        let new_len = (audio.len() as f32 / speed) as usize;
        let mut resampled = Vec::with_capacity(new_len);
        
        for i in 0..new_len {
            let src_idx = (i as f32 * speed) as usize;
            if src_idx < audio.len() {
                resampled.push(audio[src_idx]);
            }
        }
        
        resampled
    }
}

// Compatibility layer for existing code
impl GGMLTtsEngine {
    /// Legacy method - kept for compatibility
    pub async fn synthesize(&mut self, _text: &str, _model_id: ModelId, _voice_id: Option<&str>) -> Result<Vec<f32>> {
        Err(anyhow::anyhow!(
            "Direct text synthesis deprecated. Use process_text() followed by synthesize_from_tokens()"
        ))
    }
    
    pub fn preprocess_text(&self, text: &str) -> String {
        // Basic text normalization
        text.trim().to_string()
    }
}