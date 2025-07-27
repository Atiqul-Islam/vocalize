//! Python bindings for Vocalize TTS engine
//!
//! This crate provides comprehensive Python bindings for the Vocalize text-to-speech engine
//! using PyO3. It exposes the full TTS functionality with proper async support.

use pyo3::prelude::*;

// Re-export submodules
mod error;
mod runtime_manager;
mod voice_manager;
mod audio_writer;
mod audio_device;

use error::{PyVocalizeError, VocalizeException};
use voice_manager::{PyVoiceManager, PyVoice, PyGender, PyVoiceStyle};
use audio_writer::{PyAudioWriter, PyAudioFormat, PyEncodingSettings};
use audio_device::{PyAudioDevice, PyAudioConfig, PyAudioDeviceInfo, PyPlaybackState};

// Re-export Python types
pub use voice_manager::PyVoice as Voice;
pub use voice_manager::PyVoiceManager as VoiceManager;
pub use audio_writer::PyAudioWriter as AudioWriter;
pub use audio_device::PyAudioDevice as AudioDevice;


/// 2025 Neural TTS synthesis using pre-processed tokens (new phoneme pipeline)
#[pyfunction]
#[pyo3(signature = (input_ids, style_vector, speed, model_id, model_path))]
fn synthesize_from_tokens_neural(
    input_ids: Vec<i64>,
    style_vector: Vec<f32>,
    speed: f32,
    model_id: Option<String>,
    model_path: String
) -> PyResult<Vec<f32>> {
    use std::time::Instant;
    let total_start = Instant::now();
    
    // Validate inputs
    let validation_start = Instant::now();
    if input_ids.is_empty() {
        return Err(PyVocalizeError::new_err("Input IDs cannot be empty".to_string()));
    }
    
    if style_vector.len() != 256 {
        return Err(PyVocalizeError::new_err(format!("Style vector must be 256 dimensions, got {}", style_vector.len())));
    }
    
    if !(0.1..=3.0).contains(&speed) {
        return Err(PyVocalizeError::new_err(format!("Speed must be between 0.1 and 3.0, got {}", speed)));
    }
    
    if input_ids.len() > 512 {
        return Err(PyVocalizeError::new_err(format!("Token sequence too long: {} tokens (max 512)", input_ids.len())));
    }
    eprintln!("  ⏱️  [Rust] Input validation: {:.3}s", validation_start.elapsed().as_secs_f32());
    
    println!("🔊 2025 TTS: Using pre-processed tokens ({} tokens, {} style dims, speed: {})", 
             input_ids.len(), style_vector.len(), speed);
    
    // Use ONNX engine directly for token-based synthesis
    use vocalize_core::{onnx_engine::OnnxTtsEngine, model::ModelId};
    
    // Create runtime for async operations
    let runtime_start = Instant::now();
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| PyVocalizeError::new_err(format!("Failed to create async runtime: {}", e)))?;
    eprintln!("  ⏱️  [Rust] Tokio runtime creation: {:.3}s", runtime_start.elapsed().as_secs_f32());
    
    rt.block_on(async {
        // Create ONNX engine with cross-platform cache directory
        let engine_start = Instant::now();
        let mut engine = OnnxTtsEngine::new_with_default_cache().await
            .map_err(|e| PyVocalizeError::new_err(format!("Failed to create ONNX engine: {}", e)))?;
        eprintln!("  ⏱️  [Rust] ONNX engine creation: {:.3}s", engine_start.elapsed().as_secs_f32());
        
        // Determine model ID
        let model = match model_id.as_deref().unwrap_or("kokoro") {
            "kokoro" => ModelId::Kokoro,
            "chatterbox" => ModelId::Chatterbox,
            "dia" => ModelId::Dia,
            _ => ModelId::Kokoro, // Default fallback
        };
        
        // Synthesize using the new token-based method with model path
        let synthesis_start = Instant::now();
        let audio_data = engine.synthesize_from_tokens(
            input_ids,
            style_vector,
            speed,
            model,
            model_path
        ).await
        .map_err(|e| PyVocalizeError::new_err(format!("Token synthesis failed: {}", e)))?;
        eprintln!("  ⏱️  [Rust] Token synthesis: {:.3}s", synthesis_start.elapsed().as_secs_f32());
        
        eprintln!("  ⏱️  [Rust] Total synthesis time: {:.3}s", total_start.elapsed().as_secs_f32());
        println!("✅ 2025 token synthesis completed: {} samples generated", audio_data.len());
        Ok(audio_data)
    })
}




/// Save neural TTS audio data to a file
#[pyfunction] 
fn save_audio_neural(audio_data: Vec<f32>, output_path: String, format: Option<String>) -> PyResult<()> {
    let format_str = format.unwrap_or_else(|| "wav".to_string());
    let audio_format = match format_str.as_str() {
        "wav" => PyAudioFormat::Wav,
        "mp3" => PyAudioFormat::Mp3,
        "flac" => PyAudioFormat::Flac,
        "ogg" => PyAudioFormat::Ogg,
        _ => return Err(PyVocalizeError::new_err(format!("Unsupported format: {format_str}"))),
    };
    
    // Validate neural audio data
    if audio_data.is_empty() {
        return Err(PyVocalizeError::new_err("Neural audio data cannot be empty".to_string()));
    }
    
    // Use the actual audio writer from vocalize-core
    use vocalize_core::{AudioWriter, AudioFormat, AudioData};
    use std::path::Path;
    
    // Convert PyAudioFormat to AudioFormat
    let core_format = match audio_format {
        PyAudioFormat::Wav => AudioFormat::Wav,
        PyAudioFormat::Mp3 => AudioFormat::Mp3,
        PyAudioFormat::Flac => AudioFormat::Flac,
        PyAudioFormat::Ogg => AudioFormat::Ogg,
    };
    
    // Create output path
    let path = Path::new(&output_path);
    
    // Create audio writer
    let writer = AudioWriter::new();
    
    // AudioData is just Vec<f32>, so use audio_data directly
    let audio_data_ref: &AudioData = &audio_data;
    
    // Create runtime for async operations
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| PyVocalizeError::new_err(format!("Failed to create async runtime: {}", e)))?;
    
    // Write audio data
    rt.block_on(async {
        writer.write_file(audio_data_ref, path, core_format, None).await
            .map_err(|e| PyVocalizeError::new_err(format!("Failed to write audio file: {}", e)))
    })?;
    
    Ok(())
}

/// Python module for Vocalize TTS functionality
#[pymodule]
fn vocalize_rust(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // Set up ONNX Runtime DLL path IMMEDIATELY on module load
    // This must happen before ANY ort code is touched
    #[cfg(target_os = "windows")]
    {
        use winapi::um::libloaderapi::{LoadLibraryExW, LOAD_WITH_ALTERED_SEARCH_PATH};
        use winapi::um::errhandlingapi::GetLastError;
        use std::ffi::OsStr;
        use std::os::windows::ffi::OsStrExt;
        use std::ptr::null_mut;
        
        if std::env::var("ORT_DYLIB_PATH").is_err() {
            // Get the site-packages path
            let sys = _py.import("sys")?;
            let prefix: String = sys.getattr("prefix")?.extract()?;
            let dll_dir = format!("{}\\Lib\\site-packages\\vocalize_rust", prefix);
            
            // Note: We bundle our own ONNX Runtime DLLs, so no need to check System32
            
            // Add directory to Python's DLL search path (for Python 3.8+)
            let os = _py.import("os")?;
            if let Ok(add_dll_dir) = os.getattr("add_dll_directory") {
                add_dll_dir.call1((dll_dir.clone(),))?;
                eprintln!("✅ Added DLL directory to Python search path: {}", dll_dir);
            }
            
            // Pre-emptively load our DLLs using Windows API
            let providers_path = format!("{}\\onnxruntime_providers_shared.dll", dll_dir);
            let onnx_path = format!("{}\\onnxruntime.dll", dll_dir);
            
            // Convert paths to wide strings for Windows API
            let providers_wide: Vec<u16> = OsStr::new(&providers_path)
                .encode_wide()
                .chain(Some(0))
                .collect();
            let onnx_wide: Vec<u16> = OsStr::new(&onnx_path)
                .encode_wide()
                .chain(Some(0))
                .collect();
            
            unsafe {
                // Load providers DLL first (dependency)
                let providers_handle = LoadLibraryExW(
                    providers_wide.as_ptr(),
                    null_mut(),
                    LOAD_WITH_ALTERED_SEARCH_PATH
                );
                
                if providers_handle.is_null() {
                    let error = GetLastError();
                    eprintln!("❌ Failed to pre-load onnxruntime_providers_shared.dll");
                    eprintln!("   Path: {}", providers_path);
                    eprintln!("   Error code: {}", error);
                } else {
                    eprintln!("✅ Pre-loaded onnxruntime_providers_shared.dll");
                }
                
                // Load main ONNX Runtime DLL
                let onnx_handle = LoadLibraryExW(
                    onnx_wide.as_ptr(),
                    null_mut(),
                    LOAD_WITH_ALTERED_SEARCH_PATH
                );
                
                if onnx_handle.is_null() {
                    let error = GetLastError();
                    eprintln!("❌ Failed to pre-load onnxruntime.dll");
                    eprintln!("   Path: {}", onnx_path);
                    eprintln!("   Error code: {}", error);
                    
                    // If pre-loading failed, it's likely a missing dependency
                    eprintln!("\n🚨 Failed to load ONNX Runtime!");
                    eprintln!("   The bundled ONNX Runtime DLL could not be loaded.");
                    eprintln!("   This might be due to missing Visual C++ Redistributables.");
                    eprintln!("\n   Solution: Install Visual C++ Redistributables from:");
                    eprintln!("   https://aka.ms/vs/17/release/vc_redist.x64.exe");
                    
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "Failed to load ONNX Runtime DLL. Please install Visual C++ Redistributables."
                    ));
                } else {
                    eprintln!("✅ Pre-loaded onnxruntime.dll");
                }
            }
            
            // Now set ORT_DYLIB_PATH for the ort crate
            // Use forward slashes for consistency with the ort crate
            let dll_path = onnx_path.replace('\\', "/");
            std::env::set_var("ORT_DYLIB_PATH", &dll_path);
            eprintln!("✅ Set ORT_DYLIB_PATH to: {}", dll_path);
        }
    }
    
    // Set up ONNX Runtime library path for Linux
    #[cfg(target_os = "linux")]
    {
        if std::env::var("ORT_DYLIB_PATH").is_err() {
            // Get the site-packages path
            let sys = _py.import("sys")?;
            let prefix: String = sys.getattr("prefix")?.extract()?;
            
            // Build the path to our bundled library
            let lib_path = format!("{}/lib/python{}.{}/site-packages/vocalize_rust/libonnxruntime.so", 
                prefix, 
                sys.getattr("version_info")?.getattr("major")?.extract::<i32>()?,
                sys.getattr("version_info")?.getattr("minor")?.extract::<i32>()?
            );
            
            // Check if the library exists
            if std::path::Path::new(&lib_path).exists() {
                std::env::set_var("ORT_DYLIB_PATH", &lib_path);
                eprintln!("✅ Set ORT_DYLIB_PATH to: {}", lib_path);
            } else {
                eprintln!("⚠️  ONNX Runtime library not found at: {}", lib_path);
                eprintln!("   Will attempt to use system ONNX Runtime if available");
            }
        }
    }
    
    // Set up ONNX Runtime library path for macOS
    #[cfg(target_os = "macos")]
    {
        if std::env::var("ORT_DYLIB_PATH").is_err() {
            // Get the site-packages path
            let sys = _py.import("sys")?;
            let prefix: String = sys.getattr("prefix")?.extract()?;
            
            // Build the path to our bundled library
            let lib_path = format!("{}/lib/python{}.{}/site-packages/vocalize_rust/libonnxruntime.dylib", 
                prefix,
                sys.getattr("version_info")?.getattr("major")?.extract::<i32>()?,
                sys.getattr("version_info")?.getattr("minor")?.extract::<i32>()?
            );
            
            // Check if the library exists
            if std::path::Path::new(&lib_path).exists() {
                std::env::set_var("ORT_DYLIB_PATH", &lib_path);
                eprintln!("✅ Set ORT_DYLIB_PATH to: {}", lib_path);
            } else {
                eprintln!("⚠️  ONNX Runtime library not found at: {}", lib_path);
                eprintln!("   Will attempt to use system ONNX Runtime if available");
            }
        }
    }
    
    // Initialize logging
    pyo3_log::init();

    // Add classes
    m.add_class::<PyVoice>()?;
    m.add_class::<PyVoiceManager>()?;
    m.add_class::<PyAudioWriter>()?;
    m.add_class::<PyAudioDevice>()?;
    m.add_class::<PyVocalizeError>()?;
    
    // Add enums
    m.add_class::<PyGender>()?;
    m.add_class::<PyVoiceStyle>()?;
    m.add_class::<PyAudioFormat>()?;
    m.add_class::<PyPlaybackState>()?;
    
    // Add configuration classes
    m.add_class::<PyEncodingSettings>()?;
    m.add_class::<PyAudioConfig>()?;
    m.add_class::<PyAudioDeviceInfo>()?;

    // Add exceptions
    m.add("VocalizeException", _py.get_type::<VocalizeException>())?;

    // Add aliased classes for backward compatibility
    m.add("Voice", _py.get_type::<PyVoice>())?;
    m.add("VoiceManager", _py.get_type::<PyVoiceManager>())?;
    m.add("AudioWriter", _py.get_type::<PyAudioWriter>())?;
    m.add("AudioDevice", _py.get_type::<PyAudioDevice>())?;
    m.add("VocalizeError", _py.get_type::<PyVocalizeError>())?;
    m.add("Gender", _py.get_type::<PyGender>())?;
    m.add("VoiceStyle", _py.get_type::<PyVoiceStyle>())?;

    // Add neural TTS functions
    m.add_function(wrap_pyfunction!(synthesize_from_tokens_neural, m)?)?;
    m.add_function(wrap_pyfunction!(save_audio_neural, m)?)?;
    
    // Add constants
    m.add("DEFAULT_SAMPLE_RATE", vocalize_core::DEFAULT_SAMPLE_RATE)?;
    m.add("DEFAULT_CHANNELS", vocalize_core::DEFAULT_CHANNELS)?;
    m.add("VERSION", env!("CARGO_PKG_VERSION"))?;
    
    Ok(())
}