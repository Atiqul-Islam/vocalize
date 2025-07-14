//! # Vocalize Core
//!
//! High-performance text-to-speech synthesis engine with neural voice generation.
//!
//! ## Features
//!
//! - Fast neural TTS synthesis using Kokoro models
//! - Cross-platform audio device support
//! - Multiple audio format output (WAV, MP3, FLAC, OGG)
//! - Real-time streaming synthesis
//! - Voice blending and customization
//!
//! ## Example
//!
//! ```rust,no_run
//! use vocalize_core::{TtsEngine, Voice, SynthesisParams, AudioDevice};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let engine = TtsEngine::new().await?;
//!     let voice = Voice::default();
//!     let params = SynthesisParams::new(voice);
//!     let audio = engine.synthesize("Hello, world!", &params).await?;
//!     
//!     let mut device = AudioDevice::new().await?;
//!     device.play(&audio).await?;
//!     
//!     Ok(())
//! }
//! ```

#![deny(missing_docs)]
#![deny(unsafe_code)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

pub mod audio_device;
pub mod audio_writer;
pub mod error;
pub mod model;
pub mod models;
pub mod onnx_engine;
pub mod tts_engine;
pub mod voice_manager;
pub mod wav_writer;

// Re-export main types for convenience
pub use audio_device::{AudioConfig, AudioDevice, AudioDeviceInfo, PlaybackState};
pub use audio_writer::{AudioFormat, AudioWriter, EncodingSettings};
pub use error::{VocalizeError, VocalizeResult};
pub use model::{ModelId, ModelInfo, ModelManager, ModelConfig};
pub use models::{TtsModel, ModelRegistry};
pub use onnx_engine::OnnxTtsEngine;
pub use tts_engine::{AudioData, SynthesisParams, TtsEngine, TtsConfig};
pub use voice_manager::{Gender, Voice, VoiceManager, VoiceStyle};

/// Version information for the vocalize-core crate
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default sample rate for audio processing (24 kHz)
pub const DEFAULT_SAMPLE_RATE: u32 = 24_000;

/// Default number of audio channels (mono)
pub const DEFAULT_CHANNELS: u16 = 1;

/// Maximum text length for synthesis (to prevent memory issues)
pub const MAX_TEXT_LENGTH: usize = 100_000;