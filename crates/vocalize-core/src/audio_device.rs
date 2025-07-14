//! Audio device management for real-time audio playback.

use crate::error::{VocalizeError, VocalizeResult};
use crate::tts_engine::AudioData;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Playback state enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaybackState {
    /// Audio is stopped
    Stopped,
    /// Audio is currently playing
    Playing,
    /// Audio is paused
    Paused,
    /// Audio playback encountered an error
    Error,
}

impl std::fmt::Display for PlaybackState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stopped => write!(f, "Stopped"),
            Self::Playing => write!(f, "Playing"),
            Self::Paused => write!(f, "Paused"),
            Self::Error => write!(f, "Error"),
        }
    }
}

/// Audio device information
#[derive(Debug, Clone, PartialEq)]
pub struct AudioDeviceInfo {
    /// Device identifier
    pub id: String,
    /// Human-readable device name
    pub name: String,
    /// Number of output channels
    pub channels: u16,
    /// Supported sample rates
    pub sample_rates: Vec<u32>,
    /// Whether this is the default device
    pub is_default: bool,
}

/// Audio device configuration
#[derive(Debug, Clone)]
pub struct AudioConfig {
    /// Target device ID (None for default)
    pub device_id: Option<String>,
    /// Sample rate for playback
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u16,
    /// Buffer size in frames
    pub buffer_size: u32,
    /// Playback latency target
    pub latency: Duration,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            device_id: None,
            sample_rate: crate::DEFAULT_SAMPLE_RATE,
            channels: crate::DEFAULT_CHANNELS,
            buffer_size: 1024,
            latency: Duration::from_millis(50),
        }
    }
}

/// Mock audio device for testing and platforms without audio support
#[derive(Debug)]
pub struct AudioDevice {
    config: AudioConfig,
    state: Arc<RwLock<PlaybackState>>,
    is_running: Arc<AtomicBool>,
    #[cfg_attr(not(test), allow(dead_code))]
    mock_mode: bool,
}

impl AudioDevice {
    /// Create a new audio device for testing (synchronous)
    #[cfg(test)]
    pub fn new_mock() -> Self {
        Self {
            config: AudioConfig::default(),
            state: Arc::new(RwLock::new(PlaybackState::Stopped)),
            is_running: Arc::new(AtomicBool::new(false)),
            mock_mode: true,
        }
    }

    /// Create a new mock audio device for Python bindings
    pub fn new_mock_for_bindings() -> Self {
        Self {
            config: AudioConfig::default(),
            state: Arc::new(RwLock::new(PlaybackState::Stopped)),
            is_running: Arc::new(AtomicBool::new(false)),
            mock_mode: true,
        }
    }

    /// Create a new audio device with default configuration
    ///
    /// # Errors
    ///
    /// Returns an error if the audio system cannot be initialized
    pub async fn new() -> VocalizeResult<Self> {
        Self::with_config(AudioConfig::default()).await
    }

    /// Create a new audio device with custom configuration
    ///
    /// # Errors
    ///
    /// Returns an error if the audio system cannot be initialized or the device is not found
    pub async fn with_config(config: AudioConfig) -> VocalizeResult<Self> {
        info!("Creating mock audio device with config: {:?}", config);

        Ok(Self {
            config,
            state: Arc::new(RwLock::new(PlaybackState::Stopped)),
            is_running: Arc::new(AtomicBool::new(false)),
            mock_mode: true,
        })
    }

    /// Get list of available audio devices
    ///
    /// # Errors
    ///
    /// Returns an error if the audio system cannot enumerate devices
    pub fn get_available_devices() -> VocalizeResult<Vec<AudioDeviceInfo>> {
        // Return mock devices for testing
        Ok(vec![
            AudioDeviceInfo {
                id: "default".to_string(),
                name: "Default Audio Device".to_string(),
                channels: 2,
                sample_rates: vec![44100, 48000],
                is_default: true,
            },
            AudioDeviceInfo {
                id: "speakers".to_string(),
                name: "Built-in Speakers".to_string(),
                channels: 2,
                sample_rates: vec![44100, 48000],
                is_default: false,
            },
        ])
    }

    /// Start audio playback
    ///
    /// # Errors
    ///
    /// Returns an error if the audio stream cannot be created or started
    pub async fn start(&self) -> VocalizeResult<()> {
        let current_state = *self.state.read().await;
        if current_state == PlaybackState::Playing {
            debug!("Audio device already playing");
            return Ok(());
        }

        info!("Starting mock audio playback");

        let mut state = self.state.write().await;
        *state = PlaybackState::Playing;
        self.is_running.store(true, Ordering::Relaxed);

        Ok(())
    }

    /// Stop audio playback
    ///
    /// # Errors
    ///
    /// Returns an error if the audio stream cannot be stopped
    pub async fn stop(&self) -> VocalizeResult<()> {
        info!("Stopping mock audio playback");

        self.is_running.store(false, Ordering::Relaxed);
        let mut state = self.state.write().await;
        *state = PlaybackState::Stopped;

        Ok(())
    }

    /// Pause audio playback
    ///
    /// # Errors
    ///
    /// Returns an error if the audio stream cannot be paused
    pub async fn pause(&self) -> VocalizeResult<()> {
        let current_state = *self.state.read().await;
        if current_state != PlaybackState::Playing {
            return Err(VocalizeError::audio_device("Cannot pause: not currently playing"));
        }

        info!("Pausing mock audio playback");
        let mut state = self.state.write().await;
        *state = PlaybackState::Paused;

        Ok(())
    }

    /// Resume audio playback
    ///
    /// # Errors
    ///
    /// Returns an error if the audio stream cannot be resumed
    pub async fn resume(&self) -> VocalizeResult<()> {
        let current_state = *self.state.read().await;
        if current_state != PlaybackState::Paused {
            return Err(VocalizeError::audio_device("Cannot resume: not currently paused"));
        }

        info!("Resuming mock audio playback");
        let mut state = self.state.write().await;
        *state = PlaybackState::Playing;

        Ok(())
    }

    /// Play audio data
    ///
    /// # Errors
    ///
    /// Returns an error if the audio cannot be queued for playback
    pub async fn play(&self, audio_data: &AudioData) -> VocalizeResult<()> {
        if audio_data.is_empty() {
            return Err(VocalizeError::invalid_input("Audio data cannot be empty"));
        }

        debug!("Mock playing {} samples", audio_data.len());

        // Simulate playback by setting state to playing briefly
        let mut state = self.state.write().await;
        *state = PlaybackState::Playing;
        self.is_running.store(true, Ordering::Relaxed);

        // Simulate playback time
        let duration = Duration::from_millis((audio_data.len() as f64 / self.config.sample_rate as f64 * 1000.0) as u64);
        tokio::time::sleep(duration.min(Duration::from_millis(100))).await; // Cap at 100ms for tests

        *state = PlaybackState::Stopped;
        self.is_running.store(false, Ordering::Relaxed);

        Ok(())
    }

    /// Play audio data and wait for completion
    ///
    /// # Errors
    ///
    /// Returns an error if playback fails or times out
    pub async fn play_blocking(&self, audio_data: &AudioData) -> VocalizeResult<()> {
        self.play(audio_data).await?;
        self.wait_for_completion().await
    }

    /// Wait for current audio to finish playing
    ///
    /// # Errors
    ///
    /// Returns an error if waiting times out
    pub async fn wait_for_completion(&self) -> VocalizeResult<()> {
        debug!("Waiting for mock audio completion");

        let timeout = Duration::from_secs(30);
        let start_time = std::time::Instant::now();

        loop {
            if start_time.elapsed() > timeout {
                return Err(VocalizeError::timeout("Audio playback timeout"));
            }

            let state = *self.state.read().await;
            if state == PlaybackState::Stopped || state == PlaybackState::Error {
                break;
            }

            if !self.is_running.load(Ordering::Relaxed) {
                break;
            }

            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        debug!("Mock audio playback completed");
        Ok(())
    }

    /// Get current playback state
    #[must_use]
    pub async fn get_state(&self) -> PlaybackState {
        *self.state.read().await
    }

    /// Check if audio is currently playing
    #[must_use]
    pub async fn is_playing(&self) -> bool {
        *self.state.read().await == PlaybackState::Playing
    }

    /// Check if audio is paused
    #[must_use]
    pub async fn is_paused(&self) -> bool {
        *self.state.read().await == PlaybackState::Paused
    }

    /// Check if audio is stopped
    #[must_use]
    pub async fn is_stopped(&self) -> bool {
        matches!(*self.state.read().await, PlaybackState::Stopped | PlaybackState::Error)
    }

    /// Get current audio configuration
    #[must_use]
    pub fn get_config(&self) -> &AudioConfig {
        &self.config
    }

    /// Get device information
    #[must_use]
    pub fn get_device_info(&self) -> Option<String> {
        Some("Mock Audio Device".to_string())
    }

    /// Get audio queue status (mock implementation)
    #[must_use]
    pub async fn get_queue_status(&self) -> (usize, usize) {
        (0, 1024) // Mock values: no data in queue, 1024 space available
    }
}

impl Drop for AudioDevice {
    fn drop(&mut self) {
        // Stop the audio stream when dropping
        self.is_running.store(false, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_playback_state_display() {
        assert_eq!(PlaybackState::Stopped.to_string(), "Stopped");
        assert_eq!(PlaybackState::Playing.to_string(), "Playing");
        assert_eq!(PlaybackState::Paused.to_string(), "Paused");
        assert_eq!(PlaybackState::Error.to_string(), "Error");
    }

    #[test]
    fn test_audio_device_info() {
        let info = AudioDeviceInfo {
            id: "test_device".to_string(),
            name: "Test Device".to_string(),
            channels: 2,
            sample_rates: vec![44100, 48000],
            is_default: true,
        };

        assert_eq!(info.id, "test_device");
        assert_eq!(info.name, "Test Device");
        assert_eq!(info.channels, 2);
        assert_eq!(info.sample_rates, vec![44100, 48000]);
        assert!(info.is_default);
    }

    #[test]
    fn test_audio_config_default() {
        let config = AudioConfig::default();
        assert_eq!(config.device_id, None);
        assert_eq!(config.sample_rate, crate::DEFAULT_SAMPLE_RATE);
        assert_eq!(config.channels, crate::DEFAULT_CHANNELS);
        assert_eq!(config.buffer_size, 1024);
        assert_eq!(config.latency, Duration::from_millis(50));
    }

    #[tokio::test]
    async fn test_audio_device_creation() {
        let device = AudioDevice::new().await.expect("Should create mock device");
        assert!(device.mock_mode);
        assert!(device.is_stopped().await);
    }

    #[tokio::test]
    async fn test_audio_device_with_config() {
        let config = AudioConfig {
            device_id: Some("test_device".to_string()),
            sample_rate: 48000,
            channels: 2,
            buffer_size: 2048,
            latency: Duration::from_millis(100),
        };

        let device = AudioDevice::with_config(config).await.expect("Should create device");
        assert_eq!(device.get_config().sample_rate, 48000);
        assert_eq!(device.get_config().channels, 2);
    }

    #[test]
    fn test_get_available_devices() {
        let devices = AudioDevice::get_available_devices().expect("Should get devices");
        assert!(!devices.is_empty());
        assert!(devices.iter().any(|d| d.is_default));
    }

    #[tokio::test]
    async fn test_audio_device_state_management() {
        let device = AudioDevice::new().await.expect("Should create device");

        // Initial state
        assert!(device.is_stopped().await);
        assert!(!device.is_playing().await);
        assert!(!device.is_paused().await);

        // Start playback
        device.start().await.expect("Should start");
        assert!(device.is_playing().await);

        // Pause playback
        device.pause().await.expect("Should pause");
        assert!(device.is_paused().await);

        // Resume playback
        device.resume().await.expect("Should resume");
        assert!(device.is_playing().await);

        // Stop playback
        device.stop().await.expect("Should stop");
        assert!(device.is_stopped().await);
    }

    #[tokio::test]
    async fn test_audio_device_play() {
        let device = AudioDevice::new().await.expect("Should create device");
        let audio_data = vec![0.1, 0.2, -0.1, -0.2];

        let result = device.play(&audio_data).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_audio_device_play_empty() {
        let device = AudioDevice::new().await.expect("Should create device");

        let result = device.play(&vec![]).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_audio_device_play_blocking() {
        let device = AudioDevice::new().await.expect("Should create device");
        let audio_data = vec![0.1, 0.2, -0.1, -0.2];

        let result = device.play_blocking(&audio_data).await;
        assert!(result.is_ok());
        assert!(device.is_stopped().await);
    }

    #[tokio::test]
    async fn test_audio_device_pause_not_playing() {
        let device = AudioDevice::new().await.expect("Should create device");

        let result = device.pause().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_audio_device_resume_not_paused() {
        let device = AudioDevice::new().await.expect("Should create device");

        let result = device.resume().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_audio_device_get_device_info() {
        let device = AudioDevice::new().await.expect("Should create device");
        let info = device.get_device_info();
        assert!(info.is_some());
        assert_eq!(info.unwrap(), "Mock Audio Device");
    }

    #[tokio::test]
    async fn test_audio_device_get_queue_status() {
        let device = AudioDevice::new().await.expect("Should create device");
        let (data, space) = device.get_queue_status().await;
        assert_eq!(data, 0);
        assert_eq!(space, 1024);
    }

    #[test]
    fn test_playback_state_equality() {
        assert_eq!(PlaybackState::Stopped, PlaybackState::Stopped);
        assert_ne!(PlaybackState::Playing, PlaybackState::Stopped);
        assert_ne!(PlaybackState::Paused, PlaybackState::Playing);
        assert_ne!(PlaybackState::Error, PlaybackState::Paused);
    }

    #[test]
    fn test_audio_device_info_equality() {
        let info1 = AudioDeviceInfo {
            id: "test".to_string(),
            name: "Test".to_string(),
            channels: 2,
            sample_rates: vec![44100],
            is_default: true,
        };

        let info2 = AudioDeviceInfo {
            id: "test".to_string(),
            name: "Test".to_string(),
            channels: 2,
            sample_rates: vec![44100],
            is_default: true,
        };

        let info3 = AudioDeviceInfo {
            id: "different".to_string(),
            name: "Different".to_string(),
            channels: 1,
            sample_rates: vec![48000],
            is_default: false,
        };

        assert_eq!(info1, info2);
        assert_ne!(info1, info3);
    }
}