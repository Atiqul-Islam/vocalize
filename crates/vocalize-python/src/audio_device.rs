//! Python bindings for audio device

use pyo3::prelude::*;
use std::collections::HashMap;
use std::time::Duration;
use vocalize_core::{AudioConfig, AudioDevice, AudioDeviceInfo, PlaybackState};

use crate::error::vocalize_error_to_pyerr;

/// Python wrapper for PlaybackState
#[pyclass(name = "PlaybackState")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyPlaybackState {
    Stopped,
    Playing,
    Paused,
    Error,
}

impl From<PlaybackState> for PyPlaybackState {
    fn from(state: PlaybackState) -> Self {
        match state {
            PlaybackState::Stopped => PyPlaybackState::Stopped,
            PlaybackState::Playing => PyPlaybackState::Playing,
            PlaybackState::Paused => PyPlaybackState::Paused,
            PlaybackState::Error => PyPlaybackState::Error,
        }
    }
}

#[pymethods]
impl PyPlaybackState {
    fn __str__(&self) -> String {
        match self {
            PyPlaybackState::Stopped => "Stopped".to_string(),
            PyPlaybackState::Playing => "Playing".to_string(),
            PyPlaybackState::Paused => "Paused".to_string(),
            PyPlaybackState::Error => "Error".to_string(),
        }
    }

    fn __repr__(&self) -> String {
        format!("PlaybackState.{}", self.__str__())
    }

    #[classattr]
    const STOPPED: PyPlaybackState = PyPlaybackState::Stopped;

    #[classattr]
    const PLAYING: PyPlaybackState = PyPlaybackState::Playing;

    #[classattr]
    const PAUSED: PyPlaybackState = PyPlaybackState::Paused;

    #[classattr]
    const ERROR: PyPlaybackState = PyPlaybackState::Error;
}

/// Python wrapper for AudioDeviceInfo
#[pyclass(name = "AudioDeviceInfo")]
#[derive(Debug, Clone)]
pub struct PyAudioDeviceInfo {
    inner: AudioDeviceInfo,
}

impl PyAudioDeviceInfo {
    pub fn new(info: AudioDeviceInfo) -> Self {
        Self { inner: info }
    }
}

#[pymethods]
impl PyAudioDeviceInfo {
    #[getter]
    fn id(&self) -> String {
        self.inner.id.clone()
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    fn channels(&self) -> u16 {
        self.inner.channels
    }

    #[getter]
    fn sample_rates(&self) -> Vec<u32> {
        self.inner.sample_rates.clone()
    }

    #[getter]
    fn is_default(&self) -> bool {
        self.inner.is_default
    }

    fn __str__(&self) -> String {
        format!("{} ({})", self.inner.name, self.inner.id)
    }

    fn __repr__(&self) -> String {
        format!(
            "AudioDeviceInfo(id='{}', name='{}', channels={}, is_default={})",
            self.inner.id, self.inner.name, self.inner.channels, self.inner.is_default
        )
    }

    fn to_dict(&self) -> HashMap<String, String> {
        let mut dict = HashMap::new();
        dict.insert("id".to_string(), self.inner.id.clone());
        dict.insert("name".to_string(), self.inner.name.clone());
        dict.insert("channels".to_string(), self.inner.channels.to_string());
        dict.insert("sample_rates".to_string(), format!("{:?}", self.inner.sample_rates));
        dict.insert("is_default".to_string(), self.inner.is_default.to_string());
        dict
    }
}

/// Python wrapper for AudioConfig
#[pyclass(name = "AudioConfig")]
#[derive(Debug, Clone)]
pub struct PyAudioConfig {
    inner: AudioConfig,
}

impl PyAudioConfig {
    pub fn new(config: AudioConfig) -> Self {
        Self { inner: config }
    }

    pub fn inner(&self) -> &AudioConfig {
        &self.inner
    }

    pub fn into_inner(self) -> AudioConfig {
        self.inner
    }
}

#[pymethods]
impl PyAudioConfig {
    #[new]
    #[pyo3(signature = (device_id=None, sample_rate=None, channels=None, buffer_size=None, latency_ms=None))]
    fn py_new(
        device_id: Option<String>,
        sample_rate: Option<u32>,
        channels: Option<u16>,
        buffer_size: Option<u32>,
        latency_ms: Option<u64>,
    ) -> Self {
        let mut config = AudioConfig::default();
        
        if let Some(id) = device_id {
            config.device_id = Some(id);
        }
        if let Some(sr) = sample_rate {
            config.sample_rate = sr;
        }
        if let Some(ch) = channels {
            config.channels = ch;
        }
        if let Some(bs) = buffer_size {
            config.buffer_size = bs;
        }
        if let Some(lat) = latency_ms {
            config.latency = Duration::from_millis(lat);
        }
        
        Self::new(config)
    }

    #[staticmethod]
    fn default() -> Self {
        Self::new(AudioConfig::default())
    }

    #[getter]
    fn device_id(&self) -> Option<String> {
        self.inner.device_id.clone()
    }

    #[getter]
    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate
    }

    #[getter]
    fn channels(&self) -> u16 {
        self.inner.channels
    }

    #[getter]
    fn buffer_size(&self) -> u32 {
        self.inner.buffer_size
    }

    #[getter]
    fn latency_ms(&self) -> u64 {
        self.inner.latency.as_millis() as u64
    }

    fn __repr__(&self) -> String {
        format!(
            "AudioConfig(device_id={:?}, sample_rate={}, channels={}, buffer_size={}, latency={}ms)",
            self.inner.device_id,
            self.inner.sample_rate,
            self.inner.channels,
            self.inner.buffer_size,
            self.inner.latency.as_millis()
        )
    }
}

/// Python wrapper for AudioDevice
#[pyclass(name = "AudioDevice")]
#[derive(Debug)]
pub struct PyAudioDevice {
    inner: AudioDevice,
}

impl PyAudioDevice {
    pub fn new(device: AudioDevice) -> Self {
        Self { inner: device }
    }
}

#[pymethods]
impl PyAudioDevice {
    #[new]
    fn py_new() -> PyResult<Self> {
        // Create mock audio device for testing
        let device = AudioDevice::new_mock_for_bindings();
        Ok(PyAudioDevice::new(device))
    }

    #[staticmethod]
    fn with_config(_config: &PyAudioConfig) -> PyResult<PyAudioDevice> {
        // Create mock audio device (ignoring config for mock implementation)
        let device = AudioDevice::new_mock_for_bindings();
        Ok(PyAudioDevice::new(device))
    }

    #[staticmethod]
    fn get_available_devices() -> PyResult<Vec<PyAudioDeviceInfo>> {
        let devices = AudioDevice::get_available_devices().map_err(vocalize_error_to_pyerr)?;
        Ok(devices.into_iter().map(PyAudioDeviceInfo::new).collect())
    }

    /// Play audio data (simplified sync version for testing)
    fn play_sync(&self, audio_data: Vec<f32>) -> PyResult<()> {
        if audio_data.is_empty() {
            return Err(crate::error::PyVocalizeError::new(
                vocalize_core::VocalizeError::invalid_input("Audio data cannot be empty".to_string())
            ).into());
        }
        Ok(())
    }

    /// Get current playback state
    fn get_state(&self) -> PyPlaybackState {
        PyPlaybackState::Stopped // Mock always returns stopped
    }

    /// Check if audio is currently playing
    fn is_playing(&self) -> bool {
        false // Mock is never playing
    }

    /// Check if audio is paused
    fn is_paused(&self) -> bool {
        false // Mock is never paused
    }

    /// Check if audio is stopped
    fn is_stopped(&self) -> bool {
        true // Mock is always stopped
    }

    /// Get current audio configuration
    fn get_config(&self) -> PyAudioConfig {
        PyAudioConfig::new(self.inner.get_config().clone())
    }

    /// Get device information
    fn get_device_info(&self) -> Option<String> {
        self.inner.get_device_info()
    }

    fn __repr__(&self) -> String {
        "AudioDevice()".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_py_playback_state_conversion() {
        assert_eq!(PyPlaybackState::from(PlaybackState::Stopped), PyPlaybackState::Stopped);
        assert_eq!(PyPlaybackState::from(PlaybackState::Playing), PyPlaybackState::Playing);
        assert_eq!(PyPlaybackState::from(PlaybackState::Paused), PyPlaybackState::Paused);
        assert_eq!(PyPlaybackState::from(PlaybackState::Error), PyPlaybackState::Error);
    }

    #[test]
    fn test_py_playback_state_string_repr() {
        assert_eq!(PyPlaybackState::Stopped.__str__(), "Stopped");
        assert_eq!(PyPlaybackState::Playing.__str__(), "Playing");
        assert_eq!(PyPlaybackState::Paused.__str__(), "Paused");
        assert_eq!(PyPlaybackState::Error.__str__(), "Error");
        
        assert_eq!(PyPlaybackState::Stopped.__repr__(), "PlaybackState.Stopped");
        assert_eq!(PyPlaybackState::Playing.__repr__(), "PlaybackState.Playing");
    }

    #[test]
    fn test_py_audio_device_info() {
        let info = AudioDeviceInfo {
            id: "test_device".to_string(),
            name: "Test Device".to_string(),
            channels: 2,
            sample_rates: vec![44100, 48000],
            is_default: true,
        };
        
        let py_info = PyAudioDeviceInfo::new(info);
        assert_eq!(py_info.id(), "test_device");
        assert_eq!(py_info.name(), "Test Device");
        assert_eq!(py_info.channels(), 2);
        assert_eq!(py_info.sample_rates(), vec![44100, 48000]);
        assert!(py_info.is_default());
    }

    #[test]
    fn test_py_audio_device_info_to_dict() {
        let info = AudioDeviceInfo {
            id: "test".to_string(),
            name: "Test".to_string(),
            channels: 1,
            sample_rates: vec![48000],
            is_default: false,
        };
        
        let py_info = PyAudioDeviceInfo::new(info);
        let dict = py_info.to_dict();
        
        assert_eq!(dict.get("id"), Some(&"test".to_string()));
        assert_eq!(dict.get("name"), Some(&"Test".to_string()));
        assert_eq!(dict.get("channels"), Some(&"1".to_string()));
        assert_eq!(dict.get("is_default"), Some(&"false".to_string()));
    }

    #[test]
    fn test_py_audio_config_creation() {
        let config = PyAudioConfig::py_new(
            Some("test_device".to_string()),
            Some(48000),
            Some(2),
            Some(2048),
            Some(100),
        );
        
        assert_eq!(config.device_id(), Some("test_device".to_string()));
        assert_eq!(config.sample_rate(), 48000);
        assert_eq!(config.channels(), 2);
        assert_eq!(config.buffer_size(), 2048);
        assert_eq!(config.latency_ms(), 100);
    }

    #[test]
    fn test_py_audio_config_default() {
        let config = PyAudioConfig::default();
        assert_eq!(config.device_id(), None);
        assert_eq!(config.sample_rate(), vocalize_core::DEFAULT_SAMPLE_RATE);
        assert_eq!(config.channels(), vocalize_core::DEFAULT_CHANNELS);
        assert_eq!(config.buffer_size(), 1024);
        assert_eq!(config.latency_ms(), 50);
    }

    #[test]
    fn test_py_audio_config_repr() {
        let config = PyAudioConfig::default();
        let repr = config.__repr__();
        
        assert!(repr.contains("AudioConfig"));
        assert!(repr.contains("sample_rate"));
        assert!(repr.contains("channels"));
    }

    #[test]
    fn test_py_audio_device_creation() {
        let device = PyAudioDevice::py_new();
        assert!(device.is_ok());
        assert_eq!(device.unwrap().__repr__(), "AudioDevice()");
    }

    #[test]
    fn test_py_audio_device_with_config() {
        let config = PyAudioConfig::default();
        let device = PyAudioDevice::with_config(&config);
        assert!(device.is_ok());
    }

    #[test]
    fn test_py_audio_device_get_available_devices() {
        let devices = PyAudioDevice::get_available_devices();
        assert!(devices.is_ok());
        
        let device_list = devices.unwrap();
        assert!(!device_list.is_empty());
    }

    #[test]
    fn test_py_audio_device_play_sync() {
        let device = PyAudioDevice::py_new().unwrap();
        
        // Valid audio
        let audio_data = vec![0.1, 0.2, -0.1, -0.2];
        assert!(device.play_sync(audio_data).is_ok());
        
        // Empty audio should fail
        assert!(device.play_sync(vec![]).is_err());
    }

    #[test]
    fn test_py_audio_device_state() {
        let device = PyAudioDevice::py_new().unwrap();
        
        assert_eq!(device.get_state(), PyPlaybackState::Stopped);
        assert!(!device.is_playing());
        assert!(!device.is_paused());
        assert!(device.is_stopped());
    }
}