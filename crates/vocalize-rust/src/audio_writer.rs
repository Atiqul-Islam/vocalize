//! Python bindings for audio writer

use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3_asyncio::tokio::future_into_py;
use std::collections::HashMap;
use std::path::Path;
use vocalize_core::{AudioFormat, AudioWriter, EncodingSettings};

use crate::error::IntoPyResult;

/// Python wrapper for AudioFormat
#[pyclass(name = "AudioFormat")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyAudioFormat {
    Wav,
    Mp3,
    Flac,
    Ogg,
}

impl From<AudioFormat> for PyAudioFormat {
    fn from(format: AudioFormat) -> Self {
        match format {
            AudioFormat::Wav => PyAudioFormat::Wav,
            AudioFormat::Mp3 => PyAudioFormat::Mp3,
            AudioFormat::Flac => PyAudioFormat::Flac,
            AudioFormat::Ogg => PyAudioFormat::Ogg,
        }
    }
}

impl From<PyAudioFormat> for AudioFormat {
    fn from(py_format: PyAudioFormat) -> Self {
        match py_format {
            PyAudioFormat::Wav => AudioFormat::Wav,
            PyAudioFormat::Mp3 => AudioFormat::Mp3,
            PyAudioFormat::Flac => AudioFormat::Flac,
            PyAudioFormat::Ogg => AudioFormat::Ogg,
        }
    }
}

#[pymethods]
impl PyAudioFormat {
    fn extension(&self) -> String {
        AudioFormat::from(*self).extension().to_string()
    }

    fn mime_type(&self) -> String {
        AudioFormat::from(*self).mime_type().to_string()
    }

    fn is_lossy(&self) -> bool {
        AudioFormat::from(*self).is_lossy()
    }

    fn description(&self) -> String {
        AudioFormat::from(*self).description().to_string()
    }

    #[staticmethod]
    fn from_extension(extension: &str) -> PyResult<PyAudioFormat> {
        let format = AudioFormat::from_extension(extension).into_py_result()?;
        Ok(PyAudioFormat::from(format))
    }

    #[staticmethod]
    fn from_path(path: &str) -> PyResult<PyAudioFormat> {
        let format = AudioFormat::from_path(path).into_py_result()?;
        Ok(PyAudioFormat::from(format))
    }

    #[staticmethod]
    fn all() -> Vec<PyAudioFormat> {
        AudioFormat::all()
            .iter()
            .map(|&f| PyAudioFormat::from(f))
            .collect()
    }

    fn __str__(&self) -> String {
        match self {
            PyAudioFormat::Wav => "WAV".to_string(),
            PyAudioFormat::Mp3 => "MP3".to_string(),
            PyAudioFormat::Flac => "FLAC".to_string(),
            PyAudioFormat::Ogg => "OGG".to_string(),
        }
    }

    fn __repr__(&self) -> String {
        format!("AudioFormat.{}", self.__str__())
    }

    #[classattr]
    const WAV: PyAudioFormat = PyAudioFormat::Wav;

    #[classattr]
    const MP3: PyAudioFormat = PyAudioFormat::Mp3;

    #[classattr]
    const FLAC: PyAudioFormat = PyAudioFormat::Flac;

    #[classattr]
    const OGG: PyAudioFormat = PyAudioFormat::Ogg;
}

/// Python wrapper for EncodingSettings
#[pyclass(name = "EncodingSettings")]
#[derive(Debug, Clone)]
pub struct PyEncodingSettings {
    inner: EncodingSettings,
}

impl PyEncodingSettings {
    pub fn new(settings: EncodingSettings) -> Self {
        Self { inner: settings }
    }

    pub fn inner(&self) -> &EncodingSettings {
        &self.inner
    }

    pub fn into_inner(self) -> EncodingSettings {
        self.inner
    }
}

#[pymethods]
impl PyEncodingSettings {
    #[new]
    fn py_new(sample_rate: u32, channels: u16) -> Self {
        Self::new(EncodingSettings::new(sample_rate, channels))
    }

    #[staticmethod]
    fn default() -> Self {
        Self::new(EncodingSettings::default())
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
    fn bit_depth(&self) -> u16 {
        self.inner.bit_depth
    }

    #[getter]
    fn quality(&self) -> Option<f32> {
        self.inner.quality
    }

    #[getter]
    fn variable_bitrate(&self) -> bool {
        self.inner.variable_bitrate
    }

    fn with_bit_depth(&self, bit_depth: u16) -> PyEncodingSettings {
        Self::new(self.inner.clone().with_bit_depth(bit_depth))
    }

    fn with_quality(&self, quality: f32) -> PyEncodingSettings {
        Self::new(self.inner.clone().with_quality(quality))
    }

    fn with_variable_bitrate(&self) -> PyEncodingSettings {
        Self::new(self.inner.clone().with_variable_bitrate())
    }

    fn with_constant_bitrate(&self) -> PyEncodingSettings {
        let mut settings = self.inner.clone();
        settings.variable_bitrate = false;
        Self::new(settings)
    }

    fn validate(&self) -> PyResult<()> {
        self.inner.validate().into_py_result()
    }

    fn __repr__(&self) -> String {
        format!(
            "EncodingSettings(sample_rate={}, channels={}, bit_depth={}, quality={:?}, variable_bitrate={})",
            self.inner.sample_rate,
            self.inner.channels,
            self.inner.bit_depth,
            self.inner.quality,
            self.inner.variable_bitrate
        )
    }

    fn to_dict(&self) -> HashMap<String, String> {
        let mut dict = HashMap::new();
        dict.insert("sample_rate".to_string(), self.inner.sample_rate.to_string());
        dict.insert("channels".to_string(), self.inner.channels.to_string());
        dict.insert("bit_depth".to_string(), self.inner.bit_depth.to_string());
        if let Some(quality) = self.inner.quality {
            dict.insert("quality".to_string(), quality.to_string());
        }
        dict.insert("variable_bitrate".to_string(), self.inner.variable_bitrate.to_string());
        dict
    }
}

/// Python wrapper for AudioWriter
#[pyclass(name = "AudioWriter")]
#[derive(Debug)]
pub struct PyAudioWriter {
    inner: AudioWriter,
}

impl PyAudioWriter {
    pub fn new(writer: AudioWriter) -> Self {
        Self { inner: writer }
    }
}

#[pymethods]
impl PyAudioWriter {
    #[new]
    fn py_new() -> Self {
        Self::new(AudioWriter::new())
    }

    fn with_settings(&self, _settings: &PyEncodingSettings) -> PyAudioWriter {
        // For now, just return a new writer (ignoring settings)
        Self::new(AudioWriter::new())
    }

    /// Write audio data to file
    fn write_file<'py>(
        &self,
        py: Python<'py>,
        audio_data: Vec<f32>,
        path: String,
        format: PyAudioFormat,
        settings: Option<&PyEncodingSettings>,
    ) -> PyResult<&'py PyAny> {
        let writer = AudioWriter::new();
        let rust_format = AudioFormat::from(format);
        let rust_settings = settings.map(|s| s.inner().clone());
        
        future_into_py(py, async move {
            writer
                .write_file(&audio_data, Path::new(&path), rust_format, rust_settings)
                .await
                .into_py_result()?;
            Ok(())
        })
    }

    /// Write audio data to file with auto-detected format
    fn write_file_auto<'py>(
        &self,
        py: Python<'py>,
        audio_data: Vec<f32>,
        path: String,
        settings: Option<&PyEncodingSettings>,
    ) -> PyResult<&'py PyAny> {
        let writer = AudioWriter::new();
        let rust_settings = settings.map(|s| s.inner().clone());
        
        future_into_py(py, async move {
            writer
                .write_file_auto(&audio_data, Path::new(&path), rust_settings)
                .await
                .into_py_result()?;
            Ok(())
        })
    }

    /// Estimate file size for given audio data and format
    fn estimate_file_size(
        &self,
        audio_data: Vec<f32>,
        format: PyAudioFormat,
        settings: &PyEncodingSettings,
    ) -> usize {
        self.inner.estimate_file_size(
            &audio_data,
            AudioFormat::from(format),
            settings.inner(),
        )
    }

    /// Check if a format is supported
    fn is_format_supported(&self, format: PyAudioFormat) -> bool {
        AudioWriter::is_format_supported(AudioFormat::from(format))
    }

    /// Get list of supported formats
    #[staticmethod]
    fn get_supported_formats() -> Vec<PyAudioFormat> {
        AudioWriter::get_supported_formats()
            .iter()
            .map(|&f| PyAudioFormat::from(f))
            .collect()
    }

    /// Validate audio data and settings
    fn validate_inputs(
        &self,
        audio_data: Vec<f32>,
        _settings: &PyEncodingSettings,
    ) -> PyResult<()> {
        // Simple validation - check if audio data is not empty
        if audio_data.is_empty() {
            return Err(crate::error::PyVocalizeError::new_err("Audio data cannot be empty".to_string()));
        }
        
        // Check for invalid samples (NaN, infinite)
        for sample in &audio_data {
            if !sample.is_finite() {
                return Err(crate::error::PyVocalizeError::new_err("Audio data contains invalid samples".to_string()));
            }
        }
        
        Ok(())
    }

    fn __repr__(&self) -> String {
        "AudioWriter()".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_py_audio_format_conversion() {
        assert_eq!(PyAudioFormat::from(AudioFormat::Wav), PyAudioFormat::Wav);
        assert_eq!(PyAudioFormat::from(AudioFormat::Mp3), PyAudioFormat::Mp3);
        assert_eq!(AudioFormat::from(PyAudioFormat::Flac), AudioFormat::Flac);
        assert_eq!(AudioFormat::from(PyAudioFormat::Ogg), AudioFormat::Ogg);
    }

    #[test]
    fn test_py_audio_format_properties() {
        assert_eq!(PyAudioFormat::Wav.extension(), "wav");
        assert_eq!(PyAudioFormat::Mp3.extension(), "mp3");
        assert_eq!(PyAudioFormat::Wav.mime_type(), "audio/wav");
        assert_eq!(PyAudioFormat::Mp3.mime_type(), "audio/mpeg");
        
        assert!(!PyAudioFormat::Wav.is_lossy());
        assert!(PyAudioFormat::Mp3.is_lossy());
        assert!(!PyAudioFormat::Flac.is_lossy());
        assert!(PyAudioFormat::Ogg.is_lossy());
    }

    #[test]
    fn test_py_audio_format_from_extension() {
        assert!(PyAudioFormat::from_extension("wav").is_ok());
        assert!(PyAudioFormat::from_extension("mp3").is_ok());
        assert!(PyAudioFormat::from_extension("WAV").is_ok());
        assert!(PyAudioFormat::from_extension("xyz").is_err());
    }

    #[test]
    fn test_py_audio_format_from_path() {
        assert!(PyAudioFormat::from_path("test.wav").is_ok());
        assert!(PyAudioFormat::from_path("/path/to/file.mp3").is_ok());
        assert!(PyAudioFormat::from_path("file.FLAC").is_ok());
        assert!(PyAudioFormat::from_path("no_extension").is_err());
    }

    #[test]
    fn test_py_audio_format_all() {
        let formats = PyAudioFormat::all();
        assert_eq!(formats.len(), 4);
        assert!(formats.contains(&PyAudioFormat::Wav));
        assert!(formats.contains(&PyAudioFormat::Mp3));
        assert!(formats.contains(&PyAudioFormat::Flac));
        assert!(formats.contains(&PyAudioFormat::Ogg));
    }

    #[test]
    fn test_py_audio_format_string_representations() {
        assert_eq!(PyAudioFormat::Wav.__str__(), "WAV");
        assert_eq!(PyAudioFormat::Mp3.__str__(), "MP3");
        assert_eq!(PyAudioFormat::Wav.__repr__(), "AudioFormat.WAV");
        assert_eq!(PyAudioFormat::Mp3.__repr__(), "AudioFormat.MP3");
    }

    #[test]
    fn test_py_encoding_settings_creation() {
        let settings = PyEncodingSettings::py_new(48000, 2);
        assert_eq!(settings.sample_rate(), 48000);
        assert_eq!(settings.channels(), 2);
        assert_eq!(settings.bit_depth(), 16); // Default
        assert_eq!(settings.quality(), None); // Default
        assert!(!settings.variable_bitrate()); // Default
    }

    #[test]
    fn test_py_encoding_settings_default() {
        let settings = PyEncodingSettings::default();
        assert_eq!(settings.sample_rate(), 24000);
        assert_eq!(settings.channels(), 1);
        assert_eq!(settings.bit_depth(), 16);
    }

    #[test]
    fn test_py_encoding_settings_modifications() {
        let settings = PyEncodingSettings::default();
        
        let with_bit_depth = settings.with_bit_depth(24);
        assert_eq!(with_bit_depth.bit_depth(), 24);
        
        let with_quality = settings.with_quality(0.8);
        assert_eq!(with_quality.quality(), Some(0.8));
        
        let with_vbr = settings.with_variable_bitrate();
        assert!(with_vbr.variable_bitrate());
        
        let with_cbr = with_vbr.with_constant_bitrate();
        assert!(!with_cbr.variable_bitrate());
    }

    #[test]
    fn test_py_encoding_settings_validation() {
        let valid_settings = PyEncodingSettings::py_new(24000, 1);
        assert!(valid_settings.validate().is_ok());
        
        // Invalid sample rate
        let invalid_settings = PyEncodingSettings::py_new(0, 1);
        assert!(invalid_settings.validate().is_err());
    }

    #[test]
    fn test_py_encoding_settings_to_dict() {
        let settings = PyEncodingSettings::py_new(48000, 2)
            .with_bit_depth(24)
            .with_quality(0.9);
        
        let dict = settings.to_dict();
        assert_eq!(dict.get("sample_rate"), Some(&"48000".to_string()));
        assert_eq!(dict.get("channels"), Some(&"2".to_string()));
        assert_eq!(dict.get("bit_depth"), Some(&"24".to_string()));
        assert_eq!(dict.get("quality"), Some(&"0.9".to_string()));
    }

    #[test]
    fn test_py_audio_writer_creation() {
        let writer = PyAudioWriter::py_new();
        assert_eq!(writer.__repr__(), "AudioWriter()");
    }

    #[test]
    fn test_py_audio_writer_with_settings() {
        let writer = PyAudioWriter::py_new();
        let settings = PyEncodingSettings::py_new(48000, 2);
        let writer_with_settings = writer.with_settings(&settings);
        assert_eq!(writer_with_settings.__repr__(), "AudioWriter()");
    }

    #[test]
    fn test_py_audio_writer_format_support() {
        let writer = PyAudioWriter::py_new();
        
        assert!(writer.is_format_supported(PyAudioFormat::Wav));
        // MP3, FLAC, OGG are not implemented yet
        assert!(!writer.is_format_supported(PyAudioFormat::Mp3));
        assert!(!writer.is_format_supported(PyAudioFormat::Flac));
        assert!(!writer.is_format_supported(PyAudioFormat::Ogg));
    }

    #[test]
    fn test_py_audio_writer_supported_formats() {
        let formats = PyAudioWriter::get_supported_formats();
        assert!(formats.contains(&PyAudioFormat::Wav));
        // Only WAV is currently supported
        assert_eq!(formats.len(), 1);
    }

    #[test]
    fn test_py_audio_writer_estimate_file_size() {
        let writer = PyAudioWriter::py_new();
        let audio_data = vec![0.1, 0.2, -0.1, -0.2];
        let settings = PyEncodingSettings::default();
        
        let size = writer.estimate_file_size(&audio_data, PyAudioFormat::Wav, &settings);
        assert!(size > 0);
    }

    #[test]
    fn test_py_audio_writer_validate_inputs() {
        let writer = PyAudioWriter::py_new();
        let settings = PyEncodingSettings::default();
        
        // Valid audio data
        let valid_audio = vec![0.1, 0.2, -0.1, -0.2];
        assert!(writer.validate_inputs(&valid_audio, &settings).is_ok());
        
        // Empty audio data
        let empty_audio: Vec<f32> = vec![];
        assert!(writer.validate_inputs(&empty_audio, &settings).is_err());
        
        // Invalid audio samples (NaN)
        let invalid_audio = vec![f32::NAN, 0.5];
        assert!(writer.validate_inputs(&invalid_audio, &settings).is_err());
    }
}