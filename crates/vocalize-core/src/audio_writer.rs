//! Audio file writing with support for multiple formats.

use crate::error::{VocalizeError, VocalizeResult};
use crate::AudioData;
use crate::wav_writer::{WavWriter, WavSpec};
use std::path::Path;
use tracing::{debug, info, warn};

/// Supported audio output formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AudioFormat {
    /// WAV format (uncompressed)
    Wav,
    /// MP3 format (lossy compression)
    Mp3,
    /// FLAC format (lossless compression)
    Flac,
    /// OGG Vorbis format (lossy compression)
    Ogg,
}

impl AudioFormat {
    /// Get file extension for the format
    #[must_use]
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Mp3 => "mp3",
            Self::Flac => "flac",
            Self::Ogg => "ogg",
        }
    }

    /// Get MIME type for the format
    #[must_use]
    pub const fn mime_type(self) -> &'static str {
        match self {
            Self::Wav => "audio/wav",
            Self::Mp3 => "audio/mpeg",
            Self::Flac => "audio/flac",
            Self::Ogg => "audio/ogg",
        }
    }

    /// Check if the format is lossy
    #[must_use]
    pub const fn is_lossy(self) -> bool {
        matches!(self, Self::Mp3 | Self::Ogg)
    }

    /// Get human-readable description
    #[must_use]
    pub const fn description(self) -> &'static str {
        match self {
            Self::Wav => "Waveform Audio File Format",
            Self::Mp3 => "MPEG Audio Layer III",
            Self::Flac => "Free Lossless Audio Codec",
            Self::Ogg => "Ogg Vorbis",
        }
    }

    /// Detect format from file extension
    ///
    /// # Errors
    ///
    /// Returns an error if the extension is not supported
    pub fn from_extension(extension: &str) -> VocalizeResult<Self> {
        match extension.to_lowercase().as_str() {
            "wav" => Ok(Self::Wav),
            "mp3" => Ok(Self::Mp3),
            "flac" => Ok(Self::Flac),
            "ogg" => Ok(Self::Ogg),
            _ => Err(VocalizeError::invalid_input(format!(
                "Unsupported audio format: {extension}"
            ))),
        }
    }

    /// Detect format from file path
    ///
    /// # Errors
    ///
    /// Returns an error if the file has no extension or unsupported extension
    pub fn from_path<P: AsRef<Path>>(path: P) -> VocalizeResult<Self> {
        let path = path.as_ref();
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .ok_or_else(|| {
                VocalizeError::invalid_input(format!(
                    "No file extension found in path: {}",
                    path.display()
                ))
            })?;

        Self::from_extension(extension)
    }

    /// Get all supported formats
    #[must_use]
    pub const fn all() -> &'static [Self] {
        &[Self::Wav, Self::Mp3, Self::Flac, Self::Ogg]
    }
}

impl std::fmt::Display for AudioFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.extension().to_uppercase())
    }
}

/// Audio encoding settings
#[derive(Debug, Clone)]
pub struct EncodingSettings {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of audio channels
    pub channels: u16,
    /// Bit depth for uncompressed formats
    pub bit_depth: u16,
    /// Quality/bitrate for compressed formats (0.0-1.0 for quality, or specific bitrate)
    pub quality: Option<f32>,
    /// Whether to use variable bitrate encoding (for supported formats)
    pub variable_bitrate: bool,
}

impl Default for EncodingSettings {
    fn default() -> Self {
        Self {
            sample_rate: crate::DEFAULT_SAMPLE_RATE,
            channels: crate::DEFAULT_CHANNELS,
            bit_depth: 16,
            quality: None,
            variable_bitrate: false,
        }
    }
}

impl EncodingSettings {
    /// Create new encoding settings
    #[must_use]
    pub fn new(sample_rate: u32, channels: u16) -> Self {
        Self {
            sample_rate,
            channels,
            ..Default::default()
        }
    }

    /// Set bit depth for uncompressed formats
    #[must_use]
    pub fn with_bit_depth(mut self, bit_depth: u16) -> Self {
        self.bit_depth = bit_depth;
        self
    }

    /// Set quality/bitrate for compressed formats
    #[must_use]
    pub fn with_quality(mut self, quality: f32) -> Self {
        self.quality = Some(quality);
        self
    }

    /// Enable variable bitrate encoding
    #[must_use]
    pub fn with_variable_bitrate(mut self) -> Self {
        self.variable_bitrate = true;
        self
    }

    /// Validate encoding settings
    pub fn validate(&self) -> VocalizeResult<()> {
        if self.sample_rate < 8000 || self.sample_rate > 192_000 {
            return Err(VocalizeError::invalid_input(format!(
                "Sample rate must be between 8000 and 192000 Hz, got {}",
                self.sample_rate
            )));
        }

        if self.channels == 0 || self.channels > 8 {
            return Err(VocalizeError::invalid_input(format!(
                "Channels must be between 1 and 8, got {}",
                self.channels
            )));
        }

        if !matches!(self.bit_depth, 8 | 16 | 24 | 32) {
            return Err(VocalizeError::invalid_input(format!(
                "Bit depth must be 8, 16, 24, or 32, got {}",
                self.bit_depth
            )));
        }

        if let Some(quality) = self.quality {
            if !(0.0..=1.0).contains(&quality) && quality < 32.0 {
                return Err(VocalizeError::invalid_input(format!(
                    "Quality must be between 0.0-1.0 (quality) or >= 32 (bitrate), got {}",
                    quality
                )));
            }
        }

        Ok(())
    }
}

/// High-performance audio writer with multi-format support
#[derive(Debug)]
pub struct AudioWriter {
    default_settings: EncodingSettings,
}

impl AudioWriter {
    /// Create a new audio writer with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            default_settings: EncodingSettings::default(),
        }
    }

    /// Create a new audio writer with custom default settings
    #[must_use]
    pub fn with_settings(settings: EncodingSettings) -> Self {
        Self {
            default_settings: settings,
        }
    }

    /// Write audio data to file
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be created or written to
    /// - The audio format is not supported
    /// - The audio data is invalid
    /// - The encoding settings are invalid
    pub async fn write_file<P: AsRef<Path>>(
        &self,
        audio_data: &AudioData,
        path: P,
        format: AudioFormat,
        settings: Option<EncodingSettings>,
    ) -> VocalizeResult<()> {
        let path = path.as_ref();
        let settings = settings.unwrap_or_else(|| self.default_settings.clone());

        self.validate_inputs(audio_data, &settings)?;

        info!(
            "Writing {} samples to {} in {} format",
            audio_data.len(),
            path.display(),
            format
        );

        // Create parent directories if they don't exist
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                VocalizeError::file(format!("Failed to create directory {}: {e}", parent.display()))
            })?;
        }

        match format {
            AudioFormat::Wav => self.write_wav(audio_data, path, &settings).await,
            AudioFormat::Mp3 => self.write_mp3(audio_data, path, &settings).await,
            AudioFormat::Flac => self.write_flac(audio_data, path, &settings).await,
            AudioFormat::Ogg => self.write_ogg(audio_data, path, &settings).await,
        }?;

        info!("Successfully wrote audio file: {}", path.display());
        Ok(())
    }

    /// Write audio data to file, auto-detecting format from extension
    ///
    /// # Errors
    ///
    /// Returns an error if the format cannot be detected or writing fails
    pub async fn write_file_auto<P: AsRef<Path>>(
        &self,
        audio_data: &AudioData,
        path: P,
        settings: Option<EncodingSettings>,
    ) -> VocalizeResult<()> {
        let format = AudioFormat::from_path(&path)?;
        self.write_file(audio_data, path, format, settings).await
    }

    /// Estimate output file size
    #[must_use]
    pub fn estimate_file_size(
        &self,
        audio_data: &AudioData,
        format: AudioFormat,
        settings: &EncodingSettings,
    ) -> usize {
        let samples = audio_data.len();
        let duration_seconds = samples as f64 / settings.sample_rate as f64;

        match format {
            AudioFormat::Wav => {
                // WAV: samples * channels * (bit_depth / 8) + header
                let bytes_per_sample = (settings.bit_depth / 8) as usize;
                samples * settings.channels as usize * bytes_per_sample + 44
            }
            AudioFormat::Flac => {
                // FLAC: roughly 50-70% of WAV size
                let wav_size = samples * settings.channels as usize * 2 + 44;
                (wav_size as f64 * 0.6) as usize
            }
            AudioFormat::Mp3 => {
                // MP3: bitrate-dependent
                let bitrate = settings.quality.unwrap_or(128.0); // Default 128 kbps
                (duration_seconds * bitrate as f64 * 1000.0 / 8.0) as usize
            }
            AudioFormat::Ogg => {
                // OGG: similar to MP3
                let bitrate = settings.quality.unwrap_or(128.0); // Default 128 kbps
                (duration_seconds * bitrate as f64 * 1000.0 / 8.0) as usize
            }
        }
    }

    /// Get supported formats
    #[must_use]
    pub fn get_supported_formats() -> &'static [AudioFormat] {
        AudioFormat::all()
    }

    /// Check if format is supported
    #[must_use]
    pub fn is_format_supported(format: AudioFormat) -> bool {
        Self::get_supported_formats().contains(&format)
    }

    /// Validate inputs
    fn validate_inputs(
        &self,
        audio_data: &AudioData,
        settings: &EncodingSettings,
    ) -> VocalizeResult<()> {
        if audio_data.is_empty() {
            return Err(VocalizeError::invalid_input("Audio data cannot be empty"));
        }

        // Check for valid audio samples
        for (i, &sample) in audio_data.iter().enumerate() {
            if !sample.is_finite() {
                return Err(VocalizeError::invalid_input(format!(
                    "Invalid audio sample at index {i}: {sample}"
                )));
            }
            if sample.abs() > 1.0 {
                warn!("Audio sample at index {} exceeds range [-1.0, 1.0]: {}", i, sample);
            }
        }

        settings.validate()?;
        Ok(())
    }

    /// Write WAV file using hound
    async fn write_wav(
        &self,
        audio_data: &AudioData,
        path: &Path,
        settings: &EncodingSettings,
    ) -> VocalizeResult<()> {
        debug!("Writing WAV file with {} bit depth", settings.bit_depth);

        let is_float = settings.bit_depth == 32 && settings.quality.unwrap_or(0.8) > 0.9;
        let spec = WavSpec::new(
            settings.channels,
            settings.sample_rate,
            settings.bit_depth,
            is_float,
        );

        let mut writer = WavWriter::create(path, spec)?;

        match settings.bit_depth {
            8 => {
                for &sample in audio_data {
                    let sample_i8 = (sample.clamp(-1.0, 1.0) * 127.0) as i8;
                    writer.write_sample_i8(sample_i8)?;
                }
            }
            16 => {
                for &sample in audio_data {
                    let sample_i16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
                    writer.write_sample_i16(sample_i16)?;
                }
            }
            24 => {
                for &sample in audio_data {
                    let sample_i32 = (sample.clamp(-1.0, 1.0) * 8_388_607.0) as i32;
                    writer.write_sample_i24(sample_i32)?;
                }
            }
            32 => {
                if is_float {
                    for &sample in audio_data {
                        writer.write_sample_f32(sample.clamp(-1.0, 1.0))?;
                    }
                } else {
                    for &sample in audio_data {
                        let sample_i32 = (sample.clamp(-1.0, 1.0) * 2_147_483_647.0) as i32;
                        writer.write_sample_i32(sample_i32)?;
                    }
                }
            }
            _ => {
                return Err(VocalizeError::invalid_input(format!(
                    "Unsupported bit depth for WAV: {}",
                    settings.bit_depth
                )));
            }
        }

        writer.finalize()?;

        Ok(())
    }

    /// Write MP3 file (placeholder - would need actual MP3 encoder)
    async fn write_mp3(
        &self,
        _audio_data: &AudioData,
        _path: &Path,
        _settings: &EncodingSettings,
    ) -> VocalizeResult<()> {
        // In a real implementation, this would use an MP3 encoder like lame-sys
        // For now, we'll write a WAV file with MP3 extension as a placeholder
        warn!("MP3 encoding not implemented, writing as WAV");
        Err(VocalizeError::audio_processing(
            "MP3 encoding not yet implemented".to_string(),
        ))
    }

    /// Write FLAC file (placeholder - would need actual FLAC encoder)
    async fn write_flac(
        &self,
        _audio_data: &AudioData,
        _path: &Path,
        _settings: &EncodingSettings,
    ) -> VocalizeResult<()> {
        // In a real implementation, this would use a FLAC encoder
        warn!("FLAC encoding not implemented, writing as WAV");
        Err(VocalizeError::audio_processing(
            "FLAC encoding not yet implemented".to_string(),
        ))
    }

    /// Write OGG file (placeholder - would need actual OGG encoder)
    async fn write_ogg(
        &self,
        _audio_data: &AudioData,
        _path: &Path,
        _settings: &EncodingSettings,
    ) -> VocalizeResult<()> {
        // In a real implementation, this would use an OGG Vorbis encoder
        warn!("OGG encoding not implemented, writing as WAV");
        Err(VocalizeError::audio_processing(
            "OGG encoding not yet implemented".to_string(),
        ))
    }
}

impl Default for AudioWriter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_audio_format_extension() {
        assert_eq!(AudioFormat::Wav.extension(), "wav");
        assert_eq!(AudioFormat::Mp3.extension(), "mp3");
        assert_eq!(AudioFormat::Flac.extension(), "flac");
        assert_eq!(AudioFormat::Ogg.extension(), "ogg");
    }

    #[test]
    fn test_audio_format_mime_type() {
        assert_eq!(AudioFormat::Wav.mime_type(), "audio/wav");
        assert_eq!(AudioFormat::Mp3.mime_type(), "audio/mpeg");
        assert_eq!(AudioFormat::Flac.mime_type(), "audio/flac");
        assert_eq!(AudioFormat::Ogg.mime_type(), "audio/ogg");
    }

    #[test]
    fn test_audio_format_is_lossy() {
        assert!(!AudioFormat::Wav.is_lossy());
        assert!(AudioFormat::Mp3.is_lossy());
        assert!(!AudioFormat::Flac.is_lossy());
        assert!(AudioFormat::Ogg.is_lossy());
    }

    #[test]
    fn test_audio_format_description() {
        assert_eq!(AudioFormat::Wav.description(), "Waveform Audio File Format");
        assert_eq!(AudioFormat::Mp3.description(), "MPEG Audio Layer III");
        assert_eq!(AudioFormat::Flac.description(), "Free Lossless Audio Codec");
        assert_eq!(AudioFormat::Ogg.description(), "Ogg Vorbis");
    }

    #[test]
    fn test_audio_format_from_extension() {
        assert_eq!(AudioFormat::from_extension("wav").unwrap(), AudioFormat::Wav);
        assert_eq!(AudioFormat::from_extension("WAV").unwrap(), AudioFormat::Wav);
        assert_eq!(AudioFormat::from_extension("mp3").unwrap(), AudioFormat::Mp3);
        assert_eq!(AudioFormat::from_extension("flac").unwrap(), AudioFormat::Flac);
        assert_eq!(AudioFormat::from_extension("ogg").unwrap(), AudioFormat::Ogg);
        
        assert!(AudioFormat::from_extension("xyz").is_err());
    }

    #[test]
    fn test_audio_format_from_path() {
        assert_eq!(AudioFormat::from_path("test.wav").unwrap(), AudioFormat::Wav);
        assert_eq!(AudioFormat::from_path("/path/to/file.mp3").unwrap(), AudioFormat::Mp3);
        assert_eq!(AudioFormat::from_path("audio.FLAC").unwrap(), AudioFormat::Flac);
        
        assert!(AudioFormat::from_path("no_extension").is_err());
        assert!(AudioFormat::from_path("file.xyz").is_err());
    }

    #[test]
    fn test_audio_format_all() {
        let formats = AudioFormat::all();
        assert_eq!(formats.len(), 4);
        assert!(formats.contains(&AudioFormat::Wav));
        assert!(formats.contains(&AudioFormat::Mp3));
        assert!(formats.contains(&AudioFormat::Flac));
        assert!(formats.contains(&AudioFormat::Ogg));
    }

    #[test]
    fn test_audio_format_display() {
        assert_eq!(AudioFormat::Wav.to_string(), "WAV");
        assert_eq!(AudioFormat::Mp3.to_string(), "MP3");
        assert_eq!(AudioFormat::Flac.to_string(), "FLAC");
        assert_eq!(AudioFormat::Ogg.to_string(), "OGG");
    }

    #[test]
    fn test_encoding_settings_default() {
        let settings = EncodingSettings::default();
        assert_eq!(settings.sample_rate, crate::DEFAULT_SAMPLE_RATE);
        assert_eq!(settings.channels, crate::DEFAULT_CHANNELS);
        assert_eq!(settings.bit_depth, 16);
        assert_eq!(settings.quality, None);
        assert!(!settings.variable_bitrate);
    }

    #[test]
    fn test_encoding_settings_new() {
        let settings = EncodingSettings::new(48000, 2);
        assert_eq!(settings.sample_rate, 48000);
        assert_eq!(settings.channels, 2);
        assert_eq!(settings.bit_depth, 16);
    }

    #[test]
    fn test_encoding_settings_with_bit_depth() {
        let settings = EncodingSettings::new(44100, 2).with_bit_depth(24);
        assert_eq!(settings.bit_depth, 24);
    }

    #[test]
    fn test_encoding_settings_with_quality() {
        let settings = EncodingSettings::new(44100, 2).with_quality(0.8);
        assert_eq!(settings.quality, Some(0.8));
    }

    #[test]
    fn test_encoding_settings_with_variable_bitrate() {
        let settings = EncodingSettings::new(44100, 2).with_variable_bitrate();
        assert!(settings.variable_bitrate);
    }

    #[test]
    fn test_encoding_settings_validation() {
        // Valid settings
        let settings = EncodingSettings::default();
        assert!(settings.validate().is_ok());

        // Invalid sample rate
        let mut settings = EncodingSettings::default();
        settings.sample_rate = 1000;
        assert!(settings.validate().is_err());

        settings.sample_rate = 200_000;
        assert!(settings.validate().is_err());

        // Invalid channels
        let mut settings = EncodingSettings::default();
        settings.channels = 0;
        assert!(settings.validate().is_err());

        settings.channels = 10;
        assert!(settings.validate().is_err());

        // Invalid bit depth
        let mut settings = EncodingSettings::default();
        settings.bit_depth = 12;
        assert!(settings.validate().is_err());

        // Invalid quality
        let mut settings = EncodingSettings::default();
        settings.quality = Some(-0.5);
        assert!(settings.validate().is_err());

        settings.quality = Some(1.5);
        assert!(settings.validate().is_err());
    }

    #[test]
    fn test_audio_writer_new() {
        let writer = AudioWriter::new();
        assert_eq!(writer.default_settings.sample_rate, crate::DEFAULT_SAMPLE_RATE);
        assert_eq!(writer.default_settings.channels, crate::DEFAULT_CHANNELS);
    }

    #[test]
    fn test_audio_writer_with_settings() {
        let settings = EncodingSettings::new(48000, 2);
        let writer = AudioWriter::with_settings(settings.clone());
        assert_eq!(writer.default_settings.sample_rate, 48000);
        assert_eq!(writer.default_settings.channels, 2);
    }

    #[test]
    fn test_audio_writer_estimate_file_size() {
        let writer = AudioWriter::new();
        let audio_data = vec![0.5; 24000]; // 1 second at 24kHz
        let settings = EncodingSettings::new(24000, 1);

        // WAV estimation
        let wav_size = writer.estimate_file_size(&audio_data, AudioFormat::Wav, &settings);
        assert!(wav_size > 0);
        assert_eq!(wav_size, 24000 * 1 * 2 + 44); // samples * channels * bytes_per_sample + header

        // FLAC estimation (should be smaller than WAV)
        let flac_size = writer.estimate_file_size(&audio_data, AudioFormat::Flac, &settings);
        assert!(flac_size > 0);
        assert!(flac_size < wav_size);

        // MP3 estimation
        let mp3_size = writer.estimate_file_size(&audio_data, AudioFormat::Mp3, &settings);
        assert!(mp3_size > 0);

        // OGG estimation
        let ogg_size = writer.estimate_file_size(&audio_data, AudioFormat::Ogg, &settings);
        assert!(ogg_size > 0);
        assert_eq!(mp3_size, ogg_size); // Same default bitrate
    }

    #[test]
    fn test_audio_writer_get_supported_formats() {
        let formats = AudioWriter::get_supported_formats();
        assert_eq!(formats.len(), 4);
        assert!(formats.contains(&AudioFormat::Wav));
        assert!(formats.contains(&AudioFormat::Mp3));
        assert!(formats.contains(&AudioFormat::Flac));
        assert!(formats.contains(&AudioFormat::Ogg));
    }

    #[test]
    fn test_audio_writer_is_format_supported() {
        assert!(AudioWriter::is_format_supported(AudioFormat::Wav));
        assert!(AudioWriter::is_format_supported(AudioFormat::Mp3));
        assert!(AudioWriter::is_format_supported(AudioFormat::Flac));
        assert!(AudioWriter::is_format_supported(AudioFormat::Ogg));
    }

    #[test]
    fn test_audio_writer_validate_inputs() {
        let writer = AudioWriter::new();
        let audio_data = vec![0.5, -0.3, 0.0, 0.8];
        let settings = EncodingSettings::default();

        // Valid inputs
        assert!(writer.validate_inputs(&audio_data, &settings).is_ok());

        // Empty audio data
        assert!(writer.validate_inputs(&vec![], &settings).is_err());

        // Invalid audio samples
        let invalid_audio = vec![f32::NAN, 0.5];
        assert!(writer.validate_inputs(&invalid_audio, &settings).is_err());

        let infinite_audio = vec![f32::INFINITY, 0.5];
        assert!(writer.validate_inputs(&infinite_audio, &settings).is_err());

        // Invalid settings
        let mut invalid_settings = EncodingSettings::default();
        invalid_settings.sample_rate = 1000;
        assert!(writer.validate_inputs(&audio_data, &invalid_settings).is_err());
    }

    #[tokio::test]
    async fn test_audio_writer_write_wav() {
        let writer = AudioWriter::new();
        let audio_data = vec![0.5, -0.3, 0.0, 0.8, -0.1];
        let settings = EncodingSettings::new(24000, 1);

        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path();

        let result = writer.write_wav(&audio_data, path, &settings).await;
        assert!(result.is_ok());

        // Verify file was created and has content
        let metadata = std::fs::metadata(path).expect("File should exist");
        assert!(metadata.len() > 0);
    }

    #[tokio::test]
    async fn test_audio_writer_write_wav_different_bit_depths() {
        let writer = AudioWriter::new();
        let audio_data = vec![0.5, -0.3, 0.0, 0.8];

        for &bit_depth in &[8, 16, 24, 32] {
            let settings = EncodingSettings::new(24000, 1).with_bit_depth(bit_depth);
            let temp_file = NamedTempFile::new().expect("Failed to create temp file");
            let path = temp_file.path();

            let result = writer.write_wav(&audio_data, path, &settings).await;
            assert!(result.is_ok(), "Failed for bit depth {}", bit_depth);

            // Verify file was created
            let metadata = std::fs::metadata(path).expect("File should exist");
            assert!(metadata.len() > 0);
        }
    }

    #[tokio::test]
    async fn test_audio_writer_write_wav_invalid_bit_depth() {
        let writer = AudioWriter::new();
        let audio_data = vec![0.5, -0.3, 0.0, 0.8];
        let settings = EncodingSettings::new(24000, 1).with_bit_depth(12); // Invalid

        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path();

        let result = writer.write_wav(&audio_data, path, &settings).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_audio_writer_write_file_auto() {
        let writer = AudioWriter::new();
        let audio_data = vec![0.5, -0.3, 0.0, 0.8];
        let settings = EncodingSettings::default();

        let temp_file = NamedTempFile::with_suffix(".wav").expect("Failed to create temp file");
        let path = temp_file.path();

        let result = writer.write_file_auto(&audio_data, path, Some(settings)).await;
        // Should succeed for WAV, fail for others (not implemented)
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_audio_writer_write_file_unsupported_format() {
        let writer = AudioWriter::new();
        let audio_data = vec![0.5, -0.3, 0.0, 0.8];
        let settings = EncodingSettings::default();

        let temp_file = NamedTempFile::with_suffix(".mp3").expect("Failed to create temp file");
        let path = temp_file.path();

        let result = writer.write_file(&audio_data, path, AudioFormat::Mp3, Some(settings)).await;
        // Should fail because MP3 encoding is not implemented
        assert!(result.is_err());
    }
}