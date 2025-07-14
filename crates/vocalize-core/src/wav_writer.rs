//! WAV file writer implementation
//! 
//! Provides functionality to write audio data in WAV/RIFF format.

use std::fs::File;
use std::io::{BufWriter, Write, Seek, SeekFrom};
use std::path::Path;
use crate::error::{VocalizeError, VocalizeResult};

/// WAV file format specification
#[derive(Debug, Clone, Copy)]
pub struct WavSpec {
    /// Number of audio channels (1 = mono, 2 = stereo)
    pub channels: u16,
    /// Sample rate in Hz (e.g., 24000, 44100, 48000)
    pub sample_rate: u32,
    /// Bits per sample (8, 16, 24, or 32)
    pub bit_depth: u16,
    /// Whether samples are floating point (only for 32-bit)
    pub is_float: bool,
}

impl WavSpec {
    /// Create a new WAV specification
    pub fn new(channels: u16, sample_rate: u32, bit_depth: u16, is_float: bool) -> Self {
        Self {
            channels,
            sample_rate,
            bit_depth,
            is_float,
        }
    }
    
    /// Get bytes per sample
    fn bytes_per_sample(&self) -> u16 {
        self.bit_depth / 8
    }
    
    /// Get byte rate (bytes per second)
    fn byte_rate(&self) -> u32 {
        self.sample_rate * u32::from(self.channels) * u32::from(self.bytes_per_sample())
    }
    
    /// Get block align (bytes per sample frame)
    fn block_align(&self) -> u16 {
        self.channels * self.bytes_per_sample()
    }
}

/// WAV file writer
pub struct WavWriter {
    writer: BufWriter<File>,
    spec: WavSpec,
    bytes_written: u32,
}

impl WavWriter {
    /// Create a new WAV file writer
    pub fn create<P: AsRef<Path>>(path: P, spec: WavSpec) -> VocalizeResult<Self> {
        let file = File::create(path.as_ref())
            .map_err(|e| VocalizeError::file(format!("Failed to create WAV file: {}", e)))?;
        
        let mut writer = BufWriter::new(file);
        
        // Write WAV header (will be updated in finalize)
        Self::write_header(&mut writer, &spec, 0)?;
        
        Ok(Self {
            writer,
            spec,
            bytes_written: 0,
        })
    }
    
    /// Write WAV/RIFF header (44 bytes)
    fn write_header(writer: &mut BufWriter<File>, spec: &WavSpec, data_size: u32) -> VocalizeResult<()> {
        // RIFF chunk
        writer.write_all(b"RIFF")?;
        writer.write_all(&(36 + data_size).to_le_bytes())?; // File size - 8
        writer.write_all(b"WAVE")?;
        
        // fmt chunk
        writer.write_all(b"fmt ")?;
        writer.write_all(&16u32.to_le_bytes())?; // fmt chunk size
        
        // Audio format (1 = PCM, 3 = IEEE float)
        let audio_format = if spec.is_float && spec.bit_depth == 32 { 3u16 } else { 1u16 };
        writer.write_all(&audio_format.to_le_bytes())?;
        
        writer.write_all(&spec.channels.to_le_bytes())?;
        writer.write_all(&spec.sample_rate.to_le_bytes())?;
        writer.write_all(&spec.byte_rate().to_le_bytes())?;
        writer.write_all(&spec.block_align().to_le_bytes())?;
        writer.write_all(&spec.bit_depth.to_le_bytes())?;
        
        // data chunk
        writer.write_all(b"data")?;
        writer.write_all(&data_size.to_le_bytes())?;
        
        writer.flush()?;
        Ok(())
    }
    
    /// Write an 8-bit sample
    pub fn write_sample_i8(&mut self, sample: i8) -> VocalizeResult<()> {
        if self.spec.bit_depth != 8 {
            return Err(VocalizeError::invalid_input("Cannot write 8-bit sample to non-8-bit WAV"));
        }
        
        // Convert signed to unsigned for WAV format
        let unsigned_sample = (sample as i16 + 128) as u8;
        self.writer.write_all(&[unsigned_sample])?;
        self.bytes_written += 1;
        Ok(())
    }
    
    /// Write a 16-bit sample
    pub fn write_sample_i16(&mut self, sample: i16) -> VocalizeResult<()> {
        if self.spec.bit_depth != 16 {
            return Err(VocalizeError::invalid_input("Cannot write 16-bit sample to non-16-bit WAV"));
        }
        
        self.writer.write_all(&sample.to_le_bytes())?;
        self.bytes_written += 2;
        Ok(())
    }
    
    /// Write a 24-bit sample
    pub fn write_sample_i24(&mut self, sample: i32) -> VocalizeResult<()> {
        if self.spec.bit_depth != 24 {
            return Err(VocalizeError::invalid_input("Cannot write 24-bit sample to non-24-bit WAV"));
        }
        
        // Write only the lower 3 bytes
        let bytes = sample.to_le_bytes();
        self.writer.write_all(&bytes[0..3])?;
        self.bytes_written += 3;
        Ok(())
    }
    
    /// Write a 32-bit integer sample
    pub fn write_sample_i32(&mut self, sample: i32) -> VocalizeResult<()> {
        if self.spec.bit_depth != 32 || self.spec.is_float {
            return Err(VocalizeError::invalid_input("Cannot write 32-bit int sample to non-32-bit-int WAV"));
        }
        
        self.writer.write_all(&sample.to_le_bytes())?;
        self.bytes_written += 4;
        Ok(())
    }
    
    /// Write a 32-bit float sample
    pub fn write_sample_f32(&mut self, sample: f32) -> VocalizeResult<()> {
        if self.spec.bit_depth != 32 || !self.spec.is_float {
            return Err(VocalizeError::invalid_input("Cannot write float sample to non-float WAV"));
        }
        
        self.writer.write_all(&sample.to_le_bytes())?;
        self.bytes_written += 4;
        Ok(())
    }
    
    /// Finalize the WAV file by updating the header with actual sizes
    pub fn finalize(mut self) -> VocalizeResult<()> {
        // Flush any remaining buffered data
        self.writer.flush()?;
        
        // Get the underlying file
        let mut file = self.writer.into_inner()
            .map_err(|e| VocalizeError::file(format!("Failed to finalize WAV writer: {}", e)))?;
        
        // Update RIFF chunk size (file size - 8)
        file.seek(SeekFrom::Start(4))?;
        file.write_all(&(36 + self.bytes_written).to_le_bytes())?;
        
        // Update data chunk size
        file.seek(SeekFrom::Start(40))?;
        file.write_all(&self.bytes_written.to_le_bytes())?;
        
        file.flush()?;
        Ok(())
    }
}

// Removed duplicate From<io::Error> implementation - already exists in error.rs

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_wav_spec() {
        let spec = WavSpec::new(1, 24000, 16, false);
        assert_eq!(spec.bytes_per_sample(), 2);
        assert_eq!(spec.byte_rate(), 48000);
        assert_eq!(spec.block_align(), 2);
        
        let stereo_spec = WavSpec::new(2, 44100, 16, false);
        assert_eq!(stereo_spec.byte_rate(), 176400);
        assert_eq!(stereo_spec.block_align(), 4);
    }
    
    #[test]
    fn test_wav_writer_creation() {
        let temp_file = NamedTempFile::new().unwrap();
        let spec = WavSpec::new(1, 24000, 16, false);
        let writer = WavWriter::create(temp_file.path(), spec);
        assert!(writer.is_ok());
    }
    
    #[test]
    fn test_write_16bit_samples() {
        let temp_file = NamedTempFile::new().unwrap();
        let spec = WavSpec::new(1, 24000, 16, false);
        let mut writer = WavWriter::create(temp_file.path(), spec).unwrap();
        
        // Write some test samples
        writer.write_sample_i16(0).unwrap();
        writer.write_sample_i16(16383).unwrap();
        writer.write_sample_i16(-16384).unwrap();
        
        let result = writer.finalize();
        assert!(result.is_ok());
        
        // Verify file exists and has correct size (44 byte header + 6 bytes data)
        let metadata = std::fs::metadata(temp_file.path()).unwrap();
        assert_eq!(metadata.len(), 50);
    }
    
    #[test]
    fn test_write_wrong_bit_depth() {
        let temp_file = NamedTempFile::new().unwrap();
        let spec = WavSpec::new(1, 24000, 16, false);
        let mut writer = WavWriter::create(temp_file.path(), spec).unwrap();
        
        // Try to write 8-bit sample to 16-bit WAV
        let result = writer.write_sample_i8(0);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_write_float_samples() {
        let temp_file = NamedTempFile::new().unwrap();
        let spec = WavSpec::new(1, 24000, 32, true);
        let mut writer = WavWriter::create(temp_file.path(), spec).unwrap();
        
        writer.write_sample_f32(0.0).unwrap();
        writer.write_sample_f32(1.0).unwrap();
        writer.write_sample_f32(-1.0).unwrap();
        
        let result = writer.finalize();
        assert!(result.is_ok());
    }
}