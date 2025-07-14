"""Tests for audio writer functionality."""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from typing import List

from vocalize import (
    AudioWriter,
    AudioFormat,
    EncodingSettings,
    TtsEngine,
    SynthesisParams,
    Voice,
    VocalizeError,
)


class TestAudioFormat:
    """Test AudioFormat enum."""
    
    def test_audio_format_values(self):
        """Test AudioFormat enum values."""
        assert hasattr(AudioFormat, 'WAV')
        assert hasattr(AudioFormat, 'MP3')
        assert hasattr(AudioFormat, 'FLAC')
        assert hasattr(AudioFormat, 'OGG')
        
    def test_audio_format_properties(self):
        """Test AudioFormat properties."""
        assert AudioFormat.WAV.extension() == "wav"
        assert AudioFormat.MP3.extension() == "mp3"
        assert AudioFormat.FLAC.extension() == "flac"
        assert AudioFormat.OGG.extension() == "ogg"
        
        assert AudioFormat.WAV.mime_type() == "audio/wav"
        assert AudioFormat.MP3.mime_type() == "audio/mpeg"
        assert AudioFormat.FLAC.mime_type() == "audio/flac"
        assert AudioFormat.OGG.mime_type() == "audio/ogg"
        
    def test_audio_format_lossy(self):
        """Test AudioFormat lossy property."""
        assert not AudioFormat.WAV.is_lossy()
        assert AudioFormat.MP3.is_lossy()
        assert not AudioFormat.FLAC.is_lossy()
        assert AudioFormat.OGG.is_lossy()
        
    def test_audio_format_description(self):
        """Test AudioFormat description."""
        descriptions = [
            AudioFormat.WAV.description(),
            AudioFormat.MP3.description(),
            AudioFormat.FLAC.description(),
            AudioFormat.OGG.description(),
        ]
        
        for desc in descriptions:
            assert isinstance(desc, str)
            assert len(desc) > 0
            
    def test_audio_format_from_extension(self):
        """Test AudioFormat from extension."""
        assert AudioFormat.from_extension("wav") == AudioFormat.WAV
        assert AudioFormat.from_extension("mp3") == AudioFormat.MP3
        assert AudioFormat.from_extension("WAV") == AudioFormat.WAV  # Case insensitive
        assert AudioFormat.from_extension("MP3") == AudioFormat.MP3
        
        with pytest.raises(VocalizeError):
            AudioFormat.from_extension("xyz")
            
    def test_audio_format_from_path(self):
        """Test AudioFormat from path."""
        assert AudioFormat.from_path("test.wav") == AudioFormat.WAV
        assert AudioFormat.from_path("/path/to/file.mp3") == AudioFormat.MP3
        assert AudioFormat.from_path("file.FLAC") == AudioFormat.FLAC
        
        with pytest.raises(VocalizeError):
            AudioFormat.from_path("no_extension")
            
        with pytest.raises(VocalizeError):
            AudioFormat.from_path("file.xyz")
            
    def test_audio_format_all(self):
        """Test AudioFormat.all()."""
        formats = AudioFormat.all()
        
        assert isinstance(formats, list)
        assert len(formats) == 4
        assert AudioFormat.WAV in formats
        assert AudioFormat.MP3 in formats
        assert AudioFormat.FLAC in formats
        assert AudioFormat.OGG in formats
        
    def test_audio_format_str_repr(self):
        """Test AudioFormat string representations."""
        assert str(AudioFormat.WAV) == "WAV"
        assert str(AudioFormat.MP3) == "MP3"
        
        assert repr(AudioFormat.WAV) == "AudioFormat.WAV"
        assert repr(AudioFormat.MP3) == "AudioFormat.MP3"


class TestEncodingSettings:
    """Test EncodingSettings class."""
    
    def test_encoding_settings_creation(self):
        """Test encoding settings creation."""
        settings = EncodingSettings(48000, 2)
        
        assert settings.sample_rate == 48000
        assert settings.channels == 2
        assert settings.bit_depth == 16  # Default
        assert settings.quality is None  # Default
        assert not settings.variable_bitrate  # Default
        
    def test_encoding_settings_default(self):
        """Test default encoding settings."""
        settings = EncodingSettings.default()
        
        assert settings.sample_rate == 24000  # DEFAULT_SAMPLE_RATE
        assert settings.channels == 1         # DEFAULT_CHANNELS
        assert settings.bit_depth == 16
        assert settings.quality is None
        assert not settings.variable_bitrate
        
    def test_encoding_settings_with_bit_depth(self):
        """Test encoding settings with bit depth."""
        settings = EncodingSettings.default()
        new_settings = settings.with_bit_depth(24)
        
        assert new_settings.bit_depth == 24
        assert settings.bit_depth == 16  # Original unchanged
        
    def test_encoding_settings_with_quality(self):
        """Test encoding settings with quality."""
        settings = EncodingSettings.default()
        new_settings = settings.with_quality(0.8)
        
        assert new_settings.quality == 0.8
        assert settings.quality is None  # Original unchanged
        
    def test_encoding_settings_with_variable_bitrate(self):
        """Test encoding settings with variable bitrate."""
        settings = EncodingSettings.default()
        vbr_settings = settings.with_variable_bitrate()
        cbr_settings = vbr_settings.with_constant_bitrate()
        
        assert vbr_settings.variable_bitrate
        assert not cbr_settings.variable_bitrate
        assert not settings.variable_bitrate  # Original unchanged
        
    def test_encoding_settings_validation(self):
        """Test encoding settings validation."""
        # Valid settings
        valid_settings = EncodingSettings(24000, 1)
        valid_settings.validate()  # Should not raise
        
        # Invalid sample rate
        invalid_settings = EncodingSettings(0, 1)
        with pytest.raises(VocalizeError):
            invalid_settings.validate()
            
    def test_encoding_settings_chaining(self):
        """Test encoding settings method chaining."""
        settings = (EncodingSettings(48000, 2)
                   .with_bit_depth(24)
                   .with_quality(0.9)
                   .with_variable_bitrate())
        
        assert settings.sample_rate == 48000
        assert settings.channels == 2
        assert settings.bit_depth == 24
        assert settings.quality == 0.9
        assert settings.variable_bitrate
        
    def test_encoding_settings_repr(self):
        """Test encoding settings repr."""
        settings = EncodingSettings(48000, 2)
        repr_str = repr(settings)
        
        assert "EncodingSettings(" in repr_str
        assert "48000" in repr_str
        assert "channels=2" in repr_str
        
    def test_encoding_settings_to_dict(self):
        """Test encoding settings to dict."""
        settings = (EncodingSettings(48000, 2)
                   .with_bit_depth(24)
                   .with_quality(0.9))
        
        data = settings.to_dict()
        
        assert isinstance(data, dict)
        assert data["sample_rate"] == "48000"
        assert data["channels"] == "2"
        assert data["bit_depth"] == "24"
        assert data["quality"] == "0.9"
        assert data["variable_bitrate"] == "false"


class TestAudioWriter:
    """Test AudioWriter class."""
    
    def test_audio_writer_creation(self):
        """Test audio writer creation."""
        writer = AudioWriter()
        assert repr(writer) == "AudioWriter()"
        
    def test_audio_writer_with_settings(self):
        """Test audio writer with settings."""
        settings = EncodingSettings(48000, 2)
        writer = AudioWriter()
        new_writer = writer.with_settings(settings)
        
        assert repr(new_writer) == "AudioWriter()"
        
    def test_audio_writer_format_support(self):
        """Test audio writer format support."""
        writer = AudioWriter()
        
        # Currently only WAV is supported
        assert writer.is_format_supported(AudioFormat.WAV)
        assert not writer.is_format_supported(AudioFormat.MP3)
        assert not writer.is_format_supported(AudioFormat.FLAC)
        assert not writer.is_format_supported(AudioFormat.OGG)
        
    def test_get_supported_formats(self):
        """Test getting supported formats."""
        formats = AudioWriter.get_supported_formats()
        
        assert isinstance(formats, list)
        assert AudioFormat.WAV in formats
        # Currently only WAV is implemented
        assert len(formats) == 1
        
    def test_audio_writer_validate_inputs(self):
        """Test audio writer input validation."""
        writer = AudioWriter()
        settings = EncodingSettings.default()
        
        # Valid audio data
        valid_audio = [0.1, 0.2, -0.1, -0.2]
        writer.validate_inputs(valid_audio, settings)  # Should not raise
        
        # Empty audio data
        with pytest.raises(VocalizeError):
            writer.validate_inputs([], settings)
            
        # Invalid audio samples (NaN)
        invalid_audio = [float('nan'), 0.5]
        with pytest.raises(VocalizeError):
            writer.validate_inputs(invalid_audio, settings)
            
    def test_audio_writer_estimate_file_size(self):
        """Test audio writer file size estimation."""
        writer = AudioWriter()
        audio_data = [0.1, 0.2, -0.1, -0.2] * 1000  # 4000 samples
        settings = EncodingSettings.default()
        
        # Test different formats
        wav_size = writer.estimate_file_size(audio_data, AudioFormat.WAV, settings)
        mp3_size = writer.estimate_file_size(audio_data, AudioFormat.MP3, settings)
        
        assert isinstance(wav_size, int)
        assert isinstance(mp3_size, int)
        assert wav_size > 0
        assert mp3_size > 0
        assert mp3_size < wav_size  # MP3 should be smaller


class TestAudioWriterFileOperations:
    """Test AudioWriter file operations."""
    
    @pytest.mark.asyncio
    async def test_write_wav_file(self):
        """Test writing WAV file."""
        writer = AudioWriter()
        audio_data = [0.1, 0.2, -0.1, -0.2] * 100
        settings = EncodingSettings.default()
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            try:
                await writer.write_file(audio_data, tmp.name, AudioFormat.WAV, settings)
                
                # Check file was created and has content
                assert os.path.exists(tmp.name)
                assert os.path.getsize(tmp.name) > 0
            finally:
                os.unlink(tmp.name)
                
    @pytest.mark.asyncio
    async def test_write_file_auto_detection(self):
        """Test writing file with auto format detection."""
        writer = AudioWriter()
        audio_data = [0.1, 0.2, -0.1, -0.2] * 100
        settings = EncodingSettings.default()
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            try:
                await writer.write_file_auto(audio_data, tmp.name, settings)
                
                # Check file was created and has content
                assert os.path.exists(tmp.name)
                assert os.path.getsize(tmp.name) > 0
            finally:
                os.unlink(tmp.name)
                
    @pytest.mark.asyncio
    async def test_write_unsupported_format(self):
        """Test writing unsupported format."""
        writer = AudioWriter()
        audio_data = [0.1, 0.2, -0.1, -0.2] * 100
        settings = EncodingSettings.default()
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            try:
                # MP3 is not implemented yet, should fail
                with pytest.raises(VocalizeError):
                    await writer.write_file(audio_data, tmp.name, AudioFormat.MP3, settings)
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)
                    
    @pytest.mark.asyncio
    async def test_write_different_bit_depths(self):
        """Test writing WAV files with different bit depths."""
        writer = AudioWriter()
        audio_data = [0.1, 0.2, -0.1, -0.2] * 100
        
        for bit_depth in [8, 16, 24, 32]:
            settings = EncodingSettings.default().with_bit_depth(bit_depth)
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                try:
                    await writer.write_file(audio_data, tmp.name, AudioFormat.WAV, settings)
                    
                    # Check file was created
                    assert os.path.exists(tmp.name)
                    assert os.path.getsize(tmp.name) > 0
                finally:
                    os.unlink(tmp.name)
                    
    @pytest.mark.asyncio
    async def test_write_different_sample_rates(self):
        """Test writing WAV files with different sample rates."""
        writer = AudioWriter()
        audio_data = [0.1, 0.2, -0.1, -0.2] * 100
        
        for sample_rate in [22050, 44100, 48000]:
            settings = EncodingSettings(sample_rate, 1)
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                try:
                    await writer.write_file(audio_data, tmp.name, AudioFormat.WAV, settings)
                    
                    # Check file was created
                    assert os.path.exists(tmp.name)
                    assert os.path.getsize(tmp.name) > 0
                finally:
                    os.unlink(tmp.name)
                    
    @pytest.mark.asyncio
    async def test_write_stereo_audio(self):
        """Test writing stereo WAV file."""
        writer = AudioWriter()
        audio_data = [0.1, 0.2, -0.1, -0.2] * 100
        settings = EncodingSettings(24000, 2)  # Stereo
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            try:
                await writer.write_file(audio_data, tmp.name, AudioFormat.WAV, settings)
                
                # Check file was created
                assert os.path.exists(tmp.name)
                assert os.path.getsize(tmp.name) > 0
            finally:
                os.unlink(tmp.name)


class TestAudioWriterIntegration:
    """Integration tests for audio writer with TTS engine."""
    
    @pytest.mark.asyncio
    async def test_save_synthesized_audio(self):
        """Test saving synthesized audio to file."""
        # Generate audio
        engine = await TtsEngine()
        voice = Voice.default()
        params = SynthesisParams(voice)
        audio_data = await engine.synthesize("Hello, world!", params)
        
        # Save audio
        writer = AudioWriter()
        settings = EncodingSettings.default()
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            try:
                await writer.write_file(audio_data, tmp.name, AudioFormat.WAV, settings)
                
                # Check file was created and has reasonable size
                assert os.path.exists(tmp.name)
                file_size = os.path.getsize(tmp.name)
                assert file_size > 1000  # Should be at least 1KB for "Hello, world!"
                
            finally:
                os.unlink(tmp.name)
                
    @pytest.mark.asyncio
    async def test_save_multiple_voices(self):
        """Test saving audio from different voices."""
        engine = await TtsEngine()
        writer = AudioWriter()
        settings = EncodingSettings.default()
        
        from vocalize import VoiceManager
        manager = VoiceManager()
        voices = manager.get_available_voices()[:2]  # Test with 2 voices
        
        for i, voice in enumerate(voices):
            params = SynthesisParams(voice)
            audio_data = await engine.synthesize(f"Hello from voice {i}", params)
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                try:
                    await writer.write_file(audio_data, tmp.name, AudioFormat.WAV, settings)
                    
                    # Check file was created
                    assert os.path.exists(tmp.name)
                    assert os.path.getsize(tmp.name) > 500
                    
                finally:
                    os.unlink(tmp.name)
                    
    @pytest.mark.asyncio
    async def test_concurrent_file_writing(self):
        """Test writing multiple files concurrently."""
        # Generate audio
        engine = await TtsEngine()
        voice = Voice.default()
        params = SynthesisParams(voice)
        
        audio_clips = await asyncio.gather(*[
            engine.synthesize(f"Clip number {i}", params)
            for i in range(3)
        ])
        
        # Write files concurrently
        writer = AudioWriter()
        settings = EncodingSettings.default()
        
        temp_files = []
        try:
            for i in range(3):
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_files.append(tmp.name)
                tmp.close()
                
            await asyncio.gather(*[
                writer.write_file(audio_clips[i], temp_files[i], AudioFormat.WAV, settings)
                for i in range(3)
            ])
            
            # Check all files were created
            for tmp_file in temp_files:
                assert os.path.exists(tmp_file)
                assert os.path.getsize(tmp_file) > 0
                
        finally:
            for tmp_file in temp_files:
                if os.path.exists(tmp_file):
                    os.unlink(tmp_file)


class TestAudioWriterErrorHandling:
    """Test audio writer error handling."""
    
    @pytest.mark.asyncio
    async def test_write_to_invalid_path(self):
        """Test writing to invalid path."""
        writer = AudioWriter()
        audio_data = [0.1, 0.2, -0.1, -0.2]
        settings = EncodingSettings.default()
        
        # Try to write to non-existent directory
        invalid_path = "/nonexistent/directory/file.wav"
        
        with pytest.raises(VocalizeError):
            await writer.write_file(audio_data, invalid_path, AudioFormat.WAV, settings)
            
    @pytest.mark.asyncio
    async def test_write_empty_audio(self):
        """Test writing empty audio data."""
        writer = AudioWriter()
        settings = EncodingSettings.default()
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            try:
                with pytest.raises(VocalizeError):
                    await writer.write_file([], tmp.name, AudioFormat.WAV, settings)
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)
                    
    @pytest.mark.asyncio
    async def test_write_with_invalid_settings(self):
        """Test writing with invalid settings."""
        writer = AudioWriter()
        audio_data = [0.1, 0.2, -0.1, -0.2]
        
        # Invalid sample rate
        invalid_settings = EncodingSettings(0, 1)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            try:
                with pytest.raises(VocalizeError):
                    await writer.write_file(audio_data, tmp.name, AudioFormat.WAV, invalid_settings)
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)


if __name__ == "__main__":
    pytest.main([__file__])