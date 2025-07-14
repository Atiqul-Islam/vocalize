"""Integration tests for the complete vocalize package."""

import pytest
import asyncio
import tempfile
import os
from typing import List, Dict, Any

from vocalize import (
    TtsEngine,
    SynthesisParams,
    Voice,
    VoiceManager,
    Gender,
    VoiceStyle,
    AudioDevice,
    AudioConfig,
    PlaybackState,
    AudioWriter,
    AudioFormat,
    EncodingSettings,
    VocalizeError,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_CHANNELS,
    MAX_TEXT_LENGTH,
    VERSION,
    synthesize_text,
    list_voices,
    play_audio,
    save_audio,
)


class TestPackageConstants:
    """Test package constants and metadata."""
    
    def test_constants_exist(self):
        """Test that all constants are available."""
        assert isinstance(DEFAULT_SAMPLE_RATE, int)
        assert isinstance(DEFAULT_CHANNELS, int)
        assert isinstance(MAX_TEXT_LENGTH, int)
        assert isinstance(VERSION, str)
        
        assert DEFAULT_SAMPLE_RATE > 0
        assert DEFAULT_CHANNELS > 0
        assert MAX_TEXT_LENGTH > 0
        assert len(VERSION) > 0
        
    def test_constants_values(self):
        """Test that constants have expected values."""
        assert DEFAULT_SAMPLE_RATE == 24000
        assert DEFAULT_CHANNELS == 1
        assert MAX_TEXT_LENGTH == 100000
        assert "." in VERSION  # Should be a version string like "0.1.0"


class TestFullTtsPipeline:
    """Test complete TTS pipeline from text to audio output."""
    
    @pytest.mark.asyncio
    async def test_basic_tts_pipeline(self):
        """Test basic TTS pipeline."""
        # 1. Create engine
        engine = await TtsEngine()
        
        # 2. Get voice
        voice = Voice.default()
        
        # 3. Create synthesis parameters
        params = SynthesisParams(voice)
        
        # 4. Synthesize text
        audio_data = await engine.synthesize("Hello, world!", params)
        
        # 5. Verify audio
        assert isinstance(audio_data, list)
        assert len(audio_data) > 0
        assert all(isinstance(sample, float) for sample in audio_data)
        
        # 6. Play audio
        device = await AudioDevice()
        await device.play_blocking(audio_data)
        
        # 7. Save audio
        writer = AudioWriter()
        settings = EncodingSettings.default()
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            try:
                await writer.write_file(audio_data, tmp.name, AudioFormat.WAV, settings)
                
                # Verify file
                assert os.path.exists(tmp.name)
                assert os.path.getsize(tmp.name) > 0
                
            finally:
                os.unlink(tmp.name)
                
    @pytest.mark.asyncio
    async def test_advanced_tts_pipeline(self):
        """Test advanced TTS pipeline with custom settings."""
        # 1. Create engine
        engine = await TtsEngine()
        
        # 2. Get voice manager and select voice
        voice_manager = VoiceManager()
        female_voices = voice_manager.get_voices_by_gender(Gender.FEMALE)
        voice = female_voices[0].with_speed(1.2).with_pitch(0.1)
        
        # 3. Create synthesis parameters with custom settings
        params = (SynthesisParams(voice)
                 .with_speed(0.9)  # Override voice speed
                 .with_streaming(1024))
        
        # 4. Synthesize with streaming
        chunks = await engine.synthesize_streaming(
            "This is a test of advanced TTS synthesis with streaming and custom parameters.",
            params
        )
        
        # 5. Combine chunks
        audio_data = []
        for chunk in chunks:
            audio_data.extend(chunk)
            
        assert len(audio_data) > 0
        
        # 6. Create custom audio device
        audio_config = AudioConfig(
            sample_rate=48000,
            channels=2,
            buffer_size=2048
        )
        device = await AudioDevice.with_config(audio_config)
        
        # 7. Play audio
        await device.play_blocking(audio_data)
        
        # 8. Save with custom encoding
        writer = AudioWriter()
        settings = (EncodingSettings(48000, 2)
                   .with_bit_depth(24)
                   .with_quality(0.9))
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            try:
                await writer.write_file(audio_data, tmp.name, AudioFormat.WAV, settings)
                
                # Verify file
                assert os.path.exists(tmp.name)
                file_size = os.path.getsize(tmp.name)
                assert file_size > 2000  # Should be substantial
                
            finally:
                os.unlink(tmp.name)
                
    @pytest.mark.asyncio
    async def test_multiple_voices_comparison(self):
        """Test synthesis with multiple voices for comparison."""
        engine = await TtsEngine()
        voice_manager = VoiceManager()
        
        # Get different types of voices
        male_voices = voice_manager.get_voices_by_gender(Gender.MALE)
        female_voices = voice_manager.get_voices_by_gender(Gender.FEMALE)
        
        test_voices = [male_voices[0], female_voices[0]]
        test_text = "This is a voice comparison test."
        
        audio_results = []
        
        for voice in test_voices:
            params = SynthesisParams(voice)
            audio_data = await engine.synthesize(test_text, params)
            
            assert len(audio_data) > 0
            audio_results.append((voice.id, audio_data))
            
        # Verify we got different audio for different voices
        assert len(audio_results) == 2
        assert audio_results[0][0] != audio_results[1][0]  # Different voice IDs
        
        # Save comparison files
        writer = AudioWriter()
        settings = EncodingSettings.default()
        
        for voice_id, audio_data in audio_results:
            with tempfile.NamedTemporaryFile(suffix=f"_{voice_id}.wav", delete=False) as tmp:
                try:
                    await writer.write_file(audio_data, tmp.name, AudioFormat.WAV, settings)
                    
                    assert os.path.exists(tmp.name)
                    assert os.path.getsize(tmp.name) > 0
                    
                finally:
                    os.unlink(tmp.name)


class TestUtilityFunctionsIntegration:
    """Test utility functions in complete workflows."""
    
    @pytest.mark.asyncio
    async def test_simple_workflow_with_utilities(self):
        """Test simple workflow using utility functions."""
        # List available voices
        voices = list_voices(language="en-US", gender="Female")
        assert len(voices) > 0
        
        voice_id = voices[0]["id"]
        
        # Synthesize text
        audio = await synthesize_text(
            "This is a test using utility functions.",
            voice=voice_id,
            speed=1.1,
            pitch=0.05
        )
        
        assert len(audio) > 0
        
        # Play audio
        await play_audio(audio, blocking=True)
        
        # Save audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            try:
                await save_audio(
                    audio,
                    tmp.name,
                    sample_rate=48000,
                    bit_depth=24
                )
                
                assert os.path.exists(tmp.name)
                assert os.path.getsize(tmp.name) > 0
                
            finally:
                os.unlink(tmp.name)
                
    @pytest.mark.asyncio
    async def test_batch_synthesis_with_utilities(self):
        """Test batch synthesis using utility functions."""
        texts = [
            "First sentence for batch processing.",
            "Second sentence with different content.",
            "Third and final sentence to complete the batch."
        ]
        
        # Get different voices for variety
        voices = list_voices(language="en")
        voice_ids = [v["id"] for v in voices[:len(texts)]]
        
        # Synthesize all texts concurrently
        synthesis_tasks = []
        for text, voice_id in zip(texts, voice_ids):
            task = synthesize_text(text, voice=voice_id, speed=1.0 + 0.1 * len(synthesis_tasks))
            synthesis_tasks.append(task)
            
        audio_results = await asyncio.gather(*synthesis_tasks)
        
        # Verify results
        assert len(audio_results) == len(texts)
        for audio in audio_results:
            assert isinstance(audio, list)
            assert len(audio) > 0
            
        # Save all files
        temp_files = []
        try:
            for i, audio in enumerate(audio_results):
                tmp = tempfile.NamedTemporaryFile(suffix=f"_batch_{i}.wav", delete=False)
                temp_files.append(tmp.name)
                tmp.close()
                
                await save_audio(audio, tmp.name)
                
            # Verify all files
            for tmp_file in temp_files:
                assert os.path.exists(tmp_file)
                assert os.path.getsize(tmp_file) > 0
                
        finally:
            for tmp_file in temp_files:
                if os.path.exists(tmp_file):
                    os.unlink(tmp_file)


class TestErrorHandlingIntegration:
    """Test error handling across the complete system."""
    
    @pytest.mark.asyncio
    async def test_engine_error_propagation(self):
        """Test that errors propagate correctly through the system."""
        engine = await TtsEngine()
        voice = Voice.default()
        
        # Test empty text
        params = SynthesisParams(voice)
        with pytest.raises(VocalizeError):
            await engine.synthesize("", params)
            
        # Test invalid voice parameters
        with pytest.raises(VocalizeError):
            voice.with_speed(10.0)  # Too high
            
        # Test invalid synthesis parameters
        with pytest.raises(VocalizeError):
            params.with_pitch(-2.0)  # Too low
            
    @pytest.mark.asyncio
    async def test_audio_device_error_handling(self):
        """Test audio device error handling."""
        device = await AudioDevice()
        
        # Test invalid state transitions
        with pytest.raises(VocalizeError):
            await device.pause()  # Can't pause when stopped
            
        # Test empty audio
        with pytest.raises(VocalizeError):
            await device.play([])
            
    @pytest.mark.asyncio
    async def test_audio_writer_error_handling(self):
        """Test audio writer error handling."""
        writer = AudioWriter()
        
        # Test unsupported format
        audio_data = [0.1, 0.2, -0.1, -0.2]
        with pytest.raises(VocalizeError):
            await writer.write_file(audio_data, "test.mp3", AudioFormat.MP3, EncodingSettings.default())
            
        # Test invalid path
        with pytest.raises(VocalizeError):
            await writer.write_file(audio_data, "/nonexistent/path.wav", AudioFormat.WAV, EncodingSettings.default())
            
    @pytest.mark.asyncio
    async def test_utility_error_handling(self):
        """Test utility function error handling."""
        # Test invalid voice in synthesize_text
        with pytest.raises(VocalizeError):
            await synthesize_text("Hello", voice="nonexistent")
            
        # Test empty audio in play_audio
        with pytest.raises(VocalizeError):
            await play_audio([])
            
        # Test empty audio in save_audio
        with pytest.raises(VocalizeError):
            await save_audio([], "test.wav")


class TestPerformanceAndScalability:
    """Test performance characteristics and scalability."""
    
    @pytest.mark.asyncio
    async def test_concurrent_synthesis(self):
        """Test concurrent synthesis performance."""
        import time
        
        engine = await TtsEngine()
        voice = Voice.default()
        params = SynthesisParams(voice)
        
        # Test concurrent synthesis
        start_time = time.time()
        
        tasks = [
            engine.synthesize(f"Concurrent synthesis test {i}", params)
            for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Verify results
        assert len(results) == 5
        for audio in results:
            assert len(audio) > 0
            
        # Should complete within reasonable time
        total_time = end_time - start_time
        assert total_time < 10.0  # 10 seconds should be more than enough
        
    @pytest.mark.asyncio
    async def test_memory_usage_with_multiple_engines(self):
        """Test memory usage with multiple engine instances."""
        engines = []
        
        # Create multiple engines
        for i in range(3):
            engine = await TtsEngine()
            engines.append(engine)
            
        # Use all engines
        voice = Voice.default()
        params = SynthesisParams(voice)
        
        tasks = [
            engine.synthesize(f"Engine {i} test", params)
            for i, engine in enumerate(engines)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verify all worked
        assert len(results) == 3
        for audio in results:
            assert len(audio) > 0
            
    @pytest.mark.asyncio
    async def test_large_text_synthesis(self):
        """Test synthesis with large text input."""
        engine = await TtsEngine()
        voice = Voice.default()
        params = SynthesisParams(voice)
        
        # Create large text (but within limits)
        large_text = "This is a test sentence. " * 100  # ~2500 characters
        
        audio_data = await engine.synthesize(large_text, params)
        
        # Should produce substantial audio
        assert len(audio_data) > 10000  # Should be many samples
        
    @pytest.mark.asyncio
    async def test_streaming_performance(self):
        """Test streaming synthesis performance."""
        engine = await TtsEngine()
        voice = Voice.default()
        params = SynthesisParams(voice).with_streaming(512)
        
        text = "This is a streaming performance test. " * 20
        
        import time
        start_time = time.time()
        
        chunks = await engine.synthesize_streaming(text, params)
        
        end_time = time.time()
        
        # Verify streaming worked
        assert len(chunks) > 1  # Should have multiple chunks
        
        total_samples = sum(len(chunk) for chunk in chunks)
        assert total_samples > 0
        
        # Should complete quickly
        synthesis_time = end_time - start_time
        assert synthesis_time < 5.0


class TestSystemCompatibility:
    """Test system compatibility and edge cases."""
    
    def test_available_audio_devices(self):
        """Test that audio devices are available."""
        devices = AudioDevice.get_available_devices()
        
        assert len(devices) > 0
        
        # Should have at least one default device
        default_devices = [d for d in devices if d.is_default]
        assert len(default_devices) > 0
        
    def test_supported_audio_formats(self):
        """Test supported audio formats."""
        formats = AudioWriter.get_supported_formats()
        
        # At least WAV should be supported
        assert AudioFormat.WAV in formats
        
        writer = AudioWriter()
        assert writer.is_format_supported(AudioFormat.WAV)
        
    def test_voice_availability(self):
        """Test voice availability and consistency."""
        manager = VoiceManager()
        voices = manager.get_available_voices()
        
        # Should have multiple voices
        assert len(voices) >= 5
        
        # Should have both genders
        genders = {v.gender for v in voices}
        assert Gender.MALE in genders
        assert Gender.FEMALE in genders
        
        # Should have multiple languages
        languages = {v.language for v in voices}
        assert len(languages) >= 2
        assert any("en" in lang for lang in languages)
        
    @pytest.mark.asyncio
    async def test_engine_statistics(self):
        """Test engine statistics tracking."""
        engine = await TtsEngine()
        voice = Voice.default()
        params = SynthesisParams(voice)
        
        # Perform some synthesis
        await engine.synthesize("Test 1", params)
        await engine.synthesize("Test 2", params)
        
        # Get statistics
        stats = await engine.get_stats()
        
        assert isinstance(stats, dict)
        assert "total_syntheses" in stats
        
        total_syntheses = int(stats["total_syntheses"])
        assert total_syntheses >= 2


class TestRegressionTests:
    """Regression tests for known issues and edge cases."""
    
    @pytest.mark.asyncio
    async def test_empty_text_handling(self):
        """Test handling of empty or whitespace-only text."""
        engine = await TtsEngine()
        voice = Voice.default()
        params = SynthesisParams(voice)
        
        # Empty text should fail
        with pytest.raises(VocalizeError):
            await engine.synthesize("", params)
            
        # Whitespace-only text should also fail
        with pytest.raises(VocalizeError):
            await engine.synthesize("   ", params)
            
    @pytest.mark.asyncio
    async def test_special_characters_synthesis(self):
        """Test synthesis with special characters."""
        engine = await TtsEngine()
        voice = Voice.default()
        params = SynthesisParams(voice)
        
        special_text = "Hello! How are you? I'm fine. 123 + 456 = 579. #hashtag @mention"
        audio = await engine.synthesize(special_text, params)
        
        assert len(audio) > 0
        
    @pytest.mark.asyncio 
    async def test_unicode_text_synthesis(self):
        """Test synthesis with unicode characters."""
        engine = await TtsEngine()
        voice = Voice.default()
        params = SynthesisParams(voice)
        
        unicode_text = "Hello world! 🌍 Nice day ☀️ Temperature: 25°C"
        audio = await engine.synthesize(unicode_text, params)
        
        assert len(audio) > 0
        
    @pytest.mark.asyncio
    async def test_parameter_boundary_values(self):
        """Test parameter boundary values."""
        voice = Voice.default()
        
        # Test minimum valid values
        voice_min_speed = voice.with_speed(0.1)  # Minimum speed
        voice_min_pitch = voice.with_pitch(-1.0)  # Minimum pitch
        
        # Test maximum valid values  
        voice_max_speed = voice.with_speed(4.0)  # Maximum speed
        voice_max_pitch = voice.with_pitch(1.0)  # Maximum pitch
        
        # All should succeed
        assert voice_min_speed.speed == 0.1
        assert voice_min_pitch.pitch == -1.0
        assert voice_max_speed.speed == 4.0
        assert voice_max_pitch.pitch == 1.0


if __name__ == "__main__":
    pytest.main([__file__])