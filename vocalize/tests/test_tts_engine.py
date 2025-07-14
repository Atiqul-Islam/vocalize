"""Tests for TTS engine functionality."""

import pytest
import asyncio
from typing import List

from vocalize import (
    TtsEngine,
    SynthesisParams,
    Voice,
    VoiceManager,
    Gender,
    VoiceStyle,
    VocalizeError,
)


class TestSynthesisParams:
    """Test SynthesisParams class."""
    
    def test_synthesis_params_creation(self):
        """Test synthesis params creation."""
        voice = Voice.default()
        params = SynthesisParams(voice)
        
        assert params.voice.id == voice.id
        assert params.speed is None
        assert params.pitch is None
        assert params.streaming_chunk_size is None
        
    def test_synthesis_params_with_speed_valid(self):
        """Test synthesis params with valid speed."""
        voice = Voice.default()
        params = SynthesisParams(voice)
        
        new_params = params.with_speed(1.5)
        assert new_params.speed == 1.5
        assert params.speed is None  # Original unchanged
        
    def test_synthesis_params_with_speed_invalid(self):
        """Test synthesis params with invalid speed."""
        voice = Voice.default()
        params = SynthesisParams(voice)
        
        with pytest.raises(VocalizeError):
            params.with_speed(0.05)  # Too low
            
        with pytest.raises(VocalizeError):
            params.with_speed(5.0)  # Too high
            
    def test_synthesis_params_with_pitch_valid(self):
        """Test synthesis params with valid pitch."""
        voice = Voice.default()
        params = SynthesisParams(voice)
        
        new_params = params.with_pitch(0.5)
        assert new_params.pitch == 0.5
        assert params.pitch is None  # Original unchanged
        
    def test_synthesis_params_with_pitch_invalid(self):
        """Test synthesis params with invalid pitch."""
        voice = Voice.default()
        params = SynthesisParams(voice)
        
        with pytest.raises(VocalizeError):
            params.with_pitch(-1.5)  # Too low
            
        with pytest.raises(VocalizeError):
            params.with_pitch(2.0)  # Too high
            
    def test_synthesis_params_with_streaming(self):
        """Test synthesis params with streaming."""
        voice = Voice.default()
        params = SynthesisParams(voice)
        
        streaming_params = params.with_streaming(1024)
        assert streaming_params.streaming_chunk_size == 1024
        assert params.streaming_chunk_size is None  # Original unchanged
        
    def test_synthesis_params_without_streaming(self):
        """Test synthesis params without streaming."""
        voice = Voice.default()
        params = SynthesisParams(voice).with_streaming(1024)
        
        no_streaming_params = params.without_streaming()
        assert no_streaming_params.streaming_chunk_size is None
        assert params.streaming_chunk_size == 1024  # Original unchanged
        
    def test_synthesis_params_repr(self):
        """Test synthesis params repr."""
        voice = Voice.default()
        params = SynthesisParams(voice)
        repr_str = repr(params)
        
        assert "SynthesisParams(" in repr_str
        assert "af_bella" in repr_str
        
    def test_synthesis_params_to_dict(self):
        """Test synthesis params to dict."""
        voice = Voice.default()
        params = SynthesisParams(voice).with_speed(1.2).with_pitch(0.1).with_streaming(512)
        
        data = params.to_dict()
        assert isinstance(data, dict)
        assert data["voice_id"] == "af_bella"
        assert data["speed"] == "1.2"
        assert data["pitch"] == "0.1"
        assert data["streaming_chunk_size"] == "512"
        
    def test_synthesis_params_chaining(self):
        """Test synthesis params method chaining."""
        voice = Voice.default()
        params = (SynthesisParams(voice)
                 .with_speed(1.3)
                 .with_pitch(0.2)
                 .with_streaming(256))
        
        assert params.speed == 1.3
        assert params.pitch == 0.2
        assert params.streaming_chunk_size == 256


class TestTtsEngine:
    """Test TtsEngine class."""
    
    @pytest.mark.asyncio
    async def test_tts_engine_creation(self):
        """Test TTS engine creation."""
        engine = await TtsEngine()
        assert repr(engine) == "TtsEngine()"
        
    @pytest.mark.asyncio
    async def test_tts_engine_is_ready(self):
        """Test TTS engine readiness check."""
        engine = await TtsEngine()
        is_ready = await engine.is_ready()
        assert isinstance(is_ready, bool)
        
    @pytest.mark.asyncio
    async def test_synthesize_basic(self):
        """Test basic text synthesis."""
        engine = await TtsEngine()
        voice = Voice.default()
        params = SynthesisParams(voice)
        
        audio_data = await engine.synthesize("Hello, world!", params)
        
        assert isinstance(audio_data, list)
        assert len(audio_data) > 0
        assert all(isinstance(sample, float) for sample in audio_data)
        
    @pytest.mark.asyncio
    async def test_synthesize_empty_text(self):
        """Test synthesis with empty text."""
        engine = await TtsEngine()
        voice = Voice.default()
        params = SynthesisParams(voice)
        
        with pytest.raises(VocalizeError):
            await engine.synthesize("", params)
            
    @pytest.mark.asyncio
    async def test_synthesize_long_text(self):
        """Test synthesis with long text."""
        engine = await TtsEngine()
        voice = Voice.default()
        params = SynthesisParams(voice)
        
        long_text = "This is a longer text for testing. " * 10
        audio_data = await engine.synthesize(long_text, params)
        
        assert isinstance(audio_data, list)
        assert len(audio_data) > 0
        
    @pytest.mark.asyncio
    async def test_synthesize_with_speed(self):
        """Test synthesis with custom speed."""
        engine = await TtsEngine()
        voice = Voice.default()
        params = SynthesisParams(voice).with_speed(1.5)
        
        audio_data = await engine.synthesize("Hello, world!", params)
        
        assert isinstance(audio_data, list)
        assert len(audio_data) > 0
        
    @pytest.mark.asyncio
    async def test_synthesize_with_pitch(self):
        """Test synthesis with custom pitch."""
        engine = await TtsEngine()
        voice = Voice.default()
        params = SynthesisParams(voice).with_pitch(0.2)
        
        audio_data = await engine.synthesize("Hello, world!", params)
        
        assert isinstance(audio_data, list)
        assert len(audio_data) > 0
        
    @pytest.mark.asyncio
    async def test_synthesize_different_voices(self):
        """Test synthesis with different voices."""
        engine = await TtsEngine()
        voice_manager = VoiceManager()
        voices = voice_manager.get_available_voices()
        
        # Test with at least 2 different voices
        for voice in voices[:2]:
            params = SynthesisParams(voice)
            audio_data = await engine.synthesize("Hello", params)
            
            assert isinstance(audio_data, list)
            assert len(audio_data) > 0
            
    @pytest.mark.asyncio
    async def test_synthesize_streaming(self):
        """Test streaming synthesis."""
        engine = await TtsEngine()
        voice = Voice.default()
        params = SynthesisParams(voice).with_streaming(1024)
        
        chunks = await engine.synthesize_streaming("This is a test for streaming synthesis.", params)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, list) for chunk in chunks)
        assert all(len(chunk) > 0 for chunk in chunks)
        
    @pytest.mark.asyncio
    async def test_synthesize_streaming_no_streaming(self):
        """Test streaming synthesis without streaming enabled."""
        engine = await TtsEngine()
        voice = Voice.default()
        params = SynthesisParams(voice)  # No streaming
        
        chunks = await engine.synthesize_streaming("Hello", params)
        
        # Should return single chunk
        assert isinstance(chunks, list)
        assert len(chunks) == 1
        assert isinstance(chunks[0], list)
        
    @pytest.mark.asyncio
    async def test_preload_models(self):
        """Test model preloading."""
        engine = await TtsEngine()
        voice_manager = VoiceManager()
        voice_ids = [v.id for v in voice_manager.get_available_voices()[:2]]
        
        # Should not raise an error
        await engine.preload_models(voice_ids)
        
    @pytest.mark.asyncio
    async def test_clear_cache(self):
        """Test cache clearing."""
        engine = await TtsEngine()
        
        # Should not raise an error
        await engine.clear_cache()
        
    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting engine statistics."""
        engine = await TtsEngine()
        
        # Perform some synthesis to generate stats
        voice = Voice.default()
        params = SynthesisParams(voice)
        await engine.synthesize("Hello", params)
        
        stats = await engine.get_stats()
        
        assert isinstance(stats, dict)
        assert "total_syntheses" in stats
        assert "total_audio_duration" in stats
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "average_synthesis_time" in stats
        assert "models_loaded" in stats
        
        # Check that synthesis count increased
        total_syntheses = int(stats["total_syntheses"])
        assert total_syntheses >= 1


class TestTtsEngineEdgeCases:
    """Test TTS engine edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_synthesize_very_long_text(self):
        """Test synthesis with very long text."""
        engine = await TtsEngine()
        voice = Voice.default()
        params = SynthesisParams(voice)
        
        # Text approaching the limit
        long_text = "A" * 50000  # 50k characters
        
        try:
            audio_data = await engine.synthesize(long_text, params)
            assert isinstance(audio_data, list)
            assert len(audio_data) > 0
        except VocalizeError:
            # This is acceptable - the engine may reject very long text
            pass
            
    @pytest.mark.asyncio
    async def test_synthesize_special_characters(self):
        """Test synthesis with special characters."""
        engine = await TtsEngine()
        voice = Voice.default()
        params = SynthesisParams(voice)
        
        special_text = "Hello! How are you? I'm fine. 123 + 456 = 579."
        audio_data = await engine.synthesize(special_text, params)
        
        assert isinstance(audio_data, list)
        assert len(audio_data) > 0
        
    @pytest.mark.asyncio
    async def test_synthesize_unicode_text(self):
        """Test synthesis with unicode text."""
        engine = await TtsEngine()
        voice = Voice.default()
        params = SynthesisParams(voice)
        
        unicode_text = "Hello world! 🌍 Nice day ☀️"
        audio_data = await engine.synthesize(unicode_text, params)
        
        assert isinstance(audio_data, list)
        assert len(audio_data) > 0
        
    @pytest.mark.asyncio
    async def test_multiple_engines(self):
        """Test creating multiple engine instances."""
        engine1 = await TtsEngine()
        engine2 = await TtsEngine()
        
        voice = Voice.default()
        params = SynthesisParams(voice)
        
        # Both engines should work independently
        audio1 = await engine1.synthesize("Hello from engine 1", params)
        audio2 = await engine2.synthesize("Hello from engine 2", params)
        
        assert isinstance(audio1, list)
        assert isinstance(audio2, list)
        assert len(audio1) > 0
        assert len(audio2) > 0


class TestTtsEnginePerformance:
    """Test TTS engine performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_synthesis_timing(self):
        """Test that synthesis completes in reasonable time."""
        import time
        
        engine = await TtsEngine()
        voice = Voice.default()
        params = SynthesisParams(voice)
        
        start_time = time.time()
        audio_data = await engine.synthesize("Hello, world!", params)
        end_time = time.time()
        
        synthesis_time = end_time - start_time
        
        # Should complete within 5 seconds (generous limit for CI)
        assert synthesis_time < 5.0
        assert len(audio_data) > 0
        
    @pytest.mark.asyncio
    async def test_concurrent_synthesis(self):
        """Test concurrent synthesis requests."""
        engine = await TtsEngine()
        voice = Voice.default()
        params = SynthesisParams(voice)
        
        # Run multiple synthesis requests concurrently
        tasks = [
            engine.synthesize(f"Hello number {i}", params)
            for i in range(3)
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        for audio_data in results:
            assert isinstance(audio_data, list)
            assert len(audio_data) > 0


if __name__ == "__main__":
    pytest.main([__file__])