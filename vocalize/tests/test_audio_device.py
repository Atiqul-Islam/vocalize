"""Tests for audio device functionality."""

import pytest
import asyncio
from typing import List, Tuple

from vocalize import (
    AudioDevice,
    AudioConfig,
    PlaybackState,
    TtsEngine,
    SynthesisParams,
    Voice,
    VocalizeError,
)


class TestPlaybackState:
    """Test PlaybackState enum."""
    
    def test_playback_state_values(self):
        """Test PlaybackState enum values."""
        assert hasattr(PlaybackState, 'STOPPED')
        assert hasattr(PlaybackState, 'PLAYING')
        assert hasattr(PlaybackState, 'PAUSED')
        assert hasattr(PlaybackState, 'ERROR')
        
    def test_playback_state_str(self):
        """Test PlaybackState string representation."""
        assert str(PlaybackState.STOPPED) == "Stopped"
        assert str(PlaybackState.PLAYING) == "Playing"
        assert str(PlaybackState.PAUSED) == "Paused"
        assert str(PlaybackState.ERROR) == "Error"
        
    def test_playback_state_repr(self):
        """Test PlaybackState repr."""
        assert repr(PlaybackState.STOPPED) == "PlaybackState.Stopped"
        assert repr(PlaybackState.PLAYING) == "PlaybackState.Playing"


class TestAudioConfig:
    """Test AudioConfig class."""
    
    def test_audio_config_creation(self):
        """Test audio config creation with all parameters."""
        config = AudioConfig(
            device_id="test_device",
            sample_rate=48000,
            channels=2,
            buffer_size=2048,
            latency_ms=100
        )
        
        assert config.device_id == "test_device"
        assert config.sample_rate == 48000
        assert config.channels == 2
        assert config.buffer_size == 2048
        assert config.latency_ms == 100
        
    def test_audio_config_partial(self):
        """Test audio config creation with some parameters."""
        config = AudioConfig(sample_rate=48000, channels=2)
        
        assert config.device_id is None
        assert config.sample_rate == 48000
        assert config.channels == 2
        assert config.buffer_size == 1024  # Default
        assert config.latency_ms == 50     # Default
        
    def test_audio_config_default(self):
        """Test default audio config."""
        config = AudioConfig.default()
        
        assert config.device_id is None
        assert config.sample_rate == 24000  # DEFAULT_SAMPLE_RATE
        assert config.channels == 1         # DEFAULT_CHANNELS
        assert config.buffer_size == 1024
        assert config.latency_ms == 50
        
    def test_audio_config_repr(self):
        """Test audio config repr."""
        config = AudioConfig.default()
        repr_str = repr(config)
        
        assert "AudioConfig(" in repr_str
        assert "sample_rate" in repr_str
        assert "channels" in repr_str
        assert "buffer_size" in repr_str
        assert "latency" in repr_str


class TestAudioDevice:
    """Test AudioDevice class."""
    
    @pytest.mark.asyncio
    async def test_audio_device_creation(self):
        """Test audio device creation."""
        device = await AudioDevice()
        assert repr(device) == "AudioDevice()"
        
    @pytest.mark.asyncio
    async def test_audio_device_with_config(self):
        """Test audio device creation with config."""
        config = AudioConfig(sample_rate=48000, channels=2)
        device = await AudioDevice.with_config(config)
        
        device_config = await device.get_config()
        assert device_config.sample_rate == 48000
        assert device_config.channels == 2
        
    def test_get_available_devices(self):
        """Test getting available audio devices."""
        devices = AudioDevice.get_available_devices()
        
        assert isinstance(devices, list)
        assert len(devices) > 0
        
        for device_info in devices:
            assert hasattr(device_info, 'id')
            assert hasattr(device_info, 'name')
            assert hasattr(device_info, 'channels')
            assert hasattr(device_info, 'sample_rates')
            assert hasattr(device_info, 'is_default')
            
            assert isinstance(device_info.id, str)
            assert isinstance(device_info.name, str)
            assert isinstance(device_info.channels, int)
            assert isinstance(device_info.sample_rates, list)
            assert isinstance(device_info.is_default, bool)
            
    def test_available_devices_have_default(self):
        """Test that available devices include a default device."""
        devices = AudioDevice.get_available_devices()
        default_devices = [d for d in devices if d.is_default]
        
        assert len(default_devices) > 0
        
    @pytest.mark.asyncio
    async def test_device_initial_state(self):
        """Test device initial state."""
        device = await AudioDevice()
        
        state = await device.get_state()
        assert state == PlaybackState.STOPPED
        
        assert await device.is_stopped()
        assert not await device.is_playing()
        assert not await device.is_paused()
        
    @pytest.mark.asyncio
    async def test_device_state_management(self):
        """Test device state transitions."""
        device = await AudioDevice()
        
        # Start playback
        await device.start()
        assert await device.is_playing()
        
        # Pause playback
        await device.pause()
        assert await device.is_paused()
        
        # Resume playback
        await device.resume()
        assert await device.is_playing()
        
        # Stop playback
        await device.stop()
        assert await device.is_stopped()
        
    @pytest.mark.asyncio
    async def test_device_invalid_state_transitions(self):
        """Test invalid state transitions raise errors."""
        device = await AudioDevice()
        
        # Cannot pause when stopped
        with pytest.raises(VocalizeError):
            await device.pause()
            
        # Cannot resume when stopped
        with pytest.raises(VocalizeError):
            await device.resume()
            
    @pytest.mark.asyncio
    async def test_device_play_audio(self):
        """Test playing audio data."""
        device = await AudioDevice()
        audio_data = [0.1, 0.2, -0.1, -0.2] * 100  # Some test audio
        
        await device.play(audio_data)
        # Device should transition through playing and back to stopped
        
    @pytest.mark.asyncio
    async def test_device_play_empty_audio(self):
        """Test playing empty audio raises error."""
        device = await AudioDevice()
        
        with pytest.raises(VocalizeError):
            await device.play([])
            
    @pytest.mark.asyncio
    async def test_device_play_blocking(self):
        """Test blocking audio playback."""
        device = await AudioDevice()
        audio_data = [0.1, 0.2, -0.1, -0.2] * 10
        
        await device.play_blocking(audio_data)
        
        # Should be stopped after blocking play completes
        assert await device.is_stopped()
        
    @pytest.mark.asyncio
    async def test_device_wait_for_completion(self):
        """Test waiting for playback completion."""
        device = await AudioDevice()
        audio_data = [0.1, 0.2, -0.1, -0.2] * 10
        
        # Start playback
        await device.play(audio_data)
        
        # Wait for completion
        await device.wait_for_completion()
        
        # Should be stopped
        assert await device.is_stopped()
        
    @pytest.mark.asyncio
    async def test_device_get_device_info(self):
        """Test getting device info."""
        device = await AudioDevice()
        info = await device.get_device_info()
        
        # Mock device returns a string
        assert info is not None
        
    @pytest.mark.asyncio
    async def test_device_get_queue_status(self):
        """Test getting queue status."""
        device = await AudioDevice()
        data_in_queue, space_available = await device.get_queue_status()
        
        assert isinstance(data_in_queue, int)
        assert isinstance(space_available, int)
        assert data_in_queue >= 0
        assert space_available >= 0


class TestAudioDeviceIntegration:
    """Integration tests for audio device with TTS engine."""
    
    @pytest.mark.asyncio
    async def test_play_synthesized_audio(self):
        """Test playing audio from TTS synthesis."""
        # Generate audio
        engine = await TtsEngine()
        voice = Voice.default()
        params = SynthesisParams(voice)
        audio_data = await engine.synthesize("Hello, world!", params)
        
        # Play audio
        device = await AudioDevice()
        await device.play_blocking(audio_data)
        
        # Should complete successfully
        assert await device.is_stopped()
        
    @pytest.mark.asyncio
    async def test_multiple_audio_playback(self):
        """Test playing multiple audio clips in sequence."""
        # Generate audio
        engine = await TtsEngine()
        voice = Voice.default()
        params = SynthesisParams(voice)
        
        audio1 = await engine.synthesize("First clip", params)
        audio2 = await engine.synthesize("Second clip", params)
        
        # Play audio clips
        device = await AudioDevice()
        
        await device.play_blocking(audio1)
        assert await device.is_stopped()
        
        await device.play_blocking(audio2)
        assert await device.is_stopped()
        
    @pytest.mark.asyncio
    async def test_concurrent_devices(self):
        """Test using multiple audio devices concurrently."""
        # Generate audio
        engine = await TtsEngine()
        voice = Voice.default()
        params = SynthesisParams(voice)
        audio_data = await engine.synthesize("Hello", params)
        
        # Create multiple devices
        device1 = await AudioDevice()
        device2 = await AudioDevice()
        
        # Play on both devices concurrently
        await asyncio.gather(
            device1.play_blocking(audio_data),
            device2.play_blocking(audio_data)
        )
        
        # Both should be stopped
        assert await device1.is_stopped()
        assert await device2.is_stopped()


class TestAudioDeviceInfo:
    """Test AudioDeviceInfo class."""
    
    def test_audio_device_info_properties(self):
        """Test audio device info properties."""
        devices = AudioDevice.get_available_devices()
        
        for device_info in devices:
            # Test string representation
            str_repr = str(device_info)
            assert device_info.name in str_repr
            assert device_info.id in str_repr
            
            # Test repr
            repr_str = repr(device_info)
            assert "AudioDeviceInfo(" in repr_str
            assert device_info.id in repr_str
            
            # Test to_dict
            data = device_info.to_dict()
            assert isinstance(data, dict)
            assert data["id"] == device_info.id
            assert data["name"] == device_info.name
            assert data["channels"] == str(device_info.channels)
            assert data["is_default"] == str(device_info.is_default).lower()
            
    def test_device_info_validation(self):
        """Test device info has valid values."""
        devices = AudioDevice.get_available_devices()
        
        for device_info in devices:
            assert len(device_info.id) > 0
            assert len(device_info.name) > 0
            assert device_info.channels > 0
            assert len(device_info.sample_rates) > 0
            assert all(rate > 0 for rate in device_info.sample_rates)


class TestAudioDeviceErrorHandling:
    """Test audio device error handling."""
    
    @pytest.mark.asyncio
    async def test_device_operations_with_invalid_data(self):
        """Test device operations with invalid data."""
        device = await AudioDevice()
        
        # Invalid audio data (empty)
        with pytest.raises(VocalizeError):
            await device.play([])
            
        # Invalid audio data (NaN values would be caught at Rust level)
        # This test depends on Rust-level validation
        
    @pytest.mark.asyncio
    async def test_device_state_error_recovery(self):
        """Test device error recovery."""
        device = await AudioDevice()
        
        # Try invalid operations
        try:
            await device.pause()  # Should fail - not playing
        except VocalizeError:
            pass
            
        # Device should still be usable
        audio_data = [0.1, 0.2] * 10
        await device.play_blocking(audio_data)
        assert await device.is_stopped()


if __name__ == "__main__":
    pytest.main([__file__])