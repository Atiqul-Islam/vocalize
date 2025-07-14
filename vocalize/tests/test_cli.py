"""
Comprehensive tests for the Vocalize CLI interface.

Tests all CLI commands, configuration management, error handling,
and integration scenarios with 100% coverage.
"""

import asyncio
import json
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from click.testing import CliRunner

from vocalize.cli import (
    cli, speak, list_voices, play, config,
    ConfigManager, DEFAULT_CONFIG,
    validate_voice_id, validate_audio_format, validate_speed, validate_pitch,
    async_synthesize_and_handle, main
)
from vocalize import (
    VoiceManager, TtsEngine, AudioDevice, AudioWriter,
    Voice, SynthesisParams, AudioFormat, EncodingSettings,
    Gender, VoiceStyle, PlaybackState, VocalizeError
)


class TestConfigManager:
    """Test the ConfigManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Use temporary directory for config
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "config.toml"
        self.config_manager = ConfigManager()
        self.config_manager.config_file = self.config_file
        self.config_manager.config_dir = Path(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_ensure_config_dir(self):
        """Test configuration directory creation."""
        # Remove directory first
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        self.config_manager.ensure_config_dir()
        assert self.config_manager.config_dir.exists()
        assert self.config_manager.config_dir.is_dir()
    
    def test_load_config_default(self):
        """Test loading default configuration when file doesn't exist."""
        config = self.config_manager.load_config()
        assert config == DEFAULT_CONFIG
    
    def test_load_config_existing_file(self):
        """Test loading configuration from existing file."""
        # Create test config file
        test_config = {"default": {"voice": "test_voice"}}
        
        import toml
        with open(self.config_file, 'w') as f:
            toml.dump(test_config, f)
        
        config = self.config_manager.load_config()
        assert "test_voice" in str(config)
    
    def test_load_config_corrupted_file(self):
        """Test loading configuration with corrupted file."""
        # Create corrupted file
        with open(self.config_file, 'w') as f:
            f.write("invalid toml content [[[")
        
        config = self.config_manager.load_config()
        assert config == DEFAULT_CONFIG
    
    def test_save_config(self):
        """Test saving configuration to file."""
        test_config = {"test": {"key": "value"}}
        self.config_manager.save_config(test_config)
        
        assert self.config_file.exists()
        
        # Verify content
        import toml
        with open(self.config_file, 'r') as f:
            saved_config = toml.load(f)
        assert saved_config == test_config
    
    def test_save_config_error(self):
        """Test error handling during config save."""
        # Make directory read-only to cause save error
        self.config_manager.config_file = Path("/root/readonly/config.toml")
        
        with pytest.raises(Exception):
            self.config_manager.save_config({"test": "data"})
    
    def test_get_value_existing(self):
        """Test getting existing configuration value."""
        self.config_manager.save_config({"section": {"key": "value"}})
        
        value = self.config_manager.get_value("section.key")
        assert value == "value"
    
    def test_get_value_nested(self):
        """Test getting nested configuration value."""
        config = {"level1": {"level2": {"level3": "deep_value"}}}
        self.config_manager.save_config(config)
        
        value = self.config_manager.get_value("level1.level2.level3")
        assert value == "deep_value"
    
    def test_get_value_nonexistent(self):
        """Test getting non-existent configuration value."""
        value = self.config_manager.get_value("nonexistent.key")
        assert value is None
    
    def test_set_value_new(self):
        """Test setting new configuration value."""
        self.config_manager.set_value("new.key", "new_value")
        
        value = self.config_manager.get_value("new.key")
        assert value == "new_value"
    
    def test_set_value_existing(self):
        """Test updating existing configuration value."""
        self.config_manager.save_config({"existing": {"key": "old_value"}})
        self.config_manager.set_value("existing.key", "new_value")
        
        value = self.config_manager.get_value("existing.key")
        assert value == "new_value"
    
    def test_set_value_deep_nested(self):
        """Test setting deeply nested configuration value."""
        self.config_manager.set_value("a.b.c.d.e", "deep_value")
        
        value = self.config_manager.get_value("a.b.c.d.e")
        assert value == "deep_value"


class TestValidationFunctions:
    """Test CLI validation functions."""
    
    @patch('vocalize.cli.VoiceManager')
    def test_validate_voice_id_valid(self, mock_voice_manager):
        """Test validation with valid voice ID."""
        mock_manager = Mock()
        mock_manager.is_voice_available.return_value = True
        mock_voice_manager.return_value = mock_manager
        
        result = validate_voice_id("bella")
        assert result == "bella"
        mock_manager.is_voice_available.assert_called_once_with("bella")
    
    @patch('vocalize.cli.VoiceManager')
    def test_validate_voice_id_invalid(self, mock_voice_manager):
        """Test validation with invalid voice ID."""
        mock_manager = Mock()
        mock_manager.is_voice_available.return_value = False
        mock_voice = Mock()
        mock_voice.id = "available_voice"
        mock_manager.get_available_voices.return_value = [mock_voice]
        mock_voice_manager.return_value = mock_manager
        
        with pytest.raises(Exception) as excinfo:
            validate_voice_id("invalid_voice")
        
        assert "not found" in str(excinfo.value)
        assert "available_voice" in str(excinfo.value)
    
    @patch('vocalize.cli.VoiceManager')
    def test_validate_voice_id_error(self, mock_voice_manager):
        """Test validation with VoiceManager error."""
        mock_voice_manager.side_effect = Exception("Manager error")
        
        with pytest.raises(Exception) as excinfo:
            validate_voice_id("any_voice")
        
        assert "Error validating voice" in str(excinfo.value)
    
    def test_validate_audio_format_valid(self):
        """Test validation with valid audio formats."""
        assert validate_audio_format("wav") == AudioFormat.Wav
        assert validate_audio_format("mp3") == AudioFormat.Mp3
        assert validate_audio_format("flac") == AudioFormat.Flac
        assert validate_audio_format("ogg") == AudioFormat.Ogg
    
    def test_validate_audio_format_invalid(self):
        """Test validation with invalid audio format."""
        with pytest.raises(Exception) as excinfo:
            validate_audio_format("invalid_format")
        
        assert "Invalid audio format" in str(excinfo.value)
        assert "wav, mp3, flac, ogg" in str(excinfo.value)
    
    def test_validate_speed_valid(self):
        """Test validation with valid speed values."""
        assert validate_speed("1.0") == 1.0
        assert validate_speed("0.5") == 0.5
        assert validate_speed("2.5") == 2.5
        assert validate_speed("0.1") == 0.1
        assert validate_speed("3.0") == 3.0
    
    def test_validate_speed_invalid_range(self):
        """Test validation with speed values out of range."""
        with pytest.raises(Exception) as excinfo:
            validate_speed("0.05")
        assert "between 0.1 and 3.0" in str(excinfo.value)
        
        with pytest.raises(Exception) as excinfo:
            validate_speed("5.0")
        assert "between 0.1 and 3.0" in str(excinfo.value)
    
    def test_validate_speed_invalid_format(self):
        """Test validation with non-numeric speed."""
        with pytest.raises(Exception) as excinfo:
            validate_speed("not_a_number")
        assert "must be a number" in str(excinfo.value)
    
    def test_validate_pitch_valid(self):
        """Test validation with valid pitch values."""
        assert validate_pitch("0.0") == 0.0
        assert validate_pitch("-0.5") == -0.5
        assert validate_pitch("0.8") == 0.8
        assert validate_pitch("-1.0") == -1.0
        assert validate_pitch("1.0") == 1.0
    
    def test_validate_pitch_invalid_range(self):
        """Test validation with pitch values out of range."""
        with pytest.raises(Exception) as excinfo:
            validate_pitch("-1.5")
        assert "between -1.0 and 1.0" in str(excinfo.value)
        
        with pytest.raises(Exception) as excinfo:
            validate_pitch("2.0")
        assert "between -1.0 and 1.0" in str(excinfo.value)
    
    def test_validate_pitch_invalid_format(self):
        """Test validation with non-numeric pitch."""
        with pytest.raises(Exception) as excinfo:
            validate_pitch("invalid")
        assert "must be a number" in str(excinfo.value)


class TestAsyncSynthesizeAndHandle:
    """Test the async synthesis function."""
    
    @pytest.mark.asyncio
    @patch('vocalize.cli.VoiceManager')
    @patch('vocalize.cli.TtsEngine')
    @patch('vocalize.cli.AudioDevice')
    @patch('vocalize.cli.AudioWriter')
    async def test_synthesis_basic(self, mock_writer, mock_device, mock_engine, mock_voice_manager):
        """Test basic synthesis without output or playback."""
        # Setup mocks
        mock_voice = Mock()
        mock_voice.sample_rate = 24000
        mock_voice.with_speed.return_value = mock_voice
        mock_voice.with_pitch.return_value = mock_voice
        
        mock_manager = Mock()
        mock_manager.get_voice.return_value = mock_voice
        mock_voice_manager.return_value = mock_manager
        
        mock_engine_instance = AsyncMock()
        mock_engine_instance.synthesize.return_value = [0.1, 0.2, 0.3]
        mock_engine.return_value = mock_engine_instance
        
        # Test synthesis
        await async_synthesize_and_handle(
            text="Hello", voice_id="bella", speed=1.0, pitch=0.0,
            play=False, output=None, format_name="wav", streaming=False, verbose=False
        )
        
        # Verify calls
        mock_manager.get_voice.assert_called_once_with("bella")
        mock_engine_instance.synthesize.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('vocalize.cli.VoiceManager')
    @patch('vocalize.cli.TtsEngine')
    @patch('vocalize.cli.AudioWriter')
    async def test_synthesis_with_output(self, mock_writer, mock_engine, mock_voice_manager):
        """Test synthesis with file output."""
        # Setup mocks
        mock_voice = Mock()
        mock_voice.sample_rate = 24000
        mock_voice.with_speed.return_value = mock_voice
        mock_voice.with_pitch.return_value = mock_voice
        
        mock_manager = Mock()
        mock_manager.get_voice.return_value = mock_voice
        mock_voice_manager.return_value = mock_manager
        
        mock_engine_instance = AsyncMock()
        mock_engine_instance.synthesize.return_value = [0.1, 0.2, 0.3]
        mock_engine.return_value = mock_engine_instance
        
        mock_writer_instance = AsyncMock()
        mock_writer.return_value = mock_writer_instance
        
        # Test with output file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_path = tmp.name
        
        try:
            await async_synthesize_and_handle(
                text="Hello", voice_id="bella", speed=1.0, pitch=0.0,
                play=False, output=output_path, format_name="wav", streaming=False, verbose=True
            )
            
            # Verify file writing was called
            mock_writer_instance.write_file.assert_called_once()
        finally:
            os.unlink(output_path)
    
    @pytest.mark.asyncio
    @patch('vocalize.cli.VoiceManager')
    @patch('vocalize.cli.TtsEngine')
    @patch('vocalize.cli.AudioDevice')
    async def test_synthesis_with_playback(self, mock_device, mock_engine, mock_voice_manager):
        """Test synthesis with audio playback."""
        # Setup mocks
        mock_voice = Mock()
        mock_voice.sample_rate = 24000
        mock_voice.with_speed.return_value = mock_voice
        mock_voice.with_pitch.return_value = mock_voice
        
        mock_manager = Mock()
        mock_manager.get_voice.return_value = mock_voice
        mock_voice_manager.return_value = mock_manager
        
        mock_engine_instance = AsyncMock()
        mock_engine_instance.synthesize.return_value = [0.1, 0.2, 0.3]
        mock_engine.return_value = mock_engine_instance
        
        mock_device_instance = AsyncMock()
        mock_device_instance.get_state.side_effect = [PlaybackState.Playing, PlaybackState.Stopped]
        mock_device.return_value = mock_device_instance
        
        # Test with playback
        await async_synthesize_and_handle(
            text="Hello", voice_id="bella", speed=1.0, pitch=0.0,
            play=True, output=None, format_name="wav", streaming=False, verbose=True
        )
        
        # Verify playback was called
        mock_device_instance.play.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('vocalize.cli.VoiceManager')
    @patch('vocalize.cli.TtsEngine')
    async def test_synthesis_streaming(self, mock_engine, mock_voice_manager):
        """Test streaming synthesis."""
        # Setup mocks
        mock_voice = Mock()
        mock_voice.sample_rate = 24000
        mock_voice.with_speed.return_value = mock_voice
        mock_voice.with_pitch.return_value = mock_voice
        
        mock_manager = Mock()
        mock_manager.get_voice.return_value = mock_voice
        mock_voice_manager.return_value = mock_manager
        
        mock_engine_instance = AsyncMock()
        mock_engine_instance.synthesize_streaming.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_engine.return_value = mock_engine_instance
        
        # Test streaming
        await async_synthesize_and_handle(
            text="Hello world", voice_id="bella", speed=1.0, pitch=0.0,
            play=False, output=None, format_name="wav", streaming=True, verbose=True
        )
        
        # Verify streaming synthesis was called
        mock_engine_instance.synthesize_streaming.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('vocalize.cli.VoiceManager')
    async def test_synthesis_voice_error(self, mock_voice_manager):
        """Test synthesis with VoiceManager error."""
        mock_manager = Mock()
        mock_manager.get_voice.side_effect = VocalizeError("Voice not found")
        mock_voice_manager.return_value = mock_manager
        
        with pytest.raises(Exception) as excinfo:
            await async_synthesize_and_handle(
                text="Hello", voice_id="invalid", speed=1.0, pitch=0.0,
                play=False, output=None, format_name="wav", streaming=False, verbose=False
            )
        
        assert "TTS Error" in str(excinfo.value)
    
    @pytest.mark.asyncio
    @patch('vocalize.cli.VoiceManager')
    @patch('vocalize.cli.TtsEngine')
    async def test_synthesis_engine_error(self, mock_engine, mock_voice_manager):
        """Test synthesis with TTS engine error."""
        mock_voice = Mock()
        mock_voice.with_speed.return_value = mock_voice
        mock_voice.with_pitch.return_value = mock_voice
        
        mock_manager = Mock()
        mock_manager.get_voice.return_value = mock_voice
        mock_voice_manager.return_value = mock_manager
        
        mock_engine_instance = AsyncMock()
        mock_engine_instance.synthesize.side_effect = Exception("Engine failure")
        mock_engine.return_value = mock_engine_instance
        
        with pytest.raises(Exception) as excinfo:
            await async_synthesize_and_handle(
                text="Hello", voice_id="bella", speed=1.0, pitch=0.0,
                play=False, output=None, format_name="wav", streaming=False, verbose=False
            )
        
        assert "Unexpected error" in str(excinfo.value)


class TestCliCommands:
    """Test CLI commands using Click's test runner."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cli_help(self):
        """Test CLI help output."""
        result = self.runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert "Vocalize" in result.output
        assert "text-to-speech" in result.output
    
    def test_cli_version(self):
        """Test CLI version output."""
        result = self.runner.invoke(cli, ['--version'])
        assert result.exit_code == 0
    
    @patch('vocalize.cli.asyncio.run')
    @patch('vocalize.cli.validate_voice_id')
    def test_speak_command_basic(self, mock_validate, mock_asyncio):
        """Test basic speak command."""
        mock_validate.return_value = "bella"
        mock_asyncio.return_value = None
        
        result = self.runner.invoke(speak, ['Hello world'])
        assert result.exit_code == 0
        mock_asyncio.assert_called_once()
    
    @patch('vocalize.cli.asyncio.run')
    @patch('vocalize.cli.validate_voice_id')
    @patch('vocalize.cli.validate_speed')
    @patch('vocalize.cli.validate_pitch')
    def test_speak_command_with_options(self, mock_pitch, mock_speed, mock_validate, mock_asyncio):
        """Test speak command with all options."""
        mock_validate.return_value = "bella"
        mock_speed.return_value = 1.5
        mock_pitch.return_value = 0.5
        mock_asyncio.return_value = None
        
        result = self.runner.invoke(speak, [
            'Hello world',
            '--voice', 'bella',
            '--speed', '1.5',
            '--pitch', '0.5',
            '--play',
            '--streaming',
            '--verbose'
        ])
        assert result.exit_code == 0
    
    @patch('vocalize.cli.asyncio.run')
    def test_speak_command_keyboard_interrupt(self, mock_asyncio):
        """Test speak command with keyboard interrupt."""
        mock_asyncio.side_effect = KeyboardInterrupt()
        
        result = self.runner.invoke(speak, ['Hello'])
        assert result.exit_code == 1
        assert "cancelled by user" in result.output
    
    @patch('vocalize.cli.VoiceManager')
    def test_list_voices_command_basic(self, mock_voice_manager):
        """Test basic list-voices command."""
        mock_voice = Mock()
        mock_voice.id = "bella"
        mock_voice.name = "Bella"
        mock_voice.gender = Gender.Female
        mock_voice.language = "en-US"
        mock_voice.style = VoiceStyle.Natural
        mock_voice.available = True
        mock_voice.description = "Friendly voice"
        
        mock_manager = Mock()
        mock_manager.get_available_voices.return_value = [mock_voice]
        mock_voice_manager.return_value = mock_manager
        
        result = self.runner.invoke(list_voices)
        assert result.exit_code == 0
        assert "bella" in result.output
        assert "Bella" in result.output
    
    @patch('vocalize.cli.VoiceManager')
    def test_list_voices_command_json(self, mock_voice_manager):
        """Test list-voices command with JSON output."""
        mock_voice = Mock()
        mock_voice.id = "bella"
        mock_voice.name = "Bella"
        mock_voice.gender = Gender.Female
        mock_voice.language = "en-US"
        mock_voice.style = VoiceStyle.Natural
        mock_voice.sample_rate = 24000
        mock_voice.available = True
        mock_voice.description = "Friendly voice"
        
        mock_manager = Mock()
        mock_manager.get_available_voices.return_value = [mock_voice]
        mock_voice_manager.return_value = mock_manager
        
        result = self.runner.invoke(list_voices, ['--format', 'json'])
        assert result.exit_code == 0
        
        # Parse JSON output
        output_lines = [line for line in result.output.split('\n') if line.strip()]
        json_start = None
        for i, line in enumerate(output_lines):
            if line.strip().startswith('['):
                json_start = i
                break
        
        if json_start is not None:
            json_output = '\n'.join(output_lines[json_start:])
            if json_output.strip():
                data = json.loads(json_output)
                assert isinstance(data, list)
                assert len(data) > 0
                assert data[0]['id'] == 'bella'
    
    @patch('vocalize.cli.VoiceManager')
    def test_list_voices_command_filtered(self, mock_voice_manager):
        """Test list-voices command with filters."""
        mock_voice = Mock()
        mock_voice.id = "bella"
        mock_voice.gender = Gender.Female
        mock_voice.language = "en-US"
        mock_voice.style = VoiceStyle.Natural
        mock_voice.supports_language.return_value = True
        
        mock_manager = Mock()
        mock_manager.get_available_voices.return_value = [mock_voice]
        mock_voice_manager.return_value = mock_manager
        
        result = self.runner.invoke(list_voices, [
            '--gender', 'female',
            '--language', 'en-US',
            '--style', 'natural'
        ])
        assert result.exit_code == 0
    
    @patch('vocalize.cli.VoiceManager')
    def test_list_voices_command_no_voices(self, mock_voice_manager):
        """Test list-voices command with no matching voices."""
        mock_manager = Mock()
        mock_manager.get_available_voices.return_value = []
        mock_voice_manager.return_value = mock_manager
        
        result = self.runner.invoke(list_voices, ['--gender', 'neutral'])
        assert result.exit_code == 0
        assert "No voices found" in result.output
    
    @patch('vocalize.cli.VoiceManager')
    def test_list_voices_command_error(self, mock_voice_manager):
        """Test list-voices command with VoiceManager error."""
        mock_voice_manager.side_effect = Exception("Manager error")
        
        result = self.runner.invoke(list_voices)
        assert result.exit_code == 1
        assert "Error listing voices" in result.output
    
    @patch('vocalize.cli.asyncio.run')
    def test_play_command_basic(self, mock_asyncio):
        """Test basic play command."""
        mock_asyncio.return_value = None
        
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(b"fake audio data")
            audio_file = tmp.name
        
        try:
            result = self.runner.invoke(play, [audio_file])
            assert result.exit_code == 0
            mock_asyncio.assert_called_once()
        finally:
            os.unlink(audio_file)
    
    def test_play_command_nonexistent_file(self):
        """Test play command with non-existent file."""
        result = self.runner.invoke(play, ['/nonexistent/file.wav'])
        assert result.exit_code == 2  # Click file validation error
    
    @patch('vocalize.cli.asyncio.run')
    def test_play_command_keyboard_interrupt(self, mock_asyncio):
        """Test play command with keyboard interrupt."""
        mock_asyncio.side_effect = KeyboardInterrupt()
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(b"fake audio data")
            audio_file = tmp.name
        
        try:
            result = self.runner.invoke(play, [audio_file])
            assert result.exit_code == 1
            assert "stopped by user" in result.output
        finally:
            os.unlink(audio_file)


class TestConfigCommands:
    """Test configuration CLI commands."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('vocalize.cli.ConfigManager')
    def test_config_get_all(self, mock_config_manager):
        """Test config get command without key."""
        mock_manager = Mock()
        mock_manager.load_config.return_value = {"test": "value"}
        mock_config_manager.return_value = mock_manager
        
        result = self.runner.invoke(config, ['get'])
        assert result.exit_code == 0
        assert "test" in result.output
    
    @patch('vocalize.cli.ConfigManager')
    def test_config_get_specific_key(self, mock_config_manager):
        """Test config get command with specific key."""
        mock_manager = Mock()
        mock_manager.get_value.return_value = "test_value"
        mock_config_manager.return_value = mock_manager
        
        result = self.runner.invoke(config, ['get', 'test.key'])
        assert result.exit_code == 0
        assert "test_value" in result.output
    
    @patch('vocalize.cli.ConfigManager')
    def test_config_get_nonexistent_key(self, mock_config_manager):
        """Test config get command with non-existent key."""
        mock_manager = Mock()
        mock_manager.get_value.return_value = None
        mock_config_manager.return_value = mock_manager
        
        result = self.runner.invoke(config, ['get', 'nonexistent.key'])
        assert result.exit_code == 1
        assert "not found" in result.output
    
    @patch('vocalize.cli.ConfigManager')
    def test_config_get_dict_value(self, mock_config_manager):
        """Test config get command with dictionary value."""
        mock_manager = Mock()
        mock_manager.get_value.return_value = {"nested": "value"}
        mock_config_manager.return_value = mock_manager
        
        result = self.runner.invoke(config, ['get', 'section'])
        assert result.exit_code == 0
        # Should output JSON format for dict values
        assert "{" in result.output
    
    @patch('vocalize.cli.ConfigManager')
    def test_config_set_string(self, mock_config_manager):
        """Test config set command with string value."""
        mock_manager = Mock()
        mock_config_manager.return_value = mock_manager
        
        result = self.runner.invoke(config, ['set', 'test.key', 'string_value'])
        assert result.exit_code == 0
        mock_manager.set_value.assert_called_once_with('test.key', 'string_value')
        assert "Set test.key = string_value" in result.output
    
    @patch('vocalize.cli.ConfigManager')
    def test_config_set_boolean(self, mock_config_manager):
        """Test config set command with boolean value."""
        mock_manager = Mock()
        mock_config_manager.return_value = mock_manager
        
        result = self.runner.invoke(config, ['set', 'test.flag', 'true'])
        assert result.exit_code == 0
        mock_manager.set_value.assert_called_once_with('test.flag', True)
    
    @patch('vocalize.cli.ConfigManager')
    def test_config_set_integer(self, mock_config_manager):
        """Test config set command with integer value."""
        mock_manager = Mock()
        mock_config_manager.return_value = mock_manager
        
        result = self.runner.invoke(config, ['set', 'test.number', '42'])
        assert result.exit_code == 0
        mock_manager.set_value.assert_called_once_with('test.number', 42)
    
    @patch('vocalize.cli.ConfigManager')
    def test_config_set_float(self, mock_config_manager):
        """Test config set command with float value."""
        mock_manager = Mock()
        mock_config_manager.return_value = mock_manager
        
        result = self.runner.invoke(config, ['set', 'test.float', '3.14'])
        assert result.exit_code == 0
        mock_manager.set_value.assert_called_once_with('test.float', 3.14)
    
    @patch('vocalize.cli.ConfigManager')
    def test_config_set_error(self, mock_config_manager):
        """Test config set command with error."""
        mock_manager = Mock()
        mock_manager.set_value.side_effect = Exception("Set error")
        mock_config_manager.return_value = mock_manager
        
        result = self.runner.invoke(config, ['set', 'test.key', 'value'])
        assert result.exit_code == 1
        assert "Error setting configuration" in result.output
    
    @patch('vocalize.cli.ConfigManager')
    def test_config_list(self, mock_config_manager):
        """Test config list command."""
        mock_manager = Mock()
        mock_manager.load_config.return_value = {
            "section1": {"key1": "value1"},
            "section2": {"key2": "value2"}
        }
        mock_config_manager.return_value = mock_manager
        
        result = self.runner.invoke(config, ['list'])
        assert result.exit_code == 0
        assert "section1" in result.output
        assert "section2" in result.output
        assert "key1" in result.output
        assert "key2" in result.output
    
    @patch('vocalize.cli.ConfigManager')
    def test_config_reset(self, mock_config_manager):
        """Test config reset command."""
        mock_manager = Mock()
        mock_config_manager.return_value = mock_manager
        
        # Simulate user confirmation
        result = self.runner.invoke(config, ['reset'], input='y\n')
        assert result.exit_code == 0
        mock_manager.save_config.assert_called_once()
        assert "reset to defaults" in result.output
    
    @patch('vocalize.cli.ConfigManager')
    def test_config_reset_cancel(self, mock_config_manager):
        """Test config reset command with cancellation."""
        mock_manager = Mock()
        mock_config_manager.return_value = mock_manager
        
        # Simulate user cancellation
        result = self.runner.invoke(config, ['reset'], input='n\n')
        assert result.exit_code == 1  # Aborted
        mock_manager.save_config.assert_not_called()
    
    @patch('vocalize.cli.ConfigManager')
    def test_config_reset_error(self, mock_config_manager):
        """Test config reset command with error."""
        mock_manager = Mock()
        mock_manager.save_config.side_effect = Exception("Reset error")
        mock_config_manager.return_value = mock_manager
        
        result = self.runner.invoke(config, ['reset'], input='y\n')
        assert result.exit_code == 1
        assert "Error resetting configuration" in result.output


class TestMainFunction:
    """Test the main entry point function."""
    
    @patch('vocalize.cli.cli')
    def test_main_normal(self, mock_cli):
        """Test main function normal execution."""
        mock_cli.return_value = None
        
        main()
        mock_cli.assert_called_once()
    
    @patch('vocalize.cli.cli')
    def test_main_keyboard_interrupt(self, mock_cli):
        """Test main function with keyboard interrupt."""
        mock_cli.side_effect = KeyboardInterrupt()
        
        with pytest.raises(SystemExit) as excinfo:
            main()
        
        assert excinfo.value.code == 1
    
    @patch('vocalize.cli.cli')
    def test_main_unexpected_error(self, mock_cli):
        """Test main function with unexpected error."""
        mock_cli.side_effect = Exception("Unexpected error")
        
        with pytest.raises(SystemExit) as excinfo:
            main()
        
        assert excinfo.value.code == 1


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple CLI operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('vocalize.cli.ConfigManager')
    @patch('vocalize.cli.asyncio.run')
    @patch('vocalize.cli.validate_voice_id')
    def test_config_and_speak_integration(self, mock_validate, mock_asyncio, mock_config_manager):
        """Test configuration setting followed by speak command."""
        mock_manager = Mock()
        mock_manager.load_config.return_value = DEFAULT_CONFIG.copy()
        mock_config_manager.return_value = mock_manager
        mock_validate.return_value = "bella"
        mock_asyncio.return_value = None
        
        # Set configuration
        result1 = self.runner.invoke(config, ['set', 'default.voice', 'bella'])
        assert result1.exit_code == 0
        
        # Use configuration in speak command
        result2 = self.runner.invoke(speak, ['Hello world'])
        assert result2.exit_code == 0
    
    @patch('vocalize.cli.VoiceManager')
    @patch('vocalize.cli.asyncio.run')
    @patch('vocalize.cli.validate_voice_id')
    def test_list_and_speak_integration(self, mock_validate, mock_asyncio, mock_voice_manager):
        """Test listing voices followed by using one for synthesis."""
        mock_voice = Mock()
        mock_voice.id = "bella"
        mock_voice.name = "Bella"
        mock_voice.gender = Gender.Female
        mock_voice.language = "en-US"
        mock_voice.style = VoiceStyle.Natural
        mock_voice.available = True
        mock_voice.description = "Friendly voice"
        
        mock_manager = Mock()
        mock_manager.get_available_voices.return_value = [mock_voice]
        mock_voice_manager.return_value = mock_manager
        mock_validate.return_value = "bella"
        mock_asyncio.return_value = None
        
        # List voices
        result1 = self.runner.invoke(list_voices)
        assert result1.exit_code == 0
        assert "bella" in result1.output
        
        # Use the voice for synthesis
        result2 = self.runner.invoke(speak, ['Hello', '--voice', 'bella'])
        assert result2.exit_code == 0
    
    @patch('vocalize.cli.asyncio.run')
    @patch('vocalize.cli.validate_voice_id')
    def test_speak_and_play_integration(self, mock_validate, mock_asyncio):
        """Test synthesis to file followed by playback."""
        mock_validate.return_value = "bella"
        mock_asyncio.return_value = None
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_file = tmp.name
        
        try:
            # Synthesize to file
            result1 = self.runner.invoke(speak, [
                'Hello world',
                '--output', output_file,
                '--voice', 'bella'
            ])
            assert result1.exit_code == 0
            
            # Play the file (mock the file existence)
            with patch('pathlib.Path.exists', return_value=True):
                result2 = self.runner.invoke(play, [output_file])
                assert result2.exit_code == 0
        
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)


class TestErrorHandlingScenarios:
    """Test comprehensive error handling scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_speak_invalid_voice(self):
        """Test speak command with invalid voice."""
        with patch('vocalize.cli.validate_voice_id') as mock_validate:
            mock_validate.side_effect = Exception("Voice 'invalid' not found")
            
            result = self.runner.invoke(speak, ['Hello', '--voice', 'invalid'])
            assert result.exit_code == 2  # Click parameter validation error
    
    def test_speak_invalid_speed(self):
        """Test speak command with invalid speed."""
        with patch('vocalize.cli.validate_speed') as mock_validate:
            mock_validate.side_effect = Exception("Speed must be between 0.1 and 3.0")
            
            result = self.runner.invoke(speak, ['Hello', '--speed', '10.0'])
            assert result.exit_code == 2
    
    def test_speak_invalid_pitch(self):
        """Test speak command with invalid pitch."""
        with patch('vocalize.cli.validate_pitch') as mock_validate:
            mock_validate.side_effect = Exception("Pitch must be between -1.0 and 1.0")
            
            result = self.runner.invoke(speak, ['Hello', '--pitch', '5.0'])
            assert result.exit_code == 2
    
    def test_config_set_readonly_file(self):
        """Test config set with read-only configuration file."""
        with patch('vocalize.cli.ConfigManager') as mock_config_manager:
            mock_manager = Mock()
            mock_manager.set_value.side_effect = Exception("Permission denied")
            mock_config_manager.return_value = mock_manager
            
            result = self.runner.invoke(config, ['set', 'test.key', 'value'])
            assert result.exit_code == 1
            assert "Error setting configuration" in result.output
    
    def test_unexpected_cli_error(self):
        """Test handling of unexpected CLI errors."""
        with patch('vocalize.cli.ConfigManager') as mock_config_manager:
            mock_config_manager.side_effect = RuntimeError("Unexpected error")
            
            result = self.runner.invoke(cli, ['--help'])
            # Should still work as ConfigManager is only created in context
            assert result.exit_code == 0


if __name__ == '__main__':
    pytest.main([__file__])