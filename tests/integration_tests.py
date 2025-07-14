#!/usr/bin/env python3
"""
Integration tests for the complete Vocalize TTS workflow.

These tests verify that the entire pipeline works correctly from text input
to audio output, including Python CLI, Rust bindings, and fallback implementations.
"""

import os
import sys
import tempfile
import subprocess
import unittest
from pathlib import Path

# Add the vocalize package to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "vocalize"))

try:
    import vocalize
    _HAS_VOCALIZE = True
except ImportError:
    _HAS_VOCALIZE = False


class IntegrationTestCase(unittest.TestCase):
    """Base class for integration tests with common setup."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_text = "Hello, this is a test of the TTS system."
        self.short_text = "Test"
        self.long_text = "This is a longer text that will be used to test the TTS system's ability to handle more complex and lengthy input strings."
        
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


@unittest.skipUnless(_HAS_VOCALIZE, "Vocalize package not available")
class VocalizePackageTests(IntegrationTestCase):
    """Test the Python package functionality."""
    
    def test_package_imports(self):
        """Test that all expected classes can be imported."""
        self.assertTrue(hasattr(vocalize, 'TtsEngine'))
        self.assertTrue(hasattr(vocalize, 'SynthesisParams'))
        self.assertTrue(hasattr(vocalize, 'Voice'))
        self.assertTrue(hasattr(vocalize, 'VoiceManager'))
        self.assertTrue(hasattr(vocalize, 'AudioWriter'))
        self.assertTrue(hasattr(vocalize, 'AudioDevice'))
        self.assertTrue(hasattr(vocalize, 'VocalizeError'))
        self.assertTrue(hasattr(vocalize, 'Gender'))
        self.assertTrue(hasattr(vocalize, 'VoiceStyle'))
    
    def test_constants(self):
        """Test that package constants are available."""
        self.assertTrue(hasattr(vocalize, 'DEFAULT_SAMPLE_RATE'))
        self.assertTrue(hasattr(vocalize, 'DEFAULT_CHANNELS'))
        self.assertTrue(hasattr(vocalize, 'MAX_TEXT_LENGTH'))
        
        # Check reasonable values
        self.assertGreater(vocalize.DEFAULT_SAMPLE_RATE, 8000)
        self.assertLessEqual(vocalize.DEFAULT_SAMPLE_RATE, 48000)
        self.assertIn(vocalize.DEFAULT_CHANNELS, [1, 2])
        self.assertGreater(vocalize.MAX_TEXT_LENGTH, 1000)
    
    def test_rust_bindings_detection(self):
        """Test detection of Rust bindings availability."""
        self.assertTrue(hasattr(vocalize, '_HAS_RUST_BINDINGS'))
        self.assertIsInstance(vocalize._HAS_RUST_BINDINGS, bool)
    
    def test_voice_creation(self):
        """Test Voice class functionality."""
        # Test default voice
        voice = vocalize.Voice.default()
        self.assertIsInstance(voice.id(), str)
        self.assertIsInstance(voice.name(), str)
        self.assertIsInstance(voice.language(), str)
        self.assertIsInstance(voice.gender(), vocalize.Gender)
        
        # Validate voice attributes
        self.assertTrue(len(voice.id()) > 0)
        self.assertTrue(len(voice.name()) > 0)
        self.assertTrue(len(voice.language()) > 0)
    
    def test_voice_manager(self):
        """Test VoiceManager functionality."""
        voice_manager = vocalize.VoiceManager()
        voices = voice_manager.get_available_voices()
        
        self.assertIsInstance(voices, list)
        self.assertGreater(len(voices), 0)
        
        # Test each voice
        for voice in voices:
            self.assertIsInstance(voice, vocalize.Voice)
            self.assertTrue(len(voice.id()) > 0)
            self.assertTrue(len(voice.name()) > 0)
        
        # Test default voice
        default_voice = voice_manager.get_default_voice()
        self.assertIsInstance(default_voice, vocalize.Voice)
    
    def test_synthesis_params(self):
        """Test SynthesisParams functionality."""
        voice = vocalize.Voice.default()
        params = vocalize.SynthesisParams(voice)
        
        # Test parameter modification
        params_with_speed = params.with_speed(1.5)
        self.assertEqual(params_with_speed.speed(), 1.5)
        
        params_with_pitch = params.with_pitch(0.2)
        self.assertEqual(params_with_pitch.pitch(), 0.2)
        
        # Test parameter validation
        with self.assertRaises(vocalize.VocalizeError):
            params.with_speed(5.0)  # Too fast
        
        with self.assertRaises(vocalize.VocalizeError):
            params.with_pitch(2.0)  # Too high
    
    def test_tts_engine_creation(self):
        """Test TTS engine creation and basic functionality."""
        engine = vocalize.TtsEngine()
        self.assertIsInstance(engine, vocalize.TtsEngine)
        
        # Test engine readiness
        ready = engine.is_ready()
        self.assertIsInstance(ready, bool)
    
    def test_basic_synthesis(self):
        """Test basic text-to-speech synthesis."""
        engine = vocalize.TtsEngine()
        voice = vocalize.Voice.default()
        params = vocalize.SynthesisParams(voice)
        
        # Test synthesis
        audio_data = engine.synthesize_sync(self.test_text, params)
        self.assertIsInstance(audio_data, list)
        self.assertGreater(len(audio_data), 0)
        
        # Validate audio data
        for sample in audio_data[:10]:  # Check first 10 samples
            self.assertIsInstance(sample, float)
            self.assertGreaterEqual(sample, -1.0)
            self.assertLessEqual(sample, 1.0)
    
    def test_synthesis_with_parameters(self):
        """Test synthesis with different parameters."""
        engine = vocalize.TtsEngine()
        voice = vocalize.Voice.default()
        
        # Test different speeds
        for speed in [0.5, 1.0, 2.0]:
            params = vocalize.SynthesisParams(voice).with_speed(speed)
            audio_data = engine.synthesize_sync(self.test_text, params)
            self.assertIsInstance(audio_data, list)
            self.assertGreater(len(audio_data), 0)
        
        # Test different pitches
        for pitch in [-0.5, 0.0, 0.5]:
            params = vocalize.SynthesisParams(voice).with_pitch(pitch)
            audio_data = engine.synthesize_sync(self.test_text, params)
            self.assertIsInstance(audio_data, list)
            self.assertGreater(len(audio_data), 0)
    
    def test_multiple_voices(self):
        """Test synthesis with different voices."""
        engine = vocalize.TtsEngine()
        voice_manager = vocalize.VoiceManager()
        voices = voice_manager.get_available_voices()
        
        for voice in voices[:3]:  # Test first 3 voices
            params = vocalize.SynthesisParams(voice)
            audio_data = engine.synthesize_sync(self.short_text, params)
            self.assertIsInstance(audio_data, list)
            self.assertGreater(len(audio_data), 0)
    
    def test_text_length_limits(self):
        """Test synthesis with different text lengths."""
        engine = vocalize.TtsEngine()
        voice = vocalize.Voice.default()
        params = vocalize.SynthesisParams(voice)
        
        # Test empty text (should fail)
        with self.assertRaises(vocalize.VocalizeError):
            engine.synthesize_sync("", params)
        
        # Test short text
        audio_data = engine.synthesize_sync(self.short_text, params)
        self.assertGreater(len(audio_data), 0)
        
        # Test longer text
        audio_data = engine.synthesize_sync(self.long_text, params)
        self.assertGreater(len(audio_data), 0)


class CLIIntegrationTests(IntegrationTestCase):
    """Test the command-line interface."""
    
    def setUp(self):
        super().setUp()
        self.vocalize_cmd = [sys.executable, "-m", "vocalize.cli"]
        self.test_output_file = os.path.join(self.temp_dir, "test_output.wav")
    
    def run_cli_command(self, args, expected_returncode=0):
        """Run a CLI command and return the result."""
        cmd = self.vocalize_cmd + args
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        if result.returncode != expected_returncode:
            print(f"Command failed: {' '.join(cmd)}")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            
        return result
    
    def test_cli_help(self):
        """Test CLI help functionality."""
        result = self.run_cli_command(["--help"])
        self.assertEqual(result.returncode, 0)
        self.assertIn("vocalize", result.stdout.lower())
        self.assertIn("text-to-speech", result.stdout.lower())
    
    def test_cli_version(self):
        """Test CLI version display."""
        result = self.run_cli_command(["--version"])
        self.assertEqual(result.returncode, 0)
        self.assertIn("vocalize", result.stdout.lower())
    
    def test_list_voices_command(self):
        """Test the list-voices command."""
        result = self.run_cli_command(["list-voices"])
        self.assertEqual(result.returncode, 0)
        self.assertIn("Available voices", result.stdout)
        
        # Test JSON output
        result = self.run_cli_command(["list-voices", "--json"])
        self.assertEqual(result.returncode, 0)
        
        # Should be valid JSON
        import json
        try:
            data = json.loads(result.stdout)
            self.assertIsInstance(data, list)
            if data:  # If voices are available
                voice = data[0]
                self.assertIn("id", voice)
                self.assertIn("name", voice)
                self.assertIn("gender", voice)
                self.assertIn("language", voice)
        except json.JSONDecodeError:
            self.fail("list-voices --json output is not valid JSON")
    
    def test_speak_command_basic(self):
        """Test basic speak command."""
        result = self.run_cli_command([
            "speak", 
            self.test_text,
            "--output", self.test_output_file
        ])
        self.assertEqual(result.returncode, 0)
        self.assertTrue(os.path.exists(self.test_output_file))
        self.assertGreater(os.path.getsize(self.test_output_file), 0)
    
    def test_speak_command_with_voice(self):
        """Test speak command with specific voice."""
        result = self.run_cli_command([
            "speak",
            self.test_text,
            "--voice", "af_bella",
            "--output", self.test_output_file
        ])
        self.assertEqual(result.returncode, 0)
        self.assertTrue(os.path.exists(self.test_output_file))
    
    def test_speak_command_with_parameters(self):
        """Test speak command with speed and pitch parameters."""
        result = self.run_cli_command([
            "speak",
            self.test_text,
            "--speed", "1.5",
            "--pitch", "0.2",
            "--output", self.test_output_file
        ])
        self.assertEqual(result.returncode, 0)
        self.assertTrue(os.path.exists(self.test_output_file))
    
    def test_speak_command_invalid_parameters(self):
        """Test speak command with invalid parameters."""
        # Invalid speed
        result = self.run_cli_command([
            "speak",
            self.test_text,
            "--speed", "10.0",
            "--output", self.test_output_file
        ], expected_returncode=1)
        self.assertNotEqual(result.returncode, 0)
        
        # Invalid pitch
        result = self.run_cli_command([
            "speak",
            self.test_text,
            "--pitch", "5.0",
            "--output", self.test_output_file
        ], expected_returncode=1)
        self.assertNotEqual(result.returncode, 0)
    
    def test_config_commands(self):
        """Test configuration management commands."""
        # Test config list
        result = self.run_cli_command(["config", "list"])
        self.assertEqual(result.returncode, 0)
        self.assertIn("configuration keys", result.stdout.lower())
        
        # Test config get
        result = self.run_cli_command(["config", "get"])
        self.assertEqual(result.returncode, 0)
        
        # Test config set and get
        result = self.run_cli_command(["config", "set", "default.speed", "1.5"])
        self.assertEqual(result.returncode, 0)
        
        result = self.run_cli_command(["config", "get", "default.speed"])
        self.assertEqual(result.returncode, 0)
        self.assertIn("1.5", result.stdout)


class FallbackModeTests(IntegrationTestCase):
    """Test fallback mode when Rust bindings are not available."""
    
    def test_fallback_synthesis(self):
        """Test synthesis using fallback implementation."""
        from vocalize.cli import VocalizeComponents
        
        # Test basic synthesis
        audio_data = VocalizeComponents.synthesize_text(self.test_text)
        self.assertIsInstance(audio_data.samples, list)
        self.assertGreater(len(audio_data.samples), 0)
        
        # Validate audio samples
        for sample in audio_data.samples[:10]:
            self.assertIsInstance(sample, float)
            self.assertGreaterEqual(sample, -1.0)
            self.assertLessEqual(sample, 1.0)
    
    def test_fallback_voice_list(self):
        """Test voice listing in fallback mode."""
        from vocalize.cli import VocalizeComponents
        
        voices = VocalizeComponents.list_voices()
        self.assertIsInstance(voices, list)
        self.assertGreater(len(voices), 0)
        
        for voice in voices:
            self.assertTrue(hasattr(voice, 'id'))
            self.assertTrue(hasattr(voice, 'name'))
            self.assertTrue(hasattr(voice, 'gender'))
            self.assertTrue(hasattr(voice, 'language'))
    
    def test_fallback_with_parameters(self):
        """Test fallback synthesis with different parameters."""
        from vocalize.cli import VocalizeComponents
        
        # Test different voices
        for voice_id in ["af_bella", "af_sarah", "af_josh", "af_adam"]:
            audio_data = VocalizeComponents.synthesize_text(
                self.short_text, 
                voice=voice_id
            )
            self.assertGreater(len(audio_data.samples), 0)
        
        # Test different speeds
        for speed in [0.5, 1.0, 2.0]:
            audio_data = VocalizeComponents.synthesize_text(
                self.short_text,
                speed=speed
            )
            self.assertGreater(len(audio_data.samples), 0)
        
        # Test different pitches
        for pitch in [-0.5, 0.0, 0.5]:
            audio_data = VocalizeComponents.synthesize_text(
                self.short_text,
                pitch=pitch
            )
            self.assertGreater(len(audio_data.samples), 0)
    
    def test_fallback_audio_saving(self):
        """Test audio saving in fallback mode."""
        from vocalize.cli import VocalizeComponents
        
        audio_data = VocalizeComponents.synthesize_text(self.test_text)
        output_file = os.path.join(self.temp_dir, "fallback_test.wav")
        
        # Test saving
        VocalizeComponents.save_audio(audio_data, output_file)
        self.assertTrue(os.path.exists(output_file))
        self.assertGreater(os.path.getsize(output_file), 0)


class AudioProcessingTests(IntegrationTestCase):
    """Test audio processing and quality."""
    
    def test_audio_data_properties(self):
        """Test properties of generated audio data."""
        from vocalize.cli import VocalizeComponents
        
        # Generate audio
        audio_data = VocalizeComponents.synthesize_text(self.test_text)
        samples = audio_data.samples
        
        # Test basic properties
        self.assertIsInstance(samples, list)
        self.assertGreater(len(samples), 0)
        
        # Test sample values
        for sample in samples:
            self.assertIsInstance(sample, float)
            self.assertFalse(math.isnan(sample), "Audio sample is NaN")
            self.assertFalse(math.isinf(sample), "Audio sample is infinite")
            self.assertGreaterEqual(sample, -1.0)
            self.assertLessEqual(sample, 1.0)
        
        # Test for silence (all zeros)
        non_zero_samples = [s for s in samples if abs(s) > 0.001]
        self.assertGreater(len(non_zero_samples), len(samples) * 0.1, 
                          "Audio appears to be mostly silent")
    
    def test_audio_length_scaling(self):
        """Test that audio length scales with text length."""
        from vocalize.cli import VocalizeComponents
        
        short_audio = VocalizeComponents.synthesize_text("Hi")
        long_audio = VocalizeComponents.synthesize_text(self.long_text)
        
        # Longer text should produce longer audio
        self.assertGreater(
            len(long_audio.samples), 
            len(short_audio.samples),
            "Longer text should produce longer audio"
        )
    
    def test_speed_affects_duration(self):
        """Test that speed parameter affects audio duration."""
        from vocalize.cli import VocalizeComponents
        
        normal_audio = VocalizeComponents.synthesize_text(self.test_text, speed=1.0)
        fast_audio = VocalizeComponents.synthesize_text(self.test_text, speed=2.0)
        slow_audio = VocalizeComponents.synthesize_text(self.test_text, speed=0.5)
        
        # Fast speech should be shorter
        self.assertLess(
            len(fast_audio.samples),
            len(normal_audio.samples),
            "Fast speech should be shorter"
        )
        
        # Slow speech should be longer
        self.assertGreater(
            len(slow_audio.samples),
            len(normal_audio.samples),
            "Slow speech should be longer"
        )


if __name__ == "__main__":
    import math
    
    # Discover and run all tests
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=".", pattern="integration_tests.py")
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)