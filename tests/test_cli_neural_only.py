"""TDD tests to ensure Python CLI uses only neural synthesis."""

import pytest
import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to sys.path to import vocalize
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vocalize.cli import VocalizeComponents


class TestNeuralOnlyCLI(unittest.TestCase):
    """Test that CLI uses only neural synthesis, no mathematical fallbacks."""
    
    def test_no_mathematical_synthesis_fallback(self):
        """Test that VocalizeComponents does not use mathematical synthesis."""
        # Mock the Rust bindings to be unavailable to trigger fallback
        with patch.dict('sys.modules', {'vocalize.vocalize_python': None}):
            # This should fail if mathematical fallback is used
            with self.assertRaises(Exception):
                # Should not fall back to mathematical synthesis
                VocalizeComponents.synthesize_text("Hello world", "af_bella", 1.0, 0.0)
    
    def test_only_rust_backend_synthesis(self):
        """Test that synthesis only works with Rust backend."""
        # Mock successful Rust synthesis
        mock_rust_module = MagicMock()
        mock_rust_module.synthesize_text.return_value = [0.1, 0.2, -0.1, 0.0]
        
        with patch.dict('sys.modules', {'vocalize.vocalize_python': mock_rust_module}):
            audio = VocalizeComponents.synthesize_text("Neural test", "af_bella", 1.0, 0.0)
            
            # Should use Rust backend
            mock_rust_module.synthesize_text.assert_called_once()
            self.assertIsInstance(audio, VocalizeComponents.AudioData)
            self.assertEqual(audio.samples, [0.1, 0.2, -0.1, 0.0])
    
    def test_no_fallback_synthesis_methods(self):
        """Test that fallback synthesis methods are removed."""
        # These methods should not exist in neural-only version
        self.assertFalse(hasattr(VocalizeComponents, '_synthesize_text_fallback'))
        self.assertFalse(hasattr(VocalizeComponents, '_get_formants'))
        self.assertFalse(hasattr(VocalizeComponents, '_generate_formant'))
        self.assertFalse(hasattr(VocalizeComponents, '_calculate_char_envelope'))
    
    def test_voices_use_neural_models(self):
        """Test that voices are mapped to neural models."""
        voices = VocalizeComponents.list_voices()
        
        # Should have voices that correspond to neural models
        voice_ids = [v.id for v in voices]
        self.assertIn("kokoro_en_us_f", voice_ids)  # Neural voice
        
        # Should not have fallback/mathematical voices
        for voice in voices:
            self.assertIn("neural", voice.name.lower() or voice.style.lower())
    
    def test_cli_requires_rust_bindings(self):
        """Test that CLI gracefully handles missing Rust bindings."""
        # Without Rust bindings, should show informative error
        with patch.dict('sys.modules', {'vocalize.vocalize_python': None}):
            with patch('builtins.print') as mock_print:
                try:
                    VocalizeComponents.synthesize_text("test", "neural_voice", 1.0, 0.0)
                except Exception:
                    pass
                
                # Should inform user about neural requirements
                printed_messages = [call.args[0] for call in mock_print.call_args_list]
                neural_message_found = any(
                    "neural" in msg.lower() or "rust" in msg.lower() 
                    for msg in printed_messages if isinstance(msg, str)
                )
                self.assertTrue(neural_message_found, "Should inform user about neural TTS requirements")


if __name__ == '__main__':
    unittest.main()