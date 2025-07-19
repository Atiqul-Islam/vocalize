"""
Vocalize - High-performance text-to-speech synthesis library

This package provides a text-to-speech CLI and Python API.
"""

# CRITICAL: Import environment setup BEFORE anything else
# This prevents ONNX Runtime deadlocks by setting thread limits early
from . import _env_setup

from ._version import __version__

__author__ = "Vocalize Contributors"
__email__ = "noreply@vocalize.ai"
__description__ = "High-performance text-to-speech synthesis library"

# Try to import Rust bindings, fall back to Python-only implementation
try:
    import vocalize_python
    _HAS_RUST_BINDINGS = True
    
    # Export main classes from Rust bindings
    from vocalize_python import (
        TtsEngine, SynthesisParams, Voice, VoiceManager, AudioWriter, AudioDevice,
        VocalizeError, Gender, VoiceStyle
    )
    
except ImportError:
    _HAS_RUST_BINDINGS = False
    
    # If Rust bindings not available, create mock classes that delegate to CLI components
    from .cli import VocalizeComponents
    
    class VocalizeError(Exception):
        """Mock VocalizeError for when Rust bindings are not available."""
        pass
    
    class Voice:
        """Mock Voice class that delegates to CLI components."""
        def __init__(self, id: str, name: str, language: str, gender: str, style: str):
            self.id = id
            self.name = name
            self.language = language
            self.gender = gender
            self.style = style
        
        @staticmethod
        def default():
            return Voice("af_bella", "Bella", "en-US", "female", "natural")
    
    class SynthesisParams:
        """Mock SynthesisParams class."""
        def __init__(self, voice: Voice):
            self.voice = voice
            self.speed = None
            self.pitch = None
            self.streaming_chunk_size = None
        
        def with_speed(self, speed: float):
            if not (0.1 <= speed <= 3.0):
                raise VocalizeError(f"Speed must be between 0.1 and 3.0, got {speed}")
            new_params = SynthesisParams(self.voice)
            new_params.speed = speed
            new_params.pitch = self.pitch
            new_params.streaming_chunk_size = self.streaming_chunk_size
            return new_params
        
        def with_pitch(self, pitch: float):
            if not (-1.0 <= pitch <= 1.0):
                raise VocalizeError(f"Pitch must be between -1.0 and 1.0, got {pitch}")
            new_params = SynthesisParams(self.voice)
            new_params.speed = self.speed
            new_params.pitch = pitch
            new_params.streaming_chunk_size = self.streaming_chunk_size
            return new_params
        
        def with_streaming(self, chunk_size: int):
            new_params = SynthesisParams(self.voice)
            new_params.speed = self.speed
            new_params.pitch = self.pitch
            new_params.streaming_chunk_size = chunk_size
            return new_params
        
        def without_streaming(self):
            new_params = SynthesisParams(self.voice)
            new_params.speed = self.speed
            new_params.pitch = self.pitch
            new_params.streaming_chunk_size = None
            return new_params
    
    class TtsEngine:
        """Mock TtsEngine class."""
        def __init__(self):
            pass
        
        def __repr__(self):
            return "TtsEngine()"
        
        async def synthesize(self, text: str, params: SynthesisParams):
            # Use CLI components for synthesis
            speed = params.speed or 1.0
            pitch = params.pitch or 0.0
            audio_data = VocalizeComponents.synthesize_text(text, params.voice.id, speed, pitch)
            return audio_data.samples
        
        async def is_ready(self):
            return True
    
    class VoiceManager:
        """Mock VoiceManager class."""
        def __init__(self):
            pass
        
        def get_available_voices(self):
            voices = VocalizeComponents.list_voices()
            return [Voice(v.id, v.name, v.language, v.gender, v.style) for v in voices]
        
        def get_default_voice(self):
            return Voice.default()
    
    class AudioWriter:
        """Mock AudioWriter class."""
        def __init__(self):
            pass
    
    class AudioDevice:
        """Mock AudioDevice class."""
        def __init__(self):
            pass
    
    class Gender:
        MALE = "male"
        FEMALE = "female"
        NEUTRAL = "neutral"
    
    class VoiceStyle:
        NATURAL = "natural"
        PROFESSIONAL = "professional"
        EXPRESSIVE = "expressive"
        CALM = "calm"
        ENERGETIC = "energetic"

# Constants
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_CHANNELS = 1
MAX_TEXT_LENGTH = 100000

__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    "__description__",
    "_HAS_RUST_BINDINGS",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_CHANNELS", 
    "MAX_TEXT_LENGTH",
    # Core classes
    "TtsEngine",
    "SynthesisParams", 
    "Voice",
    "VoiceManager",
    "AudioWriter",
    "AudioDevice",
    "VocalizeError",
    "Gender",
    "VoiceStyle",
]