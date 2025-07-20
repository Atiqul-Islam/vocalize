"""
Command-line interface for Vocalize TTS library.

This module provides a comprehensive CLI for text-to-speech synthesis,
voice management, audio playback, and model management.

Example usage:
    vocalize speak "Hello, world!" --voice bella --play
    vocalize list-voices --gender female
    vocalize models list
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import time
import platformdirs

# Check if verbose mode is requested early
_verbose = "--verbose" in sys.argv

# Import our reliable Python model manager
if _verbose:
    _import_start = time.perf_counter()
from .model_manager import ModelManager, ensure_model_available
if _verbose:
    print(f"  ⏱️  Import ModelManager: {time.perf_counter() - _import_start:.3f}s")

try:
    import sounddevice as sd
    _HAS_AUDIO = True
except (ImportError, OSError) as e:
    _HAS_AUDIO = False
    if "PortAudio" in str(e):
        print("Warning: PortAudio not available. Install with: sudo apt-get install portaudio19-dev")
    else:
        print("Warning: sounddevice not available. Install with: uv add sounddevice")

# Default values (no configuration needed)
DEFAULT_VOICE = "af_alloy"
DEFAULT_SPEED = 1.0
DEFAULT_PITCH = 0.0
DEFAULT_FORMAT = "wav"


class Timer:
    """Context manager for timing operations."""
    def __init__(self, name, verbose=False):
        self.name = name
        self.verbose = verbose
        self.start = None
        
    def __enter__(self):
        if self.verbose:
            self.start = time.perf_counter()
        return self
        
    def __exit__(self, *args):
        if self.verbose and self.start is not None:
            elapsed = time.perf_counter() - self.start
            print(f"  ⏱️  {self.name}: {elapsed:.3f}s")




class VocalizeComponents:
    """Real implementations for Vocalize TTS components using Rust backend."""
    
    class Voice:
        def __init__(self, id: str, name: str, gender: str = "unknown", 
                     language: str = "en", style: str = "neutral"):
            self.id = id
            self.name = name
            self.gender = gender
            self.language = language
            self.style = style
    
    class AudioData:
        def __init__(self, samples: List[float]):
            self.samples = samples
    
    @staticmethod
    def synthesize_text(text: str, voice: str = "kokoro_en_us_f", speed: float = 1.0, 
                       pitch: float = 0.0) -> 'VocalizeComponents.AudioData':
        """Neural speech synthesis using Rust ONNX TTS engine with Python model management."""
        if not text.strip():
            return VocalizeComponents.AudioData([])
        
        try:
            print(f"🎙️  Starting neural synthesis - text: '{text}', voice: {voice}")
            
            # Map voice to model ID for reliable Python downloads
            voice_to_model = {
                "kokoro_en_us_f": "kokoro",
                "kokoro_en_us_m": "kokoro", 
                "chatterbox_en_f": "chatterbox", 
                "dia_en_premium": "dia",
            }
            
            model_id = voice_to_model.get(voice, "kokoro")  # Default to kokoro
            print(f"📦 Model required: {model_id}")
            
            # CRITICAL: Ensure model is downloaded using reliable Python client
            print(f"🔍 Checking if model '{model_id}' is available...")
            if not ensure_model_available(model_id):
                raise RuntimeError(f"Failed to download required model: {model_id}")
            
            print(f"✅ Model '{model_id}' is ready")
            
            # Import the Rust neural TTS bindings
            from . import vocalize_rust
            print("DEBUG: Successfully imported vocalize_rust")
            
            # Use neural ONNX TTS engine for synthesis (Rust loads from Python-managed cache)
            print("DEBUG: Calling vocalize_rust.synthesize_neural()...")
            samples = vocalize_rust.synthesize_neural(text, voice, speed, pitch)
            print(f"✅ Got {len(samples)} audio samples from neural synthesis")
            
            return VocalizeComponents.AudioData(samples)
            
        except (ImportError, ModuleNotFoundError) as e:
            print(f"❌ Error: Neural TTS engine not available ({e}).")
            print("This version requires the neural TTS engine. Please install with: maturin develop")
            print("Neural TTS provides superior quality compared to mathematical synthesis.")
            raise RuntimeError("Neural TTS engine required - no fallback synthesis available") from e
    
    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available neural TTS models."""
        return [
            "kokoro_en_us_f",    # Kokoro TTS female English
            "kokoro_en_us_m",    # Kokoro TTS male English  
            "chatterbox_en_f",   # Chatterbox English female
            "dia_en_premium",    # Dia premium English
        ]
    
    @staticmethod
    def list_voices() -> List['VocalizeComponents.Voice']:
        """List available neural TTS voices."""
        return [
            VocalizeComponents.Voice("kokoro_en_us_f", "Kokoro Female", "female", "en-US", "neural_natural"),
            VocalizeComponents.Voice("kokoro_en_us_m", "Kokoro Male", "male", "en-US", "neural_natural"),
            VocalizeComponents.Voice("chatterbox_en_f", "Chatterbox English", "female", "en-US", "neural_fast"),
            VocalizeComponents.Voice("dia_en_premium", "Dia Premium", "female", "en-US", "neural_premium"),
        ]
    
    @staticmethod
    def save_audio(audio_data: 'VocalizeComponents.AudioData', 
                   output_path: str, format: str = "wav"):
        """Save audio to file using Rust backend."""
        try:
            # Use Rust backend for high-quality audio writing
            from . import vocalize_rust
            vocalize_rust.save_audio_neural(audio_data.samples, output_path, format)
            print(f"Saved neural TTS audio to {output_path} in {format} format")
        except (ImportError, ModuleNotFoundError) as e:
            print(f"Error: Neural audio writer not available ({e}).")
            print("This version requires the neural TTS engine for audio writing.")
            raise RuntimeError("Neural audio writer required - no fallback available") from e
    
    
    @staticmethod
    def play_audio(audio_data: 'VocalizeComponents.AudioData'):
        """Real audio playback through computer speakers."""
        if not _HAS_AUDIO:
            print("Error: sounddevice not available. Install with: uv add sounddevice")
            return
        
        if not audio_data.samples:
            print("Warning: No audio data to play")
            return
        
        try:
            # Convert to numpy array and play
            import numpy as np
            audio_array = np.array(audio_data.samples, dtype=np.float32)
            sample_rate = 24000  # Match the synthesis rate
            
            print(f"Playing {len(audio_array)} samples at {sample_rate}Hz...")
            sd.play(audio_array, samplerate=sample_rate)
            sd.wait()  # Wait until playback is finished
            print("Playback completed.")
            
        except Exception as e:
            print(f"Error playing audio: {e}")


def synthesize_with_tokens(text: str, voice: str, speed: float, pitch: float, model: str) -> 'VocalizeComponents.AudioData':
    """Synthesize using token-based approach for better compatibility."""
    verbose = _verbose  # Use global verbose flag
    try:
        print(f"🎙️  Starting phoneme-based synthesis - text: '{text}', voice: {voice}")
        
        # Use the phoneme processor to convert text to tokens
        with Timer("Import KokoroPhonemeProcessor", verbose):
            from .model_manager import KokoroPhonemeProcessor
        from pathlib import Path
        
        if model == "kokoro":
            # Use cross-platform cache directory that matches Rust implementation
            cache_base = platformdirs.user_cache_dir("vocalize", "Vocalize")
            cache_dir = Path(cache_base) / "models" / "models--direct_download" / "local"
            processor = KokoroPhonemeProcessor(cache_dir)
            
            # Process text to tokens with proper speed
            result = processor.process_text(text, voice)
            result['speed'] = speed  # Override speed
            
            print(f"📝 Generated {len(result['input_ids'])} tokens for synthesis")
            
            # Import the Rust neural TTS bindings
            from . import vocalize_rust
            print("DEBUG: Successfully imported vocalize_rust for token synthesis")
            
            # Use token-based neural synthesis
            print("DEBUG: Calling vocalize_rust.synthesize_from_tokens_neural()...")
            print(f"DEBUG: input_ids length: {len(result['input_ids'])}")
            print(f"DEBUG: style vector length: {len(result['style'])}")
            print(f"DEBUG: style vector range: [{min(result['style']):.3f}, {max(result['style']):.3f}]")
            print(f"DEBUG: speed: {result['speed']}")
            
            samples = vocalize_rust.synthesize_from_tokens_neural(
                result['input_ids'],
                result['style'],
                result['speed'],
                model
            )
            print(f"✅ Got {len(samples)} audio samples from token synthesis")
            
            return VocalizeComponents.AudioData(samples)
        else:
            # Fall back to the original synthesis for non-Kokoro models
            return VocalizeComponents.synthesize_text(text, voice, speed, pitch)
            
    except Exception as e:
        print(f"❌ Token synthesis failed: {e}")
        # Fall back to original synthesis
        return VocalizeComponents.synthesize_text(text, voice, speed, pitch)


def handle_speak_command(args):
    """Handle the 'speak' command with model and voice selection."""
    start_time = time.perf_counter()
    verbose = args.verbose
    text = args.text
    model = args.model or "kokoro"
    voice = args.voice
    speed = args.speed or DEFAULT_SPEED
    pitch = args.pitch or DEFAULT_PITCH
    output = args.output
    play = args.play
    format = args.format or DEFAULT_FORMAT
    
    # Import VoiceManager for voice selection
    with Timer("Import VoiceManager", verbose):
        from .voice_manager import VoiceManager
    
    with Timer("Import ModelManager", verbose):
        from .model_manager import ModelManager
    
    # Initialize managers
    with Timer("Initialize ModelManager", verbose):
        manager = ModelManager()
    
    with Timer("Initialize VoiceManager", verbose):
        voice_manager = VoiceManager(str(manager.cache_dir))
    
    # Ensure model is available
    with Timer("ensure_model_available", verbose):
        if not ensure_model_available(model):
            print(f"❌ Failed to download model: {model}")
            return
    
    # Get voice from user input or use Python default
    if not voice:
        voice = DEFAULT_VOICE  # Use Python default af_alloy
        if verbose:
            print(f"No voice specified, using default: {voice}")
    else:
        if verbose:
            print(f"Using specified voice: {voice}")
    
    print(f"🎙️  Synthesizing text: '{text}'")
    print(f"📦 Model: {model}, 🎵 Voice: {voice}, ⚡ Speed: {speed}, 🎛️  Pitch: {pitch}")
    
    # Use token-based synthesis for better compatibility
    with Timer("Speech synthesis", verbose):
        audio_data = synthesize_with_tokens(text, voice, speed, pitch, model)
    
    # Save to file if requested
    if output:
        with Timer("Save audio file", verbose):
            VocalizeComponents.save_audio(audio_data, output, format)
        print(f"💾 Audio saved to: {output}")
    
    # Play audio if requested
    if play:
        with Timer("Play audio", verbose):
            VocalizeComponents.play_audio(audio_data)
    
    if not output and not play:
        print("Note: Use --output to save audio or --play to hear it")
    
    # Show timing if verbose
    if verbose:
        elapsed = time.perf_counter() - start_time
        print(f"\n⏱️  Total execution time: {elapsed:.3f}s")


def handle_list_voices_command(args):
    """Handle the 'list-voices' command with model-specific voice discovery."""
    start_time = time.perf_counter()
    verbose = args.verbose
    model = args.model or "kokoro"
    
    # Import VoiceManager for voice discovery
    with Timer("Import VoiceManager", verbose):
        from .voice_manager import VoiceManager
    
    with Timer("Import ModelManager", verbose):
        from .model_manager import ModelManager
    
    # Initialize managers
    with Timer("Initialize ModelManager", verbose):
        manager = ModelManager()
    
    with Timer("Initialize VoiceManager", verbose):
        voice_manager = VoiceManager(str(manager.cache_dir))
    
    # Fast path: Try to discover voices from cache first
    with Timer("discover_voices_fast_path", verbose):
        voices = voice_manager.discover_voices(model)
    
    # If no voices found, ensure model is available and try again
    if not voices:
        with Timer("ensure_model_available", verbose):
            if not ensure_model_available(model):
                print(f"❌ Failed to download model: {model}")
                return
        
        # For Kokoro, voices are already included in the main model download
        if model == "kokoro":
            print(f"📦 Kokoro voices are included in the main model")
        else:
            # Download voices if needed for other models
            print(f"📦 Downloading voices for model: {model}")
            if not manager.download_model_with_voices(model):
                print(f"❌ Failed to download voices for model: {model}")
                return
        
        # Try discovering voices again after ensuring model
        with Timer("discover_voices_retry", verbose):
            voices = voice_manager.discover_voices(model)
    
    if not voices:
        print(f"❌ No voices found for model: {model}")
        return
    
    # Filter by criteria
    if hasattr(args, 'gender') and args.gender:
        voices = [v for v in voices if v.gender.lower() == args.gender.lower()]
    
    if hasattr(args, 'language') and args.language:
        voices = [v for v in voices if args.language.lower() in v.language.lower()]
    
    # Output format
    if hasattr(args, 'json') and args.json:
        voice_list = []
        for voice in voices:
            voice_list.append({
                "id": voice.id,
                "name": voice.name,
                "gender": voice.gender,
                "language": voice.language,
                "file_path": voice.file_path
            })
        print(json.dumps(voice_list, indent=2))
    else:
        print(f"🎵 Available voices for {model} ({len(voices)}):")
        print("=" * 60)
        for voice in voices:
            print(f"  {voice.id:<16} | {voice.name:<20} | {voice.gender:<8} | {voice.language}")
        print(f"\nUsage: vocalize speak \"Hello world\" --model {model} --voice <voice_id>")
    
    # Show timing if verbose
    if verbose:
        elapsed = time.perf_counter() - start_time
        print(f"\n⏱️  Total execution time: {elapsed:.3f}s")


def handle_play_command(args):
    """Handle the 'play' command."""
    input_file = args.input
    
    if not Path(input_file).exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)
    
    print(f"Playing audio file: {input_file}")
    # For now, just show that we would play the file
    # In a full implementation, we'd load the audio file and play it
    print("Note: Audio file playback not yet implemented")


def handle_models_command(args):
    """Handle the 'models' command for reliable Python-based model management."""
    manager = ModelManager()
    
    if not args.models_action:
        print("Error: No model action specified. Use 'models --help' for options.")
        return
    
    if args.models_action == "list":
        print("Neural TTS Models:")
        print("=" * 60)
        for model_id in manager.list_available_models():
            model_info = manager.get_model_info(model_id)
            status = "✅ cached" if manager.is_model_cached(model_id) else "⬜ not cached"
            print(f"  {model_id:<12} - {model_info.name} ({model_info.size_mb}MB) {status}")
            print(f"                Repository: {model_info.repo_id}")
            print(f"                Files: {', '.join(model_info.files)}")
            print()
        
        print(f"Cache size: {manager.get_cache_size()}")
        print(f"Cache location: {manager.cache_dir}")
    
    elif args.models_action == "download":
        model_id = args.model_id
        force = getattr(args, 'force', False)
        
        print(f"📥 Downloading model: {model_id}")
        if manager.download_model(model_id, force=force):
            print(f"✅ Successfully downloaded {model_id}")
        else:
            print(f"❌ Failed to download {model_id}")
            sys.exit(1)
    
    elif args.models_action == "clear":
        model_id = getattr(args, 'model_id', None)
        
        if model_id:
            print(f"🗑️  Clearing cache for model: {model_id}")
        else:
            print("🗑️  Clearing all model cache")
        
        if manager.clear_cache(model_id):
            print("✅ Cache cleared successfully")
        else:
            print("❌ Failed to clear cache")
            sys.exit(1)
    
    elif args.models_action == "status":
        model_id = args.model_id
        
        if model_id not in manager.list_available_models():
            print(f"❌ Unknown model: {model_id}")
            print(f"Available models: {', '.join(manager.list_available_models())}")
            sys.exit(1)
        
        model_info = manager.get_model_info(model_id)
        cached = manager.is_model_cached(model_id)
        
        print(f"Model: {model_info.name}")
        print(f"ID: {model_id}")
        print(f"Repository: {model_info.repo_id}")
        print(f"Size: {model_info.size_mb}MB")
        print(f"Files: {', '.join(model_info.files)}")
        print(f"Status: {'✅ Cached' if cached else '⬜ Not cached'}")
        
        if cached:
            for filename in model_info.files:
                path = manager.get_model_path(model_id, filename)
                if path:
                    file_size = path.stat().st_size
                    print(f"  📄 {filename}: {file_size:,} bytes at {path}")
    else:
        print(f"Unknown models action: {args.models_action}")



def create_parser():
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="vocalize",
        description="High-performance text-to-speech synthesis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  vocalize speak "Hello, world!" --voice bella --play
  vocalize speak "Save this to file" --output hello.wav
  vocalize list-voices --gender female --json
  vocalize play audio.wav
  vocalize models list
  vocalize models download kokoro
        """.strip()
    )
    
    parser.add_argument(
        "--version", 
        action="version", 
        version="vocalize 0.1.0"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed timing information"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Speak command
    speak_parser = subparsers.add_parser("speak", help="Synthesize text to speech")
    speak_parser.add_argument("text", help="Text to synthesize")
    speak_parser.add_argument("--model", "-m", default="kokoro", help="TTS model to use (default: kokoro)")
    speak_parser.add_argument("--voice", "-v", help="Voice to use for synthesis (e.g., af_alloy, am_adam)")
    speak_parser.add_argument("--speed", "-s", type=float, help="Speech speed (0.1-3.0)")
    speak_parser.add_argument("--pitch", "-p", type=float, help="Pitch adjustment (-1.0 to 1.0)")
    speak_parser.add_argument("--output", "-o", help="Output file path")
    speak_parser.add_argument("--format", "-f", choices=["wav", "mp3", "flac", "ogg"], 
                            help="Output format")
    speak_parser.add_argument("--play", action="store_true", 
                            help="Play audio through speakers")
    
    # List voices command
    list_voices_parser = subparsers.add_parser("list-voices", help="List available voices")
    list_voices_parser.add_argument("--model", "-m", default="kokoro", help="TTS model to list voices for (default: kokoro)")
    list_voices_parser.add_argument("--gender", "-g", choices=["male", "female"], 
                                   help="Filter by gender")
    list_voices_parser.add_argument("--language", "-l", help="Filter by language code")
    list_voices_parser.add_argument("--style", help="Filter by voice style")
    list_voices_parser.add_argument("--json", action="store_true", 
                                   help="Output in JSON format")
    
    # Play command
    play_parser = subparsers.add_parser("play", help="Play audio file")
    play_parser.add_argument("input", help="Input audio file path")
    
    # Models command - new reliable Python-based model management
    models_parser = subparsers.add_parser("models", help="Manage neural TTS models")
    models_subparsers = models_parser.add_subparsers(dest="models_action",
                                                     help="Model management actions")
    
    # Models list
    models_subparsers.add_parser("list", help="List available and cached models")
    
    # Models download
    models_download_parser = models_subparsers.add_parser("download", help="Download a model")
    models_download_parser.add_argument("model_id", help="Model ID to download (e.g., kokoro)")
    models_download_parser.add_argument("--force", action="store_true", help="Force redownload")
    
    # Models clear
    models_clear_parser = models_subparsers.add_parser("clear", help="Clear model cache")
    models_clear_parser.add_argument("model_id", nargs="?", help="Specific model to clear (or all)")
    
    # Models status
    models_status_parser = models_subparsers.add_parser("status", help="Check model status")
    models_status_parser.add_argument("model_id", help="Model ID to check")
    
    return parser


def main():
    """Main CLI entry point."""
    # Fast path: Handle help and version without expensive imports
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', '--version']:
        parser = create_parser()
        args = parser.parse_args()
        return
    
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == "speak":
            handle_speak_command(args)
        elif args.command == "list-voices":
            handle_list_voices_command(args)
        elif args.command == "play":
            handle_play_command(args)
        elif args.command == "models":
            handle_models_command(args)
        else:
            parser.print_help()
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()