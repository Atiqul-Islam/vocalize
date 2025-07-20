# Vocalize Neural TTS

Production-ready neural text-to-speech synthesis using 2025 state-of-the-art models with Rust ONNX Runtime backend and Python CLI.

## 📜 License

Vocalize is dual-licensed:

- **Non-commercial use**: [PolyForm Noncommercial License 1.0.0](LICENSE)
  - ✅ Personal projects, research, education
  - ✅ Open source projects (non-commercial)
  - ✅ Non-profit organizations
  
- **Commercial use**: Requires a commercial license
  - 📋 Request via: [GitHub Issues](https://github.com/vocalize/vocalize/issues)
  - 📄 See [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md) for details

For questions about which license you need, please see our [commercial licensing guide](COMMERCIAL-LICENSE.md).

## 🧠 Neural TTS Architecture

- **Pure Neural Synthesis**: Uses Kokoro TTS and other 2025 neural models
- **ONNX Runtime**: Self-contained deployment with no external DLL dependencies  
- **Zero Fallbacks**: No mathematical synthesis - neural models only
- **Premium Models**: Support for multiple neural TTS models with different characteristics

## Quick Start

### Requirements
- Python 3.8+
- UV package manager
- Rust (for building from source)
- PortAudio (for audio playback only)

### Windows

```powershell
# Navigate to the project directory
cd C:\Users\iatiq\Documents\dev\vocalize

# List available neural voices
uv run python -m vocalize.cli list-voices

# Generate neural speech and save to file
uv run python -m vocalize.cli speak "Hello world!" --voice kokoro_en_us_f --output hello.wav

# Try different neural voices
uv run python -m vocalize.cli speak "Neural male voice" --voice kokoro_en_us_m --output male.wav
uv run python -m vocalize.cli speak "Premium quality" --voice dia_en_premium --output premium.wav

# Control speed and pitch
uv run python -m vocalize.cli speak "Fast neural speech" --speed 2.0 --output fast.wav
uv run python -m vocalize.cli speak "High pitch neural" --pitch 0.5 --output high.wav

# Use the Windows shortcuts (optional)
.\vocalize.bat speak "Using batch file with neural TTS" --output batch.wav
.\vocalize.ps1 speak "Using PowerShell with neural TTS" --output ps.wav
```

### Linux/macOS

```bash
# Navigate to the project directory
cd /path/to/vocalize

# List available neural voices
uv run python -m vocalize.cli list-voices

# Generate neural speech and save to file
uv run python -m vocalize.cli speak "Hello neural world!" --voice kokoro_en_us_f --output hello.wav

# For audio playback, install PortAudio first:
# Linux: sudo apt-get install portaudio19-dev
# macOS: brew install portaudio

# Then you can play neural audio directly:
uv run python -m vocalize.cli speak "Neural TTS playback!" --voice kokoro_en_us_f --play
```

## 🎙️ Available Neural Voices

- **kokoro_en_us_f** (female, neural_natural) - Kokoro TTS Female (82MB, fastest)
- **kokoro_en_us_m** (male, neural_natural) - Kokoro TTS Male (82MB, fastest)
- **chatterbox_en_f** (female, neural_fast) - Chatterbox English (150MB, balanced)
- **dia_en_premium** (female, neural_premium) - Dia Premium (1.6GB, highest quality)

## 📋 Commands

### speak
Generate neural speech from text:
```bash
uv run python -m vocalize.cli speak "Your text here" [options]

Options:
  --voice/-v VOICE       Neural voice to use (kokoro_en_us_f, kokoro_en_us_m, dia_en_premium, etc.)
  --speed/-s SPEED       Speech speed (0.1-3.0, default: 1.0)
  --pitch/-p PITCH       Pitch adjustment (-1.0 to 1.0, default: 0.0)
  --output/-o FILE       Output file path
  --format/-f FORMAT     Output format (wav, mp3, flac, ogg)
  --play                 Play neural audio through speakers
```

### list-voices
List available neural voices:
```bash
uv run python -m vocalize.cli list-voices [options]

Options:
  --gender/-g GENDER     Filter by gender (male, female)
  --language/-l LANG     Filter by language code
  --style STYLE          Filter by neural voice style
  --json                 Output in JSON format
```

### config
Manage configuration:
```bash
uv run python -m vocalize.cli config get [key]
uv run python -m vocalize.cli config set key value
uv run python -m vocalize.cli config list
```

## Architecture

- **Rust Backend**: High-performance TTS synthesis with neural models
- **Python Frontend**: User-friendly CLI with cross-platform compatibility
- **Self-contained**: Bundles all dependencies including ONNX Runtime
- **UV Managed**: Modern Python package management
- **Platform-specific builds**: Native wheels for Windows, Linux, and macOS

## Requirements

- Python 3.8+
- Rust (for building from source)
- UV package manager
- PortAudio (for audio playback only)
- **Windows**: WSL2 (Windows Subsystem for Linux) for building
- **macOS**: Xcode Command Line Tools (`xcode-select --install`)

## Development

For detailed development setup and build instructions, see [DEVELOPMENT.md](DEVELOPMENT.md).

### Quick Start for Developers

**Prerequisites:**
- Python 3.8+ with UV package manager
- Rust toolchain
- Platform-specific requirements:
  - **Windows**: WSL2 with 7-Zip
  - **Linux**: build-essential
  - **macOS**: Xcode Command Line Tools

#### Building on Windows (via WSL)
```bash
# In WSL terminal
cd /mnt/c/Users/[your-username]/Documents/dev/personal/vocalize
./build_windows.sh

# In Windows terminal
uv sync
uv pip install crates/target/wheels/vocalize_rust-0.1.0-cp38-abi3-win_amd64_bundled.whl --force-reinstall
uv run python -m vocalize
```

#### Building on Linux
```bash
# Build wheel with bundled dependencies
./build_linux.sh

# Install
uv sync
uv pip install crates/target/wheels/vocalize_rust-*_bundled.whl --force-reinstall --python-platform linux
uv run python -m vocalize
```

#### Building on macOS
```bash
# Build wheel with bundled dependencies
./build_macos.sh

# Install
uv sync
uv pip install crates/target/wheels/vocalize_rust-*_bundled.whl --force-reinstall
uv run python -m vocalize
```

For other platforms and detailed instructions, see [DEVELOPMENT.md](DEVELOPMENT.md).

## Examples

```bash
# Basic usage
uv run python -m vocalize.cli speak "Hello, world!" --output hello.wav

# Different voices and settings
uv run python -m vocalize.cli speak "Female voice" --voice af_bella --output female.wav
uv run python -m vocalize.cli speak "Male voice" --voice af_josh --output male.wav
uv run python -m vocalize.cli speak "Fast speech" --speed 2.0 --output fast.wav
uv run python -m vocalize.cli speak "Slow speech" --speed 0.5 --output slow.wav
uv run python -m vocalize.cli speak "High pitch" --pitch 0.5 --output high.wav
uv run python -m vocalize.cli speak "Low pitch" --pitch -0.5 --output low.wav

# Configuration
uv run python -m vocalize.cli config set default.voice af_josh
uv run python -m vocalize.cli config set default.speed 1.2
uv run python -m vocalize.cli config get
```

The generated WAV files are high-quality 24kHz mono audio that can be played with any audio player.