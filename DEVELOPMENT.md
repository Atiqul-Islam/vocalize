# Vocalize Development Guide

This comprehensive guide covers development setup, build processes, architecture details, and troubleshooting for the Vocalize TTS project.

## Overview

Vocalize is a neural text-to-speech system with:
- **Rust Core** (`vocalize-core`): High-performance TTS synthesis engine
- **Python Bindings** (`vocalize-python`): PyO3-based Python API
- **Cross-compilation**: Build on WSL for Windows deployment
- **Self-contained**: Bundles all dependencies including ONNX Runtime and VC++ redistributables

## Project Structure

```
vocalize/
├── crates/
│   ├── vocalize-core/          # Rust TTS engine
│   │   ├── src/
│   │   │   ├── lib.rs          # Core library
│   │   │   ├── onnx_engine.rs  # ONNX Runtime integration
│   │   │   └── model/          # Neural model implementations
│   │   └── Cargo.toml
│   └── vocalize-python/        # Python bindings
│       ├── src/
│       │   └── lib.rs          # PyO3 module with DLL loading
│       ├── build.rs            # Build script for ONNX Runtime
│       └── Cargo.toml
├── vocalize/                   # Python package
│   ├── __init__.py             # Package initialization
│   ├── _env_setup.py           # Environment setup
│   ├── cli.py                  # Command-line interface
│   └── model_manager.py        # Model management
├── build_and_bundle_complete.sh # WSL build script
└── pyproject.toml              # Python project configuration
```

## Prerequisites

### All Platforms
- Python 3.8+ 
- [UV package manager](https://github.com/astral-sh/uv)
- Rust toolchain (latest stable)

### Windows Development
- WSL2 (Windows Subsystem for Linux)
- 7-Zip installed in WSL (`sudo apt install p7zip-full`)
- Visual Studio 2022 or Build Tools (for MSVC)

### Linux
- Build essentials (`sudo apt install build-essential`)
- OpenSSL development headers (`sudo apt install libssl-dev`)

### macOS
- Xcode Command Line Tools (`xcode-select --install`)

## Environment Setup by Platform

### Linux (Ubuntu/Debian)

#### Prerequisites
```bash
# System dependencies
sudo apt update
sudo apt install build-essential python3 python3-pip python3-dev pkg-config

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Build Process
```bash
# Clone and setup
git clone https://github.com/vocalize/vocalize.git
cd vocalize

# Build and develop
uv sync --dev
uv run maturin develop --release
```

### macOS

#### Prerequisites
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Build Process
```bash
# Detect architecture and build accordingly
if [[ $(uname -m) == "arm64" ]]; then
    # Apple Silicon
    uv run maturin develop --release --target aarch64-apple-darwin
else
    # Intel Mac
    uv run maturin develop --release --target x86_64-apple-darwin
fi
```

### Windows with WSL2 (RECOMMENDED APPROACH)

#### Prerequisites
1. **Install WSL2** (if not already installed):
   ```powershell
   # In PowerShell as Administrator
   wsl --install
   # Restart computer if prompted
   ```

2. **In WSL (Ubuntu):**
   ```bash
   # Update system
   sudo apt update && sudo apt upgrade

   # Install prerequisites
   sudo apt install build-essential python3 python3-pip python3-dev pkg-config p7zip-full

   # Install Rust
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source $HOME/.cargo/env

   # Add Windows target for cross-compilation
   rustup target add x86_64-pc-windows-msvc

   # Install maturin for Python packaging
   pip install maturin

   # Install UV
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

#### Build Process (Current Production Method)

##### Step 1: Build on WSL
```bash
# Navigate to project directory in WSL
cd /mnt/c/Users/[your-username]/Documents/dev/personal/vocalize

# Run the unified build script
./build_and_bundle_complete.sh
```

This script:
1. Builds the Rust extension using maturin with the Windows target
2. Downloads and extracts ONNX Runtime 1.22.1
3. Downloads and bundles VC++ redistributable DLLs
4. Creates a self-contained wheel with all dependencies

##### Step 2: Install on Windows
```powershell
# In Windows terminal
cd C:\Users\[your-username]\Documents\dev\personal\vocalize

# Sync Python environment
uv sync

# Install the bundled wheel
uv pip install target/wheels/vocalize_python-0.1.0-cp38-abi3-win_amd64_bundled.whl --force-reinstall

# Test the installation
uv run python -m vocalize
```

## DLL Handling Strategy (Windows)

### The Problem: System32 ONNX Runtime Conflict

Windows may have an incompatible ONNX Runtime version in System32 (commonly v1.17.1) that conflicts with our bundled version (v1.22.x). Windows DLL loading mechanism will return the already-loaded DLL regardless of the path specified.

### Our Solution: Pre-emptive DLL Loading

We implement pre-emptive DLL loading in `vocalize-python/src/lib.rs`:

```rust
// Pre-load our DLLs using Windows API before ort crate initializes
LoadLibraryExW(providers_dll_path, null, LOAD_WITH_ALTERED_SEARCH_PATH);
LoadLibraryExW(onnxruntime_dll_path, null, LOAD_WITH_ALTERED_SEARCH_PATH);
```

This ensures our bundled DLLs are loaded first, preventing System32 interference.

### Bundled Dependencies

The Windows wheel includes:
- `onnxruntime.dll` (v1.22.1)
- `onnxruntime_providers_shared.dll`
- VC++ Redistributable DLLs:
  - `vcruntime140.dll`, `vcruntime140_1.dll`
  - `msvcp140.dll`, `msvcp140_1.dll`, `msvcp140_2.dll`
  - `vccorlib140.dll`, `concrt140.dll`

## Architecture Details

### Cross-Compilation Approach

We use WSL for building Windows binaries because:
1. **Consistent build environment** - Avoids Windows-specific build issues
2. **Better tooling** - Linux tools like 7zip work more reliably
3. **Automated DLL bundling** - Script can extract and bundle dependencies
4. **No Visual Studio required** - Uses maturin's cross-compilation capabilities

### Environment Variables

#### Build Time
- `RUST_LOG=debug` - Enable debug logging
- `VOCALIZE_DEBUG=1` - Enable vocalize-specific debug output

#### Runtime
- `ORT_DYLIB_PATH` - Path to ONNX Runtime DLL (set automatically)
- `OMP_NUM_THREADS` - OpenMP thread limit (set in _env_setup.py)
- `MKL_NUM_THREADS` - Intel MKL thread limit

## Troubleshooting

### ONNX Runtime Version Conflict

**Symptom**: 
```
ort 2.0.0-rc.10 is not compatible with the ONNX Runtime binary found at `onnxruntime.dll`; expected GetVersionString to return '1.22.x', but got '1.17.1'
```

**Cause**: Windows System32 contains an older ONNX Runtime version

**Solution**: Our pre-emptive loading handles this automatically. If issues persist:
```cmd
# Run as Administrator to temporarily rename System32 version
ren C:\Windows\System32\onnxruntime.dll onnxruntime.dll.bak
```

### LoadLibraryExW Failed

**Symptom**: 
```
An error occurred while attempting to load the ONNX Runtime binary at `path/to/onnxruntime.dll`: LoadLibraryExW failed
```

**Causes**:
1. Missing dependencies (onnxruntime_providers_shared.dll)
2. Path formatting issues
3. DLL already loaded from different location

**Solutions**:
1. Ensure both ONNX Runtime DLLs are bundled
2. Check the build output for "Copied: onnxruntime_providers_shared.dll"
3. Verify the wheel contents include all DLLs

### Common Build Issues

#### 1. "maturin failed: Need a Python interpreter"

**Error:**
```
Need a Python interpreter to compile for Windows without PyO3's `generate-import-lib` feature
```

**Solutions:**
1. **Enable generate-import-lib** (Recommended):
   - Already configured in `crates/vocalize-python/Cargo.toml`
   - Uses PyO3's automatic import library generation

2. **Specify Python interpreter path:**
   ```bash
   maturin build --release --target x86_64-pc-windows-gnu --zig -i /path/to/windows/python.exe
   ```

3. **Install Python in WSL:**
   ```bash
   sudo apt install python3 python3-dev
   ```

### 2. "Error calling dlltool: No such file or directory"

**Error:**
```
Error calling dlltool 'x86_64-w64-mingw32-dlltool': No such file or directory (os error 2)
```

**Solutions:**
1. **Use generate-import-lib feature** (Already enabled)
2. **Install MinGW-w64** (if using GNU target):
   ```bash
   sudo apt install mingw-w64
   ```
3. **Or use cargo-xwin for MSVC target:**
   ```bash
   cargo install cargo-xwin
   ```

### 3. "DirectML.lib not found"

**Error:**
```
lld-link: error: could not open 'DirectML.lib': No such file or directory
```

**Cause:** DirectML is a Windows-only library not available in WSL

**Solution:** DirectML is disabled in .cargo/config.toml and Cargo.toml files:
```bash
# The project is configured to avoid DirectML automatically
# .cargo/config.toml sets ORT_STRATEGY=download and ORT_USE_DIRECTML=0
# Cargo.toml files use default-features = false for ORT

# Build with cargo-xwin (recommended)
cargo xwin build --release --target x86_64-pc-windows-msvc --manifest-path crates/vocalize-python/Cargo.toml

# Or use maturin directly
maturin build --release --target x86_64-pc-windows-msvc
```

### 4. "can't find crate for `core`"

**Error:**
```
error[E0463]: can't find crate for `core`
= note: the `x86_64-pc-windows-gnu` target may not be installed
```

**Solution:** Add the Windows target:
```bash
rustup target add x86_64-pc-windows-gnu
```

### PYO3_CONFIG_FILE Errors

**Error:**
```
error: PYO3_CONFIG_FILE is set but does not contain a valid config
```

**Solution**: The build script handles this automatically. Ensure you're using the latest `build_and_bundle_complete.sh`.

## Development Workflow

### Daily Development Cycle

1. **Make changes** in your preferred editor
2. **Build on WSL**: `./build_and_bundle_complete.sh`
3. **Install on Windows**: 
   ```powershell
   uv pip install target/wheels/vocalize_python-0.1.0-cp38-abi3-win_amd64_bundled.whl --force-reinstall
   ```
4. **Test**: `uv run python -m vocalize`

### Testing Changes

```bash
# Quick smoke test
uv run python -c "import vocalize; print('✓ Import successful')"

# Test synthesis
uv run python -m vocalize speak "Test" --output test.wav

# Run full test suite
uv run pytest
```

## Performance Tips

### 1. Build Performance
- Use `--release` flag for optimized builds
- Enable LTO in `Cargo.toml` (already configured)
- Use parallel compilation: `export CARGO_BUILD_JOBS=4`

### 2. Development Workflow
- Use `maturin develop` for rapid iteration on Linux/macOS
- Use `cargo xwin build` or `maturin build` for Windows cross-compilation
- Cache builds: `export CARGO_TARGET_DIR=~/.cargo/target`

### 3. Memory Usage
- Limit ONNX Runtime threads in production
- Use `strip=true` in release builds (already configured)
- Monitor memory usage with system tools

## Testing

### Unit Tests
```bash
# Rust tests
cargo test

# Python tests (after building)
uv run pytest
```

### Integration Tests
```bash
# Test installation
uv run python -c "import vocalize; print('✓ Import successful')"

# Test basic functionality
uv run python -c "
import vocalize
engine = vocalize.VocalizeEngine()
print('✓ Engine creation successful')
"
```

## Contributing

1. **Follow the build process** for your platform
2. **Run tests** before submitting PRs
3. **Update documentation** for any changes to build process
4. **Test cross-platform** when possible

## Useful Commands

### Development
```bash
# Clean build
cargo clean

# Check without building
cargo check

# Format code
cargo fmt

# Lint
cargo clippy

# Update dependencies
cargo update
```

### Python
```bash
# Reinstall in development mode
uv run maturin develop --release

# Build wheel
uv run maturin build --release

# Install wheel
uv pip install target/wheels/*.whl --force-reinstall
```

## Common Development Tasks

### Adding a New Voice Model

1. Add the model file to `crates/vocalize-core/models/`
2. Update the model registry in `model/manager.rs`
3. Add voice metadata in `model_manager.py`
4. Test the new voice:
   ```bash
   uv run python -m vocalize speak "Test" --voice new_voice_id
   ```

### Updating ONNX Runtime

1. Update version in `crates/vocalize-python/build.rs`:
   ```rust
   let onnx_version = "1.22.1";  // Change this
   ```
2. Test thoroughly for compatibility
3. Update documentation if needed

### Debugging DLL Issues

Enable verbose output:
```bash
# Set environment variables
export RUST_LOG=debug
export VOCALIZE_DEBUG=1

# Run with backtrace
RUST_BACKTRACE=1 uv run python -m vocalize
```

## Release Process

1. Update version in:
   - All `Cargo.toml` files
   - `pyproject.toml`
2. Build release wheel: `./build_and_bundle_complete.sh`
3. Test on clean Windows system
4. Create GitHub release with changelog

## Getting Help

1. **Check this guide** for common issues
2. **Enable debug logging** to get more information
3. **Search GitHub issues** for similar problems
4. **Create detailed bug reports** with:
   - Error messages and stack traces
   - Steps to reproduce
   - Environment: `rustc --version`, `python --version`
   - Build output from `./build_and_bundle_complete.sh`

## Additional Resources

- [PyO3 Documentation](https://pyo3.rs/)
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [Maturin Documentation](https://maturin.rs/)
- [Rust Cross-compilation Guide](https://rust-lang.github.io/rustup/cross-compilation.html)