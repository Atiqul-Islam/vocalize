# Vocalize TTS Performance Optimization Plan

## Current Performance Baseline

**Test Command**: `uv run python -m vocalize --verbose speak "Hello, world!" --play`

**Total Execution Time**: 14.912s

### Breakdown:
- Environment setup: ~0.5s (DLL loading, thread config)
- Model/Voice Manager init: 0.016s
- Model availability check: 0.001s
- **Speech synthesis: 12.187s** ← MAIN BOTTLENECK (81.7%)
- Audio playback: 2.706s (actual playback time)

### Detailed Synthesis Breakdown (12.187s):
1. Import KokoroPhonemeProcessor: 0.000s
2. Load NPZ voices file: ~0.5s (estimated)
3. Token generation: ~0.5s (including ttstokenizer import warning)
4. Rust synthesis call: ~11s (estimated), includes:
   - Tokio runtime creation
   - ONNX engine creation
   - Model loading from disk
   - Session pool creation (4 sessions)
   - Actual inference
   - Audio data extraction

## Root Cause Analysis

### Primary Issues:
1. **No model persistence**: Each CLI call creates new ONNX engine and loads model from scratch
2. **Redundant initialization**: Multiple validation checks for model/voice availability
3. **Expensive imports**: NLTK data checks on every import
4. **No pre-warming**: Model isn't warmed up before first inference
5. **Excessive async overhead**: Full Tokio runtime for single operation

### Secondary Issues:
1. Voice embeddings loaded from NPZ on every call
2. No caching of tokenizer or phoneme processor
3. Session pool might be oversized (4 threads for single inference)
4. Multiple Python/Rust boundary crossings

## Phase 1: Quick Wins + Instrumentation (Immediate Implementation)

### 1.1 Remove Redundant Checks

**File: vocalize/cli.py**
- Line 273-276: Remove `ensure_model_available()` call - it's called again during synthesis
- Line 335-357: Skip voice discovery when voice is already specified
- Line 259-270: Combine VoiceManager and ModelManager imports/init

**File: vocalize/_env_setup.py**
- Line 26-49: Make NLTK data download conditional (check if already exists)
- Only run download if `--download-nltk` flag is passed

**Expected Savings**: ~1.1s

### 1.2 Add Comprehensive Timing Logs

Add timing instrumentation at these locations:

**Python Side (vocalize/cli.py)**:
```
[CLI_START]
├─ Import phase
│  ├─ ModelManager import
│  ├─ VoiceManager import
│  └─ vocalize_rust import
├─ Initialization phase
│  ├─ ModelManager init
│  ├─ VoiceManager init
│  └─ Model availability check
├─ Synthesis phase
│  ├─ synthesize_with_tokens start
│  ├─ KokoroPhonemeProcessor import
│  ├─ Voice loading from NPZ
│  ├─ Token generation
│  └─ vocalize_rust.synthesize_from_tokens_neural call
└─ Post-processing phase
   ├─ Audio data handling
   └─ Playback/save
```

**Rust Side (crates/vocalize-rust/src/lib.rs)**:
```
synthesize_from_tokens_neural:
├─ Parameter validation
├─ Tokio runtime creation
├─ ONNX engine operations
│  ├─ Engine creation (new_with_default_cache)
│  ├─ Model loading (load_model)
│  └─ Inference (synthesize_from_tokens)
└─ Result handling
```

**Rust Side (crates/vocalize-core/src/onnx_engine.rs)**:
```
OnnxTtsEngine operations:
├─ new_with_default_cache
│  ├─ Project directories setup
│  ├─ ORT environment setup
│  └─ ort::init().commit()
├─ load_model
│  ├─ Model path resolution
│  ├─ Session pool creation
│  └─ Individual session creation (× pool_size)
└─ synthesize_from_tokens
   ├─ Input validation
   ├─ Session acquisition
   ├─ Tensor creation
   ├─ ONNX inference (session.run)
   └─ Audio extraction
```

### 1.3 Optimize Import Chain

**File: vocalize/cli.py**
- Move `from .model_manager import KokoroPhonemeProcessor` inside function (line 200)
- Make sounddevice import lazy (only import when --play is used)

**File: vocalize/model_manager.py**
- Lazy import numpy, requests, huggingface_hub
- Only import when actually downloading or processing

**Expected Savings**: ~0.2s

## Phase 2: Core Optimizations (After Analysis)

### 2.1 Implement Model Persistence for Library Usage

Create new API structure:
```
vocalize.Engine class:
- load_model(model_id) - Explicit model loading
- synthesize(text, voice, speed) - Fast synthesis with loaded model
- synthesize_auto(text, voice, speed) - Auto-load for CLI usage
```

### 2.2 Optimize ONNX Initialization

**Investigate**:
- Why ort::init() takes so long
- Session pool size optimization (try 2 instead of 4)
- GraphOptimizationLevel settings
- Memory pattern optimization impact

### 2.3 Resource Caching Strategy

**Cache these resources**:
- ONNX engine instance (Rust side)
- Tokio runtime (single instance)
- Voice embeddings (after first load)
- Tokenizer instance

## Phase 3: Advanced Optimizations (If Needed)

### 3.1 Model Optimization
- Quantize model to INT8
- Pre-compile to ORT format
- Investigate model pruning

### 3.2 Alternative Architectures
- Background service/daemon mode
- Process pooling for batch operations
- Shared memory for model data

### 3.3 Platform-Specific Optimizations
- Memory-mapped model files
- GPU acceleration (where available)
- Native CPU optimizations

## Implementation Checklist

### Immediate Tasks (Phase 1):
- [ ] Add timing logs to Python synthesis path
- [ ] Add timing logs to Rust synthesis path
- [ ] Remove redundant model availability check
- [ ] Skip voice discovery when voice specified
- [ ] Make NLTK download conditional
- [ ] Lazy import expensive Python modules
- [ ] Test and measure improvements

### Analysis Tasks:
- [ ] Profile with timing data
- [ ] Identify unexpected delays
- [ ] Memory usage analysis
- [ ] CPU usage patterns

### Core Optimization Tasks (Phase 2):
- [ ] Design Engine API for library usage
- [ ] Implement model persistence
- [ ] Add pre-warming capability
- [ ] Optimize session pool size
- [ ] Cache voice embeddings

## Success Metrics

### Phase 1 Goals:
- Remove ~1.1s from total time
- Get detailed timing breakdown
- Identify the mystery 6-7s delay

### Phase 2 Goals:
- CLI mode: <3s total (from 14.9s)
- Library mode first call: <3s
- Library mode subsequent: <200ms

### Final Goals:
- Best-in-class TTS performance
- Maintain audio quality
- Keep cross-platform compatibility

## Testing Requirements

### Performance Tests:
1. Baseline measurement before changes
2. Measurement after each optimization
3. Test with different text lengths
4. Test with different voices
5. Memory usage profiling

### Regression Tests:
1. Audio quality comparison
2. Cross-platform compatibility
3. Thread safety for library mode
4. Error handling

## Code Locations Reference

### Key Files to Modify:
1. `vocalize/cli.py` - CLI entry point, redundant checks
2. `vocalize/_env_setup.py` - Environment initialization
3. `vocalize/model_manager.py` - Model loading logic
4. `crates/vocalize-rust/src/lib.rs` - Python bindings, DLL loading
5. `crates/vocalize-core/src/onnx_engine.rs` - Core ONNX logic
6. `crates/vocalize-core/src/onnx_engine/session_pool.rs` - Session management

### Critical Functions:
1. `handle_speak_command()` - CLI entry point
2. `synthesize_with_tokens()` - Python synthesis wrapper
3. `synthesize_from_tokens_neural()` - Rust synthesis entry
4. `OnnxTtsEngine::new_with_default_cache()` - Engine creation
5. `OnnxTtsEngine::synthesize_from_tokens()` - Core inference

## Notes for Implementers

1. **Start with Phase 1** - Quick wins + instrumentation
2. **Measure everything** - Don't assume, profile
3. **Test incrementally** - Verify each change
4. **Preserve quality** - Speed isn't everything
5. **Document changes** - Update this plan with findings

## Current Hypothesis

The 12s synthesis time likely breaks down as:
- 2-3s: ONNX Runtime initialization + model loading
- 1s: Session pool creation
- 0.5s: Voice embedding loading
- 0.5s: Tokenization
- **7-8s: Unknown overhead** ← This needs investigation

The unknown overhead might be:
- Blocking I/O in async context
- Excessive memory allocation/copying
- Lock contention
- First-run JIT compilation
- Windows-specific DLL issues

Finding and fixing this unknown overhead is the key to achieving <3s performance.