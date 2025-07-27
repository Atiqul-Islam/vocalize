# Vocalize TTS Performance Optimization Plan V2
## Target: Sub-1 Second Synthesis

## Current Status After Phase 1 Implementation

### Performance Improvements Achieved:
- **Original baseline**: 14.912s total
- **After Phase 1**: 10.245s total (31% improvement)
- **Key wins**:
  - Session pool creation: 6.4s → 1.3s (✅ 78% reduction)
  - Removed redundant checks: ~0.5s saved
  - NLTK made conditional: ~0.2s saved

### Current Performance Breakdown:
| Component | Time | % of Total |
|-----------|------|------------|
| Token processing (ttstokenizer) | 4.2s | 41% |
| Session creation | 1.3s | 13% |
| ONNX inference | 1.2s | 12% |
| ort::init() | 0.7s | 7% |
| Voice NPZ loading | 0.15s | 1.5% |
| Other overhead | 0.5s | 5% |
| **Total synthesis** | **~7.4s** | **72%** |
| Audio playback | 2.6s | 25% |
| **Total execution** | **~10.2s** | **100%** |

### Main Bottlenecks Identified:
1. **ttstokenizer**: 4.2s - Even when "cached", still extremely slow
2. **Session creation**: 1.3s - Happens every CLI call
3. **ONNX inference**: 1.2s - Slower than expected for 18 tokens

## Phase 2A: Token Processing Optimization (Target: Save 4s)

### Option 1: Pre-computed Token Cache ⭐ Recommended
Create a persistent cache of pre-computed tokens for common words and phrases.

**Implementation**:
```python
# vocalize/token_cache.py
import pickle
from pathlib import Path

class TokenCache:
    def __init__(self, cache_path="tokens.cache"):
        self.cache = self._load_cache(cache_path)
    
    def get_tokens(self, text):
        # Check cache first
        if text in self.cache:
            return self.cache[text]
        
        # Compute only if not cached
        tokens = self._compute_tokens(text)
        self._update_cache(text, tokens)
        return tokens
```

**Build cache script**:
```python
# scripts/build_token_cache.py
# Pre-compute tokens for:
# - Top 10,000 English words
# - Common phrases
# - Numbers 0-1000
# - Common names
```

**Expected**: 4.2s → <100ms

### Option 2: Direct Character Input
Test if Kokoro model accepts character sequences directly, bypassing phoneme conversion.

**Implementation**:
```python
# In KokoroPhonemeProcessor.process_text()
if use_direct_chars:
    # Skip ttstokenizer entirely
    char_tokens = [ord(c) for c in text]
    return {"input_ids": char_tokens, "style": style_vector}
```

### Option 3: Alternative Fast G2P
Replace ttstokenizer with faster alternatives:
- **gruut**: Comprehensive with SSML support
- **g2p_en**: Optimized for English
- **epitran**: Rule-based, very fast

## Phase 2B: ONNX Runtime Optimization (Target: Save 1-2s)

### 1. Offline Model Optimization
Pre-optimize the model to eliminate runtime optimization overhead.

**One-time optimization script**:
```python
# scripts/optimize_model.py
import onnxruntime as ort

# Create optimized model
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.optimized_model_filepath = "kokoro-optimized.onnx"

# Load original model and save optimized version
session = ort.InferenceSession("kokoro-v1.0.onnx", sess_options)
print("Optimized model saved to kokoro-optimized.onnx")
```

### 2. Session Configuration Updates
**File**: `crates/vocalize-core/src/onnx_engine/session_pool.rs`

```rust
// Use pre-optimized model
.with_optimization_level(GraphOptimizationLevel.ORT_DISABLE_ALL)?
// Single thread for small model
.with_intra_threads(1)?
.with_inter_threads(1)?
// Disable CPU spinning
.with_session_config_entry("session.intra_op.allow_spinning", "0")?
// Disable memory arena for faster startup
.enable_cpu_mem_arena(false)?
```

### 3. Model Quantization
Reduce model size and inference time with INT8 quantization.

```bash
# Quantize model (2-4x speedup)
python -m onnxruntime.quantization.quantize_dynamic \
    --model_input kokoro-v1.0.onnx \
    --model_output kokoro-v1.0-int8.onnx \
    --per_channel
```

## Phase 3: Architecture Changes for <1s

### Option A: Daemon/Service Mode ⭐ Recommended

Create a persistent background service that keeps the model loaded.

**Architecture**:
```
┌─────────────┐     HTTP      ┌──────────────┐
│   CLI       │ ─────────────→ │  TTS Server  │
│  (client)   │ ←───────────── │  (FastAPI)   │
└─────────────┘    Audio       └──────────────┘
                                      ↓
                               ┌──────────────┐
                               │ Loaded Model │
                               │   (cached)   │
                               └──────────────┘
```

**Implementation Files**:
1. `vocalize/server.py` - FastAPI server with loaded model
2. `vocalize/client.py` - HTTP client for CLI
3. `vocalize/cli.py` - Updated to use client mode

**Server Features**:
- Auto-start on first request
- Keep-alive with timeout
- Concurrent request handling
- Health check endpoint

### Option B: Shared Memory IPC

Use OS-level shared memory for zero-copy model access.

**Implementation**:
- Python `multiprocessing.shared_memory`
- Model loaded into shared memory segment
- CLI processes attach to existing segment
- Platform-specific considerations

### Option C: Process Pool

Maintain a pool of pre-initialized worker processes.

**Implementation**:
- `multiprocessing.Pool` with persistent workers
- Each worker has model pre-loaded
- Job queue for synthesis requests

## Phase 4: Advanced Optimizations

### 1. GPU Acceleration
Enable hardware acceleration for massive speedup.

**CUDA Support**:
```python
providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
    }),
    'CPUExecutionProvider',
]
```

**DirectML (Windows)**:
```python
providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
```

**Expected**: 8x speedup on compatible hardware

### 2. Streaming Inference
Process and play audio in chunks for perceived low latency.

**Implementation**:
- Split inference into overlapping chunks
- Use circular buffer for audio
- Start playback after first chunk
- Perceived latency: <100ms

### 3. Batch Processing
Process multiple texts in parallel for better throughput.

## Implementation Timeline

### Week 1: Token Cache (High Priority)
- [ ] Day 1-2: Build token cache infrastructure
- [ ] Day 3: Create pre-computation script
- [ ] Day 4: Integrate with KokoroPhonemeProcessor
- [ ] Day 5: Test and benchmark

### Week 2: ONNX Optimizations
- [ ] Day 1: Create offline optimized model
- [ ] Day 2: Update session configuration
- [ ] Day 3: Test quantized model quality
- [ ] Day 4-5: Benchmark and tune

### Week 3: Service Architecture
- [ ] Day 1-2: Implement FastAPI server
- [ ] Day 3: Create client library
- [ ] Day 4: Update CLI integration
- [ ] Day 5: Add auto-start logic

### Week 4: Polish & Advanced Features
- [ ] Day 1-2: GPU support
- [ ] Day 3-4: Streaming implementation
- [ ] Day 5: Final testing and documentation

## Expected Results

### After Phase 2A+2B (Token Cache + ONNX):
| Component | Current | Target | Improvement |
|-----------|---------|---------|-------------|
| Token processing | 4.2s | 0.1s | -4.1s |
| Session creation | 1.3s | 1.3s | - |
| ONNX inference | 1.2s | 0.5s | -0.7s |
| **Total** | **7.4s** | **2.0s** | **-5.4s** |

### After Phase 3 (Service Mode):
| Component | Current | Target | Improvement |
|-----------|---------|---------|-------------|
| Token processing | 0.1s | 0.1s | - |
| Session creation | 1.3s | 0s | -1.3s |
| ONNX inference | 0.5s | 0.5s | - |
| **Total** | **2.0s** | **0.6s** | **-1.4s** |

### With GPU/Advanced:
- **Target: <200ms** achievable
- 40x improvement from original baseline

## Testing Strategy

### 1. Performance Benchmarks
```python
# tests/benchmark_performance.py
def benchmark_synthesis():
    texts = [
        "Hello world",  # Short
        "The quick brown fox...",  # Medium
        "Lorem ipsum..." * 10  # Long
    ]
    
    for text in texts:
        measure_time(synthesize, text)
```

### 2. Quality Validation
- A/B test original vs optimized audio
- MOS (Mean Opinion Score) testing
- Verify no quality degradation

### 3. Load Testing
```bash
# Test concurrent requests
locust -f tests/load_test.py --users 100 --spawn-rate 10
```

### 4. Platform Testing
- Windows 10/11
- Ubuntu 20.04/22.04
- macOS 12+
- ARM64 support

## Risk Mitigation

### 1. Quality Degradation
- Keep original model as fallback
- A/B testing framework
- User-selectable quality modes

### 2. Platform Compatibility
- Conditional optimizations
- Fallback implementations
- Clear error messages

### 3. Memory Usage
- Monitor memory consumption
- Implement cleanup routines
- Set resource limits

## Success Criteria

1. **Primary Goal**: <1s synthesis time ✅
2. **Stretch Goal**: <200ms with GPU
3. **Maintain**: Audio quality (MOS ≥ 4.35)
4. **Support**: All current platforms
5. **Scalability**: Handle 100+ RPS in server mode

## Conclusion

This plan provides a clear path from the current 10.2s to sub-1 second synthesis through:
1. **Immediate wins** with token caching (save 4s)
2. **ONNX optimizations** (save 1-2s)
3. **Architecture changes** for persistent model (save 1.3s)
4. **Advanced features** for <200ms performance

The modular approach allows incremental improvements while maintaining compatibility and quality.