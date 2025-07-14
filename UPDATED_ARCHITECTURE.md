# Vocalize 2025 TTS Architecture - Updated

## Overview

This document describes the updated architecture for Vocalize TTS using the 2025 Kokoro neural model with proper phoneme-based processing.

## Architecture Flow

```
Text Input ("Hello World")
    ↓
[Python Layer]
KokoroPhonemeProcessor (ttstokenizer)
    ↓ G2P Conversion
Phoneme Tokens [0, 50, 83, 54, 156, 57, 135, 16, 65, 156, 85, 54, 46, 0]
    ↓
[Rust Layer via PyO3]
OnnxTtsEngine.synthesize_from_tokens()
    ↓ ONNX Runtime Inference
Audio Samples (Vec<f32>)
    ↓
WAV File Output
```

## Key Components

### 1. Python Layer (`vocalize/model_manager.py`)

**KokoroPhonemeProcessor**
- Uses `ttstokenizer` (IPATokenizer) for G2P conversion
- Converts text to phoneme-based tokens
- Adds padding tokens (0) at start and end
- Loads voice embeddings from NPZ file
- Returns token IDs, style vector, and speed

```python
def process_text(self, text: str, voice_id: str = "af_sarah") -> dict:
    tokens = self.tokenizer(text)  # Returns numpy array
    input_ids = [0] + tokens.tolist()[:510] + [0]
    style_vector = self._get_voice_embedding(voice_id)  # From NPZ file
    return {
        "input_ids": input_ids,
        "style": style_vector,
        "speed": 1.0,
        "voice_id": voice_id
    }
```

**Voice Loading**
- `voices-v1.0.bin` is an NPZ (numpy compressed) file
- Contains 54 voices with names like af_sarah, am_adam, etc.
- Each voice has shape (510, 1, 256)
- Style vector extracted as `voice[0, 0, :]` (first frame)

### 2. Rust Layer (`crates/vocalize-core/src/onnx_engine.rs`)

**OnnxTtsEngine**
- Accepts pre-processed tokens from Python
- No longer handles text processing or tokenization
- Performs ONNX neural inference
- Returns audio samples

```rust
pub async fn synthesize_from_tokens(
    &mut self, 
    input_ids: Vec<i64>, 
    style_vector: Vec<f32>, 
    speed: f32,
    model_id: ModelId
) -> Result<Vec<f32>>
```

### 3. PyO3 Bindings (`crates/vocalize-python/src/lib.rs`)

**synthesize_from_tokens_neural()**
- Bridge between Python and Rust
- Validates inputs
- Calls Rust ONNX engine
- Returns audio data to Python

## ONNX Model Specifications (Kokoro v1.0)

Based on 2025 research and actual model inspection:

### Model Inputs

1. **`tokens`** (NOT "input_ids")
   - Type: int64 tensor
   - Shape: [1, sequence_length]
   - Description: Tokenized phoneme IDs with padding
   - The kokoro-v1.0.onnx model uses "tokens"

2. **`style`**
   - Type: float32 tensor
   - Shape: [1, 256]
   - Description: Voice embedding vector
   - Controls voice characteristics

3. **`speed`**
   - Type: float32 tensor
   - Shape: [1]
   - Description: Speaking rate multiplier
   - Default: 1.0

### Model Output

- **`audio`**: float32 tensor containing raw audio samples at 24kHz

## Implementation Status

### ✅ Completed
1. **Phoneme Processing**: ttstokenizer integration working
2. **PyO3 Bindings**: synthesize_from_tokens_neural() implemented
3. **ONNX Inference**: Neural synthesis produces audio
4. **Voice Loading**: NPZ file loading implemented
5. **WAV Generation**: Proper WAV files created

### Voice Data Format
The `voices-v1.0.bin` file is an NPZ (numpy compressed archive) containing:
- 54 different voices (af_sarah, am_adam, etc.)
- Each voice is a numpy array of shape (510, 1, 256)
- The style vector is extracted from the first frame: `voice[0, 0, :]`
- Values are in range approximately [-1.35, 1.31]

### Commands to Build and Test
```bash
# 1. Build the Rust extension
uv run maturin develop

# 2. Test the complete pipeline
uv run python3 test_wav_generation.py

# 3. Play the generated audio
# On Windows: start hello_world_phoneme_pipeline.wav
# On Linux/Mac: open hello_world_phoneme_pipeline.wav
```

## Dependencies

### Python
- `ttstokenizer`: Phoneme tokenization (G2P)
- `huggingface_hub`: Model management
- `onnx`: Model inspection
- `numpy`: Array operations

### Rust
- `ort`: ONNX Runtime bindings
- `tokio`: Async runtime
- `pyo3`: Python bindings

## Testing

### Test Script: `test_wav_generation.py`
1. Initializes KokoroPhonemeProcessor
2. Converts "Hello World" to phoneme tokens
3. Calls Rust synthesis via PyO3
4. Saves WAV file
5. Validates output

### Expected Output
```
✅ Text processed: 14 tokens, 256 dimensions
✅ Rust synthesis completed: ~48000 samples
✅ WAV file created: hello_world_phoneme_pipeline.wav
✅ Duration: ~2.0 seconds at 24kHz
```

## Future Enhancements

1. **Real Voice Embeddings**: Load actual voice data from `voices-v1.0.bin` instead of random vectors
2. **Multiple Models**: Support for Chatterbox and Dia models
3. **Streaming**: Implement streaming synthesis for real-time applications
4. **Voice Cloning**: Use custom voice embeddings

## References

- [Kokoro-82M Model](https://huggingface.co/hexgrad/Kokoro-82M)
- [ONNX Community Kokoro](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX)
- [thewh1teagle/kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) - Reference implementation
- [2025 TTS Research](https://dev.to/emojiiii/running-kokoro-82m-onnx-tts-model-in-the-browser-eeh)

---

*Last Updated: July 2025*