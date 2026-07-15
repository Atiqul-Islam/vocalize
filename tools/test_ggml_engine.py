#!/usr/bin/env python3
"""
Test GGML engine implementation
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add vocalize to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    # Import GGML functions from Rust module
    from vocalize_rust import process_text_to_phonemes, synthesize_ggml
except ImportError as e:
    print(f"Error importing vocalize_rust: {e}")
    print("Make sure to build the Rust module first with: maturin develop")
    sys.exit(1)

def test_phoneme_processing():
    """Test fast phoneme processing"""
    print("\n=== Testing Phoneme Processing ===")
    
    test_texts = [
        "Hello world",
        "This is a test of the speech synthesis system",
        "One two three four five",
        "The quick brown fox jumps over the lazy dog"
    ]
    
    for text in test_texts:
        start = time.time()
        try:
            phonemes = process_text_to_phonemes(text)
            elapsed = (time.time() - start) * 1000
            print(f"✓ '{text}' -> {len(phonemes)} phonemes in {elapsed:.1f}ms")
            print(f"  Phonemes: {phonemes[:20]}..." if len(phonemes) > 20 else f"  Phonemes: {phonemes}")
        except Exception as e:
            print(f"✗ Failed to process '{text}': {e}")

def test_ggml_synthesis():
    """Test GGML synthesis"""
    print("\n=== Testing GGML Synthesis ===")
    
    # Create test inputs
    test_tokens = [0, 44, 15, 48, 5, 13, 19, 8, 50, 3, 16, 48, 33, 0]  # "hello world" phonemes
    style_vector = np.random.randn(256).astype(np.float32).tolist()
    
    # Test different speeds
    speeds = [1.0, 0.5, 2.0]
    
    for speed in speeds:
        print(f"\nTesting with speed={speed}")
        start = time.time()
        
        try:
            # Use dummy model path for now - will need actual GGUF model
            audio = synthesize_ggml(
                test_tokens,
                style_vector,
                speed,
                "piper-amy-medium",
                "dummy_model.gguf"
            )
            
            elapsed = time.time() - start
            duration_sec = len(audio) / 22050  # Assuming 22kHz sample rate
            
            print(f"✓ Generated {len(audio)} samples ({duration_sec:.2f}s audio) in {elapsed:.3f}s")
            print(f"  RTF (Real-Time Factor): {elapsed/duration_sec:.2f}x")
            
            # Check audio properties
            audio_np = np.array(audio)
            print(f"  Audio range: [{audio_np.min():.3f}, {audio_np.max():.3f}]")
            print(f"  Audio mean: {audio_np.mean():.3f}, std: {audio_np.std():.3f}")
            
        except Exception as e:
            print(f"✗ Synthesis failed: {e}")
            import traceback
            traceback.print_exc()

def test_full_pipeline():
    """Test complete pipeline from text to audio"""
    print("\n=== Testing Full Pipeline ===")
    
    test_text = "Hello, this is a test of the GGML speech synthesis engine."
    
    # Step 1: Process text to phonemes
    print(f"\n1. Processing text: '{test_text}'")
    try:
        start = time.time()
        phonemes = process_text_to_phonemes(test_text)
        phoneme_time = time.time() - start
        print(f"✓ Got {len(phonemes)} phonemes in {phoneme_time*1000:.1f}ms")
    except Exception as e:
        print(f"✗ Phoneme processing failed: {e}")
        return
    
    # Step 2: Synthesize audio
    print("\n2. Synthesizing audio...")
    try:
        start = time.time()
        style_vector = np.zeros(256, dtype=np.float32).tolist()  # Neutral style
        
        audio = synthesize_ggml(
            phonemes,
            style_vector,
            1.0,  # Normal speed
            "piper-amy-medium",
            "dummy_model.gguf"
        )
        
        synthesis_time = time.time() - start
        total_time = phoneme_time + synthesis_time
        
        print(f"✓ Generated {len(audio)} samples in {synthesis_time:.3f}s")
        print(f"\nTotal pipeline time: {total_time:.3f}s")
        
        if total_time < 2.0:
            print("🎉 SUCCESS: Achieved <2s latency goal!")
        else:
            print(f"⚠️  Pipeline took {total_time:.3f}s (goal: <2s)")
            
    except Exception as e:
        print(f"✗ Synthesis failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("GGML Engine Test Suite")
    print("=" * 50)
    
    # Test individual components
    test_phoneme_processing()
    
    # Only test synthesis if we have a model
    # In real usage, we'd download and convert a model first
    print("\n⚠️  Note: GGML synthesis tests require a converted GGUF model")
    print("   Run 'python tools/convert_piper_to_gguf.py' first to create one")
    
    # Uncomment when we have actual GGUF models
    # test_ggml_synthesis()
    # test_full_pipeline()
    
    print("\n" + "=" * 50)
    print("Test suite completed!")

if __name__ == "__main__":
    main()