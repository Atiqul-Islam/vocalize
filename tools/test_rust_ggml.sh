#!/bin/bash
# Test GGML implementation in Rust

echo "=== Testing GGML Rust Implementation ==="
echo

# Navigate to the vocalize-core crate
cd crates/vocalize-core

echo "1. Running cargo check..."
cargo check --all-features
if [ $? -ne 0 ]; then
    echo "❌ Cargo check failed"
    exit 1
fi
echo "✅ Cargo check passed"
echo

echo "2. Running tensor operation tests..."
cargo test tensor_ops::tests -- --nocapture
if [ $? -ne 0 ]; then
    echo "❌ Tensor ops tests failed"
    exit 1
fi
echo "✅ Tensor ops tests passed"
echo

echo "3. Running phoneme processor tests..."
cargo test phoneme_processor::tests -- --nocapture
if [ $? -ne 0 ]; then
    echo "❌ Phoneme processor tests failed"
    exit 1
fi
echo "✅ Phoneme processor tests passed"
echo

echo "4. Running GGUF format tests..."
cargo test gguf_format -- --nocapture
if [ $? -ne 0 ]; then
    echo "❌ GGUF format tests failed"
    exit 1
fi
echo "✅ GGUF format tests passed"
echo

echo "5. Building release version..."
cargo build --release
if [ $? -ne 0 ]; then
    echo "❌ Release build failed"
    exit 1
fi
echo "✅ Release build succeeded"
echo

echo "=== All GGML tests passed! ==="