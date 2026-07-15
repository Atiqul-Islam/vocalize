#!/usr/bin/env python3
"""
Convert Piper ONNX models to GGUF format
Follows GGML quantization patterns for fast inference
"""

import onnx
import numpy as np
import struct
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import time

class PiperToGGUFConverter:
    """Convert Piper ONNX models to GGUF format"""
    
    GGUF_MAGIC = 0x46554747  # "GGUF"
    GGUF_VERSION = 3
    
    # GGML data types
    GGML_TYPE_F32 = 0
    GGML_TYPE_F16 = 1
    GGML_TYPE_Q4_0 = 2
    GGML_TYPE_Q4_1 = 3
    GGML_TYPE_Q5_0 = 6
    GGML_TYPE_Q5_1 = 7
    GGML_TYPE_Q8_0 = 8
    
    def __init__(self):
        self.quantization_map = {
            "f32": (self.GGML_TYPE_F32, self.quantize_f32),
            "f16": (self.GGML_TYPE_F16, self.quantize_f16),
            "q8_0": (self.GGML_TYPE_Q8_0, self.quantize_q8_0),
            "q4_0": (self.GGML_TYPE_Q4_0, self.quantize_q4_0),
        }
    
    def convert(self, onnx_path: Path, output_path: Path, quantization: str = "q8_0"):
        """Convert ONNX model to GGUF format"""
        start_time = time.time()
        
        print(f"Loading ONNX model from {onnx_path}...")
        model = onnx.load(str(onnx_path))
        
        # Load config
        config_path = Path(str(onnx_path) + ".json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"Model info:")
        print(f"  - Audio sample rate: {config['audio']['sample_rate']} Hz")
        print(f"  - Phoneme type: {config['espeak']['voice']}")
        print(f"  - Model type: VITS")
        
        # Extract weights
        print(f"\nExtracting weights...")
        weights = {}
        total_params = 0
        for init in model.graph.initializer:
            tensor = onnx.numpy_helper.to_array(init)
            weights[init.name] = tensor
            total_params += tensor.size
            
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Total tensors: {len(weights)}")
        
        # Calculate sizes
        original_size = sum(t.nbytes for t in weights.values())
        print(f"  - Original size: {original_size / (1024**2):.1f} MB")
        
        print(f"\nConverting to GGUF with {quantization} quantization...")
        
        # Write GGUF file
        with open(output_path, 'wb') as f:
            # Write header
            self._write_header(f)
            
            # Write metadata
            self._write_metadata(f, config, len(weights))
            
            # Write tensor info
            tensor_infos = self._write_tensor_info(f, weights, quantization)
            
            # Align to 32-byte boundary for tensor data
            current_pos = f.tell()
            alignment = 32
            padding = (alignment - (current_pos % alignment)) % alignment
            f.write(b'\x00' * padding)
            
            # Write tensor data
            self._write_tensor_data(f, weights, tensor_infos, quantization)
        
        # Report results
        output_size = output_path.stat().st_size
        compression_ratio = original_size / output_size
        conversion_time = time.time() - start_time
        
        print(f"\n✅ Conversion complete!")
        print(f"  - Output file: {output_path}")
        print(f"  - Output size: {output_size / (1024**2):.1f} MB")
        print(f"  - Compression ratio: {compression_ratio:.1f}x")
        print(f"  - Conversion time: {conversion_time:.1f}s")
    
    def _write_header(self, f):
        """Write GGUF header"""
        f.write(struct.pack('<I', self.GGUF_MAGIC))
        f.write(struct.pack('<I', self.GGUF_VERSION))
        
    def _write_metadata(self, f, config: Dict[str, Any], tensor_count: int):
        """Write model metadata in GGUF format"""
        metadata = {
            "general.architecture": "piper",
            "general.name": "piper-vits",
            "general.description": "Piper VITS TTS model converted to GGUF",
            "piper.model_type": "vits",
            "piper.sample_rate": config["audio"]["sample_rate"],
            "piper.num_symbols": config["num_symbols"],
            "piper.num_speakers": config["num_speakers"],
            "piper.phoneme_type": config["espeak"]["voice"],
            "piper.language": config.get("language", {"code": "en"})["code"],
        }
        
        # Write tensor count
        f.write(struct.pack('<Q', tensor_count))
        
        # Write metadata count
        f.write(struct.pack('<Q', len(metadata)))
        
        # Write each metadata entry
        for key, value in metadata.items():
            self._write_string(f, key)
            self._write_metadata_value(f, value)
    
    def _write_string(self, f, s: str):
        """Write string in GGUF format"""
        encoded = s.encode('utf-8')
        f.write(struct.pack('<Q', len(encoded)))
        f.write(encoded)
    
    def _write_metadata_value(self, f, value):
        """Write metadata value with type tag"""
        if isinstance(value, bool):
            f.write(struct.pack('<I', 7))  # GGUF_METADATA_VALUE_TYPE_BOOL
            f.write(struct.pack('<?', value))
        elif isinstance(value, int):
            if -128 <= value <= 127:
                f.write(struct.pack('<I', 0))  # GGUF_METADATA_VALUE_TYPE_INT8
                f.write(struct.pack('<b', value))
            else:
                f.write(struct.pack('<I', 4))  # GGUF_METADATA_VALUE_TYPE_INT32
                f.write(struct.pack('<i', value))
        elif isinstance(value, float):
            f.write(struct.pack('<I', 6))  # GGUF_METADATA_VALUE_TYPE_FLOAT32
            f.write(struct.pack('<f', value))
        elif isinstance(value, str):
            f.write(struct.pack('<I', 8))  # GGUF_METADATA_VALUE_TYPE_STRING
            self._write_string(f, value)
        else:
            # Convert to string as fallback
            f.write(struct.pack('<I', 8))  # GGUF_METADATA_VALUE_TYPE_STRING
            self._write_string(f, str(value))
    
    def _write_tensor_info(self, f, weights: Dict[str, np.ndarray], quantization: str) -> Dict[str, Tuple[int, int]]:
        """Write tensor information and return offset info"""
        tensor_infos = {}
        data_offset = 0
        
        ggml_type, _ = self.quantization_map[quantization]
        
        for name, tensor in weights.items():
            # Write tensor name
            self._write_string(f, name)
            
            # Write number of dimensions
            n_dims = len(tensor.shape)
            f.write(struct.pack('<I', n_dims))
            
            # Write dimensions (in reverse order for GGML)
            for dim in reversed(tensor.shape):
                f.write(struct.pack('<Q', dim))
            
            # Write type
            f.write(struct.pack('<I', ggml_type))
            
            # Write offset
            f.write(struct.pack('<Q', data_offset))
            
            # Calculate size for this tensor
            if quantization == "f32":
                size = tensor.size * 4
            elif quantization == "f16":
                size = tensor.size * 2
            elif quantization == "q8_0":
                # Q8_0: 32 values -> 32 bytes + 2 bytes scale = 34 bytes per block
                size = (tensor.size // 32) * 34
                if tensor.size % 32 != 0:
                    size += 34  # Partial block
            elif quantization == "q4_0":
                # Q4_0: 32 values -> 16 bytes + 2 bytes scale = 18 bytes per block
                size = (tensor.size // 32) * 18
                if tensor.size % 32 != 0:
                    size += 18  # Partial block
            
            tensor_infos[name] = (data_offset, size)
            data_offset += size
            
        return tensor_infos
    
    def _write_tensor_data(self, f, weights: Dict[str, np.ndarray], tensor_infos: Dict, quantization: str):
        """Write quantized tensor data"""
        _, quantize_fn = self.quantization_map[quantization]
        
        total_tensors = len(weights)
        for i, (name, tensor) in enumerate(weights.items(), 1):
            print(f"\r  Quantizing tensors: {i}/{total_tensors}", end='', flush=True)
            
            # Quantize and write
            quantized_data = quantize_fn(tensor)
            f.write(quantized_data)
        
        print()  # New line after progress
    
    def quantize_f32(self, tensor: np.ndarray) -> bytes:
        """No quantization - keep as float32"""
        return tensor.astype(np.float32).tobytes()
    
    def quantize_f16(self, tensor: np.ndarray) -> bytes:
        """Quantize to float16"""
        return tensor.astype(np.float16).tobytes()
    
    def quantize_q8_0(self, tensor: np.ndarray) -> bytes:
        """Quantize to 8-bit (GGML Q8_0 format)"""
        result = bytearray()
        
        flat_tensor = tensor.flatten().astype(np.float32)
        block_size = 32
        
        for i in range(0, len(flat_tensor), block_size):
            block = flat_tensor[i:i+block_size]
            if len(block) < block_size:
                # Pad last block
                block = np.pad(block, (0, block_size - len(block)), constant_values=0)
            
            # Find scale (max absolute value)
            max_val = np.max(np.abs(block))
            scale = max_val / 127.0 if max_val > 0 else 1.0
            
            # Quantize
            quantized = np.clip(np.round(block / scale), -128, 127).astype(np.int8)
            
            # Write scale as float16
            result.extend(struct.pack('<e', scale))
            
            # Write quantized values
            result.extend(quantized.tobytes())
        
        return bytes(result)
    
    def quantize_q4_0(self, tensor: np.ndarray) -> bytes:
        """Quantize to 4-bit (GGML Q4_0 format)"""
        result = bytearray()
        
        flat_tensor = tensor.flatten().astype(np.float32)
        block_size = 32
        
        for i in range(0, len(flat_tensor), block_size):
            block = flat_tensor[i:i+block_size]
            if len(block) < block_size:
                block = np.pad(block, (0, block_size - len(block)), constant_values=0)
            
            # Find scale
            max_val = np.max(np.abs(block))
            scale = max_val / 7.0 if max_val > 0 else 1.0
            
            # Write scale as float16
            result.extend(struct.pack('<e', scale))
            
            # Quantize to 4-bit (stored as pairs in bytes)
            for j in range(0, block_size, 2):
                val1 = int(np.clip(np.round(block[j] / scale), -8, 7)) + 8
                val2 = int(np.clip(np.round(block[j+1] / scale), -8, 7)) + 8 if j+1 < block_size else 8
                
                # Pack two 4-bit values into one byte
                byte_val = (val1 & 0x0F) | ((val2 & 0x0F) << 4)
                result.append(byte_val)
        
        return bytes(result)


def main():
    parser = argparse.ArgumentParser(description="Convert Piper ONNX models to GGUF")
    parser.add_argument('--input', '-i', type=str, required=True, help='Input ONNX model path')
    parser.add_argument('--output', '-o', type=str, help='Output GGUF path (default: input.gguf)')
    parser.add_argument('--quantization', '-q', type=str, default='q8_0',
                        choices=['f32', 'f16', 'q8_0', 'q4_0'],
                        help='Quantization type (default: q8_0)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1
    
    # Check config exists
    config_path = Path(str(input_path) + ".json")
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix('.gguf')
        # Add quantization suffix
        if args.quantization != 'f32':
            output_path = output_path.with_stem(f"{output_path.stem}-{args.quantization}")
    
    # Convert
    converter = PiperToGGUFConverter()
    try:
        converter.convert(input_path, output_path, args.quantization)
        return 0
    except Exception as e:
        print(f"\nError during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())