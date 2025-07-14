"""
Early environment setup for ONNX Runtime.

This module MUST be imported before any other imports to prevent
ONNX Runtime from initializing with default multi-threaded settings
that cause deadlocks on certain systems.
"""

import os
import sys
from pathlib import Path

# CRITICAL: Set thread limits BEFORE any library initialization
# This prevents deadlocks in ONNX Runtime on WSL and other systems
os.environ.update({
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1', 
    'NUMEXPR_NUM_THREADS': '1',
    'ORT_DISABLE_SPINNING': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'VECLIB_MAXIMUM_THREADS': '1',
    'BLIS_NUM_THREADS': '1',
})

# Set ONNX Runtime library path for load-dynamic feature
# This ensures the bundled libonnxruntime.so is used
vocalize_root = Path(__file__).parent.parent
onnx_lib_path = vocalize_root / "crates" / "vocalize-core" / "onnxruntime" / "libonnxruntime.so"

if onnx_lib_path.exists():
    os.environ['ORT_DYLIB_PATH'] = str(onnx_lib_path)
    # Also add to LD_LIBRARY_PATH for Linux systems
    if sys.platform.startswith('linux'):
        ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        onnx_dir = str(onnx_lib_path.parent)
        if onnx_dir not in ld_path:
            os.environ['LD_LIBRARY_PATH'] = f"{onnx_dir}:{ld_path}" if ld_path else onnx_dir

# Debug output if verbose mode is detected
if '--verbose' in sys.argv or os.environ.get('VOCALIZE_DEBUG'):
    print(f"🔧 Environment setup complete:")
    print(f"   OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")
    print(f"   ORT_DYLIB_PATH: {os.environ.get('ORT_DYLIB_PATH')}")
    print(f"   Threading: Single-threaded mode enabled")