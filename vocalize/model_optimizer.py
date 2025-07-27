"""
Model optimization utilities for Vocalize TTS.
Handles quantization of ONNX models.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from onnxruntime.quantization import preprocess
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    # Don't print warning here - it's printed too early

from .config import get_config


class ModelOptimizer:
    """Handles ONNX model optimization and quantization."""
    
    def __init__(self, cache_dir: Optional[Path] = None, model_manager=None):
        """Initialize model optimizer with cache directory."""
        if cache_dir is None:
            import platformdirs
            cache_base = platformdirs.user_cache_dir("vocalize", "Vocalize")
            cache_dir = Path(cache_base)
        
        self.cache_dir = Path(cache_dir)
        # Use the same directory structure as downloaded models
        self.models_dir = self.cache_dir / "models" / "models--direct_download" / "local"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Use provided ModelManager or create one
        if model_manager is None:
            from .model_manager import ModelManager
            self.model_manager = ModelManager()
        else:
            self.model_manager = model_manager
    
    def get_model_path(self, variant: str = "original") -> Optional[Path]:
        """Get path to a model variant."""
        # For original model, use ModelManager's path
        if variant == "original":
            # First check if model is cached
            if not self.model_manager.is_model_cached("kokoro"):
                return None
            model_path = self.model_manager.get_model_path("kokoro", "kokoro-v1.0.onnx")
            if model_path:
                return Path(model_path)
            return None
        
        # For optimized variants, use our own directory
        filenames = {
            "int8": "kokoro-v1.0-int8.onnx"
        }
        
        if variant not in filenames:
            return None
        
        path = self.models_dir / filenames[variant]
        
        return path if path.exists() else None
    
    def quantize(self, force: bool = False, per_channel: bool = True) -> bool:
        """
        Download pre-quantized INT8 version of the model.
        
        The pre-quantized model uses selective layer quantization for optimal quality.
        
        Args:
            force: Force redownload even if exists
            per_channel: Not used (kept for API compatibility)
            
        Returns:
            bool: True if successful, False otherwise
        """
        quantized_path = self.models_dir / "kokoro-v1.0-int8.onnx"
        
        if quantized_path.exists() and not force:
            print("Quantized model already exists. Use --force to redownload.")
            return True
        
        print("🎯 Downloading pre-quantized INT8 model (ConvInteger optimized)...")
        print("  This model uses selective layer quantization and ConvInteger ops for optimal CPU performance")
        
        try:
            # Import requests for downloading
            try:
                import requests
            except ImportError:
                print("Error: requests not available. Install with: pip install requests")
                return False
            
            # Pre-quantized INT8 model URL (using ConvInteger version for best CPU performance)
            int8_url = "https://github.com/taylorchu/kokoro-onnx/releases/download/v0.2.0/kokoro-quant-convinteger.onnx"
            
            print(f"  📥 Downloading from: {int8_url}")
            
            # Download with progress
            response = requests.get(int8_url, stream=True)
            response.raise_for_status()
            
            # Get file size if available
            total_size = int(response.headers.get('content-length', 0))
            if total_size > 0:
                print(f"  📦 File size: {total_size / (1024*1024):.1f} MB")
            
            # Write to file
            with open(quantized_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = downloaded / total_size * 100
                        print(f"\r  ⏳ Progress: {progress:.1f}%", end='', flush=True)
            
            if total_size > 0:
                print()  # New line after progress
            
            # Get file sizes for comparison
            original_path = self.get_model_path("original")
            if original_path:
                original_size = original_path.stat().st_size / (1024 * 1024)
                quantized_size = quantized_path.stat().st_size / (1024 * 1024)
                reduction = (1 - quantized_size / original_size) * 100
                
                print(f"✅ Pre-quantized INT8 model downloaded successfully")
                print(f"  Original: {original_path.name} ({original_size:.1f} MB)")
                print(f"  Quantized: {quantized_path.name} ({quantized_size:.1f} MB)")
                print(f"  Size reduction: {reduction:.1f}%")
                print(f"  Expected speedup: 2-4x on CPU")
            else:
                quantized_size = quantized_path.stat().st_size / (1024 * 1024)
                print(f"✅ Pre-quantized INT8 model downloaded successfully")
                print(f"  Quantized: {quantized_path.name} ({quantized_size:.1f} MB)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error downloading pre-quantized model: {e}")
            # Clean up partial download
            if quantized_path.exists():
                quantized_path.unlink()
            return False
    
    def _validate_quantized_model(self, original_path: Path, quantized_path: Path) -> bool:
        """Validate quantized model produces similar outputs."""
        try:
            # Create test input
            test_input_ids = np.array([[0] + list(range(1, 20)) + [0]], dtype=np.int64)
            test_style = np.random.randn(1, 256).astype(np.float32) * 0.1
            test_speed = np.array([1.0], dtype=np.float32)
            
            # Run original model
            original_session = ort.InferenceSession(str(original_path))
            
            # Get input names from original model
            original_inputs = {inp.name for inp in original_session.get_inputs()}
            print(f"  Original model inputs: {original_inputs}")
            
            # Prepare inputs based on what the model expects
            original_feed = {}
            if "input_ids" in original_inputs:
                original_feed["input_ids"] = test_input_ids
            if "tokens" in original_inputs:
                original_feed["tokens"] = test_input_ids
            if "style" in original_inputs:
                original_feed["style"] = test_style
            if "speed" in original_inputs:
                original_feed["speed"] = test_speed
                
            original_outputs = original_session.run(None, original_feed)
            
            # Run quantized model
            quantized_session = ort.InferenceSession(str(quantized_path))
            
            # Get input names from quantized model
            quantized_inputs = {inp.name for inp in quantized_session.get_inputs()}
            print(f"  Quantized model inputs: {quantized_inputs}")
            
            # Prepare inputs based on what the quantized model expects
            quantized_feed = {}
            if "input_ids" in quantized_inputs:
                quantized_feed["input_ids"] = test_input_ids
            if "tokens" in quantized_inputs:
                quantized_feed["tokens"] = test_input_ids
            if "style" in quantized_inputs:
                quantized_feed["style"] = test_style
            if "speed" in quantized_inputs:
                quantized_feed["speed"] = test_speed
                
            quantized_outputs = quantized_session.run(None, quantized_feed)
            
            # Compare outputs
            original_audio = original_outputs[0]
            quantized_audio = quantized_outputs[0]
            
            # Calculate metrics
            mse = np.mean((original_audio - quantized_audio) ** 2)
            max_diff = np.max(np.abs(original_audio - quantized_audio))
            
            print(f"  Quality metrics: MSE={mse:.6f}, Max diff={max_diff:.6f}")
            
            # Check if within acceptable range
            return mse < 0.001 and max_diff < 0.1
            
        except Exception as e:
            print(f"  Validation error: {e}")
            return False
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get status of all optimizations."""
        status = {
            "quantize": {
                "enabled": get_config().get('optimizations.quantization_enabled', False),
                "built": False,
                "path": None,
                "size_mb": None
            },
            "original": {
                "path": None,
                "size_mb": None
            }
        }
        
        # Check original model
        original_path = self.get_model_path("original")
        if original_path:
            status["original"]["path"] = str(original_path)
            status["original"]["size_mb"] = original_path.stat().st_size / (1024 * 1024)
        
        # Check quantized model
        quantized_path = self.get_model_path("int8")
        if quantized_path:
            status["quantize"]["built"] = True
            status["quantize"]["path"] = str(quantized_path)
            status["quantize"]["size_mb"] = quantized_path.stat().st_size / (1024 * 1024)
        
        return status
    
    def get_active_model_path(self) -> Path:
        """Get the path to the currently active model based on enabled optimizations."""
        config = get_config()
        quantize_enabled = config.get('optimizations.quantization_enabled', False)
        
        if quantize_enabled:
            path = self.get_model_path("int8")
            if path:
                return path
        
        # Default to original
        path = self.get_model_path("original")
        if path:
            return path
        
        raise FileNotFoundError("No model found. Download with: vocalize models download kokoro")