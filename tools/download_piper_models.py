#!/usr/bin/env python3
"""
Download and convert Piper TTS models to GGUF format
Following Vocalize DEVELOPMENT.md build patterns
"""

import os
import json
import requests
import tarfile
import struct
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import argparse
import hashlib

class PiperModelDownloader:
    """Download Piper models and prepare for GGUF conversion"""
    
    # Piper model registry - using models with permissive licenses
    PIPER_MODELS = {
        "en_US-amy-medium": {
            "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx",
            "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json",
            "size_mb": 63,
            "quality": "medium",
            "language": "en-US",
            "license": "MIT",
            "description": "American English female voice, medium quality"
        },
        "en_US-amy-low": {
            "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/low/en_US-amy-low.onnx",
            "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/low/en_US-amy-low.onnx.json",
            "size_mb": 30,
            "quality": "low",
            "language": "en-US",
            "license": "MIT",
            "description": "American English female voice, low quality (fastest)"
        },
        "en_US-danny-low": {
            "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/danny/low/en_US-danny-low.onnx",
            "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/danny/low/en_US-danny-low.onnx.json",
            "size_mb": 30,
            "quality": "low",
            "language": "en-US",
            "license": "MIT",
            "description": "American English male voice, low quality"
        },
        "en_GB-alan-low": {
            "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/low/en_GB-alan-low.onnx",
            "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/low/en_GB-alan-low.onnx.json",
            "size_mb": 30,
            "quality": "low",
            "language": "en-GB",
            "license": "MIT",
            "description": "British English male voice, low quality"
        }
    }
    
    def __init__(self, cache_dir: Path = None):
        if cache_dir is None:
            # Use same cache structure as DEVELOPMENT.md
            self.cache_dir = Path.home() / ".cache" / "vocalize" / "models"
        else:
            self.cache_dir = Path(cache_dir)
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using cache directory: {self.cache_dir}")
        
    def list_models(self) -> None:
        """List all available models"""
        print("\nAvailable Piper Models:")
        print("-" * 80)
        print(f"{'Model ID':<25} {'Quality':<10} {'Size':<10} {'Language':<10} {'License':<10}")
        print("-" * 80)
        
        for model_id, info in self.PIPER_MODELS.items():
            print(f"{model_id:<25} {info['quality']:<10} {info['size_mb']:<10}MB {info['language']:<10} {info['license']:<10}")
            print(f"  {info['description']}")
        print()
    
    def download_model(self, model_id: str, force: bool = False) -> Path:
        """Download Piper ONNX model and config"""
        if model_id not in self.PIPER_MODELS:
            available = ", ".join(self.PIPER_MODELS.keys())
            raise ValueError(f"Unknown model: {model_id}. Available: {available}")
            
        model_info = self.PIPER_MODELS[model_id]
        model_path = self.cache_dir / f"{model_id}.onnx"
        config_path = self.cache_dir / f"{model_id}.onnx.json"
        
        # Download model
        if model_path.exists() and not force:
            print(f"✓ Model {model_id} already cached at {model_path}")
        else:
            print(f"📥 Downloading {model_id} ({model_info['size_mb']}MB)...")
            self._download_file(model_info['url'], model_path)
            print(f"✓ Downloaded model to {model_path}")
            
        # Download config
        if config_path.exists() and not force:
            print(f"✓ Config already cached at {config_path}")
        else:
            print(f"📥 Downloading {model_id} config...")
            self._download_file(model_info['config_url'], config_path)
            print(f"✓ Downloaded config to {config_path}")
            
        # Verify download
        if self._verify_model(model_path, config_path):
            print(f"✅ Model {model_id} ready for conversion")
        else:
            raise ValueError(f"Model verification failed for {model_id}")
            
        return model_path
    
    def _download_file(self, url: str, path: Path):
        """Download with progress bar"""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='', flush=True)
        print()  # New line after progress
    
    def _verify_model(self, model_path: Path, config_path: Path) -> bool:
        """Verify downloaded model files"""
        # Check files exist
        if not model_path.exists() or not config_path.exists():
            return False
            
        # Check model file size (should be > 10MB at least)
        if model_path.stat().st_size < 10 * 1024 * 1024:
            print(f"⚠️  Model file seems too small: {model_path.stat().st_size} bytes")
            return False
            
        # Try to load config
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Verify it has expected fields
                required_fields = ["audio", "espeak", "inference"]
                for field in required_fields:
                    if field not in config:
                        print(f"⚠️  Config missing required field: {field}")
                        return False
        except Exception as e:
            print(f"⚠️  Failed to load config: {e}")
            return False
            
        return True
    
    def get_model_info(self, model_id: str) -> Dict:
        """Get information about a model"""
        if model_id not in self.PIPER_MODELS:
            raise ValueError(f"Unknown model: {model_id}")
        return self.PIPER_MODELS[model_id]
    
    def download_all_models(self, quality: Optional[str] = None) -> None:
        """Download all models or models of specific quality"""
        models_to_download = []
        
        for model_id, info in self.PIPER_MODELS.items():
            if quality is None or info['quality'] == quality:
                models_to_download.append(model_id)
        
        print(f"Will download {len(models_to_download)} models")
        
        for i, model_id in enumerate(models_to_download, 1):
            print(f"\n[{i}/{len(models_to_download)}] Processing {model_id}")
            try:
                self.download_model(model_id)
            except Exception as e:
                print(f"❌ Failed to download {model_id}: {e}")
                continue


def main():
    parser = argparse.ArgumentParser(description="Download Piper TTS models")
    parser.add_argument('--model', type=str, help='Model ID to download')
    parser.add_argument('--list', action='store_true', help='List available models')
    parser.add_argument('--all', action='store_true', help='Download all models')
    parser.add_argument('--quality', type=str, choices=['low', 'medium', 'high'], 
                        help='Download all models of specific quality')
    parser.add_argument('--cache-dir', type=str, help='Cache directory')
    parser.add_argument('--force', action='store_true', help='Force re-download')
    
    args = parser.parse_args()
    
    downloader = PiperModelDownloader(
        cache_dir=Path(args.cache_dir) if args.cache_dir else None
    )
    
    if args.list:
        downloader.list_models()
    elif args.all:
        downloader.download_all_models(quality=args.quality)
    elif args.model:
        downloader.download_model(args.model, force=args.force)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python download_piper_models.py --list")
        print("  python download_piper_models.py --model en_US-amy-medium")
        print("  python download_piper_models.py --all --quality low")


if __name__ == "__main__":
    main()