"""
Reliable model management using Python huggingface_hub.

This module handles all model downloads and caching using the mature Python
huggingface_hub client, eliminating the unreliable Rust hf-hub hanging issues.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import platformdirs

# Configure HuggingFace Hub for cross-platform compatibility
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"

# Lazy import expensive modules only when needed for faster CLI startup
_HAS_HF_HUB = None
_HAS_REQUESTS = None

def _import_huggingface_hub():
    """Lazy import of huggingface_hub for faster CLI startup."""
    global _HAS_HF_HUB
    if _HAS_HF_HUB is None:
        try:
            from huggingface_hub import hf_hub_download, HfApi
            _HAS_HF_HUB = True
            return hf_hub_download, HfApi
        except ImportError:
            print("Warning: huggingface_hub not available. Install with: uv add huggingface-hub")
            _HAS_HF_HUB = False
            return None, None
    elif _HAS_HF_HUB:
        from huggingface_hub import hf_hub_download, HfApi
        return hf_hub_download, HfApi
    else:
        return None, None


def _import_requests():
    """Lazy import of requests for faster CLI startup."""
    global _HAS_REQUESTS
    if _HAS_REQUESTS is None:
        try:
            import requests
            _HAS_REQUESTS = True
            return requests
        except ImportError:
            print("Warning: requests not available. Install with: uv add requests")
            _HAS_REQUESTS = False
            return None
    elif _HAS_REQUESTS:
        import requests
        return requests
    else:
        return None


def _import_shutil():
    """Lazy import of shutil for faster CLI startup."""
    import shutil
    return shutil


@dataclass
class ModelInfo:
    """Information about a neural TTS model."""
    id: str
    name: str
    repo_id: str
    files: List[str]
    size_mb: int
    description: str
    
    def __post_init__(self):
        """Validate model info after creation."""
        if not self.id or not self.repo_id or not self.files:
            raise ValueError(f"Invalid model info: {self}")


class ModelManager:
    """Manages neural TTS model downloads and caching."""
    
    # Model registry with 2025 working repository information
    MODELS = {
        "kokoro": ModelInfo(
            id="kokoro",
            name="Kokoro TTS",
            repo_id="direct_download",  # Special flag for direct GitHub downloads
            files=["kokoro-v1.0.onnx", "voices-v1.0.bin"],  # 2025 unified model files
            size_mb=410,  # Combined model + voices size
            description="2025 optimized neural TTS model (82M parameters)"
        ),
    }
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize model manager with cache directory."""
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            # Use cross-platform cache directory that matches Rust implementation
            # This creates paths like:
            # Windows: C:\Users\{user}\AppData\Local\Vocalize\vocalize\cache\models
            # macOS: /Users/{user}/Library/Caches/ai.Vocalize.vocalize/models  
            # Linux: /home/{user}/.cache/vocalize/models
            cache_base = platformdirs.user_cache_dir("vocalize", "Vocalize")
            self.cache_dir = Path(cache_base) / "models"
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # HuggingFace API will be initialized lazily when needed
        self.hf_api = None
        
    def is_model_cached(self, model_id: str) -> bool:
        """Check if a model is already downloaded and cached."""
        if model_id not in self.MODELS:
            return False
            
        model_info = self.MODELS[model_id]
        
        # Handle 2025 direct download models
        if model_info.repo_id == "direct_download":
            model_local_dir = self.cache_dir / "models--direct_download" / "local"
        else:
            # Check in local directory first (no symlinks)
            model_local_dir = self.cache_dir / f"models--{model_info.repo_id.replace('/', '--')}" / "local"
        
        # Check if all required files exist in local directory
        if model_local_dir.exists():
            all_files_exist = all(
                (model_local_dir / filename).exists() and (model_local_dir / filename).is_file()
                for filename in model_info.files
            )
            if all_files_exist:
                return True
        
        # Fallback: check in snapshots directory (legacy symlink structure)
        model_cache_dir = self.cache_dir / f"models--{model_info.repo_id.replace('/', '--')}"
        if model_cache_dir.exists():
            for snapshot_dir in model_cache_dir.glob("snapshots/*"):
                if all((snapshot_dir / filename).exists() for filename in model_info.files):
                    return True
        
        return False
    
    def get_model_path(self, model_id: str, filename: str) -> Optional[Path]:
        """Get the local path to a cached model file."""
        if not self.is_model_cached(model_id):
            return None
            
        model_info = self.MODELS[model_id]
        
        # Handle 2025 direct download models
        if model_info.repo_id == "direct_download":
            model_local_dir = self.cache_dir / "models--direct_download" / "local"
        else:
            # Check in local directory first (no symlinks)
            model_local_dir = self.cache_dir / f"models--{model_info.repo_id.replace('/', '--')}" / "local"
        local_file_path = model_local_dir / filename
        if local_file_path.exists() and local_file_path.is_file():
            return local_file_path
        
        # Fallback: check in snapshots directory (legacy symlink structure)
        model_cache_dir = self.cache_dir / f"models--{model_info.repo_id.replace('/', '--')}"
        for snapshot_dir in model_cache_dir.glob("snapshots/*"):
            file_path = snapshot_dir / filename
            if file_path.exists():
                return file_path
        
        return None
    
    def download_model(self, model_id: str, force: bool = False) -> bool:
        """
        Download a model using either HuggingFace hub or direct download for 2025 models.
        
        Args:
            model_id: ID of the model to download
            force: Force redownload even if cached
            
        Returns:
            True if download successful, False otherwise
        """
        if model_id not in self.MODELS:
            print(f"Error: Unknown model '{model_id}'. Available: {list(self.MODELS.keys())}")
            return False
        
        model_info = self.MODELS[model_id]
        
        # Check if already cached
        if not force and self.is_model_cached(model_id):
            print(f"✓ Model '{model_id}' already cached")
            return True
        
        print(f"📥 Downloading {model_info.name} ({model_info.size_mb}MB)")
        
        try:
            # Handle 2025 direct download models
            if model_info.repo_id == "direct_download" and model_id == "kokoro":
                return self._download_kokoro_2025(force)
            
            # Handle traditional HuggingFace models
            hf_hub_download, HfApi = _import_huggingface_hub()
            if not hf_hub_download:
                print("Error: huggingface_hub not available. Install with: uv add huggingface-hub")
                return False
            
            # Download each required file from HuggingFace
            for filename in model_info.files:
                print(f"  📄 Downloading {filename}...")
                
                # Create a local directory for this model to avoid symlinks
                model_local_dir = self.cache_dir / f"models--{model_info.repo_id.replace('/', '--')}" / "local"
                model_local_dir.mkdir(parents=True, exist_ok=True)
                
                # Use hf_hub_download with local_dir to avoid symlinks completely
                local_path = hf_hub_download(
                    repo_id=model_info.repo_id,
                    filename=filename,
                    local_dir=str(model_local_dir),
                    local_dir_use_symlinks=False,  # Disable symlinks for cross-platform compatibility
                    resume_download=True,  # Enable resumable downloads
                    force_download=force,  # Force redownload if requested
                )
                
                print(f"  ✓ Downloaded to {local_path}")
            
            print(f"✅ Successfully downloaded {model_info.name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to download {model_info.name}: {e}")
            return False
    
    def _download_kokoro_2025(self, force: bool = False) -> bool:
        """Download 2025 Kokoro model files directly from GitHub releases."""
        requests = _import_requests()
        if not requests:
            print("Error: requests not available. Install with: uv add requests")
            return False
        
        # 2025 working model URLs
        model_urls = {
            "kokoro-v1.0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
            "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
        }
        
        # Create local directory
        model_local_dir = self.cache_dir / "models--direct_download" / "local"
        model_local_dir.mkdir(parents=True, exist_ok=True)
        
        for filename, url in model_urls.items():
            local_file = model_local_dir / filename
            
            # Skip if exists and not forcing
            if local_file.exists() and not force:
                print(f"  ✓ {filename} already exists")
                continue
            
            print(f"  📄 Downloading {filename} from GitHub releases...")
            
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                with open(local_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                print(f"  ✓ Downloaded {filename} ({local_file.stat().st_size // (1024*1024)}MB)")
                
            except Exception as e:
                print(f"  ❌ Failed to download {filename}: {e}")
                return False
        
        print("✅ Successfully downloaded 2025 Kokoro model files")
        return True
    
    def list_available_models(self) -> List[str]:
        """List all available models."""
        return list(self.MODELS.keys())
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get information about a model."""
        return self.MODELS.get(model_id)
    
    def clear_cache(self, model_id: Optional[str] = None) -> bool:
        """
        Clear model cache.
        
        Args:
            model_id: Specific model to clear, or None to clear all
            
        Returns:
            True if successful
        """
        try:
            if model_id:
                if model_id not in self.MODELS:
                    print(f"Error: Unknown model '{model_id}'")
                    return False
                    
                model_info = self.MODELS[model_id]
                model_cache_dir = self.cache_dir / f"models--{model_info.repo_id.replace('/', '--')}"
                
                if model_cache_dir.exists():
                    shutil = _import_shutil()
                    shutil.rmtree(model_cache_dir)
                    print(f"✅ Cleared cache for {model_info.name}")
                else:
                    print(f"ℹ️  No cache found for {model_info.name}")
            else:
                # Clear entire cache
                if self.cache_dir.exists():
                    shutil = _import_shutil()
                    shutil.rmtree(self.cache_dir)
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    print("✅ Cleared all model cache")
                    
            return True
            
        except Exception as e:
            print(f"❌ Failed to clear cache: {e}")
            return False
    
    def get_cache_size(self) -> str:
        """Get total cache size in human-readable format."""
        if not self.cache_dir.exists():
            return "0 B"
            
        total_size = 0
        for file_path in self.cache_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        # Convert to human readable
        for unit in ['B', 'KB', 'MB', 'GB']:
            if total_size < 1024.0:
                return f"{total_size:.1f} {unit}"
            total_size /= 1024.0
        return f"{total_size:.1f} TB"

    def download_model_with_voices(self, model_id: str, force: bool = False) -> bool:
        """
        Download complete model package including all voices.
        
        Args:
            model_id: ID of the model to download
            force: Force redownload even if cached
            
        Returns:
            True if download successful, False otherwise
        """
        hf_hub_download, HfApi = _import_huggingface_hub()
        if not hf_hub_download:
            print("Error: huggingface_hub not available. Install with: uv add huggingface-hub")
            return False
            
        if model_id not in self.MODELS:
            print(f"Error: Unknown model '{model_id}'. Available: {list(self.MODELS.keys())}")
            return False
        
        model_info = self.MODELS[model_id]
        
        # Step 1: Download base model
        base_success = self.download_model(model_id, force)
        if not base_success:
            return False
        
        # Step 2: Download all voices for the model
        try:
            if model_id == "kokoro":
                # Download all Kokoro voices based on research
                voice_files = self._get_kokoro_voice_files()
                
                for voice_file in voice_files:
                    print(f"  📄 Downloading voice: {voice_file}")
                    
                    # Create local directory structure
                    model_local_dir = self.cache_dir / f"models--{model_info.repo_id.replace('/', '--')}" / "local"
                    
                    try:
                        local_path = hf_hub_download(
                            repo_id=model_info.repo_id,
                            filename=voice_file,
                            local_dir=str(model_local_dir),
                            local_dir_use_symlinks=False,
                            resume_download=True,
                            force_download=force,
                        )
                        print(f"  ✓ Downloaded: {local_path}")
                    except Exception as e:
                        # Some voices might not exist - continue with others
                        print(f"  ⚠️  Voice {voice_file} not available: {e}")
                        continue
            
            print(f"✅ Complete model package downloaded: {model_info.name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to download voices: {e}")
            return False
    
    def _get_kokoro_voice_files(self) -> List[str]:
        """Get all Kokoro voice files based on research"""
        voice_files = [
            # American English (11F, 9M)
            "voices/af_heart.bin", "voices/af_alloy.bin", "voices/af_aoede.bin", 
            "voices/af_bella.bin", "voices/af_jessica.bin", "voices/af_kore.bin", 
            "voices/af_nicole.bin", "voices/af_nova.bin", "voices/af_river.bin", 
            "voices/af_sarah.bin", "voices/af_sky.bin",
            "voices/am_adam.bin", "voices/am_echo.bin", "voices/am_eric.bin", 
            "voices/am_fenrir.bin", "voices/am_liam.bin", "voices/am_michael.bin", 
            "voices/am_onyx.bin", "voices/am_puck.bin", "voices/am_santa.bin",
            
            # British English (4F, 4M)
            "voices/bf_alice.bin", "voices/bf_emma.bin", "voices/bf_isabella.bin", 
            "voices/bf_lily.bin", "voices/bm_daniel.bin", "voices/bm_fable.bin", 
            "voices/bm_george.bin", "voices/bm_lewis.bin",
            
            # Japanese (4F, 1M)
            "voices/jf_alpha.bin", "voices/jf_gongitsune.bin", "voices/jf_nezumi.bin", 
            "voices/jf_tebukuro.bin", "voices/jm_kumo.bin",
            
            # Mandarin Chinese (4F, 4M)
            "voices/zf_xiaobei.bin", "voices/zf_xiaoni.bin", "voices/zf_xiaoxiao.bin", 
            "voices/zf_xiaoyi.bin", "voices/zm_yunjian.bin", "voices/zm_yunxi.bin", 
            "voices/zm_yunxia.bin", "voices/zm_yunyang.bin",
            
            # Spanish (1F, 2M)
            "voices/ef_dora.bin", "voices/em_alex.bin", "voices/em_santa.bin",
            
            # French (1F)
            "voices/ff_siwis.bin",
            
            # Hindi (2F, 2M)
            "voices/hf_alpha.bin", "voices/hf_beta.bin", "voices/hm_omega.bin", 
            "voices/hm_psi.bin",
            
            # Italian (1F, 1M)
            "voices/if_sara.bin", "voices/im_nicola.bin",
            
            # Brazilian Portuguese (1F, 2M)
            "voices/pf_dora.bin", "voices/pm_alex.bin", "voices/pm_santa.bin"
        ]
        return voice_files


def ensure_model_available(model_id: str, cache_dir: Optional[str] = None) -> bool:
    """
    Ensure a model is available for use, downloading if necessary.
    
    This is the main entry point for the CLI to ensure models are ready.
    
    Args:
        model_id: Model to ensure is available
        cache_dir: Optional cache directory override
        
    Returns:
        True if model is available, False otherwise
    """
    manager = ModelManager(cache_dir)
    
    # Check if already cached
    if manager.is_model_cached(model_id):
        return True
    
    # Download if not cached
    print(f"Model '{model_id}' not found in cache. Downloading...")
    return manager.download_model(model_id)


class KokoroPhonemeProcessor:
    """Handles text-to-phoneme conversion and tokenization for Kokoro TTS."""
    
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.tokenizer = None
        self.phoneme_config = None
        self.voices = None
        self._load_phoneme_config()
        self._load_voices()
        
    def _load_phoneme_config(self):
        """Load phoneme-to-token mapping configuration."""
        config_file = self.model_dir / "phoneme_config.json"
        if config_file.exists():
            import json
            with open(config_file, 'r') as f:
                self.phoneme_config = json.load(f)
                print("✅ Loaded phoneme configuration")
    
    def _load_voices(self):
        """Load voice embeddings from NPZ file."""
        voices_file = self.model_dir / "voices-v1.0.bin"
        if voices_file.exists():
            try:
                # NPZ loading requires full numpy, not tinynumpy
                import numpy as np
                self.voices = np.load(str(voices_file))
                print(f"✅ Loaded {len(self.voices.files)} voices from NPZ file")
                # List available voices
                print(f"   Available voices: {', '.join(sorted(self.voices.files)[:10])}...")
            except Exception as e:
                print(f"⚠️  Failed to load voices: {e}")
                self.voices = None
        else:
            print(f"⚠️  Voices file not found: {voices_file}")
            self.voices = None
        
    def setup_tokenizer(self):
        """Setup ttstokenizer for phoneme processing."""
        try:
            # Try to import ttstokenizer
            from ttstokenizer import IPATokenizer
            self.tokenizer = IPATokenizer()
            print("✅ ttstokenizer loaded successfully")
            return True
        except ImportError:
            print("⚠️  ttstokenizer not available. Install with: pip install ttstokenizer")
            return False
    
    def process_text(self, text: str, voice_id: str = "af_alloy") -> dict:
        """
        Convert text to tokens ready for Kokoro ONNX inference.
        
        Returns:
            dict with 'input_ids', 'style', 'speed' for ONNX model
        """
        # For testing: create mock tokenization until ttstokenizer is properly installed
        if not self.tokenizer:
            if not self.setup_tokenizer():
                print("⚠️  Using mock tokenization for testing (ttstokenizer not available)")
                # Mock tokenization: convert text to character-based tokens
                char_tokens = [ord(c) % 256 for c in text.lower()]  # Simple char-to-token mapping
                input_ids = [0] + char_tokens[:510] + [0]  # Add padding, ensure max 512
                
                # Load real voice embedding or fall back to random
                style_vector = self._get_voice_embedding(voice_id)
                
                return {
                    "input_ids": input_ids,
                    "style": style_vector,  # Already a list
                    "speed": 1.0,
                    "voice_id": voice_id
                }
        
        # Real tokenization with ttstokenizer
        tokens = self.tokenizer(text)
        # ttstokenizer returns numpy array of token IDs
        if hasattr(tokens, 'tolist'):
            # Convert numpy array to list and add padding
            token_list = tokens.tolist()[:510]  # Limit length
            input_ids = [0] + token_list + [0]  # Add padding tokens
        else:
            # Fallback if unexpected format
            input_ids = [0] + list(tokens)[:510] + [0]
        
        # Ensure max length constraint
        if len(input_ids) > 512:
            input_ids = input_ids[:512]
        
        # Load real voice embedding from NPZ file
        style_vector = self._get_voice_embedding(voice_id)
        
        return {
            "input_ids": input_ids,
            "style": style_vector,
            "speed": 1.0,
            "voice_id": voice_id
        }
    
    def _get_voice_embedding(self, voice_id: str) -> list:
        """Get voice embedding from loaded NPZ file with safe fallback."""
        # Import voice alias resolution
        from .voice_manager import VOICE_ALIASES
        
        # Resolve alias (e.g., "bella" → "af_bella")
        resolved_voice_id = VOICE_ALIASES.get(voice_id.lower(), voice_id)
        
        # Try exact match first
        if self.voices is not None and resolved_voice_id in self.voices.files:
            # Get the voice array (shape: 510x1x256)
            voice_array = self.voices[resolved_voice_id]
            # Extract the first frame's style vector (256 dimensions)
            style_vector = voice_array[0, 0, :].tolist()
            print(f"✅ Loaded voice embedding for '{resolved_voice_id}' (range: [{min(style_vector):.3f}, {max(style_vector):.3f}])")
            return style_vector
        
        # Try original voice_id if alias didn't work
        if self.voices is not None and voice_id in self.voices.files:
            voice_array = self.voices[voice_id]
            style_vector = voice_array[0, 0, :].tolist()
            print(f"✅ Loaded voice embedding for '{voice_id}' (range: [{min(style_vector):.3f}, {max(style_vector):.3f}])")
            return style_vector
        
        # Voice not found - show warning and use default
        print(f"⚠️  Voice '{voice_id}' not found, using default 'af_alloy'")
        
        # SAFE FALLBACK: Use default voice instead of random values
        default_voices = ["af_alloy", "af_bella", "af_sarah"]  # Known good voices
        for default_voice in default_voices:
            if self.voices is not None and default_voice in self.voices.files:
                voice_array = self.voices[default_voice]
                style_vector = voice_array[0, 0, :].tolist()
                print(f"✅ Loaded fallback voice embedding '{default_voice}' (range: [{min(style_vector):.3f}, {max(style_vector):.3f}])")
                return style_vector
        
        # Final emergency fallback: Create safe neutral vector
        print(f"❌ No voices available in NPZ file, using neutral embedding")
        # Use small values centered around 0 (safe for neural networks)
        import random
        random.seed(42)  # Deterministic for consistency
        return [random.gauss(0, 0.1) for _ in range(256)]  # Small gaussian noise around 0


if __name__ == "__main__":
    # Simple CLI for testing
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python model_manager.py <command> [args]")
        print("Commands:")
        print("  list                    - List available models")
        print("  download <model_id>     - Download a model")
        print("  status <model_id>       - Check if model is cached")
        print("  clear [model_id]        - Clear cache")
        print("  size                    - Show cache size")
        sys.exit(1)
    
    manager = ModelManager()
    command = sys.argv[1]
    
    if command == "list":
        print("Available models:")
        for model_id in manager.list_available_models():
            info = manager.get_model_info(model_id)
            status = "✓ cached" if manager.is_model_cached(model_id) else "○ not cached"
            print(f"  {model_id:<12} - {info.name} ({info.size_mb}MB) {status}")
            
    elif command == "download" and len(sys.argv) >= 3:
        model_id = sys.argv[2]
        success = manager.download_model(model_id)
        sys.exit(0 if success else 1)
        
    elif command == "status" and len(sys.argv) >= 3:
        model_id = sys.argv[2]
        if manager.is_model_cached(model_id):
            print(f"✓ Model '{model_id}' is cached")
        else:
            print(f"○ Model '{model_id}' is not cached")
            
    elif command == "clear":
        model_id = sys.argv[2] if len(sys.argv) >= 3 else None
        manager.clear_cache(model_id)
        
    elif command == "size":
        print(f"Cache size: {manager.get_cache_size()}")
        
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)