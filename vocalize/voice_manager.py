"""
Voice management system for TTS models.

This module handles voice discovery, downloading, and loading for TTS models.
Each model has its own set of compatible voices that are downloaded as complete packages.
"""

import os
import json
import math
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

# Voice alias mapping for user-friendly names
VOICE_ALIASES = {
    # User-friendly names → Kokoro voice IDs
    "bella": "af_bella",
    "alloy": "af_alloy", 
    "sarah": "af_sarah",
    "adam": "am_adam",
    "echo": "am_echo",
    "nova": "af_nova",
    "heart": "af_heart",
    "jessica": "af_jessica",
    "nicole": "af_nicole",
    "river": "af_river",
    "sky": "af_sky",
    "eric": "am_eric",
    "liam": "am_liam",
    "michael": "am_michael",
    "onyx": "am_onyx",
    "fenrir": "am_fenrir",
    "puck": "am_puck",
    # Additional common aliases
    "female": "af_sarah",
    "male": "am_adam",
    "default": "af_sarah",
}

try:
    import tinynumpy as np  # Lightweight alternative to numpy for faster imports
except ImportError:
    import numpy as np  # Fallback to regular numpy

@dataclass
class VoiceInfo:
    """Information about a voice for TTS models."""
    id: str
    name: str
    gender: str
    language: str
    file_path: str

class VoiceManager:
    """Manages voice discovery, downloading, and loading for TTS models."""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.voice_cache_file = self.cache_dir / "voice_cache.json"
    
    def validate_style_vector(self, style_vector: List[float]) -> bool:
        """Validate style vector to prevent neural network instability."""
        if not style_vector or len(style_vector) != 256:
            return False
        
        # Check for NaN/Inf values (causes immediate model corruption)
        if any(not math.isfinite(x) for x in style_vector):
            print("⚠️ Style vector contains NaN/Inf values")
            return False
        
        # Check for extreme values (causes gradient explosion)
        if any(abs(x) > 10.0 for x in style_vector):
            print("⚠️ Style vector contains extreme values")
            return False
        
        # Check for all zeros (indicates failed loading)
        if all(abs(x) < 0.001 for x in style_vector):
            print("⚠️ Style vector appears to be all zeros")
            return False
        
        # Check for uniform random values (indicates fallback to random)
        mean_val = sum(style_vector) / len(style_vector)
        if abs(mean_val) < 0.01:  # Random [-1,1] should have mean ~0
            variance = sum((x - mean_val) ** 2 for x in style_vector) / len(style_vector)
            if variance > 0.8:  # High variance suggests random values
                print("⚠️ Style vector appears to be random values")
                return False
        
        return True
    
    def resolve_voice_alias(self, voice_id: str) -> str:
        """Resolve user-friendly voice names to actual Kokoro voice IDs."""
        # Try exact match first
        if voice_id.startswith(('af_', 'am_', 'bf_', 'bm_', 'ef_', 'em_', 'ff_', 'hf_', 'hm_', 'if_', 'im_', 'jf_', 'jm_', 'pf_', 'pm_', 'zf_', 'zm_')):
            return voice_id
        
        # Try alias mapping
        resolved = VOICE_ALIASES.get(voice_id.lower())
        if resolved:
            print(f"🔄 Resolved voice alias: '{voice_id}' → '{resolved}'")
            return resolved
        
        # Return original if no alias found
        return voice_id
    
    def _load_voice_cache(self) -> Dict:
        """Load voice cache from JSON file for fast access."""
        if self.voice_cache_file.exists():
            try:
                with open(self.voice_cache_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {}
    
    def _create_voice_cache_from_npz(self, model_id: str, npz_path: Path) -> Dict:
        """Create voice cache from NPZ file (slow operation)."""
        print(f"🔄 Creating voice cache from NPZ file...")
        # NPZ loading requires full numpy, not tinynumpy
        import numpy as np_full
        voices_data = np_full.load(str(npz_path))
        
        cache_data = {
            "kokoro": {
                "voices": [],
                "last_updated": str(npz_path.stat().st_mtime)
            }
        }
        
        for voice_id in sorted(voices_data.files):
            cache_data["kokoro"]["voices"].append({
                "id": voice_id,
                "name": self._parse_voice_name(voice_id),
                "gender": self._parse_gender(voice_id),
                "language": self._parse_language(voice_id),
                "file_path": str(npz_path)
            })
        
        # Save cache to file
        self.voice_cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.voice_cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        print(f"✅ Voice cache created with {len(cache_data['kokoro']['voices'])} voices")
        return cache_data
    
    def _discover_voices_from_cache(self, model_id: str) -> List[VoiceInfo]:
        """Fast voice discovery using cached data."""
        cache = self._load_voice_cache()
        
        if model_id in cache:
            voices = []
            for voice_data in cache[model_id]["voices"]:
                voices.append(VoiceInfo(
                    id=voice_data["id"],
                    name=voice_data["name"],
                    gender=voice_data["gender"],
                    language=voice_data["language"],
                    file_path=voice_data["file_path"]
                ))
            print(f"✅ Discovered {len(voices)} {model_id} voices from cache")
            return voices
        
        return []
    
    def discover_voices(self, model_id: str) -> List[VoiceInfo]:
        """Discover all voices for a model - fast cache first, then NPZ if needed."""
        
        # Try fast cache lookup first
        voices = self._discover_voices_from_cache(model_id)
        if voices:
            return voices
        
        # Fallback to slow NPZ loading and cache creation
        if model_id == "kokoro":
            return self._discover_kokoro_voices_slow()
        
        # For other models, use the original file-based discovery
        return self._discover_voices_from_files(model_id)
    
    def _discover_kokoro_voices_slow(self) -> List[VoiceInfo]:
        """Slow path: Load Kokoro voices from NPZ and create cache."""
        # Look for Kokoro NPZ file
        model_cache = self.cache_dir / "models--direct_download" / "local"
        npz_path = model_cache / "voices-v1.0.bin"
        
        if npz_path.exists():
            # Create cache from NPZ file
            cache_data = self._create_voice_cache_from_npz("kokoro", npz_path)
            # Return voices from the cache we just created
            return self._discover_voices_from_cache("kokoro")
        
        return []
    
    def _discover_voices_from_files(self, model_id: str) -> List[VoiceInfo]:
        """Original file-based voice discovery for non-Kokoro models."""
        voices = []
        
        # Calculate model cache directory based on model_id
        model_cache = self.cache_dir / f"models--{model_id.replace('/', '--')}"
        
        voices_dirs = [
            model_cache / "local" / "voices",
            model_cache / "local",
        ]
        
        # Also check in snapshots directory
        snapshots_dir = model_cache / "snapshots"
        if snapshots_dir.exists():
            for snapshot in snapshots_dir.iterdir():
                if snapshot.is_dir():
                    voices_dirs.extend([
                        snapshot / "voices",
                        snapshot,
                    ])
        
        voice_files_found = set()
        
        for voices_dir in voices_dirs:
            if not voices_dir.exists():
                continue
                
            for voice_file in voices_dir.glob("*.bin"):
                # Avoid duplicates
                if voice_file.name in voice_files_found:
                    continue
                voice_files_found.add(voice_file.name)
                
                # Extract voice info from filename
                voice_id = voice_file.stem
                voices.append(VoiceInfo(
                    id=voice_id,
                    name=self._parse_voice_name(voice_id),
                    gender=self._parse_gender(voice_id),
                    language=self._parse_language(voice_id),
                    file_path=str(voice_file)
                ))
        
        return voices
    
    def load_voice_embedding(self, voice_file: str) -> List[float]:
        """Load voice embedding from binary file"""
        if not os.path.exists(voice_file):
            raise FileNotFoundError(f"Voice file not found: {voice_file}")
        
        try:
            # Load binary voice embedding
            voice_data = np.fromfile(voice_file, dtype=np.float32)
            if len(voice_data) == 0:
                raise ValueError("Voice embedding file is empty")
            return voice_data.tolist()
        except Exception as e:
            raise RuntimeError(f"Failed to load voice embedding: {e}")
    
    def get_default_voice(self, model_id: str) -> Optional[str]:
        """Get default voice for a model"""
        voices = self.discover_voices(model_id)
        if not voices:
            return None
        
        # Prefer female voices as default
        for voice in voices:
            if 'female' in voice.gender.lower() or 'f' in voice.id.lower():
                return voice.id
        
        # Return first available voice
        return voices[0].id
    
    def _parse_voice_name(self, voice_id: str) -> str:
        """Parse human-readable voice name from ID"""
        # Handle common voice ID patterns
        name_mappings = {
            'af_alloy': 'Alloy (Female)',
            'af_bella': 'Bella (Female)', 
            'af_nova': 'Nova (Female)',
            'af_aoede': 'Aoede (Female)',
            'af_heart': 'Heart (Female)',
            'af_jessica': 'Jessica (Female)',
            'af_kore': 'Kore (Female)',
            'af_nicole': 'Nicole (Female)',
            'af_river': 'River (Female)',
            'af_sarah': 'Sarah (Female)',
            'af_sky': 'Sky (Female)',
            'am_adam': 'Adam (Male)',
            'am_echo': 'Echo (Male)',
            'am_eric': 'Eric (Male)',
            'am_fenrir': 'Fenrir (Male)',
            'am_liam': 'Liam (Male)',
            'am_michael': 'Michael (Male)',
            'am_onyx': 'Onyx (Male)',
            'am_puck': 'Puck (Male)',
        }
        
        if voice_id in name_mappings:
            return name_mappings[voice_id]
        
        # Generic parsing
        return voice_id.replace('_', ' ').title()
    
    def _parse_gender(self, voice_id: str) -> str:
        """Parse gender from voice ID"""
        voice_lower = voice_id.lower()
        if voice_lower.startswith('af_') or 'female' in voice_lower:
            return 'female'
        elif voice_lower.startswith('am_') or 'male' in voice_lower:
            return 'male'
        elif voice_lower.startswith('bf_'):
            return 'female'
        elif voice_lower.startswith('bm_'):
            return 'male'
        return 'neutral'
    
    def _parse_language(self, voice_id: str) -> str:
        """Parse language from voice ID"""
        voice_lower = voice_id.lower()
        if 'en' in voice_lower or voice_id.startswith(('af_', 'am_')):
            return 'english'
        elif 'zh' in voice_lower or 'chinese' in voice_lower:
            return 'chinese'
        elif 'ja' in voice_lower or 'japanese' in voice_lower:
            return 'japanese'
        return 'english'  # Default to English for Kokoro
    
    def _discover_kokoro_voices(self) -> List[VoiceInfo]:
        """Discover Kokoro voices from NPZ file"""
        voices = []
        
        # Check multiple possible locations for the voices NPZ file
        possible_paths = [
            self.cache_dir / "models--direct_download" / "local" / "voices-v1.0.bin",
            self.cache_dir / "models--onnx-community--Kokoro-82M-v1.0-ONNX" / "local" / "voices-v1.0.bin",
            self.cache_dir / "models--hexgrad--Kokoro-82M" / "local" / "voices-v1.0.bin",
        ]
        
        voices_file = None
        for path in possible_paths:
            if path.exists():
                voices_file = path
                break
        
        if not voices_file:
            print("⚠️  No Kokoro voices NPZ file found")
            return voices
        
        try:
            # Load voices from NPZ file - requires full numpy
            import numpy as np_full
            voices_data = np_full.load(str(voices_file))
            
            for voice_id in sorted(voices_data.files):
                voices.append(VoiceInfo(
                    id=voice_id,
                    name=self._parse_voice_name(voice_id),
                    gender=self._parse_gender(voice_id),
                    language=self._parse_language(voice_id),
                    file_path=str(voices_file)  # Point to the NPZ file
                ))
            
            print(f"✅ Discovered {len(voices)} Kokoro voices from NPZ file")
            
        except Exception as e:
            print(f"❌ Failed to load Kokoro voices from NPZ file: {e}")
        
        return voices