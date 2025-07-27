"""
Configuration management for Vocalize using TOML files.
Follows Poetry's approach for cross-platform configuration storage.
"""

import toml
import platformdirs
from pathlib import Path
from typing import Dict, Any, Optional


class VocalizeConfig:
    """Manages Vocalize configuration with TOML files."""
    
    def __init__(self):
        # Use platformdirs for cross-platform configuration directory
        self.config_dir = Path(platformdirs.user_config_dir("vocalize", "Vocalize"))
        self.config_file = self.config_dir / "config.toml"
        
        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from TOML file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return toml.load(f)
            except Exception:
                # If config is corrupted, start fresh
                return self._get_default_config()
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "optimizations": {
                "token_cache_enabled": False,
                "quantization_enabled": False
            }
        }
    
    def get(self, key: str, default=None):
        """
        Get configuration value from TOML config file.
        """
        # Check config file
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value and save to file."""
        keys = key.split('.')
        config = self._config
        
        # Navigate to the nested key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        
        # Save to file
        self._save_config()
    
    def _save_config(self):
        """Save configuration to TOML file."""
        try:
            with open(self.config_file, 'w') as f:
                toml.dump(self._config, f)
        except Exception as e:
            print(f"Warning: Could not save config: {e}")
    
    def get_all_optimizations(self) -> Dict[str, bool]:
        """Get all optimization settings."""
        return {
            "token_cache": self.get("optimizations.token_cache_enabled", False),
            "quantize": self.get("optimizations.quantization_enabled", False)
        }


# Global config instance
_config = None

def get_config() -> VocalizeConfig:
    """Get or create the global config instance."""
    global _config
    if _config is None:
        _config = VocalizeConfig()
    return _config