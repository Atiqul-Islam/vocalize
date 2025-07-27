"""
Token cache for fast phoneme lookup.
Automatically creates cache on first run and provides O(1) lookup for common words.
"""

import sqlite3
import pickle
import time
import os
from pathlib import Path
from typing import Dict, List, Optional
import platformdirs
import threading

from .config import get_config


class TokenCache:
    """
    Persistent token cache for fast phoneme lookup.
    Uses SQLite for efficient storage and retrieval.
    """
    
    CACHE_VERSION = "1.0"
    _instance_lock = threading.Lock()
    _shared_instance = None
    
    def __new__(cls, cache_dir: Optional[Path] = None):
        """Ensure singleton instance for thread safety."""
        if cls._shared_instance is None:
            with cls._instance_lock:
                if cls._shared_instance is None:
                    cls._shared_instance = super().__new__(cls)
        return cls._shared_instance
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize token cache with automatic creation on first run if enabled."""
        # Skip initialization if already done
        if hasattr(self, '_initialized'):
            return
        
        # Check if token cache is enabled
        self.enabled = get_config().get('optimizations.token_cache_enabled', False)
        
        if not self.enabled:
            self._initialized = True
            return
        
        if cache_dir is None:
            # Use same cache directory as models
            cache_base = platformdirs.user_cache_dir("vocalize", "Vocalize")
            cache_dir = Path(cache_base)
        
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "token_cache.db"
        self._lock = threading.Lock()
        
        # Check if cache exists and is valid
        if not self._is_cache_valid():
            print("Token cache not found or outdated. Building cache...")
            self._build_initial_cache()
        else:
            print(f"✅ Using existing token cache (v{self.CACHE_VERSION})")
        
        # Connect to cache
        self._connect()
        self._initialized = True
    
    def _is_cache_valid(self) -> bool:
        """Check if cache exists and has correct version."""
        if not self.cache_file.exists():
            return False
        
        try:
            conn = sqlite3.connect(str(self.cache_file))
            cursor = conn.cursor()
            
            # Check if metadata table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='metadata'
            """)
            if not cursor.fetchone():
                conn.close()
                return False
            
            # Check version
            cursor.execute("SELECT value FROM metadata WHERE key='version'")
            result = cursor.fetchone()
            conn.close()
            
            return result and result[0] == self.CACHE_VERSION
        except Exception as e:
            print(f"Cache validation error: {e}")
            return False
    
    def _connect(self):
        """Connect to cache database with thread-safe connection."""
        # SQLite connections are not thread-safe, so we create one per thread
        self.local = threading.local()
    
    def _get_connection(self):
        """Get thread-local database connection."""
        if not hasattr(self.local, 'conn'):
            self.local.conn = sqlite3.connect(str(self.cache_file))
            self.local.conn.row_factory = sqlite3.Row
        return self.local.conn
    
    def _build_initial_cache(self):
        """Build initial cache with common words."""
        try:
            from .cache_builder import build_default_cache
            build_default_cache(self.cache_file)
        except ImportError:
            print("Warning: cache_builder not found. Creating empty cache.")
            self._create_empty_cache()
    
    def _create_empty_cache(self):
        """Create empty cache database with schema."""
        conn = sqlite3.connect(str(self.cache_file))
        cursor = conn.cursor()
        
        # Create schema
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS token_cache (
                text TEXT PRIMARY KEY,
                tokens BLOB NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        
        # Set version
        cursor.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES ('version', ?)",
            (self.CACHE_VERSION,)
        )
        
        cursor.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES ('build_time', ?)",
            (str(int(time.time())),)
        )
        
        conn.commit()
        conn.close()
    
    def get_tokens(self, text: str) -> Optional[List[int]]:
        """
        Get tokens for text from cache.
        Returns None if not found or cache is disabled.
        """
        if not self.enabled:
            return None
            
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Normalize text for lookup
            normalized_text = text.lower().strip()
            
            cursor.execute("SELECT tokens FROM token_cache WHERE text = ?", (normalized_text,))
            result = cursor.fetchone()
            
            if result:
                # Deserialize tokens
                return pickle.loads(result['tokens'])
            return None
        except Exception as e:
            print(f"Cache lookup error: {e}")
            return None
    
    def add_tokens(self, text: str, tokens: List[int]):
        """Add new text-token pair to cache if enabled."""
        if not self.enabled:
            return
            
        try:
            with self._lock:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                # Normalize text for storage
                normalized_text = text.lower().strip()
                
                cursor.execute(
                    "INSERT OR REPLACE INTO token_cache (text, tokens) VALUES (?, ?)",
                    (normalized_text, pickle.dumps(tokens))
                )
                conn.commit()
        except Exception as e:
            print(f"Cache write error: {e}")
    
    def get_stats(self) -> Dict[str, any]:
        """Get cache statistics."""
        if not self.enabled:
            return {
                'entries': 0,
                'size_bytes': 0,
                'size_mb': 0,
                'file_size_mb': 0,
                'enabled': False
            }
            
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) as count FROM token_cache")
            count = cursor.fetchone()['count']
            
            cursor.execute("SELECT SUM(LENGTH(tokens)) as size FROM token_cache")
            size = cursor.fetchone()['size'] or 0
            
            # Get cache file size
            file_size = self.cache_file.stat().st_size if self.cache_file.exists() else 0
            
            return {
                'entries': count,
                'size_bytes': size,
                'size_mb': size / (1024 * 1024),
                'file_size_mb': file_size / (1024 * 1024),
                'enabled': True
            }
        except Exception as e:
            print(f"Stats error: {e}")
            return {
                'entries': 0,
                'size_bytes': 0,
                'size_mb': 0,
                'file_size_mb': 0,
                'enabled': True
            }
    
    def clear(self):
        """Clear all cache entries."""
        if not self.enabled:
            print("Token cache is disabled")
            return
            
        try:
            with self._lock:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("DELETE FROM token_cache")
                conn.commit()
                print("✅ Token cache cleared")
        except Exception as e:
            print(f"Cache clear error: {e}")
    
    def close(self):
        """Close database connection."""
        if not self.enabled:
            return
            
        if hasattr(self, 'local') and hasattr(self.local, 'conn'):
            self.local.conn.close()
            delattr(self.local, 'conn')