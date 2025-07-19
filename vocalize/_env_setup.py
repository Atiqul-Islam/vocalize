"""
Early environment setup for threading optimization.

This module MUST be imported before any other imports to prevent
threading conflicts and ensure optimal performance.
"""

import os
import sys

# CRITICAL: Set thread limits BEFORE any library initialization
# Use optimal number of threads for performance
import multiprocessing
num_threads = str(multiprocessing.cpu_count())
os.environ.update({
    'OMP_NUM_THREADS': num_threads,
    'MKL_NUM_THREADS': num_threads, 
    'NUMEXPR_NUM_THREADS': num_threads,
    'ORT_DISABLE_SPINNING': '0',  # Enable spinning for better performance
    'OPENBLAS_NUM_THREADS': num_threads,
    'VECLIB_MAXIMUM_THREADS': num_threads,
    'BLIS_NUM_THREADS': num_threads,
})

# Download required NLTK data if not present
def ensure_nltk_data():
    """Download required NLTK data if not available."""
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.tag import pos_tag
        
        # Test if averaged_perceptron_tagger_eng is available
        try:
            pos_tag(['test'])
        except LookupError:
            print("📥 Downloading NLTK data (averaged_perceptron_tagger_eng)...")
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)
            nltk.download('punkt', quiet=True)  # Often needed with tagger
            
    except ImportError:
        # NLTK not available, skip
        pass
    except Exception:
        # Silent fail for NLTK issues
        pass

# Download NLTK data on import
ensure_nltk_data()

# Debug output if verbose mode is detected
if '--verbose' in sys.argv or os.environ.get('VOCALIZE_DEBUG'):
    print(f"🔧 Environment setup complete:")
    print(f"   OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")
    print(f"   Threading: Multi-threaded mode ({num_threads} threads)")