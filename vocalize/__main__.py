"""Main entry point for vocalize package."""

# Set thread limits before ANY imports to prevent deadlocks
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'
os.environ['ORT_DISABLE_SPINNING'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['VECLIB_MAXIMUM_THREADS'] = '4'

from .cli import main

if __name__ == "__main__":
    main()