"""
Optimization command handlers for Vocalize CLI.
Handles cache and quantization commands.
"""

import sys
import platformdirs
from pathlib import Path
from typing import Optional

from .token_cache import TokenCache
from .cache_builder import build_default_cache
from .model_optimizer import ModelOptimizer
from .config import get_config


def handle_optimize_command(args):
    """Handle the 'optimize' command for all optimizations."""
    if not args.optimize_action:
        print("Error: No optimization action specified. Use 'optimize --help' for options.")
        return
    
    if args.optimize_action == "cache":
        handle_optimize_cache_command(args)
    elif args.optimize_action == "quantize":
        handle_optimize_quantize_command(args)
    elif args.optimize_action == "enable":
        # Enable all optimizations
        handle_optimize_enable_all_command(args)
    elif args.optimize_action == "disable":
        # Disable all optimizations
        handle_optimize_disable_all_command(args)
    elif args.optimize_action == "status":
        handle_optimize_status_command(args)
    else:
        print(f"Unknown optimize action: {args.optimize_action}")


def handle_optimize_cache_command(args):
    """Handle cache optimization commands."""
    cache_base = platformdirs.user_cache_dir("vocalize", "Vocalize")
    cache_file = Path(cache_base) / "token_cache.db"
    
    if not args.cache_action:
        print("Error: No cache action specified. Use 'optimize cache --help' for options.")
        return
    
    if args.cache_action == "enable":
        # Enable token cache
        config = get_config()
        config.set("optimizations.token_cache_enabled", True)
        print("✅ Token cache enabled")
        
        # Build cache if it doesn't exist
        if not cache_file.exists():
            print("\nBuilding token cache for first use...")
            build_default_cache(cache_file)
        else:
            print("\nToken cache already exists. Run 'vocalize optimize cache build --force' to rebuild.")
    
    elif args.cache_action == "disable":
        # Disable token cache
        config = get_config()
        config.set("optimizations.token_cache_enabled", False)
        print("✅ Token cache disabled")
    
    elif args.cache_action == "build":
        # Check if cache is enabled
        config = get_config()
        if not config.get('optimizations.token_cache_enabled', False):
            print("Token cache is disabled. Enable it first with: vocalize optimize cache enable")
            return
        
        force = getattr(args, 'force', False)
        
        if cache_file.exists() and not force:
            print("Token cache already exists. Use --force to rebuild.")
            # Show current stats
            cache = TokenCache()
            stats = cache.get_stats()
            print(f"\nCurrent cache statistics:")
            print(f"  Entries: {stats['entries']:,}")
            print(f"  Size: {stats['size_mb']:.2f} MB")
            return
        
        print("Building token cache...")
        build_default_cache(cache_file, force=force)
    
    elif args.cache_action == "clear":
        if cache_file.exists():
            cache_file.unlink()
            print("✅ Token cache cleared")
        else:
            print("No cache to clear")
    
    elif args.cache_action == "stats":
        if not cache_file.exists():
            print("No token cache found.")
            print("Status: ", end="")
            config = get_config()
            if config.get('optimizations.token_cache_enabled', False):
                print("Enabled (will be built on first use)")
            else:
                print("Disabled")
            return
        
        cache = TokenCache()
        stats = cache.get_stats()
        print(f"Token Cache Statistics:")
        print(f"  Status: {'Enabled' if stats.get('enabled', False) else 'Disabled'}")
        if stats.get('enabled', False):
            print(f"  Location: {cache_file}")
            print(f"  Entries: {stats['entries']:,}")
            print(f"  Data size: {stats['size_mb']:.2f} MB")
            print(f"  File size: {stats['file_size_mb']:.2f} MB")
    
    else:
        print(f"Unknown cache action: {args.cache_action}")



def handle_optimize_quantize_command(args):
    """Handle quantization subcommands."""
    if not args.quantize_action:
        # Show info if no subcommand
        show_quantization_info()
        return
    
    if args.quantize_action == "enable":
        handle_quantize_enable_command(args)
    elif args.quantize_action == "disable":
        handle_quantize_disable_command(args)
    elif args.quantize_action == "status":
        show_quantization_info()
    else:
        print(f"Unknown quantize action: {args.quantize_action}")


def show_quantization_info():
    """Show information about quantization."""
    # Import ModelManager lazily
    from .model_manager import ModelManager
    
    # Create ModelManager and pass it to optimizer
    manager = ModelManager()
    optimizer = ModelOptimizer(model_manager=manager)
    
    print("Quantization (8-bit)")
    print("=" * 50)
    print("\nQuantization uses a pre-optimized INT8 model with selective layer quantization:")
    print("  • Reduces model size by ~75%")
    print("  • Faster inference on CPU (2-4x speedup)")
    print("  • Minimal quality trade-off (selective layer quantization)")
    print("  • Pre-quantized by experts for optimal performance")
    
    # Check current status
    opt_status = optimizer.get_optimization_status()
    
    print(f"\nStatus: {'Enabled ✓' if opt_status['quantize']['enabled'] else 'Disabled ✗'}")
    
    if opt_status['quantize']['built']:
        print(f"Model: {Path(opt_status['quantize']['path']).name}")
        if opt_status['original']['size_mb'] and opt_status['quantize']['size_mb']:
            reduction = (1 - opt_status['quantize']['size_mb'] / opt_status['original']['size_mb']) * 100
            print(f"Size: {opt_status['quantize']['size_mb']:.1f} MB ({reduction:.1f}% compression)")
    else:
        print("Model: Not built")
    
    print("\nExpected benefits:")
    print("  • 4x smaller model size")
    print("  • 2-3x faster CPU inference")
    print("  • Lower memory usage")
    
    print("\nCommands:")
    print("  vocalize optimize quantize enable    - Enable quantization (auto-downloads model)")
    print("  vocalize optimize quantize disable   - Disable quantization")
    print("  vocalize optimize quantize status    - Show this info")


def handle_quantize_enable_command(args):
    """Enable quantization and auto-download pre-quantized model if needed."""
    from .model_manager import ModelManager
    manager = ModelManager()
    optimizer = ModelOptimizer(model_manager=manager)
    
    # Check if quantized model exists, download if needed
    opt_status = optimizer.get_optimization_status()
    if not opt_status['quantize']['built']:
        print("Downloading pre-quantized INT8 model...")
        success = optimizer.quantize()
        if not success:
            print("❌ Failed to download pre-quantized model")
            return
    
    # Enable quantization
    config = get_config()
    config.set("optimizations.quantization_enabled", True)
    print("✅ Quantization (8bit) enabled")


def handle_quantize_disable_command(args):
    """Disable quantization."""
    config = get_config()
    config.set("optimizations.quantization_enabled", False)
    print("✅ Quantization (8bit) disabled")




def handle_optimize_enable_all_command(args):
    """Enable all optimizations."""
    print("Enabling all optimizations...")
    
    # Import ModelManager for graph/quantize
    from .model_manager import ModelManager
    manager = ModelManager()
    optimizer = ModelOptimizer(model_manager=manager)
    
    # Get config instance
    config = get_config()
    
    # Enable token cache
    config.set("optimizations.token_cache_enabled", True)
    print("✅ Token cache enabled")
    
    # Build cache if it doesn't exist
    cache_base = platformdirs.user_cache_dir("vocalize", "Vocalize")
    cache_file = Path(cache_base) / "token_cache.db"
    if not cache_file.exists():
        print("  Building token cache for first use...")
        build_default_cache(cache_file)
    
    # Check if quantized model exists, download if needed
    opt_status = optimizer.get_optimization_status()
    if not opt_status['quantize']['built']:
        print("Downloading pre-quantized INT8 model...")
        success = optimizer.quantize()
        if not success:
            print("❌ Failed to download pre-quantized model")
            return
    
    # Enable quantization
    config.set("optimizations.quantization_enabled", True)
    print("✅ Quantization (8bit) enabled")
    
    print("\n🎉 All optimizations enabled!")


def handle_optimize_disable_all_command(args):
    """Disable all optimizations."""
    print("Disabling all optimizations...")
    
    # Get config instance
    config = get_config()
    
    # Disable token cache
    config.set("optimizations.token_cache_enabled", False)
    print("✅ Token cache disabled")
    
    # Disable quantization
    config.set("optimizations.quantization_enabled", False)
    print("✅ Quantization (8bit) disabled")
    
    print("\n✅ All optimizations disabled")


def handle_optimize_status_command(args):
    """Show status of all optimizations."""
    print("Vocalize Optimization Status")
    print("=" * 40)
    
    # Get config
    config = get_config()
    
    # Token cache status
    cache_enabled = config.get('optimizations.token_cache_enabled', False)
    cache_base = platformdirs.user_cache_dir("vocalize", "Vocalize")
    cache_file = Path(cache_base) / "token_cache.db"
    
    print("\nToken Cache:")
    print(f"  Status: {'Enabled ✓' if cache_enabled else 'Disabled ✗'}")
    
    if cache_file.exists():
        cache = TokenCache()
        stats = cache.get_stats()
        if stats.get('enabled', False):
            print(f"  Entries: {stats['entries']:,}")
            print(f"  Size: {stats['size_mb']:.2f} MB")
            # Could add hit rate if we track it
    else:
        print("  Not built yet")
    
    # Model optimizations status
    # Import ModelManager lazily
    from .model_manager import ModelManager
    
    # Create ModelManager and pass it to optimizer
    manager = ModelManager()
    optimizer = ModelOptimizer(model_manager=manager)
    opt_status = optimizer.get_optimization_status()
    
    print("\nQuantization (8bit):")
    print(f"  Status: {'Enabled ✓' if opt_status['quantize']['enabled'] else 'Disabled ✗'}")
    if opt_status['quantize']['built']:
        print(f"  Model: {Path(opt_status['quantize']['path']).name} ({opt_status['quantize']['size_mb']:.1f} MB)")
        if opt_status['original']['size_mb']:
            reduction = (1 - opt_status['quantize']['size_mb'] / opt_status['original']['size_mb']) * 100
            print(f"  Compression: {reduction:.1f}% smaller than original")
    else:
        print("  Model: Not downloaded")
        print("  Download with: vocalize optimize quantize enable")
    
    # Active model
    try:
        active_path = optimizer.get_active_model_path()
        print(f"\nActive Model: {active_path.name}")
        
        # Performance estimate
        if cache_enabled and opt_status['quantize']['enabled']:
            print("Performance: ~1.1s (vs 5.4s baseline)")
        elif cache_enabled:
            print("Performance: ~2.8s (vs 5.4s baseline)")
        elif opt_status['quantize']['enabled']:
            print("Performance: ~3.5s (vs 5.4s baseline)")
        else:
            print("Performance: ~5.4s (baseline)")
    except FileNotFoundError as e:
        print(f"\n⚠️  {e}")


