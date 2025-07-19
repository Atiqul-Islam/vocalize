#!/bin/bash
# macOS build script for vocalize
# This script builds the macOS wheel with maturin and bundles all required libraries

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions for colored output
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_info() { echo -e "ℹ️  $1"; }

echo "🔨 Building macOS wheel with bundled ONNX Runtime..."
echo "================================================="
echo ""

# Get project directory
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# Detect architecture
ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" ]]; then
    TARGET="aarch64-apple-darwin"
    ARCH_NAME="arm64"
    print_info "Detected Apple Silicon (arm64)"
elif [[ "$ARCH" == "x86_64" ]]; then
    TARGET="x86_64-apple-darwin"
    ARCH_NAME="x86_64"
    print_info "Detected Intel Mac (x86_64)"
else
    print_error "Unsupported architecture: $ARCH"
    exit 1
fi

# Clean previous builds
print_info "Cleaning previous macOS builds..."
rm -rf target/wheels/vocalize_python-*-macosx*.whl 2>/dev/null || true

# Step 1: Build the wheel with maturin
echo ""
echo "Step 1: Building wheel with maturin..."
echo "========================================="

maturin build --release \
    --target "$TARGET" \
    --manifest-path crates/vocalize-python/Cargo.toml || {
    print_error "Failed to build wheel with maturin"
    exit 1
}

# Find the built wheel
WHEEL_FILE=$(ls target/wheels/vocalize_python-*-macosx*.whl 2>/dev/null | head -1)
if [ -z "$WHEEL_FILE" ]; then
    print_error "No macOS wheel found after build!"
    exit 1
fi

print_success "Built wheel: $(basename "$WHEEL_FILE")"

# Step 2: Find ONNX Runtime libraries
echo ""
echo "Step 2: Locating ONNX Runtime libraries..."
echo "========================================="

# Find ONNX Runtime library from build directory
ONNX_LIB_DIR=$(find target -path "*/build/vocalize-python-*/out/onnxruntime/lib" -type d 2>/dev/null | head -1)
if [ -z "$ONNX_LIB_DIR" ]; then
    print_error "ONNX Runtime library directory not found in build artifacts!"
    print_info "Looking for pattern: target/*/build/vocalize-python-*/out/onnxruntime/lib"
    exit 1
fi

print_success "Found ONNX Runtime libraries at: $ONNX_LIB_DIR"

# Verify libraries exist
if [ ! -f "$ONNX_LIB_DIR/libonnxruntime.1.22.0.dylib" ] && [ ! -f "$ONNX_LIB_DIR/libonnxruntime.dylib" ]; then
    print_error "libonnxruntime.dylib not found in $ONNX_LIB_DIR"
    exit 1
fi

# Step 3: Create temporary directory and extract wheel
echo ""
echo "Step 3: Extracting wheel for modification..."
echo "========================================="

TEMP_DIR=$(mktemp -d)
print_info "Working in temporary directory: $TEMP_DIR"

cd "$TEMP_DIR"

# Extract wheel
python3 -m zipfile -e "$PROJECT_DIR/$WHEEL_FILE" . || {
    print_error "Failed to extract wheel"
    cd "$PROJECT_DIR"
    rm -rf "$TEMP_DIR"
    exit 1
}

# Find the package directory
PACKAGE_DIR=$(find . -type d -name "vocalize_python" | head -1)
if [ -z "$PACKAGE_DIR" ]; then
    print_error "vocalize_python directory not found in wheel"
    cd "$PROJECT_DIR"
    rm -rf "$TEMP_DIR"
    exit 1
fi

print_success "Found package directory: $PACKAGE_DIR"

# Step 4: Copy ONNX Runtime libraries
echo ""
echo "Step 4: Bundling ONNX Runtime libraries..."
echo "========================================="

# Copy main library (handle different naming conventions)
if [ -f "$PROJECT_DIR/$ONNX_LIB_DIR/libonnxruntime.1.22.0.dylib" ]; then
    cp "$PROJECT_DIR/$ONNX_LIB_DIR/libonnxruntime.1.22.0.dylib" "$PACKAGE_DIR/" || {
        print_error "Failed to copy libonnxruntime.1.22.0.dylib"
        cd "$PROJECT_DIR"
        rm -rf "$TEMP_DIR"
        exit 1
    }
    print_success "Copied libonnxruntime.1.22.0.dylib"
    
    # Create symlink
    cd "$PACKAGE_DIR"
    ln -s libonnxruntime.1.22.0.dylib libonnxruntime.dylib
    print_success "Created symlink"
elif [ -f "$PROJECT_DIR/$ONNX_LIB_DIR/libonnxruntime.dylib" ]; then
    cp "$PROJECT_DIR/$ONNX_LIB_DIR/libonnxruntime.dylib" "$PACKAGE_DIR/" || {
        print_error "Failed to copy libonnxruntime.dylib"
        cd "$PROJECT_DIR"
        rm -rf "$TEMP_DIR"
        exit 1
    }
    print_success "Copied libonnxruntime.dylib"
fi

# Copy providers shared library if it exists
if [ -f "$PROJECT_DIR/$ONNX_LIB_DIR/libonnxruntime_providers_shared.dylib" ]; then
    cp "$PROJECT_DIR/$ONNX_LIB_DIR/libonnxruntime_providers_shared.dylib" "$PACKAGE_DIR/"
    print_success "Copied libonnxruntime_providers_shared.dylib"
fi

cd "$TEMP_DIR"

# Step 5: Update RECORD file
echo ""
echo "Step 5: Updating wheel metadata..."
echo "========================================="

# Find RECORD file
RECORD_FILE=$(find . -name "RECORD" -path "*dist-info/*" | head -1)
if [ -z "$RECORD_FILE" ]; then
    print_error "RECORD file not found"
    cd "$PROJECT_DIR"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Update RECORD with new files
python3 -c "
import csv
import hashlib
import base64
import os

record_file = '$RECORD_FILE'
package_dir = '$PACKAGE_DIR'

# Read existing records
records = []
with open(record_file, 'r', newline='') as f:
    reader = csv.reader(f)
    records = list(reader)

# Add new library entries
libs_to_add = []

# Check which files were actually copied
if os.path.exists(os.path.join(package_dir, 'libonnxruntime.1.22.0.dylib')):
    libs_to_add.extend(['libonnxruntime.1.22.0.dylib', 'libonnxruntime.dylib'])
elif os.path.exists(os.path.join(package_dir, 'libonnxruntime.dylib')):
    libs_to_add.append('libonnxruntime.dylib')

if os.path.exists(os.path.join(package_dir, 'libonnxruntime_providers_shared.dylib')):
    libs_to_add.append('libonnxruntime_providers_shared.dylib')

for lib in libs_to_add:
    lib_path = os.path.join(package_dir, lib)
    if os.path.exists(lib_path) and not os.path.islink(lib_path):
        # Calculate hash only for real files, not symlinks
        with open(lib_path, 'rb') as f:
            data = f.read()
            hash_digest = hashlib.sha256(data).digest()
            hash_b64 = base64.urlsafe_b64encode(hash_digest).decode().rstrip('=')
            size = len(data)
        
        # Add to records
        record_path = os.path.join('vocalize_python', lib).replace(os.sep, '/')
        records.append([record_path, f'sha256={hash_b64}', str(size)])
        print(f'Added {lib} to RECORD')
    elif os.path.islink(lib_path):
        # For symlinks, just add without hash
        record_path = os.path.join('vocalize_python', lib).replace(os.sep, '/')
        records.append([record_path, '', ''])
        print(f'Added symlink {lib} to RECORD')

# Write updated RECORD
with open(record_file, 'w', newline='') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerows(records)
" || {
    print_error "Failed to update RECORD file"
    cd "$PROJECT_DIR"
    rm -rf "$TEMP_DIR"
    exit 1
}

print_success "Updated RECORD file"

# Step 6: Repack wheel
echo ""
echo "Step 6: Creating bundled wheel..."
echo "========================================="

# Create new wheel filename
BUNDLED_WHEEL="${WHEEL_FILE%.whl}_bundled.whl"
BUNDLED_WHEEL_NAME=$(basename "$BUNDLED_WHEEL")

# Create the wheel
python3 -m zipfile -c "$PROJECT_DIR/$BUNDLED_WHEEL" . || {
    print_error "Failed to create bundled wheel"
    cd "$PROJECT_DIR"
    rm -rf "$TEMP_DIR"
    exit 1
}

# Cleanup
cd "$PROJECT_DIR"
rm -rf "$TEMP_DIR"

print_success "Created bundled macOS wheel: $BUNDLED_WHEEL_NAME"

# Step 7: Optional - Use delocate for better compatibility
echo ""
echo "Step 7: Checking for delocate..."
echo "========================================="

if command -v delocate-wheel &> /dev/null; then
    print_info "Running delocate to ensure wheel compatibility..."
    delocate-wheel -w target/wheels -v "$BUNDLED_WHEEL" || {
        print_warning "delocate failed, but wheel may still work"
    }
else
    print_warning "delocate not found. Install with: pip install delocate"
    print_info "delocate helps ensure wheel compatibility across macOS versions"
fi

# Step 8: Display summary
echo ""
echo "✨ Build Summary"
echo "================"
echo "Architecture:   $ARCH_NAME"
echo "Target:         $TARGET"
echo "Original wheel: $(basename "$WHEEL_FILE")"
echo "Bundled wheel:  $BUNDLED_WHEEL_NAME"
echo "Location:       target/wheels/"
echo ""
echo "📦 Bundled libraries:"
if [ -f "$ONNX_LIB_DIR/libonnxruntime.1.22.0.dylib" ]; then
    echo "  - libonnxruntime.1.22.0.dylib"
    echo "  - libonnxruntime.dylib (symlink)"
else
    echo "  - libonnxruntime.dylib"
fi
if [ -f "$ONNX_LIB_DIR/libonnxruntime_providers_shared.dylib" ]; then
    echo "  - libonnxruntime_providers_shared.dylib"
fi
echo ""
echo "🚀 To install and test:"
echo "  uv pip install $BUNDLED_WHEEL --force-reinstall"
echo "  uv run python -m vocalize speak \"Hello macOS\" --output test.wav"
echo ""
print_success "Build complete!"