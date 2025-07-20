#!/bin/bash
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

echo "🔨 Building Linux wheel with bundled ONNX Runtime..."
echo "================================================="
echo ""

# Get project directory
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# Clean previous builds
print_info "Cleaning previous Linux builds..."
rm -rf crates/target/wheels/vocalize_rust-*-manylinux*.whl 2>/dev/null || true
rm -rf crates/target/wheels/vocalize_rust-*-linux*.whl 2>/dev/null || true

# Step 1: Build the wheel with maturin
echo ""
echo "Step 1: Building wheel with maturin..."
echo "========================================="

maturin build --release \
    --manifest-path crates/vocalize-rust/Cargo.toml \
    --interpreter python3.10 || {
    print_error "Failed to build wheel with maturin"
    exit 1
}

# Find the built wheel
WHEEL_FILE=$(ls crates/target/wheels/vocalize_rust-*-manylinux*.whl 2>/dev/null | head -1)
if [ -z "$WHEEL_FILE" ]; then
    WHEEL_FILE=$(ls crates/target/wheels/vocalize_rust-*-linux*.whl 2>/dev/null | head -1)
fi

if [ -z "$WHEEL_FILE" ]; then
    print_error "No Linux wheel found after build!"
    exit 1
fi

print_success "Built wheel: $(basename "$WHEEL_FILE")"

# Step 2: Find ONNX Runtime libraries
echo ""
echo "Step 2: Locating ONNX Runtime libraries..."
echo "========================================="

# Find ONNX Runtime library from build directory
ONNX_LIB_DIR=$(find crates/target -path "*/build/vocalize-rust-*/out/onnxruntime/lib" -type d 2>/dev/null | head -1)
if [ -z "$ONNX_LIB_DIR" ]; then
    print_error "ONNX Runtime library directory not found in build artifacts!"
    print_info "Looking for pattern: crates/target/*/build/vocalize-rust-*/out/onnxruntime/lib"
    exit 1
fi

print_success "Found ONNX Runtime libraries at: $ONNX_LIB_DIR"

# Verify libraries exist
if [ ! -f "$ONNX_LIB_DIR/libonnxruntime.so.1.22.0" ]; then
    print_error "libonnxruntime.so.1.22.0 not found in $ONNX_LIB_DIR"
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
PACKAGE_DIR=$(find . -type d -name "vocalize_rust" | head -1)
if [ -z "$PACKAGE_DIR" ]; then
    print_error "vocalize_rust directory not found in wheel"
    cd "$PROJECT_DIR"
    rm -rf "$TEMP_DIR"
    exit 1
fi

print_success "Found package directory: $PACKAGE_DIR"

# Step 4: Copy ONNX Runtime libraries
echo ""
echo "Step 4: Bundling ONNX Runtime libraries..."
echo "========================================="

# Copy main library
cp "$PROJECT_DIR/$ONNX_LIB_DIR/libonnxruntime.so.1.22.0" "$PACKAGE_DIR/" || {
    print_error "Failed to copy libonnxruntime.so.1.22.0"
    cd "$PROJECT_DIR"
    rm -rf "$TEMP_DIR"
    exit 1
}
print_success "Copied libonnxruntime.so.1.22.0"

# Create symlinks
cd "$PACKAGE_DIR"
ln -s libonnxruntime.so.1.22.0 libonnxruntime.so.1
ln -s libonnxruntime.so.1 libonnxruntime.so
print_success "Created symlinks"

# Copy providers shared library if it exists
if [ -f "$PROJECT_DIR/$ONNX_LIB_DIR/libonnxruntime_providers_shared.so" ]; then
    cp "$PROJECT_DIR/$ONNX_LIB_DIR/libonnxruntime_providers_shared.so" .
    print_success "Copied libonnxruntime_providers_shared.so"
fi

cd "$TEMP_DIR"

# Step 5: Update RECORD file
echo ""
echo "Step 5: Updating wheel metadata..."
echo "========================================="

# Find RECORD file (note: dist-info uses dash, not underscore)
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
libs_to_add = [
    'libonnxruntime.so',
    'libonnxruntime.so.1',
    'libonnxruntime.so.1.22.0',
]

# Check if providers shared library was copied
if os.path.exists(os.path.join(package_dir, 'libonnxruntime_providers_shared.so')):
    libs_to_add.append('libonnxruntime_providers_shared.so')

for lib in libs_to_add:
    lib_path = os.path.join(package_dir, lib)
    if os.path.exists(lib_path):
        # Calculate hash
        with open(lib_path, 'rb') as f:
            data = f.read()
            hash_digest = hashlib.sha256(data).digest()
            hash_b64 = base64.urlsafe_b64encode(hash_digest).decode().rstrip('=')
            size = len(data)
        
        # Add to records
        record_path = os.path.join('vocalize_rust', lib).replace(os.sep, '/')
        records.append([record_path, f'sha256={hash_b64}', str(size)])
        print(f'Added {lib} to RECORD')

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

print_success "Created bundled Linux wheel: $BUNDLED_WHEEL_NAME"

# Step 7: Display summary
echo ""
echo "✨ Build Summary"
echo "================"
echo "Original wheel: $(basename "$WHEEL_FILE")"
echo "Bundled wheel:  $BUNDLED_WHEEL_NAME"
echo "Location:       crates/target/wheels/"
echo ""
echo "📦 Bundled libraries:"
echo "  - libonnxruntime.so.1.22.0"
echo "  - libonnxruntime.so.1 (symlink)"
echo "  - libonnxruntime.so (symlink)"
if [ -f "$ONNX_LIB_DIR/libonnxruntime_providers_shared.so" ]; then
    echo "  - libonnxruntime_providers_shared.so"
fi
echo ""
echo "🚀 To install and test:"
echo "  uv pip install $BUNDLED_WHEEL --force-reinstall"
echo "  uv run python -m vocalize speak \"Hello Linux\" --output test.wav"
echo ""
print_success "Build complete!"