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
# Wheels are named for the DISTRIBUTION (vocalize-tts -> vocalize_tts-*), not
# for the Rust crate. They were vocalize_rust-* back when this script built
# crates/vocalize-rust/pyproject.toml directly.
print_info "Cleaning previous Linux builds..."
rm -rf crates/target/wheels/vocalize_tts-*-manylinux*.whl 2>/dev/null || true
rm -rf crates/target/wheels/vocalize_tts-*-linux*.whl 2>/dev/null || true

# Step 1: Build the wheel with maturin
echo ""
echo "Step 1: Building wheel with maturin..."
echo "========================================="

# Build from the ROOT pyproject.toml -- do NOT pass --manifest-path here.
#
# --manifest-path made maturin use crates/vocalize-rust/pyproject.toml, which
# declares no `python-packages`. That produced a wheel containing ONLY the
# compiled extension: no `vocalize` package and no `vocalize` command, published
# under the wrong distribution name (vocalize-rust instead of vocalize-tts).
#
# The root pyproject.toml already points maturin at the same crate via
# [tool.maturin] manifest-path, and additionally ships the `vocalize` package
# and the console script. Same extension, complete wheel.
maturin build --release \
    --interpreter python3.10 || {
    print_error "Failed to build wheel with maturin"
    exit 1
}

# Find the built wheel
WHEEL_FILE=$(ls crates/target/wheels/vocalize_tts-*-manylinux*.whl 2>/dev/null | head -1)
if [ -z "$WHEEL_FILE" ]; then
    WHEEL_FILE=$(ls crates/target/wheels/vocalize_tts-*-linux*.whl 2>/dev/null | head -1)
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

# Repack OVER the original wheel. Do NOT invent a new filename.
#
# This used to write "${WHEEL_FILE%.whl}_bundled.whl", which produced
# vocalize_tts-0.1.0-cp38-abi3-manylinux_2_34_x86_64_bundled.whl. A wheel
# filename is parsed positionally as
#   {distribution}-{version}(-{build})?-{python}-{abi}-{platform}.whl
# so that "_bundled" suffix landed inside the PLATFORM tag, making it
# "manylinux_2_34_x86_64_bundled" -- not a platform tag any installer
# recognises. The bundled wheel, i.e. the actual release artifact, was therefore
# uninstallable:
#   uv pip install ..._bundled.whl
#   error: A path dependency is incompatible with the current platform
# Verified: the same bytes under the correct filename install cleanly.
#
# Replacing the original in place is what auditwheel/delvewheel do; the
# unbundled wheel is an intermediate and must not be shipped or kept around to
# be published by mistake.
BUNDLED_WHEEL="$WHEEL_FILE"
BUNDLED_WHEEL_NAME=$(basename "$BUNDLED_WHEEL")

# Create the wheel
rm -f "$PROJECT_DIR/$BUNDLED_WHEEL"
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

# Step 6b: Verify the wheel is actually shippable
#
# This gate exists because every defect it checks for has already shipped once:
# a wheel with no `vocalize` package, no console script, and a platform tag
# corrupted by a "_bundled" suffix that made it uninstallable. Fail the build
# rather than hand over a broken artifact.
echo ""
echo "Step 6b: Verifying wheel contents..."
echo "========================================="

if ! python3 - "$PROJECT_DIR/$BUNDLED_WHEEL" <<'PYEOF'
import sys, zipfile, re
from pathlib import Path

path = Path(sys.argv[1])
errors = []

# The filename must parse as a valid wheel name; the platform tag in particular
# must not have picked up a suffix.
stem = path.name[:-4] if path.name.endswith(".whl") else path.name
parts = stem.split("-")
if len(parts) not in (5, 6):
    errors.append(f"filename does not parse as a wheel: {path.name}")
else:
    platform_tag = parts[-1]
    if not re.fullmatch(r"(manylinux[0-9_]*_(x86_64|aarch64)|linux_(x86_64|aarch64)|any)", platform_tag):
        errors.append(f"invalid platform tag {platform_tag!r} in {path.name}")

z = zipfile.ZipFile(path)
names = z.namelist()

if not any(n.startswith("vocalize/") and n.endswith(".py") for n in names):
    errors.append("wheel contains no `vocalize` Python package")
if not any(n == "vocalize/cli.py" for n in names):
    errors.append("wheel is missing vocalize/cli.py")
if not any("vocalize_rust" in n and n.endswith(".so") for n in names):
    errors.append("wheel contains no compiled vocalize_rust extension")
if not any("libonnxruntime.so" in n for n in names):
    errors.append("wheel does not bundle libonnxruntime.so")

ep = [n for n in names if n.endswith("entry_points.txt")]
if not ep:
    errors.append("wheel declares no entry_points.txt (no `vocalize` command)")
elif "vocalize.cli:main" not in z.read(ep[0]).decode():
    errors.append("entry_points.txt does not point at vocalize.cli:main")

if errors:
    for e in errors:
        print(f"  FAIL: {e}")
    sys.exit(1)

print(f"  OK: valid platform tag {parts[-1]}")
print(f"  OK: {sum(1 for n in names if n.startswith('vocalize/'))} vocalize/ package files")
print("  OK: vocalize_rust extension present")
print("  OK: libonnxruntime bundled")
print("  OK: `vocalize` console script -> vocalize.cli:main")
PYEOF
then
    print_error "Wheel verification FAILED - refusing to ship this artifact"
    exit 1
fi

print_success "Wheel verification passed"

# Step 7: Display summary
echo ""
echo "✨ Build Summary"
echo "================"
echo "Wheel:          $BUNDLED_WHEEL_NAME"
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
# The wheel now ships the `vocalize` console script, so this no longer has to be
# run from a checkout. The old instructions used `uv run python -m vocalize`,
# which only worked because ./vocalize/ was importable from the repo cwd -- the
# wheel itself contained no Python package at all.
echo "  vocalize speak \"Hello Linux\" --output test.wav"
echo ""
print_success "Build complete!"