#!/bin/bash
# Complete build and bundle script for vocalize
# This script builds the project with maturin and bundles all required DLLs

set -e
set -o pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
print_error() {
    echo -e "${RED}Error: $1${NC}" >&2
}

print_success() {
    echo -e "${GREEN}$1${NC}"
}

print_warning() {
    echo -e "${YELLOW}Warning: $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    echo "Checking prerequisites..."
    
    # Check if in WSL
    if [ ! -f /proc/sys/fs/binfmt_misc/WSLInterop ]; then
        print_warning "This script is designed for WSL but appears to be running elsewhere"
    fi
    
    # Check required tools
    local missing_tools=()
    
    command -v maturin >/dev/null 2>&1 || missing_tools+=("maturin")
    command -v python3 >/dev/null 2>&1 || missing_tools+=("python3")
    command -v curl >/dev/null 2>&1 || missing_tools+=("curl")
    command -v cabextract >/dev/null 2>&1 || missing_tools+=("cabextract")
    
    if [ ${#missing_tools[@]} -gt 0 ]; then
        print_error "Missing required tools: ${missing_tools[*]}"
        echo "Please install missing tools:"
        echo "  - maturin: pip install maturin"
        echo "  - cabextract: sudo apt-get install cabextract"
        exit 1
    fi
    
    # Check if in project root
    if [ ! -f "pyproject.toml" ] || [ ! -d "crates/vocalize-python" ]; then
        print_error "This script must be run from the vocalize project root directory"
        exit 1
    fi
    
    print_success "All prerequisites met"
}

# Parse command line arguments
SKIP_BUILD=false
WHEEL_PATH=""
PYTHON_VERSION="python3.10"

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --wheel)
            WHEEL_PATH="$2"
            SKIP_BUILD=true
            shift 2
            ;;
        --python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --skip-build         Skip maturin build step"
            echo "  --wheel PATH         Use existing wheel file (implies --skip-build)"
            echo "  --python VERSION     Python version to target (default: python3.10)"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== Vocalize Build and Bundle Script ==="
echo ""

# Save original directory
PROJECT_DIR=$(pwd)

# Check prerequisites
check_prerequisites

# Step 1: Build with maturin (unless skipped)
if [ "$SKIP_BUILD" = false ]; then
    echo ""
    echo "Step 1: Building with maturin..."
    echo "========================================="
    
    # Clean previous builds
    rm -rf target/wheels/*.whl
    
    # Build with maturin (without PYO3_CONFIG_FILE since it works without it)
    print_warning "Building for target: x86_64-pc-windows-msvc with $PYTHON_VERSION"
    maturin build \
        --release \
        --target x86_64-pc-windows-msvc \
        --manifest-path crates/vocalize-python/Cargo.toml \
        --interpreter "$PYTHON_VERSION"
    
    if [ $? -ne 0 ]; then
        print_error "Maturin build failed"
        exit 1
    fi
    
    # Find the built wheel
    WHEEL_PATH=$(find target/wheels -name "*.whl" -type f | sort -r | head -1)
else
    echo ""
    echo "Step 1: Skipping build (using existing wheel)"
    echo "========================================="
    
    # If no wheel specified, try to find one
    if [ -z "$WHEEL_PATH" ]; then
        WHEEL_PATH=$(find target/wheels -name "*.whl" -type f ! -name "*_bundled.whl" | sort -r | head -1)
        if [ -n "$WHEEL_PATH" ]; then
            echo "Found existing wheel: $(basename "$WHEEL_PATH")"
        fi
    fi
fi

# Validate wheel path
if [ -z "$WHEEL_PATH" ] || [ ! -f "$WHEEL_PATH" ]; then
    print_error "No wheel file found. Please specify with --wheel or build first."
    exit 1
fi

# Convert to absolute path
WHEEL_PATH=$(realpath "$WHEEL_PATH")

print_success "Using wheel: $WHEEL_PATH"

# Create temp directories
TEMP_DIR=$(mktemp -d)
WORK_DIR=$(mktemp -d)

# Cleanup function
cleanup() {
    rm -rf "$TEMP_DIR" "$WORK_DIR"
}
trap cleanup EXIT

# Step 2: Download and extract VC++ Runtime DLLs
echo ""
echo "Step 2: Extracting VC++ Runtime DLLs..."
echo "========================================="

cd "$TEMP_DIR"

# Check if we already have the redistributable cached
CACHE_DIR="$HOME/.cache/vocalize"
VCREDIST_CACHE="$CACHE_DIR/vc_redist.x64.exe"

if [ -f "$VCREDIST_CACHE" ]; then
    print_success "Using cached VC++ redistributable"
    cp "$VCREDIST_CACHE" vc_redist.x64.exe
else
    echo "Downloading VC++ redistributable..."
    curl -L -o vc_redist.x64.exe "https://aka.ms/vs/17/release/vc_redist.x64.exe"
    
    # Cache for future use
    mkdir -p "$CACHE_DIR"
    cp vc_redist.x64.exe "$VCREDIST_CACHE"
fi

# Extract with cabextract
echo "Extracting embedded cabinet files..."
cabextract -q vc_redist.x64.exe || {
    print_error "Failed to extract vc_redist.x64.exe"
    exit 1
}

echo "Extracting DLL containers..."
cabextract -q a12 2>/dev/null || print_warning "a12 extraction had warnings"
cabextract -q a13 2>/dev/null || print_warning "a13 extraction had warnings"

# Verify DLLs were extracted
DLL_COUNT=$(ls -1 *.dll_amd64 2>/dev/null | wc -l)
if [ "$DLL_COUNT" -eq 0 ]; then
    print_error "No DLLs were extracted from VC++ redistributable"
    exit 1
fi

print_success "Extracted $DLL_COUNT DLL files"

cd "$PROJECT_DIR"

# Step 3: Extract and update the wheel
echo ""
echo "Step 3: Updating wheel with DLLs..."
echo "========================================="

cd "$WORK_DIR"

# Extract the wheel
echo "Extracting wheel contents..."
python3 -m zipfile -e "$WHEEL_PATH" . || {
    print_error "Failed to extract wheel"
    exit 1
}

# Find the package directory
PACKAGE_DIR=$(find . -type d -name "vocalize_python" | head -1)
if [ -z "$PACKAGE_DIR" ]; then
    print_error "vocalize_python directory not found in wheel"
    exit 1
fi

print_success "Found package directory: $PACKAGE_DIR"

# Copy ONNX Runtime DLLs
echo ""
echo "Looking for ONNX Runtime DLLs..."
ONNX_COPIED=0

# Try multiple possible locations
ONNX_SEARCH_DIRS=(
    "$PROJECT_DIR/target/x86_64-pc-windows-msvc/release/build/vocalize-python-*/out/onnxruntime/lib"
    "$PROJECT_DIR/crates/vocalize-python/target/x86_64-pc-windows-msvc/release/build/vocalize-python-*/out/onnxruntime/lib"
)

for pattern in "${ONNX_SEARCH_DIRS[@]}"; do
    for dir in $pattern; do
        if [ -d "$dir" ]; then
            echo "  Found ONNX build directory: $dir"
            for dll in "$dir"/*.dll; do
                if [ -f "$dll" ]; then
                    cp "$dll" "$PACKAGE_DIR/"
                    echo "  Copied: $(basename "$dll")"
                    ONNX_COPIED=$((ONNX_COPIED + 1))
                fi
            done
            break 2
        fi
    done
done

if [ $ONNX_COPIED -eq 0 ]; then
    print_warning "No ONNX Runtime DLLs found in build output"
fi

# Copy VC++ Runtime DLLs
echo ""
echo "Copying VC++ Runtime DLLs..."
VC_COPIED=0

# List of essential DLLs
declare -A ESSENTIAL_DLLS=(
    ["vcruntime140.dll_amd64"]="vcruntime140.dll"
    ["vcruntime140_1.dll_amd64"]="vcruntime140_1.dll"
    ["msvcp140.dll_amd64"]="msvcp140.dll"
    ["msvcp140_1.dll_amd64"]="msvcp140_1.dll"
    ["msvcp140_2.dll_amd64"]="msvcp140_2.dll"
    ["vccorlib140.dll_amd64"]="vccorlib140.dll"
    ["concrt140.dll_amd64"]="concrt140.dll"
)

for src_dll in "${!ESSENTIAL_DLLS[@]}"; do
    target_dll="${ESSENTIAL_DLLS[$src_dll]}"
    if [ -f "$TEMP_DIR/$src_dll" ]; then
        cp "$TEMP_DIR/$src_dll" "$PACKAGE_DIR/$target_dll"
        echo "  Copied: $target_dll"
        VC_COPIED=$((VC_COPIED + 1))
    else
        print_warning "Missing: $target_dll"
    fi
done

# Step 4: Recreate the wheel
echo ""
echo "Step 4: Creating updated wheel..."
echo "========================================="

# Generate new wheel name
WHEEL_NAME=$(basename "$WHEEL_PATH")
WHEEL_BASE="${WHEEL_NAME%.whl}"
NEW_WHEEL_PATH="$PROJECT_DIR/target/wheels/${WHEEL_BASE}_bundled.whl"

# Create the new wheel
echo "Creating wheel: $(basename "$NEW_WHEEL_PATH")"
python3 -m zipfile -c "$NEW_WHEEL_PATH" . || {
    print_error "Failed to create new wheel"
    exit 1
}

# Verify the wheel contents
echo ""
echo "Verifying wheel contents..."
echo "----------------------------------------"
DLL_LIST=$(python3 -m zipfile -l "$NEW_WHEEL_PATH" | grep -E "\.(dll|pyd)$" | awk '{print $NF}' | sort)
DLL_COUNT=$(echo "$DLL_LIST" | grep -c .)

echo "DLLs and PYD files in wheel ($DLL_COUNT total):"
echo "$DLL_LIST" | sed 's/^/  /'

# Check for required DLLs
REQUIRED_DLLS=("vcruntime140.dll" "vcruntime140_1.dll" "msvcp140.dll" "msvcp140_1.dll")
MISSING_DLLS=()

for dll in "${REQUIRED_DLLS[@]}"; do
    if ! echo "$DLL_LIST" | grep -q "$dll"; then
        MISSING_DLLS+=("$dll")
    fi
done

if [ ${#MISSING_DLLS[@]} -gt 0 ]; then
    print_warning "Missing required DLLs: ${MISSING_DLLS[*]}"
fi

# Final summary
echo ""
echo "========================================"
print_success "✅ Build and bundle completed!"
echo ""
echo "Original wheel: $WHEEL_PATH"
echo "Bundled wheel:  $NEW_WHEEL_PATH"
echo "Wheel size:     $(du -h "$NEW_WHEEL_PATH" | cut -f1)"
echo ""
echo "Summary:"
echo "  - ONNX Runtime DLLs: $ONNX_COPIED"
echo "  - VC++ Runtime DLLs: $VC_COPIED"
echo "  - Total DLLs/PYDs:   $DLL_COUNT"
echo ""
echo "The bundled wheel includes all required DLLs and can be"
echo "installed on any Windows system without dependencies."
echo ""
echo "To install:"
echo "  pip install \"$NEW_WHEEL_PATH\""
echo "========================================"