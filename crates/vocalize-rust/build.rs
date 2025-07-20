use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    
    let target = env::var("TARGET").unwrap();
    let out_dir = env::var("OUT_DIR").unwrap();
    
    // Setup ONNX Runtime for all supported platforms
    setup_onnx_runtime(&out_dir, &target);
    
    // Note: Library bundling is handled by platform-specific build scripts
    // - Windows: build_and_bundle_complete.sh
    // - Linux: build_linux.sh (or auditwheel)
    // - macOS: build_macos.sh (or delocate)
}

fn setup_onnx_runtime(out_dir: &str, target: &str) {
    println!("cargo:warning=Setting up ONNX Runtime for target: {}", target);
    
    // Download ONNX Runtime if needed
    let onnx_dir = download_onnx_runtime(out_dir, target);
    
    // Convert WSL path to Windows path if running in WSL and building for Windows
    if target.contains("windows") && is_wsl() {
        let lib_location = wsl_to_windows_path(&onnx_dir.join("lib"));
        println!("cargo:warning=Converted WSL path to Windows: {}", lib_location);
    }
    
    // Platform-specific notes:
    // - Windows: DLLs will be bundled by build script after build
    // - Linux/macOS: .so/.dylib files will be bundled by wheel build process
}

fn download_onnx_runtime(out_dir: &str, target: &str) -> PathBuf {
    let onnx_version = "1.22.0";
    let onnx_dir = PathBuf::from(out_dir).join("onnxruntime");
    
    // Determine platform-specific details
    let (platform, arch, lib_name, archive_ext) = if target.contains("windows") {
        let arch = if target.contains("x86_64") { "x64" } else { "x86" };
        ("win", arch, "onnxruntime.dll", "zip")
    } else if target.contains("linux") {
        let arch = if target.contains("x86_64") { "x64" } else if target.contains("aarch64") { "aarch64" } else { "x64" };
        ("linux", arch, "libonnxruntime.so", "tgz")
    } else if target.contains("darwin") {
        let arch = if target.contains("aarch64") { "arm64" } else { "x86_64" };
        ("osx", arch, "libonnxruntime.dylib", "tgz")
    } else {
        panic!("Unsupported platform: {}", target);
    };
    
    // Check if already downloaded
    if onnx_dir.join("lib").join(lib_name).exists() {
        println!("cargo:warning=ONNX Runtime already downloaded");
        return onnx_dir;
    }
    
    println!("cargo:warning=Downloading ONNX Runtime v{} for {}-{}", onnx_version, platform, arch);
    
    // Create directory
    fs::create_dir_all(&onnx_dir).expect("Failed to create ONNX Runtime directory");
    
    // Build download URL
    let url = format!(
        "https://github.com/microsoft/onnxruntime/releases/download/v{}/onnxruntime-{}-{}-{}.{}",
        onnx_version, platform, arch, onnx_version, archive_ext
    );
    
    // Download using curl
    let archive_path = onnx_dir.join(format!("onnxruntime.{}", archive_ext));
    let output = Command::new("curl")
        .args(&["-L", "-f", "-o", archive_path.to_str().unwrap(), &url])
        .output()
        .expect("Failed to execute curl");
    
    if !output.status.success() {
        eprintln!("curl stderr: {}", String::from_utf8_lossy(&output.stderr));
        panic!("Failed to download ONNX Runtime from: {}", url);
    }
    
    // Extract based on archive type
    if archive_ext == "zip" {
        // Extract ZIP file (Windows)
        let extract_script = format!(
            r#"
import zipfile
import os
import shutil

archive_path = '{}'
extract_dir = '{}'

# Extract the zip file
with zipfile.ZipFile(archive_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# Find the extracted directory and move contents up
for item in os.listdir(extract_dir):
    if item.startswith('onnxruntime-'):
        extracted = os.path.join(extract_dir, item)
        for subitem in ['lib', 'include']:
            src = os.path.join(extracted, subitem)
            dst = os.path.join(extract_dir, subitem)
            if os.path.exists(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.move(src, dst)
        # Remove the now-empty directory
        if os.path.exists(extracted) and not os.listdir(extracted):
            os.rmdir(extracted)
        break
"#,
            archive_path.to_string_lossy().replace("\\", "/"),
            onnx_dir.to_string_lossy().replace("\\", "/")
        );
        
        let status = Command::new("python3")
            .arg("-c")
            .arg(&extract_script)
            .status()
            .expect("Failed to execute Python");
        
        if !status.success() {
            panic!("Failed to extract ONNX Runtime");
        }
    } else {
        // Extract TAR.GZ file (Linux/macOS)
        let extract_script = format!(
            r#"
import tarfile
import os
import shutil

archive_path = '{}'
extract_dir = '{}'

# Extract the tar.gz file
with tarfile.open(archive_path, 'r:gz') as tar:
    tar.extractall(extract_dir)

# Find the extracted directory and move contents up
for item in os.listdir(extract_dir):
    if item.startswith('onnxruntime-'):
        extracted = os.path.join(extract_dir, item)
        for subitem in ['lib', 'include']:
            src = os.path.join(extracted, subitem)
            dst = os.path.join(extract_dir, subitem)
            if os.path.exists(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.move(src, dst)
        # Remove the now-empty directory
        if os.path.exists(extracted) and not os.listdir(extracted):
            os.rmdir(extracted)
        break
"#,
            archive_path.to_string_lossy().replace("\\", "/"),
            onnx_dir.to_string_lossy().replace("\\", "/")
        );
        
        let status = Command::new("python3")
            .arg("-c")
            .arg(&extract_script)
            .status()
            .expect("Failed to execute Python");
        
        if !status.success() {
            panic!("Failed to extract ONNX Runtime");
        }
    }
    
    // Clean up archive file
    let _ = fs::remove_file(&archive_path);
    
    println!("cargo:warning=ONNX Runtime downloaded successfully");
    onnx_dir
}

// Removed: copy_dlls_to_package - DLL bundling handled by delvewheel via cibuildwheel

fn is_wsl() -> bool {
    // Check if running in WSL by looking for WSL-specific files
    Path::new("/proc/sys/fs/binfmt_misc/WSLInterop").exists() ||
    env::var("WSL_DISTRO_NAME").is_ok()
}

fn wsl_to_windows_path(wsl_path: &Path) -> String {
    let path_str = wsl_path.to_string_lossy();
    
    // Convert /mnt/c/... to C:\...
    if path_str.starts_with("/mnt/") && path_str.len() > 6 {
        let drive = path_str.chars().nth(5).unwrap();
        let rest = &path_str[6..];
        format!("{}:{}", drive.to_ascii_uppercase(), rest.replace("/", "\\"))
    } else {
        // Fallback for paths not under /mnt
        path_str.replace("/", "\\")
    }
}

// VC++ Runtime functions removed - delvewheel will handle these dependencies

// Removed: download_and_extract_vcredist - replaced by delvewheel

// Removed: try_layout_extraction - replaced by delvewheel

// Removed: try_7zip_extraction - replaced by delvewheel

// Removed: download_vcredist_exe - replaced by delvewheel

// Removed: extract_dlls_from_layout - replaced by delvewheel

// Removed: list_directory_contents - replaced by delvewheel

// Removed: extract_msi_with_msiexec - replaced by delvewheel

// Removed: extract_msi_with_7zip - replaced by delvewheel

// Removed: find_and_copy_dlls_recursive - replaced by delvewheel

// Removed: bundle_onnx_runtime_libs - library bundling handled by platform-specific build scripts

