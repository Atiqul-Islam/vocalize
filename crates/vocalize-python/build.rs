use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    
    let target = env::var("TARGET").unwrap();
    let out_dir = env::var("OUT_DIR").unwrap();
    
    // Only handle Windows targets  
    if target.contains("windows") {
        setup_windows_onnx_runtime(&out_dir, &target);
        // DLLs will be bundled in the wheel package like ONNX Runtime PyPI package
    }
}

fn setup_windows_onnx_runtime(out_dir: &str, target: &str) {
    println!("cargo:warning=Setting up ONNX Runtime for Windows target: {}", target);
    
    // Download ONNX Runtime if needed
    let onnx_dir = download_onnx_runtime(out_dir);
    
    // Convert WSL path to Windows path if running in WSL
    let lib_location = if is_wsl() {
        let windows_path = wsl_to_windows_path(&onnx_dir.join("lib"));
        println!("cargo:warning=Converted WSL path to Windows: {}", windows_path);
        windows_path
    } else {
        onnx_dir.join("lib").to_string_lossy().to_string()
    };
    
    // Download ONNX Runtime for load-dynamic feature
    // DLLs will be bundled in the Python wheel package (like ONNX Runtime PyPI approach)
    // The ort crate will load them at runtime via ORT_DYLIB_PATH
    
    // VC++ runtime DLLs will be bundled by PowerShell script after build
}

fn download_onnx_runtime(out_dir: &str) -> PathBuf {
    let onnx_version = "1.22.1";
    let onnx_dir = PathBuf::from(out_dir).join("onnxruntime");
    
    // Check if already downloaded
    if onnx_dir.join("lib").join("onnxruntime.dll").exists() {
        println!("cargo:warning=ONNX Runtime already downloaded");
        return onnx_dir;
    }
    
    println!("cargo:warning=Downloading ONNX Runtime v{}", onnx_version);
    
    // Create directory
    fs::create_dir_all(&onnx_dir).expect("Failed to create ONNX Runtime directory");
    
    // Download URL
    let url = format!(
        "https://github.com/microsoft/onnxruntime/releases/download/v{}/onnxruntime-win-x64-{}.zip",
        onnx_version, onnx_version
    );
    
    // Download using curl (available in WSL)
    let zip_path = onnx_dir.join("onnxruntime.zip");
    let output = Command::new("curl")
        .args(&["-L", "-f", "-o", zip_path.to_str().unwrap(), &url])
        .output()
        .expect("Failed to execute curl");
    
    if !output.status.success() {
        eprintln!("curl stderr: {}", String::from_utf8_lossy(&output.stderr));
        panic!("Failed to download ONNX Runtime from: {}", url);
    }
    
    // Extract using Python's zipfile module (cross-platform)
    let extract_script = format!(
        r#"
import zipfile
import os
import shutil

zip_path = '{}'
extract_dir = '{}'

# Extract the zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# Find the extracted directory and move contents up
for item in os.listdir(extract_dir):
    if item.startswith('onnxruntime-win-x64'):
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
        zip_path.to_string_lossy().replace("\\", "/"),
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
    
    // Clean up zip file
    let _ = fs::remove_file(&zip_path);
    
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

