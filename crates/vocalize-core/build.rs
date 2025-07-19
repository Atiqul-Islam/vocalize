use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=ORT_LIB_LOCATION");
    println!("cargo:rerun-if-env-changed=ORT_STRATEGY");
    
    // When using system strategy with our LLVM-built ONNX Runtime
    if env::var("ORT_STRATEGY").as_deref() == Ok("system") {
        // Check if we're cross-compiling for Windows
        let target = env::var("TARGET").unwrap();
        
        if target.contains("windows") {
            // For Windows targets, we need to ensure the LLVM-built DLLs are found
            // The actual path will be set by the build environment
            if let Ok(ort_lib_location) = env::var("ORT_LIB_LOCATION") {
                println!("cargo:warning=Using LLVM-built ONNX Runtime from: {}", ort_lib_location);
                
                // Add the library path for linking
                println!("cargo:rustc-link-search=native={}", ort_lib_location);
                
                // For Windows, we need to copy the DLLs to the output directory
                let out_dir = env::var("OUT_DIR").unwrap();
                let dll_src = PathBuf::from(&ort_lib_location).parent().unwrap().join("bin");
                
                if dll_src.exists() {
                    println!("cargo:warning=DLLs should be copied from: {:?}", dll_src);
                }
            } else {
                println!("cargo:warning=ORT_LIB_LOCATION not set. Please build ONNX Runtime with build-onnxruntime-windows-llvm.sh");
            }
        }
    }
}