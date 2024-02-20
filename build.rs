use std::{env, path::PathBuf};

fn main() {
    let cuda_root = std::env::var("CUDA_PATH").unwrap();

    println!("cargo:rustc-link-search={cuda_root}/lib/x64");
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cudadevrt");
    println!("cargo:rustc-link-lib=nvrtc");
    println!("cargo:rustc-link-lib=nvjpeg");

    let cuda_header = format!("{cuda_root}/include/cuda.h");
    let nvjpeg_header = format!("{cuda_root}/include/nvjpeg.h");
    let nvrtc_header = format!("{cuda_root}/include/nvrtc.h");
    println!("cargo:rerun-if-changed='{cuda_header}'");
    println!("cargo:rerun-if-changed='{nvjpeg_header}'");
    println!("cargo:rerun-if-changed='{nvrtc_header}'");

    let bindings = bindgen::Builder::default()
        .clang_arg(format!("-I{cuda_root}"))
        .header(cuda_header)
        .header(nvjpeg_header)
        .header(nvrtc_header)
        .generate_inline_functions(true)
        .generate()
        .expect("unable to generate nvjpeg bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("bindings file write failed");
}
