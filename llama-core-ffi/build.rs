use std::path::PathBuf;

fn main() {
    let crate_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = crate_dir.join("include");
    std::fs::create_dir_all(&out_dir).expect("create include dir");

    cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_language(cbindgen::Language::C)
        .with_include_guard("LLAMA_CORE_FFI_H")
        .with_cpp_compat(true)
        .generate()
        .expect("cbindgen generation")
        .write_to_file(out_dir.join("llama-core-ffi.h"));

    println!("cargo:rerun-if-changed=src/lib.rs");
}
