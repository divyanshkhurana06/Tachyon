//! Compiles the Objective-C bridge to Metal (src/gpu/shim.m) with the
//! system clang and links it in. No crates: just clang, ar and the SDK.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/gpu/shim.m");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let obj = out.join("shim.o");
    let lib = out.join("libtachyon_metal.a");

    let status = Command::new("clang")
        .args(["-c", "-O2", "-fobjc-arc", "-Wall", "src/gpu/shim.m", "-o"])
        .arg(&obj)
        .status()
        .expect("could not run clang (install the Xcode command line tools)");
    assert!(status.success(), "compiling src/gpu/shim.m failed");

    let _ = std::fs::remove_file(&lib);
    let status = Command::new("ar")
        .arg("crs")
        .arg(&lib)
        .arg(&obj)
        .status()
        .expect("could not run ar");
    assert!(status.success(), "ar failed");

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=tachyon_metal");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    // Without CoreGraphics, command-line tools may see no default GPU.
    println!("cargo:rustc-link-lib=framework=CoreGraphics");
    println!("cargo:rustc-link-lib=dylib=objc");
}
