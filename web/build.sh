#!/usr/bin/env bash
# Builds the browser version (WebAssembly with SIMD) into web/tachyon.wasm.
# Serve the repository root and open /web/ (see the README).
set -euo pipefail
cd "$(dirname "$0")/.."
RUSTFLAGS="-C target-feature=+simd128" CARGO_TARGET_DIR=target/web \
  cargo rustc --release --lib --target wasm32-unknown-unknown --crate-type cdylib
cp target/web/wasm32-unknown-unknown/release/tachyon.wasm web/tachyon.wasm
echo "built web/tachyon.wasm ($(wc -c < web/tachyon.wasm) bytes)"
