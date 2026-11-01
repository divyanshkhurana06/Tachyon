//! Tachyon: an LLM inference engine written from scratch.
//!
//! Loads Hugging Face checkpoints (Qwen2 / Llama-style transformers) straight
//! from `.safetensors`, tokenizes with its own byte-level BPE, and runs the
//! model on the CPU (NEON + a spin-then-park thread pool) or on Apple GPUs
//! (Metal compute shaders), in bf16 or quantized to 8 or 4 bits.

use std::fmt;

pub mod chat;
pub mod config;
pub mod engine;
pub mod generate;
pub mod json;
pub mod kernels;
pub mod model;
pub mod pool;
pub mod safetensors;
pub mod sampler;
pub mod tokenizer;
pub mod weights;

#[derive(Debug)]
pub struct Error(pub String);

impl Error {
    pub fn msg(s: impl Into<String>) -> Error {
        Error(s.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Error {
        Error(e.to_string())
    }
}

impl From<String> for Error {
    fn from(s: String) -> Error {
        Error(s)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[macro_export]
macro_rules! bail {
    ($($arg:tt)*) => { return Err($crate::Error(format!($($arg)*))) };
}
