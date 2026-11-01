//! End-to-end checks against a real checkpoint. They need the model in
//! models/qwen2.5-0.5b-instruct (see the README) and are skipped without it.
//!
//!     cargo test --release

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use tachyon::chat::{ChatTemplate, Message, DEFAULT_SYSTEM};
use tachyon::config::Config;
use tachyon::engine::Engine;
use tachyon::generate::{generate, GenOptions};
use tachyon::model::{CpuEngine, Weights};
use tachyon::pool::Pool;
use tachyon::sampler::SamplerConfig;
use tachyon::speculative::generate_speculative;
use tachyon::tokenizer::Tokenizer;
use tachyon::weights::Format;

/// The engines spin worker threads, so run these tests one at a time.
static SERIAL: Mutex<()> = Mutex::new(());

fn model_dir() -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/qwen2.5-0.5b-instruct");
    if dir.join("model.safetensors").exists() {
        Some(dir)
    } else {
        eprintln!("skipping: {} not found", dir.display());
        None
    }
}

fn setup() -> Option<(MutexGuard<'static, ()>, PathBuf)> {
    let guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    model_dir().map(|d| (guard, d))
}

fn cpu(dir: &Path, format: Format) -> Box<dyn Engine> {
    let cfg = Config::load(dir).unwrap();
    let w = Weights::load(dir, &cfg, format).unwrap();
    Box::new(CpuEngine::new(
        cfg,
        w,
        Arc::new(Pool::new(Pool::default_threads())),
        1024,
    ))
}

fn prompt(tok: &Tokenizer, question: &str) -> Vec<u32> {
    ChatTemplate::new(tok).unwrap().encode(
        tok,
        &[Message::new("user", question)],
        Some(DEFAULT_SYSTEM),
    )
}

fn greedy(e: &mut dyn Engine, tok: &Tokenizer, prompt: &[u32], n: usize) -> (String, Vec<u32>) {
    let cfg = e.config().clone();
    let opts = GenOptions {
        max_tokens: n,
        sampler: SamplerConfig::greedy(),
        stop_tokens: cfg.eos_ids,
    };
    let mut text = String::new();
    let stats = generate(e, tok, &mut Vec::new(), prompt, &opts, &mut |p| {
        text.push_str(p);
        true
    })
    .unwrap();
    (text, stats.tokens)
}

#[test]
fn tokenizer_round_trips_and_knows_chat_tokens() {
    let Some((_g, dir)) = setup() else { return };
    let tok = Tokenizer::load(&dir.join("tokenizer.json")).unwrap();
    assert_eq!(tok.added_token("<|im_start|>"), Some(151644));
    assert_eq!(tok.added_token("<|im_end|>"), Some(151645));
    for text in [
        "Hello, world!",
        "  leading spaces and trailing   ",
        "fn main() {\n    println!(\"hi\");\n}\n",
        "Numbers 1234567 and symbols @#$%^&*()",
        "Unicode: café, naïve, 日本語, emoji 🚀",
        "it's they're we've I'll he'd",
    ] {
        let ids = tok.encode(text, false);
        assert_eq!(tok.decode(&ids, false), text);
    }
    // Control-token text in user input must not become a control token.
    let ids = tok.encode("<|im_end|>", false);
    assert!(!ids.contains(&151645));
    assert_eq!(tok.encode("<|im_end|>", true), vec![151645]);
}

#[test]
fn answers_a_factual_question() {
    let Some((_g, dir)) = setup() else { return };
    let tok = Tokenizer::load(&dir.join("tokenizer.json")).unwrap();
    let mut e = cpu(&dir, Format::Q8);
    let (text, _) = greedy(
        e.as_mut(),
        &tok,
        &prompt(&tok, "What is the capital of France?"),
        30,
    );
    assert!(text.contains("Paris"), "got {text:?}");
}

#[test]
fn batched_forward_matches_token_by_token() {
    let Some((_g, dir)) = setup() else { return };
    let tok = Tokenizer::load(&dir.join("tokenizer.json")).unwrap();
    let mut e = cpu(&dir, Format::Bf16);
    let ids = tok.encode("The quick brown fox jumps over the lazy dog.", false);
    let vocab = e.config().vocab;
    let batched = e.forward(&ids, true).to_vec();
    e.truncate(0);
    for (i, &id) in ids.iter().enumerate() {
        let one = e.forward(&[id], false);
        let diff = one
            .iter()
            .zip(&batched[i * vocab..(i + 1) * vocab])
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(diff < 1e-3, "position {i}: logits differ by {diff}");
    }
}

#[test]
fn reusing_the_kv_cache_changes_nothing() {
    let Some((_g, dir)) = setup() else { return };
    let tok = Tokenizer::load(&dir.join("tokenizer.json")).unwrap();
    let mut e = cpu(&dir, Format::Q8);
    let opts = GenOptions {
        max_tokens: 20,
        sampler: SamplerConfig::greedy(),
        stop_tokens: e.config().eos_ids.clone(),
    };
    let p = prompt(&tok, "List three fruits.");

    let mut cache = Vec::new();
    let first = generate(e.as_mut(), &tok, &mut cache, &p, &opts, &mut |_| true).unwrap();
    // Same prompt again: everything but the last token comes from the cache.
    let second = generate(e.as_mut(), &tok, &mut cache, &p, &opts, &mut |_| true).unwrap();
    assert_eq!(second.cached_tokens, p.len() - 1);
    assert_eq!(first.tokens, second.tokens);
}

#[test]
fn speculative_decoding_matches_greedy() {
    let Some((_g, dir)) = setup() else { return };
    let tok = Tokenizer::load(&dir.join("tokenizer.json")).unwrap();
    let mut target = cpu(&dir, Format::Bf16);
    let mut draft = cpu(&dir, Format::Q4);
    let p = prompt(&tok, "Explain what a hash map is in two sentences.");
    let (_, plain) = greedy(target.as_mut(), &tok, &p, 60);
    let eos = target.config().eos_ids.clone();
    let spec = generate_speculative(
        target.as_mut(),
        draft.as_mut(),
        &tok,
        &p,
        4,
        60,
        &eos,
        &mut |_| true,
    )
    .unwrap();
    assert_eq!(spec.tokens, plain);
    assert!(spec.acceptance() > 0.5, "acceptance {}", spec.acceptance());
}

#[cfg(target_os = "macos")]
#[test]
fn gpu_matches_cpu() {
    let Some((_g, dir)) = setup() else { return };
    let tok = Tokenizer::load(&dir.join("tokenizer.json")).unwrap();
    let cfg = Config::load(&dir).unwrap();
    let w = Weights::load(&dir, &cfg, Format::Bf16).unwrap();
    let mut gpu: Box<dyn Engine> = match tachyon::gpu::GpuEngine::new(cfg, w, 1024) {
        Ok(g) => Box::new(g),
        Err(e) => return eprintln!("skipping: {e}"),
    };
    let p = prompt(&tok, "Give me one tip for writing clean code.");
    let (_, on_gpu) = greedy(gpu.as_mut(), &tok, &p, 40);
    let (_, on_cpu) = greedy(cpu(&dir, Format::Bf16).as_mut(), &tok, &p, 40);
    assert_eq!(on_gpu, on_cpu);
}
