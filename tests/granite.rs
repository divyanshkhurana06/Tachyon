//! End-to-end checks against Granite 3.1 1B-A400M Instruct, a mixture of
//! experts (32 per layer, 8 per token). They need the model in
//! models/granite-3.1-1b-a400m-instruct (see the README) and are skipped
//! without it.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use tachyon::chat::{ChatTemplate, Message};
use tachyon::config::Config;
use tachyon::engine::{Engine, Session};
use tachyon::model::{CpuBackend, Weights};
use tachyon::pool::Pool;
use tachyon::sampler::argmax;
use tachyon::tokenizer::Tokenizer;
use tachyon::weights::Format;

static SERIAL: Mutex<()> = Mutex::new(());

fn setup() -> Option<(MutexGuard<'static, ()>, PathBuf)> {
    let guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/granite-3.1-1b-a400m-instruct");
    if dir.join("model.safetensors").exists() {
        Some((guard, dir))
    } else {
        eprintln!("skipping: {} not found", dir.display());
        None
    }
}

fn greedy(e: &mut dyn Engine, prompt: &[u32], n: usize, stop: u32) -> Vec<u32> {
    let vocab = e.config().vocab;
    e.truncate(0);
    let mut out = vec![argmax(&e.forward(prompt, false)[..vocab])];
    while out.len() < n && *out.last().unwrap() != stop {
        let next = argmax(&e.forward(&out[out.len() - 1..], false)[..vocab]);
        out.push(next);
    }
    out
}

#[test]
fn tokenizer_and_template() {
    let Some((_g, dir)) = setup() else { return };
    let tok = Tokenizer::load(&dir.join("tokenizer.json")).unwrap();
    let text = std::fs::read_to_string(dir.join("../../bench/eval.txt")).unwrap();
    assert_eq!(tok.decode(&tok.encode(&text, false), false), text);
    let tpl = ChatTemplate::new(&tok).unwrap();
    let p = tpl.encode(&tok, &[Message::new("user", "Hi!")], None);
    let shown = tok.decode(&p, false);
    assert!(shown.starts_with(
        "<|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.\nToday's Date: "
    ));
    assert!(shown.ends_with(
        "You are Granite, developed by IBM.<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>Hi!<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
    ));
    assert_eq!(tpl.end_of_turn(), 0);
}

/// The experts are routed the same on both backends: identical greedy
/// tokens, through the per-(token, expert) path for decode and the
/// per-expert GEMMs for the prompt.
#[test]
fn answers_and_gpu_matches_cpu() {
    let Some((_g, dir)) = setup() else { return };
    let tok = Tokenizer::load(&dir.join("tokenizer.json")).unwrap();
    let cfg = Config::load(&dir).unwrap();
    assert_eq!((cfg.experts, cfg.experts_per_token), (32, 8));
    let tpl = ChatTemplate::new(&tok).unwrap();
    let p = tpl.encode(
        &tok,
        &[Message::new(
            "user",
            "What is the capital of France? Answer in one sentence.",
        )],
        None,
    );
    let pool = Arc::new(Pool::new(Pool::default_threads()));
    let w = Weights::load(&dir, &cfg, Format::Bf16).unwrap();
    let mut cpu = Session::new(Box::new(CpuBackend::new(cfg.clone(), w, pool, 16)));
    let on_cpu = greedy(&mut cpu, &p, 24, tpl.end_of_turn());
    let reply = tok.decode(&on_cpu, true);
    assert!(reply.contains("Paris"), "{reply:?}");

    #[cfg(target_os = "macos")]
    {
        let w = Weights::load(&dir, &cfg, Format::Bf16).unwrap();
        match tachyon::gpu::GpuBackend::new(cfg.clone(), w, 16, 256, false) {
            Ok(g) => {
                let on_gpu = greedy(&mut Session::new(Box::new(g)), &p, 24, tpl.end_of_turn());
                assert_eq!(on_gpu, on_cpu);
            }
            Err(e) => eprintln!("skipping the GPU: {e}"),
        }
    }
}
