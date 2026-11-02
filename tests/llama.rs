//! End-to-end checks against Llama 3.2 1B Instruct. They need the model in
//! models/llama-3.2-1b-instruct (see the README) and are skipped without it.

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
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/llama-3.2-1b-instruct");
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
fn tokenizer_and_template_match_llama_3() {
    let Some((_g, dir)) = setup() else { return };
    let tok = Tokenizer::load(&dir.join("tokenizer.json")).unwrap();
    assert_eq!(tok.encode("Hello world!", false), [9906, 1917, 0]);
    // Digits go in threes, and a space before them stands alone.
    let ids = tok.encode("in 2023", false);
    let pieces: Vec<String> = ids.iter().map(|&i| tok.decode(&[i], false)).collect();
    assert_eq!(pieces, ["in", " ", "202", "3"]);

    // The prompt Llama 3.2's own template renders, up to the date.
    let tpl = ChatTemplate::new(&tok).unwrap();
    let p = tpl.encode(&tok, &[Message::new("user", "Hi!")], None);
    assert_eq!(
        p[..18],
        [
            128000, 128006, 9125, 128007, 271, 38766, 1303, 33025, 2696, 25, 6790, 220, 2366, 18,
            198, 15724, 2696, 25
        ]
    );
    assert_eq!(p[p.len() - 4..], [128006, 78191, 128007, 271]);
    assert_eq!(tpl.end_of_turn(), 128009);

    let text = std::fs::read_to_string(dir.join("../../bench/eval.txt")).unwrap();
    assert_eq!(tok.decode(&tok.encode(&text, false), false), text);
}

#[test]
fn answers_and_gpu_matches_cpu() {
    let Some((_g, dir)) = setup() else { return };
    let tok = Tokenizer::load(&dir.join("tokenizer.json")).unwrap();
    let cfg = Config::load(&dir).unwrap();
    assert!(cfg.rope_scaling.is_some());
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
