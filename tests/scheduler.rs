//! The continuous-batching scheduler under memory pressure, on a real
//! checkpoint (skipped without models/qwen2.5-0.5b-instruct): requests that
//! don't all fit in the KV cache are preempted, swapped out or recomputed,
//! and still produce exactly what they would alone.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;

use tachyon::chat::{ChatTemplate, Message, DEFAULT_SYSTEM};
use tachyon::config::Config;
use tachyon::model::{CpuBackend, Weights};
use tachyon::pool::Pool;
use tachyon::sampler::SamplerConfig;
use tachyon::scheduler::{Event, Request, Scheduler, SchedulerOptions, Summary};
use tachyon::tokenizer::Tokenizer;
use tachyon::weights::Format;

static SERIAL: Mutex<()> = Mutex::new(());

fn setup() -> Option<(MutexGuard<'static, ()>, PathBuf)> {
    let guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/qwen2.5-0.5b-instruct");
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} not found", dir.display());
        return None;
    }
    Some((guard, dir))
}

const QUESTIONS: [&str; 4] = [
    "Describe how a bicycle works.",
    "Write a short story about a lighthouse keeper.",
    "Explain why the sky is blue.",
    "List some tips for learning to cook.",
];

fn scheduler(dir: &Path, blocks: usize, max_seqs: usize, swap_tokens: usize) -> Scheduler {
    let cfg = Config::load(dir).unwrap();
    let w = Weights::load(dir, &cfg, Format::Q8).unwrap();
    let pool = Arc::new(Pool::new(Pool::default_threads()));
    let tok = Arc::new(Tokenizer::load(&dir.join("tokenizer.json")).unwrap());
    Scheduler::spawn(
        Box::new(CpuBackend::new(cfg, w, pool, blocks)),
        tok,
        SchedulerOptions {
            max_seqs,
            max_seq_len: 1024,
            swap_tokens,
        },
    )
}

/// Submits a greedy request that never stops early, returning its events.
fn submit(
    s: &Scheduler,
    prompt: Vec<u32>,
    max_tokens: usize,
    priority: i32,
) -> mpsc::Receiver<Event> {
    let (tx, rx) = mpsc::channel();
    s.submit(Request {
        prompt,
        max_tokens,
        sampler: SamplerConfig {
            temperature: 0.0,
            ..SamplerConfig::default()
        },
        stop_tokens: Vec::new(),
        constraint: None,
        events: tx,
        cancel: Arc::new(AtomicBool::new(false)),
        logprobs: None,
        adapter: None,
        visible: Vec::new(),
        priority,
    });
    rx
}

fn collect(rx: mpsc::Receiver<Event>) -> (String, Summary) {
    let mut text = String::new();
    for e in rx {
        match e {
            Event::Text(t) => text.push_str(&t),
            Event::Done(s) => return (text, s),
            Event::Error(e) => panic!("{e}"),
            Event::Logprob(_) => {}
        }
    }
    panic!("no summary")
}

fn prompts(dir: &Path) -> Vec<Vec<u32>> {
    let tok = Tokenizer::load(&dir.join("tokenizer.json")).unwrap();
    let tpl = ChatTemplate::new(&tok).unwrap();
    QUESTIONS
        .iter()
        .map(|q| tpl.encode(&tok, &[Message::new("user", q)], Some(DEFAULT_SYSTEM)))
        .collect()
}

/// Every request at once; their texts, and the scheduler's preemption and
/// swap counts.
fn run_all(s: Scheduler, ps: &[Vec<u32>], n: usize) -> (Vec<String>, u64, u64) {
    let rxs: Vec<_> = ps.iter().map(|p| submit(&s, p.clone(), n, 0)).collect();
    let texts = rxs.into_iter().map(|rx| collect(rx).0).collect();
    let st = &s.stats;
    (
        texts,
        st.preemptions.load(Ordering::Relaxed),
        st.swaps.load(Ordering::Relaxed),
    )
}

#[test]
fn preempted_requests_match_running_alone() {
    let Some((_g, dir)) = setup() else { return };
    let ps = prompts(&dir);
    let n = 120;
    // Room for everything at once.
    let (want, pre, _) = run_all(scheduler(&dir, 64, 8, 0), &ps, n);
    assert_eq!(pre, 0);
    // Four requests of ~160 tokens in a 320-token cache.
    let (swapped, pre, swaps) = run_all(scheduler(&dir, 20, 8, 4096), &ps, n);
    eprintln!("swap space: {pre} preemptions, {swaps} swapped out");
    assert!(pre > 0 && swaps == pre);
    assert_eq!(swapped, want);
    let (recomputed, pre, swaps) = run_all(scheduler(&dir, 20, 8, 0), &ps, n);
    eprintln!("no swap space: {pre} preemptions, recomputed");
    assert!(pre > 0 && swaps == 0);
    assert_eq!(recomputed, want);
}

#[test]
fn urgent_requests_go_first() {
    let Some((_g, dir)) = setup() else { return };
    let ps = prompts(&dir);
    // One at a time: while the first runs, the rest queue by priority.
    let s = scheduler(&dir, 64, 1, 0);
    let order = Arc::new(Mutex::new(Vec::new()));
    let first = submit(&s, ps[0].clone(), 60, 0);
    // Once it is generating, queue the others.
    while !matches!(first.recv().unwrap(), Event::Text(_)) {}
    let rest: Vec<_> = [(1, 5), (2, 0), (3, -1)]
        .into_iter()
        .map(|(i, p)| (i, submit(&s, ps[i].clone(), 8, p)))
        .collect();
    let waits: Vec<_> = std::iter::once((0, first))
        .chain(rest)
        .map(|(i, rx)| {
            let order = order.clone();
            thread::spawn(move || {
                collect(rx);
                order.lock().unwrap().push(i);
            })
        })
        .collect();
    waits.into_iter().for_each(|w| w.join().unwrap());
    assert_eq!(*order.lock().unwrap(), [0, 3, 2, 1]);
}

/// Blocks swapped out of one place in the GPU's cache and into another give
/// the same next-token logits.
#[cfg(target_os = "macos")]
#[test]
fn gpu_blocks_swap_out_and_back_in() {
    use tachyon::engine::Backend;
    use tachyon::kv::{Batch, Logits};
    let Some((_g, dir)) = setup() else { return };
    let cfg = Config::load(&dir).unwrap();
    let w = Weights::load(&dir, &cfg, Format::Q8).unwrap();
    let mut gpu = match tachyon::gpu::GpuBackend::new(cfg.clone(), w, 16, 256, true) {
        Ok(g) => g,
        Err(e) => return eprintln!("skipping: {e}"),
    };
    let prompt = &prompts(&dir)[1][..32];
    let mut batch = Batch::default();
    batch.push(&[3, 5, 6], 0, prompt, Logits::None);
    gpu.forward(&batch);
    let next = |gpu: &mut tachyon::gpu::GpuBackend, table: &[u32]| {
        let mut b = Batch::default();
        b.push(table, 32, &[prompt[0]], Logits::Last);
        gpu.forward(&b)[..cfg.vocab].to_vec()
    };
    let want = next(&mut gpu, &[3, 5, 6]);
    let saved = gpu.save_blocks(&[3, 5]).unwrap();
    gpu.load_blocks(&[9, 12], &saved);
    assert_eq!(gpu.save_blocks(&[9, 12]).unwrap(), saved);
    assert_eq!(next(&mut gpu, &[9, 12, 6]), want);
}
