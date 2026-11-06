//! LoRA adapters on a real checkpoint (skipped without
//! models/qwen2.5-0.5b-instruct). The adapters are random, written to disk in
//! PEFT's format and loaded back the way `--lora` loads them.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{mpsc, Arc, Mutex, MutexGuard};

use tachyon::config::Config;
use tachyon::engine::Backend;
use tachyon::kernels::{bf16_to_f32, f32_to_bf16};
use tachyon::kv::{Batch, Logits};
use tachyon::lora::{Adapter, Adapters, MODULES};
use tachyon::model::{CpuBackend, Weights};
use tachyon::pool::Pool;
use tachyon::sampler::{argmax, SamplerConfig};
use tachyon::scheduler::{Event, Request, Scheduler, SchedulerOptions};
use tachyon::tokenizer::Tokenizer;
use tachyon::weights::{Format, Matrix};

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

/// One projection's (out, in, A, B).
type Proj = (usize, usize, Vec<f32>, Vec<f32>);

/// A synthetic adapter: its rank, alpha, and every projection,
/// `[layer][module]`.
struct Synthetic {
    rank: usize,
    alpha: f32,
    rslora: bool,
    mats: Vec<Vec<Proj>>,
}

impl Synthetic {
    fn new(c: &Config, rank: usize, alpha: f32, rslora: bool, seed: u32) -> Synthetic {
        let mut s = seed;
        let mut rand = move || {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
        };
        let shapes = [
            (c.q_dim(), c.hidden),
            (c.kv_dim(), c.hidden),
            (c.kv_dim(), c.hidden),
            (c.hidden, c.q_dim()),
            (c.intermediate, c.hidden),
            (c.intermediate, c.hidden),
            (c.hidden, c.intermediate),
        ];
        let mats = (0..c.layers)
            .map(|_| {
                shapes
                    .iter()
                    .map(|&(out, inp)| {
                        let a = (0..rank * inp)
                            .map(|_| rand() / (inp as f32).sqrt())
                            .collect();
                        let b = (0..out * rank).map(|_| rand() * 0.02).collect();
                        (out, inp, a, b)
                    })
                    .collect()
            })
            .collect();
        Synthetic {
            rank,
            alpha,
            rslora,
            mats,
        }
    }

    fn scale(&self) -> f32 {
        if self.rslora {
            self.alpha / (self.rank as f32).sqrt()
        } else {
            self.alpha / self.rank as f32
        }
    }

    /// Writes `adapter_config.json` and `adapter_model.safetensors`.
    fn write(&self, dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        let targets: Vec<String> = MODULES.iter().map(|m| format!("\"{m}\"")).collect();
        std::fs::write(
            dir.join("adapter_config.json"),
            format!(
                r#"{{"peft_type": "LORA", "r": {}, "lora_alpha": {}, "use_rslora": {},
                    "target_modules": [{}], "modules_to_save": null, "bias": "none"}}"#,
                self.rank,
                self.alpha,
                self.rslora,
                targets.join(", ")
            ),
        )
        .unwrap();
        let mut header = Vec::new();
        let mut data: Vec<u8> = Vec::new();
        for (l, mats) in self.mats.iter().enumerate() {
            for (m, (out, inp, a, b)) in mats.iter().enumerate() {
                let block = if m < 4 { "self_attn" } else { "mlp" };
                let base = format!("base_model.model.model.layers.{l}.{block}.{}", MODULES[m]);
                for (which, shape, v) in [("A", [self.rank, *inp], a), ("B", [*out, self.rank], b)]
                {
                    let start = data.len();
                    data.extend(v.iter().flat_map(|x| x.to_le_bytes()));
                    header.push(format!(
                        r#""{base}.lora_{which}.weight": {{"dtype": "F32", "shape": [{}, {}], "data_offsets": [{start}, {}]}}"#,
                        shape[0],
                        shape[1],
                        data.len()
                    ));
                }
            }
        }
        let header = format!("{{{}}}", header.join(", "));
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header.as_bytes());
        file.extend(data);
        std::fs::write(dir.join("adapter_model.safetensors"), file).unwrap();
    }

    /// Adds `scale * B A` to every projection of `w` (bf16), as merging the
    /// adapter into the checkpoint would.
    fn merge_into(&self, w: &mut Weights, c: &Config) {
        let scale = self.scale() as f64;
        for (layer, mats) in w.layers.iter_mut().zip(&self.mats) {
            let mut row0 = 0;
            for (m, (out, inp, a, b)) in mats.iter().enumerate() {
                let target = match m {
                    0..=2 => &mut layer.wqkv,
                    3 => &mut layer.wo,
                    4 | 5 => &mut layer.w_gate_up,
                    _ => &mut layer.w_down,
                };
                if m == 3 || m == 4 || m == 6 {
                    row0 = 0;
                }
                let Matrix::Bf16 { data, cols, .. } = target else {
                    panic!("merging needs bf16 weights");
                };
                assert_eq!(*cols, *inp);
                for r in 0..*out {
                    for k in 0..*inp {
                        let delta: f64 = (0..self.rank)
                            .map(|j| b[r * self.rank + j] as f64 * a[j * inp + k] as f64)
                            .sum();
                        let at = (row0 + r) * inp + k;
                        data[at] =
                            f32_to_bf16((bf16_to_f32(data[at]) as f64 + scale * delta) as f32);
                    }
                }
                row0 += out;
            }
        }
        let _ = c;
    }
}

fn cpu(cfg: &Config, w: Weights) -> CpuBackend {
    let pool = Arc::new(Pool::new(Pool::default_threads()));
    CpuBackend::new(cfg.clone(), w, pool, 64)
}

/// The last token's logits for `tokens`, run alone with `adapter`.
fn last_logits(b: &mut dyn Backend, tokens: &[u32], adapter: u32) -> Vec<f32> {
    let vocab = b.config().vocab;
    let table: Vec<u32> = (0..b.num_blocks() as u32).collect();
    let mut batch = Batch::default();
    batch.push_with(&table, 0, tokens, Logits::Last, adapter);
    b.forward(&batch)[..vocab].to_vec()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn prompts(dir: &Path) -> Vec<Vec<u32>> {
    let tok = Tokenizer::load(&dir.join("tokenizer.json")).unwrap();
    [
        "The quick brown fox jumps over the lazy dog because",
        "def fibonacci(n):\n    if n < 2:\n        return",
        "Paris is the capital of",
    ]
    .iter()
    .map(|t| tok.encode(t, false))
    .collect()
}

fn adapters(cfg: &Config, scratch: &Path) -> (Vec<Synthetic>, Adapters) {
    let synth = vec![
        Synthetic::new(cfg, 8, 16.0, false, 7),
        Synthetic::new(cfg, 16, 16.0, true, 8),
    ];
    let list = synth
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let dir = scratch.join(format!("adapter{i}"));
            s.write(&dir);
            Adapter::load(&format!("a{i}"), &dir).unwrap()
        })
        .collect();
    let packed = Adapters::new(cfg, list).unwrap();
    (synth, packed)
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tachyon-lora-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

#[test]
fn adapters_match_merged_weights() {
    let Some((_g, dir)) = setup() else { return };
    let cfg = Config::load(&dir).unwrap();
    let tmp = scratch("merge");
    let (synth, packed) = adapters(&cfg, &tmp);
    assert_eq!(packed.rank, 16);
    let mut backend = cpu(&cfg, Weights::load(&dir, &cfg, Format::Bf16).unwrap());
    backend.load_adapters(packed).unwrap();

    for (i, s) in synth.iter().enumerate() {
        let mut w = Weights::load(&dir, &cfg, Format::Bf16).unwrap();
        s.merge_into(&mut w, &cfg);
        let mut merged = cpu(&cfg, w);
        for p in prompts(&dir) {
            let base = last_logits(&mut backend, &p, 0);
            let lora = last_logits(&mut backend, &p, i as u32 + 1);
            let want = last_logits(&mut merged, &p, 0);
            let (effect, err) = (max_diff(&lora, &base), max_diff(&lora, &want));
            eprintln!(
                "adapter {i}: changes logits by {effect:.3}, differs from merged by {err:.4}"
            );
            assert!(
                effect > 1.0,
                "the adapter barely changes anything ({effect})"
            );
            assert!(err < 0.1 * effect, "{err} vs {effect}");
            assert_eq!(argmax(&lora), argmax(&want));
        }
    }
    let _ = std::fs::remove_dir_all(&tmp);
}

/// Sequences with different adapters (and none) in one batch get what they
/// would alone.
fn check_mixed_batch(b: &mut dyn Backend, ps: &[Vec<u32>]) {
    let vocab = b.config().vocab;
    let ids = [1u32, 2, 0];
    let alone: Vec<Vec<f32>> = ps
        .iter()
        .zip(ids)
        .map(|(p, id)| last_logits(b, p, id))
        .collect();
    let mut batch = Batch::default();
    for (i, (p, id)) in ps.iter().zip(ids).enumerate() {
        let table: Vec<u32> = (i as u32 * 16..(i as u32 + 1) * 16).collect();
        batch.push_with(&table, 0, p, Logits::Last, id);
    }
    let together = b.forward(&batch).to_vec();
    for (i, want) in alone.iter().enumerate() {
        let got = &together[i * vocab..(i + 1) * vocab];
        assert!(
            max_diff(got, want) < 1e-3,
            "sequence {i}: {}",
            max_diff(got, want)
        );
        assert_eq!(argmax(got), argmax(want));
    }
}

#[test]
fn mixed_adapter_batches_match_running_alone() {
    let Some((_g, dir)) = setup() else { return };
    let cfg = Config::load(&dir).unwrap();
    let tmp = scratch("mixed");
    let (_, packed) = adapters(&cfg, &tmp);
    let mut backend = cpu(&cfg, Weights::load(&dir, &cfg, Format::Q8).unwrap());
    backend.load_adapters(packed).unwrap();
    check_mixed_batch(&mut backend, &prompts(&dir));
    let _ = std::fs::remove_dir_all(&tmp);
}

#[cfg(target_os = "macos")]
#[test]
fn gpu_adapters_match_cpu() {
    let Some((_g, dir)) = setup() else { return };
    let cfg = Config::load(&dir).unwrap();
    let tmp = scratch("gpu");
    let (_, packed) = adapters(&cfg, &tmp);
    let (_, packed2) = adapters(&cfg, &tmp);
    let w = Weights::load(&dir, &cfg, Format::Bf16).unwrap();
    let mut gpu = match tachyon::gpu::GpuBackend::new(cfg.clone(), w, 64, 1024, false) {
        Ok(g) => g,
        Err(e) => return eprintln!("skipping: {e}"),
    };
    gpu.load_adapters(packed).unwrap();
    let mut backend = cpu(&cfg, Weights::load(&dir, &cfg, Format::Bf16).unwrap());
    backend.load_adapters(packed2).unwrap();
    let ps = prompts(&dir);
    check_mixed_batch(&mut gpu, &ps);
    // Compare what each adapter changes (bf16 weights, so both sides keep
    // activations in floats).
    for (p, id) in ps.iter().zip([1u32, 2]) {
        let change = |b: &mut dyn Backend| -> Vec<f32> {
            let base = last_logits(b, p, 0);
            let with = last_logits(b, p, id);
            with.iter().zip(&base).map(|(a, b)| a - b).collect()
        };
        let (on_gpu, on_cpu) = (change(&mut gpu), change(&mut backend));
        let effect = on_cpu.iter().fold(0f32, |m, v| m.max(v.abs()));
        let diff = max_diff(&on_gpu, &on_cpu);
        eprintln!(
            "adapter {id}: changes logits by {effect:.3}; GPU and CPU changes differ by {diff:.4}"
        );
        assert!(effect > 1.0 && diff < 0.01, "{diff} vs {effect}");
    }
    let _ = std::fs::remove_dir_all(&tmp);
}

/// Submits a greedy request (with `adapter`, by name) and returns its event
/// stream.
fn submit(s: &Scheduler, prompt: &[u32], adapter: Option<&str>, n: usize) -> mpsc::Receiver<Event> {
    let (tx, rx) = mpsc::channel();
    s.submit(Request {
        prompt: prompt.to_vec(),
        max_tokens: n,
        sampler: SamplerConfig {
            temperature: 0.0,
            ..SamplerConfig::default()
        },
        stop_tokens: Vec::new(),
        constraint: None,
        events: tx,
        cancel: Arc::new(AtomicBool::new(false)),
        logprobs: None,
        adapter: adapter.map(str::to_string),
        visible: Vec::new(),
        priority: 0,
    });
    rx
}

/// The whole reply, or the error.
fn reply(rx: mpsc::Receiver<Event>) -> Result<String, String> {
    let mut text = String::new();
    for e in rx {
        match e {
            Event::Text(t) => text.push_str(&t),
            Event::Done(_) => return Ok(text),
            Event::Error(e) => return Err(e),
            Event::Logprob(_) => {}
        }
    }
    Err("no reply".into())
}

/// Adapters loaded and unloaded while serving: requests name them, get
/// what they'd get from adapters loaded at startup, can't name one that is
/// gone, and keep one that is in use from being unloaded. An adapter loaded
/// into a freed id starts with a clean prefix cache.
#[test]
fn adapters_load_and_unload_while_serving() {
    let Some((_g, dir)) = setup() else { return };
    let cfg = Config::load(&dir).unwrap();
    let tmp = scratch("serve");
    let synth = [
        Synthetic::new(&cfg, 8, 16.0, false, 7),
        Synthetic::new(&cfg, 16, 16.0, true, 8),
    ];
    for (i, sy) in synth.iter().enumerate() {
        sy.write(&tmp.join(format!("adapter{i}")));
    }
    let load =
        |name: &str, i: usize| Adapter::load(name, &tmp.join(format!("adapter{i}"))).unwrap();
    let tok = Arc::new(Tokenizer::load(&dir.join("tokenizer.json")).unwrap());
    let s = Scheduler::spawn(
        Box::new(cpu(&cfg, Weights::load(&dir, &cfg, Format::Q8).unwrap())),
        tok.clone(),
        SchedulerOptions {
            max_seqs: 8,
            max_seq_len: 512,
            swap_tokens: 0,
        },
    );
    // Long enough to fill cache blocks, which the next requests reuse.
    let text =
        std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("bench/eval.txt"))
            .unwrap();
    let prompt = tok.encode(&text, false)[..60].to_vec();
    let run = |adapter: Option<&str>| reply(submit(&s, &prompt, adapter, 12));

    assert!(run(Some("a0")).unwrap_err().contains("not loaded"));
    let base = run(None).unwrap();
    s.load_adapter(load("a0", 0)).unwrap();
    let a0 = run(Some("a0")).unwrap();
    assert_ne!(a0, base, "the adapter should change the reply");
    s.load_adapter(load("a1", 1)).unwrap();
    assert_eq!(s.adapters(), ["a0", "a1"]);
    let a1 = run(Some("a1")).unwrap();
    assert_ne!(a1, a0);
    // Padding a0 to a1's rank changes nothing, and the base is untouched.
    assert_eq!(run(Some("a0")).unwrap(), a0);
    assert_eq!(run(None).unwrap(), base);
    assert!(s
        .load_adapter(load("a0", 1))
        .unwrap_err()
        .contains("already loaded"));

    // In use: a long reply holds a1.
    let busy = submit(&s, &prompt, Some("a1"), 200);
    assert!(matches!(busy.recv().unwrap(), Event::Text(_)));
    assert!(s.unload_adapter("a1").unwrap_err().contains("in use"));
    reply(busy).unwrap();

    s.unload_adapter("a0").unwrap();
    assert_eq!(s.adapters(), ["a1"]);
    assert!(run(Some("a0")).unwrap_err().contains("not loaded"));
    assert_eq!(run(Some("a1")).unwrap(), a1);
    // a1's weights under a new name take a0's freed id; a0's cached blocks
    // for this prompt must not leak into it.
    s.load_adapter(load("b", 1)).unwrap();
    assert_eq!(s.adapters(), ["b", "a1"]);
    assert_eq!(run(Some("b")).unwrap(), a1);
    s.unload_adapter("b").unwrap();
    s.unload_adapter("a1").unwrap();
    assert!(s.adapters().is_empty());
    assert_eq!(run(None).unwrap(), base);
    let _ = std::fs::remove_dir_all(&tmp);
}
