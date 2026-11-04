//! LoRA adapters on a real checkpoint (skipped without
//! models/qwen2.5-0.5b-instruct). The adapters are random, written to disk in
//! PEFT's format and loaded back the way `--lora` loads them.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use tachyon::config::Config;
use tachyon::engine::Backend;
use tachyon::kernels::{bf16_to_f32, f32_to_bf16};
use tachyon::kv::{Batch, Logits};
use tachyon::lora::{Adapter, Adapters, MODULES};
use tachyon::model::{CpuBackend, Weights};
use tachyon::pool::Pool;
use tachyon::sampler::argmax;
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
