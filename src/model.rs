//! Transformer weights and the CPU forward pass, for the Qwen2 / Llama
//! family: pre-norm decoder blocks with RMSNorm, grouped-query attention with
//! rotary position embeddings, and a SiLU-gated MLP.

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::thread;

use crate::config::Config;
use crate::engine::Engine;
use crate::kernels::{axpy, dot_f32, rmsnorm, silu};
use crate::pool::{Pool, SyncPtr};
use crate::safetensors::SafeTensors;
use crate::weights::{Format, Matrix};
use crate::{bail, Result};

pub struct LayerWeights {
    pub attn_norm: Vec<f32>,
    /// q, k and v projections stacked into one matrix: one pass over the
    /// input instead of three.
    pub wqkv: Matrix,
    pub bqkv: Option<Vec<f32>>,
    pub wo: Matrix,
    pub mlp_norm: Vec<f32>,
    /// gate and up projections, stacked the same way.
    pub w_gate_up: Matrix,
    pub w_down: Matrix,
}

pub struct Weights {
    pub format: Format,
    pub embed: Matrix,
    /// `None` when the output projection shares the embedding matrix.
    pub lm_head: Option<Matrix>,
    pub layers: Vec<LayerWeights>,
    pub norm: Vec<f32>,
}

impl Weights {
    /// Reads the checkpoint and converts it to `format`, one layer per core.
    pub fn load(dir: &Path, c: &Config, format: Format) -> Result<Weights> {
        let st = SafeTensors::open_dir(dir)?;
        let next = AtomicUsize::new(0);
        let done = Mutex::new(Vec::new());
        let threads = Pool::default_threads().min(c.layers + 1);
        // The output layer is the most sensitive matrix to quantize, so
        // 4-bit models keep it (and the embedding it may share) at 8 bits,
        // as llama.cpp does.
        let outer = if format == Format::Q4 {
            Format::Q8
        } else {
            format
        };

        // The embedding (the single biggest matrix) loads alongside the layers.
        let embed = thread::scope(|s| {
            let embed = s.spawn(|| matrix(&st, &["model.embed_tokens.weight"], outer));
            for _ in 0..threads {
                s.spawn(|| loop {
                    let l = next.fetch_add(1, Relaxed);
                    if l >= c.layers {
                        break;
                    }
                    let layer = load_layer(&st, c, l, format);
                    done.lock().unwrap().push((l, layer));
                });
            }
            embed.join().unwrap()
        })?;

        let mut done = done.into_inner().unwrap();
        done.sort_by_key(|(l, _)| *l);
        let layers = done
            .into_iter()
            .map(|(_, l)| l)
            .collect::<Result<Vec<_>>>()?;
        let lm_head = if c.tied_embeddings || !st.contains("lm_head.weight") {
            None
        } else {
            Some(matrix(&st, &["lm_head.weight"], outer)?)
        };
        if embed.rows() < c.vocab || embed.cols() != c.hidden {
            bail!(
                "embedding is {}x{}, expected {}x{}",
                embed.rows(),
                embed.cols(),
                c.vocab,
                c.hidden
            );
        }
        Ok(Weights {
            format,
            embed,
            lm_head,
            layers,
            norm: vector(&st, "model.norm.weight")?,
        })
    }

    pub fn bytes(&self) -> usize {
        let layer = |l: &LayerWeights| {
            l.wqkv.bytes() + l.wo.bytes() + l.w_gate_up.bytes() + l.w_down.bytes()
        };
        self.embed.bytes()
            + self.lm_head.as_ref().map_or(0, Matrix::bytes)
            + self.layers.iter().map(layer).sum::<usize>()
    }

    pub fn lm_head(&self) -> &Matrix {
        self.lm_head.as_ref().unwrap_or(&self.embed)
    }
}

fn load_layer(st: &SafeTensors, c: &Config, l: usize, format: Format) -> Result<LayerWeights> {
    let n = |s: &str| format!("model.layers.{l}.{s}");
    let biases = ["q_proj", "k_proj", "v_proj"].map(|p| n(&format!("self_attn.{p}.bias")));
    let bqkv = if biases.iter().all(|b| st.contains(b)) {
        let mut b = Vec::new();
        for name in &biases {
            b.extend(st.tensor(name)?.to_f32());
        }
        Some(b)
    } else {
        None
    };
    let layer = LayerWeights {
        attn_norm: vector(st, &n("input_layernorm.weight"))?,
        wqkv: matrix(
            st,
            &[
                &n("self_attn.q_proj.weight"),
                &n("self_attn.k_proj.weight"),
                &n("self_attn.v_proj.weight"),
            ],
            format,
        )?,
        bqkv,
        wo: matrix(st, &[&n("self_attn.o_proj.weight")], format)?,
        mlp_norm: vector(st, &n("post_attention_layernorm.weight"))?,
        w_gate_up: matrix(
            st,
            &[&n("mlp.gate_proj.weight"), &n("mlp.up_proj.weight")],
            format,
        )?,
        w_down: matrix(st, &[&n("mlp.down_proj.weight")], format)?,
    };
    let qkv = c.q_dim() + 2 * c.kv_dim();
    let shapes = [
        (&layer.wqkv, qkv, c.hidden),
        (&layer.wo, c.hidden, c.q_dim()),
        (&layer.w_gate_up, 2 * c.intermediate, c.hidden),
        (&layer.w_down, c.hidden, c.intermediate),
    ];
    for (m, rows, cols) in shapes {
        if (m.rows(), m.cols()) != (rows, cols) {
            bail!(
                "layer {l}: a matrix is {}x{}, expected {rows}x{cols}",
                m.rows(),
                m.cols()
            );
        }
    }
    Ok(layer)
}

/// Stacks the named 2-D tensors row-wise into one matrix.
fn matrix(st: &SafeTensors, names: &[&str], format: Format) -> Result<Matrix> {
    let mut data = Vec::new();
    let (mut rows, mut cols) = (0, None);
    for name in names {
        let t = st.tensor(name)?;
        let &[r, c] = t.shape else {
            bail!("{name} is not a matrix")
        };
        if cols.is_some_and(|cols| cols != c) {
            bail!("{name} has {c} columns, unlike the tensors stacked with it");
        }
        cols = Some(c);
        rows += r;
        data.extend(t.to_f32());
    }
    Ok(Matrix::from_f32(&data, rows, cols.unwrap_or(0), format))
}

fn vector(st: &SafeTensors, name: &str) -> Result<Vec<f32>> {
    Ok(st.tensor(name)?.to_f32())
}

/// Rotary embedding frequencies: `theta^(-2i/d)` for each pair of dims.
pub fn rope_frequencies(c: &Config) -> Vec<f64> {
    let half = c.head_dim / 2;
    (0..half)
        .map(|i| c.rope_theta.powf(-2.0 * i as f64 / c.head_dim as f64))
        .collect()
}

/// Rotates each head's dims `i` and `i + d/2` by `pos * freq[i]`
/// (the "rotate half" layout Hugging Face checkpoints use).
fn rope(v: &mut [f32], head_dim: usize, cos: &[f32], sin: &[f32]) {
    let half = head_dim / 2;
    for head in v.chunks_exact_mut(head_dim) {
        let (a, b) = head.split_at_mut(half);
        for i in 0..half {
            let (x, y) = (a[i], b[i]);
            a[i] = x * cos[i] - y * sin[i];
            b[i] = y * cos[i] + x * sin[i];
        }
    }
}

#[derive(Default)]
struct Scratch {
    x: Vec<f32>,
    xb: Vec<f32>,
    qkv: Vec<f32>,
    att: Vec<f32>,
    proj: Vec<f32>,
    gate_up: Vec<f32>,
    hidden: Vec<f32>,
    logits: Vec<f32>,
}

fn zeroed(v: &mut Vec<f32>, n: usize) {
    v.clear();
    v.resize(n, 0.0);
}

pub struct CpuEngine {
    cfg: Config,
    w: Weights,
    pool: Arc<Pool>,
    freqs: Vec<f64>,
    max_seq: usize,
    pos: usize,
    /// `[layer][position][kv_dim]`
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
    s: Scratch,
}

impl CpuEngine {
    pub fn new(cfg: Config, w: Weights, pool: Arc<Pool>, max_seq: usize) -> CpuEngine {
        let max_seq = max_seq.min(cfg.max_positions);
        let cache = cfg.layers * max_seq * cfg.kv_dim();
        CpuEngine {
            freqs: rope_frequencies(&cfg),
            k_cache: vec![0.0; cache],
            v_cache: vec![0.0; cache],
            cfg,
            w,
            pool,
            max_seq,
            pos: 0,
            s: Scratch::default(),
        }
    }

    pub fn weights(&self) -> &Weights {
        &self.w
    }
}

/// Causal attention for `t` new tokens of one layer: token `i` attends to
/// every cached position up to and including its own. One task per
/// (token, head); heads in a group share the same K/V head.
#[allow(clippy::too_many_arguments)]
fn attention(
    c: &Config,
    pool: &Pool,
    k_cache: &[f32],
    v_cache: &[f32],
    pos0: usize,
    t: usize,
    qkv: &[f32],
    att: &mut Vec<f32>,
) {
    let (hd, heads, kv_dim, q_dim) = (c.head_dim, c.heads, c.kv_dim(), c.q_dim());
    let group = heads / c.kv_heads;
    let qkv_dim = q_dim + 2 * kv_dim;
    let scale = 1.0 / (hd as f32).sqrt();
    zeroed(att, t * q_dim);
    let out = SyncPtr::new(att);

    pool.parallel_for(t * heads, 1, |tasks| {
        let mut scores = Vec::new();
        for task in tasks {
            let (i, h) = (task / heads, task % heads);
            let q = &qkv[i * qkv_dim + h * hd..][..hd];
            let kv = (h / group) * hd;
            let n = pos0 + i + 1;

            scores.clear();
            scores.extend((0..n).map(|p| dot_f32(q, &k_cache[p * kv_dim + kv..][..hd]) * scale));
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for s in scores.iter_mut() {
                *s = (*s - max).exp();
                sum += *s;
            }
            // SAFETY: each (token, head) owns its own slice of `att`.
            let o = unsafe { out.slice(i * q_dim + h * hd, hd) };
            for (p, &s) in scores.iter().enumerate() {
                axpy(o, s / sum, &v_cache[p * kv_dim + kv..][..hd]);
            }
        }
    });
}

impl Engine for CpuEngine {
    fn config(&self) -> &Config {
        &self.cfg
    }

    fn name(&self) -> String {
        format!("cpu {} x{}", self.w.format.name(), self.pool.threads())
    }

    fn pos(&self) -> usize {
        self.pos
    }

    fn max_seq(&self) -> usize {
        self.max_seq
    }

    fn truncate(&mut self, pos: usize) {
        self.pos = self.pos.min(pos);
    }

    fn forward(&mut self, tokens: &[u32], all: bool) -> &[f32] {
        let CpuEngine {
            cfg: c,
            w,
            pool,
            freqs,
            max_seq,
            pos,
            k_cache,
            v_cache,
            s,
        } = self;
        let t = tokens.len();
        assert!(t > 0, "forward needs at least one token");
        assert!(*pos + t <= *max_seq, "context window full");
        let (dim, kv_dim, q_dim, inter, hd) =
            (c.hidden, c.kv_dim(), c.q_dim(), c.intermediate, c.head_dim);
        let qkv_dim = q_dim + 2 * kv_dim;
        let half = hd / 2;

        zeroed(&mut s.x, t * dim);
        zeroed(&mut s.xb, t * dim);
        zeroed(&mut s.qkv, t * qkv_dim);
        zeroed(&mut s.proj, t * dim);
        zeroed(&mut s.gate_up, t * 2 * inter);
        zeroed(&mut s.hidden, t * inter);
        for (i, &tok) in tokens.iter().enumerate() {
            w.embed
                .row_into(tok as usize, &mut s.x[i * dim..(i + 1) * dim]);
        }

        // Rotation angles for each new position, shared by every layer.
        let mut cos = vec![0f32; t * half];
        let mut sin = vec![0f32; t * half];
        for i in 0..t {
            for j in 0..half {
                let angle = (*pos + i) as f64 * freqs[j];
                cos[i * half + j] = angle.cos() as f32;
                sin[i * half + j] = angle.sin() as f32;
            }
        }

        for (l, layer) in w.layers.iter().enumerate() {
            // attention block: x += o(attention(rope(qkv(norm(x)))))
            for i in 0..t {
                rmsnorm(
                    &mut s.xb[i * dim..(i + 1) * dim],
                    &s.x[i * dim..(i + 1) * dim],
                    &layer.attn_norm,
                    c.norm_eps,
                );
            }
            layer.wqkv.matmul(pool, &s.xb, t, &mut s.qkv);
            let base = l * *max_seq * kv_dim;
            for i in 0..t {
                let row = &mut s.qkv[i * qkv_dim..(i + 1) * qkv_dim];
                if let Some(b) = &layer.bqkv {
                    row.iter_mut().zip(b).for_each(|(v, b)| *v += b);
                }
                let (cs, sn) = (
                    &cos[i * half..(i + 1) * half],
                    &sin[i * half..(i + 1) * half],
                );
                let (q, kv) = row.split_at_mut(q_dim);
                let (k, v) = kv.split_at_mut(kv_dim);
                rope(q, hd, cs, sn);
                rope(k, hd, cs, sn);
                let at = base + (*pos + i) * kv_dim;
                k_cache[at..at + kv_dim].copy_from_slice(k);
                v_cache[at..at + kv_dim].copy_from_slice(v);
            }
            let layer_cache = base..base + *max_seq * kv_dim;
            attention(
                c,
                pool,
                &k_cache[layer_cache.clone()],
                &v_cache[layer_cache],
                *pos,
                t,
                &s.qkv,
                &mut s.att,
            );
            layer.wo.matmul(pool, &s.att, t, &mut s.proj);
            s.x.iter_mut().zip(&s.proj).for_each(|(x, p)| *x += p);

            // MLP block: x += down(silu(gate(norm(x))) * up(norm(x)))
            for i in 0..t {
                rmsnorm(
                    &mut s.xb[i * dim..(i + 1) * dim],
                    &s.x[i * dim..(i + 1) * dim],
                    &layer.mlp_norm,
                    c.norm_eps,
                );
            }
            layer.w_gate_up.matmul(pool, &s.xb, t, &mut s.gate_up);
            for i in 0..t {
                let (g, u) = s.gate_up[i * 2 * inter..(i + 1) * 2 * inter].split_at(inter);
                for (h, (&g, &u)) in s.hidden[i * inter..(i + 1) * inter]
                    .iter_mut()
                    .zip(g.iter().zip(u))
                {
                    *h = silu(g) * u;
                }
            }
            layer.w_down.matmul(pool, &s.hidden, t, &mut s.proj);
            s.x.iter_mut().zip(&s.proj).for_each(|(x, p)| *x += p);
        }
        *pos += t;

        // Only rows we return logits for go through the (large) output layer.
        let first = if all { 0 } else { t - 1 };
        let n = t - first;
        for i in 0..n {
            let src = (first + i) * dim;
            rmsnorm(
                &mut s.xb[i * dim..(i + 1) * dim],
                &s.x[src..src + dim],
                &w.norm,
                c.norm_eps,
            );
        }
        let head = w.lm_head();
        zeroed(&mut s.logits, n * head.rows());
        head.matmul(pool, &s.xb[..n * dim], n, &mut s.logits);
        &s.logits
    }
}
