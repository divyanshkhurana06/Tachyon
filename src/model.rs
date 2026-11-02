//! Transformer weights and the CPU forward pass, for the Qwen2 / Llama
//! family: pre-norm decoder blocks with RMSNorm, grouped-query attention with
//! rotary position embeddings, and a SiLU-gated MLP, or a mixture of SiLU
//! experts (Granite MoE).

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::thread;

use crate::config::Config;
use crate::engine::Backend;
use crate::imatrix::{Importance, LayerImportance};
use crate::kernels::{axpy, dot_f32, rmsnorm, silu};
use crate::kv::{slot, Batch, BLOCK};
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
    /// Per-head RMSNorm weights for q and k, applied before RoPE (Qwen3).
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    pub wo: Matrix,
    pub mlp_norm: Vec<f32>,
    /// gate and up projections, stacked the same way. In a mixture of
    /// experts, every expert's matrices stacked in turn: expert `e` owns
    /// rows `e * 2 * intermediate..` of `w_gate_up` and `e * hidden..` of
    /// `w_down`.
    pub w_gate_up: Matrix,
    pub w_down: Matrix,
    /// Mixture of experts: the router, a score per expert for each token.
    pub router: Option<Matrix>,
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
        Weights::load_with(dir, c, format, None)
    }

    /// [`Weights::load`], fitting 4-bit matrices to an importance matrix
    /// (mean squared inputs per channel, from calibration).
    pub fn load_with(
        dir: &Path,
        c: &Config,
        format: Format,
        importance: Option<&Importance>,
    ) -> Result<Weights> {
        let st = SafeTensors::open_dir(dir)?;
        let next = AtomicUsize::new(0);
        let done = Mutex::new(Vec::new());
        let threads = Pool::default_threads().min(c.layers + 1);
        // The output layer is the most sensitive matrix to quantize, so
        // 4-bit models keep it (and the embedding it may share) at 8 bits,
        // as llama.cpp does.
        let outer = format.for_matrix("output");

        // The embedding (the single biggest matrix) loads alongside the layers.
        let embed = thread::scope(|s| {
            let embed = s.spawn(|| matrix(&st, &["model.embed_tokens.weight"], (outer, None), 1.0));
            for _ in 0..threads {
                s.spawn(|| loop {
                    let l = next.fetch_add(1, Relaxed);
                    if l >= c.layers {
                        break;
                    }
                    let layer = load_layer(&st, c, l, format, importance.map(|i| &i.layers[l]));
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
            Some(matrix(&st, &["lm_head.weight"], (outer, None), 1.0)?)
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
        // Granite divides logits by `logits_scaling`; scaling the final norm's
        // weights does it before the (possibly shared) output matrix.
        let mut norm = vector(&st, "model.norm.weight")?;
        norm.iter_mut().for_each(|g| *g /= c.logits_scaling);
        Ok(Weights {
            format,
            embed,
            lm_head,
            layers,
            norm,
        })
    }

    pub fn bytes(&self) -> usize {
        let layer = |l: &LayerWeights| {
            l.wqkv.bytes()
                + l.wo.bytes()
                + l.w_gate_up.bytes()
                + l.w_down.bytes()
                + l.router.as_ref().map_or(0, Matrix::bytes)
        };
        self.embed.bytes()
            + self.lm_head.as_ref().map_or(0, Matrix::bytes)
            + self.layers.iter().map(layer).sum::<usize>()
    }

    pub fn lm_head(&self) -> &Matrix {
        self.lm_head.as_ref().unwrap_or(&self.embed)
    }
}

fn load_layer(
    st: &SafeTensors,
    c: &Config,
    l: usize,
    format: Format,
    imp: Option<&LayerImportance>,
) -> Result<LayerWeights> {
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
    let fmt = |name: &str| (format.for_matrix(name), imp.and_then(|i| i.get(name)));
    // Granite scales each block's output before adding it back; folding the
    // factor into the matrices that produce those outputs costs nothing.
    let residual = c.residual_multiplier;
    let moe = |s: &str| n(&format!("block_sparse_moe.{s}.weight"));
    let (w_gate_up, w_down, router) = if c.experts > 0 {
        (
            matrix(st, &[&moe("input_linear")], fmt("gate_up"), 1.0)?,
            matrix(st, &[&moe("output_linear")], fmt("down"), residual)?,
            // The router is tiny and picks the experts: kept at bf16.
            Some(matrix(
                st,
                &[&moe("router.layer")],
                (Format::Bf16, None),
                1.0,
            )?),
        )
    } else {
        (
            matrix(
                st,
                &[&n("mlp.gate_proj.weight"), &n("mlp.up_proj.weight")],
                fmt("gate_up"),
                1.0,
            )?,
            matrix(st, &[&n("mlp.down_proj.weight")], fmt("down"), residual)?,
            None,
        )
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
            fmt("qkv"),
            1.0,
        )?,
        bqkv,
        q_norm: optional(st, &n("self_attn.q_norm.weight")),
        k_norm: optional(st, &n("self_attn.k_norm.weight")),
        wo: matrix(st, &[&n("self_attn.o_proj.weight")], fmt("o"), residual)?,
        mlp_norm: vector(st, &n("post_attention_layernorm.weight"))?,
        w_gate_up,
        w_down,
        router,
    };
    let qkv = c.q_dim() + 2 * c.kv_dim();
    let e = c.experts.max(1);
    let mut shapes = vec![
        (&layer.wqkv, qkv, c.hidden),
        (&layer.wo, c.hidden, c.q_dim()),
        (&layer.w_gate_up, e * 2 * c.intermediate, c.hidden),
        (&layer.w_down, e * c.hidden, c.intermediate),
    ];
    if let Some(r) = &layer.router {
        shapes.push((r, c.experts, c.hidden));
    }
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

/// Stacks the named tensors row-wise into one matrix, times `scale`. A 3-D
/// tensor (a stack of experts' matrices) counts as its matrices stacked.
fn matrix(
    st: &SafeTensors,
    names: &[&str],
    (format, importance): (Format, Option<&[f32]>),
    scale: f32,
) -> Result<Matrix> {
    let mut data = Vec::new();
    let (mut rows, mut cols) = (0, None);
    for name in names {
        let t = st.tensor(name)?;
        let (r, c) = match *t.shape {
            [r, c] => (r, c),
            [e, r, c] => (e * r, c),
            _ => bail!("{name} is not a matrix"),
        };
        if cols.is_some_and(|cols| cols != c) {
            bail!("{name} has {c} columns, unlike the tensors stacked with it");
        }
        cols = Some(c);
        rows += r;
        data.extend(t.to_f32());
    }
    if scale != 1.0 {
        data.iter_mut().for_each(|v| *v *= scale);
    }
    Ok(Matrix::from_f32_weighted(
        &data,
        rows,
        cols.unwrap_or(0),
        format,
        importance,
    ))
}

fn optional(st: &SafeTensors, name: &str) -> Option<Vec<f32>> {
    st.tensor(name).ok().map(|t| t.to_f32())
}

fn vector(st: &SafeTensors, name: &str) -> Result<Vec<f32>> {
    Ok(st.tensor(name)?.to_f32())
}

/// Rotary embedding frequencies: `theta^(-2i/d)` for each pair of dims.
pub fn rope_frequencies(c: &Config) -> Vec<f64> {
    let half = c.head_dim / 2;
    (0..half)
        .map(|i| {
            let f = c.rope_theta.powf(-2.0 * i as f64 / c.head_dim as f64);
            c.rope_scaling.as_ref().map_or(f, |s| s.apply(f))
        })
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
    /// Mixture of experts: router scores, and one expert's tokens' inputs
    /// and outputs.
    route: Vec<f32>,
    expert_in: Vec<f32>,
    expert_out: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

fn zeroed(v: &mut Vec<f32>, n: usize) {
    v.clear();
    v.resize(n, 0.0);
}

/// RMSNorm each `head_dim`-sized head of `v` in place.
fn head_norm(v: &mut [f32], head_dim: usize, w: &[f32], eps: f32) {
    for head in v.chunks_exact_mut(head_dim) {
        let ss = dot_f32(head, head) / head_dim as f32;
        let inv = 1.0 / (ss + eps).sqrt();
        head.iter_mut().zip(w).for_each(|(x, g)| *x *= inv * g);
    }
}

pub struct CpuBackend {
    cfg: Config,
    w: Weights,
    pool: Arc<Pool>,
    freqs: Vec<f64>,
    blocks: usize,
    /// `[layer][slot][kv_dim]`, slot = block * BLOCK + offset
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
    s: Scratch,
    /// When set, every forward pass adds its linear layers' squared inputs
    /// here (see `imatrix.rs`).
    pub calibration: Option<Importance>,
}

impl CpuBackend {
    /// `blocks` sets the KV cache size: `blocks * BLOCK` token positions,
    /// shared by every sequence.
    pub fn new(cfg: Config, w: Weights, pool: Arc<Pool>, blocks: usize) -> CpuBackend {
        let cache = cfg.layers * blocks * BLOCK * cfg.kv_dim();
        CpuBackend {
            freqs: rope_frequencies(&cfg),
            k_cache: vec![0.0; cache],
            v_cache: vec![0.0; cache],
            cfg,
            w,
            pool,
            blocks,
            s: Scratch::default(),
            calibration: None,
        }
    }
}

/// Causal attention for every row of one layer: each token attends to its
/// own sequence's positions up to and including its own, found through the
/// sequence's block table. One task per (token, head); heads in a group
/// share a K/V head.
fn attention(
    c: &Config,
    pool: &Pool,
    k_cache: &[f32],
    v_cache: &[f32],
    b: &Batch,
    qkv: &[f32],
    att: &mut Vec<f32>,
) {
    let (hd, heads, kv_dim, q_dim) = (c.head_dim, c.heads, c.kv_dim(), c.q_dim());
    let group = heads / c.kv_heads;
    let qkv_dim = q_dim + 2 * kv_dim;
    let scale = c.attention_scale();
    let t = b.len();
    zeroed(att, t * q_dim);
    let out = SyncPtr::new(att);

    pool.parallel_for(t * heads, 1, |tasks| {
        let mut scores = Vec::new();
        for task in tasks {
            let (i, h) = (task / heads, task % heads);
            let q = &qkv[i * qkv_dim + h * hd..][..hd];
            let kv = (h / group) * hd;
            let table = &b.tables[b.table_of[i] as usize..];
            let n = b.positions[i] as usize + 1;

            scores.clear();
            scores.extend(
                (0..n).map(|p| dot_f32(q, &k_cache[slot(table, p) * kv_dim + kv..][..hd]) * scale),
            );
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for s in scores.iter_mut() {
                *s = (*s - max).exp();
                sum += *s;
            }
            // SAFETY: each (token, head) owns its own slice of `att`.
            let o = unsafe { out.slice(i * q_dim + h * hd, hd) };
            for (p, &s) in scores.iter().enumerate() {
                axpy(o, s / sum, &v_cache[slot(table, p) * kv_dim + kv..][..hd]);
            }
        }
    });
}

/// Picks each token's experts: the `k` highest router scores, weighted by a
/// softmax over those `k` (Granite's gating). Returns (expert, weight) pairs.
pub fn route(scores: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut top: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    top.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    top.truncate(k);
    let max = top.first().map_or(0.0, |t| t.1);
    let sum: f32 = top.iter().map(|t| (t.1 - max).exp()).sum();
    top.iter()
        .map(|&(e, v)| (e, (v - max).exp() / sum))
        .collect()
}

/// The mixture-of-experts MLP over the normed inputs in `s.xb`, into
/// `s.proj`: each expert runs once over all the tokens routed to it, and
/// its outputs are added to theirs, weighted by the router.
fn moe(
    c: &Config,
    pool: &Pool,
    layer: &LayerWeights,
    router: &Matrix,
    s: &mut Scratch,
    t: usize,
    mut cal: Option<&mut LayerImportance>,
) {
    let (dim, inter, experts) = (c.hidden, c.intermediate, c.experts);
    zeroed(&mut s.route, t * experts);
    router.matmul(pool, &s.xb, t, &mut s.route);
    let mut by_expert: Vec<Vec<(usize, f32)>> = vec![Vec::new(); experts];
    for i in 0..t {
        for (e, g) in route(
            &s.route[i * experts..(i + 1) * experts],
            c.experts_per_token,
        ) {
            by_expert[e].push((i, g));
        }
    }
    zeroed(&mut s.proj, t * dim);
    for (e, tokens) in by_expert.iter().enumerate() {
        let m = tokens.len();
        if m == 0 {
            continue;
        }
        s.expert_in.clear();
        for &(i, _) in tokens {
            s.expert_in.extend_from_slice(&s.xb[i * dim..(i + 1) * dim]);
        }
        zeroed(&mut s.gate_up, m * 2 * inter);
        let rows = e * 2 * inter..(e + 1) * 2 * inter;
        layer
            .w_gate_up
            .matmul_rows(pool, &s.expert_in, m, &mut s.gate_up, rows);
        zeroed(&mut s.hidden, m * inter);
        for j in 0..m {
            let (g, u) = s.gate_up[j * 2 * inter..(j + 1) * 2 * inter].split_at(inter);
            for (h, (&g, &u)) in s.hidden[j * inter..(j + 1) * inter]
                .iter_mut()
                .zip(g.iter().zip(u))
            {
                *h = silu(g) * u;
            }
        }
        if let Some(cal) = cal.as_mut() {
            Importance::accumulate(&mut cal.down, &s.hidden);
        }
        zeroed(&mut s.expert_out, m * dim);
        let rows = e * dim..(e + 1) * dim;
        layer
            .w_down
            .matmul_rows(pool, &s.hidden, m, &mut s.expert_out, rows);
        for (j, &(i, g)) in tokens.iter().enumerate() {
            axpy(
                &mut s.proj[i * dim..(i + 1) * dim],
                g,
                &s.expert_out[j * dim..(j + 1) * dim],
            );
        }
    }
}

impl Backend for CpuBackend {
    fn config(&self) -> &Config {
        &self.cfg
    }

    fn name(&self) -> String {
        format!("cpu {} x{}", self.w.format.name(), self.pool.threads())
    }

    fn num_blocks(&self) -> usize {
        self.blocks
    }

    fn max_batch(&self) -> usize {
        256
    }

    fn forward(&mut self, b: &Batch) -> &[f32] {
        let CpuBackend {
            cfg: c,
            w,
            pool,
            freqs,
            k_cache,
            v_cache,
            s,
            calibration,
            ..
        } = self;
        let t = b.len();
        assert!(t > 0, "empty batch");
        let (dim, kv_dim, q_dim, inter, hd) =
            (c.hidden, c.kv_dim(), c.q_dim(), c.intermediate, c.head_dim);
        let qkv_dim = q_dim + 2 * kv_dim;
        let half = hd / 2;
        let layer_cache = self.blocks * BLOCK * kv_dim;
        if let Some(cal) = calibration.as_mut() {
            cal.tokens += t as u64;
        }

        zeroed(&mut s.x, t * dim);
        zeroed(&mut s.xb, t * dim);
        zeroed(&mut s.qkv, t * qkv_dim);
        zeroed(&mut s.proj, t * dim);
        zeroed(&mut s.gate_up, t * 2 * inter);
        zeroed(&mut s.hidden, t * inter);
        for (i, &tok) in b.tokens.iter().enumerate() {
            let row = &mut s.x[i * dim..(i + 1) * dim];
            w.embed.row_into(tok as usize, row);
            if c.embedding_multiplier != 1.0 {
                row.iter_mut().for_each(|v| *v *= c.embedding_multiplier);
            }
        }

        // Rotation angles for each token's position, shared by every layer.
        zeroed(&mut s.cos, t * half);
        zeroed(&mut s.sin, t * half);
        for (i, &pos) in b.positions.iter().enumerate() {
            for (j, f) in freqs.iter().enumerate() {
                let angle = pos as f64 * f;
                s.cos[i * half + j] = angle.cos() as f32;
                s.sin[i * half + j] = angle.sin() as f32;
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
            if let Some(cal) = calibration.as_mut() {
                Importance::accumulate(&mut cal.layers[l].qkv, &s.xb);
            }
            layer.wqkv.matmul(pool, &s.xb, t, &mut s.qkv);
            let (kc, vc) = (
                &mut k_cache[l * layer_cache..(l + 1) * layer_cache],
                &mut v_cache[l * layer_cache..(l + 1) * layer_cache],
            );
            for i in 0..t {
                let row = &mut s.qkv[i * qkv_dim..(i + 1) * qkv_dim];
                if let Some(bias) = &layer.bqkv {
                    row.iter_mut().zip(bias).for_each(|(v, b)| *v += b);
                }
                let (cs, sn) = (
                    &s.cos[i * half..(i + 1) * half],
                    &s.sin[i * half..(i + 1) * half],
                );
                let (q, kv) = row.split_at_mut(q_dim);
                let (k, v) = kv.split_at_mut(kv_dim);
                if let (Some(qn), Some(kn)) = (&layer.q_norm, &layer.k_norm) {
                    head_norm(q, hd, qn, c.norm_eps);
                    head_norm(k, hd, kn, c.norm_eps);
                }
                rope(q, hd, cs, sn);
                rope(k, hd, cs, sn);
                let at = b.slots[i] as usize * kv_dim;
                kc[at..at + kv_dim].copy_from_slice(k);
                vc[at..at + kv_dim].copy_from_slice(v);
            }
            attention(c, pool, kc, vc, b, &s.qkv, &mut s.att);
            if let Some(cal) = calibration.as_mut() {
                Importance::accumulate(&mut cal.layers[l].o, &s.att);
            }
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
            if let Some(cal) = calibration.as_mut() {
                Importance::accumulate(&mut cal.layers[l].gate_up, &s.xb);
            }
            if let Some(router) = &layer.router {
                moe(
                    c,
                    pool,
                    layer,
                    router,
                    s,
                    t,
                    calibration.as_mut().map(|cal| &mut cal.layers[l]),
                );
                s.x.iter_mut().zip(&s.proj).for_each(|(x, p)| *x += p);
                continue;
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
            if let Some(cal) = calibration.as_mut() {
                Importance::accumulate(&mut cal.layers[l].down, &s.hidden);
            }
            layer.w_down.matmul(pool, &s.hidden, t, &mut s.proj);
            s.x.iter_mut().zip(&s.proj).for_each(|(x, p)| *x += p);
        }

        // Only rows we return logits for go through the (large) output layer.
        let n = b.logit_rows.len();
        for (i, &row) in b.logit_rows.iter().enumerate() {
            let src = row as usize * dim;
            rmsnorm(
                &mut s.xb[i * dim..(i + 1) * dim],
                &s.x[src..src + dim],
                &w.norm,
                c.norm_eps,
            );
        }
        let head = w.lm_head();
        zeroed(&mut s.logits, n * head.rows());
        if n > 0 {
            head.matmul(pool, &s.xb[..n * dim], n, &mut s.logits);
        }
        &s.logits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_to_the_top_k_with_a_softmax_over_them() {
        let r = route(&[0.5, 3.0, -1.0, 3.0, 2.0], 3);
        let experts: Vec<usize> = r.iter().map(|&(e, _)| e).collect();
        assert_eq!(experts, [1, 3, 4]); // ties go to the lower expert
        let (a, c) = (r[0].1, r[2].1);
        assert!((r.iter().map(|x| x.1).sum::<f32>() - 1.0).abs() < 1e-6);
        assert!((a / c - 1f32.exp()).abs() < 1e-5);
    }
}
