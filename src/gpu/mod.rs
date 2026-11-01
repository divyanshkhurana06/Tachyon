//! Metal backend for Apple GPUs.
//!
//! A forward pass is encoded into a single command buffer, about nine kernel
//! dispatches per layer (see `kernels.metal`), and run with one
//! commit-and-wait. Apple Silicon's CPU and GPU share memory, so token
//! embeddings and RoPE angles are written straight into GPU buffers and the
//! logits are read back in place: no copies across a bus.

use std::ffi::{c_char, c_void, CStr, CString};
use std::mem::size_of;

use crate::config::Config;
use crate::engine::Engine;
use crate::model::{rope_frequencies, Weights};
use crate::weights::Matrix;
use crate::{Error, Result};

const SOURCE: &str = include_str!("kernels.metal");

/// Must match `SIMDS`, `R` and `ATT_THREADS` in kernels.metal.
const MAT_SIMDS: usize = 2;
const ROWS_PER_GROUP: usize = MAT_SIMDS * 4;
const ATT_THREADS: u32 = 256;

/// Most tokens one forward pass takes (prefill chunks, speculative checks).
pub const MAX_BATCH: usize = 128;

#[derive(Clone, Copy)]
#[repr(u32)]
enum Kernel {
    MatBf16,
    MatQ8,
    MatQ4,
    RmsNorm,
    RopeStore,
    Attention,
    SiluMul,
}

const KERNELS: [&str; 7] = [
    "matmul_bf16",
    "matmul_q8",
    "matmul_q4",
    "rmsnorm",
    "rope_store",
    "attention",
    "silu_mul",
];

#[repr(C)]
struct TmArg {
    buf: *mut c_void,
    offset: u64,
}

extern "C" {
    fn tm_create(
        source: *const c_char,
        kernels: *const *const c_char,
        n: usize,
        err: *mut c_char,
        errlen: usize,
    ) -> *mut c_void;
    fn tm_destroy(ctx: *mut c_void);
    fn tm_device_name(ctx: *mut c_void, out: *mut c_char, len: usize);
    fn tm_buffer(ctx: *mut c_void, len: usize) -> *mut c_void;
    fn tm_buffer_contents(buf: *mut c_void) -> *mut c_void;
    fn tm_buffer_free(buf: *mut c_void);
    fn tm_begin(ctx: *mut c_void);
    #[allow(clippy::too_many_arguments)]
    fn tm_dispatch(
        ctx: *mut c_void,
        pipeline: u32,
        args: *const TmArg,
        nargs: u32,
        params: *const c_void,
        plen: u32,
        gx: u32,
        gy: u32,
        gz: u32,
        tx: u32,
        ty: u32,
        tz: u32,
    );
    fn tm_end(ctx: *mut c_void, err: *mut c_char, errlen: usize) -> i32;
}

struct Context(*mut c_void);

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { tm_destroy(self.0) };
    }
}

/// A GPU buffer in shared (CPU-visible) memory.
struct Buffer {
    ptr: *mut c_void,
    len: usize,
}

impl Buffer {
    fn new(ctx: &Context, len: usize) -> Result<Buffer> {
        let ptr = unsafe { tm_buffer(ctx.0, len) };
        if ptr.is_null() {
            return Err(Error(format!("could not allocate a {len}-byte GPU buffer")));
        }
        Ok(Buffer { ptr, len })
    }

    fn from_slice<T: Copy>(ctx: &Context, data: &[T]) -> Result<Buffer> {
        let bytes = std::mem::size_of_val(data);
        let b = Buffer::new(ctx, bytes)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                b.contents() as *mut u8,
                bytes,
            )
        };
        Ok(b)
    }

    fn contents(&self) -> *mut c_void {
        unsafe { tm_buffer_contents(self.ptr) }
    }

    /// The memory belongs to the GPU buffer, not to `self`, so handing out
    /// a mutable view from `&self` is sound as long as views don't overlap.
    ///
    /// # Safety
    /// No GPU work may be using the buffer, and `n` must fit in it.
    #[allow(clippy::mut_from_ref)]
    unsafe fn floats(&self, n: usize) -> &mut [f32] {
        assert!(n * 4 <= self.len);
        std::slice::from_raw_parts_mut(self.contents() as *mut f32, n)
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe { tm_buffer_free(self.ptr) };
    }
}

struct GpuMatrix {
    kernel: Kernel,
    rows: usize,
    cols: usize,
    /// bf16: [data]; quantized: [scales, quants]
    bufs: Vec<Buffer>,
}

impl GpuMatrix {
    fn upload(ctx: &Context, m: &Matrix) -> Result<GpuMatrix> {
        let (kernel, bufs) = match m {
            Matrix::Bf16 { data, .. } => (Kernel::MatBf16, vec![Buffer::from_slice(ctx, data)?]),
            Matrix::Q8 { scales, qs, .. } => (
                Kernel::MatQ8,
                vec![
                    Buffer::from_slice(ctx, scales)?,
                    Buffer::from_slice(ctx, qs)?,
                ],
            ),
            Matrix::Q4 { scales, qs, .. } => (
                Kernel::MatQ4,
                vec![
                    Buffer::from_slice(ctx, scales)?,
                    Buffer::from_slice(ctx, qs)?,
                ],
            ),
        };
        if m.cols() % 32 != 0 {
            return Err(Error::msg("GPU matrices need a multiple of 32 columns"));
        }
        Ok(GpuMatrix {
            kernel,
            rows: m.rows(),
            cols: m.cols(),
            bufs,
        })
    }
}

struct GpuLayer {
    attn_norm: Buffer,
    wqkv: GpuMatrix,
    bqkv: Option<Buffer>,
    wo: GpuMatrix,
    mlp_norm: Buffer,
    w_gate_up: GpuMatrix,
    w_down: GpuMatrix,
}

#[repr(C)]
struct MatParams {
    rows: u32,
    cols: u32,
    t: u32,
    accumulate: u32,
    has_bias: u32,
}

#[repr(C)]
struct NormParams {
    n: u32,
    eps: f32,
}

#[repr(C)]
struct RopeParams {
    head_dim: u32,
    heads: u32,
    kv_heads: u32,
    qkv_dim: u32,
    q_dim: u32,
    kv_dim: u32,
    pos0: u32,
}

#[repr(C)]
struct AttnParams {
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    qkv_dim: u32,
    q_dim: u32,
    kv_dim: u32,
    pos0: u32,
    max_seq: u32,
    scale: f32,
}

#[repr(C)]
struct SiluParams {
    inter: u32,
    n: u32,
}

/// Activations for up to MAX_BATCH tokens, plus the KV cache.
struct Activations {
    x: Buffer,
    xb: Buffer,
    qkv: Buffer,
    att: Buffer,
    gate_up: Buffer,
    hidden: Buffer,
    scores: Buffer,
    logits: Buffer,
    cos: Buffer,
    sin: Buffer,
    k_cache: Buffer,
    v_cache: Buffer,
}

pub struct GpuEngine {
    cfg: Config,
    device: String,
    format: &'static str,
    // Field order matters for drop: buffers before the context.
    act: Activations,
    layers: Vec<GpuLayer>,
    norm: Buffer,
    lm_head: GpuMatrix,
    /// Kept on the CPU for embedding lookups.
    embed: Matrix,
    freqs: Vec<f64>,
    max_seq: usize,
    pos: usize,
    ctx: Context,
}

// Metal objects may be used from any thread, one thread at a time; the
// engine is only ever driven through `&mut self`.
unsafe impl Send for GpuEngine {}

impl GpuEngine {
    pub fn new(cfg: Config, w: Weights, max_seq: usize) -> Result<GpuEngine> {
        let source = CString::new(SOURCE).unwrap();
        let names: Vec<CString> = KERNELS.iter().map(|k| CString::new(*k).unwrap()).collect();
        let name_ptrs: Vec<*const c_char> = names.iter().map(|n| n.as_ptr()).collect();
        let mut err = [0 as c_char; 1024];
        let raw = unsafe {
            tm_create(
                source.as_ptr(),
                name_ptrs.as_ptr(),
                name_ptrs.len(),
                err.as_mut_ptr(),
                err.len(),
            )
        };
        if raw.is_null() {
            let msg = unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy();
            return Err(Error(format!("Metal setup failed: {msg}")));
        }
        let ctx = Context(raw);
        let mut name = [0 as c_char; 256];
        unsafe { tm_device_name(ctx.0, name.as_mut_ptr(), name.len()) };
        let device = unsafe { CStr::from_ptr(name.as_ptr()) }
            .to_string_lossy()
            .into_owned();

        if ![64, 128].contains(&cfg.head_dim) {
            return Err(Error(format!(
                "the GPU attention kernel needs head_dim 64 or 128, not {}",
                cfg.head_dim
            )));
        }
        let max_seq = max_seq.min(cfg.max_positions);
        let layers = w
            .layers
            .iter()
            .map(|l| {
                Ok(GpuLayer {
                    attn_norm: Buffer::from_slice(&ctx, &l.attn_norm)?,
                    wqkv: GpuMatrix::upload(&ctx, &l.wqkv)?,
                    bqkv: l
                        .bqkv
                        .as_ref()
                        .map(|b| Buffer::from_slice(&ctx, b))
                        .transpose()?,
                    wo: GpuMatrix::upload(&ctx, &l.wo)?,
                    mlp_norm: Buffer::from_slice(&ctx, &l.mlp_norm)?,
                    w_gate_up: GpuMatrix::upload(&ctx, &l.w_gate_up)?,
                    w_down: GpuMatrix::upload(&ctx, &l.w_down)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let lm_head = GpuMatrix::upload(&ctx, w.lm_head())?;
        let norm = Buffer::from_slice(&ctx, &w.norm)?;

        let f = size_of::<f32>();
        let b = MAX_BATCH;
        let qkv_dim = cfg.q_dim() + 2 * cfg.kv_dim();
        let cache = cfg.layers * max_seq * cfg.kv_dim() * f;
        let act = Activations {
            x: Buffer::new(&ctx, b * cfg.hidden * f)?,
            xb: Buffer::new(&ctx, b * cfg.hidden * f)?,
            qkv: Buffer::new(&ctx, b * qkv_dim * f)?,
            att: Buffer::new(&ctx, b * cfg.q_dim() * f)?,
            gate_up: Buffer::new(&ctx, b * 2 * cfg.intermediate * f)?,
            hidden: Buffer::new(&ctx, b * cfg.intermediate * f)?,
            scores: Buffer::new(&ctx, b * cfg.heads * max_seq * f)?,
            logits: Buffer::new(&ctx, b * lm_head.rows * f)?,
            cos: Buffer::new(&ctx, b * cfg.head_dim / 2 * f)?,
            sin: Buffer::new(&ctx, b * cfg.head_dim / 2 * f)?,
            k_cache: Buffer::new(&ctx, cache)?,
            v_cache: Buffer::new(&ctx, cache)?,
        };

        let format = w.format.name();
        let Weights { embed, .. } = w; // everything else now lives on the GPU
        Ok(GpuEngine {
            freqs: rope_frequencies(&cfg),
            cfg,
            device,
            format,
            act,
            layers,
            norm,
            lm_head,
            embed,
            max_seq,
            pos: 0,
            ctx,
        })
    }

    fn dispatch<P>(
        &self,
        k: Kernel,
        args: &[(&Buffer, usize)],
        params: &P,
        groups: [usize; 3],
        threads: [usize; 3],
    ) {
        let args: Vec<TmArg> = args
            .iter()
            .map(|(b, off)| TmArg {
                buf: b.ptr,
                offset: *off as u64,
            })
            .collect();
        unsafe {
            tm_dispatch(
                self.ctx.0,
                k as u32,
                args.as_ptr(),
                args.len() as u32,
                params as *const P as *const c_void,
                size_of::<P>() as u32,
                groups[0] as u32,
                groups[1] as u32,
                groups[2] as u32,
                threads[0] as u32,
                threads[1] as u32,
                threads[2] as u32,
            )
        };
    }

    /// `y[i] (+)= m · x[i]` for `t` rows of `x`.
    #[allow(clippy::too_many_arguments)]
    fn matmul(
        &self,
        m: &GpuMatrix,
        x: (&Buffer, usize),
        y: (&Buffer, usize),
        t: usize,
        bias: Option<&Buffer>,
        accumulate: bool,
    ) {
        let p = MatParams {
            rows: m.rows as u32,
            cols: m.cols as u32,
            t: t as u32,
            accumulate: accumulate as u32,
            has_bias: bias.is_some() as u32,
        };
        let bias = (bias.unwrap_or(y.0), 0);
        let mut args: Vec<(&Buffer, usize)> = m.bufs.iter().map(|b| (b, 0)).collect();
        args.extend([x, y, bias]);
        self.dispatch(
            m.kernel,
            &args,
            &p,
            [m.rows.div_ceil(ROWS_PER_GROUP), 1, 1],
            [32 * MAT_SIMDS, 1, 1],
        );
    }

    fn rmsnorm(&self, x: (&Buffer, usize), w: &Buffer, out: &Buffer, rows: usize) {
        let p = NormParams {
            n: self.cfg.hidden as u32,
            eps: self.cfg.norm_eps,
        };
        self.dispatch(
            Kernel::RmsNorm,
            &[x, (w, 0), (out, 0)],
            &p,
            [rows, 1, 1],
            [256, 1, 1],
        );
    }
}

impl Engine for GpuEngine {
    fn config(&self) -> &Config {
        &self.cfg
    }

    fn name(&self) -> String {
        format!("metal {} ({})", self.format, self.device)
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
        let t = tokens.len();
        assert!(
            t > 0 && t <= MAX_BATCH,
            "the GPU engine takes 1..={MAX_BATCH} tokens per call"
        );
        assert!(self.pos + t <= self.max_seq, "context window full");
        let c = &self.cfg;
        let (dim, kv_dim, q_dim, inter, hd) =
            (c.hidden, c.kv_dim(), c.q_dim(), c.intermediate, c.head_dim);
        let qkv_dim = q_dim + 2 * kv_dim;
        let half = hd / 2;
        let a = &self.act;

        // Inputs, written directly into GPU-visible memory.
        unsafe {
            let x = a.x.floats(t * dim);
            for (i, &tok) in tokens.iter().enumerate() {
                self.embed
                    .row_into(tok as usize, &mut x[i * dim..(i + 1) * dim]);
            }
            let (cos, sin) = (a.cos.floats(t * half), a.sin.floats(t * half));
            for i in 0..t {
                for j in 0..half {
                    let angle = (self.pos + i) as f64 * self.freqs[j];
                    cos[i * half + j] = angle.cos() as f32;
                    sin[i * half + j] = angle.sin() as f32;
                }
            }
        }

        let rope = RopeParams {
            head_dim: hd as u32,
            heads: c.heads as u32,
            kv_heads: c.kv_heads as u32,
            qkv_dim: qkv_dim as u32,
            q_dim: q_dim as u32,
            kv_dim: kv_dim as u32,
            pos0: self.pos as u32,
        };
        let attn = AttnParams {
            heads: c.heads as u32,
            kv_heads: c.kv_heads as u32,
            head_dim: hd as u32,
            qkv_dim: qkv_dim as u32,
            q_dim: q_dim as u32,
            kv_dim: kv_dim as u32,
            pos0: self.pos as u32,
            max_seq: self.max_seq as u32,
            scale: 1.0 / (hd as f32).sqrt(),
        };
        let silu = SiluParams {
            inter: inter as u32,
            n: (t * inter) as u32,
        };

        unsafe { tm_begin(self.ctx.0) };
        for (l, layer) in self.layers.iter().enumerate() {
            let cache_off = l * self.max_seq * kv_dim * 4;
            self.rmsnorm((&a.x, 0), &layer.attn_norm, &a.xb, t);
            self.matmul(
                &layer.wqkv,
                (&a.xb, 0),
                (&a.qkv, 0),
                t,
                layer.bqkv.as_ref(),
                false,
            );
            self.dispatch(
                Kernel::RopeStore,
                &[
                    (&a.qkv, 0),
                    (&a.k_cache, cache_off),
                    (&a.v_cache, cache_off),
                    (&a.cos, 0),
                    (&a.sin, 0),
                ],
                &rope,
                [1, c.heads + c.kv_heads, t],
                [half, 1, 1],
            );
            self.dispatch(
                Kernel::Attention,
                &[
                    (&a.qkv, 0),
                    (&a.k_cache, cache_off),
                    (&a.v_cache, cache_off),
                    (&a.scores, 0),
                    (&a.att, 0),
                ],
                &attn,
                [c.heads, t, 1],
                [ATT_THREADS as usize, 1, 1],
            );
            self.matmul(&layer.wo, (&a.att, 0), (&a.x, 0), t, None, true);
            self.rmsnorm((&a.x, 0), &layer.mlp_norm, &a.xb, t);
            self.matmul(
                &layer.w_gate_up,
                (&a.xb, 0),
                (&a.gate_up, 0),
                t,
                None,
                false,
            );
            self.dispatch(
                Kernel::SiluMul,
                &[(&a.gate_up, 0), (&a.hidden, 0)],
                &silu,
                [(t * inter).div_ceil(256), 1, 1],
                [256, 1, 1],
            );
            self.matmul(&layer.w_down, (&a.hidden, 0), (&a.x, 0), t, None, true);
        }
        let first = if all { 0 } else { t - 1 };
        let n = t - first;
        self.rmsnorm((&a.x, first * dim * 4), &self.norm, &a.xb, n);
        self.matmul(&self.lm_head, (&a.xb, 0), (&a.logits, 0), n, None, false);

        let mut err = [0 as c_char; 1024];
        if unsafe { tm_end(self.ctx.0, err.as_mut_ptr(), err.len()) } != 0 {
            let msg = unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy();
            panic!("GPU command buffer failed: {msg}");
        }
        self.pos += t;
        unsafe { self.act.logits.floats(n * self.lm_head.rows) }
    }
}
