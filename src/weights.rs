//! Weight matrices in each storage format, and the parallel matrix multiply
//! over them.

use crate::kernels::{self, dot_bf16, dot_q4, dot_q4_x4, dot_q8, dot_q8_x4, f32_to_bf16, QK};
use crate::pool::{Pool, SyncPtr};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// 16 bits per weight, the checkpoint's own precision.
    Bf16,
    /// 8.5 bits per weight (int8 plus one f32 scale per 32).
    Q8,
    /// 4.5 bits per weight for most matrices. The small, sensitive q/k/v
    /// projection and the output layer stay at 8 bits.
    Q4,
    /// "Medium" 4-bit: like `Q4`, plus the MLP down projection at 8 bits.
    Q4M,
}

impl Format {
    pub fn parse(s: &str) -> Option<Format> {
        match s.to_ascii_lowercase().as_str() {
            "bf16" => Some(Format::Bf16),
            "q8" | "int8" => Some(Format::Q8),
            "q4" | "int4" => Some(Format::Q4),
            "q4m" => Some(Format::Q4M),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Format::Bf16 => "bf16",
            Format::Q8 => "q8",
            Format::Q4 => "q4",
            Format::Q4M => "q4m",
        }
    }

    /// Storage for one kind of matrix (`qkv`, `o`, `gate_up`, `down`,
    /// `output`) under this format. Measured on Qwen2.5-0.5B, keeping q/k/v
    /// at 8 bits cuts 4-bit perplexity by 0.6 for 3% more size; the down
    /// projection by 0.5 for 14% more.
    pub fn for_matrix(self, which: &str) -> Format {
        match (self, which) {
            (Format::Q4 | Format::Q4M, "qkv" | "output") | (Format::Q4M, "down") => Format::Q8,
            (Format::Q4M, _) => Format::Q4,
            (f, _) => f,
        }
    }
}

/// A `rows x cols` matrix stored row-major, as in PyTorch's `Linear`
/// (`out_features x in_features`), so each output is one contiguous row.
pub enum Matrix {
    Bf16 {
        rows: usize,
        cols: usize,
        data: Vec<u16>,
    },
    Q8 {
        rows: usize,
        cols: usize,
        scales: Vec<f32>,
        qs: Vec<i8>,
    },
    Q4 {
        rows: usize,
        cols: usize,
        scales: Vec<f32>,
        qs: Vec<u8>,
    },
}

impl Matrix {
    pub fn from_f32(data: &[f32], rows: usize, cols: usize, format: Format) -> Matrix {
        assert_eq!(data.len(), rows * cols);
        // Quantized blocks must not straddle rows; fall back if they would.
        let format = if cols % QK != 0 { Format::Bf16 } else { format };
        let blocks = rows * cols / QK;
        match format {
            Format::Bf16 => Matrix::Bf16 {
                rows,
                cols,
                data: data.iter().map(|&v| f32_to_bf16(v)).collect(),
            },
            Format::Q8 => {
                let (mut scales, mut qs) = (vec![0.0; blocks], vec![0i8; rows * cols]);
                kernels::quantize_q8(data, &mut scales, &mut qs);
                Matrix::Q8 {
                    rows,
                    cols,
                    scales,
                    qs,
                }
            }
            Format::Q4 | Format::Q4M => {
                let (mut scales, mut qs) = (vec![0.0; blocks], vec![0u8; rows * cols / 2]);
                kernels::quantize_q4(data, &mut scales, &mut qs);
                Matrix::Q4 {
                    rows,
                    cols,
                    scales,
                    qs,
                }
            }
        }
    }

    /// Like [`Matrix::from_f32`], but 4-bit rows are fitted using
    /// `importance` (one value per column; see `imatrix.rs`).
    pub fn from_f32_weighted(
        data: &[f32],
        rows: usize,
        cols: usize,
        format: Format,
        importance: Option<&[f32]>,
    ) -> Matrix {
        let (Some(imp), Format::Q4 | Format::Q4M) = (importance, format) else {
            return Matrix::from_f32(data, rows, cols, format);
        };
        if cols % QK != 0 || imp.len() != cols {
            return Matrix::from_f32(data, rows, cols, format);
        }
        let nb = cols / QK;
        let (mut scales, mut qs) = (vec![0.0; rows * nb], vec![0u8; rows * cols / 2]);
        for r in 0..rows {
            kernels::quantize_q4_weighted(
                &data[r * cols..(r + 1) * cols],
                imp,
                &mut scales[r * nb..(r + 1) * nb],
                &mut qs[r * cols / 2..(r + 1) * cols / 2],
            );
        }
        Matrix::Q4 {
            rows,
            cols,
            scales,
            qs,
        }
    }

    pub fn rows(&self) -> usize {
        match *self {
            Matrix::Bf16 { rows, .. } | Matrix::Q8 { rows, .. } | Matrix::Q4 { rows, .. } => rows,
        }
    }

    pub fn cols(&self) -> usize {
        match *self {
            Matrix::Bf16 { cols, .. } | Matrix::Q8 { cols, .. } | Matrix::Q4 { cols, .. } => cols,
        }
    }

    pub fn format(&self) -> Format {
        match self {
            Matrix::Bf16 { .. } => Format::Bf16,
            Matrix::Q8 { .. } => Format::Q8,
            Matrix::Q4 { .. } => Format::Q4,
        }
    }

    pub fn bytes(&self) -> usize {
        match self {
            Matrix::Bf16 { data, .. } => data.len() * 2,
            Matrix::Q8 { scales, qs, .. } => scales.len() * 4 + qs.len(),
            Matrix::Q4 { scales, qs, .. } => scales.len() * 4 + qs.len(),
        }
    }

    /// Row `r` as f32 (used for embedding lookups).
    pub fn row_into(&self, r: usize, out: &mut [f32]) {
        let cols = self.cols();
        let nb = cols / QK;
        match self {
            Matrix::Bf16 { data, .. } => {
                for (o, &w) in out.iter_mut().zip(&data[r * cols..(r + 1) * cols]) {
                    *o = kernels::bf16_to_f32(w);
                }
            }
            Matrix::Q8 { scales, qs, .. } => kernels::dequantize_q8(
                &scales[r * nb..(r + 1) * nb],
                &qs[r * cols..(r + 1) * cols],
                out,
            ),
            Matrix::Q4 { scales, qs, .. } => kernels::dequantize_q4(
                &scales[r * nb..(r + 1) * nb],
                &qs[r * cols / 2..(r + 1) * cols / 2],
                out,
            ),
        }
    }

    /// `out[i] = self · x[i]` for each of the `t` input rows: `x` is
    /// `t x cols`, `out` is `t x rows`.
    ///
    /// The loop over inputs sits inside the loop over weight rows, so with
    /// several tokens (prefill, speculative verification) each weight row is
    /// read from memory once and reused from cache.
    pub fn matmul(&self, pool: &Pool, x: &[f32], t: usize, out: &mut [f32]) {
        let (rows, cols) = (self.rows(), self.cols());
        assert_eq!(x.len(), t * cols);
        assert_eq!(out.len(), t * rows);
        let grain = (rows / (pool.threads() * 8)).clamp(4, 256);
        let out = SyncPtr::new(out);
        match self {
            Matrix::Bf16 { data, .. } => pool.parallel_for(rows, grain, |range| {
                for r in range {
                    let w = &data[r * cols..(r + 1) * cols];
                    for i in 0..t {
                        // SAFETY: each (i, r) is written by exactly one thread.
                        unsafe {
                            out.write(i * rows + r, dot_bf16(w, &x[i * cols..(i + 1) * cols]))
                        };
                    }
                }
            }),
            Matrix::Q8 { scales, qs, .. } => {
                let (xs, xq) = quantize_input(x, t, cols);
                let nb = cols / QK;
                let tok = |i: usize| (&xs[i * nb..(i + 1) * nb], &xq[i * cols..(i + 1) * cols]);
                pool.parallel_for(rows, grain, |range| {
                    for r in range {
                        let (ws, wq) =
                            (&scales[r * nb..(r + 1) * nb], &qs[r * cols..(r + 1) * cols]);
                        // Four tokens at a time share each weight block load.
                        let mut i = 0;
                        while i + 4 <= t {
                            let ys = dot_q8_x4(
                                ws,
                                wq,
                                [0, 1, 2, 3].map(|j| tok(i + j).0),
                                [0, 1, 2, 3].map(|j| tok(i + j).1),
                            );
                            for (j, y) in ys.into_iter().enumerate() {
                                unsafe { out.write((i + j) * rows + r, y) };
                            }
                            i += 4;
                        }
                        for i in i..t {
                            let (s, q) = tok(i);
                            unsafe { out.write(i * rows + r, dot_q8(ws, wq, s, q)) };
                        }
                    }
                })
            }
            Matrix::Q4 { scales, qs, .. } => {
                let (xs, xq) = quantize_input(x, t, cols);
                let nb = cols / QK;
                let tok = |i: usize| (&xs[i * nb..(i + 1) * nb], &xq[i * cols..(i + 1) * cols]);
                pool.parallel_for(rows, grain, |range| {
                    for r in range {
                        let (ws, wq) = (
                            &scales[r * nb..(r + 1) * nb],
                            &qs[r * cols / 2..(r + 1) * cols / 2],
                        );
                        let mut i = 0;
                        while i + 4 <= t {
                            let ys = dot_q4_x4(
                                ws,
                                wq,
                                [0, 1, 2, 3].map(|j| tok(i + j).0),
                                [0, 1, 2, 3].map(|j| tok(i + j).1),
                            );
                            for (j, y) in ys.into_iter().enumerate() {
                                unsafe { out.write((i + j) * rows + r, y) };
                            }
                            i += 4;
                        }
                        for i in i..t {
                            let (s, q) = tok(i);
                            unsafe { out.write(i * rows + r, dot_q4(ws, wq, s, q)) };
                        }
                    }
                })
            }
        }
    }
}

fn quantize_input(x: &[f32], t: usize, cols: usize) -> (Vec<f32>, Vec<i8>) {
    let (mut xs, mut xq) = (vec![0.0; t * cols / QK], vec![0i8; t * cols]);
    kernels::quantize_q8(x, &mut xs, &mut xq);
    (xs, xq)
}
