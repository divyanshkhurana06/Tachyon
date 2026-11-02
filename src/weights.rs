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
        self.matmul_rows(pool, x, t, out, 0..self.rows());
    }

    /// [`Matrix::matmul`] with only the rows in `range`: `out` is
    /// `t x range.len()`. A mixture-of-experts layer stores its experts'
    /// matrices stacked, and runs one expert's rows at a time.
    pub fn matmul_rows(
        &self,
        pool: &Pool,
        x: &[f32],
        t: usize,
        out: &mut [f32],
        range: std::ops::Range<usize>,
    ) {
        let (rows, cols, row0) = (range.len(), self.cols(), range.start);
        assert!(range.end <= self.rows());
        assert_eq!(x.len(), t * cols);
        assert_eq!(out.len(), t * rows);
        // Even, so the I8MM path's row pairs don't straddle chunks.
        let grain = (rows / (pool.threads() * 8)).clamp(4, 256) & !1;
        let out = SyncPtr::new(out);
        match self {
            Matrix::Bf16 { data, .. } => pool.parallel_for(rows, grain, |range| {
                for r in range {
                    let w = &data[(row0 + r) * cols..(row0 + r + 1) * cols];
                    for i in 0..t {
                        // SAFETY: each (i, r) is written by exactly one thread.
                        unsafe {
                            out.write(i * rows + r, dot_bf16(w, &x[i * cols..(i + 1) * cols]))
                        };
                    }
                }
            }),
            Matrix::Q8 { scales, qs, .. } => int_matmul(
                pool,
                &Q8Rows {
                    scales,
                    qs,
                    cols,
                    row0,
                },
                rows,
                cols,
                x,
                t,
                grain,
                out,
            ),
            Matrix::Q4 { scales, qs, .. } => int_matmul(
                pool,
                &Q4Rows {
                    scales,
                    qs,
                    cols,
                    row0,
                },
                rows,
                cols,
                x,
                t,
                grain,
                out,
            ),
        }
    }
}

/// The rows of an 8- or 4-bit matrix, for the shared integer matmul.
trait IntRows: Sync {
    fn dot(&self, r: usize, xs: &[f32], xq: &[i8]) -> f32;
    fn dot_x4(&self, r: usize, xs: [&[f32]; 4], xq: [&[i8]; 4]) -> [f32; 4];
    /// Rows `r` and `r + 1` against token pairs (see `kernels::pair_tokens`).
    fn dot2_pairs<const P: usize>(
        &self,
        r: usize,
        ps: [&[f32]; P],
        pq: [&[i8]; P],
    ) -> [[f32; 4]; P];
}

struct Q8Rows<'a> {
    scales: &'a [f32],
    qs: &'a [i8],
    cols: usize,
    /// Row `r` is the matrix's row `row0 + r`.
    row0: usize,
}

impl Q8Rows<'_> {
    fn row(&self, r: usize) -> (&[f32], &[i8]) {
        let (nb, cols, r) = (self.cols / QK, self.cols, self.row0 + r);
        (
            &self.scales[r * nb..(r + 1) * nb],
            &self.qs[r * cols..(r + 1) * cols],
        )
    }
}

impl IntRows for Q8Rows<'_> {
    fn dot(&self, r: usize, xs: &[f32], xq: &[i8]) -> f32 {
        let (ws, wq) = self.row(r);
        dot_q8(ws, wq, xs, xq)
    }
    fn dot_x4(&self, r: usize, xs: [&[f32]; 4], xq: [&[i8]; 4]) -> [f32; 4] {
        let (ws, wq) = self.row(r);
        dot_q8_x4(ws, wq, xs, xq)
    }
    fn dot2_pairs<const P: usize>(
        &self,
        r: usize,
        ps: [&[f32]; P],
        pq: [&[i8]; P],
    ) -> [[f32; 4]; P] {
        let (a, b) = (self.row(r), self.row(r + 1));
        kernels::dot2_q8_pairs([a.0, b.0], [a.1, b.1], ps, pq)
    }
}

struct Q4Rows<'a> {
    scales: &'a [f32],
    qs: &'a [u8],
    cols: usize,
    row0: usize,
}

impl Q4Rows<'_> {
    fn row(&self, r: usize) -> (&[f32], &[u8]) {
        let (nb, half, r) = (self.cols / QK, self.cols / 2, self.row0 + r);
        (
            &self.scales[r * nb..(r + 1) * nb],
            &self.qs[r * half..(r + 1) * half],
        )
    }
}

impl IntRows for Q4Rows<'_> {
    fn dot(&self, r: usize, xs: &[f32], xq: &[i8]) -> f32 {
        let (ws, wq) = self.row(r);
        dot_q4(ws, wq, xs, xq)
    }
    fn dot_x4(&self, r: usize, xs: [&[f32]; 4], xq: [&[i8]; 4]) -> [f32; 4] {
        let (ws, wq) = self.row(r);
        dot_q4_x4(ws, wq, xs, xq)
    }
    fn dot2_pairs<const P: usize>(
        &self,
        r: usize,
        ps: [&[f32]; P],
        pq: [&[i8]; P],
    ) -> [[f32; 4]; P] {
        let (a, b) = (self.row(r), self.row(r + 1));
        kernels::dot2_q4_pairs([a.0, b.0], [a.1, b.1], ps, pq)
    }
}

/// `out = m · x` for quantized `m`, with `x` quantized to 8 bits first.
/// With I8MM and several tokens, rows go two at a time against token pairs
/// on the matrix instructions; otherwise (and for an odd row or token left
/// over) each row takes four tokens per weight load with SDOT.
#[allow(clippy::too_many_arguments)]
fn int_matmul(
    pool: &Pool,
    m: &impl IntRows,
    rows: usize,
    cols: usize,
    x: &[f32],
    t: usize,
    grain: usize,
    out: SyncPtr<f32>,
) {
    let (xs, xq) = quantize_input(x, t, cols);
    let nb = cols / QK;
    let tok = |i: usize| (&xs[i * nb..(i + 1) * nb], &xq[i * cols..(i + 1) * cols]);
    let paired = (t >= 2 && kernels::has_i8mm()).then(|| kernels::pair_tokens(&xs, &xq, t, cols));
    pool.parallel_for(rows, grain, |range| {
        // SAFETY: each (token, row) of `out` is written by exactly one thread.
        let put = |i: usize, r: usize, y: f32| unsafe { out.write(i * rows + r, y) };
        let mut r = range.start;
        if let Some((ps, pq)) = &paired {
            let pair = |p: usize| {
                (
                    &ps[p * 2 * nb..(p + 1) * 2 * nb],
                    &pq[p * 2 * cols..(p + 1) * 2 * cols],
                )
            };
            // Rows r and r + 1 against tokens 2p and 2p + 1.
            let store = |r: usize, p: usize, y: [f32; 4]| {
                put(2 * p, r, y[0]);
                put(2 * p + 1, r, y[1]);
                put(2 * p, r + 1, y[2]);
                put(2 * p + 1, r + 1, y[3]);
            };
            while r + 2 <= range.end {
                let mut p = 0;
                while p + 4 <= t / 2 {
                    let g = [pair(p), pair(p + 1), pair(p + 2), pair(p + 3)];
                    let ys = m.dot2_pairs(r, g.map(|x| x.0), g.map(|x| x.1));
                    for (k, y) in ys.into_iter().enumerate() {
                        store(r, p + k, y);
                    }
                    p += 4;
                }
                while p + 2 <= t / 2 {
                    let (a, b) = (pair(p), pair(p + 1));
                    let ys = m.dot2_pairs(r, [a.0, b.0], [a.1, b.1]);
                    store(r, p, ys[0]);
                    store(r, p + 1, ys[1]);
                    p += 2;
                }
                if p < t / 2 {
                    store(r, p, m.dot2_pairs(r, [pair(p).0], [pair(p).1])[0]);
                }
                if t % 2 == 1 {
                    let (s, q) = tok(t - 1);
                    put(t - 1, r, m.dot(r, s, q));
                    put(t - 1, r + 1, m.dot(r + 1, s, q));
                }
                r += 2;
            }
        }
        for r in r..range.end {
            // Four tokens at a time share each weight block load.
            let mut i = 0;
            while i + 4 <= t {
                let ys = m.dot_x4(
                    r,
                    [0, 1, 2, 3].map(|j| tok(i + j).0),
                    [0, 1, 2, 3].map(|j| tok(i + j).1),
                );
                for (j, y) in ys.into_iter().enumerate() {
                    put(i + j, r, y);
                }
                i += 4;
            }
            for i in i..t {
                let (s, q) = tok(i);
                put(i, r, m.dot(r, s, q));
            }
        }
    });
}

fn quantize_input(x: &[f32], t: usize, cols: usize) -> (Vec<f32>, Vec<i8>) {
    let (mut xs, mut xq) = (vec![0.0; t * cols / QK], vec![0i8; t * cols]);
    kernels::quantize_q8(x, &mut xs, &mut xq);
    (xs, xq)
}
