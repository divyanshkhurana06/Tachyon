//! Weight matrices in each storage format, and the parallel matrix multiply
//! over them.

use crate::kernels::{
    self, dot_bf16, dot_q4, dot_q4_x4, dot_q8, dot_q8_x4, f32_to_bf16, QK, SUPER, SUPER_WORDS,
};
use crate::pool::{Pool, SyncPtr};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// 16 bits per weight, the checkpoint's own precision.
    Bf16,
    /// 8.5 bits per weight (int8 plus one f16 scale per 32).
    Q8,
    /// 4.625 bits per weight for most matrices (4-bit values, and 8-bit
    /// scales and offsets per 32 in super-blocks of 256). The small,
    /// sensitive q/k/v projection and the output layer stay at 8 bits.
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
        /// f16, one per block.
        scales: Vec<u16>,
        qs: Vec<i8>,
    },
    Q4 {
        rows: usize,
        cols: usize,
        /// Packed scales and offsets, five words per super-block of a row
        /// (see `kernels::quantize_q4_packed`).
        scales: Vec<u32>,
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
                let (mut scales, mut qs) = (vec![0; blocks], vec![0i8; rows * cols]);
                kernels::quantize_q8_weights(data, &mut scales, &mut qs);
                Matrix::Q8 {
                    rows,
                    cols,
                    scales,
                    qs,
                }
            }
            Format::Q4 | Format::Q4M => Matrix::q4_offset(data, rows, cols, None),
        }
    }

    /// 4-bit with per-block offsets, fitted to `importance` if given.
    fn q4_offset(data: &[f32], rows: usize, cols: usize, importance: Option<&[f32]>) -> Matrix {
        let words = q4_words(cols);
        let mut scales = vec![0u32; rows * words];
        let mut qs = vec![0u8; rows * cols / 2];
        for r in 0..rows {
            kernels::quantize_q4_packed(
                &data[r * cols..(r + 1) * cols],
                importance,
                &mut scales[r * words..(r + 1) * words],
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
        Matrix::q4_offset(data, rows, cols, Some(imp))
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
            Matrix::Q8 { scales, qs, .. } => scales.len() * 2 + qs.len(),
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
            Matrix::Q8 { scales, qs, .. } => {
                let mut d = vec![0.0; nb];
                kernels::unpack_f16(&scales[r * nb..(r + 1) * nb], &mut d);
                kernels::dequantize_q8(&d, &qs[r * cols..(r + 1) * cols], out)
            }
            Matrix::Q4 { scales, qs, .. } => {
                let w = q4_words(cols);
                let (mut d, mut c) = (vec![0.0; nb], vec![0.0; nb]);
                kernels::unpack_q4(&scales[r * w..(r + 1) * w], &mut d, &mut c);
                kernels::dequantize_q4(&d, &qs[r * cols / 2..(r + 1) * cols / 2], out);
                for (block, &c) in out.chunks_exact_mut(QK).zip(&c) {
                    block.iter_mut().for_each(|v| *v += c);
                }
            }
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

/// Words of packed 4-bit scales per row of `cols` weights.
pub fn q4_words(cols: usize) -> usize {
    (cols / QK).div_ceil(SUPER) * SUPER_WORDS
}

/// Rows whose scales are expanded together, in one loop.
const GROUP: usize = 8;

/// Per-block scales (and 4-bit offsets) of up to [`GROUP`] rows as f32, for
/// several tokens: stored compactly, they are expanded a few rows at a time
/// into this scratch space and reused for every token. (A single token's
/// kernels convert them in registers instead; see `kernels::Scales`.)
#[derive(Default)]
struct Scratch {
    d: Vec<f32>,
    c: Vec<f32>,
}

thread_local! {
    /// Each thread's scratch, kept between matmuls.
    static SCRATCH: std::cell::RefCell<Scratch> = Default::default();
}

/// One row's expanded scales and offsets.
#[derive(Clone, Copy)]
struct Row<'a> {
    d: &'a [f32],
    c: &'a [f32],
}

/// The rows of an 8- or 4-bit matrix, for the shared integer matmul.
trait IntRows: Sync {
    /// Expands the scales (and offsets) of rows `r..r + n` into `d` and
    /// `c`, a row's blocks after another's: `nb` apart for 8-bit rows,
    /// padded to whole super-blocks for 4-bit ones.
    fn scales(&self, r: usize, n: usize, d: &mut [f32], c: &mut [f32]);
    /// Row `r` times one token, with the scales as stored: the same value
    /// as [`IntRows::dot`] with them expanded.
    fn dot1(&self, r: usize, xs: &[f32], xq: &[i8]) -> f32;
    fn dot(&self, r: usize, s: Row, xs: &[f32], xq: &[i8]) -> f32;
    fn dot_x4(&self, r: usize, s: Row, xs: [&[f32]; 4], xq: [&[i8]; 4]) -> [f32; 4];
    /// Whether rows have per-block offsets: their share of a row times a
    /// token is the offsets times the token's blocks' input sums (`scale *
    /// sum of quantized values`), which [`IntRows::offset`] adds up.
    fn has_offsets(&self) -> bool {
        false
    }
    fn offset(&self, _r: usize, _sums: &[f32]) -> f32 {
        0.0
    }
    /// Rows `r` and `r + 1` against token pairs (see `kernels::pair_tokens`).
    fn dot2_pairs<const P: usize>(
        &self,
        r: usize,
        s: [Row; 2],
        ps: [&[f32]; P],
        pq: [&[i8]; P],
    ) -> [[f32; 4]; P];
}

struct Q8Rows<'a> {
    scales: &'a [u16],
    qs: &'a [i8],
    cols: usize,
    /// Row `r` is the matrix's row `row0 + r`.
    row0: usize,
}

impl Q8Rows<'_> {
    fn quants(&self, r: usize) -> &[i8] {
        let r = self.row0 + r;
        &self.qs[r * self.cols..(r + 1) * self.cols]
    }
}

impl IntRows for Q8Rows<'_> {
    fn scales(&self, r: usize, n: usize, d: &mut [f32], _: &mut [f32]) {
        let (nb, r) = (self.cols / QK, self.row0 + r);
        kernels::unpack_f16(&self.scales[r * nb..(r + n) * nb], &mut d[..n * nb]);
    }
    fn dot1(&self, r: usize, xs: &[f32], xq: &[i8]) -> f32 {
        let (nb, row) = (self.cols / QK, self.row0 + r);
        let ws = kernels::F16Scales(&self.scales[row * nb..(row + 1) * nb]);
        kernels::dot_q8_with(ws, self.quants(r), xs, xq)
    }
    fn dot(&self, r: usize, s: Row, xs: &[f32], xq: &[i8]) -> f32 {
        dot_q8(s.d, self.quants(r), xs, xq)
    }
    fn dot_x4(&self, r: usize, s: Row, xs: [&[f32]; 4], xq: [&[i8]; 4]) -> [f32; 4] {
        dot_q8_x4(s.d, self.quants(r), xs, xq)
    }
    fn dot2_pairs<const P: usize>(
        &self,
        r: usize,
        s: [Row; 2],
        ps: [&[f32]; P],
        pq: [&[i8]; P],
    ) -> [[f32; 4]; P] {
        let w = [self.quants(r), self.quants(r + 1)];
        kernels::dot2_q8_pairs([s[0].d, s[1].d], w, ps, pq)
    }
}

struct Q4Rows<'a> {
    scales: &'a [u32],
    qs: &'a [u8],
    cols: usize,
    row0: usize,
}

impl Q4Rows<'_> {
    fn quants(&self, r: usize) -> &[u8] {
        let (half, r) = (self.cols / 2, self.row0 + r);
        &self.qs[r * half..(r + 1) * half]
    }
    fn words(&self, r: usize) -> &[u32] {
        let (w, r) = (q4_words(self.cols), self.row0 + r);
        &self.scales[r * w..(r + 1) * w]
    }
}

impl IntRows for Q4Rows<'_> {
    fn scales(&self, r: usize, n: usize, d: &mut [f32], c: &mut [f32]) {
        // Whole super-blocks: past the row's end the packed bytes are zero.
        let (padded, w) = (
            (self.cols / QK).div_ceil(SUPER) * SUPER,
            q4_words(self.cols),
        );
        for i in 0..n {
            let row = self.row0 + r + i;
            kernels::unpack_q4(
                &self.scales[row * w..(row + 1) * w],
                &mut d[i * padded..(i + 1) * padded],
                &mut c[i * padded..(i + 1) * padded],
            );
        }
    }
    fn has_offsets(&self) -> bool {
        true
    }
    fn offset(&self, r: usize, sums: &[f32]) -> f32 {
        kernels::dot_offsets(self.words(r), sums)
    }
    fn dot1(&self, r: usize, xs: &[f32], xq: &[i8]) -> f32 {
        let ws = kernels::PackedScales::new(self.words(r), self.cols / QK);
        kernels::dot_q4_with(ws, self.quants(r), xs, xq)
    }
    fn dot(&self, r: usize, s: Row, xs: &[f32], xq: &[i8]) -> f32 {
        dot_q4(s.d, self.quants(r), xs, xq)
    }
    fn dot_x4(&self, r: usize, s: Row, xs: [&[f32]; 4], xq: [&[i8]; 4]) -> [f32; 4] {
        dot_q4_x4(s.d, self.quants(r), xs, xq)
    }
    fn dot2_pairs<const P: usize>(
        &self,
        r: usize,
        s: [Row; 2],
        ps: [&[f32]; P],
        pq: [&[i8]; P],
    ) -> [[f32; 4]; P] {
        let w = [self.quants(r), self.quants(r + 1)];
        kernels::dot2_q4_pairs([s[0].d, s[1].d], w, ps, pq)
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
    // For per-block offsets: each token's blocks' input sums, padded with
    // zeros to whole super-blocks (`padded` per token), as the offset dots
    // take them. Expanded offsets are padded the same way.
    let padded = nb.div_ceil(SUPER) * SUPER;
    let mut sums = Vec::new();
    if m.has_offsets() {
        sums.resize(t * padded, 0.0);
        for i in 0..t {
            for b in 0..nb {
                let k = i * nb + b;
                let q: i32 = xq[k * QK..(k + 1) * QK].iter().map(|&q| q as i32).sum();
                sums[i * padded + b] = xs[k] * q as f32;
            }
        }
    }
    let stride = if m.has_offsets() { padded } else { nb };
    pool.parallel_for(rows, grain, |range| {
        // SAFETY: each (token, row) of `out` is written by exactly one thread.
        let put = |i: usize, r: usize, s: Option<Row>, y: f32| {
            let y = match (sums.is_empty(), s) {
                (true, _) => y,
                (false, None) => y + m.offset(r, &sums[i * padded..(i + 1) * padded]),
                (false, Some(s)) => {
                    y + kernels::dot_offsets_expanded(s.c, &sums[i * padded..(i + 1) * padded])
                }
            };
            unsafe { out.write(i * rows + r, y) }
        };
        if t == 1 {
            for r in range {
                put(0, r, None, m.dot1(r, &xs, &xq));
            }
            return;
        }
        SCRATCH.with_borrow_mut(|scratch| {
            scratch.d.resize(GROUP * stride, 0.0);
            scratch.c.resize(GROUP * stride, 0.0);
            for g in range.clone().step_by(GROUP) {
                let end = (g + GROUP).min(range.end);
                m.scales(g, end - g, &mut scratch.d, &mut scratch.c);
                let row = |r: usize| Row {
                    d: &scratch.d[(r - g) * stride..][..nb],
                    c: &scratch.c[(r - g) * stride..][..stride],
                };
                let mut r = g;
                if let Some((ps, pq)) = &paired {
                    let pair = |p: usize| {
                        (
                            &ps[p * 2 * nb..(p + 1) * 2 * nb],
                            &pq[p * 2 * cols..(p + 1) * 2 * cols],
                        )
                    };
                    while r + 2 <= end {
                        let s = [row(r), row(r + 1)];
                        // Rows r and r + 1 against tokens 2p and 2p + 1.
                        let store = |p: usize, y: [f32; 4]| {
                            put(2 * p, r, Some(s[0]), y[0]);
                            put(2 * p + 1, r, Some(s[0]), y[1]);
                            put(2 * p, r + 1, Some(s[1]), y[2]);
                            put(2 * p + 1, r + 1, Some(s[1]), y[3]);
                        };
                        let mut p = 0;
                        while p + 4 <= t / 2 {
                            let g = [pair(p), pair(p + 1), pair(p + 2), pair(p + 3)];
                            let ys = m.dot2_pairs(r, s, g.map(|x| x.0), g.map(|x| x.1));
                            for (k, y) in ys.into_iter().enumerate() {
                                store(p + k, y);
                            }
                            p += 4;
                        }
                        while p + 2 <= t / 2 {
                            let (a, b) = (pair(p), pair(p + 1));
                            let ys = m.dot2_pairs(r, s, [a.0, b.0], [a.1, b.1]);
                            store(p, ys[0]);
                            store(p + 1, ys[1]);
                            p += 2;
                        }
                        if p < t / 2 {
                            store(p, m.dot2_pairs(r, s, [pair(p).0], [pair(p).1])[0]);
                        }
                        if t % 2 == 1 {
                            let (xs, xq) = tok(t - 1);
                            put(t - 1, r, Some(s[0]), m.dot(r, s[0], xs, xq));
                            put(t - 1, r + 1, Some(s[1]), m.dot(r + 1, s[1], xs, xq));
                        }
                        r += 2;
                    }
                }
                for r in r..end {
                    let s = row(r);
                    // Four tokens at a time share each weight block load.
                    let mut i = 0;
                    while i + 4 <= t {
                        let ys = m.dot_x4(
                            r,
                            s,
                            [0, 1, 2, 3].map(|j| tok(i + j).0),
                            [0, 1, 2, 3].map(|j| tok(i + j).1),
                        );
                        for (j, y) in ys.into_iter().enumerate() {
                            put(i + j, r, Some(s), y);
                        }
                        i += 4;
                    }
                    for i in i..t {
                        let (xs, xq) = tok(i);
                        put(i, r, Some(s), m.dot(r, s, xs, xq));
                    }
                }
            }
        });
    });
}

fn quantize_input(x: &[f32], t: usize, cols: usize) -> (Vec<f32>, Vec<i8>) {
    let (mut xs, mut xq) = (vec![0.0; t * cols / QK], vec![0i8; t * cols]);
    kernels::quantize_q8(x, &mut xs, &mut xq);
    (xs, xq)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    }

    /// Every path through the integer matmul (token pairs, fours, an odd
    /// token, row chunks) gives each output the dequantized row times the
    /// token's 8-bit activations.
    #[test]
    fn matmul_matches_dequantized_rows() {
        let pool = Pool::new(3);
        let (rows, cols) = (37, 11 * QK);
        let w = data(rows * cols, 1);
        for format in [Format::Q8, Format::Q4] {
            let m = Matrix::from_f32(&w, rows, cols, format);
            let mut deq = vec![0.0; rows * cols];
            for r in 0..rows {
                m.row_into(r, &mut deq[r * cols..(r + 1) * cols]);
            }
            for t in [1, 2, 3, 5, 8, 9] {
                let x = data(t * cols, t as u64 + 7);
                let mut out = vec![0.0; t * rows];
                m.matmul(&pool, &x, t, &mut out);
                // The activations as the kernels see them, 8-bit.
                let (xs, xq) = quantize_input(&x, t, cols);
                let mut xd = vec![0.0; t * cols];
                kernels::dequantize_q8(&xs, &xq, &mut xd);
                // Each token alone gives exactly its result in the batch.
                for i in 0..t {
                    let mut alone = vec![0.0; rows];
                    m.matmul(&pool, &x[i * cols..(i + 1) * cols], 1, &mut alone);
                    assert_eq!(
                        alone,
                        out[i * rows..(i + 1) * rows],
                        "{format:?} t={t} token {i}"
                    );
                }
                for i in 0..t {
                    let xi = &xd[i * cols..(i + 1) * cols];
                    for r in 0..rows {
                        let want: f32 = deq[r * cols..(r + 1) * cols]
                            .iter()
                            .zip(xi)
                            .map(|(a, b)| a * b)
                            .sum();
                        let got = out[i * rows + r];
                        assert!(
                            (got - want).abs() < 1e-4,
                            "{format:?} t={t} token {i} row {r}: {got} vs {want}"
                        );
                    }
                }
            }
        }
    }
}
