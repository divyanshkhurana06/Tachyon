//! Inner loops: a dot product for each weight format, quantizers, and the
//! small element-wise ops around them. NEON intrinsics on aarch64 (every
//! Apple Silicon Mac), AVX2 on x86-64, SIMD128 in the browser, plain Rust
//! elsewhere.
//!
//! Quantized formats store weights in blocks of [`QK`] values sharing a
//! scale (f16 for 8-bit, 6-bit and packed for 4-bit), which the dot products
//! take expanded to f32 a row at a time. Activations are quantized to 8 bits
//! on the fly so the inner loop is integer multiply-accumulate, then each
//! block's sum is scaled once.

/// Values per quantization block.
pub const QK: usize = 32;

#[inline]
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Round to nearest, ties to even (what PyTorch does).
pub fn f32_to_bf16(f: f32) -> u16 {
    if f.is_nan() {
        return 0x7FC0;
    }
    let bits = f.to_bits();
    let round = 0x7FFF + ((bits >> 16) & 1);
    (bits.wrapping_add(round) >> 16) as u16
}

/// Rounds to the nearest half-precision float, ties to even.
pub fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xFF) as i32;
    let mant = x & 0x7F_FFFF;
    if exp == 0xFF {
        return sign | 0x7C00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1F {
        return sign | 0x7C00;
    }
    // The mantissa with its leading 1, and how far it shifts to fit 10 bits
    // (further for subnormals).
    let (full, shift, base) = if e > 0 {
        (mant, 13, (e as u32) << 10)
    } else {
        (mant | 0x80_0000, (14 - e) as u32, 0)
    };
    if shift > 25 {
        return sign;
    }
    let mut h = base | (full >> shift);
    let rem = full & ((1 << shift) - 1);
    let half = 1 << (shift - 1);
    if rem > half || (rem == half && h & 1 == 1) {
        h += 1; // a carry rolls into the exponent, as it should
    }
    sign | h as u16
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1F) as u32;
    let frac = (h & 0x3FF) as u32;
    let bits = match (exp, frac) {
        (0, 0) => sign,
        (0, mut f) => {
            // subnormal: renormalize
            let mut e = 127 - 15 + 1;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((f & 0x3FF) << 13)
        }
        (31, f) => sign | 0x7F80_0000 | (f << 13),
        (e, f) => sign | ((e + 127 - 15) << 23) | (f << 13),
    };
    f32::from_bits(bits)
}

/// Symmetric 8-bit: each block stores `max|x| / 127` and values in -127..=127.
pub fn quantize_q8(x: &[f32], scales: &mut [f32], qs: &mut [i8]) {
    for ((block, q), scale) in x
        .chunks_exact(QK)
        .zip(qs.chunks_exact_mut(QK))
        .zip(scales.iter_mut())
    {
        let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        for (qi, &v) in q.iter_mut().zip(block) {
            *qi = (v * id).round() as i8;
        }
        *scale = d;
    }
}

/// 4-bit, llama.cpp's Q4_0 layout: 16 bytes per block, value j in the low
/// nibble of byte j and value j+16 in the high nibble, stored offset by 8.
/// The scale maps the largest-magnitude value exactly to -8.
pub fn quantize_q4(x: &[f32], scales: &mut [f32], qs: &mut [u8]) {
    for ((block, q), scale) in x
        .chunks_exact(QK)
        .zip(qs.chunks_exact_mut(QK / 2))
        .zip(scales.iter_mut())
    {
        let max = block
            .iter()
            .copied()
            .fold(0f32, |m, v| if v.abs() > m.abs() { v } else { m });
        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let nibble = |v: f32| ((v * id + 8.5) as i32).clamp(0, 15) as u8;
        for j in 0..QK / 2 {
            q[j] = nibble(block[j]) | (nibble(block[j + QK / 2]) << 4);
        }
        *scale = d;
    }
}

/// f16 to f32 for finite values without branches, so loops of it
/// vectorize: the bits shifted into place, then rescaled by the difference
/// in exponent bias (which also renormalizes subnormals).
#[inline]
pub fn f16_finite(h: u16) -> f32 {
    let (h, sign) = (h as u32 & 0x7FFF, (h as u32 & 0x8000) << 16);
    f32::from_bits(sign | (h << 13)) * f32::from_bits(0x7780_0000) // 2^112
}

/// Weights' 8-bit blocks, as [`quantize_q8`] but with each scale stored as
/// f16 (8.5 bits per weight, llama.cpp's Q8_0): values are rounded against
/// the f16 scale, so they stay within -127..=127 of it.
pub fn quantize_q8_weights(x: &[f32], scales: &mut [u16], qs: &mut [i8]) {
    for ((block, q), scale) in x
        .chunks_exact(QK)
        .zip(qs.chunks_exact_mut(QK))
        .zip(scales.iter_mut())
    {
        let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
        *scale = f32_to_f16(amax / 127.0);
        let d = f16_finite(*scale);
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        for (qi, &v) in q.iter_mut().zip(block) {
            *qi = (v * id).round().clamp(-127.0, 127.0) as i8;
        }
    }
}

/// f16 scales to f32, a row at a time (NEON on aarch64: rows of 8-bit
/// weights can be as short as 16 blocks, where a plain loop costs a quarter
/// as much as the dot product).
pub fn unpack_f16(h: &[u16], out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    let done = neon::unpack_f16_scales(h, out);
    #[cfg(not(target_arch = "aarch64"))]
    let done = 0;
    for (o, &h) in out[done..].iter_mut().zip(&h[done..]) {
        *o = f16_finite(h);
    }
}

/// 4-bit blocks per super-block of packed scales.
pub const SUPER: usize = 8;
/// Words of packed scales per super-block.
pub const SUPER_WORDS: usize = 5;

/// Packed 4-bit scales: `value = d * (q - 8) + c` per block as before, but
/// each [`SUPER`] blocks (256 weights) of a row share five words: f16 steps
/// `D` and `M` (word 0), then each block's 8-bit scale `s` in 0..=255 (words
/// 1 and 2, a byte per block) and signed offset `o` in -128..=127 (words 3
/// and 4), for `d = D * s` and `c = M * o`. With 16 bytes of 4-bit values
/// per block that is 4.625 bits per weight, from 6 with an f32 scale and
/// offset per block. (llama.cpp's Q4_K packs 6-bit fields into 4.5 bits per
/// weight; whole bytes widen to floats in a few vector instructions, and the
/// fields are expanded once per row per matrix multiply, so for a single
/// token their cost is on the same scale as the dot product's.) A row's last
/// super-block may hold fewer blocks.
pub fn quantize_q4_packed(
    row: &[f32],
    importance: Option<&[f32]>,
    packed: &mut [u32],
    qs: &mut [u8],
) {
    let sigma2 = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
    let nb = row.len() / QK;
    for (k, words) in packed.chunks_exact_mut(SUPER_WORDS).enumerate() {
        let blocks = k * SUPER..((k + 1) * SUPER).min(nb);
        // Each block's own best scale and offset, unconstrained.
        let mut fits = Vec::with_capacity(SUPER);
        for b in blocks.clone() {
            let x = &row[b * QK..(b + 1) * QK];
            let mut w = [0f32; QK];
            for j in 0..QK {
                let imp = importance.map_or(1.0, |i| i[b * QK + j]);
                w[j] = imp * (sigma2 + x[j] * x[j]).sqrt();
            }
            let (d, c, _) = fit_q4_offset(x, &w);
            fits.push((d.max(0.0), c, w));
        }
        let dmax = fits.iter().fold(0f32, |m, f| m.max(f.0));
        let cmax = fits.iter().fold(0f32, |m, f| m.max(f.1.abs()));
        let (dh, mh) = (f32_to_f16(dmax / 255.0), f32_to_f16(cmax / 127.0));
        let (step_d, step_c) = (f16_finite(dh), f16_finite(mh));
        words.fill(0);
        words[0] = dh as u32 | (mh as u32) << 16;
        for (j, (b, (d, c, w))) in blocks.zip(fits).enumerate() {
            let x = &row[b * QK..(b + 1) * QK];
            // The four roundings of (scale, offset) to the grid around the
            // free fit, each with its levels refitted; the closest wins.
            let near = |v: f32, step: f32, lo: f32, hi: f32| {
                let f = if step > 0.0 { (v / step).floor() } else { 0.0 };
                [f.clamp(lo, hi), (f + 1.0).clamp(lo, hi)]
            };
            let mut best = (f32::INFINITY, 0u32, 0u32, [8u8; QK]);
            for s in near(d, step_d, 0.0, 255.0) {
                for o in near(c, step_c, -128.0, 127.0) {
                    let (dq, cq) = (step_d * s, step_c * o);
                    let id = if dq > 0.0 { 1.0 / dq } else { 0.0 };
                    let mut levels = [8u8; QK];
                    let mut err = 0.0;
                    for i in 0..QK {
                        let l = ((x[i] - cq) * id + 8.0).round().clamp(0.0, 15.0);
                        levels[i] = l as u8;
                        err += w[i] * (x[i] - dq * (l - 8.0) - cq).powi(2);
                    }
                    if err < best.0 {
                        best = (err, s as u32, o as i32 as u8 as u32, levels);
                    }
                }
            }
            let (_, s, o, levels) = best;
            words[1 + j / 4] |= s << (8 * (j % 4));
            words[3 + j / 4] |= o << (8 * (j % 4));
            let q = &mut qs[b * QK / 2..(b + 1) * QK / 2];
            for i in 0..QK / 2 {
                q[i] = levels[i] | (levels[i + QK / 2] << 4);
            }
        }
    }
}

/// One row's packed 4-bit scales ([`quantize_q4_packed`]) as each block's
/// `d` and `c`. This runs once per row per matrix multiply, so for a single
/// token it costs about as much as the dot product itself unless it is
/// vectorized (NEON on aarch64).
pub fn unpack_q4(packed: &[u32], d: &mut [f32], c: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    neon::unpack_q4_scales(packed, d, c);
    #[cfg(not(target_arch = "aarch64"))]
    unpack_q4_scalar(packed, d, c);
}

#[cfg_attr(target_arch = "aarch64", allow(dead_code))] // tests compare against it
fn unpack_q4_scalar(packed: &[u32], d: &mut [f32], c: &mut [f32]) {
    let nb = d.len();
    let supers = packed.chunks_exact(SUPER_WORDS).take(nb.div_ceil(SUPER));
    for ((w, d), c) in supers.zip(d.chunks_mut(SUPER)).zip(c.chunks_mut(SUPER)) {
        let (step_d, step_c) = (f16_finite(w[0] as u16), f16_finite((w[0] >> 16) as u16));
        for (j, (d, c)) in d.iter_mut().zip(c).enumerate() {
            let (s, o) = (w[1 + j / 4] >> (8 * (j % 4)), w[3 + j / 4] >> (8 * (j % 4)));
            *d = step_d * (s as u8) as f32;
            *c = step_c * (o as u8 as i8) as f32;
        }
    }
}

#[cfg(target_endian = "big")]
compile_error!("packed scales are read as little-endian bytes");

/// Where a row's per-block weight scales come from, for the single-token
/// kernels, which convert compact scales in registers as they go: a
/// matrix-vector product's dot products are bound by their chain of
/// multiply-adds, so the conversion fits in spare issue slots, where
/// expanding a row into memory first costs about a tenth of decode time.
/// Every source gives a block exactly the scale the expanded rows hold.
pub trait Scales: Copy {
    fn blocks(self) -> usize;
    /// # Safety
    /// `b < self.blocks()`.
    unsafe fn get(self, b: usize) -> f32;
    /// Blocks `8k..8k + 8`, the same values as [`Scales::get`].
    ///
    /// # Safety
    /// `8k + 8 <= self.blocks()`.
    #[inline(always)]
    unsafe fn eight(self, k: usize) -> [f32; SUPER] {
        std::array::from_fn(|j| self.get(k * SUPER + j))
    }
}

impl Scales for &[f32] {
    fn blocks(self) -> usize {
        self.len()
    }
    #[inline(always)]
    unsafe fn get(self, b: usize) -> f32 {
        *self.get_unchecked(b)
    }
    #[inline(always)]
    unsafe fn eight(self, k: usize) -> [f32; SUPER] {
        *(self.as_ptr().add(k * SUPER) as *const [f32; SUPER])
    }
}

/// f16 scales, one per block (8-bit weights).
#[derive(Clone, Copy)]
pub struct F16Scales<'a>(pub &'a [u16]);

impl Scales for F16Scales<'_> {
    fn blocks(self) -> usize {
        self.0.len()
    }
    #[inline(always)]
    unsafe fn get(self, b: usize) -> f32 {
        f16_finite(*self.0.get_unchecked(b))
    }
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn eight(self, k: usize) -> [f32; SUPER] {
        neon::f16_eight(self.0.as_ptr().add(k * SUPER))
    }
}

/// A row's packed 4-bit scales ([`quantize_q4_packed`]).
#[derive(Clone, Copy)]
pub struct PackedScales<'a> {
    words: &'a [u32],
    blocks: usize,
}

impl<'a> PackedScales<'a> {
    pub fn new(words: &'a [u32], blocks: usize) -> PackedScales<'a> {
        assert!(words.len() >= blocks.div_ceil(SUPER) * SUPER_WORDS);
        PackedScales { words, blocks }
    }
}

impl Scales for PackedScales<'_> {
    fn blocks(self) -> usize {
        self.blocks
    }
    #[inline(always)]
    unsafe fn get(self, b: usize) -> f32 {
        let w = self.words.as_ptr().add(b / SUPER * SUPER_WORDS);
        // A block's scale byte: words 1 and 2 as bytes.
        let s = *(w.add(1) as *const u8).add(b % SUPER);
        f16_finite(*w as u16) * s as f32
    }
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn eight(self, k: usize) -> [f32; SUPER] {
        neon::packed_eight(self.words.as_ptr().add(k * SUPER_WORDS))
    }
}

/// `f` on the scales expanded to f32 (for a kernel that takes them so).
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
fn expanded<R>(ws: impl Scales, f: impl FnOnce(&[f32]) -> R) -> R {
    let nb = ws.blocks();
    let (mut stack, mut heap) = ([0f32; 512], Vec::new());
    let buf = if nb <= stack.len() {
        &mut stack[..nb]
    } else {
        heap.resize(nb, 0.0);
        &mut heap[..]
    };
    for (b, v) in buf.iter_mut().enumerate() {
        // SAFETY: b < nb.
        *v = unsafe { ws.get(b) };
    }
    f(buf)
}

/// [`dot_q8`] with the scales as they are stored: the same result.
pub fn dot_q8_with(ws: impl Scales, wq: &[i8], xs: &[f32], xq: &[i8]) -> f32 {
    simd::dot_q8_with(ws, wq, xs, xq)
}

/// [`dot_q4`] with the scales as they are stored: the same result.
pub fn dot_q4_with(ws: impl Scales, wq: &[u8], xs: &[f32], xq: &[i8]) -> f32 {
    simd::dot_q4_with(ws, wq, xs, xq)
}

/// A row's packed 4-bit offsets against a token's blocks' input sums
/// (`scale * sum of quantized values` per block): `sum_b c_b sums_b`, read
/// straight from the packed bytes. `sums` is padded with zeros to whole
/// super-blocks (a row's packed bytes past its end are zero too). The same
/// result as [`dot_offsets_expanded`] on the offsets [`unpack_q4`] gives:
/// eight lanes, one per block of a super-block, added up at the end.
pub fn dot_offsets(words: &[u32], sums: &[f32]) -> f32 {
    assert!(sums.len() % SUPER == 0 && words.len() >= sums.len() / SUPER * SUPER_WORDS);
    #[cfg(target_arch = "aarch64")]
    return neon::dot_offsets(words, sums);
    #[cfg(not(target_arch = "aarch64"))]
    offsets_lanes(sums, |k, c| {
        let w = &words[k * SUPER_WORDS..];
        let step = f16_finite((w[0] >> 16) as u16);
        for (j, c) in c.iter_mut().enumerate() {
            *c = step * ((w[3 + j / 4] >> (8 * (j % 4))) as u8 as i8) as f32;
        }
    })
}

/// [`dot_offsets`] on expanded offsets (as many as `sums`), for several
/// tokens.
pub fn dot_offsets_expanded(c: &[f32], sums: &[f32]) -> f32 {
    assert!(sums.len() % SUPER == 0 && c.len() >= sums.len());
    #[cfg(target_arch = "aarch64")]
    return neon::dot_offsets_expanded(c, sums);
    #[cfg(not(target_arch = "aarch64"))]
    offsets_lanes(sums, |k, out| {
        out.copy_from_slice(&c[k * SUPER..(k + 1) * SUPER])
    })
}

/// The lanes both offset dots share: `fill(k, c)` writes super-block `k`'s
/// offsets.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn offsets_lanes(sums: &[f32], mut fill: impl FnMut(usize, &mut [f32; SUPER])) -> f32 {
    let mut acc = [0f32; SUPER];
    for (k, x) in sums.chunks_exact(SUPER).enumerate() {
        let mut c = [0f32; SUPER];
        fill(k, &mut c);
        for j in 0..SUPER {
            acc[j] = c[j].mul_add(x[j], acc[j]);
        }
    }
    let a: [f32; 4] = std::array::from_fn(|j| acc[j] + acc[j + 4]);
    (a[0] + a[1]) + (a[2] + a[3])
}

/// (scale, offset, levels) for one block, minimizing `sum_j w_j (x_j -
/// value_j)^2` by a weighted least-squares fit for each of several
/// roundings, as llama.cpp's k-quants search. Weights are `w_j = importance_j
/// * sqrt(sigma^2 + x_j^2)` (importance 1 without an importance matrix).
fn fit_q4_offset(x: &[f32], w: &[f32]) -> (f32, f32, [u8; QK]) {
    let (lo, hi) = x
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    if hi <= lo {
        return (0.0, lo, [8; QK]); // a constant block: all offset
    }
    let levels_for = |iscale: f32| {
        let mut l = [0u8; QK];
        for (l, &v) in l.iter_mut().zip(x) {
            *l = ((v - lo) * iscale).round().clamp(0.0, 15.0) as u8;
        }
        l
    };
    let err = |a: f32, m: f32, l: &[u8; QK]| {
        x.iter()
            .zip(l)
            .zip(w)
            .map(|((&v, &l), &w)| w * (v - a * l as f32 - m).powi(2))
            .sum::<f32>()
    };
    // The plain choice: the block's range in 15 steps from its minimum.
    let plain = levels_for(15.0 / (hi - lo));
    let mut best = (
        (hi - lo) / 15.0,
        lo,
        plain,
        err((hi - lo) / 15.0, lo, &plain),
    );
    for step in -10..=10 {
        let l = levels_for((15.0 + 0.1 * step as f32) / (hi - lo));
        // weighted least squares for x ≈ a * l + m
        let (mut sw, mut sl, mut sll, mut sx, mut slx) = (0f32, 0f32, 0f32, 0f32, 0f32);
        for ((&v, &l), &w) in x.iter().zip(&l).zip(w) {
            let l = l as f32;
            sw += w;
            sl += w * l;
            sll += w * l * l;
            sx += w * v;
            slx += w * l * v;
        }
        let det = sw * sll - sl * sl;
        if det <= 0.0 {
            continue;
        }
        let a = (sw * slx - sl * sx) / det;
        let m = (sll * sx - sl * slx) / det;
        let e = err(a, m, &l);
        if e < best.3 {
            best = (a, m, l, e);
        }
    }
    let (a, m, l, _) = best;
    // value = a * l + m = a * (l - 8) + (m + 8a)
    (a, m + 8.0 * a, l)
}

pub fn dequantize_q8(scales: &[f32], qs: &[i8], out: &mut [f32]) {
    for ((o, q), &d) in out
        .chunks_exact_mut(QK)
        .zip(qs.chunks_exact(QK))
        .zip(scales)
    {
        for (o, &q) in o.iter_mut().zip(q) {
            *o = q as f32 * d;
        }
    }
}

pub fn dequantize_q4(scales: &[f32], qs: &[u8], out: &mut [f32]) {
    for ((o, q), &d) in out
        .chunks_exact_mut(QK)
        .zip(qs.chunks_exact(QK / 2))
        .zip(scales)
    {
        for j in 0..QK / 2 {
            o[j] = ((q[j] & 0xF) as i32 - 8) as f32 * d;
            o[j + QK / 2] = ((q[j] >> 4) as i32 - 8) as f32 * d;
        }
    }
}

pub fn rmsnorm(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let ss = dot_f32(x, x) / x.len() as f32;
    let inv = 1.0 / (ss + eps).sqrt();
    for ((o, &v), &g) in out.iter_mut().zip(x).zip(w) {
        *o = v * inv * g;
    }
}

/// `exp` of each value, vectorized (NEON on aarch64): `2^n p(r)` with
/// `x = n ln 2 + r` and Cephes' degree-6 polynomial, within 2 ulp of `exp`
/// for inputs from -87 to 88 (below that it gives the smallest normal
/// float rather than 0). Softmax calls it once per score, so libm's scalar
/// `expf` was a seventh of CPU attention.
pub fn exp_inplace(xs: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    neon::exp_inplace(xs);
    #[cfg(not(target_arch = "aarch64"))]
    for x in xs {
        *x = exp_scalar(*x);
    }
}

const EXP_P: [f32; 6] = [
    1.987_569_2e-4,
    1.398_199_9e-3,
    8.333_452e-3,
    4.166_579_6e-2,
    0.166_666_65,
    0.5,
];

#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn exp_scalar(x: f32) -> f32 {
    let x = x.clamp(-87.336_55, 88.7);
    let n = (x * std::f32::consts::LOG2_E).round();
    let r = x - n * 0.693_359_4 - n * -2.121_944_4e-4;
    let mut y = EXP_P[0];
    for &c in &EXP_P[1..] {
        y = y * r + c;
    }
    let y = y * (r * r) + r + 1.0;
    f32::from_bits((y.to_bits() as i32 + ((n as i32) << 23)) as u32)
}

/// `q · k * scale` for each key, appended to `out` (attention scores). For
/// 64-wide heads on aarch64 the query stays in registers across keys.
pub fn attention_scores<'a>(
    q: &[f32],
    keys: impl Iterator<Item = &'a [f32]>,
    scale: f32,
    out: &mut Vec<f32>,
) {
    #[cfg(target_arch = "aarch64")]
    if q.len() == 64 {
        return neon::scores_64(q, keys, scale, out);
    }
    out.extend(keys.map(|k| dot_f32(q, k) * scale));
}

/// `out += w_p * row_p` over the rows (attention's weighted sum of
/// values). For 64-wide heads on aarch64 the output stays in registers.
pub fn weighted_sum<'a>(w: &[f32], rows: impl Iterator<Item = &'a [f32]>, out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    if out.len() == 64 {
        return neon::weighted_sum_64(w, rows, out);
    }
    for (&w, row) in w.iter().zip(rows) {
        axpy(out, w, row);
    }
}

/// `out = silu(g) * u` for each element (the SwiGLU MLP's gate), with
/// the vectorized exp.
pub fn silu_mul(g: &[f32], u: &[f32], out: &mut [f32]) {
    for (o, &g) in out.iter_mut().zip(g) {
        *o = -g;
    }
    exp_inplace(out);
    for ((o, &g), &u) in out.iter_mut().zip(g).zip(u) {
        *o = g / (1.0 + *o) * u;
    }
}

/// out += a * x
pub fn axpy(out: &mut [f32], a: f32, x: &[f32]) {
    for (o, &v) in out.iter_mut().zip(x) {
        *o += a * v;
    }
}

/// Interleaves quantized activations two tokens at a time for the I8MM
/// kernels ([`dot2_q8_pairs`]): for tokens 2p and 2p+1, every 8 values of
/// one are followed by the same 8 of the other, so each 16 bytes is the 8x2
/// matrix `smmla` takes, and their block scales sit side by side. A last odd
/// token is left out. Returns (scales, quants), `2 * cols / QK` scales and
/// `2 * cols` quants per pair.
pub fn pair_tokens(xs: &[f32], xq: &[i8], t: usize, cols: usize) -> (Vec<f32>, Vec<i8>) {
    let nb = cols / QK;
    let pairs = t / 2;
    let (mut ps, mut pq) = (vec![0.0; pairs * 2 * nb], vec![0i8; pairs * 2 * cols]);
    for p in 0..pairs {
        let (a, b) = (2 * p * cols, (2 * p + 1) * cols);
        for (blk, s) in ps[p * 2 * nb..(p + 1) * 2 * nb]
            .chunks_exact_mut(2)
            .enumerate()
        {
            s[0] = xs[2 * p * nb + blk];
            s[1] = xs[(2 * p + 1) * nb + blk];
        }
        for (c, chunk) in pq[p * 2 * cols..(p + 1) * 2 * cols]
            .chunks_exact_mut(16)
            .enumerate()
        {
            chunk[..8].copy_from_slice(&xq[a + c * 8..a + c * 8 + 8]);
            chunk[8..].copy_from_slice(&xq[b + c * 8..b + c * 8 + 8]);
        }
    }
    (ps, pq)
}

// Each platform's kernels, for the functions every one of them has.
#[cfg(target_arch = "aarch64")]
use neon as simd;
#[cfg(not(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
)))]
use scalar as simd;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use wasm as simd;
#[cfg(target_arch = "x86_64")]
use x86 as simd;

#[cfg(target_arch = "aarch64")]
pub use neon::{
    dot2_q4_pairs, dot2_q8_pairs, dot4_q4_pairs, dot4_q8_pairs, dot_bf16, dot_f32, dot_q4,
    dot_q4_x4, dot_q8, dot_q8_x4, has_i8mm,
};

/// Four 8-bit weight rows against `P` token pairs: rows 0 and 1, then rows
/// 2 and 3, each as [`dot2_q8_pairs`] gives them. (Only aarch64 has a
/// kernel for it: elsewhere there are no matrix instructions to pair up.)
#[cfg(not(target_arch = "aarch64"))]
pub fn dot4_q8_pairs<const P: usize>(
    ws: [&[f32]; 4],
    wq: [&[i8]; 4],
    ps: [&[f32]; P],
    pq: [&[i8]; P],
) -> [[[f32; 4]; P]; 2] {
    [
        dot2_q8_pairs([ws[0], ws[1]], [wq[0], wq[1]], ps, pq),
        dot2_q8_pairs([ws[2], ws[3]], [wq[2], wq[3]], ps, pq),
    ]
}

/// [`dot4_q8_pairs`] for 4-bit rows.
#[cfg(not(target_arch = "aarch64"))]
pub fn dot4_q4_pairs<const P: usize>(
    ws: [&[f32]; 4],
    wq: [&[u8]; 4],
    ps: [&[f32]; P],
    pq: [&[i8]; P],
) -> [[[f32; 4]; P]; 2] {
    [
        dot2_q4_pairs([ws[0], ws[1]], [wq[0], wq[1]], ps, pq),
        dot2_q4_pairs([ws[2], ws[3]], [wq[2], wq[3]], ps, pq),
    ]
}
#[cfg(not(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
)))]
pub use scalar::{
    dot2_q4_pairs, dot2_q8_pairs, dot_bf16, dot_f32, dot_q4, dot_q4_x4, dot_q8, dot_q8_x4, has_i8mm,
};
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub use scalar::{dot2_q4_pairs, dot2_q8_pairs, has_i8mm};
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub use wasm::{dot_bf16, dot_f32, dot_q4, dot_q4_x4, dot_q8, dot_q8_x4};

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[path = "kernels_wasm.rs"]
mod wasm;

#[cfg(target_arch = "x86_64")]
pub use scalar::{dot2_q4_pairs, dot2_q8_pairs, has_i8mm};
#[cfg(target_arch = "x86_64")]
pub use x86::{dot_bf16, dot_f32, dot_q4, dot_q4_x4, dot_q8, dot_q8_x4};

#[cfg(target_arch = "x86_64")]
#[path = "kernels_x86.rs"]
mod x86;

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{f16_finite, Scales, QK, SUPER, SUPER_WORDS};
    use std::arch::aarch64::*;
    use std::sync::atomic::{AtomicU8, Ordering::Relaxed};

    /// The first multiple of 8 of [`super::unpack_f16`], returning how
    /// many: [`f16_finite`] on 8 lanes.
    pub fn unpack_f16_scales(h: &[u16], out: &mut [f32]) -> usize {
        let n = h.len().min(out.len()) / 8 * 8;
        // SAFETY: i + 8 <= n is in bounds of both; NEON is always present.
        unsafe {
            let bias = vdupq_n_f32(f32::from_bits(0x7780_0000));
            let (magnitude, sign) = (vdupq_n_u32(0x7FFF), vdupq_n_u32(0x8000));
            for i in (0..n).step_by(8) {
                let v = vld1q_u16(h.as_ptr().add(i));
                for (j, half) in [vget_low_u16(v), vget_high_u16(v)].into_iter().enumerate() {
                    let w = vmovl_u16(half);
                    let bits = vorrq_u32(
                        vshlq_n_u32::<13>(vandq_u32(w, magnitude)),
                        vshlq_n_u32::<16>(vandq_u32(w, sign)),
                    );
                    let f = vmulq_f32(vreinterpretq_f32_u32(bits), bias);
                    vst1q_f32(out.as_mut_ptr().add(i + 4 * j), f);
                }
            }
        }
        n
    }

    /// [`super::unpack_q4`], four blocks per instruction: each word of
    /// fields is broadcast and shifted by a different amount per lane.
    pub fn unpack_q4_scales(packed: &[u32], d: &mut [f32], c: &mut [f32]) {
        let nb = d.len();
        assert!(c.len() == nb && packed.len() >= nb.div_ceil(SUPER) * SUPER_WORDS);
        let (mut ds, mut cs) = ([0f32; SUPER], [0f32; SUPER]);
        for k in 0..nb.div_ceil(SUPER) {
            let n = (nb - k * SUPER).min(SUPER);
            // SAFETY: the bounds are checked above (a partial last
            // super-block goes through a buffer); NEON is always present on
            // aarch64.
            unsafe {
                let w = packed.as_ptr().add(k * SUPER_WORDS);
                let step_d = f16_finite(*w as u16);
                let step_c = f16_finite((*w >> 16) as u16);
                let s = vmovl_u8(vld1_u8(w.add(1) as *const u8));
                let o = vmovl_s8(vld1_s8(w.add(3) as *const i8));
                let (pd, pc) = if n == SUPER {
                    (d.as_mut_ptr().add(k * SUPER), c.as_mut_ptr().add(k * SUPER))
                } else {
                    (ds.as_mut_ptr(), cs.as_mut_ptr())
                };
                let f = |v: uint16x4_t| vcvtq_f32_u32(vmovl_u16(v));
                let g = |v: int16x4_t| vcvtq_f32_s32(vmovl_s16(v));
                vst1q_f32(pd, vmulq_n_f32(f(vget_low_u16(s)), step_d));
                vst1q_f32(pd.add(4), vmulq_n_f32(f(vget_high_u16(s)), step_d));
                vst1q_f32(pc, vmulq_n_f32(g(vget_low_s16(o)), step_c));
                vst1q_f32(pc.add(4), vmulq_n_f32(g(vget_high_s16(o)), step_c));
            }
            if n < SUPER {
                d[k * SUPER..].copy_from_slice(&ds[..n]);
                c[k * SUPER..].copy_from_slice(&cs[..n]);
            }
        }
    }

    /// A CPU feature check, done once.
    fn cached(flag: &AtomicU8, detect: fn() -> bool) -> bool {
        match flag.load(Relaxed) {
            1 => false,
            2 => true,
            _ => {
                let has = detect();
                flag.store(if has { 2 } else { 1 }, Relaxed);
                has
            }
        }
    }

    /// Whether this CPU has SDOT (ARMv8.4 dot product; every Apple Silicon
    /// chip does).
    fn dotprod() -> bool {
        static HAS: AtomicU8 = AtomicU8::new(0);
        cached(&HAS, || std::arch::is_aarch64_feature_detected!("dotprod"))
    }

    /// Whether this CPU has I8MM, the int8 matrix multiply instructions
    /// (Apple M2 and later).
    pub fn has_i8mm() -> bool {
        static HAS: AtomicU8 = AtomicU8::new(0);
        cached(&HAS, || std::arch::is_aarch64_feature_detected!("i8mm"))
    }

    /// `acc += dot(a, b)` four bytes at a time: 16 int8 multiply-adds into
    /// four int32 lanes in one instruction, where plain NEON needs a widening
    /// multiply and a pairwise add per 8. Rust's intrinsic for it isn't
    /// stable yet, so it is a line of inline assembly.
    #[inline]
    #[target_feature(enable = "dotprod")]
    unsafe fn sdot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
        let mut acc = acc;
        std::arch::asm!(
            "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
            acc = inout(vreg) acc,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack, preserves_flags)
        );
        acc
    }

    /// `acc += a · bᵀ` for a 2x8 matrix `a` and an 8x2 matrix `b` (each a
    /// 16-byte vector, row after row): the 2x2 int32 result is
    /// `[a0·b0, a0·b1, a1·b0, a1·b1]`. 32 multiply-adds in one instruction,
    /// twice SDOT's 16.
    #[inline]
    #[target_feature(enable = "i8mm")]
    unsafe fn smmla(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
        let mut acc = acc;
        std::arch::asm!(
            "smmla {acc:v}.4s, {a:v}.16b, {b:v}.16b",
            acc = inout(vreg) acc,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack, preserves_flags)
        );
        acc
    }

    /// The first (`HI = false`) or second 8 bytes of `a` followed by the same
    /// of `b`.
    #[inline(always)]
    unsafe fn zip64<const HI: bool>(a: int8x16_t, b: int8x16_t) -> int8x16_t {
        let (a, b) = (vreinterpretq_s64_s8(a), vreinterpretq_s64_s8(b));
        vreinterpretq_s8_s64(if HI {
            vzip2q_s64(a, b)
        } else {
            vzip1q_s64(a, b)
        })
    }

    /// Two weight rows against `P` token pairs laid out by `pair_tokens`.
    /// Per block, the rows' weights become four 2x8 matrices, each multiplied
    /// by a pair's matching 8x2 slice: one `smmla` covers 2 rows x 2 tokens
    /// x 8 values.
    #[target_feature(enable = "i8mm")]
    unsafe fn dot2_pairs<const P: usize, const Q4: bool>(
        ws: [&[f32]; 2],
        wq: [*const u8; 2],
        ps: [&[f32]; P],
        pq: [&[i8]; P],
    ) -> [[f32; 4]; P] {
        let nb = ws[0].len();
        assert!(ws[1].len() >= nb);
        for p in 0..P {
            assert!(pq[p].len() >= nb * 2 * QK && ps[p].len() >= nb * 2);
        }
        let mut acc = [vdupq_n_f32(0.0); P];
        for b in 0..nb {
            let [r0, r1] = wq.map(|w| {
                if Q4 {
                    unpack_q4(w.add(b * QK / 2))
                } else {
                    let p = w.add(b * QK) as *const i8;
                    (vld1q_s8(p), vld1q_s8(p.add(16)))
                }
            });
            let w = [
                zip64::<false>(r0.0, r1.0),
                zip64::<true>(r0.0, r1.0),
                zip64::<false>(r0.1, r1.1),
                zip64::<true>(r0.1, r1.1),
            ];
            let wd = vcombine_f32(
                vdup_n_f32(*ws[0].get_unchecked(b)),
                vdup_n_f32(*ws[1].get_unchecked(b)),
            );
            for p in 0..P {
                let q = pq[p].as_ptr().add(b * 2 * QK);
                let mut s = vdupq_n_s32(0);
                for (k, &w) in w.iter().enumerate() {
                    s = smmla(s, w, vld1q_s8(q.add(16 * k)));
                }
                let xd = vld1_f32(ps[p].as_ptr().add(2 * b));
                acc[p] = vfmaq_f32(
                    acc[p],
                    vcvtq_f32_s32(s),
                    vmulq_f32(wd, vcombine_f32(xd, xd)),
                );
            }
        }
        acc.map(|a| {
            let mut out = [0.0; 4];
            vst1q_f32(out.as_mut_ptr(), a);
            out
        })
    }

    /// [`dot2_pairs`] for two pairs of rows at once: each token pair's
    /// activations are loaded once for all four rows, which halves the
    /// loads (and scale products) per matrix instruction. Every output is
    /// computed exactly as `dot2_pairs` computes it.
    #[target_feature(enable = "i8mm")]
    unsafe fn dot4_pairs<const P: usize, const Q4: bool>(
        ws: [&[f32]; 4],
        wq: [*const u8; 4],
        ps: [&[f32]; P],
        pq: [&[i8]; P],
    ) -> [[[f32; 4]; P]; 2] {
        let nb = ws[0].len();
        assert!(ws.iter().all(|w| w.len() >= nb));
        for p in 0..P {
            assert!(pq[p].len() >= nb * 2 * QK && ps[p].len() >= nb * 2);
        }
        let mut acc = [[vdupq_n_f32(0.0); P]; 2];
        for b in 0..nb {
            let r = wq.map(|w| {
                if Q4 {
                    unpack_q4(w.add(b * QK / 2))
                } else {
                    let p = w.add(b * QK) as *const i8;
                    (vld1q_s8(p), vld1q_s8(p.add(16)))
                }
            });
            let w = [0, 2].map(|h| {
                let (a, c) = (r[h], r[h + 1]);
                [
                    zip64::<false>(a.0, c.0),
                    zip64::<true>(a.0, c.0),
                    zip64::<false>(a.1, c.1),
                    zip64::<true>(a.1, c.1),
                ]
            });
            let wd = [0, 2].map(|h| {
                vcombine_f32(
                    vdup_n_f32(*ws[h].get_unchecked(b)),
                    vdup_n_f32(*ws[h + 1].get_unchecked(b)),
                )
            });
            for p in 0..P {
                let q = pq[p].as_ptr().add(b * 2 * QK);
                let x = [0, 1, 2, 3].map(|k| vld1q_s8(q.add(16 * k)));
                let xd = vld1_f32(ps[p].as_ptr().add(2 * b));
                let xd = vcombine_f32(xd, xd);
                for h in 0..2 {
                    let mut s = vdupq_n_s32(0);
                    for (&w, &x) in w[h].iter().zip(&x) {
                        s = smmla(s, w, x);
                    }
                    acc[h][p] = vfmaq_f32(acc[h][p], vcvtq_f32_s32(s), vmulq_f32(wd[h], xd));
                }
            }
        }
        acc.map(|a| {
            a.map(|a| {
                let mut out = [0.0; 4];
                vst1q_f32(out.as_mut_ptr(), a);
                out
            })
        })
    }

    /// Four 8-bit weight rows against `P` token pairs: rows 0 and 1, then
    /// rows 2 and 3, as [`dot2_q8_pairs`] gives them. Needs [`has_i8mm`].
    pub fn dot4_q8_pairs<const P: usize>(
        ws: [&[f32]; 4],
        wq: [&[i8]; 4],
        ps: [&[f32]; P],
        pq: [&[i8]; P],
    ) -> [[[f32; 4]; P]; 2] {
        assert!(has_i8mm() && wq.iter().all(|w| w.len() >= ws[0].len() * QK));
        unsafe { dot4_pairs::<P, false>(ws, wq.map(|w| w.as_ptr() as *const u8), ps, pq) }
    }

    /// [`dot4_q8_pairs`] for 4-bit rows.
    pub fn dot4_q4_pairs<const P: usize>(
        ws: [&[f32]; 4],
        wq: [&[u8]; 4],
        ps: [&[f32]; P],
        pq: [&[i8]; P],
    ) -> [[[f32; 4]; P]; 2] {
        assert!(has_i8mm() && wq.iter().all(|w| w.len() >= ws[0].len() * QK / 2));
        unsafe { dot4_pairs::<P, true>(ws, wq.map(|w| w.as_ptr()), ps, pq) }
    }

    /// Two 8-bit weight rows against `P` token pairs: for each pair
    /// `[row0·t0, row0·t1, row1·t0, row1·t1]`. Needs [`has_i8mm`].
    pub fn dot2_q8_pairs<const P: usize>(
        ws: [&[f32]; 2],
        wq: [&[i8]; 2],
        ps: [&[f32]; P],
        pq: [&[i8]; P],
    ) -> [[f32; 4]; P] {
        assert!(has_i8mm() && wq.iter().all(|w| w.len() >= ws[0].len() * QK));
        unsafe { dot2_pairs::<P, false>(ws, wq.map(|w| w.as_ptr() as *const u8), ps, pq) }
    }

    /// Two 4-bit weight rows against `P` token pairs, as [`dot2_q8_pairs`].
    pub fn dot2_q4_pairs<const P: usize>(
        ws: [&[f32]; 2],
        wq: [&[u8]; 2],
        ps: [&[f32]; P],
        pq: [&[i8]; P],
    ) -> [[f32; 4]; P] {
        assert!(has_i8mm() && wq.iter().all(|w| w.len() >= ws[0].len() * QK / 2));
        unsafe { dot2_pairs::<P, true>(ws, wq.map(|w| w.as_ptr()), ps, pq) }
    }

    /// The 32 weights of a 4-bit block as two vectors of signed bytes.
    #[inline(always)]
    unsafe fn unpack_q4(p: *const u8) -> (int8x16_t, int8x16_t) {
        let raw = vld1q_u8(p);
        let eight = vdupq_n_s8(8);
        (
            vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, vdupq_n_u8(0x0F))), eight),
            vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(raw)), eight),
        )
    }

    /// One weight row against `N` tokens' activations: each weight block is
    /// loaded (and for 4-bit, unpacked) once and used for every token, which
    /// is what makes batched prefill faster than N separate dots.
    ///
    /// Each block's exact integer sum is scaled into one running total per
    /// token with a fused multiply-add, the same arithmetic `dot2_pairs`
    /// does per lane, so a token's result doesn't depend on which kernel or
    /// which batch it went through.
    #[target_feature(enable = "dotprod")]
    unsafe fn dot_rows_sdot<const N: usize, const Q4: bool, S: Scales>(
        ws: S,
        wq: *const u8,
        xs: [&[f32]; N],
        xq: [&[i8]; N],
    ) -> [f32; N] {
        let nb = ws.blocks();
        for j in 0..N {
            assert!(xq[j].len() >= nb * QK && xs[j].len() >= nb);
        }
        let mut acc = [0f32; N];
        let mut block = |b: usize, wd: f32| {
            let (w0, w1) = if Q4 {
                unpack_q4(wq.add(b * QK / 2))
            } else {
                let p = wq.add(b * QK) as *const i8;
                (vld1q_s8(p), vld1q_s8(p.add(16)))
            };
            for j in 0..N {
                let px = xq[j].as_ptr().add(b * QK);
                let s = sdot(
                    sdot(vdupq_n_s32(0), w0, vld1q_s8(px)),
                    w1,
                    vld1q_s8(px.add(16)),
                );
                acc[j] = (vaddvq_s32(s) as f32).mul_add(wd * *xs[j].get_unchecked(b), acc[j]);
            }
        };
        // Scales eight blocks at a time, so compact ones are converted with
        // vector instructions; then any blocks left over one by one.
        for k in 0..nb / SUPER {
            let wd = ws.eight(k);
            for (j, &wd) in wd.iter().enumerate() {
                block(k * SUPER + j, wd);
            }
        }
        for b in nb / SUPER * SUPER..nb {
            block(b, ws.get(b));
        }
        acc
    }

    pub fn dot_q8(ws: &[f32], wq: &[i8], xs: &[f32], xq: &[i8]) -> f32 {
        dot_q8_with(ws, wq, xs, xq)
    }

    pub fn dot_q8_with<S: Scales>(ws: S, wq: &[i8], xs: &[f32], xq: &[i8]) -> f32 {
        if dotprod() {
            assert!(wq.len() >= ws.blocks() * QK);
            unsafe { dot_rows_sdot::<1, false, S>(ws, wq.as_ptr() as *const u8, [xs], [xq])[0] }
        } else {
            super::expanded(ws, |d| dot_q8_neon(d, wq, xs, xq))
        }
    }

    pub fn dot_q4_with<S: Scales>(ws: S, wq: &[u8], xs: &[f32], xq: &[i8]) -> f32 {
        if dotprod() {
            assert!(wq.len() >= ws.blocks() * QK / 2);
            unsafe { dot_rows_sdot::<1, true, S>(ws, wq.as_ptr(), [xs], [xq])[0] }
        } else {
            super::expanded(ws, |d| dot_q4_neon(d, wq, xs, xq))
        }
    }

    pub fn scores_64<'a>(
        q: &[f32],
        keys: impl Iterator<Item = &'a [f32]>,
        scale: f32,
        out: &mut Vec<f32>,
    ) {
        assert!(q.len() == 64);
        // SAFETY: the query and every key are checked to be 64 long.
        unsafe {
            let qv: [float32x4_t; 16] = std::array::from_fn(|i| vld1q_f32(q.as_ptr().add(4 * i)));
            for k in keys {
                assert!(k.len() >= 64);
                let p = k.as_ptr();
                let mut a = [vdupq_n_f32(0.0); 4];
                for i in 0..16 {
                    a[i % 4] = vfmaq_f32(a[i % 4], qv[i], vld1q_f32(p.add(4 * i)));
                }
                let s = vaddq_f32(vaddq_f32(a[0], a[1]), vaddq_f32(a[2], a[3]));
                out.push(vaddvq_f32(s) * scale);
            }
        }
    }

    pub fn weighted_sum_64<'a>(w: &[f32], rows: impl Iterator<Item = &'a [f32]>, out: &mut [f32]) {
        assert!(out.len() == 64);
        // SAFETY: the output and every row are checked to be 64 long.
        unsafe {
            let mut o: [float32x4_t; 16] =
                std::array::from_fn(|i| vld1q_f32(out.as_ptr().add(4 * i)));
            for (&w, row) in w.iter().zip(rows) {
                assert!(row.len() >= 64);
                let p = row.as_ptr();
                for (i, o) in o.iter_mut().enumerate() {
                    *o = vfmaq_n_f32(*o, vld1q_f32(p.add(4 * i)), w);
                }
            }
            for (i, o) in o.iter().enumerate() {
                vst1q_f32(out.as_mut_ptr().add(4 * i), *o);
            }
        }
    }

    pub fn exp_inplace(xs: &mut [f32]) {
        // SAFETY: whole groups of four are in bounds; the rest go through
        // a buffer. NEON is always present.
        unsafe fn exp4(x: float32x4_t) -> float32x4_t {
            let x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-87.336_55)), vdupq_n_f32(88.7));
            let n = vrndnq_f32(vmulq_n_f32(x, std::f32::consts::LOG2_E));
            let r = vfmsq_f32(
                vfmsq_f32(x, n, vdupq_n_f32(0.693_359_4)),
                n,
                vdupq_n_f32(-2.121_944_4e-4),
            );
            let mut y = vdupq_n_f32(super::EXP_P[0]);
            for &c in &super::EXP_P[1..] {
                y = vfmaq_f32(vdupq_n_f32(c), y, r);
            }
            let y = vaddq_f32(vfmaq_f32(r, y, vmulq_f32(r, r)), vdupq_n_f32(1.0));
            let e = vshlq_n_s32::<23>(vcvtq_s32_f32(n));
            vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(y), e))
        }
        let whole = xs.len() / 4 * 4;
        unsafe {
            for i in (0..whole).step_by(4) {
                let p = xs.as_mut_ptr().add(i);
                vst1q_f32(p, exp4(vld1q_f32(p)));
            }
            if whole < xs.len() {
                let mut t = [0f32; 4];
                t[..xs.len() - whole].copy_from_slice(&xs[whole..]);
                vst1q_f32(t.as_mut_ptr(), exp4(vld1q_f32(t.as_ptr())));
                let n = xs.len() - whole;
                xs[whole..].copy_from_slice(&t[..n]);
            }
        }
    }

    /// Eight f16 scales as [`f16_finite`] gives them.
    #[inline(always)]
    pub unsafe fn f16_eight(h: *const u16) -> [f32; SUPER] {
        let v = vld1q_u16(h);
        let (magnitude, sign) = (vdupq_n_u32(0x7FFF), vdupq_n_u32(0x8000));
        let bias = vdupq_n_f32(f32::from_bits(0x7780_0000));
        let f = |half: uint16x4_t| {
            let w = vmovl_u16(half);
            let bits = vorrq_u32(
                vshlq_n_u32::<13>(vandq_u32(w, magnitude)),
                vshlq_n_u32::<16>(vandq_u32(w, sign)),
            );
            vmulq_f32(vreinterpretq_f32_u32(bits), bias)
        };
        std::mem::transmute::<[float32x4_t; 2], [f32; SUPER]>([
            f(vget_low_u16(v)),
            f(vget_high_u16(v)),
        ])
    }

    /// A super-block's eight 4-bit scales as [`unpack_q4_scales`] gives them.
    #[inline(always)]
    pub unsafe fn packed_eight(w: *const u32) -> [f32; SUPER] {
        let step = f16_finite(*w as u16);
        let s = vmovl_u8(vld1_u8(w.add(1) as *const u8));
        let f = |v: uint16x4_t| vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(v)), step);
        std::mem::transmute::<[float32x4_t; 2], [f32; SUPER]>([
            f(vget_low_u16(s)),
            f(vget_high_u16(s)),
        ])
    }

    /// [`super::dot_offsets`] and [`super::dot_offsets_expanded`]: blocks
    /// 0..4 and 4..8 of each super-block in two vector lanes, multiply-added
    /// in registers, then summed. Both build the same offsets (the packed
    /// version as [`unpack_q4_scales`] does) and add them in the same order.
    /// The callers checked `sums` is whole super-blocks that `offsets`
    /// covers.
    #[inline(always)]
    unsafe fn offsets_lanes(
        sums: &[f32],
        offsets: impl Fn(usize) -> (float32x4_t, float32x4_t),
    ) -> f32 {
        let (mut lo, mut hi) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
        for k in 0..sums.len() / SUPER {
            let (c0, c1) = offsets(k);
            let p = sums.as_ptr().add(k * SUPER);
            lo = vfmaq_f32(lo, c0, vld1q_f32(p));
            hi = vfmaq_f32(hi, c1, vld1q_f32(p.add(4)));
        }
        vaddvq_f32(vaddq_f32(lo, hi))
    }

    pub fn dot_offsets(words: &[u32], sums: &[f32]) -> f32 {
        // SAFETY: see `offsets_lanes`; NEON is always present.
        unsafe {
            offsets_lanes(sums, |k| {
                let w = words.as_ptr().add(k * SUPER_WORDS);
                let step = f16_finite((*w >> 16) as u16);
                let o = vmovl_s8(vld1_s8(w.add(3) as *const i8));
                (
                    vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(o))), step),
                    vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(o))), step),
                )
            })
        }
    }

    pub fn dot_offsets_expanded(c: &[f32], sums: &[f32]) -> f32 {
        // SAFETY: see `offsets_lanes`.
        unsafe {
            offsets_lanes(sums, |k| {
                let p = c.as_ptr().add(k * SUPER);
                (vld1q_f32(p), vld1q_f32(p.add(4)))
            })
        }
    }

    pub fn dot_q4(ws: &[f32], wq: &[u8], xs: &[f32], xq: &[i8]) -> f32 {
        dot_q4_with(ws, wq, xs, xq)
    }

    /// Four tokens against one 8-bit weight row.
    pub fn dot_q8_x4(ws: &[f32], wq: &[i8], xs: [&[f32]; 4], xq: [&[i8]; 4]) -> [f32; 4] {
        if dotprod() {
            assert!(wq.len() >= ws.len() * QK);
            unsafe { dot_rows_sdot::<4, false, &[f32]>(ws, wq.as_ptr() as *const u8, xs, xq) }
        } else {
            [0, 1, 2, 3].map(|j| dot_q8_neon(ws, wq, xs[j], xq[j]))
        }
    }

    /// Four tokens against one 4-bit weight row.
    pub fn dot_q4_x4(ws: &[f32], wq: &[u8], xs: [&[f32]; 4], xq: [&[i8]; 4]) -> [f32; 4] {
        if dotprod() {
            assert!(wq.len() >= ws.len() * QK / 2);
            unsafe { dot_rows_sdot::<4, true, &[f32]>(ws, wq.as_ptr(), xs, xq) }
        } else {
            [0, 1, 2, 3].map(|j| dot_q4_neon(ws, wq, xs[j], xq[j]))
        }
    }

    pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut i = 0;
        let mut sum = unsafe {
            let mut acc = [vdupq_n_f32(0.0); 4];
            while i + 16 <= n {
                for (k, acc) in acc.iter_mut().enumerate() {
                    *acc = vfmaq_f32(
                        *acc,
                        vld1q_f32(pa.add(i + 4 * k)),
                        vld1q_f32(pb.add(i + 4 * k)),
                    );
                }
                i += 16;
            }
            vaddvq_f32(vaddq_f32(
                vaddq_f32(acc[0], acc[1]),
                vaddq_f32(acc[2], acc[3]),
            ))
        };
        while i < n {
            sum += a[i] * b[i];
            i += 1;
        }
        sum
    }

    /// bf16 is the top half of an f32, so widening is a zip with zeros.
    pub fn dot_bf16(w: &[u16], x: &[f32]) -> f32 {
        let n = w.len().min(x.len());
        let (pw, px) = (w.as_ptr(), x.as_ptr());
        let mut i = 0;
        let mut sum = unsafe {
            let zero = vdupq_n_u16(0);
            let mut acc = [vdupq_n_f32(0.0); 4];
            while i + 16 <= n {
                let w0 = vld1q_u16(pw.add(i));
                let w1 = vld1q_u16(pw.add(i + 8));
                let f = [
                    vreinterpretq_f32_u16(vzip1q_u16(zero, w0)),
                    vreinterpretq_f32_u16(vzip2q_u16(zero, w0)),
                    vreinterpretq_f32_u16(vzip1q_u16(zero, w1)),
                    vreinterpretq_f32_u16(vzip2q_u16(zero, w1)),
                ];
                for k in 0..4 {
                    acc[k] = vfmaq_f32(acc[k], f[k], vld1q_f32(px.add(i + 4 * k)));
                }
                i += 16;
            }
            vaddvq_f32(vaddq_f32(
                vaddq_f32(acc[0], acc[1]),
                vaddq_f32(acc[2], acc[3]),
            ))
        };
        while i < n {
            sum += super::bf16_to_f32(w[i]) * x[i];
            i += 1;
        }
        sum
    }

    /// Sum of one block's 32 int8 products, as four i32 lanes.
    #[inline(always)]
    unsafe fn block_dot(w0: int8x16_t, w1: int8x16_t, x0: int8x16_t, x1: int8x16_t) -> int32x4_t {
        let mut s = vpaddlq_s16(vmull_s8(vget_low_s8(w0), vget_low_s8(x0)));
        s = vpadalq_s16(s, vmull_high_s8(w0, x0));
        s = vpadalq_s16(s, vmull_s8(vget_low_s8(w1), vget_low_s8(x1)));
        vpadalq_s16(s, vmull_high_s8(w1, x1))
    }

    fn dot_q8_neon(ws: &[f32], wq: &[i8], xs: &[f32], xq: &[i8]) -> f32 {
        let nb = ws.len();
        assert!(wq.len() >= nb * QK && xq.len() >= nb * QK && xs.len() >= nb);
        unsafe {
            let mut acc = vdupq_n_f32(0.0);
            for b in 0..nb {
                let (pw, px) = (wq.as_ptr().add(b * QK), xq.as_ptr().add(b * QK));
                let s = block_dot(
                    vld1q_s8(pw),
                    vld1q_s8(pw.add(16)),
                    vld1q_s8(px),
                    vld1q_s8(px.add(16)),
                );
                let d = *ws.get_unchecked(b) * *xs.get_unchecked(b);
                acc = vfmaq_f32(acc, vcvtq_f32_s32(s), vdupq_n_f32(d));
            }
            vaddvq_f32(acc)
        }
    }

    fn dot_q4_neon(ws: &[f32], wq: &[u8], xs: &[f32], xq: &[i8]) -> f32 {
        let nb = ws.len();
        assert!(wq.len() >= nb * QK / 2 && xq.len() >= nb * QK && xs.len() >= nb);
        unsafe {
            let mask = vdupq_n_u8(0x0F);
            let eight = vdupq_n_s8(8);
            let mut acc = vdupq_n_f32(0.0);
            for b in 0..nb {
                let raw = vld1q_u8(wq.as_ptr().add(b * QK / 2));
                let w0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask)), eight);
                let w1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(raw)), eight);
                let px = xq.as_ptr().add(b * QK);
                let s = block_dot(w0, w1, vld1q_s8(px), vld1q_s8(px.add(16)));
                let d = *ws.get_unchecked(b) * *xs.get_unchecked(b);
                acc = vfmaq_f32(acc, vcvtq_f32_s32(s), vdupq_n_f32(d));
            }
            vaddvq_f32(acc)
        }
    }
}

#[cfg_attr(
    any(target_arch = "aarch64", target_feature = "simd128"),
    allow(dead_code)
)]
mod scalar {
    use super::{bf16_to_f32, Scales, QK};

    pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    pub fn dot_bf16(w: &[u16], x: &[f32]) -> f32 {
        w.iter().zip(x).map(|(&w, &x)| bf16_to_f32(w) * x).sum()
    }

    pub fn dot_q8(ws: &[f32], wq: &[i8], xs: &[f32], xq: &[i8]) -> f32 {
        dot_q8_with(ws, wq, xs, xq)
    }

    pub fn dot_q8_with(ws: impl Scales, wq: &[i8], xs: &[f32], xq: &[i8]) -> f32 {
        (0..ws.blocks())
            .map(|b| {
                let s: i32 = (0..QK)
                    .map(|j| wq[b * QK + j] as i32 * xq[b * QK + j] as i32)
                    .sum();
                // SAFETY: b < blocks.
                s as f32 * unsafe { ws.get(b) } * xs[b]
            })
            .sum()
    }

    pub fn dot_q4(ws: &[f32], wq: &[u8], xs: &[f32], xq: &[i8]) -> f32 {
        dot_q4_with(ws, wq, xs, xq)
    }

    pub fn dot_q4_with(ws: impl Scales, wq: &[u8], xs: &[f32], xq: &[i8]) -> f32 {
        (0..ws.blocks())
            .map(|b| {
                let mut s = 0i32;
                for j in 0..QK / 2 {
                    let byte = wq[b * QK / 2 + j];
                    s += ((byte & 0xF) as i32 - 8) * xq[b * QK + j] as i32;
                    s += ((byte >> 4) as i32 - 8) * xq[b * QK + j + QK / 2] as i32;
                }
                // SAFETY: b < blocks.
                s as f32 * unsafe { ws.get(b) } * xs[b]
            })
            .sum()
    }

    pub fn dot_q8_x4(ws: &[f32], wq: &[i8], xs: [&[f32]; 4], xq: [&[i8]; 4]) -> [f32; 4] {
        [0, 1, 2, 3].map(|j| dot_q8(ws, wq, xs[j], xq[j]))
    }

    pub fn dot_q4_x4(ws: &[f32], wq: &[u8], xs: [&[f32]; 4], xq: [&[i8]; 4]) -> [f32; 4] {
        [0, 1, 2, 3].map(|j| dot_q4(ws, wq, xs[j], xq[j]))
    }

    pub fn has_i8mm() -> bool {
        false
    }

    /// Reads one token back out of a pair laid out by `pair_tokens`.
    fn unpair(ps: &[f32], pq: &[i8], second: bool) -> (Vec<f32>, Vec<i8>) {
        let o = second as usize;
        let s = ps.iter().skip(o).step_by(2).copied().collect();
        let q = pq
            .chunks_exact(16)
            .flat_map(|c| c[o * 8..o * 8 + 8].iter().copied())
            .collect();
        (s, q)
    }

    fn pairs<const P: usize>(
        ps: [&[f32]; P],
        pq: [&[i8]; P],
        dot: impl Fn(usize, &[f32], &[i8]) -> f32,
    ) -> [[f32; 4]; P] {
        std::array::from_fn(|p| {
            let (s0, q0) = unpair(ps[p], pq[p], false);
            let (s1, q1) = unpair(ps[p], pq[p], true);
            [
                dot(0, &s0, &q0),
                dot(0, &s1, &q1),
                dot(1, &s0, &q0),
                dot(1, &s1, &q1),
            ]
        })
    }

    pub fn dot2_q8_pairs<const P: usize>(
        ws: [&[f32]; 2],
        wq: [&[i8]; 2],
        ps: [&[f32]; P],
        pq: [&[i8]; P],
    ) -> [[f32; 4]; P] {
        pairs(ps, pq, |r, s, q| dot_q8(ws[r], wq[r], s, q))
    }

    pub fn dot2_q4_pairs<const P: usize>(
        ws: [&[f32]; 2],
        wq: [&[u8]; 2],
        ps: [&[f32]; P],
        pq: [&[i8]; P],
    ) -> [[f32; 4]; P] {
        pairs(ps, pq, |r, s, q| dot_q4(ws[r], wq[r], s, q))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn simd_matches_scalar() {
        let (a, b) = (data(896, 1), data(896, 2));
        let exact: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!((dot_f32(&a, &b) - exact).abs() < 1e-3);

        let wb: Vec<u16> = a.iter().map(|&v| f32_to_bf16(v)).collect();
        assert!((dot_bf16(&wb, &b) - scalar::dot_bf16(&wb, &b)).abs() < 1e-3);

        let nb = a.len() / QK;
        let (mut ws, mut wq, mut xs, mut xq) = (
            vec![0.0; nb],
            vec![0i8; a.len()],
            vec![0.0; nb],
            vec![0i8; a.len()],
        );
        quantize_q8(&a, &mut ws, &mut wq);
        quantize_q8(&b, &mut xs, &mut xq);
        let q8 = dot_q8(&ws, &wq, &xs, &xq);
        assert!((q8 - scalar::dot_q8(&ws, &wq, &xs, &xq)).abs() < 1e-3);
        assert!(
            (q8 - exact).abs() < 0.05 * exact.abs().max(1.0),
            "q8 {q8} vs {exact}"
        );

        let mut w4 = vec![0u8; a.len() / 2];
        quantize_q4(&a, &mut ws, &mut w4);
        let q4 = dot_q4(&ws, &w4, &xs, &xq);
        assert!((q4 - scalar::dot_q4(&ws, &w4, &xs, &xq)).abs() < 1e-3);
    }

    #[test]
    fn four_token_dots_match_single_dots() {
        let cols = 896;
        let nb = cols / QK;
        let w = data(cols, 21);
        let (mut ws, mut wq8, mut wq4) = (vec![0.0; nb], vec![0i8; cols], vec![0u8; cols / 2]);
        quantize_q8(&w, &mut ws, &mut wq8);
        let mut ws4 = vec![0.0; nb];
        quantize_q4(&w, &mut ws4, &mut wq4);
        let xs_all: Vec<(Vec<f32>, Vec<i8>)> = (0..4)
            .map(|j| {
                let x = data(cols, 30 + j);
                let (mut s, mut q) = (vec![0.0; nb], vec![0i8; cols]);
                quantize_q8(&x, &mut s, &mut q);
                (s, q)
            })
            .collect();
        let xs: [&[f32]; 4] = [0, 1, 2, 3].map(|j| xs_all[j].0.as_slice());
        let xq: [&[i8]; 4] = [0, 1, 2, 3].map(|j| xs_all[j].1.as_slice());
        let q8 = dot_q8_x4(&ws, &wq8, xs, xq);
        let q4 = dot_q4_x4(&ws4, &wq4, xs, xq);
        for j in 0..4 {
            assert_eq!(q8[j], dot_q8(&ws, &wq8, xs[j], xq[j]));
            assert_eq!(q4[j], dot_q4(&ws4, &wq4, xs[j], xq[j]));
            assert!((q8[j] - scalar::dot_q8(&ws, &wq8, xs[j], xq[j])).abs() < 1e-3);
            assert!((q4[j] - scalar::dot_q4(&ws4, &wq4, xs[j], xq[j])).abs() < 1e-3);
        }
    }

    #[test]
    fn pair_dots_match_single_dots() {
        let (cols, t) = (896, 5);
        let nb = cols / QK;
        type Row = (Vec<f32>, Vec<i8>, Vec<f32>, Vec<u8>);
        let rows: Vec<Row> = (0..2)
            .map(|r| {
                let w = data(cols, 40 + r);
                let (mut s8, mut q8) = (vec![0.0; nb], vec![0i8; cols]);
                let (mut s4, mut q4) = (vec![0.0; nb], vec![0u8; cols / 2]);
                quantize_q8(&w, &mut s8, &mut q8);
                quantize_q4(&w, &mut s4, &mut q4);
                (s8, q8, s4, q4)
            })
            .collect();
        let x: Vec<f32> = (0..t).flat_map(|i| data(cols, 50 + i as u32)).collect();
        let (mut xs, mut xq) = (vec![0.0; t * nb], vec![0i8; t * cols]);
        quantize_q8(&x, &mut xs, &mut xq);
        let tok = |i: usize| (&xs[i * nb..(i + 1) * nb], &xq[i * cols..(i + 1) * cols]);
        let (ps, pq) = pair_tokens(&xs, &xq, t, cols);
        assert_eq!(pq.len(), 2 * 2 * cols);
        let pair = |p: usize| {
            (
                &ps[p * 2 * nb..(p + 1) * 2 * nb],
                &pq[p * 2 * cols..(p + 1) * 2 * cols],
            )
        };
        let close = |a: f32, b: f32| (a - b).abs() <= 1e-4 * b.abs().max(1.0);

        let q8 = scalar::dot2_q8_pairs::<2>(
            [&rows[0].0, &rows[1].0],
            [&rows[0].1, &rows[1].1],
            [pair(0).0, pair(1).0],
            [pair(0).1, pair(1).1],
        );
        let fast8 = has_i8mm().then(|| {
            dot2_q8_pairs::<2>(
                [&rows[0].0, &rows[1].0],
                [&rows[0].1, &rows[1].1],
                [pair(0).0, pair(1).0],
                [pair(0).1, pair(1).1],
            )
        });
        let fast4 = has_i8mm().then(|| {
            dot2_q4_pairs::<2>(
                [&rows[0].2, &rows[1].2],
                [&rows[0].3, &rows[1].3],
                [pair(0).0, pair(1).0],
                [pair(0).1, pair(1).1],
            )
        });
        for p in 0..2 {
            for (k, (r, i)) in [(0, 2 * p), (0, 2 * p + 1), (1, 2 * p), (1, 2 * p + 1)]
                .into_iter()
                .enumerate()
            {
                let want8 = dot_q8(&rows[r].0, &rows[r].1, tok(i).0, tok(i).1);
                let want4 = dot_q4(&rows[r].2, &rows[r].3, tok(i).0, tok(i).1);
                assert!(close(q8[p][k], want8), "{} vs {want8}", q8[p][k]);
                // Bit for bit, so batching never changes a token's result.
                if let (Some(f8), Some(f4)) = (fast8, fast4) {
                    assert_eq!(f8[p][k], want8);
                    assert_eq!(f4[p][k], want4);
                }
            }
        }
    }

    #[test]
    fn quantizers_round_trip() {
        let x = data(256, 7);
        let mut s = vec![0.0; 8];
        let (mut q8, mut q4, mut out) = (vec![0i8; 256], vec![0u8; 128], vec![0.0; 256]);
        quantize_q8(&x, &mut s, &mut q8);
        dequantize_q8(&s, &q8, &mut out);
        assert!(x
            .iter()
            .zip(&out)
            .all(|(a, b)| (a - b).abs() <= 1.0 / 127.0));
        quantize_q4(&x, &mut s, &mut q4);
        dequantize_q4(&s, &q4, &mut out);
        assert!(x
            .iter()
            .zip(&out)
            .all(|(a, b)| (a - b).abs() <= 1.0 / 8.0 + 1e-6));
    }

    #[test]
    fn offset_q4_beats_symmetric_q4_on_its_own_measure() {
        let x = data(64 * QK, 11);
        let imp: Vec<f32> = data(64 * QK, 12)
            .iter()
            .map(|v| v * v * 10.0 + 0.01)
            .collect();
        let nb = x.len() / QK;
        let sigma2 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let werr = |out: &[f32], imp: Option<&[f32]>| -> f32 {
            x.iter()
                .zip(out)
                .enumerate()
                .map(|(j, (&a, &b))| {
                    imp.map_or(1.0, |i| i[j]) * (sigma2 + a * a).sqrt() * (a - b).powi(2)
                })
                .sum()
        };
        let (mut s, mut q) = (vec![0.0; nb], vec![0u8; x.len() / 2]);
        let mut sym = vec![0.0; x.len()];
        quantize_q4(&x, &mut s, &mut q);
        dequantize_q4(&s, &q, &mut sym);
        for imp in [None, Some(imp.as_slice())] {
            let mut packed = vec![0u32; nb.div_ceil(SUPER) * SUPER_WORDS];
            let mut q = vec![0u8; x.len() / 2];
            quantize_q4_packed(&x, imp, &mut packed, &mut q);
            let (mut d, mut c) = (vec![0.0; nb], vec![0.0; nb]);
            unpack_q4(&packed, &mut d, &mut c);
            let mut out = vec![0.0; x.len()];
            dequantize_q4(&d, &q, &mut out);
            for (block, &c) in out.chunks_exact_mut(QK).zip(&c) {
                block.iter_mut().for_each(|v| *v += c);
            }
            // Against each block's own free fit, in f32: packing the scales
            // into 6 bits costs little.
            let mut free = vec![0.0; x.len()];
            for (b, out) in free.chunks_exact_mut(QK).enumerate() {
                let xb = &x[b * QK..(b + 1) * QK];
                let w: Vec<f32> = (0..QK)
                    .map(|j| imp.map_or(1.0, |i| i[b * QK + j]) * (sigma2 + xb[j] * xb[j]).sqrt())
                    .collect();
                let (d, c, l) = fit_q4_offset(xb, &w);
                for j in 0..QK {
                    out[j] = d * (l[j] as f32 - 8.0) + c;
                }
            }
            let (e, e_sym, e_free) = (werr(&out, imp), werr(&sym, imp), werr(&free, imp));
            assert!(e < e_sym * 0.8, "{e} vs {e_sym}");
            assert!(e < e_free * 1.15, "{e} vs {e_free}");
        }
    }

    #[test]
    fn exp_is_within_two_ulp() {
        let mut xs: Vec<f32> = (0..200_000).map(|i| -87.0 + i as f32 * 0.000_875).collect();
        xs.extend([0.0, -0.0, 1.0, -1e-30, -50.5, -87.3]);
        let want: Vec<f32> = xs.iter().map(|x| x.exp()).collect();
        exp_inplace(&mut xs);
        for (x, (got, want)) in xs.iter().zip(want.iter()).enumerate() {
            let ulps = (got.to_bits() as i64 - want.to_bits() as i64).abs();
            assert!(ulps <= 2, "{x}: {got} vs {want} ({ulps} ulp)");
        }
    }

    #[test]
    fn packed_scales_round_trip() {
        for h in 0..0x7C00u16 {
            assert_eq!(f16_finite(h).to_bits(), f16_to_f32(h).to_bits());
            assert_eq!(
                f16_finite(h | 0x8000).to_bits(),
                f16_to_f32(h | 0x8000).to_bits()
            );
        }
        // A partial last super-block, and blocks of one value.
        let mut x = data(11 * QK, 5);
        x[3 * QK..4 * QK].fill(0.25);
        x[9 * QK..10 * QK].fill(0.0);
        let mut packed = vec![0u32; 2 * SUPER_WORDS];
        let mut q = vec![0u8; x.len() / 2];
        quantize_q4_packed(&x, None, &mut packed, &mut q);
        let (mut d, mut c) = (vec![0.0; 11], vec![0.0; 11]);
        unpack_q4(&packed, &mut d, &mut c);
        // The vectorized unpacking matches the plain one on any bits.
        let any: Vec<u32> = data(15, 9).iter().map(|v| v.to_bits() >> 2).collect();
        for nb in [8, 11, 24] {
            let (mut d1, mut c1, mut d2, mut c2) =
                (vec![0.0; nb], vec![0.0; nb], vec![0.0; nb], vec![0.0; nb]);
            unpack_q4(&any, &mut d1, &mut c1);
            unpack_q4_scalar(&any, &mut d2, &mut c2);
            assert_eq!((d1, c1), (d2, c2));
        }
        let mut out = vec![0.0; x.len()];
        dequantize_q4(&d, &q, &mut out);
        for (block, &c) in out.chunks_exact_mut(QK).zip(&c) {
            block.iter_mut().for_each(|v| *v += c);
        }
        assert!(out[9 * QK..10 * QK].iter().all(|&v| v == 0.0));
        assert!(out[3 * QK..4 * QK].iter().all(|&v| (v - 0.25).abs() < 0.01));
        let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
        let err = (x
            .iter()
            .zip(&out)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / x.len() as f32)
            .sqrt();
        assert!(err < rms * 0.12, "{err} vs {rms}");
        let (mut s, mut q8) = (vec![0u16; 11], vec![0i8; x.len()]);
        quantize_q8_weights(&x, &mut s, &mut q8);
        let mut ds = vec![0.0; 11];
        unpack_f16(&s, &mut ds);
        dequantize_q8(&ds, &q8, &mut out);
        assert!(x
            .iter()
            .zip(&out)
            .all(|(a, b)| (a - b).abs() <= 0.5 / 127.0 + 1e-3));
    }

    #[test]
    fn half_floats() {
        assert_eq!(bf16_to_f32(f32_to_bf16(1.5)), 1.5);
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        for (v, h) in [
            (1.0, 0x3C00),
            (-2.0, 0xC000),
            (65504.0, 0x7BFF),
            (65520.0, 0x7C00), // halfway to 65536: ties to even, which overflows
            (2f32.powi(-24), 0x0001),
            (2f32.powi(-14), 0x0400),
            (1.0 + 2f32.powi(-11), 0x3C00), // halfway: down to even
            (1.0 + 3.0 * 2f32.powi(-11), 0x3C02),
        ] {
            assert_eq!(f32_to_f16(v), h, "{v}");
        }
        let x = data(1000, 3);
        assert!(x
            .iter()
            .all(|&v| (f16_to_f32(f32_to_f16(v * 100.0)) - v * 100.0).abs() <= 0.05));
    }
}
