//! Inner loops: a dot product for each weight format, quantizers, and the
//! small element-wise ops around them. NEON intrinsics on aarch64 (every
//! Apple Silicon Mac), AVX2 on x86-64, SIMD128 in the browser, plain Rust
//! elsewhere.
//!
//! Quantized formats store weights in blocks of [`QK`] values sharing one f32
//! scale. Activations are quantized to 8 bits on the fly so the inner loop is
//! integer multiply-accumulate, then each block's sum is scaled once.

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

/// 4-bit with an offset per block: value = `d * (q - 8) + c`, q in 0..=15,
/// so a block's 16 levels span its own range instead of one centred on
/// zero. Same nibble layout as [`quantize_q4`]. For each block the scale and
/// offset minimize `sum_j w_j (x_j - value_j)^2` (a weighted least-squares
/// fit for each of several roundings, as llama.cpp's k-quants search), with
/// `w_j = importance_j * sqrt(sigma^2 + x_j^2)` and importance 1 without an
/// importance matrix.
pub fn quantize_q4_offset(
    row: &[f32],
    importance: Option<&[f32]>,
    scales: &mut [f32],
    offsets: &mut [f32],
    qs: &mut [u8],
) {
    let sigma2 = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
    for (b, block) in row.chunks_exact(QK).enumerate() {
        let mut w = [0f32; QK];
        for j in 0..QK {
            let imp = importance.map_or(1.0, |i| i[b * QK + j]);
            w[j] = imp * (sigma2 + block[j] * block[j]).sqrt();
        }
        let (d, c, levels) = fit_q4_offset(block, &w);
        let q = &mut qs[b * QK / 2..(b + 1) * QK / 2];
        for j in 0..QK / 2 {
            q[j] = levels[j] | (levels[j + QK / 2] << 4);
        }
        scales[b] = d;
        offsets[b] = c;
    }
}

/// (scale, offset, levels) for one block; see [`quantize_q4_offset`].
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

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
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

#[cfg(target_arch = "aarch64")]
pub use neon::{
    dot2_q4_pairs, dot2_q8_pairs, dot_bf16, dot_f32, dot_q4, dot_q4_x4, dot_q8, dot_q8_x4, has_i8mm,
};
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
    use super::QK;
    use std::arch::aarch64::*;

    use std::sync::atomic::{AtomicU8, Ordering::Relaxed};

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
    unsafe fn dot_rows_sdot<const N: usize, const Q4: bool>(
        ws: &[f32],
        wq: *const u8,
        xs: [&[f32]; N],
        xq: [&[i8]; N],
    ) -> [f32; N] {
        let nb = ws.len();
        for j in 0..N {
            assert!(xq[j].len() >= nb * QK && xs[j].len() >= nb);
        }
        let mut acc = [0f32; N];
        for b in 0..nb {
            let (w0, w1) = if Q4 {
                unpack_q4(wq.add(b * QK / 2))
            } else {
                let p = wq.add(b * QK) as *const i8;
                (vld1q_s8(p), vld1q_s8(p.add(16)))
            };
            let wd = *ws.get_unchecked(b);
            for j in 0..N {
                let px = xq[j].as_ptr().add(b * QK);
                let s = sdot(
                    sdot(vdupq_n_s32(0), w0, vld1q_s8(px)),
                    w1,
                    vld1q_s8(px.add(16)),
                );
                acc[j] = (vaddvq_s32(s) as f32).mul_add(wd * *xs[j].get_unchecked(b), acc[j]);
            }
        }
        acc
    }

    pub fn dot_q8(ws: &[f32], wq: &[i8], xs: &[f32], xq: &[i8]) -> f32 {
        if dotprod() {
            assert!(wq.len() >= ws.len() * QK);
            unsafe { dot_rows_sdot::<1, false>(ws, wq.as_ptr() as *const u8, [xs], [xq])[0] }
        } else {
            dot_q8_neon(ws, wq, xs, xq)
        }
    }

    pub fn dot_q4(ws: &[f32], wq: &[u8], xs: &[f32], xq: &[i8]) -> f32 {
        if dotprod() {
            assert!(wq.len() >= ws.len() * QK / 2);
            unsafe { dot_rows_sdot::<1, true>(ws, wq.as_ptr(), [xs], [xq])[0] }
        } else {
            dot_q4_neon(ws, wq, xs, xq)
        }
    }

    /// Four tokens against one 8-bit weight row.
    pub fn dot_q8_x4(ws: &[f32], wq: &[i8], xs: [&[f32]; 4], xq: [&[i8]; 4]) -> [f32; 4] {
        if dotprod() {
            assert!(wq.len() >= ws.len() * QK);
            unsafe { dot_rows_sdot::<4, false>(ws, wq.as_ptr() as *const u8, xs, xq) }
        } else {
            [0, 1, 2, 3].map(|j| dot_q8_neon(ws, wq, xs[j], xq[j]))
        }
    }

    /// Four tokens against one 4-bit weight row.
    pub fn dot_q4_x4(ws: &[f32], wq: &[u8], xs: [&[f32]; 4], xq: [&[i8]; 4]) -> [f32; 4] {
        if dotprod() {
            assert!(wq.len() >= ws.len() * QK / 2);
            unsafe { dot_rows_sdot::<4, true>(ws, wq.as_ptr(), xs, xq) }
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
    use super::{bf16_to_f32, QK};

    pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    pub fn dot_bf16(w: &[u16], x: &[f32]) -> f32 {
        w.iter().zip(x).map(|(&w, &x)| bf16_to_f32(w) * x).sum()
    }

    pub fn dot_q8(ws: &[f32], wq: &[i8], xs: &[f32], xq: &[i8]) -> f32 {
        (0..ws.len())
            .map(|b| {
                let s: i32 = (0..QK)
                    .map(|j| wq[b * QK + j] as i32 * xq[b * QK + j] as i32)
                    .sum();
                s as f32 * ws[b] * xs[b]
            })
            .sum()
    }

    pub fn dot_q4(ws: &[f32], wq: &[u8], xs: &[f32], xq: &[i8]) -> f32 {
        (0..ws.len())
            .map(|b| {
                let mut s = 0i32;
                for j in 0..QK / 2 {
                    let byte = wq[b * QK / 2 + j];
                    s += ((byte & 0xF) as i32 - 8) * xq[b * QK + j] as i32;
                    s += ((byte >> 4) as i32 - 8) * xq[b * QK + j + QK / 2] as i32;
                }
                s as f32 * ws[b] * xs[b]
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
            let (mut s, mut o, mut q) = (vec![0.0; nb], vec![0.0; nb], vec![0u8; x.len() / 2]);
            quantize_q4_offset(&x, imp, &mut s, &mut o, &mut q);
            let mut out = vec![0.0; x.len()];
            dequantize_q4(&s, &q, &mut out);
            for (block, &c) in out.chunks_exact_mut(QK).zip(&o) {
                block.iter_mut().for_each(|v| *v += c);
            }
            assert!(
                werr(&out, imp) < werr(&sym, imp) * 0.8,
                "{} vs {}",
                werr(&out, imp),
                werr(&sym, imp)
            );
        }
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
