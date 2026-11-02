//! Inner loops: a dot product for each weight format, quantizers, and the
//! small element-wise ops around them. NEON intrinsics on aarch64 (every
//! Apple Silicon Mac), plain Rust elsewhere.
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

/// [`quantize_q4`] for one row, guided by `importance[j]`, the mean squared
/// input that column `j` multiplies (see `imatrix.rs`). Each block picks the
/// scale minimizing `sum_j w_j (x_j - d q_j)^2` with
/// `w_j = importance_j * sqrt(sigma^2 + x_j^2)` (llama.cpp's weighting;
/// sigma^2 is the row's mean squared weight). The plain scale is among the
/// candidates, so the weighted error is never worse than [`quantize_q4`]'s.
pub fn quantize_q4_weighted(row: &[f32], importance: &[f32], scales: &mut [f32], qs: &mut [u8]) {
    let sigma2 = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
    for (b, ((block, q), scale)) in row
        .chunks_exact(QK)
        .zip(qs.chunks_exact_mut(QK / 2))
        .zip(scales.iter_mut())
        .enumerate()
    {
        let imp = &importance[b * QK..(b + 1) * QK];
        let mut w = [0f32; QK];
        for j in 0..QK {
            w[j] = imp[j] * (sigma2 + block[j] * block[j]).sqrt();
        }
        let (d, ints) = fit_q4_weighted(block, &w);
        for j in 0..QK / 2 {
            q[j] = (ints[j] + 8) as u8 | (((ints[j + QK / 2] + 8) as u8) << 4);
        }
        *scale = d;
    }
}

fn fit_q4_weighted(x: &[f32], w: &[f32]) -> (f32, [i8; QK]) {
    let max = x
        .iter()
        .copied()
        .fold(0f32, |m, v| if v.abs() > m.abs() { v } else { m });
    if max == 0.0 {
        return (0.0, [0; QK]);
    }
    let err = |d: f32, q: &[i8; QK]| {
        x.iter()
            .zip(q)
            .zip(w)
            .map(|((&v, &q), &w)| w * (v - d * q as f32).powi(2))
            .sum::<f32>()
    };
    let ints_for = |iscale: f32| {
        let mut q = [0i8; QK];
        for (q, &v) in q.iter_mut().zip(x) {
            *q = (v * iscale).round().clamp(-8.0, 7.0) as i8;
        }
        q
    };
    // The plain choice: the largest value maps exactly to -8.
    let plain = ints_for(-8.0 / max);
    let mut best = (max / -8.0, plain, err(max / -8.0, &plain));
    for step in -9..=9 {
        let q = ints_for(-(8.0 + 0.1 * step as f32) / max);
        let (mut sxq, mut sqq) = (0f32, 0f32);
        for ((&v, &q), &w) in x.iter().zip(&q).zip(w) {
            sxq += w * v * q as f32;
            sqq += w * (q as f32) * (q as f32);
        }
        if sqq > 0.0 {
            let d = sxq / sqq;
            let e = err(d, &q);
            if e < best.2 {
                best = (d, q, e);
            }
        }
    }
    (best.0, best.1)
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

#[cfg(target_arch = "aarch64")]
pub use neon::{dot_bf16, dot_f32, dot_q4, dot_q4_x4, dot_q8, dot_q8_x4};
#[cfg(not(target_arch = "aarch64"))]
pub use scalar::{dot_bf16, dot_f32, dot_q4, dot_q4_x4, dot_q8, dot_q8_x4};

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::QK;
    use std::arch::aarch64::*;

    /// Whether this CPU has SDOT (ARMv8.4 dot product; every Apple Silicon
    /// chip does). Checked once.
    fn dotprod() -> bool {
        use std::sync::atomic::{AtomicU8, Ordering::Relaxed};
        static HAS: AtomicU8 = AtomicU8::new(0);
        match HAS.load(Relaxed) {
            1 => false,
            2 => true,
            _ => {
                let has = std::arch::is_aarch64_feature_detected!("dotprod");
                HAS.store(if has { 2 } else { 1 }, Relaxed);
                has
            }
        }
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
        let mut acc = [vdupq_n_f32(0.0); N];
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
                acc[j] = vfmaq_f32(
                    acc[j],
                    vcvtq_f32_s32(s),
                    vdupq_n_f32(wd * *xs[j].get_unchecked(b)),
                );
            }
        }
        acc.map(|a| vaddvq_f32(a))
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

#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
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
    fn weighted_q4_never_loses_to_plain_on_its_own_measure() {
        let x = data(64 * QK, 11);
        let imp: Vec<f32> = data(64 * QK, 12)
            .iter()
            .map(|v| v * v * 10.0 + 0.01)
            .collect();
        let nb = x.len() / QK;
        let (mut s, mut q) = (vec![0.0; nb], vec![0u8; x.len() / 2]);
        let (mut out_w, mut out_p) = (vec![0.0; x.len()], vec![0.0; x.len()]);
        quantize_q4_weighted(&x, &imp, &mut s, &mut q);
        dequantize_q4(&s, &q, &mut out_w);
        quantize_q4(&x, &mut s, &mut q);
        dequantize_q4(&s, &q, &mut out_p);
        let sigma2 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let werr = |out: &[f32]| -> f32 {
            x.iter()
                .zip(out)
                .zip(&imp)
                .map(|((&a, &b), &i)| i * (sigma2 + a * a).sqrt() * (a - b).powi(2))
                .sum()
        };
        assert!(
            werr(&out_w) <= werr(&out_p) * 1.0001,
            "{} vs {}",
            werr(&out_w),
            werr(&out_p)
        );
    }

    #[test]
    fn half_floats() {
        assert_eq!(bf16_to_f32(f32_to_bf16(1.5)), 1.5);
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
    }
}
