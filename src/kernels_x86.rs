//! x86-64 kernels: AVX2 and FMA, picked at run time. The build targets the
//! x86-64 baseline, so one binary runs anywhere and a CPU without AVX2 takes
//! the scalar path.
//!
//! int8 products use `maddubs` (unsigned by signed bytes into pairs of 16-bit
//! sums): |w| times x carrying w's sign for 8-bit weights, and the raw 0..15
//! nibbles times x, less 8 times x's sum, for 4-bit ones. Both are exact.
//! Each block's eight lane sums are scaled into eight running totals, the
//! same instructions in the same order for one token or four, so batching
//! never changes a result.

use super::{scalar, QK};
use std::arch::x86_64::*;
use std::sync::atomic::{AtomicU8, Ordering::Relaxed};

/// Whether this CPU has AVX2 and FMA (Intel since Haswell, AMD since Zen),
/// checked once.
fn avx2() -> bool {
    static HAS: AtomicU8 = AtomicU8::new(0);
    match HAS.load(Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let has = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
            HAS.store(if has { 2 } else { 1 }, Relaxed);
            has
        }
    }
}

#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn hsum(v: __m256) -> f32 {
    let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_movehdup_ps(s));
    _mm_cvtss_f32(s)
}

/// 32 int8 weights by 32 int8 activations: eight lane sums of four products.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn block_q8(w: __m256i, x: __m256i) -> __m256 {
    let p = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(x, w));
    _mm256_cvtepi32_ps(_mm256_madd_epi16(p, _mm256_set1_epi16(1)))
}

/// 32 4-bit weights (16 bytes: low nibbles are values 0..16, high nibbles
/// 16..32) as unsigned 0..15.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn nibbles(p: *const u8) -> __m256i {
    let raw = _mm_loadu_si128(p as *const __m128i);
    let mask = _mm_set1_epi8(0x0F);
    let lo = _mm_and_si128(raw, mask);
    let hi = _mm_and_si128(_mm_srli_epi16::<4>(raw), mask);
    _mm256_set_m128i(hi, lo)
}

/// The same for 4-bit weights stored as q + 8: sum (q - 8) x.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn block_q4(w: __m256i, x: __m256i) -> __m256 {
    let p = _mm256_maddubs_epi16(w, x);
    let eights = _mm256_maddubs_epi16(_mm256_set1_epi8(8), x);
    _mm256_cvtepi32_ps(_mm256_madd_epi16(
        _mm256_sub_epi16(p, eights),
        _mm256_set1_epi16(1),
    ))
}

#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn load(p: *const i8) -> __m256i {
    _mm256_loadu_si256(p as *const __m256i)
}

#[target_feature(enable = "avx2,fma")]
unsafe fn q8(ws: &[f32], wq: &[i8], xs: &[f32], xq: &[i8]) -> f32 {
    let mut acc = _mm256_setzero_ps();
    for b in 0..ws.len() {
        let s = block_q8(load(wq.as_ptr().add(b * QK)), load(xq.as_ptr().add(b * QK)));
        let d = *ws.get_unchecked(b) * *xs.get_unchecked(b);
        acc = _mm256_fmadd_ps(s, _mm256_set1_ps(d), acc);
    }
    hsum(acc)
}

#[target_feature(enable = "avx2,fma")]
unsafe fn q8_x4(ws: &[f32], wq: &[i8], xs: [&[f32]; 4], xq: [&[i8]; 4]) -> [f32; 4] {
    let mut acc = [_mm256_setzero_ps(); 4];
    for b in 0..ws.len() {
        let w = load(wq.as_ptr().add(b * QK));
        for (j, acc) in acc.iter_mut().enumerate() {
            let s = block_q8(w, load(xq[j].as_ptr().add(b * QK)));
            let d = *ws.get_unchecked(b) * *xs[j].get_unchecked(b);
            *acc = _mm256_fmadd_ps(s, _mm256_set1_ps(d), *acc);
        }
    }
    [hsum(acc[0]), hsum(acc[1]), hsum(acc[2]), hsum(acc[3])]
}

#[target_feature(enable = "avx2,fma")]
unsafe fn q4(ws: &[f32], wq: &[u8], xs: &[f32], xq: &[i8]) -> f32 {
    let mut acc = _mm256_setzero_ps();
    for b in 0..ws.len() {
        let s = block_q4(
            nibbles(wq.as_ptr().add(b * QK / 2)),
            load(xq.as_ptr().add(b * QK)),
        );
        let d = *ws.get_unchecked(b) * *xs.get_unchecked(b);
        acc = _mm256_fmadd_ps(s, _mm256_set1_ps(d), acc);
    }
    hsum(acc)
}

#[target_feature(enable = "avx2,fma")]
unsafe fn q4_x4(ws: &[f32], wq: &[u8], xs: [&[f32]; 4], xq: [&[i8]; 4]) -> [f32; 4] {
    let mut acc = [_mm256_setzero_ps(); 4];
    for b in 0..ws.len() {
        let w = nibbles(wq.as_ptr().add(b * QK / 2));
        for (j, acc) in acc.iter_mut().enumerate() {
            let s = block_q4(w, load(xq[j].as_ptr().add(b * QK)));
            let d = *ws.get_unchecked(b) * *xs[j].get_unchecked(b);
            *acc = _mm256_fmadd_ps(s, _mm256_set1_ps(d), *acc);
        }
    }
    [hsum(acc[0]), hsum(acc[1]), hsum(acc[2]), hsum(acc[3])]
}

#[target_feature(enable = "avx2,fma")]
unsafe fn f32s(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut acc = [_mm256_setzero_ps(); 4];
    let mut i = 0;
    while i + 32 <= n {
        for (k, acc) in acc.iter_mut().enumerate() {
            let o = i + 8 * k;
            *acc = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(o)), _mm256_loadu_ps(pb.add(o)), *acc);
        }
        i += 32;
    }
    while i + 8 <= n {
        acc[0] = _mm256_fmadd_ps(
            _mm256_loadu_ps(pa.add(i)),
            _mm256_loadu_ps(pb.add(i)),
            acc[0],
        );
        i += 8;
    }
    let all = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
    hsum(all) + (i..n).map(|j| a[j] * b[j]).sum::<f32>()
}

#[target_feature(enable = "avx2,fma")]
unsafe fn bf16s(w: &[u16], x: &[f32]) -> f32 {
    let n = w.len().min(x.len());
    let mut acc = [_mm256_setzero_ps(); 2];
    let mut i = 0;
    while i + 16 <= n {
        for (k, acc) in acc.iter_mut().enumerate() {
            let h = _mm_loadu_si128(w.as_ptr().add(i + 8 * k) as *const __m128i);
            let f = _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(h)));
            *acc = _mm256_fmadd_ps(f, _mm256_loadu_ps(x.as_ptr().add(i + 8 * k)), *acc);
        }
        i += 16;
    }
    hsum(_mm256_add_ps(acc[0], acc[1]))
        + (i..n).map(|j| super::bf16_to_f32(w[j]) * x[j]).sum::<f32>()
}

fn check_q8(ws: &[f32], wq: usize, xs: &[f32], xq: &[i8]) {
    let nb = ws.len();
    assert!(wq >= nb * QK && xq.len() >= nb * QK && xs.len() >= nb);
}

pub fn dot_q8(ws: &[f32], wq: &[i8], xs: &[f32], xq: &[i8]) -> f32 {
    check_q8(ws, wq.len(), xs, xq);
    if avx2() {
        unsafe { q8(ws, wq, xs, xq) }
    } else {
        scalar::dot_q8(ws, wq, xs, xq)
    }
}

pub fn dot_q4(ws: &[f32], wq: &[u8], xs: &[f32], xq: &[i8]) -> f32 {
    check_q8(ws, wq.len() * 2, xs, xq);
    if avx2() {
        unsafe { q4(ws, wq, xs, xq) }
    } else {
        scalar::dot_q4(ws, wq, xs, xq)
    }
}

pub fn dot_q8_x4(ws: &[f32], wq: &[i8], xs: [&[f32]; 4], xq: [&[i8]; 4]) -> [f32; 4] {
    (0..4).for_each(|j| check_q8(ws, wq.len(), xs[j], xq[j]));
    if avx2() {
        unsafe { q8_x4(ws, wq, xs, xq) }
    } else {
        scalar::dot_q8_x4(ws, wq, xs, xq)
    }
}

pub fn dot_q4_x4(ws: &[f32], wq: &[u8], xs: [&[f32]; 4], xq: [&[i8]; 4]) -> [f32; 4] {
    (0..4).for_each(|j| check_q8(ws, wq.len() * 2, xs[j], xq[j]));
    if avx2() {
        unsafe { q4_x4(ws, wq, xs, xq) }
    } else {
        scalar::dot_q4_x4(ws, wq, xs, xq)
    }
}

pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    if avx2() {
        unsafe { f32s(a, b) }
    } else {
        scalar::dot_f32(a, b)
    }
}

pub fn dot_bf16(w: &[u16], x: &[f32]) -> f32 {
    if avx2() {
        unsafe { bf16s(w, x) }
    } else {
        scalar::dot_bf16(w, x)
    }
}
