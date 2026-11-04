//! WebAssembly SIMD (128-bit) kernels, for the browser build: int8 products
//! widen to 16 bits and pair up with `i32x4.dot_i16x8_s`. Each block's
//! integer sum is scaled into one running total, the same arithmetic for one
//! token or four, so batching never changes a result.

use super::{bf16_to_f32, QK};
use std::arch::wasm32::*;

/// The sum of a block's 32 int8 products.
#[inline(always)]
unsafe fn block(w0: v128, w1: v128, x: *const i8) -> i32 {
    let (x0, x1) = (
        v128_load(x as *const v128),
        v128_load(x.add(16) as *const v128),
    );
    let d = |w: v128, x: v128| {
        i32x4_add(
            i32x4_dot_i16x8(i16x8_extend_low_i8x16(w), i16x8_extend_low_i8x16(x)),
            i32x4_dot_i16x8(i16x8_extend_high_i8x16(w), i16x8_extend_high_i8x16(x)),
        )
    };
    let s = i32x4_add(d(w0, x0), d(w1, x1));
    i32x4_extract_lane::<0>(s)
        + i32x4_extract_lane::<1>(s)
        + i32x4_extract_lane::<2>(s)
        + i32x4_extract_lane::<3>(s)
}

pub fn dot_q8(ws: &[f32], wq: &[i8], xs: &[f32], xq: &[i8]) -> f32 {
    let nb = ws.len();
    assert!(wq.len() >= nb * QK && xq.len() >= nb * QK && xs.len() >= nb);
    let mut acc = 0.0;
    for b in 0..nb {
        let s = unsafe {
            let p = wq.as_ptr().add(b * QK);
            let (w0, w1) = (
                v128_load(p as *const v128),
                v128_load(p.add(16) as *const v128),
            );
            block(w0, w1, xq.as_ptr().add(b * QK))
        };
        acc += s as f32 * (ws[b] * xs[b]);
    }
    acc
}

pub fn dot_q4(ws: &[f32], wq: &[u8], xs: &[f32], xq: &[i8]) -> f32 {
    let nb = ws.len();
    assert!(wq.len() >= nb * QK / 2 && xq.len() >= nb * QK && xs.len() >= nb);
    let mut acc = 0.0;
    for b in 0..nb {
        let s = unsafe {
            let raw = v128_load(wq.as_ptr().add(b * QK / 2) as *const v128);
            let eight = i8x16_splat(8);
            let w0 = i8x16_sub(v128_and(raw, u8x16_splat(0x0F)), eight);
            let w1 = i8x16_sub(u8x16_shr(raw, 4), eight);
            block(w0, w1, xq.as_ptr().add(b * QK))
        };
        acc += s as f32 * (ws[b] * xs[b]);
    }
    acc
}

pub fn dot_q8_x4(ws: &[f32], wq: &[i8], xs: [&[f32]; 4], xq: [&[i8]; 4]) -> [f32; 4] {
    [0, 1, 2, 3].map(|j| dot_q8(ws, wq, xs[j], xq[j]))
}

pub fn dot_q4_x4(ws: &[f32], wq: &[u8], xs: [&[f32]; 4], xq: [&[i8]; 4]) -> [f32; 4] {
    [0, 1, 2, 3].map(|j| dot_q4(ws, wq, xs[j], xq[j]))
}

fn sum(v: v128) -> f32 {
    f32x4_extract_lane::<0>(v)
        + f32x4_extract_lane::<1>(v)
        + f32x4_extract_lane::<2>(v)
        + f32x4_extract_lane::<3>(v)
}

pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut acc = f32x4_splat(0.0);
    let mut i = 0;
    while i + 4 <= n {
        unsafe {
            let x = v128_load(a.as_ptr().add(i) as *const v128);
            let y = v128_load(b.as_ptr().add(i) as *const v128);
            acc = f32x4_add(acc, f32x4_mul(x, y));
        }
        i += 4;
    }
    sum(acc) + (i..n).map(|j| a[j] * b[j]).sum::<f32>()
}

pub fn dot_bf16(w: &[u16], x: &[f32]) -> f32 {
    let n = w.len().min(x.len());
    let mut acc = f32x4_splat(0.0);
    let mut i = 0;
    while i + 4 <= n {
        unsafe {
            let h = v128_load64_zero(w.as_ptr().add(i) as *const u64);
            let f = i32x4_shl(u32x4_extend_low_u16x8(h), 16);
            acc = f32x4_add(
                acc,
                f32x4_mul(f, v128_load(x.as_ptr().add(i) as *const v128)),
            );
        }
        i += 4;
    }
    sum(acc) + (i..n).map(|j| bf16_to_f32(w[j]) * x[j]).sum::<f32>()
}
