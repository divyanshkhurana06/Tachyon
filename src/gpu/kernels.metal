// Compute kernels for the Metal backend. Compiled at runtime from this
// source by the OS's built-in Metal compiler (no Xcode needed).

#include <metal_stdlib>
using namespace metal;

// Matrix multiply. Each simdgroup (32 lanes) computes R output rows: a lane
// loads 4 input values once and reuses them from registers for all R rows,
// so input reads don't swamp the weight reads (for 4-bit weights the input
// is 8x more bytes than the weights). Lanes take consecutive 4-element
// slices, so both weight and input loads are contiguous across the
// simdgroup. With several tokens, TB of them share each weight load.
constant uint SIMDS = 2; // simdgroups per threadgroup
constant uint R = 4;     // rows per simdgroup
constant uint TB = 8;    // tokens per pass

struct MatParams {
    uint rows;
    uint cols;
    uint t;
    uint accumulate; // y += W·x instead of y = W·x (fuses the residual add)
    uint has_bias;
};

inline void write_out(device float *y, device const float *bias, constant MatParams &p, uint token, uint row, float s) {
    ulong idx = (ulong)token * p.rows + row;
    if (p.has_bias) s += bias[row];
    y[idx] = p.accumulate ? y[idx] + s : s;
}

// Loaders: weights (row, k..k+4) as float4. `cols` is a multiple of 32.
struct LoadBf16 {
    device const ushort *w;
    uint cols;
    float4 operator()(uint row, uint k) const {
        ushort4 q = *(device const ushort4 *)(w + (ulong)row * cols + k);
        return float4(as_type<float>(uint(q.x) << 16), as_type<float>(uint(q.y) << 16),
                      as_type<float>(uint(q.z) << 16), as_type<float>(uint(q.w) << 16));
    }
};

struct LoadQ8 {
    device const float *scales;
    device const char *qs;
    uint cols;
    float4 operator()(uint row, uint k) const {
        ulong e = (ulong)row * cols + k;
        return float4(*(device const char4 *)(qs + e)) * scales[e / 32];
    }
};

// Q4 blocks: 16 bytes per 32 weights; byte j holds weight j (low nibble) and
// weight j+16 (high nibble), each stored offset by 8.
struct LoadQ4 {
    device const float *scales;
    device const uchar *qs;
    uint cols;
    float4 operator()(uint row, uint k) const {
        ulong e = (ulong)row * cols + k;
        ulong block = e / 32;
        uint j = e % 32;
        uchar4 c = *(device const uchar4 *)(qs + block * 16 + (j & 15));
        float4 v = j < 16 ? float4(c & uchar4(0xF)) : float4(c >> uchar4(4));
        return (v - 8.0f) * scales[block];
    }
};

template <typename L>
inline void matmul_impl(thread const L &load, device const float *x, device float *y, device const float *bias,
                        constant MatParams &p, uint row0, uint lane) {
    if (p.t == 1) { // decode: the hot path, kept lean
        float acc[R] = {0, 0, 0, 0};
        for (uint k = lane * 4; k < p.cols; k += 128) {
            float4 xv = *(device const float4 *)(x + k);
            for (uint r = 0; r < R; r++) {
                if (row0 + r < p.rows) acc[r] += dot(load(row0 + r, k), xv);
            }
        }
        for (uint r = 0; r < R; r++) {
            float s = simd_sum(acc[r]);
            if (lane == 0 && row0 + r < p.rows) write_out(y, bias, p, 0, row0 + r, s);
        }
        return;
    }
    for (uint base = 0; base < p.t; base += TB) {
        uint nb = min(TB, p.t - base);
        float acc[R * TB] = {0};
        for (uint k = lane * 4; k < p.cols; k += 128) {
            float4 xv[TB];
            for (uint j = 0; j < TB; j++) {
                if (j < nb) xv[j] = *(device const float4 *)(x + (ulong)(base + j) * p.cols + k);
            }
            for (uint r = 0; r < R; r++) {
                if (row0 + r >= p.rows) break;
                float4 wf = load(row0 + r, k);
                for (uint j = 0; j < TB; j++) {
                    if (j < nb) acc[r * TB + j] += dot(wf, xv[j]);
                }
            }
        }
        for (uint r = 0; r < R; r++) {
            for (uint j = 0; j < TB; j++) {
                if (j < nb) {
                    float s = simd_sum(acc[r * TB + j]);
                    if (lane == 0 && row0 + r < p.rows) write_out(y, bias, p, base + j, row0 + r, s);
                }
            }
        }
    }
}

kernel void matmul_bf16(device const ushort *w [[buffer(0)]], device const float *x [[buffer(1)]],
                        device float *y [[buffer(2)]], device const float *bias [[buffer(3)]],
                        constant MatParams &p [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    LoadBf16 load = {w, p.cols};
    matmul_impl(load, x, y, bias, p, (tg * SIMDS + sg) * R, lane);
}

kernel void matmul_q8(device const float *scales [[buffer(0)]], device const char *qs [[buffer(1)]],
                      device const float *x [[buffer(2)]], device float *y [[buffer(3)]],
                      device const float *bias [[buffer(4)]], constant MatParams &p [[buffer(5)]],
                      uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    LoadQ8 load = {scales, qs, p.cols};
    matmul_impl(load, x, y, bias, p, (tg * SIMDS + sg) * R, lane);
}

kernel void matmul_q4(device const float *scales [[buffer(0)]], device const uchar *qs [[buffer(1)]],
                      device const float *x [[buffer(2)]], device float *y [[buffer(3)]],
                      device const float *bias [[buffer(4)]], constant MatParams &p [[buffer(5)]],
                      uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    LoadQ4 load = {scales, qs, p.cols};
    matmul_impl(load, x, y, bias, p, (tg * SIMDS + sg) * R, lane);
}

constant uint BLOCK = 16; // KV cache block size, as in kv.rs

struct NormParams {
    uint n;
    float eps;
    uint gather; // read input row rows[i] instead of row i
};

// One threadgroup per output row.
kernel void rmsnorm(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
                    device float *out [[buffer(2)]], device const uint *rows [[buffer(3)]],
                    constant NormParams &p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]], uint size [[threads_per_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[32];
    device const float *xr = x + (ulong)(p.gather ? rows[row] : row) * p.n;
    device float *o = out + (ulong)row * p.n;
    float ss = 0;
    for (uint i = tid; i < p.n; i += size) ss += xr[i] * xr[i];
    ss = simd_sum(ss);
    if (lane == 0) partial[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0;
    for (uint i = 0; i < (size + 31) / 32; i++) total += partial[i];
    float inv = rsqrt(total / p.n + p.eps);
    for (uint i = tid; i < p.n; i += size) o[i] = xr[i] * inv * w[i];
}

struct HeadNormParams {
    uint head_dim;
    uint heads;
    uint qkv_dim;
    uint q_dim;
    float eps;
};

// Qwen3: RMSNorm over each q and k head before RoPE. One threadgroup of
// head_dim threads per (head, token); heads past `heads` are k heads.
kernel void head_norm(device float *qkv [[buffer(0)]], device const float *qw [[buffer(1)]],
                      device const float *kw [[buffer(2)]], constant HeadNormParams &p [[buffer(3)]],
                      uint2 tg [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[4];
    uint h = tg.x, i = tg.y;
    bool is_q = h < p.heads;
    device float *v = qkv + (ulong)i * p.qkv_dim + (is_q ? h * p.head_dim : p.q_dim + (h - p.heads) * p.head_dim);
    float x = v[d];
    float ss = simd_sum(x * x);
    if (lane == 0) partial[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0;
    for (uint j = 0; j < p.head_dim / 32; j++) total += partial[j];
    v[d] = x * rsqrt(total / p.head_dim + p.eps) * (is_q ? qw[d] : kw[d]);
}

struct RopeParams {
    uint head_dim;
    uint heads;
    uint kv_heads;
    uint qkv_dim;
    uint q_dim;
    uint kv_dim;
};

// Rotates q in place, and writes rotated k and plain v into this layer's KV
// cache at each token's slot. Grid: (head_dim/2, heads + kv_heads, tokens).
kernel void rope_store(device float *qkv [[buffer(0)]], device float *kc [[buffer(1)]], device float *vc [[buffer(2)]],
                       device const float *cosb [[buffer(3)]], device const float *sinb [[buffer(4)]],
                       device const uint *slots [[buffer(5)]], constant RopeParams &p [[buffer(6)]],
                       uint3 gid [[thread_position_in_grid]]) {
    uint half_dim = p.head_dim / 2;
    uint d = gid.x, h = gid.y, i = gid.z;
    if (d >= half_dim || h >= p.heads + p.kv_heads) return;
    float c = cosb[i * half_dim + d], s = sinb[i * half_dim + d];
    device float *row = qkv + (ulong)i * p.qkv_dim;
    if (h < p.heads) {
        device float *q = row + h * p.head_dim;
        float a = q[d], b = q[d + half_dim];
        q[d] = a * c - b * s;
        q[d + half_dim] = b * c + a * s;
    } else {
        uint kh = h - p.heads;
        device const float *k = row + p.q_dim + kh * p.head_dim;
        device const float *v = row + p.q_dim + p.kv_dim + kh * p.head_dim;
        ulong at = (ulong)slots[i] * p.kv_dim + kh * p.head_dim;
        float a = k[d], b = k[d + half_dim];
        kc[at + d] = a * c - b * s;
        kc[at + d + half_dim] = b * c + a * s;
        vc[at + d] = v[d];
        vc[at + d + half_dim] = v[d + half_dim];
    }
}

struct AttnParams {
    uint heads;
    uint kv_heads;
    uint head_dim;
    uint qkv_dim;
    uint q_dim;
    uint kv_dim;
    uint max_seq;
    float scale;
};

constant uint ATT_THREADS = 256;
constant uint ATT_SIMDS = ATT_THREADS / 32;

inline ulong cache_slot(device const uint *table, uint s) {
    return (ulong)table[s / BLOCK] * BLOCK + s % BLOCK;
}

// One threadgroup per (head, token). Each token attends to its own
// sequence's positions 0..=pos, found through the sequence's block table.
// Scores go to a scratch buffer so any context length fits; three passes:
// q·k (a thread per position), softmax, weighted sum of v.
kernel void attention(device const float *qkv [[buffer(0)]], device const float *kc [[buffer(1)]],
                      device const float *vc [[buffer(2)]], device float *scores [[buffer(3)]],
                      device float *out [[buffer(4)]], device const uint *positions [[buffer(5)]],
                      device const uint *table_of [[buffer(6)]], device const uint *tables [[buffer(7)]],
                      constant AttnParams &p [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[ATT_SIMDS];
    threadgroup float partial[ATT_THREADS];
    threadgroup float4 q4[32]; // head_dim <= 128
    uint h = tg.x, i = tg.y;
    uint n = positions[i] + 1;
    device const uint *table = tables + table_of[i];
    uint kv = (h / (p.heads / p.kv_heads)) * p.head_dim;
    uint d4 = p.head_dim / 4;
    device const float4 *q = (device const float4 *)(qkv + (ulong)i * p.qkv_dim + h * p.head_dim);
    device float *sc = scores + ((ulong)i * p.heads + h) * p.max_seq;
    if (tid < d4) q4[tid] = q[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 1. scores: one thread per cached position, a full q·k each
    float local_max = -INFINITY;
    for (uint s = tid; s < n; s += ATT_THREADS) {
        device const float4 *k = (device const float4 *)(kc + cache_slot(table, s) * p.kv_dim + kv);
        float d = 0;
        for (uint e = 0; e < d4; e++) d += dot(q4[e], k[e]);
        d *= p.scale;
        sc[s] = d;
        local_max = max(local_max, d);
    }
    local_max = simd_max(local_max);
    if (lane == 0) red[sg] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = -INFINITY;
    for (uint j = 0; j < ATT_SIMDS; j++) m = max(m, red[j]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. softmax numerators and their sum
    float local_sum = 0;
    for (uint s = tid; s < n; s += ATT_THREADS) {
        float e = exp(sc[s] - m);
        sc[s] = e;
        local_sum += e;
    }
    local_sum = simd_sum(local_sum);
    if (lane == 0) red[sg] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    float total = 0;
    for (uint j = 0; j < ATT_SIMDS; j++) total += red[j];

    // 3. out[d] = sum_s p_s v[s][d], positions split into ATT_THREADS/head_dim stripes
    uint stripes = ATT_THREADS / p.head_dim;
    uint d = tid % p.head_dim, stripe = tid / p.head_dim;
    float acc = 0;
    for (uint s = stripe; s < n; s += stripes) acc += sc[s] * vc[cache_slot(table, s) * p.kv_dim + kv + d];
    partial[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < p.head_dim) {
        float r = 0;
        for (uint j = 0; j < stripes; j++) r += partial[j * p.head_dim + tid];
        out[(ulong)i * p.q_dim + h * p.head_dim + tid] = r / total;
    }
}

struct SiluParams {
    uint inter;
    uint n;
};

// hidden = silu(gate) * up, with gate and up side by side in each row.
kernel void silu_mul(device const float *gu [[buffer(0)]], device float *hidden [[buffer(1)]],
                     constant SiluParams &p [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) return;
    uint i = gid / p.inter, j = gid % p.inter;
    float g = gu[(ulong)i * 2 * p.inter + j];
    float u = gu[(ulong)i * 2 * p.inter + p.inter + j];
    hidden[gid] = g / (1.0f + exp(-g)) * u;
}
