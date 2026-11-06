// Compute kernels for the Metal backend. Compiled at runtime from this
// source by the OS's built-in Metal compiler (no Xcode needed).

#include <metal_simdgroup_matrix>
#include <metal_stdlib>
using namespace metal;

// KV cache element type: the host prepends `#define KV_HALF` to store keys
// and values as f16 (half the memory and attention bandwidth).
#ifdef KV_HALF
typedef half kv_t;
#else
typedef float kv_t;
#endif
typedef vec<kv_t, 4> kv4_t;

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

// Loaders: the weights of `row` at columns k..k+4 and k+16..k+20, where k
// is 0, 4, 8 or 12 into a 32-column block (`cols` is a multiple of 32). A
// 4-bit block keeps those eight in the same four bytes, so one load and one
// scale serve all of them.
struct W8 {
    float4 lo, hi;
};

inline float4 bf16x4(ushort4 q) {
    return float4(as_type<float>(uint(q.x) << 16), as_type<float>(uint(q.y) << 16),
                  as_type<float>(uint(q.z) << 16), as_type<float>(uint(q.w) << 16));
}

struct LoadBf16 {
    device const ushort *w;
    uint cols;
    W8 operator()(uint row, uint k) const {
        device const ushort *p = w + (ulong)row * cols + k;
        return {bf16x4(*(device const ushort4 *)p), bf16x4(*(device const ushort4 *)(p + 16))};
    }
};

// Q8 blocks: 32 int8 weights and an f16 scale.
struct LoadQ8 {
    device const half *scales;
    device const char *qs;
    uint cols;
    W8 operator()(uint row, uint k) const {
        ulong e = (ulong)row * cols + k;
        float s = scales[e / 32];
        return {float4(*(device const char4 *)(qs + e)) * s, float4(*(device const char4 *)(qs + e + 16)) * s};
    }
};

// Q4 blocks: 16 bytes per 32 weights; byte j holds weight j (low nibble) and
// weight j+16 (high nibble), each stored offset by 8. A weight is
// (q - 8) * scale + offset. Scales and offsets are packed per 8 blocks of a
// row in five words (see kernels::quantize_q4_packed): f16 steps D and M,
// then a byte per block of scale (unsigned) and of offset (signed) in units
// of them.
inline float2 q4_scale(device const uint *packed, uint row, uint cols, uint block) {
    device const uint *w = packed + ((ulong)row * ((cols / 32 + 7) / 8) + block / 8) * 5;
    float2 step = float2(as_type<half2>(w[0]));
    uint j = block % 8;
    uint s = (w[1 + j / 4] >> (8 * (j % 4))) & 255;
    int o = as_type<char4>(w[3 + j / 4])[j % 4];
    return float2(step.x * float(s), step.y * float(o));
}

struct LoadQ4 {
    device const uint *scales;
    device const uchar *qs;
    uint cols;
    W8 operator()(uint row, uint k) const {
        ulong e = (ulong)row * cols + k;
        uchar4 c = *(device const uchar4 *)(qs + (e / 32) * 16 + k % 32);
        float2 dc = q4_scale(scales, row, cols, k / 32);
        return {(float4(c & uchar4(0xF)) - 8.0f) * dc.x + dc.y, (float4(c >> uchar4(4)) - 8.0f) * dc.x + dc.y};
    }
};

// Each lane's first column: per 256 columns, lane l takes block l / 4,
// columns (l % 4) * 4 and 16 more.
inline uint lane_k(uint lane) { return (lane / 4) * 32 + (lane % 4) * 4; }

template <typename L>
inline void matmul_impl(thread const L &load, device const float *x, device float *y, device const float *bias,
                        constant MatParams &p, uint row0, uint lane) {
    if (p.t == 1) { // decode: the hot path, kept lean
        float acc[R] = {0, 0, 0, 0};
        for (uint k = lane_k(lane); k < p.cols; k += 256) {
            float4 x0 = *(device const float4 *)(x + k), x1 = *(device const float4 *)(x + k + 16);
            for (uint r = 0; r < R; r++) {
                if (row0 + r < p.rows) {
                    W8 w = load(row0 + r, k);
                    acc[r] += dot(w.lo, x0) + dot(w.hi, x1);
                }
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
        for (uint k = lane_k(lane); k < p.cols; k += 256) {
            float4 x0[TB], x1[TB];
            for (uint j = 0; j < TB; j++) {
                if (j < nb) {
                    device const float *xj = x + (ulong)(base + j) * p.cols + k;
                    x0[j] = *(device const float4 *)xj;
                    x1[j] = *(device const float4 *)(xj + 16);
                }
            }
            for (uint r = 0; r < R; r++) {
                if (row0 + r >= p.rows) break;
                W8 w = load(row0 + r, k);
                for (uint j = 0; j < TB; j++) {
                    if (j < nb) acc[r * TB + j] += dot(w.lo, x0[j]) + dot(w.hi, x1[j]);
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

kernel void matmul_q8(device const half *scales [[buffer(0)]], device const char *qs [[buffer(1)]],
                      device const float *x [[buffer(2)]], device float *y [[buffer(3)]],
                      device const float *bias [[buffer(4)]], constant MatParams &p [[buffer(5)]],
                      uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    LoadQ8 load = {scales, qs, p.cols};
    matmul_impl(load, x, y, bias, p, (tg * SIMDS + sg) * R, lane);
}

kernel void matmul_q4(device const uint *scales [[buffer(0)]], device const uchar *qs [[buffer(1)]],
                      device const float *x [[buffer(2)]], device float *y [[buffer(3)]],
                      device const float *bias [[buffer(4)]], constant MatParams &p [[buffer(5)]],
                      uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    LoadQ4 load = {scales, qs, p.cols};
    matmul_impl(load, x, y, bias, p, (tg * SIMDS + sg) * R, lane);
}

// Tiled matrix multiply for batches (prefill, many sequences decoding at
// once), on the GPU's 8x8 matrix units. Each threadgroup computes a tile of
// GM_TOK tokens x GM_ROWS output rows. Per step of 32 columns (one
// quantization block), the input tile and the dequantized weight tile are
// staged in threadgroup memory once and shared by four simdgroups, each of
// which accumulates a 16x16 piece with simdgroup_multiply_accumulate.
// Weights are staged as half (bf16, and 8 or 4-bit values times their
// scale, fit in half's precision) and inputs as float, which keeps the whole
// tile in 8 KB: the output tile reuses the same memory once the loop is done,
// so more threadgroups fit on each core.
constant uint GM_TOK = 32;
constant uint TOK_BLOCKS = GM_TOK / 8;
constant uint GM_ROWS = 64;
constant uint GM_K = 32;

// Tile loaders: 16 weights of `row` starting at column `k` (a multiple of
// 16) into `d`, as halves.
struct TileBf16 {
    device const ushort *w;
    uint cols;
    void operator()(threadgroup half *d, uint row, uint k) const {
        device const ushort4 *s = (device const ushort4 *)(w + (ulong)row * cols + k);
        for (uint i = 0; i < 4; i++) {
            ushort4 v = s[i];
            *(threadgroup half4 *)(d + i * 4) = half4(as_type<float4>(uint4(v) << 16));
        }
    }
};

struct TileQ8 {
    device const half *scales;
    device const char *qs;
    uint cols;
    void operator()(threadgroup half *d, uint row, uint k) const {
        ulong e = (ulong)row * cols + k;
        float s = scales[e / 32];
        device const char4 *q = (device const char4 *)(qs + e);
        for (uint i = 0; i < 4; i++) {
            *(threadgroup half4 *)(d + i * 4) = half4(float4(q[i]) * s);
        }
    }
};

struct TileQ4 {
    device const uint *scales;
    device const uchar *qs;
    uint cols;
    void operator()(threadgroup half *d, uint row, uint k) const {
        ulong e = (ulong)row * cols + k;
        bool high = (k % 32) >= 16;
        device const uchar4 *q = (device const uchar4 *)(qs + (e / 32) * 16);
        float2 dc = q4_scale(scales, row, cols, k / 32);
        float s = dc.x, o = dc.y;
        for (uint i = 0; i < 4; i++) {
            uchar4 c = q[i];
            float4 f = ((high ? float4(c >> uchar4(4)) : float4(c & uchar4(0xF))) - 8.0f) * s + o;
            *(threadgroup half4 *)(d + i * 4) = half4(f);
        }
    }
};

// Which rows a GEMM reads and writes: a dense layer's tokens in order, or
// the tokens routed to one expert (gathered by moe_group), whose rows of
// the stacked expert matrices start at `base`.
struct DenseRows {
    device float *y;
    device const float *bias;
    uint t, rows, accumulate, has_bias;
    uint count() const { return t; }
    ulong input(uint i) const { return i; }
    uint weight(uint r) const { return r; }
    void store(uint i, uint r, float s) const {
        ulong idx = (ulong)i * rows + r;
        if (has_bias) s += bias[r];
        y[idx] = accumulate ? y[idx] + s : s;
    }
};

struct ExpertRows {
    device float *y;
    device const uint *slots; // this expert's slots, `n` of them
    uint n, rows, base, k, per_slot;
    uint count() const { return n; }
    ulong input(uint i) const { return per_slot ? slots[i] : slots[i] / k; }
    uint weight(uint r) const { return base + r; }
    void store(uint i, uint r, float s) const { y[(ulong)slots[i] * rows + r] = s; }
};

template <typename L, typename G>
inline void gemm_impl(thread const L &tile, thread const G &g, device const float *x, uint rows, uint cols,
                      uint2 tg, uint tid, uint sg, threadgroup float *shared) {
    uint row0 = tg.x * GM_ROWS, tok0 = tg.y * GM_TOK, n = g.count();
    if (tok0 >= n) return;
    threadgroup float *xs = shared;
    threadgroup half *ws = (threadgroup half *)(shared + GM_TOK * GM_K);
    threadgroup float *os = shared;
    // Each simdgroup owns 16 output rows; one 8x8 accumulator per
    // (8-token block, 8-row block). Every dequantized weight tile is used
    // for all GM_TOK tokens before the next is loaded.
    simdgroup_float8x8 acc[TOK_BLOCKS * 2];
    for (uint i = 0; i < TOK_BLOCKS * 2; i++) acc[i] = simdgroup_float8x8(0);

    for (uint k0 = 0; k0 < cols; k0 += GM_K) {
        // inputs: GM_TOK tokens x 32 columns
        for (uint e = tid; e < GM_TOK * GM_K / 4; e += 128) {
            uint t = e / (GM_K / 4), c = (e % (GM_K / 4)) * 4;
            float4 v = tok0 + t < n ? *(device const float4 *)(x + g.input(tok0 + t) * cols + k0 + c) : float4(0);
            xs[t * GM_K + c + 0] = v.x;
            xs[t * GM_K + c + 1] = v.y;
            xs[t * GM_K + c + 2] = v.z;
            xs[t * GM_K + c + 3] = v.w;
        }
        // weights: 64 rows x 32 columns, 16 per thread
        {
            uint r = tid / 2, half_k = (tid % 2) * 16;
            threadgroup half *d = ws + r * GM_K + half_k;
            if (row0 + r < rows) {
                tile(d, g.weight(row0 + r), k0 + half_k);
            } else {
                for (uint i = 0; i < 16; i++) d[i] = 0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < GM_K; kk += 8) {
            simdgroup_half8x8 b0, b1;
            simdgroup_load(b0, ws + (sg * 16) * GM_K + kk, GM_K, ulong2(0, 0), true);
            simdgroup_load(b1, ws + (sg * 16 + 8) * GM_K + kk, GM_K, ulong2(0, 0), true);
            for (uint tb = 0; tb < TOK_BLOCKS; tb++) {
                simdgroup_float8x8 a;
                simdgroup_load(a, xs + tb * 8 * GM_K + kk, GM_K);
                simdgroup_multiply_accumulate(acc[tb * 2], a, b0, acc[tb * 2]);
                simdgroup_multiply_accumulate(acc[tb * 2 + 1], a, b1, acc[tb * 2 + 1]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint tb = 0; tb < TOK_BLOCKS; tb++) {
        simdgroup_store(acc[tb * 2], os + tb * 8 * GM_ROWS + sg * 16, GM_ROWS);
        simdgroup_store(acc[tb * 2 + 1], os + tb * 8 * GM_ROWS + sg * 16 + 8, GM_ROWS);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < GM_TOK * GM_ROWS; i += 128) {
        uint t = i / GM_ROWS, r = i % GM_ROWS;
        if (tok0 + t < n && row0 + r < rows) g.store(tok0 + t, row0 + r, os[i]);
    }
}

kernel void gemm_bf16(device const ushort *w [[buffer(0)]], device const float *x [[buffer(1)]],
                      device float *y [[buffer(2)]], device const float *bias [[buffer(3)]],
                      constant MatParams &p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float shared[GM_TOK * GM_ROWS];
    TileBf16 tile = {w, p.cols};
    DenseRows g = {y, bias, p.t, p.rows, p.accumulate, p.has_bias};
    gemm_impl(tile, g, x, p.rows, p.cols, tg, tid, sg, shared);
}

kernel void gemm_q8(device const half *scales [[buffer(0)]], device const char *qs [[buffer(1)]],
                    device const float *x [[buffer(2)]], device float *y [[buffer(3)]],
                    device const float *bias [[buffer(4)]], constant MatParams &p [[buffer(5)]],
                    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float shared[GM_TOK * GM_ROWS];
    TileQ8 tile = {scales, qs, p.cols};
    DenseRows g = {y, bias, p.t, p.rows, p.accumulate, p.has_bias};
    gemm_impl(tile, g, x, p.rows, p.cols, tg, tid, sg, shared);
}

kernel void gemm_q4(device const uint *scales [[buffer(0)]], device const uchar *qs [[buffer(1)]],
                    device const float *x [[buffer(2)]], device float *y [[buffer(3)]],
                    device const float *bias [[buffer(4)]], constant MatParams &p [[buffer(5)]],
                    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float shared[GM_TOK * GM_ROWS];
    TileQ4 tile = {scales, qs, p.cols};
    DenseRows g = {y, bias, p.t, p.rows, p.accumulate, p.has_bias};
    gemm_impl(tile, g, x, p.rows, p.cols, tg, tid, sg, shared);
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
kernel void rope_store(device float *qkv [[buffer(0)]], device kv_t *kc [[buffer(1)]], device kv_t *vc [[buffer(2)]],
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
        kc[at + d] = kv_t(a * c - b * s);
        kc[at + d + half_dim] = kv_t(b * c + a * s);
        vc[at + d] = kv_t(v[d]);
        vc[at + d + half_dim] = kv_t(v[d + half_dim]);
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

constant uint ATT_THREADS = 1024;
constant uint ATT_SIMDS = ATT_THREADS / 32;

inline ulong cache_slot(device const uint *table, uint s) {
    return (ulong)table[s / BLOCK] * BLOCK + s % BLOCK;
}

// One threadgroup per (head, token), for tokens decoding alone (runs of a
// sequence's tokens go to attention_prefill). Each token attends to its own
// sequence's positions 0..=pos, found through the sequence's block table.
// Scores go to a scratch buffer so any context length fits; three passes:
// q·k (a thread per position), softmax, weighted sum of v.
kernel void attention(device const float *qkv [[buffer(0)]], device const kv_t *kc [[buffer(1)]],
                      device const kv_t *vc [[buffer(2)]], device float *scores [[buffer(3)]],
                      device float *out [[buffer(4)]], device const uint *positions [[buffer(5)]],
                      device const uint *table_of [[buffer(6)]], device const uint *tables [[buffer(7)]],
                      device const uint *tokens [[buffer(8)]], constant AttnParams &p [[buffer(9)]],
                      uint2 tg [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[ATT_SIMDS];
    threadgroup float4 partial[ATT_THREADS];
    threadgroup float4 q4[32]; // head_dim <= 128
    uint h = tg.x, i = tokens[tg.y];
    uint n = positions[i] + 1;
    device const uint *table = tables + table_of[i];
    uint kv = (h / (p.heads / p.kv_heads)) * p.head_dim;
    uint d4 = p.head_dim / 4;
    device const float4 *q = (device const float4 *)(qkv + (ulong)i * p.qkv_dim + h * p.head_dim);
    device float *sc = scores + ((ulong)tg.y * p.heads + h) * p.max_seq;
    if (tid < d4) q4[tid] = q[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 1. scores: one thread per cached position, a full q·k each
    float local_max = -INFINITY;
    for (uint s = tid; s < n; s += ATT_THREADS) {
        device const kv4_t *k = (device const kv4_t *)(kc + cache_slot(table, s) * p.kv_dim + kv);
        float d = 0;
        for (uint e = 0; e < d4; e++) d += dot(q4[e], float4(k[e]));
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

    // 3. out = sum_s p_s v[s]: a thread takes 4 dims of every
    // (ATT_THREADS / d4)th position
    uint stripes = ATT_THREADS / d4;
    uint c = tid % d4, stripe = tid / d4;
    float4 acc = 0;
    for (uint s = stripe; s < n; s += stripes)
        acc += sc[s] * float4(*(device const kv4_t *)(vc + cache_slot(table, s) * p.kv_dim + kv + c * 4));
    partial[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < d4) {
        float4 r = 0;
        for (uint j = 0; j < stripes; j++) r += partial[j * d4 + tid];
        *(device float4 *)(out + (ulong)i * p.q_dim + h * p.head_dim + tid * 4) = r / total;
    }
}

// Attention for runs of consecutive tokens of one sequence (prefill), in
// the style of FlashAttention. A threadgroup takes one head and up to FA_Q
// tokens, 8 per simdgroup. It walks the sequence's cache FA_KV positions
// at a time: their keys and values are loaded into threadgroup memory once
// for all the queries, scores come from the 8x8 matrix units, and a running
// max and sum per query keep the softmax exact without ever storing a full
// row of scores. The output is rescaled as the max grows, by multiplying
// with a diagonal matrix.
constant uint FA_SIMDS = 4;
constant uint FA_Q = FA_SIMDS * 8;
#ifndef HEAD_DIM
#define HEAD_DIM 128
#endif
// The head size, fixed when the kernels are compiled for a model, so the
// query and output matrices and the staging memory are no bigger than it.
// Threadgroup memory decides how many threadgroups share a core, so it is
// kept to about 11 KB: for 128-wide heads by taking half as many cached
// positions per step and staging queries and outputs in two rounds.
constant uint FA_MAX_DIM = HEAD_DIM;
constant uint FA_ROUNDS = HEAD_DIM > 64 ? 2 : 1;
// cached positions per step: 8 KB of keys and values
constant uint FA_KV = 8192 / (2 * HEAD_DIM * sizeof(kv_t));
typedef simdgroup_matrix<kv_t, 8, 8> kv8x8;

kernel void attention_prefill(device const float *qkv [[buffer(0)]], device const kv_t *kc [[buffer(1)]],
                              device const kv_t *vc [[buffer(2)]], device float *out [[buffer(3)]],
                              device const uint *positions [[buffer(4)]], device const uint *table_of [[buffer(5)]],
                              device const uint *tables [[buffer(6)]], device const uint2 *tiles [[buffer(7)]],
                              constant AttnParams &p [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]]) {
    // Queries on the way in, keys and values per block, outputs at the end.
    threadgroup float shared[FA_Q * FA_MAX_DIM / FA_ROUNDS];
    threadgroup float scores[FA_SIMDS][8 * FA_KV];
    threadgroup float diag[FA_SIMDS][64];

    uint h = tg.x, first = tiles[tg.y].x, count = tiles[tg.y].y;
    const uint D = HEAD_DIM, d4 = D / 4;
    uint kv = (h / (p.heads / p.kv_heads)) * D;
    device const uint *table = tables + table_of[first];
    uint n = positions[first + count - 1] + 1; // cached positions any query sees

    // This simdgroup's 8 queries, pre-scaled; rows past the run are zero.
    // Staged through `shared` FA_SIMDS / FA_ROUNDS simdgroups at a time.
    uint q0 = sg * 8;
    const uint per_round = FA_SIMDS / FA_ROUNDS;
    threadgroup float *qs = shared + (sg % per_round) * 8 * D;
    for (uint e = lane; e < 64; e += 32) diag[sg][e] = 0;
    // Queries in half precision (like the keys), which halves the registers
    // they take; outputs stay in float.
    simdgroup_half8x8 q[FA_MAX_DIM / 8];
    simdgroup_float8x8 o[FA_MAX_DIM / 8];
    threadgroup half *qh = (threadgroup half *)shared + sg * 8 * D;
    for (uint e = lane; e < 8 * d4; e += 32) {
        uint r = e / d4, c = (e % d4) * 4;
        float4 v = q0 + r < count ? *(device const float4 *)(qkv + (ulong)(first + q0 + r) * p.qkv_dim + h * D + c)
                                  : float4(0);
        *(threadgroup half4 *)(qh + r * D + c) = half4(v * p.scale);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < D / 8; j++) {
        simdgroup_load(q[j], qh + j * 8, D);
        o[j] = simdgroup_float8x8(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Lanes 4r..4r+3 own query row r's softmax state and FA_KV/4 of its
    // columns. Rows past the run copy the last real query.
    uint row = lane / 4, col = (lane % 4) * (FA_KV / 4);
    uint qpos = positions[first + min(q0 + row, count - 1)];
    float m = -INFINITY, l = 0;

    threadgroup kv_t *ks = (threadgroup kv_t *)shared;
    threadgroup kv_t *vs = ks + FA_KV * D;
    for (uint b0 = 0; b0 < n; b0 += FA_KV) {
        // keys and values of positions b0..b0+FA_KV, zero past n
        for (uint e = tid; e < FA_KV * d4; e += FA_SIMDS * 32) {
            uint j = e / d4, c = (e % d4) * 4;
            bool ok = b0 + j < n;
            ulong at = (ok ? cache_slot(table, b0 + j) : 0) * p.kv_dim + kv + c;
            *(threadgroup kv4_t *)(ks + j * D + c) = ok ? *(device const kv4_t *)(kc + at) : kv4_t(0);
            *(threadgroup kv4_t *)(vs + j * D + c) = ok ? *(device const kv4_t *)(vc + at) : kv4_t(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // scores = q · kᵀ, 8 queries x FA_KV positions
        threadgroup float *sc = scores[sg];
        for (uint c = 0; c < FA_KV / 8; c++) {
            simdgroup_float8x8 acc = simdgroup_float8x8(0);
            for (uint j = 0; j < D / 8; j++) {
                kv8x8 kt;
                simdgroup_load(kt, ks + c * 8 * D + j * 8, D, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, q[j], kt, acc);
            }
            simdgroup_store(acc, sc + c * 8, FA_KV);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // running softmax: mask the future, rescale by the new max
        float v[FA_KV / 4], top = -INFINITY;
        for (uint j = 0; j < FA_KV / 4; j++) {
            v[j] = b0 + col + j <= qpos ? sc[row * FA_KV + col + j] : -INFINITY;
            top = max(top, v[j]);
        }
        float mx = max(m, quad_max(top));
        float sum = 0;
        for (uint j = 0; j < FA_KV / 4; j++) {
            float e = exp(v[j] - mx);
            sc[row * FA_KV + col + j] = e;
            sum += e;
        }
        float scale = exp(m - mx);
        l = l * scale + quad_sum(sum);
        m = mx;
        if (lane % 4 == 0) diag[sg][row * 9] = scale;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // o = diag(scale) · o + p · v
        simdgroup_float8x8 dm;
        simdgroup_load(dm, diag[sg], 8);
        for (uint j = 0; j < D / 8; j++) {
            simdgroup_float8x8 t;
            simdgroup_multiply(t, dm, o[j]);
            o[j] = t;
        }
        for (uint c = 0; c < FA_KV / 8; c++) {
            simdgroup_float8x8 pm;
            simdgroup_load(pm, sc + c * 8, FA_KV);
            for (uint j = 0; j < D / 8; j++) {
                kv8x8 vm;
                simdgroup_load(vm, vs + c * 8 * D + j * 8, D);
                simdgroup_multiply_accumulate(o[j], pm, vm, o[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // o / l, staged so only the run's rows are written (in rounds, as the
    // queries came in)
    if (lane % 4 == 0) diag[sg][row * 9] = 1.0f / l;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 dm;
    simdgroup_load(dm, diag[sg], 8);
    for (uint round = 0; round < FA_ROUNDS; round++) {
        if (sg / per_round == round) {
            for (uint j = 0; j < D / 8; j++) {
                simdgroup_float8x8 t;
                simdgroup_multiply(t, dm, o[j]);
                simdgroup_store(t, qs + j * 8, D);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < 8 * d4; e += 32) {
                uint r = e / d4, c = (e % d4) * 4;
                if (q0 + r < count)
                    *(device float4 *)(out + (ulong)(first + q0 + r) * p.q_dim + h * D + c) =
                        *(threadgroup float4 *)(qs + r * D + c);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Mixture of experts. Each token's k chosen experts give it k "slots"
// (slot = token * k + j); the experts' matrices are stacked, expert e owning
// rows e * rows... A slot's product reads only its expert's rows, so a token
// costs k small matrix-vector products and the other experts are never read.
struct MoeParams {
    uint rows; // per expert
    uint cols;
    uint k;
    uint per_slot; // inputs are one row per slot (else one per token)
    uint t;
    uint experts;
};

template <typename L>
inline void moe_impl(thread const L &load, device const float *x, device float *y, device const uint *ids,
                     constant MoeParams &p, uint2 tg, uint sg, uint lane) {
    uint slot = tg.y, row0 = (tg.x * SIMDS + sg) * R;
    uint base = ids[slot] * p.rows;
    device const float *xs = x + (ulong)(p.per_slot ? slot : slot / p.k) * p.cols;
    float acc[R] = {0, 0, 0, 0};
    for (uint k = lane_k(lane); k < p.cols; k += 256) {
        float4 x0 = *(device const float4 *)(xs + k), x1 = *(device const float4 *)(xs + k + 16);
        for (uint r = 0; r < R; r++) {
            if (row0 + r < p.rows) {
                W8 w = load(base + row0 + r, k);
                acc[r] += dot(w.lo, x0) + dot(w.hi, x1);
            }
        }
    }
    for (uint r = 0; r < R; r++) {
        float s = simd_sum(acc[r]);
        if (lane == 0 && row0 + r < p.rows) y[(ulong)slot * p.rows + row0 + r] = s;
    }
}

kernel void moe_bf16(device const ushort *w [[buffer(0)]], device const float *x [[buffer(1)]],
                     device float *y [[buffer(2)]], device const uint *ids [[buffer(3)]],
                     constant MoeParams &p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    LoadBf16 load = {w, p.cols};
    moe_impl(load, x, y, ids, p, tg, sg, lane);
}

kernel void moe_q8(device const half *scales [[buffer(0)]], device const char *qs [[buffer(1)]],
                   device const float *x [[buffer(2)]], device float *y [[buffer(3)]],
                   device const uint *ids [[buffer(4)]], constant MoeParams &p [[buffer(5)]],
                   uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]]) {
    LoadQ8 load = {scales, qs, p.cols};
    moe_impl(load, x, y, ids, p, tg, sg, lane);
}

kernel void moe_q4(device const uint *scales [[buffer(0)]], device const uchar *qs [[buffer(1)]],
                   device const float *x [[buffer(2)]], device float *y [[buffer(3)]],
                   device const uint *ids [[buffer(4)]], constant MoeParams &p [[buffer(5)]],
                   uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]]) {
    LoadQ4 load = {scales, qs, p.cols};
    moe_impl(load, x, y, ids, p, tg, sg, lane);
}

// Lists each expert's slots, for the expert GEMMs (prefill): a thread per
// expert scans the batch's choices.
kernel void moe_group(device const uint *ids [[buffer(0)]], device uint *counts [[buffer(1)]],
                      device uint *lists [[buffer(2)]], constant MoeParams &p [[buffer(3)]],
                      uint e [[thread_position_in_grid]]) {
    if (e >= p.experts) return;
    uint n = 0;
    for (uint s = 0; s < p.t * p.k; s++) {
        if (ids[s] == e) lists[(ulong)e * p.t + n++] = s;
    }
    counts[e] = n;
}

// Weight buffers first (padded to three), then inputs, outputs, the
// experts' slot counts and lists.
#define MOE_GEMM(NAME, TILE, ARGS, INIT)                                                                   \
    kernel void NAME(ARGS, device const float *x [[buffer(3)]], device float *y [[buffer(4)]],             \
                     device const uint *counts [[buffer(5)]], device const uint *lists [[buffer(6)]],       \
                     constant MoeParams &p [[buffer(7)]], uint3 tg [[threadgroup_position_in_grid]],        \
                     uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) { \
        threadgroup float shared[GM_TOK * GM_ROWS];                                                         \
        TILE tile = INIT;                                                                                   \
        uint e = tg.z;                                                                                      \
        ExpertRows g = {y, lists + (ulong)e * p.t, counts[e], p.rows, e * p.rows, p.k, p.per_slot};         \
        gemm_impl(tile, g, x, p.rows, p.cols, tg.xy, tid, sg, shared);                                      \
    }
#define COMMA ,
MOE_GEMM(moe_gemm_bf16, TileBf16,
         device const ushort *w [[buffer(0)]] COMMA device const float *unused [[buffer(1)]] COMMA
             device const float *unused2 [[buffer(2)]],
         (TileBf16{w, p.cols}))
MOE_GEMM(moe_gemm_q8, TileQ8,
         device const half *scales [[buffer(0)]] COMMA device const char *qs [[buffer(1)]] COMMA
             device const float *unused [[buffer(2)]],
         (TileQ8{scales, qs, p.cols}))
MOE_GEMM(moe_gemm_q4, TileQ4,
         device const uint *scales [[buffer(0)]] COMMA device const uchar *qs [[buffer(1)]] COMMA
             device const float *unused [[buffer(2)]],
         (TileQ4{scales, qs, p.cols}))

// A thread per token: its k highest router scores (ties to the lower
// expert, as on the CPU), weighted by a softmax over those k.
kernel void moe_route(device const float *scores [[buffer(0)]], device uint *ids [[buffer(1)]],
                      device float *gates [[buffer(2)]], constant MoeParams &p [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= p.t) return;
    device const float *s = scores + (ulong)i * p.experts;
    ulong taken = 0;
    float top[16];
    for (uint j = 0; j < p.k; j++) {
        float best = -INFINITY;
        uint at = 0;
        for (uint e = 0; e < p.experts; e++) {
            if (!((taken >> e) & 1) && s[e] > best) {
                best = s[e];
                at = e;
            }
        }
        taken |= 1ul << at;
        top[j] = best;
        ids[i * p.k + j] = at;
    }
    float sum = 0;
    for (uint j = 0; j < p.k; j++) sum += exp(top[j] - top[0]);
    for (uint j = 0; j < p.k; j++) gates[i * p.k + j] = exp(top[j] - top[0]) / sum;
}

// x[token] += sum over its slots of gate * expert output.
kernel void moe_combine(device float *x [[buffer(0)]], device const float *out [[buffer(1)]],
                        device const float *gates [[buffer(2)]], constant MoeParams &p [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= p.t * p.rows) return;
    uint i = gid / p.rows, d = gid % p.rows;
    float s = 0;
    for (uint j = 0; j < p.k; j++) s += gates[i * p.k + j] * out[((ulong)i * p.k + j) * p.rows + d];
    x[gid] += s;
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

// ---- LoRA adapters: y += B (A x) per token, each with its own adapter ----
//
// After Punica's SGMV: the batch's tokens are grouped by adapter on the CPU
// (`order`), and cut into tiles of up to LORA_TILE tokens that share one
// (`tiles`: first index into `order`, count, adapter). Each row of A and B
// is then read once per tile instead of once per token, and one dispatch
// serves a batch that mixes adapters. Tokens without an adapter are in no
// tile. A holds `segs * rank` rows per adapter (q, k and v each have their
// own); output row r reads the `rank` values of its segment.

#define LORA_TILE 8

struct LoraParams {
    uint rows;
    uint cols;
    uint rank; // a multiple of 4
    uint segs;
    uint end0; // where segments 0 and 1 end
    uint end1;
};

// h[token][j] = A_adapter[j] . x[token]: one threadgroup per (j, tile), its
// 256 threads splitting the columns.
kernel void lora_shrink(device const half *A [[buffer(0)]], device const float *x [[buffer(1)]],
                        device float *h [[buffer(2)]], device const uint *order [[buffer(3)]],
                        device const uint *tiles [[buffer(4)]], constant LoraParams &p [[buffer(5)]],
                        uint2 tg [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[8][LORA_TILE];
    uint inner = p.segs * p.rank, j = tg.x;
    device const uint *tile = tiles + tg.y * 3;
    uint first = tile[0], n = tile[1], id = tile[2];
    device const half4 *a = (device const half4 *)(A + ((ulong)(id - 1) * inner + j) * p.cols);
    device const float4 *xs[LORA_TILE];
    float s[LORA_TILE];
    for (uint k = 0; k < LORA_TILE; k++) {
        xs[k] = (device const float4 *)(x + (ulong)order[first + min(k, n - 1)] * p.cols);
        s[k] = 0.0f;
    }
    for (uint c = tid; c < p.cols / 4; c += 256) {
        float4 av = float4(a[c]);
        for (uint k = 0; k < LORA_TILE; k++) {
            if (k < n) s[k] += dot(av, xs[k][c]);
        }
    }
    for (uint k = 0; k < LORA_TILE; k++) {
        float v = simd_sum(s[k]);
        if (lane == 0) part[sg][k] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < n) {
        float v = 0.0f;
        for (uint g = 0; g < 8; g++) v += part[g][tid];
        h[(ulong)order[first + tid] * inner + j] = v;
    }
}

// y[token][r] += B_adapter[r] . h[token][segment of r]: one thread per
// (output row, tile).
kernel void lora_expand(device const half *B [[buffer(0)]], device const float *h [[buffer(1)]],
                        device float *y [[buffer(2)]], device const uint *order [[buffer(3)]],
                        device const uint *tiles [[buffer(4)]], constant LoraParams &p [[buffer(5)]],
                        uint2 tg [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]]) {
    uint r = tg.x * 256 + tid;
    if (r >= p.rows) return;
    device const uint *tile = tiles + tg.y * 3;
    uint first = tile[0], n = tile[1], id = tile[2];
    uint seg = r < p.end0 ? 0 : (r < p.end1 ? 1 : 2);
    uint inner = p.segs * p.rank;
    device const half4 *b = (device const half4 *)(B + ((ulong)(id - 1) * p.rows + r) * p.rank);
    device const float4 *hs[LORA_TILE];
    float s[LORA_TILE];
    for (uint k = 0; k < LORA_TILE; k++) {
        hs[k] = (device const float4 *)(h + (ulong)order[first + min(k, n - 1)] * inner + seg * p.rank);
        s[k] = 0.0f;
    }
    for (uint j = 0; j < p.rank / 4; j++) {
        float4 bv = float4(b[j]);
        for (uint k = 0; k < LORA_TILE; k++) {
            if (k < n) s[k] += dot(bv, hs[k][j]);
        }
    }
    for (uint k = 0; k < n; k++) y[(ulong)order[first + k] * p.rows + r] += s[k];
}
