# Tachyon

**An LLM inference engine written from scratch in Rust, with hand-written ARM NEON and Metal GPU kernels, and zero dependencies.**

Tachyon loads real Hugging Face checkpoints (Qwen2.5, Llama-style models) and runs them on a MacBook's CPU or GPU. It handles 8-bit and 4-bit quantization and speculative decoding, and serves an OpenAI-compatible API. Everything between the weights file and the HTTP response is in this repository: the JSON parser, safetensors reader, BPE tokenizer, thread pool, SIMD kernels, GPU shaders, sampler and web server. `Cargo.toml` has no dependencies.

![Chat page served by Tachyon](docs/chat-ui.jpg)

## Results

Qwen2.5-0.5B-Instruct on a MacBook Air M3 (8-core CPU, 10-core GPU, 16 GB):

| backend | weights | prefill | decode | perplexity |
|---|---:|---:|---:|---:|
| CPU, bf16 | 988 MB | 138 tok/s | 60.7 tok/s | 17.94 |
| CPU, 8-bit | 556 MB | 225 tok/s | 82.7 tok/s | 18.08 |
| CPU, 4-bit | 377 MB | 190 tok/s | 93.4 tok/s | 20.65 |
| GPU (Metal), bf16 | 988 MB | 565 tok/s | 61.6 tok/s | 17.94 |
| GPU (Metal), 8-bit | 556 MB | 543 tok/s | 88.7 tok/s | 17.99 |
| GPU (Metal), 4-bit | 377 MB | 404 tok/s | 96.3 tok/s | 20.71 |

From `tachyon bench`: best of 3 runs, prefill 256 tokens, decode 128, perplexity over 979 tokens of [`bench/eval.txt`](bench/eval.txt). Measured with other apps running, which moves decode speed by about ±15% between runs; the fastest runs seen were 101 tok/s (CPU) and 112 tok/s (GPU) for 8-bit.

- **Decode** is one token at a time, which is how chat replies are generated. Each token reads every weight once, so speed is set by memory bandwidth, and smaller weights mean more tokens per second.
- **Prefill** processes the prompt in batches of 64 tokens. The GPU is 2–3x faster here because batching turns it into a compute problem.
- **Perplexity** measures how surprised the model is by a held-out text (lower is better). It is the standard way to see what quantization costs: 8-bit is nearly free, 4-bit costs about 15% on a model this small. The CPU and GPU give identical bf16 perplexity because they compute the same function.

### Speculative decoding: up to 2.2x faster

Qwen2.5-**1.5B** (bf16) as the target, drafted by the 0.5B model in 4-bit, greedy, 200 tokens, on the GPU:

| prompt | drafted per step | drafts accepted | tokens per target pass | speed | vs. plain |
|---|---:|---:|---:|---:|---:|
| code | 4 | 81% | 4.2 | 41.0 tok/s | **1.89x** |
| code | 6 | 80% | 5.6 | 50.5 tok/s | **2.19x** |
| prose | 4 | 39% | 2.5 | 29.0 tok/s | 1.26x |

Plain decoding of the 1.5B model runs at 21.7 tok/s (23.1 on the prose prompt). In every run the output is token-for-token identical to plain greedy decoding, and a test checks this. Code is predictable, so the small model guesses most tokens right; a story is not.

On the CPU, speculation is slightly *slower* (0.9x). Checking 5 tokens in one pass costs 5x the arithmetic, and the CPU runs out of compute long before memory bandwidth. The GPU has compute to spare, so checking 5 tokens costs about the same as checking one. Where verification is cheap decides whether speculation helps.

## What's inside

| | |
|---|---|
| [`safetensors.rs`](src/safetensors.rs) | Memory-maps checkpoints (via raw `mmap`), parses the header with [`json.rs`](src/json.rs), handles sharded files |
| [`tokenizer.rs`](src/tokenizer.rs) | Byte-level BPE from `tokenizer.json`. The pre-tokenizer regex is reimplemented by hand (no regex engine); control tokens can't be forged from user text |
| [`model.rs`](src/model.rs) | Transformer forward pass: RMSNorm, grouped-query attention, rotary embeddings, SiLU-gated MLP. Q/K/V and gate/up are stacked into single matrices at load time, so each layer does 4 matmuls instead of 7 |
| [`kernels.rs`](src/kernels.rs) | NEON dot products for bf16 (widened with a zip), int8×int8 and int4×int8 (`vmull_s8` + pairwise accumulate), plus the quantizers |
| [`pool.rs`](src/pool.rs) | A thread pool for ~150 tiny jobs per token. Workers spin briefly before parking, so a dispatch starts in about a microsecond |
| [`gpu/`](src/gpu) | Metal backend: [`kernels.metal`](src/gpu/kernels.metal) (matmul for each format, attention, RoPE, norms) and an [Objective-C bridge](src/gpu/shim.m) compiled by `build.rs` with plain `clang` |
| [`generate.rs`](src/generate.rs) | Prefill and decode loop, with KV-cache prefix reuse |
| [`speculative.rs`](src/speculative.rs) | Draft-then-verify decoding with KV-cache rollback |
| [`server.rs`](src/server.rs) | HTTP/1.1 on `std::net`: `/v1/chat/completions` with server-sent-event streaming, `/v1/models`, and a chat page |

### Design notes

- **Decode is a memory-bandwidth problem.** A 0.5B model in bf16 is ~1 GB, all of which is read for every token, so at 100 GB/s the ceiling is ~100 tok/s whatever the arithmetic. Most of the design follows from that: quantize the weights, read each weight once per batch of tokens, and keep everything else out of the way.
- **Quantization.** Weights are stored in blocks of 32 sharing one scale (llama.cpp's Q8_0/Q4_0 layouts). On the CPU, activations are quantized to int8 on the fly so the inner loop is integer multiply-accumulate. 4-bit models keep the output layer at 8 bits, as llama.cpp does: that buys 4% perplexity for 68 MB.
- **The thread pool was the biggest CPU win.** With one thread per core, any background process preempts a spinning worker, and every matmul then waits for it. Leaving two efficiency cores free more than doubled decode speed (50 → 114 tok/s for 8-bit).
- **GPU matmul.** Each simdgroup computes four output rows, so each input value is loaded once and reused from registers. With 4-bit weights the input vector is 8x more bytes than the weights, so without this reuse the kernel is bound by input reads. This took 8-bit decode from 89 to 105 tok/s and prefill from ~400 to ~570 tok/s.
- **Unified memory.** On Apple Silicon the CPU writes embeddings and RoPE angles straight into GPU buffers and reads logits in place. A whole forward pass is one command buffer: ~220 dispatches, encoded in ~70 µs.
- **Prefix caching.** The KV cache remembers which tokens it holds. A new request that starts the same way (same system prompt, earlier chat turns) only computes the new part.

### How it's verified

`cargo test --release` runs 18 tests. Unit tests cover each kernel against a scalar reference, the quantizers, JSON, the tokenizer's splitting rules and the thread pool. End-to-end tests against the real model check that:

- the GPU produces exactly the same greedy tokens as the CPU;
- a batched forward pass gives the same logits as feeding tokens one by one;
- reusing the KV cache gives the same output as recomputing it;
- speculative decoding gives exactly the same output as plain greedy decoding;
- the tokenizer round-trips code, Unicode and emoji, and won't turn `<|im_end|>` typed by a user into a control token;
- the model answers "What is the capital of France?" with Paris.

## Try it

Needs Rust and, for the GPU backend, macOS on Apple Silicon (no Xcode needed, just the command line tools).

```bash
# 1. Get a model (~1 GB, Apache-2.0)
mkdir -p models/qwen2.5-0.5b-instruct && cd models/qwen2.5-0.5b-instruct
for f in config.json generation_config.json tokenizer.json model.safetensors; do
  curl -LO "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/$f"
done
cd ../..

# 2. Build
cargo build --release
```

```bash
./target/release/tachyon run --gpu --prompt "Explain recursion to a five-year-old."
./target/release/tachyon chat --quant q4          # multi-turn chat in the terminal
./target/release/tachyon serve --gpu              # chat page at http://localhost:8080
./target/release/tachyon bench                    # the table above
./target/release/tachyon speculate --gpu --model models/qwen2.5-1.5b-instruct \
    --quant bf16 --draft-model models/qwen2.5-0.5b-instruct --draft q4 --prompt "..."
```

Any OpenAI client works against `serve`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
stream = client.chat.completions.create(
    model="qwen2.5-0.5b-instruct",
    messages=[{"role": "user", "content": "Write a haiku about Rust."}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

Run `tachyon help` for every option (`--quant bf16|q8|q4`, `--temp`, `--top-k`, `--top-p`, `--ctx`, `--threads`, ...).

## Limitations

- One request at a time against one KV cache (continuous batching is next on the [roadmap](ROADMAP.md)).
- Speculative decoding is greedy only; sampling would need the rejection-sampling rule.
- Model support: Qwen2 / Llama / Mistral-style architectures without `rope_scaling`, so not yet Llama 3.1+ or Qwen3.
- The tokenizer skips NFC normalization and treats Unicode letters slightly more broadly than `\p{L}`; this matters only for some scripts with combining marks.

## Roadmap

See [ROADMAP.md](ROADMAP.md): kernel fusion and `simdgroup_matrix` prefill on the GPU, k-quants, continuous batching with a paged KV cache, constrained JSON output, MoE models, CUDA and WebGPU backends.

## License

MIT
