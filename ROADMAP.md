# Roadmap

Each milestone is self-contained: pick one, build it, benchmark it, and you
have a new result to talk about. "Done when" is the bar for calling it
finished; every one should end with a number in the README.

## Done (v0.1)

- [x] safetensors loader (mmap), JSON parser, byte-level BPE tokenizer
- [x] Qwen2 / Llama-family transformer on the CPU: NEON kernels, spin-then-park thread pool
- [x] bf16, 8-bit and 4-bit weights; perplexity benchmark per format
- [x] Metal GPU backend: whole forward pass in one command buffer
- [x] KV cache with prefix reuse across requests
- [x] Speculative decoding (greedy), verified identical to plain decoding
- [x] OpenAI-compatible server with streaming, plus a chat page
- [x] 18 tests, including GPU-equals-CPU and speculative-equals-greedy

## v0.2: Make the GPU fast

The M3 GPU has ~100 GB/s; decode currently uses 60–70 GB/s with bf16 and less
with 4-bit, where fixed per-layer costs dominate.

- **Kernel fusion.** Fold RMSNorm into the following matmul (each simdgroup
  can recompute the norm of an 896-float vector for free) and SiLU·up into
  the down projection. Nine dispatches per layer become six.
- **Prefill with `simdgroup_matrix`.** Apple GPUs have 8x8 matrix units;
  a tiled matmul for batches of 64+ tokens should be several times faster
  than the current mat-vec loop.
- **Flash-decoding attention.** Split long contexts across threadgroups and
  merge partial softmaxes, so a 4k-token context doesn't run on 14 threadgroups.
- **f16 KV cache.** Halves cache memory and attention bandwidth.

*Done when:* q4 decode beats 150 tok/s on an M3 and prefill beats 2,000 tok/s.

## v0.3: Better 4-bit quality

4-bit costs ~15% perplexity on a 0.5B model today.

- **k-quants:** 256-weight super-blocks with 6-bit sub-block scales (llama.cpp's Q4_K).
- **Activation-aware scaling (AWQ):** run calibration text, find the input
  channels with large activations, and scale them up before quantizing.
- Evaluate on a standard set (WikiText-2) as well as `bench/eval.txt`.

*Done when:* 4-bit perplexity is within 5% of bf16.

## v0.4: Serve many users at once

- **Continuous batching:** a scheduler that adds and removes sequences from
  the running batch every step, instead of one request at a time.
- **Paged KV cache (vLLM's PagedAttention):** fixed-size cache blocks with a
  block table per sequence, so memory isn't reserved for the longest possible reply.
- **Load test:** throughput and p50/p99 latency at 1, 4, 16 concurrent clients.

*Done when:* 16 concurrent clients get more than 5x the total tokens/second of one.

## v0.5: Smarter decoding

- **Speculative sampling** (not just greedy): the rejection-sampling rule
  that keeps the target's exact output distribution at any temperature.
- **Prompt-lookup decoding:** draft tokens by copying n-grams from the prompt.
  No draft model needed, and great for code editing and summarization.
- **Constrained decoding:** JSON-schema / grammar-guided output, masking
  invalid tokens at each step (`response_format` in the API).
- **Stop strings and logprobs** in the API.

*Done when:* `response_format: {type: "json_object"}` never returns invalid JSON.

## v0.6: More models

- **Llama 3.x:** `rope_scaling` type `llama3` (frequency-dependent scaling).
- **Qwen3:** per-head RMSNorm on q and k.
- **Mixture of experts (Qwen-MoE / Mixtral):** a router plus top-k expert
  matmuls. Only active experts' weights are read per token, which changes
  the bandwidth math in interesting ways.
- **LoRA adapters,** hot-swappable per request.

## v0.7: Other platforms

- **CUDA backend** (the same kernels in CUDA C, loaded through the driver API).
- **WebGPU / WASM:** run the 0.5B model in a browser tab; the tokenizer and
  CPU path are already dependency-free Rust.
- **x86 SIMD:** AVX2 versions of the four dot-product kernels.

## Always-on

- Fuzz the JSON parser, safetensors reader and tokenizer (`cargo fuzz`).
- Compare tokenizer output with Hugging Face's on a large corpus.
- Flamegraphs for every performance change, before and after.
