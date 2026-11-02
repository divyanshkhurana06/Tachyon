# Roadmap

Each milestone is self-contained: pick one, build it, benchmark it, and you
have a new result to talk about. "Done when" is the bar for calling it
finished; every one should end with a number in the README.

## Done

### v0.1: a working engine

- [x] safetensors loader (mmap), JSON parser, byte-level BPE tokenizer
- [x] Qwen2 / Llama-family transformer on the CPU: NEON kernels, spin-then-park thread pool
- [x] bf16, 8-bit and 4-bit weights; perplexity benchmark per format
- [x] Metal GPU backend: whole forward pass in one command buffer
- [x] Speculative decoding (greedy), verified identical to plain decoding
- [x] OpenAI-compatible server with streaming, plus a chat page

### v0.2: serving, smarter decoding, faster GPU

- [x] **Paged KV cache** (vLLM's PagedAttention): 16-token blocks, per-sequence block tables
- [x] **Prefix caching across requests**: full blocks are content-addressed by a hash chain and shared
- [x] **Continuous batching** scheduler: decode tokens and prompt chunks share each forward pass; preemption by recompute when the cache fills
- [x] **`simdgroup_matrix` GEMM** on the GPU: 3x faster prefill, 3x more throughput at batch 32
- [x] Load test tool; 9.8x total throughput at 32 concurrent users
- [x] **JSON mode**: constrained decoding with a pushdown automaton; output always parses
- [x] Stop strings, `ignore_eos`, Qwen3's `enable_thinking`
- [x] **Speculative sampling** at any temperature (the rejection-sampling rule), with a statistical test
- [x] **Prompt-lookup decoding**: drafting with no draft model
- [x] Mixed-precision 4-bit (`q4`, `q4m`), chosen by measuring each matrix's sensitivity
- [x] f16 KV cache on the GPU (lossless here, half the memory)
- [x] **Qwen3** support (per-head q/k norm), checked on both backends
- [x] Property tests on hostile input; CI on Apple Silicon

### v0.3: agents and quality

- [x] **Importance matrix** (`calibrate`, `--imatrix`): activation-weighted 4-bit scale fitting; 4-bit quality loss vs bf16 down from 11% to 8.5%
- [x] **JSON Schema structured output**: a schema-compiled pushdown matcher (objects, arrays, enums, unions, `$ref`)
- [x] **Tool calling**: OpenAI `tools` / `tool_choice` / `tool_calls`, with schema-constrained forced calls
- [x] **Logprobs** and `top_logprobs`, streamed or not

### v0.4: measured against llama.cpp

- [x] Head-to-head with llama.cpp on the same Mac: decode on par (GPU and CPU), GPU prefill ~64%
- [x] Attention with 1,024 threads per head: +26% decode at 2k context
- [x] 32-token GEMM tiles and 128-token prefill chunks: +15-18% prefill
- [x] f16 KV cache by default on the GPU
- Tried and rejected after A/B measurement: RMSNorm/SiLU kernel fusion (no gain), grouped-query flash-decoding attention (slower with 2 KV heads), 64-token half-precision tiles (no clear win)

## Next

### Close the prefill gap

GPU prefill is ~64% of llama.cpp's, CPU prefill ~20%.

- **GPU:** half-precision `simdgroup_matrix` tiles with a register layout that doesn't spill (the first 64-token attempt did), and double-buffered tile loads.
- **CPU:** an optional Accelerate (AMX) path for batched matmuls, kept behind a feature flag so the default build stays dependency-free.
- **Long context:** a flash-attention kernel on the matrix units, so 8k+ contexts don't fall off.

*Done when:* GPU prefill is within 80% of llama.cpp.

### Better 4-bit

Least-squares scale fitting without activation statistics made perplexity
worse; with the importance matrix, `q4` is 8.5% behind bf16 (it was 11%).

- **k-quants:** 256-weight super-blocks with 6-bit sub-block scales and minimums.
- A bigger, more varied calibration set, and per-layer format choices driven by it.

*Done when:* `q4` perplexity is within 5% of bf16.

### Serving

- Swap-to-CPU preemption instead of recompute, and request priorities.
- JSON Schema `pattern`, numeric ranges, and optional properties.
- Multiple LoRA adapters served from one base model.

### Models

- **Llama 3.x:** `rope_scaling` type `llama3` (needs gated model access to test).
- **Mixture of experts** (Qwen-MoE / Mixtral): a router and top-k expert matmuls.

### Other platforms

- **CUDA backend** (the same kernels in CUDA C, through the driver API).
- **WebGPU / WASM:** the 0.5B model in a browser tab.
- **x86 SIMD:** AVX2 versions of the four dot-product kernels.

### Always on

- Compare tokenizer output with Hugging Face's on a large corpus.
- Fuzz with `cargo fuzz` (the property tests are a dependency-free start).
- Flamegraphs for every performance change, before and after.
