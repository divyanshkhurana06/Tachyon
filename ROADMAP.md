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

### v0.5: closing the prefill gap, and Llama

- [x] **FlashAttention-style prefill kernel** on the matrix units: +30% prefill at 256 tokens, 2x at 2,048
- [x] **Half-precision GEMM weight tiles**, with the output tile reusing their memory: +11-14% prefill
- [x] GPU prefill on par with llama.cpp at 256 tokens (was 64%)
- [x] **SDOT** inline assembly and four tokens per weight load on the CPU: 1.7x prefill (42% of llama.cpp, was 20%), 2.7x throughput at batch 4
- [x] Decode attention reads four value dims per thread: +11% decode at 2k context
- [x] **Llama 3.2**: Llama 3 RoPE scaling, `ignore_merges` tokenization, Llama 3 chat format; GPU and CPU agree token for token
- Tried and rejected: 64-token tiles and half-precision activations again (no gain), attention threadgroups sized to the context (slower on long prompts), 256-token prefill chunks, splitting keys across lanes in decode attention

### v0.6: I8MM, and agents on Llama

- [x] **I8MM** (`smmla`) CPU matmuls: two weight rows by two tokens per instruction, eight tokens per weight load. CPU prefill +45% (about 65% of llama.cpp, from 42%), batch throughput +40-60% from 8 sequences up
- [x] **Batch-invariant CPU results**: every int8 path does the same arithmetic in the same order, so batching never changes a sequence's logits (checked bit for bit)
- [x] **Llama 3 tool calling** in its own format, with forced calls schema-constrained; indented JSON matching Python's `json.dumps`
- Tried and rejected: split-context decode attention with threadgroup-memory scores, unrolled value loads (no faster); 16- and 64-query prefill attention tiles (slower)

## Next

### Long context

At 2,048 tokens, prefill is 65% of llama.cpp's and decode 87%. Attention is a third of prefill time there and an eighth of decode. Splitting a head's context across threadgroups, scores in threadgroup memory, and smaller or larger query tiles were all measured and didn't help, so the next attempts are structural:

- **Decode:** a single-pass kernel with an online softmax per simdgroup (no scores pass, no barriers until the merge), reading keys a cache block at a time.
- **Prefill:** double-buffered key/value tiles, so loading the next tile overlaps the matrix multiplies on this one.

*Done when:* both are within 90% of llama.cpp at 2k and 8k tokens.

### CPU prefill

About 65% of llama.cpp. Next: repack weights at load into the interleaved layout `smmla` reads, so the per-block shuffles go away, and fuse activation quantization into the previous op.

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

- SmolLM2 (a GPT-2 style pre-tokenizer).
- **Mixture of experts** (Qwen-MoE / Mixtral): a router and top-k expert matmuls.

### Other platforms

- **CUDA backend** (the same kernels in CUDA C, through the driver API).
- **WebGPU / WASM:** the 0.5B model in a browser tab.
- **x86 SIMD:** AVX2 versions of the four dot-product kernels.

### Always on

- Compare tokenizer output with Hugging Face's on a large corpus.
- Fuzz with `cargo fuzz` (the property tests are a dependency-free start).
- Flamegraphs for every performance change, before and after.
