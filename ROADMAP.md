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

### v0.7: mixture of experts, the browser, better 4-bit

- [x] **Mixture of experts** (Granite 3.1 1B-A400M): top-k routing, per-(token, expert) products for decode and a GEMM per expert over gathered tokens for prefill, all on the GPU in one command buffer; GPU and CPU agree token for token
- [x] **Browser build**: the library compiled to WebAssembly with SIMD kernels, a worker and a chat page; Qwen2.5-0.5B at 49 tok/s on one thread
- [x] **4-bit with per-block offsets**, fitted by weighted least squares: `q4m` 3.2% from bf16 (was 7.7%), `q4` 7.3% (was 11.3%), 5.9% with the importance matrix
- [x] **Tokenizer checked against llama.cpp** (`tachyon tokenize`, `bench/compare_tokenizer.sh`): 118,401 tokens identical after fixing non-special added tokens
- [x] **JSON Schema optional properties** (`required`), GPT-2 style pre-tokenization, Granite's chat format and scalings

### v0.8: many adapters, memory pressure, stricter output, x86

- [x] **Multi-LoRA serving**: PEFT adapters loaded beside the base model, chosen per request by `model`; batches mix adapters, with tokens grouped by adapter so each adapter is read once per step (SGMV-style kernels on the GPU, f16; bf16 on the CPU). Four rank-16 adapters on every projection keep 89-92% of base throughput on the GPU; checked against merged weights
- [x] **Swap-out preemption and priorities**: a preempted sequence's cache goes to host memory and comes back through the prefix cache; requests carry vLLM's `priority`. Same throughput as recompute under the load test, lower median time to first token
- [x] **JSON Schema patterns, formats, lengths and number ranges**: a regex engine (Thompson NFA over characters) inside the schema matcher; bounded numbers checked digit by digit for a reachable value
- [x] **Granite tool calling** in its own format, and its template's default system message fixed
- [x] **AVX2 kernels** for x86-64, picked at run time, and a Linux CI job that runs every test on them

### v0.9: smaller and faster quantized weights

- [x] **Packed 4-bit scales**: each block's scale and offset as 8-bit multiples of f16 steps shared by 256 weights, refitted to that grid. 4.625 bits per weight for the 4-bit matrices (from 6), with the same quality by KL divergence; models 6-18% smaller
- [x] **f16 scales for 8-bit weights** (llama.cpp's Q8_0 layout): 8.5 bits per weight, from 9
- [x] **Compact scales converted where they're used**: the single-token CPU kernels convert eight scales at a time in registers; GPU lanes take eight weights (both nibbles of four bytes) per scale. 4-bit decode 13-26% faster on the GPU, as fast as before on the CPU; GPU decode at chat lengths now on par with llama.cpp
- [x] **`tachyon kld`**: KL divergence from bf16 and top-token agreement, which showed the importance matrix cutting `q4m`'s divergence by 40% where short-text perplexity said it didn't help
- [x] **LoRA adapters loaded and unloaded while serving** (vLLM's `/v1/load_lora_adapter` and `/v1/unload_lora_adapter`), serialized with requests in the scheduler, with a prefix-cache salt per adapter ever loaded
- [x] **Priority preemption**: a more urgent request with no room preempts less urgent running ones; under a full 8-sequence server its time to first token went from 9.3 s to 0.034 s
- [x] **Prefill attention compiled for the model's head size**, with half-precision queries and less threadgroup memory for 128-wide heads, so more threadgroups share a core: 2,048-token prefill +47% (Qwen2.5-0.5B) to +64% (Qwen3-0.6B), 90% of llama.cpp from 63%. GPU prefill in 512-token passes
- Tried and rejected: skipping the output rescale when no maximum moves, half-precision probabilities (no faster); 256-token CPU prefill passes (12% slower)
- Tried and rejected: 6-bit fields packed as in Q4_K (4.5 bits per weight, but unpacking them cost 20% of CPU decode); expanding a row's scales into memory before its dot product, even eight rows at a time (about a tenth of decode on short rows)

## Next

### Long context

At 2,048 tokens, prefill is 90% of llama.cpp's and decode about 92%. Prefill attention got there by fitting more threadgroups on each core; attention is now about a tenth of prefill time at 2k for 64-wide heads. Next:

- **Decode:** a single-pass kernel with an online softmax per simdgroup (no scores pass, no barriers until the merge), reading keys a cache block at a time.
- **Prefill at 8k:** measure, then double-buffered key/value tiles so loading the next tile overlaps the matrix multiplies on this one, and fewer registers for 128-wide heads (the output matrices alone take 32 per lane).

*Done when:* both are within 90% of llama.cpp at 8k tokens too.

### CPU prefill

About 65% of llama.cpp. Next: repack weights at load into the interleaved layout `smmla` reads, so the per-block shuffles go away, and fuse activation quantization into the previous op.

### Quantization

- **3-bit** matrices in the same packed layout, chosen per matrix by measured KL divergence, for a model under 300 MB.
- A bigger, more varied calibration set, and the importance matrix computed on the GPU.

### Serving

- LoRA adapters on mixture-of-experts MLPs.
- Fewer LoRA dispatches at batch 1 (8 per layer now; launches are most of the remaining 10%).

### Models

- SmolLM2 (GPT-2 style pre-tokenization is in; it adds a digit splitter).
- Larger mixtures of experts (OLMoE, Qwen-MoE): the same kernels, more memory.

### Other platforms

- **CUDA backend** (the same kernels in CUDA C, through the driver API).
- **WebGPU** for the browser build (it runs on the CPU today), and threads through shared memory where the page is cross-origin isolated.
- **x86:** AVX-512 VNNI kernels, and a pair-of-tokens path like I8MM's.

### Always on

- Compare tokenizer output on a large multilingual corpus (llama.cpp agrees on this repository's 118k tokens).
- Fuzz with `cargo fuzz` (the property tests are a dependency-free start).
- Flamegraphs for every performance change, before and after.
