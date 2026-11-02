# Tachyon

[![CI](https://github.com/divyanshkhurana06/Tachyon/actions/workflows/ci.yml/badge.svg)](https://github.com/divyanshkhurana06/Tachyon/actions/workflows/ci.yml)

**An LLM inference engine and serving stack written from scratch in Rust, with hand-written ARM NEON and Metal GPU kernels, and zero dependencies.**

Tachyon runs real Hugging Face checkpoints (Llama 3.2, Qwen2.5, Qwen3) on a MacBook's CPU or GPU, and serves many users at once the way vLLM does: a paged KV cache with prefix sharing, continuous batching, and an OpenAI-compatible API. It quantizes to 8 and 4 bits (with activation-aware calibration), decodes speculatively (with a draft model or with no model at all), and supports the API features agents rely on: structured output that matches a JSON Schema, tool calling and logprobs. Everything between the weights file and the HTTP response is in this repository, including the JSON parser, safetensors reader, BPE tokenizer, thread pool, SIMD kernels, GPU shaders, scheduler and web server. `Cargo.toml` has no dependencies.

![Chat page served by Tachyon](docs/chat-ui.jpg)

## Results

All on a MacBook Air M3 (8-core CPU, 10-core GPU, 16 GB), with other apps running.

### Serving many users: 9.8x the throughput of one

`tachyon serve --gpu` with Qwen2.5-0.5B (8-bit), driven by `tachyon loadtest`: each client streams 128-token replies back to back.

| concurrent users | total throughput | per user | mean batch | time to first token (p50) |
|---:|---:|---:|---:|---:|
| 1 | 103 tok/s | 105 tok/s | 1.0 | 0.02 s |
| 4 | 246 tok/s | 61 tok/s | 4.0 | 0.03 s |
| 16 | 768 tok/s | 49 tok/s | 15.9 | 0.07 s |
| 32 | **1,014 tok/s** | 32 tok/s | 31.9 | 0.09 s |

Decoding one sequence reads every weight once per token, so it is limited by memory bandwidth. Batching N sequences shares that read N ways, and the GPU's matrix units do the extra arithmetic almost for free. The engine alone, one token per sequence per step:

| batch | GPU step time | GPU throughput | CPU throughput |
|---:|---:|---:|---:|
| 1 | 7.6 ms | 132 tok/s | 150 tok/s |
| 4 | 13.6 ms | 294 tok/s | 404 tok/s |
| 8 | 14.9 ms | 536 tok/s | 477 tok/s |
| 16 | 20.3 ms | 790 tok/s | 502 tok/s |
| 32 | 27.6 ms | 1,157 tok/s | 494 tok/s |
| 64 | 48.3 ms | 1,324 tok/s | 458 tok/s |

The CPU gets 3.3x by batch 16, with sequences sharing each weight load in its int8 matrix instructions, and is then limited by arithmetic, not memory, so extra sequences just take turns. From `tachyon bench-batch`, 8-bit weights.

95% of the server's step time is the forward pass; sampling, streaming and scheduling are the other 5%.

### Single-sequence speed and quality

Qwen2.5-0.5B-Instruct, one sequence at a time:

| backend | format | weights | prefill | decode | perplexity |
|---|---|---:|---:|---:|---:|
| GPU | bf16 | 988 MB | 2,500 tok/s | 78.0 tok/s | 17.95 |
| GPU | 8-bit | 556 MB | 2,512 tok/s | 126.0 tok/s | 17.99 |
| GPU | `q4m` (mixed 4/8-bit) | 441 MB | 2,454 tok/s | 127.8 tok/s | 19.32 |
| GPU | `q4` (mostly 4-bit) | 389 MB | 2,325 tok/s | 126.7 tok/s | 19.97 |
| GPU | `q4` + importance matrix | 389 MB | | | **19.48** |
| GPU | `q4m` + importance matrix | 441 MB | | | **19.11** |
| CPU | bf16 | 988 MB | 173 tok/s | 73.7 tok/s | 17.94 |
| CPU | 8-bit | 556 MB | 727 tok/s | 134.6 tok/s | 18.00 |
| CPU | `q4m` | 441 MB | 690 tok/s | 143.2 tok/s | 19.32 |
| CPU | `q4` | 389 MB | 642 tok/s | 128.0 tok/s | 20.04 |

From `tachyon bench`: best of 3 runs, prefill 256 tokens, decode 128, perplexity over 979 tokens of [`bench/eval.txt`](bench/eval.txt). Speeds move by ±20% with background load, CPU decode most of all.

- **Decode** is one token at a time, which is how chat replies are generated. Smaller weights mean fewer bytes per token, so faster decoding.
- **Prefill** is processing the prompt. The GPU does it on its matrix units (`simdgroup_matrix`), for the matrix multiplies and for attention; the CPU with ARM's int8 matrix instructions (I8MM), two weight rows by two tokens per instruction.
- **Perplexity** measures how surprised the model is by a held-out text (lower is better), and shows what each format costs. CPU and GPU agree in bf16 (to 0.002) because they compute the same function. `q4` and `q4m` are mixed precision: the matrices that measurably matter most stay at 8 bits (see below).

The **importance matrix** (`tachyon calibrate`, then `--imatrix`) runs calibration text through the model, records how large each linear layer's inputs are per channel, and fits each 4-bit block's scale to minimize error weighted by those activations (llama.cpp's imatrix, in the spirit of AWQ). It cuts 4-bit's quality loss against bf16 from 11% to 8.5%, at no cost in size or speed. The calibration text ([`bench/calib.txt`](bench/calib.txt)) is separate from the evaluation text.

Qwen3-0.6B on the GPU: 55 tok/s in bf16, 86 tok/s in 8-bit.

**Llama 3.2 1B Instruct** (Llama 3's scaled RoPE, tokenizer and chat format): on the GPU in 8-bit, 1,140 tok/s prefill and 55 tok/s decode; perplexity 16.64 in 8-bit against 16.66 in bf16, where CPU and GPU agree exactly.

### Against llama.cpp

Same model (Qwen2.5-0.5B-Instruct, 8-bit: Tachyon's `q8` against Qwen's official Q8_0 GGUF), same MacBook Air M3, llama.cpp build b12040 with its Metal backend, runs alternated between the two engines. llama.cpp numbers are from `llama-bench`; decode is measured after the given amount of context (`-d`, and `tachyon bench --context`).

| | llama.cpp | Tachyon | Tachyon vs llama.cpp |
|---|---:|---:|---:|
| GPU prefill, 256 tokens | 2,540–3,315 tok/s | 2,670 tok/s | **on par** |
| GPU prefill, 2,048 tokens | 2,905–3,077 tok/s | 1,934 tok/s | 65% |
| GPU decode, 256-token context | 137–141 tok/s | 129–131 tok/s | 93% |
| GPU decode, 2,048-token context | 134 tok/s | 116 tok/s | 87% |
| CPU prefill, 6 threads | 979–1,176 tok/s | 655–727 tok/s | ~65% |
| CPU decode, 6 threads | 126–150 tok/s | 114–135 tok/s | ~90% |

Short prompts and chat-length decoding are close to llama.cpp, a mature engine with years of Apple-specific tuning, from code with no dependencies. The remaining gaps are long contexts, where attention dominates, and CPU prefill, which llama.cpp sends through Apple's Accelerate framework to the AMX matrix units while Tachyon uses ARM's int8 matrix instructions. Over the last two releases GPU prefill went from 64% to on par and CPU prefill from 20% to about 65%.

### Speculative decoding: up to 2.2x faster

A drafter guesses several tokens; the big model checks them all in one batched pass and keeps the ones it agrees with. Greedy runs produce output **token-for-token identical** to plain decoding; sampled runs keep the target's exact output distribution (the rejection-sampling rule, checked by a statistical test). On the GPU:

| target | drafter | sampling | drafts accepted | tokens per pass | speedup |
|---|---|---|---:|---:|---:|
| Qwen2.5-1.5B bf16 | 0.5B in 4-bit, 6 per step | greedy, code prompt | 80% | 5.6 | **2.19x** |
| Qwen2.5-1.5B bf16 | 0.5B in 4-bit, 4 per step | greedy, code prompt | 81% | 4.2 | **1.89x** |
| Qwen2.5-1.5B bf16 | 0.5B in 4-bit, 5 per step | temperature 0.7 | 73% | 4.6 | **1.89x** |
| Qwen2.5-1.5B bf16 | 0.5B in 4-bit, 4 per step | greedy, a short story | 39% | 2.5 | 1.26x |
| Qwen2.5-1.5B bf16 | **prompt lookup** (no model) | greedy, editing code | 23% | 2.0 | **1.83x** |
| Qwen2.5-0.5B bf16 | **prompt lookup** (no model) | greedy, editing code | 30% | 2.2 | **1.75x** |

Prompt lookup drafts by finding the last few tokens earlier in the context and proposing what followed them, which is ideal when the reply copies from the prompt (edits, summaries, answering over a document). On the CPU, speculation doesn't pay: checking 5 tokens costs 5x the arithmetic, and the CPU runs out of compute before bandwidth.

### Structured output, tool calling and logprobs

**JSON Schema.** With `response_format: {"type": "json_schema", ...}`, the schema is compiled to a pushdown matcher that runs over the bytes of each candidate token, and the sampler only picks tokens that keep the output a valid prefix of a matching document. Objects get exactly the declared keys in order, integers are integers, enums are enum values, arrays respect `minItems`/`maxItems`; `anyOf`, nullable types and `$ref` work. Asked at temperature 1.0 to invent a movie as `{title, year, genres (1-3 from a list), rating, sequel (string or null)}`, 3 of 3 replies matched exactly. `json_object` mode (any valid object) works the same way.

**Tool calling.** OpenAI's `tools` / `tool_choice` / `tool_calls` / `role: "tool"` round trip, in each model's own format: Qwen's `<tool_call>` blocks, or Llama 3's tool definitions in the first user turn and calls as bare `{"name": .., "parameters": ..}` replies. With `tool_choice: "auto"` the model decides and the server parses its calls (streamed as `tool_calls` deltas). With `"required"` or a named function, the call is opened for it (Qwen) and its arguments are constrained to that function's parameter schema, so they always fit it. (Llama 3.2 1B tends to call again after a result while tools are on offer; `tool_choice: "none"` gets its answer.)

**Logprobs.** `logprobs` and `top_logprobs` (up to 20) report each generated token's log-probability under the model and the most likely alternatives, streamed or not.

Property tests check the matchers against the JSON parser on tens of thousands of random inputs, and end-to-end tests check the real model's constrained output.

## Architecture

```text
 HTTP threads ──► Scheduler (one thread owns the model)
  (SSE out)        │  each step: 1 token from every decoding sequence
                   │           + prompt chunks of new ones, up to the budget
                   │  sample all rows in parallel (top-k in one pass, JSON mask)
                   ▼
              Backend::forward(Batch) ── CPU: NEON + thread pool
                   │                    └ GPU: one Metal command buffer
                   ▼
           paged KV cache (16-token blocks, block tables, hash-chained prefix cache)
```

| | |
|---|---|
| [`kv.rs`](src/kv.rs) | Paged KV cache: block allocator, per-sequence block tables, prefix cache with LRU eviction |
| [`scheduler.rs`](src/scheduler.rs) | Continuous batching, chunked prefill, preemption by recompute, parallel sampling |
| [`server.rs`](src/server.rs) | HTTP/1.1 on `std::net`: `/v1/chat/completions` (streaming, `stop`, `response_format`, `ignore_eos`), `/v1/models`, `/info`, chat page |
| [`model.rs`](src/model.rs) | CPU forward pass over a batch: RMSNorm, grouped-query attention through block tables, RoPE (with Llama 3's scaling), SiLU MLP, Qwen3's q/k norm |
| [`gpu/`](src/gpu) | Metal backend: [`kernels.metal`](src/gpu/kernels.metal) (matrix-vector kernels for decode, `simdgroup_matrix` GEMM for batches, FlashAttention-style prefill attention and paged decode attention, RoPE, norms) and an [Objective-C bridge](src/gpu/shim.m) compiled by `build.rs` |
| [`kernels.rs`](src/kernels.rs) | NEON kernels for bf16, int8×int8 and int4×int8: I8MM matrix instructions for batches, SDOT for single tokens; the quantizers |
| [`speculative.rs`](src/speculative.rs) | Speculative decoding and sampling, model and prompt-lookup drafters |
| [`constrain.rs`](src/constrain.rs) | JSON pushdown automaton for constrained decoding |
| [`sampler.rs`](src/sampler.rs) | Temperature, top-k (single pass), top-p, repetition penalty, constraint masks |
| [`tokenizer.rs`](src/tokenizer.rs) | Byte-level BPE from `tokenizer.json`, with the pre-tokenizer regex reimplemented by hand |
| [`chat.rs`](src/chat.rs) | ChatML and Llama 3 chat formats, and each one's tool calls |
| [`safetensors.rs`](src/safetensors.rs), [`json.rs`](src/json.rs), [`pool.rs`](src/pool.rs) | mmap'd weights, JSON, a spin-then-park thread pool |

### Design notes

- **Decode is a memory-bandwidth problem.** A 0.5B model in bf16 is ~1 GB, all of it read for every token: at ~100 GB/s the ceiling is ~100 tok/s whatever the arithmetic. Hence quantization for one user, and batching for many.
- **Paged KV cache.** Sequences grow 16 positions at a time instead of reserving their maximum length, so the cache holds as many conversations as actually fit. Full blocks are registered under a hash of the entire token prefix up to them; a new request whose prompt starts the same way takes those blocks instead of recomputing them.
- **The matrix units changed the batching story.** The first batched GPU kernel reused each weight for 8 tokens and read 8 bytes of activations per byte of weights; 4 tokens cost twice as much as 1. A tiled GEMM on `simdgroup_matrix` (16 tokens × 64 rows per threadgroup, tiles staged in threadgroup memory) made batch 32 run at 1,300 tok/s instead of 410, and prefill 3x faster.
- **Then attention was.** With the matrix multiplies on the matrix units, attention took 30% of prefill: it ran one 1,024-thread threadgroup per head per token, mostly idle at short contexts, each rereading the same keys. Runs of a sequence's consecutive tokens now go to a FlashAttention-style kernel: one threadgroup takes 32 queries of a head, walks the cache 32 positions at a time with those positions' keys and values staged once in threadgroup memory, computes scores and outputs on the 8x8 matrix units, and keeps the softmax exact with a running max and sum, rescaling the output through a diagonal matrix multiply. Prefill went from 2,000 to 2,670 tok/s on a 256-token prompt and from 950 to 1,930 on a 2,048-token one; decode steps keep the per-token kernel, which suits one query.
- **Then the scheduler was the bottleneck.** With a fast engine, 25% of each step went to sampling: copying 600 KB of logits per sequence and partially sorting 150k entries. A single-pass top-k (insert only what beats the current k-th best) and parallel sampling brought that to 5%, taking 32 users from 695 to 1,014 tok/s.
- **Quantization by measurement.** Keeping each matrix type at 8 bits in turn showed q/k/v buys 0.6 perplexity for 3% more memory and the down projection 0.5 for 14%; `q4` and `q4m` are built from that. Fitting 4-bit scales by least squares lowered weight error but made perplexity *worse*: what matters is error times the input it meets. Weighting the fit by measured activations (the importance matrix) is what finally helped.
- **Constrained decoding is a matcher plus a sampler filter.** JSON mode, JSON Schema and forced tool calls are all the same mechanism: a byte-level automaton that says whether a token's bytes can come next, and a sampler that ranks the likely tokens first and asks about those, so the check costs a few hundred probes per token, not 150,000.
- **The CPU's int8 path is SDOT.** One `sdot` instruction multiplies 16 pairs of int8 values and adds them into four int32 lanes, against four instructions for the widening multiply-adds it replaced (written as inline assembly, since Rust's intrinsic for it is unstable). Each weight block is also loaded once for four tokens. Prefill went from about 300 to 490 tok/s and batch-4 decoding from about 200 to 400 tok/s.
- **Then I8MM.** The M2 and later also have `smmla`, which multiplies a 2x8 int8 matrix by an 8x2 one: two weight rows by two tokens, 32 multiply-adds per instruction. Activations are interleaved two tokens at a time once per matmul so the instruction reads them directly, and each weight block is loaded once for eight tokens. CPU prefill went up another 45%, to about 700 tok/s.
- **Batch invariance.** Every int8 path (single token, four tokens with SDOT, token pairs with I8MM) scales each block's exact integer sum into one running total with a fused multiply-add, in the same order, so a sequence's logits on the CPU are bit-for-bit the same whether it runs alone or batched with any others. A kernel test checks that equality exactly, not within a tolerance.
- **The thread pool.** With one spinning worker per core, any background process preempts one and every matmul waits for it. Leaving two efficiency cores free doubled CPU decode speed.
- **Measure under identical conditions, or not at all.** GPU speed on a laptop moves with power source, Low Power Mode and background load (one run measured 105 tok/s, the same code later 131). Every optimization here was decided by building both versions and alternating them in the same session. Kept and rejected so far:
  - kept: **1,024 threads per attention head** for decode (+26% at 2,048 tokens of context), with **4 value dims per thread** in its last pass (+11% more); **32-token GEMM tiles fed 128-token prefill chunks** (+15-18% prefill), with **weights staged as half precision** (+11-14%, and the tile's threadgroup memory down from 20 KB to 8 KB); the **FlashAttention-style prefill kernel** (+30% at 256 tokens, 2x at 2,048); **SDOT with four tokens per weight load** (1.7x CPU prefill), then **I8MM with eight** (another +45%); an **f16 KV cache** by default (lossless here, and llama.cpp's default too);
  - rejected: **fusing RMSNorm and SiLU into the matmuls** (no faster: launches inside one command buffer are cheap); **grouped-query flash-decoding attention** (reads each key once instead of 7 times, but with only 2 KV heads it left most of the GPU idle); **64-token GEMM tiles**, **half-precision activation tiles** and **256-token prefill chunks** (no faster); **sizing attention threadgroups to the context** (+15% on short prompts, slower on long ones: the per-token kernel was the wrong shape, which the prefill kernel fixed); **splitting each key across lanes**, **unrolling the value loads** and **splitting the context across threadgroups** (with a combine pass and scores kept in threadgroup memory) in decode attention, and **16- or 64-query tiles** in prefill attention (no faster, or slower: 16 queries share each key and value load too little, and 32 is the sweet spot).

### How it's verified

`cargo test --release` runs 59 tests, in CI on Apple Silicon. GitHub's virtual Macs expose a GPU that returns zeros, so the engine self-tests the GPU at startup and refuses a broken one; there the GPU comparisons skip, and they run on real hardware. Beyond unit tests of every kernel against a scalar reference, the tests check against the real model that:

- several sequences batched together produce exactly the tokens each produces alone, on CPU and GPU;
- the GPU produces exactly the same greedy tokens as the CPU, for Qwen2.5, Qwen3 and Llama 3.2;
- the prefill attention kernel, fed a whole prompt through a scrambled block table, gives the same logits as the per-token kernel (largest difference 0.009);
- Llama 3.2's tokenizer and chat format give the known token ids of Llama 3's prompt header and assistant turn, and it calls tools in its own format;
- the CPU's int8 kernels give bit-identical results for a token whichever path (alone, in fours, in pairs) computes it;
- speculative decoding (model or prompt-lookup drafter) is identical to plain greedy decoding, and speculative sampling reproduces the target distribution (200,000 trials);
- reusing the KV cache gives the same output as recomputing it;
- JSON mode always parses, schema-constrained output has the requested shape, and forced tool calls fit the function's schema;
- the tokenizer round-trips code, Unicode and emoji and won't let user text forge control tokens;
- the JSON parser, safetensors reader and pre-tokenizer survive thousands of random and corrupted inputs, and the JSON and JSON Schema matchers never accept anything the parser rejects.

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

For Llama 3.2 1B, download the same four files from [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) (gated: accept Meta's license on Hugging Face first) into `models/llama-3.2-1b-instruct`, and pass `--model models/llama-3.2-1b-instruct`.

```bash
./target/release/tachyon run --gpu --prompt "Explain recursion to a five-year-old."
./target/release/tachyon chat --quant q4m                 # multi-turn chat in the terminal
./target/release/tachyon serve --gpu                      # many users; chat page at http://localhost:8080
./target/release/tachyon loadtest --clients 1,16,32       # against a running server
./target/release/tachyon bench                            # speed and perplexity per format
./target/release/tachyon bench --context 2048             # decode speed at a long context
./target/release/tachyon bench-batch                      # throughput vs batch size
./target/release/tachyon speculate --gpu --lookup --prompt "Add type hints to this code: ..."
./target/release/tachyon calibrate && ./target/release/tachyon run --quant q4 \
    --imatrix models/qwen2.5-0.5b-instruct/imatrix.bin --prompt "..."  # better 4-bit
./target/release/tachyon run --schema movie.schema.json --prompt "Invent a movie."
```

Any OpenAI client works against `serve`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
reply = client.chat.completions.create(
    model="qwen2.5-0.5b-instruct",
    messages=[{"role": "user", "content": "Describe a cat as JSON."}],
    response_format={"type": "json_object"},
)
print(reply.choices[0].message.content)

# Tool calling
tools = [{"type": "function", "function": {
    "name": "get_weather",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "unit": {"enum": ["celsius", "fahrenheit"]}}}}}]
reply = client.chat.completions.create(
    model="qwen2.5-0.5b-instruct",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
)
print(reply.choices[0].message.tool_calls)  # get_weather(city="Tokyo", unit=...)
```

Run `tachyon help` for every option.

## Limitations

- Apple Silicon for the GPU path; the CPU path is portable but only has NEON kernels (other CPUs fall back to scalar code).
- JSON Schema support covers what structured output needs (objects with all properties required and in order, arrays, enums, unions told apart by their first character, `$ref`), not string `pattern`s or numeric ranges.
- No mixture-of-experts models yet.
- The tokenizer skips NFC normalization and treats Unicode letters slightly more broadly than `\p{L}`; this matters only for some scripts with combining marks.

## Roadmap

See [ROADMAP.md](ROADMAP.md): long-context attention, I8MM for CPU prefill, k-quants, MoE models, CUDA and WebGPU backends.

## License

MIT
