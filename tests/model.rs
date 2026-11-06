//! End-to-end checks against a real checkpoint. They need the model in
//! models/qwen2.5-0.5b-instruct (see the README) and are skipped without it.
//!
//!     cargo test --release

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use tachyon::chat::{ChatTemplate, Message, DEFAULT_SYSTEM};
use tachyon::config::Config;
use tachyon::engine::{Engine, Session};
use tachyon::generate::{generate, GenOptions};
use tachyon::model::{CpuBackend, Weights};
use tachyon::pool::Pool;
use tachyon::sampler::SamplerConfig;
use tachyon::speculative::generate_speculative;
use tachyon::tokenizer::Tokenizer;
use tachyon::weights::Format;

/// The engines spin worker threads, so run these tests one at a time.
static SERIAL: Mutex<()> = Mutex::new(());

fn model_dir() -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/qwen2.5-0.5b-instruct");
    if dir.join("model.safetensors").exists() {
        Some(dir)
    } else {
        eprintln!("skipping: {} not found", dir.display());
        None
    }
}

fn setup() -> Option<(MutexGuard<'static, ()>, PathBuf)> {
    let guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    model_dir().map(|d| (guard, d))
}

fn cpu(dir: &Path, format: Format) -> Box<dyn Engine> {
    let cfg = Config::load(dir).unwrap();
    let w = Weights::load(dir, &cfg, format).unwrap();
    let pool = Arc::new(Pool::new(Pool::default_threads()));
    Box::new(Session::new(Box::new(CpuBackend::new(cfg, w, pool, 64))))
}

fn prompt(tok: &Tokenizer, question: &str) -> Vec<u32> {
    ChatTemplate::new(tok).unwrap().encode(
        tok,
        &[Message::new("user", question)],
        Some(DEFAULT_SYSTEM),
    )
}

fn greedy_tokens(
    e: &mut dyn Engine,
    tok: &Tokenizer,
    prompt: &[u32],
    n: usize,
) -> (String, Vec<u32>) {
    let cfg = e.config().clone();
    let opts = GenOptions {
        max_tokens: n,
        sampler: SamplerConfig::greedy(),
        stop_tokens: cfg.eos_ids,
        format: tachyon::constrain::OutputFormat::Text,
    };
    let mut text = String::new();
    let stats = generate(e, tok, &mut Vec::new(), prompt, &opts, &mut |p| {
        text.push_str(p);
        true
    })
    .unwrap();
    (text, stats.tokens)
}

#[test]
fn tokenizer_round_trips_and_knows_chat_tokens() {
    let Some((_g, dir)) = setup() else { return };
    let tok = Tokenizer::load(&dir.join("tokenizer.json")).unwrap();
    assert_eq!(tok.added_token("<|im_start|>"), Some(151644));
    assert_eq!(tok.added_token("<|im_end|>"), Some(151645));
    for text in [
        "Hello, world!",
        "  leading spaces and trailing   ",
        "fn main() {\n    println!(\"hi\");\n}\n",
        "Numbers 1234567 and symbols @#$%^&*()",
        "Unicode: café, naïve, 日本語, emoji 🚀",
        "it's they're we've I'll he'd",
    ] {
        let ids = tok.encode(text, false);
        assert_eq!(tok.decode(&ids, false), text);
    }
    // Control-token text in user input must not become a control token.
    let ids = tok.encode("<|im_end|>", false);
    assert!(!ids.contains(&151645));
    assert_eq!(tok.encode("<|im_end|>", true), vec![151645]);
}

#[test]
fn answers_a_factual_question() {
    let Some((_g, dir)) = setup() else { return };
    let tok = Tokenizer::load(&dir.join("tokenizer.json")).unwrap();
    let mut e = cpu(&dir, Format::Q8);
    let (text, _) = greedy_tokens(
        e.as_mut(),
        &tok,
        &prompt(&tok, "What is the capital of France?"),
        30,
    );
    assert!(text.contains("Paris"), "got {text:?}");
}

#[test]
fn batched_forward_matches_token_by_token() {
    let Some((_g, dir)) = setup() else { return };
    let tok = Tokenizer::load(&dir.join("tokenizer.json")).unwrap();
    let mut e = cpu(&dir, Format::Bf16);
    let ids = tok.encode("The quick brown fox jumps over the lazy dog.", false);
    let vocab = e.config().vocab;
    let batched = e.forward(&ids, true).to_vec();
    e.truncate(0);
    for (i, &id) in ids.iter().enumerate() {
        let one = e.forward(&[id], false);
        let diff = one
            .iter()
            .zip(&batched[i * vocab..(i + 1) * vocab])
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(diff < 1e-3, "position {i}: logits differ by {diff}");
    }
}

#[test]
fn reusing_the_kv_cache_changes_nothing() {
    let Some((_g, dir)) = setup() else { return };
    let tok = Tokenizer::load(&dir.join("tokenizer.json")).unwrap();
    let mut e = cpu(&dir, Format::Q8);
    let opts = GenOptions {
        max_tokens: 20,
        sampler: SamplerConfig::greedy(),
        stop_tokens: e.config().eos_ids.clone(),
        format: tachyon::constrain::OutputFormat::Text,
    };
    let p = prompt(&tok, "List three fruits.");

    let mut cache = Vec::new();
    let first = generate(e.as_mut(), &tok, &mut cache, &p, &opts, &mut |_| true).unwrap();
    // Same prompt again: everything but the last token comes from the cache.
    let second = generate(e.as_mut(), &tok, &mut cache, &p, &opts, &mut |_| true).unwrap();
    assert_eq!(second.cached_tokens, p.len() - 1);
    assert_eq!(first.tokens, second.tokens);
}

#[test]
fn speculative_decoding_matches_greedy() {
    use tachyon::speculative::{LookupDrafter, ModelDrafter};
    let Some((_g, dir)) = setup() else { return };
    let tok = Tokenizer::load(&dir.join("tokenizer.json")).unwrap();
    let mut target = cpu(&dir, Format::Bf16);
    let mut draft = cpu(&dir, Format::Q4);
    let greedy = SamplerConfig::greedy();
    let eos = target.config().eos_ids.clone();

    let p = prompt(&tok, "Explain what a hash map is in two sentences.");
    let (_, plain) = greedy_tokens(target.as_mut(), &tok, &p, 60);
    let mut drafter = ModelDrafter::new(draft.as_mut(), greedy.clone(), tok.len());
    let spec = generate_speculative(
        target.as_mut(),
        &mut drafter,
        &tok,
        &p,
        4,
        60,
        &eos,
        &greedy,
        &mut |_| true,
    )
    .unwrap();
    assert_eq!(spec.tokens, plain);
    assert!(spec.acceptance() > 0.5, "acceptance {}", spec.acceptance());

    // Prompt lookup: a prompt the reply will copy from.
    let p = prompt(&tok, "Repeat this sentence exactly: The quick brown fox jumps over the lazy dog near the river bank.");
    let (_, plain) = greedy_tokens(target.as_mut(), &tok, &p, 40);
    let spec = generate_speculative(
        target.as_mut(),
        &mut LookupDrafter::default(),
        &tok,
        &p,
        6,
        40,
        &eos,
        &greedy,
        &mut |_| true,
    )
    .unwrap();
    assert_eq!(spec.tokens, plain);
    assert!(
        spec.tokens_per_pass() > 1.5,
        "lookup accepted too little: {}",
        spec.tokens_per_pass()
    );
}

#[cfg(target_os = "macos")]
#[test]
fn gpu_matches_cpu() {
    let Some((_g, dir)) = setup() else { return };
    let tok = Tokenizer::load(&dir.join("tokenizer.json")).unwrap();
    let cfg = Config::load(&dir).unwrap();
    let w = Weights::load(&dir, &cfg, Format::Bf16).unwrap();
    let mut gpu: Box<dyn Engine> = match tachyon::gpu::GpuBackend::new(cfg, w, 64, 1024, false) {
        Ok(g) => Box::new(Session::new(Box::new(g))),
        Err(e) => return eprintln!("skipping: {e}"),
    };
    let p = prompt(&tok, "Give me one tip for writing clean code.");
    let (_, on_gpu) = greedy_tokens(gpu.as_mut(), &tok, &p, 40);
    let (_, on_cpu) = greedy_tokens(cpu(&dir, Format::Bf16).as_mut(), &tok, &p, 40);
    assert_eq!(on_gpu, on_cpu);
}

/// Each quantized format means the same weights on the GPU as on the CPU:
/// next-token distributions agree closely through the GPU's GEMM (a
/// prompt), small-batch and single-token paths. Not exactly, because the
/// CPU also rounds activations to 8 bits; a misread scale or nibble would
/// be far outside this.
#[cfg(target_os = "macos")]
#[test]
fn gpu_reads_quantized_weights_as_the_cpu_does() {
    use tachyon::engine::Backend;
    use tachyon::kv::{Batch, Logits};
    let Some((_g, dir)) = setup() else { return };
    let tok = Tokenizer::load(&dir.join("tokenizer.json")).unwrap();
    let cfg = Config::load(&dir).unwrap();
    let text =
        std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("bench/eval.txt"))
            .unwrap();
    let ids = &tok.encode(&text, false)[..36];
    let pool = Arc::new(Pool::new(Pool::default_threads()));
    let log_softmax = |row: &[f32]| {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let lse = max
            + row
                .iter()
                .map(|&l| (l as f64 - max).exp())
                .sum::<f64>()
                .ln();
        row.iter().map(|&l| l as f64 - lse).collect::<Vec<f64>>()
    };
    for format in [Format::Q8, Format::Q4M, Format::Q4] {
        let w = Weights::load(&dir, &cfg, format).unwrap();
        let mut gpu = match tachyon::gpu::GpuBackend::new(cfg.clone(), w, 16, 256, true) {
            Ok(g) => g,
            Err(e) => return eprintln!("skipping: {e}"),
        };
        let w = Weights::load(&dir, &cfg, format).unwrap();
        let mut cpu = CpuBackend::new(cfg.clone(), w, pool.clone(), 16);
        let table: Vec<u32> = (0..3).collect();
        let mut worst = 0f64;
        // 32 tokens, then 3, then 1.
        for (start, end) in [(0, 32), (32, 35), (35, 36)] {
            let mut b = Batch::default();
            b.push(&table, start, &ids[start..end], Logits::All);
            let on_gpu = gpu.forward(&b).to_vec();
            let on_cpu = cpu.forward(&b);
            let rows = on_gpu
                .chunks_exact(cfg.vocab)
                .zip(on_cpu.chunks_exact(cfg.vocab));
            // Position 0 has the model's outsized first-token activations,
            // where the CPU's 8-bit rounding shows most: left out.
            for (g, c) in rows.skip((start == 0) as usize) {
                let (lg, lc) = (log_softmax(g), log_softmax(c));
                let kl: f64 = lc.iter().zip(&lg).map(|(&c, &g)| c.exp() * (c - g)).sum();
                worst = worst.max(kl);
            }
        }
        eprintln!("{}: worst KL between backends {worst:.5}", format.name());
        assert!(worst < 0.02, "{}: {worst}", format.name());
    }
}

/// More single tokens in one batch than one dispatch of the per-token
/// attention kernel takes: each still gets what it gets alone.
#[cfg(target_os = "macos")]
#[test]
fn gpu_attention_takes_more_tokens_than_one_dispatch() {
    use tachyon::engine::Backend;
    use tachyon::kv::{Batch, Logits};
    let Some((_g, dir)) = setup() else { return };
    let tok = Tokenizer::load(&dir.join("tokenizer.json")).unwrap();
    let cfg = Config::load(&dir).unwrap();
    let w = Weights::load(&dir, &cfg, Format::Q8).unwrap();
    let mut gpu = match tachyon::gpu::GpuBackend::new(cfg.clone(), w, 2 * 300 + 8, 64, true) {
        Ok(g) => g,
        Err(e) => return eprintln!("skipping: {e}"),
    };
    let ids = tok.encode("one two three", false);
    // 300 sequences of two tokens: their first tokens in one batch, then
    // their second tokens (each alone in its sequence: per-token attention).
    let n = 300;
    let table = |i: usize| [2 * i as u32, 2 * i as u32 + 1];
    let token = |i: usize, j: usize| ids[(i + j) % ids.len()];
    let watched = [0, 255, 256, 299];
    let mut b = Batch::default();
    for i in 0..n {
        b.push(&table(i), 0, &[token(i, 0)], Logits::None);
    }
    gpu.forward(&b);
    b.clear();
    for i in 0..n {
        let want = if watched.contains(&i) {
            Logits::Last
        } else {
            Logits::None
        };
        b.push(&table(i), 1, &[token(i, 1)], want);
    }
    let together = gpu.forward(&b).to_vec();
    for (k, &i) in watched.iter().enumerate() {
        let mut one = Batch::default();
        one.push(&table(i), 1, &[token(i, 1)], Logits::Last);
        let alone = gpu.forward(&one)[..cfg.vocab].to_vec();
        let got = &together[k * cfg.vocab..(k + 1) * cfg.vocab];
        let diff = got
            .iter()
            .zip(&alone)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(diff < 0.05, "sequence {i}: {diff}");
    }
}

/// The prefill attention kernel (a run of tokens at once, through a
/// scrambled block table) against the per-token kernel, over many cache
/// blocks and query tiles.
#[cfg(target_os = "macos")]
#[test]
fn gpu_prefill_attention_matches_token_by_token() {
    use tachyon::engine::Backend;
    use tachyon::kv::{Batch, Logits};
    let Some((_g, dir)) = setup() else { return };
    let tok = Tokenizer::load(&dir.join("tokenizer.json")).unwrap();
    let cfg = Config::load(&dir).unwrap();
    let load = || {
        let w = Weights::load(&dir, &cfg, Format::Q8).unwrap();
        tachyon::gpu::GpuBackend::new(cfg.clone(), w, 64, 1024, true)
    };
    let mut whole = match load() {
        Ok(g) => g,
        Err(e) => return eprintln!("skipping: {e}"),
    };
    let mut alone = Session::new(Box::new(load().unwrap()));
    let text =
        std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("bench/eval.txt"))
            .unwrap();
    let ids = &tok.encode(&text, false)[..200];

    let table: Vec<u32> = (0..64).rev().collect();
    let mut batch = Batch::default();
    batch.push(&table, 0, ids, Logits::Last);
    let at_once = whole.forward(&batch)[..cfg.vocab].to_vec();
    let mut step = Vec::new();
    for &id in ids {
        step = alone.forward(&[id], false)[..cfg.vocab].to_vec();
    }
    let diff = at_once
        .iter()
        .zip(&step)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("largest logit difference {diff}");
    assert_eq!(
        tachyon::sampler::argmax(&at_once),
        tachyon::sampler::argmax(&step)
    );
    assert!(diff < 0.05, "logits differ by {diff}");
}

/// Greedy tokens with no stop token, through the single-sequence API.
fn greedy_raw(e: &mut dyn Engine, prompt: &[u32], n: usize) -> Vec<u32> {
    let vocab = e.config().vocab;
    e.truncate(0);
    let mut next = tachyon::sampler::argmax(&e.forward(prompt, false)[..vocab]);
    let mut out = vec![next];
    while out.len() < n {
        next = tachyon::sampler::argmax(&e.forward(&[next], false)[..vocab]);
        out.push(next);
    }
    out
}

/// Decodes several prompts in lockstep, all in one batch per step, through
/// the paged cache, and checks each matches decoding it alone.
fn check_batched_matches_alone(
    mut backend: Box<dyn tachyon::engine::Backend>,
    alone: &mut dyn Engine,
    tok: &Tokenizer,
) {
    use tachyon::kv::{Batch, BlockPool, Logits};
    let prompts: Vec<Vec<u32>> = [
        "Name a planet.",
        "Write one word that rhymes with cat, then explain.",
        "2+2=",
    ]
    .iter()
    .map(|q| prompt(tok, q))
    .collect();
    let n = 12;
    let expected: Vec<Vec<u32>> = prompts.iter().map(|p| greedy_raw(alone, p, n)).collect();

    let vocab = backend.config().vocab;
    let mut pool = BlockPool::new(backend.num_blocks());
    let mut seqs: Vec<_> = prompts.iter().map(|p| pool.start(p)).collect();
    let mut outs: Vec<Vec<u32>> = vec![Vec::new(); prompts.len()];
    let mut batch = Batch::default();
    for step in 0..n {
        batch.clear();
        for (i, seq) in seqs.iter_mut().enumerate() {
            let new: Vec<u32> = if step == 0 {
                prompts[i][seq.len..].to_vec()
            } else {
                vec![*outs[i].last().unwrap()]
            };
            assert!(pool.reserve(seq, seq.len + new.len()));
            batch.push(&seq.blocks, seq.len, &new, Logits::Last);
            seq.len += new.len();
        }
        let logits = backend.forward(&batch).to_vec();
        for (i, row) in logits.chunks_exact(vocab).enumerate() {
            outs[i].push(tachyon::sampler::argmax(&row[..tok.len()]));
        }
    }
    assert_eq!(outs, expected);
}

#[test]
fn batched_sequences_match_running_alone() {
    let Some((_g, dir)) = setup() else { return };
    let tok = Tokenizer::load(&dir.join("tokenizer.json")).unwrap();
    let cfg = Config::load(&dir).unwrap();
    let pool = Arc::new(Pool::new(Pool::default_threads()));
    let backend = CpuBackend::new(
        cfg.clone(),
        Weights::load(&dir, &cfg, Format::Q8).unwrap(),
        pool,
        64,
    );
    check_batched_matches_alone(Box::new(backend), cpu(&dir, Format::Q8).as_mut(), &tok);

    #[cfg(target_os = "macos")]
    if let Ok(g) = tachyon::gpu::GpuBackend::new(
        cfg.clone(),
        Weights::load(&dir, &cfg, Format::Q8).unwrap(),
        64,
        1024,
        false,
    ) {
        let alone = tachyon::gpu::GpuBackend::new(
            cfg.clone(),
            Weights::load(&dir, &cfg, Format::Q8).unwrap(),
            64,
            1024,
            false,
        )
        .unwrap();
        check_batched_matches_alone(Box::new(g), &mut Session::new(Box::new(alone)), &tok);
    }
}

#[test]
fn json_mode_always_parses() {
    let Some((_g, dir)) = setup() else { return };
    let tok = Tokenizer::load(&dir.join("tokenizer.json")).unwrap();
    let mut e = cpu(&dir, Format::Q8);
    // Prompts that don't ask for JSON: the constraint alone must force it.
    for (i, q) in [
        "Tell me a joke.",
        "What is 2 + 2?",
        "Write a haiku about rain.",
    ]
    .iter()
    .enumerate()
    {
        let opts = GenOptions {
            max_tokens: 120,
            sampler: SamplerConfig {
                temperature: 1.0,
                seed: i as u64 + 1,
                ..SamplerConfig::default()
            },
            stop_tokens: e.config().eos_ids.clone(),
            format: tachyon::constrain::OutputFormat::JsonObject,
        };
        let mut text = String::new();
        let stats = generate(
            e.as_mut(),
            &tok,
            &mut Vec::new(),
            &prompt(&tok, q),
            &opts,
            &mut |p| {
                text.push_str(p);
                true
            },
        )
        .unwrap();
        if stats.finish == tachyon::generate::Finish::Stop {
            assert!(
                tachyon::json::Json::parse(&text).is_ok(),
                "invalid JSON for {q:?}: {text}"
            );
        }
        assert!(text.trim_start().starts_with('{'), "{text}");
    }
}

/// Qwen3 adds per-head q/k RMSNorm; check it on both backends.
/// Needs models/qwen3-0.6b; skipped without it.
#[test]
fn qwen3_runs_and_backends_agree() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/qwen3-0.6b");
    if !dir.join("model.safetensors").exists() {
        return eprintln!("skipping: {} not found", dir.display());
    }
    let tok = Tokenizer::load(&dir.join("tokenizer.json")).unwrap();
    let tpl = ChatTemplate::new(&tok).unwrap();
    let mut p = tpl.encode(
        &tok,
        &[Message::new(
            "user",
            "What is the capital of France? Answer briefly.",
        )],
        None,
    );
    tpl.skip_thinking(&tok, &mut p);

    #[cfg_attr(not(target_os = "macos"), allow(unused_variables))]
    let (text, on_cpu) = greedy_tokens(cpu(&dir, Format::Bf16).as_mut(), &tok, &p, 20);
    assert!(text.contains("Paris"), "{text}");
    #[cfg(target_os = "macos")]
    {
        let cfg = Config::load(&dir).unwrap();
        let w = Weights::load(&dir, &cfg, Format::Bf16).unwrap();
        if let Ok(g) = tachyon::gpu::GpuBackend::new(cfg, w, 64, 1024, false) {
            let (_, on_gpu) = greedy_tokens(&mut Session::new(Box::new(g)), &tok, &p, 20);
            assert_eq!(on_gpu, on_cpu);
        }
    }
}

#[test]
fn json_schema_output_has_the_requested_shape() {
    use tachyon::constrain::OutputFormat;
    use tachyon::json::Json;
    use tachyon::schema::{JsonSchema, Schema};
    let Some((_g, dir)) = setup() else { return };
    let tok = Tokenizer::load(&dir.join("tokenizer.json")).unwrap();
    let mut e = cpu(&dir, Format::Q8);
    let schema = Json::parse(
        r#"{"type":"object","properties":{"city":{"type":"string"},"population":{"type":"integer"},"coastal":{"type":"boolean"},"landmarks":{"type":"array","items":{"type":"string"},"maxItems":2}},"required":["city","population","coastal","landmarks"]}"#,
    )
    .unwrap();
    let format = OutputFormat::JsonSchema(JsonSchema::new(Schema::parse(&schema).unwrap()));
    for seed in 1..=3 {
        let opts = GenOptions {
            // Room to finish: at temperature 1.0 a reply can spend many
            // tokens on whitespace, which JSON allows anywhere between values.
            max_tokens: 300,
            sampler: SamplerConfig {
                temperature: 1.0,
                seed,
                ..SamplerConfig::default()
            },
            stop_tokens: e.config().eos_ids.clone(),
            format: format.clone(),
        };
        let mut text = String::new();
        generate(
            e.as_mut(),
            &tok,
            &mut Vec::new(),
            &prompt(&tok, "Tell me about a city."),
            &opts,
            &mut |p| {
                text.push_str(p);
                true
            },
        )
        .unwrap();
        let v = Json::parse(&text).unwrap_or_else(|e| panic!("{e}: {text}"));
        let keys: Vec<&str> = v
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, _)| k.as_str())
            .collect();
        assert_eq!(
            keys,
            ["city", "population", "coastal", "landmarks"],
            "{text}"
        );
        assert!(
            v.get("population").unwrap().as_u64().is_some()
                || v.get("population")
                    .unwrap()
                    .as_f64()
                    .is_some_and(|n| n.fract() == 0.0)
        );
        assert!(v.get("coastal").unwrap().as_bool().is_some());
        assert!(v.get("landmarks").unwrap().as_array().unwrap().len() <= 2);
    }
}

/// String patterns, formats, lengths and number ranges hold in sampled
/// output.
#[test]
fn json_schema_patterns_and_ranges_hold() {
    use tachyon::constrain::OutputFormat;
    use tachyon::json::Json;
    use tachyon::regex::Regex;
    use tachyon::schema::{JsonSchema, Schema};
    let Some((_g, dir)) = setup() else { return };
    let tok = Tokenizer::load(&dir.join("tokenizer.json")).unwrap();
    let mut e = cpu(&dir, Format::Q8);
    let schema = Json::parse(
        r#"{"type":"object","properties":{
            "id":{"type":"string","pattern":"[A-Z]{2}-\\d{4}"},
            "name":{"type":"string","minLength":3,"maxLength":12},
            "joined":{"type":"string","format":"date"},
            "age":{"type":"integer","minimum":18,"maximum":65},
            "rating":{"type":"number","minimum":0,"exclusiveMaximum":5}},
          "required":["id","name","joined","age","rating"]}"#,
    )
    .unwrap();
    let format = OutputFormat::JsonSchema(JsonSchema::new(Schema::parse(&schema).unwrap()));
    let id = Regex::new("[A-Z]{2}-\\d{4}").unwrap();
    let date = Regex::new(tachyon::regex::format_pattern("date").unwrap()).unwrap();
    for seed in 1..=4 {
        let opts = GenOptions {
            max_tokens: 300,
            sampler: SamplerConfig {
                temperature: 1.0,
                seed,
                ..SamplerConfig::default()
            },
            stop_tokens: e.config().eos_ids.clone(),
            format: format.clone(),
        };
        let mut text = String::new();
        generate(
            e.as_mut(),
            &tok,
            &mut Vec::new(),
            &prompt(&tok, "Make up an employee record."),
            &opts,
            &mut |p| {
                text.push_str(p);
                true
            },
        )
        .unwrap();
        let v = Json::parse(&text).unwrap_or_else(|e| panic!("{e}: {text}"));
        let s = |k: &str| v.get(k).and_then(Json::as_str).unwrap().to_string();
        let n = |k: &str| v.get(k).and_then(Json::as_f64).unwrap();
        assert!(id.is_match(&s("id")), "{text}");
        assert!((3..=12).contains(&s("name").chars().count()), "{text}");
        assert!(date.is_match(&s("joined")), "{text}");
        assert!(
            (18.0..=65.0).contains(&n("age")) && n("age").fract() == 0.0,
            "{text}"
        );
        assert!((0.0..5.0).contains(&n("rating")), "{text}");
    }
}

/// A forced call: the reply opens with <tool_call> and the arguments are
/// constrained to the function's schema, so they always fit it.
#[test]
fn forced_tool_calls_fit_the_function_schema() {
    use tachyon::chat::parse_tool_calls;
    use tachyon::constrain::OutputFormat;
    use tachyon::json::Json;
    use tachyon::schema::{JsonSchema, Schema};
    let Some((_g, dir)) = setup() else { return };
    let tok = Tokenizer::load(&dir.join("tokenizer.json")).unwrap();
    let tpl = ChatTemplate::new(&tok).unwrap();
    let mut e = cpu(&dir, Format::Q8);
    let params = r#"{"type":"object","properties":{"city":{"type":"string"},"unit":{"enum":["celsius","fahrenheit"]}}}"#;
    let tool = Json::parse(&format!(
        r#"{{"type":"function","function":{{"name":"get_weather","parameters":{params}}}}}"#
    ))
    .unwrap();
    let schema = Json::parse(&format!(r#"{{"type":"object","properties":{{"name":{{"const":"get_weather"}},"arguments":{params}}},"required":["name","arguments"]}}"#)).unwrap();

    let mut p = tpl.encode_with_tools(
        &tok,
        &[Message::new("user", "Is it cold in Helsinki?")],
        Some(DEFAULT_SYSTEM),
        &[tool],
    );
    assert!(
        p.contains(&tok.added_token("<tool_call>").unwrap()),
        "tool markup should use the added token"
    );
    p.extend(tok.encode("<tool_call>\n", true));
    for seed in 1..=3 {
        let opts = GenOptions {
            max_tokens: 60,
            sampler: SamplerConfig {
                temperature: 1.0,
                seed,
                ..SamplerConfig::default()
            },
            stop_tokens: e.config().eos_ids.clone(),
            format: OutputFormat::JsonSchema(JsonSchema::new(Schema::parse(&schema).unwrap())),
        };
        let mut text = String::new();
        generate(e.as_mut(), &tok, &mut Vec::new(), &p, &opts, &mut |s| {
            text.push_str(s);
            true
        })
        .unwrap();
        let (_, calls) = parse_tool_calls(&format!("<tool_call>{text}</tool_call>"));
        assert_eq!(calls.len(), 1, "{text}");
        assert_eq!(calls[0].name, "get_weather");
        assert!(
            calls[0]
                .arguments
                .get("city")
                .and_then(Json::as_str)
                .is_some(),
            "{text}"
        );
        assert!(matches!(
            calls[0].arguments.get("unit").and_then(Json::as_str),
            Some("celsius" | "fahrenheit")
        ));
    }
}
