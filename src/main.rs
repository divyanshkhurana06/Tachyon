use std::io::{self, BufRead, BufReader, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use tachyon::chat::{default_system, ChatTemplate, Message};
use tachyon::config::Config;
use tachyon::constrain::OutputFormat;
use tachyon::engine::{Backend, Engine, Session};
use tachyon::generate::{generate, GenOptions, GenStats, PREFILL_CHUNK};
use tachyon::json::{obj, Json};
use tachyon::kv::BLOCK;
use tachyon::model::{CpuBackend, Weights};
use tachyon::pool::Pool;
use tachyon::sampler::{argmax, SamplerConfig};
use tachyon::scheduler::{Scheduler, SchedulerOptions};
use tachyon::server::Server;
use tachyon::tokenizer::Tokenizer;
use tachyon::weights::Format;
use tachyon::{bail, Error, Result};

const USAGE: &str = "\
tachyon: an LLM inference engine written from scratch

usage: tachyon <command> [options]

commands:
  run        generate a reply to --prompt
  chat       interactive multi-turn chat
  serve      OpenAI-compatible API and chat page, batching all users together
  loadtest   many concurrent clients against a running `serve`
  speculate  plain vs speculative decoding of --prompt, side by side
             (greedy and checked identical, or sampled with --temp)
  bench      speed and quality of each weight format
  bench-batch  decode throughput at batch sizes 1 to 64 (the engine alone)
  ppl        perplexity of the model on --file
  calibrate  measure activations on --file (default bench/calib.txt) and save
             an importance matrix that improves 4-bit quantization
  tokenize   print the token ids of --prompt, or of --file

options:
  --model DIR        model folder (default models/qwen2.5-0.5b-instruct)
  --quant F          bf16 | q8 | q4m | q4 (default q8)
  --prompt TEXT      prompt for `run`
  --system TEXT      system prompt
  --raw              no chat template: continue --prompt as plain text
  -n, --max-tokens N tokens to generate (default 512)
  --temp T           temperature, 0 = greedy (default 0.7)
  --top-k K          (default 20)
  --top-p P          (default 0.8)
  --repeat-penalty R (default 1.05)
  --seed S
  --gpu              run on the GPU (Metal) instead of the CPU
  --threads N        CPU threads (default: performance cores + half the rest)
  --ctx N            context window in tokens (default 4096)
  --file PATH        text for `ppl` / `bench` (default bench/eval.txt)
  --formats LIST     formats for `bench` (default bf16,q8,q4m,q4)
  --context N        `bench` decodes after N tokens of context (default 256)
  --backends LIST    backends for `bench` (default cpu,gpu)
  --draft F          speculative decoding with a draft model in format F
  --draft-model DIR  draft model folder (default: the same model)
  --draft-k K        tokens drafted per step (default 4)
  --lookup           `speculate` with prompt-lookup drafting instead of a model
  --port P           port for `serve` (default 8080)
  --host H           address for `serve` (default 127.0.0.1)
  --kv-tokens N      KV cache size for `serve`, shared by all users (default 16384)
  --max-seqs N       most requests `serve` runs at once (default 16)
  --json             constrain the reply to a JSON object
  --schema FILE      constrain the reply to JSON matching this JSON Schema
  --kv-f32           keep the GPU KV cache in f32 (the default, f16, uses half
                     the memory and measured lossless)
  --no-think         Qwen3: answer directly, without a <think> block
  --imatrix FILE     fit 4-bit weights to this importance matrix (from `calibrate`)
  --lora NAME=DIR    load a LoRA adapter (PEFT format); repeat for more. `serve`
                     runs a request with the adapter its `model` names;
                     `run` and `chat` use --adapter, or the first
  --adapter NAME     which adapter `run` and `chat` use
  --out FILE         where `calibrate` writes (default <model>/imatrix.bin)
  --url URL          server for `loadtest` (default http://127.0.0.1:8080)
  --clients LIST     concurrency levels for `loadtest` (default 1,4,16)
  --requests N       requests per level for `loadtest` (default 32)";

struct Args {
    cmd: String,
    model: PathBuf,
    format: Format,
    prompt: Option<String>,
    system: Option<String>,
    raw: bool,
    max_tokens: usize,
    sampler: SamplerConfig,
    threads: usize,
    ctx: usize,
    file: PathBuf,
    formats: Vec<Format>,
    gpu: bool,
    backends: Vec<bool>,
    draft: Option<Format>,
    draft_model: Option<PathBuf>,
    draft_k: usize,
    port: u16,
    host: String,
    kv_tokens: usize,
    max_seqs: usize,
    json: bool,
    schema: Option<PathBuf>,
    url: String,
    clients: Vec<usize>,
    requests: usize,
    lookup: bool,
    temp_set: bool,
    kv_f16: bool,
    context: Option<usize>,
    no_think: bool,
    imatrix: Option<PathBuf>,
    out: Option<PathBuf>,
    file_set: bool,
    lora: Vec<(String, PathBuf)>,
    adapter: Option<String>,
}

fn parse_args() -> Result<Args> {
    let mut it = std::env::args().skip(1);
    let cmd = it.next().unwrap_or_else(|| "help".into());
    let mut a = Args {
        cmd,
        model: PathBuf::from("models/qwen2.5-0.5b-instruct"),
        format: Format::Q8,
        prompt: None,
        system: None,
        raw: false,
        max_tokens: 512,
        sampler: SamplerConfig::default(),
        threads: Pool::default_threads(),
        ctx: 4096,
        file: PathBuf::from("bench/eval.txt"),
        formats: vec![Format::Bf16, Format::Q8, Format::Q4M, Format::Q4],
        gpu: false,
        backends: if cfg!(target_os = "macos") {
            vec![false, true]
        } else {
            vec![false]
        },
        draft: None,
        draft_model: None,
        draft_k: 4,
        port: 8080,
        host: "127.0.0.1".into(),
        kv_tokens: 16384,
        max_seqs: 16,
        json: false,
        schema: None,
        url: "http://127.0.0.1:8080".into(),
        clients: vec![1, 4, 16],
        requests: 32,
        lookup: false,
        temp_set: false,
        kv_f16: true,
        context: None,
        no_think: false,
        imatrix: None,
        out: None,
        file_set: false,
        lora: Vec::new(),
        adapter: None,
    };
    while let Some(flag) = it.next() {
        let mut value = || {
            it.next()
                .ok_or_else(|| Error(format!("{flag} needs a value")))
        };
        fn num<T: std::str::FromStr>(flag: &str, v: String) -> Result<T> {
            v.parse()
                .map_err(|_| Error(format!("{flag}: invalid number {v:?}")))
        }
        let format = |v: &str| {
            Format::parse(v)
                .ok_or_else(|| Error(format!("unknown format {v:?} (bf16, q8, q4m, q4)")))
        };
        match flag.as_str() {
            "--model" | "-m" => a.model = value()?.into(),
            "--quant" | "-q" => a.format = format(&value()?)?,
            "--prompt" | "-p" => a.prompt = Some(value()?),
            "--system" => a.system = Some(value()?),
            "--raw" => a.raw = true,
            "--max-tokens" | "-n" => a.max_tokens = num(&flag, value()?)?,
            "--temp" => {
                a.sampler.temperature = num(&flag, value()?)?;
                a.temp_set = true;
            }
            "--lookup" => a.lookup = true,
            "--kv-f16" => a.kv_f16 = true,
            "--kv-f32" => a.kv_f16 = false,
            "--context" => a.context = Some(num(&flag, value()?)?),
            "--no-think" => a.no_think = true,
            "--imatrix" => a.imatrix = Some(value()?.into()),
            "--lora" => {
                let v = value()?;
                let (name, dir) = match v.split_once('=') {
                    Some((n, d)) => (n.to_string(), PathBuf::from(d)),
                    None => {
                        let dir = PathBuf::from(&v);
                        let name = dir
                            .file_name()
                            .map_or(v.clone(), |n| n.to_string_lossy().into_owned());
                        (name, dir)
                    }
                };
                a.lora.push((name, dir));
            }
            "--adapter" => a.adapter = Some(value()?),
            "--out" => a.out = Some(value()?.into()),
            "--top-k" => a.sampler.top_k = num(&flag, value()?)?,
            "--top-p" => a.sampler.top_p = num(&flag, value()?)?,
            "--repeat-penalty" => a.sampler.repetition_penalty = num(&flag, value()?)?,
            "--seed" => a.sampler.seed = num(&flag, value()?)?,
            "--threads" | "-t" => a.threads = num(&flag, value()?)?,
            "--ctx" => a.ctx = num(&flag, value()?)?,
            "--file" => {
                a.file = value()?.into();
                a.file_set = true;
            }
            "--formats" => a.formats = value()?.split(',').map(format).collect::<Result<_>>()?,
            "--gpu" => a.gpu = true,
            "--cpu" => a.gpu = false,
            "--backends" => {
                a.backends = value()?
                    .split(',')
                    .map(|b| match b {
                        "cpu" => Ok(false),
                        "gpu" | "metal" => Ok(true),
                        _ => bail!("unknown backend {b:?} (cpu, gpu)"),
                    })
                    .collect::<Result<_>>()?
            }
            "--draft" => a.draft = Some(format(&value()?)?),
            "--draft-model" => a.draft_model = Some(value()?.into()),
            "--draft-k" => a.draft_k = num(&flag, value()?)?,
            "--port" => a.port = num(&flag, value()?)?,
            "--host" => a.host = value()?,
            "--kv-tokens" => a.kv_tokens = num(&flag, value()?)?,
            "--max-seqs" => a.max_seqs = num(&flag, value()?)?,
            "--json" => a.json = true,
            "--schema" => a.schema = Some(value()?.into()),
            "--url" => a.url = value()?,
            "--requests" => a.requests = num(&flag, value()?)?,
            "--clients" => {
                a.clients = value()?
                    .split(',')
                    .map(|c| {
                        c.parse()
                            .map_err(|_| Error(format!("--clients: bad number {c:?}")))
                    })
                    .collect::<Result<_>>()?
            }
            "-h" | "--help" => a.cmd = "help".into(),
            _ => bail!("unknown option {flag} (see `tachyon help`)"),
        }
    }
    Ok(a)
}

fn main() -> ExitCode {
    let result = parse_args().and_then(|a| match a.cmd.as_str() {
        "run" => run(&a),
        "chat" => chat(&a),
        "bench" => bench(&a),
        "ppl" => perplexity_cmd(&a),
        "serve" => serve(&a),
        "speculate" => speculate(&a),
        "loadtest" => loadtest(&a),
        "bench-batch" => bench_batch(&a),
        "calibrate" => calibrate(&a),
        "tokenize" => tokenize(&a),
        "help" => {
            println!("{USAGE}");
            Ok(())
        }
        other => bail!("unknown command {other:?} (see `tachyon help`)"),
    });
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

struct Loaded {
    engine: Box<dyn Engine>,
    tok: Tokenizer,
    cfg: Config,
    bytes: usize,
}

fn load(a: &Args, format: Format, gpu: bool, pool: &Arc<Pool>) -> Result<Loaded> {
    load_from(a, &a.model, format, gpu, pool)
}

struct LoadedBackend {
    backend: Box<dyn Backend>,
    tok: Tokenizer,
    cfg: Config,
    bytes: usize,
}

/// Loads a model onto the CPU or GPU with a KV cache of `blocks` blocks.
fn load_backend(
    a: &Args,
    dir: &std::path::Path,
    format: Format,
    gpu: bool,
    pool: &Arc<Pool>,
    blocks: usize,
) -> Result<LoadedBackend> {
    let cfg = Config::load(dir)?;
    let tok = Tokenizer::load(&dir.join("tokenizer.json"))?;
    let started = Instant::now();
    let importance = match &a.imatrix {
        Some(path) => Some(tachyon::imatrix::Importance::load(path, &cfg)?.normalized()),
        None => None,
    };
    let weights = Weights::load_with(dir, &cfg, format, importance.as_ref())?;
    let bytes = weights.bytes();
    let mut backend: Box<dyn Backend> = if gpu {
        gpu_backend(cfg.clone(), weights, blocks, a.ctx, a.kv_f16)?
    } else {
        Box::new(CpuBackend::new(cfg.clone(), weights, pool.clone(), blocks))
    };
    if !a.lora.is_empty() && dir == a.model {
        let adapters = tachyon::lora::Adapters::load(&cfg, &a.lora)?;
        eprintln!(
            "loaded LoRA adapters {} (rank {}, {:.0} MB)",
            adapters.names.join(", "),
            adapters.rank,
            adapters.bytes() as f64 / 1e6
        );
        backend.load_adapters(adapters)?;
    }
    eprintln!(
        "loaded {} ({:.0}M params, {}, {:.0} MB) in {:.2}s on {}",
        dir.file_name()
            .map_or("model".into(), |n| n.to_string_lossy()),
        cfg.params() as f64 / 1e6,
        format.name(),
        bytes as f64 / 1e6,
        started.elapsed().as_secs_f64(),
        backend.name()
    );
    Ok(LoadedBackend {
        backend,
        tok,
        cfg,
        bytes,
    })
}

/// Loads a model as a single sequence with a `--ctx`-token cache.
fn load_from(
    a: &Args,
    dir: &std::path::Path,
    format: Format,
    gpu: bool,
    pool: &Arc<Pool>,
) -> Result<Loaded> {
    let m = load_backend(a, dir, format, gpu, pool, a.ctx.div_ceil(BLOCK))?;
    let mut session = Session::new(m.backend);
    if !a.lora.is_empty() && dir == a.model {
        let id = match &a.adapter {
            Some(name) => a.lora.iter().position(|(n, _)| n == name).ok_or_else(|| {
                Error(format!("--adapter {name}: no --lora adapter has that name"))
            })?,
            None => 0,
        };
        session.set_adapter(id as u32 + 1);
    }
    Ok(Loaded {
        engine: Box::new(session),
        tok: m.tok,
        cfg: m.cfg,
        bytes: m.bytes,
    })
}

#[cfg(target_os = "macos")]
fn gpu_backend(
    cfg: Config,
    weights: Weights,
    blocks: usize,
    max_seq: usize,
    kv_f16: bool,
) -> Result<Box<dyn Backend>> {
    Ok(Box::new(tachyon::gpu::GpuBackend::new(
        cfg, weights, blocks, max_seq, kv_f16,
    )?))
}

#[cfg(not(target_os = "macos"))]
fn gpu_backend(_: Config, _: Weights, _: usize, _: usize, _: bool) -> Result<Box<dyn Backend>> {
    bail!("the GPU backend needs macOS (Metal)")
}

fn stop_tokens(cfg: &Config, tok: &Tokenizer) -> Vec<u32> {
    let mut stop = cfg.eos_ids.clone();
    for s in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>"] {
        if let Some(id) = tok.added_token(s) {
            if !stop.contains(&id) {
                stop.push(id);
            }
        }
    }
    stop
}

fn print_stats(s: &GenStats) {
    eprintln!(
        "\n[{} prompt tokens ({} cached) at {:.0} tok/s · {} generated at {:.1} tok/s · {}]",
        s.prompt_tokens,
        s.cached_tokens,
        s.prefill_tps(),
        s.tokens.len(),
        s.decode_tps(),
        s.finish.as_str()
    );
}

fn print_piece(piece: &str) -> bool {
    print!("{piece}");
    io::stdout().flush().is_ok()
}

fn run(a: &Args) -> Result<()> {
    let prompt = a
        .prompt
        .as_deref()
        .ok_or_else(|| Error::msg("run needs --prompt"))?;
    let pool = Arc::new(Pool::new(a.threads));
    let mut m = load(a, a.format, a.gpu, &pool)?;
    let tokens = if a.raw {
        m.tok.encode(prompt, false)
    } else {
        chat_prompt(a, &m.tok, &m.cfg, prompt)?
    };
    let opts = GenOptions {
        max_tokens: a.max_tokens,
        sampler: a.sampler.clone(),
        stop_tokens: stop_tokens(&m.cfg, &m.tok),
        format: output_format(a)?,
    };
    if a.raw {
        print!("{prompt}");
    }
    let stats = generate(
        m.engine.as_mut(),
        &m.tok,
        &mut Vec::new(),
        &tokens,
        &opts,
        &mut print_piece,
    )?;
    println!();
    print_stats(&stats);
    Ok(())
}

fn chat(a: &Args) -> Result<()> {
    let pool = Arc::new(Pool::new(a.threads));
    let mut m = load(a, a.format, a.gpu, &pool)?;
    let tpl = ChatTemplate::new(&m.tok)?;
    let opts = GenOptions {
        max_tokens: a.max_tokens,
        sampler: a.sampler.clone(),
        stop_tokens: stop_tokens(&m.cfg, &m.tok),
        format: output_format(a)?,
    };

    // The conversation as tokens. Each turn appends to it, and generate()
    // finds everything but the new user message already in the KV cache.
    let system = a.system.as_deref().or(default_system(&m.cfg.model_type));
    let mut convo = Vec::new();
    tpl.begin(&m.tok, &mut convo, system);
    let mut cache = Vec::new();
    eprintln!("chat with the model; /reset clears the conversation, Ctrl-D quits\n");

    let stdin = io::stdin();
    loop {
        eprint!("> ");
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "/reset" {
            convo.truncate(0);
            tpl.begin(&m.tok, &mut convo, system);
            eprintln!("(conversation cleared)");
            continue;
        }
        let mut prompt = convo.clone();
        tpl.push(&m.tok, &mut prompt, "user", line);
        tpl.open_assistant(&m.tok, &mut prompt);
        if a.no_think {
            tpl.skip_thinking(&m.tok, &mut prompt);
        }
        if prompt.len() + a.max_tokens > m.engine.max_seq() {
            eprintln!("(context window full; use /reset)");
            continue;
        }
        let stats = generate(
            m.engine.as_mut(),
            &m.tok,
            &mut cache,
            &prompt,
            &opts,
            &mut print_piece,
        )?;
        println!();
        print_stats(&stats);
        convo = prompt;
        convo.extend_from_slice(&stats.tokens);
        tpl.close_turn(&mut convo);
    }
    Ok(())
}

/// Perplexity: exp of the average negative log-likelihood the model gives
/// each next token of the text. Lower is better; it is the standard way to
/// see how much quality a quantization format costs.
fn perplexity(engine: &mut dyn Engine, tokens: &[u32]) -> f64 {
    engine.truncate(0);
    let mut nll = 0.0;
    let mut count = 0;
    for start in (0..tokens.len() - 1).step_by(PREFILL_CHUNK) {
        let chunk = &tokens[start..(start + PREFILL_CHUNK).min(tokens.len() - 1)];
        let vocab = engine.config().vocab;
        let logits = engine.forward(chunk, true);
        for (i, row) in logits.chunks_exact(vocab).enumerate() {
            let target = tokens[start + i + 1] as usize;
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let lse = max as f64
                + row
                    .iter()
                    .map(|&l| ((l - max) as f64).exp())
                    .sum::<f64>()
                    .ln();
            nll += lse - row[target] as f64;
            count += 1;
        }
    }
    (nll / count as f64).exp()
}

fn eval_tokens(a: &Args, tok: &Tokenizer, max: usize) -> Result<Vec<u32>> {
    let text = std::fs::read_to_string(&a.file)
        .map_err(|e| Error(format!("cannot read {}: {e}", a.file.display())))?;
    let mut tokens = tok.encode(&text, false);
    tokens.truncate(max);
    if tokens.len() < 2 {
        bail!("{} is too short", a.file.display());
    }
    Ok(tokens)
}

/// Token ids as a list, the way `llama-tokenize --ids` prints them, so the
/// two tokenizers can be compared on any text.
fn tokenize(a: &Args) -> Result<()> {
    let tok = Tokenizer::load(&a.model.join("tokenizer.json"))?;
    let text = match &a.prompt {
        Some(p) => p.clone(),
        None => std::fs::read_to_string(&a.file)
            .map_err(|e| Error(format!("cannot read {}: {e}", a.file.display())))?,
    };
    let ids: Vec<String> = tok
        .encode(&text, false)
        .iter()
        .map(u32::to_string)
        .collect();
    println!("[{}]", ids.join(", "));
    Ok(())
}

fn perplexity_cmd(a: &Args) -> Result<()> {
    let pool = Arc::new(Pool::new(a.threads));
    let mut m = load(a, a.format, a.gpu, &pool)?;
    let tokens = eval_tokens(a, &m.tok, m.engine.max_seq())?;
    let started = Instant::now();
    let ppl = perplexity(m.engine.as_mut(), &tokens);
    println!(
        "perplexity {ppl:.3} over {} tokens ({:.1}s)",
        tokens.len(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

/// For each format: load time, weight size, prefill and decode speed, and
/// perplexity, as a Markdown table.
fn bench(a: &Args) -> Result<()> {
    // Decode is measured after `--context` tokens (default 256): attention
    // reads every cached position, so long contexts are slower.
    let prefill_len = a.context.unwrap_or(256);
    const DECODE: usize = 128;
    const RUNS: usize = 3;
    let pool = Arc::new(Pool::new(a.threads));
    let mut rows = Vec::new();
    for &gpu in &a.backends {
        for &format in &a.formats {
            let started = Instant::now();
            let mut m = load(a, format, gpu, &pool)?;
            let load_secs = started.elapsed().as_secs_f64();
            let bytes = m.bytes;
            let tokens = eval_tokens(a, &m.tok, 1024)?;
            // The eval text, repeated as needed to fill the context.
            let context: Vec<u32> = tokens.iter().copied().cycle().take(prefill_len).collect();
            if prefill_len + DECODE > m.engine.max_seq() {
                bail!(
                    "--context {prefill_len} plus {DECODE} decoded tokens exceeds --ctx {}",
                    m.engine.max_seq()
                );
            }
            let e = m.engine.as_mut();

            // warm-up, so page faults and cold caches don't count
            e.truncate(0);
            e.forward(&tokens[..PREFILL_CHUNK], false);

            // Best of RUNS: other programs on the machine only ever slow a
            // run down, so the fastest run is the closest to the engine's speed.
            let (mut prefill, mut decode) = (0f64, 0f64);
            for _ in 0..RUNS {
                e.truncate(0);
                let t0 = Instant::now();
                let mut next = 0;
                for chunk in context[..prefill_len].chunks(PREFILL_CHUNK) {
                    next = argmax(e.forward(chunk, false));
                }
                prefill = prefill.max(prefill_len as f64 / t0.elapsed().as_secs_f64());

                let t0 = Instant::now();
                for _ in 0..DECODE {
                    next = argmax(e.forward(&[next], false));
                }
                decode = decode.max(DECODE as f64 / t0.elapsed().as_secs_f64());
            }

            let ppl = perplexity(e, &tokens);
            eprintln!(
                "{}: prefill {prefill:.0} tok/s, decode {decode:.1} tok/s, ppl {ppl:.3}",
                format.name()
            );
            rows.push((
                e.name(),
                bytes,
                load_secs,
                prefill,
                decode,
                ppl,
                tokens.len(),
            ));
        }
    }
    println!("\n| backend | weights | load | prefill | decode | perplexity |");
    println!("|---|---:|---:|---:|---:|---:|");
    for (name, bytes, load, prefill, decode, ppl, _) in &rows {
        println!("| {name} | {:.0} MB | {load:.1} s | {prefill:.0} tok/s | {decode:.1} tok/s | {ppl:.3} |", *bytes as f64 / 1e6);
    }
    if let Some(r) = rows.first() {
        println!(
            "\nprefill {prefill_len} tokens, decode {DECODE} tokens (best of {RUNS} runs), perplexity over {} tokens of {}",
            r.6,
            a.file.display()
        );
    }
    Ok(())
}

fn chat_prompt(a: &Args, tok: &Tokenizer, cfg: &Config, prompt: &str) -> Result<Vec<u32>> {
    if a.raw {
        return Ok(tok.encode(prompt, false));
    }
    let tpl = ChatTemplate::new(tok)?;
    let system = a.system.as_deref().or(default_system(&cfg.model_type));
    let mut tokens = tpl.encode(tok, &[Message::new("user", prompt)], system);
    if a.no_think {
        tpl.skip_thinking(tok, &mut tokens);
    }
    Ok(tokens)
}

/// Runs the same greedy generation plainly and speculatively, checks the
/// outputs match, and reports the speedup.
fn speculate(a: &Args) -> Result<()> {
    use tachyon::speculative::{generate_speculative, Drafter, LookupDrafter, ModelDrafter};
    let prompt = a
        .prompt
        .as_deref()
        .ok_or_else(|| Error::msg("speculate needs --prompt"))?;
    let pool = Arc::new(Pool::new(a.threads));
    let mut target = load(a, a.format, a.gpu, &pool)?;
    let tokens = chat_prompt(a, &target.tok, &target.cfg, prompt)?;
    let stop = stop_tokens(&target.cfg, &target.tok);
    // Greedy unless a temperature was asked for: then the outputs can be
    // compared token for token.
    let sampling = if a.temp_set {
        a.sampler.clone()
    } else {
        SamplerConfig::greedy()
    };
    let greedy = sampling.temperature <= 0.0;
    let opts = GenOptions {
        max_tokens: a.max_tokens,
        sampler: sampling.clone(),
        stop_tokens: stop.clone(),
        format: OutputFormat::Text,
    };

    let mode = if greedy {
        "greedy".into()
    } else {
        format!("sampled at temperature {}", sampling.temperature)
    };
    eprintln!("\n--- plain decoding, {mode}, on {}", target.engine.name());
    let plain = generate(
        target.engine.as_mut(),
        &target.tok,
        &mut Vec::new(),
        &tokens,
        &opts,
        &mut print_piece,
    )?;
    println!();
    print_stats(&plain);

    let mut draft_model;
    let mut lookup = LookupDrafter::default();
    let mut model_drafter;
    let drafter: &mut dyn Drafter = if a.lookup {
        &mut lookup
    } else {
        let draft_dir = a.draft_model.clone().unwrap_or_else(|| a.model.clone());
        draft_model = load_from(a, &draft_dir, a.draft.unwrap_or(Format::Q4), a.gpu, &pool)?;
        model_drafter = ModelDrafter::new(
            draft_model.engine.as_mut(),
            sampling.clone(),
            target.tok.len(),
        );
        &mut model_drafter
    };
    eprintln!(
        "\n--- speculative: {} drafts up to {} tokens per step",
        drafter.name(),
        a.draft_k
    );
    let spec = generate_speculative(
        target.engine.as_mut(),
        drafter,
        &target.tok,
        &tokens,
        a.draft_k,
        a.max_tokens,
        &stop,
        &sampling,
        &mut |p| if greedy { true } else { print_piece(p) },
    )?;
    if !greedy {
        println!();
    }
    let same = spec.tokens == plain.tokens;
    eprintln!(
        "{} tokens at {:.1} tok/s ({:.2}x) · {:.0}% of drafts accepted · {:.2} tokens per target pass{}",
        spec.tokens.len(),
        spec.decode_tps(),
        spec.decode_tps() / plain.decode_tps(),
        spec.acceptance() * 100.0,
        spec.tokens_per_pass(),
        match (greedy, same) {
            (true, true) => " · output identical to plain decoding",
            (true, false) => " · output DIFFERS from plain decoding",
            (false, _) => " · sampled, so outputs differ by design",
        }
    );
    if greedy && !same {
        bail!("speculative output differs from plain greedy output");
    }
    Ok(())
}

fn serve(a: &Args) -> Result<()> {
    let pool = Arc::new(Pool::new(a.threads));
    let blocks = a.kv_tokens.max(a.ctx).div_ceil(BLOCK);
    let m = load_backend(a, &a.model, a.format, a.gpu, &pool, blocks)?;
    let template = ChatTemplate::new(&m.tok)?;
    let stop = stop_tokens(&m.cfg, &m.tok);
    let backend = m.backend.name();
    let tok = Arc::new(m.tok);
    let scheduler = Scheduler::spawn(
        m.backend,
        tok.clone(),
        SchedulerOptions {
            max_seqs: a.max_seqs.min(128),
            max_seq_len: a.ctx,
        },
    );
    let server = Server {
        scheduler,
        tok,
        template,
        stop,
        model_id: a
            .model
            .file_name()
            .map_or("model".into(), |n| n.to_string_lossy().into_owned()),
        backend,
        params: m.cfg.params(),
        context: a.ctx,
        system: a
            .system
            .clone()
            .or(default_system(&m.cfg.model_type).map(str::to_string)),
        kv_tokens: blocks * BLOCK,
        defaults: a.sampler.clone(),
        max_tokens: a.max_tokens,
        adapters: a.lora.iter().map(|(n, _)| n.clone()).collect(),
    };
    tachyon::server::serve(server, &format!("{}:{}", a.host, a.port))
}

const LOAD_PROMPTS: [&str; 8] = [
    "Explain how a hash map works.",
    "Write a short poem about the ocean.",
    "What are the main causes of inflation?",
    "Describe the water cycle to a child.",
    "Give three tips for writing clean code.",
    "Summarize the plot of a heist movie you invent.",
    "How does a CPU cache improve performance?",
    "Write a polite email declining a meeting.",
];

struct Sample {
    ttft: f64,
    total: f64,
    tokens: usize,
    mean_batch: f64,
}

/// One streamed chat completion, timed from the client's side.
fn one_request(addr: &str, prompt: &str, max_tokens: usize, seed: u64) -> Result<Sample> {
    let body = obj([
        (
            "messages",
            Json::Arr(vec![obj([
                ("role", "user".into()),
                ("content", prompt.into()),
            ])]),
        ),
        ("max_tokens", max_tokens.into()),
        ("stream", true.into()),
        ("ignore_eos", true.into()),
        ("seed", seed.into()),
    ])
    .to_string();
    let started = Instant::now();
    let mut conn =
        TcpStream::connect(addr).map_err(|e| Error(format!("cannot connect to {addr}: {e}")))?;
    write!(conn, "POST /v1/chat/completions HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len())?;
    let mut lines = BufReader::new(conn).lines();
    let status = lines.next().transpose()?.unwrap_or_default();
    if !status.contains(" 200 ") {
        let rest: Vec<String> = lines.map_while(|l| l.ok()).collect();
        bail!("{status}: {}", rest.last().map_or("", String::as_str));
    }
    let (mut ttft, mut tokens, mut mean_batch) = (None, 0, 0.0);
    for line in lines {
        let line = line?;
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data == "[DONE]" {
            break;
        }
        let j = Json::parse(data)?;
        if let Some(e) = j.get("error") {
            bail!("server error: {e}");
        }
        let content = j
            .get("choices")
            .and_then(Json::as_array)
            .and_then(|c| c.first()?.get("delta")?.get("content")?.as_str());
        if content.is_some_and(|c| !c.is_empty()) {
            ttft.get_or_insert(started.elapsed().as_secs_f64());
        }
        if let Some(n) = j
            .get("usage")
            .and_then(|u| u.get("completion_tokens"))
            .and_then(Json::as_usize)
        {
            tokens = n;
            mean_batch = j
                .get("timings")
                .and_then(|t| t.get("mean_batch"))
                .and_then(Json::as_f64)
                .unwrap_or(0.0);
        }
    }
    let total = started.elapsed().as_secs_f64();
    Ok(Sample {
        ttft: ttft.unwrap_or(total),
        total,
        tokens,
        mean_batch,
    })
}

fn percentile(xs: &mut [f64], p: f64) -> f64 {
    xs.sort_by(f64::total_cmp);
    xs[((xs.len() - 1) as f64 * p).round() as usize]
}

/// Hammers a running server with concurrent streaming clients and reports
/// total throughput, per-user speed and latency at each concurrency level.
fn loadtest(a: &Args) -> Result<()> {
    let addr = a
        .url
        .trim_start_matches("http://")
        .trim_end_matches('/')
        .to_string();
    let mut rows = Vec::new();
    for &clients in &a.clients {
        let per_client = a.requests.div_ceil(clients).max(1);
        let started = Instant::now();
        let results: Vec<Result<Sample>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..clients)
                .map(|c| {
                    let addr = &addr;
                    s.spawn(move || {
                        (0..per_client)
                            .map(|r| {
                                let i = c * per_client + r;
                                one_request(
                                    addr,
                                    LOAD_PROMPTS[i % LOAD_PROMPTS.len()],
                                    a.max_tokens,
                                    a.sampler.seed.wrapping_add(i as u64),
                                )
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect()
        });
        let wall = started.elapsed().as_secs_f64();
        let samples = results.into_iter().collect::<Result<Vec<_>>>()?;
        let tokens: usize = samples.iter().map(|s| s.tokens).sum();
        let mut ttft: Vec<f64> = samples.iter().map(|s| s.ttft).collect();
        let mut latency: Vec<f64> = samples.iter().map(|s| s.total).collect();
        let mut speed: Vec<f64> = samples
            .iter()
            .map(|s| s.tokens.saturating_sub(1) as f64 / (s.total - s.ttft).max(1e-9))
            .collect();
        let batch = samples.iter().map(|s| s.mean_batch).sum::<f64>() / samples.len() as f64;
        let row = (
            clients,
            samples.len(),
            tokens as f64 / wall,
            percentile(&mut speed, 0.5),
            batch,
            percentile(&mut ttft, 0.5),
            percentile(&mut ttft, 0.99),
            percentile(&mut latency, 0.5),
            percentile(&mut latency, 0.99),
        );
        eprintln!(
            "{clients} clients: {:.0} tok/s total, {:.1} tok/s each, TTFT p50 {:.2}s",
            row.2, row.3, row.5
        );
        rows.push(row);
    }
    println!("\n| clients | requests | total throughput | per user | mean batch | TTFT p50 | TTFT p99 | latency p50 | latency p99 |");
    println!("|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    for (c, n, total, each, batch, t50, t99, l50, l99) in rows {
        println!("| {c} | {n} | {total:.0} tok/s | {each:.1} tok/s | {batch:.1} | {t50:.2} s | {t99:.2} s | {l50:.1} s | {l99:.1} s |");
    }
    println!(
        "\n{} tokens per request (ignore_eos), streamed",
        a.max_tokens
    );
    Ok(())
}

/// Decode throughput of the engine alone as the batch grows: each step
/// feeds one token to each of `b` sequences that already hold a prompt.
fn bench_batch(a: &Args) -> Result<()> {
    use tachyon::kv::{Batch, BlockPool, Logits};
    const PROMPT: usize = 128;
    const STEPS: usize = 16;
    let sizes = [1, 4, 8, 16, 32, 64];
    let pool = Arc::new(Pool::new(a.threads));
    let mut rows = Vec::new();
    for &gpu in &a.backends {
        let blocks = (sizes[sizes.len() - 1] * (PROMPT + STEPS + 1)).div_ceil(BLOCK) + 64;
        let mut m = load_backend(a, &a.model, a.format, gpu, &pool, blocks)?;
        let prompt = eval_tokens(a, &m.tok, PROMPT)?;
        let name = m.backend.name();
        let budget = m.backend.max_batch();
        for &b in &sizes {
            let mut kv = BlockPool::new(blocks);
            let mut seqs: Vec<_> = (0..b).map(|_| kv.start(&[])).collect();
            let mut batch = Batch::default();
            // With --lora, sequence i runs adapter i mod n: every step mixes them.
            let adapter = |i: usize| match a.lora.len() {
                0 => 0,
                n => (i % n) as u32 + 1,
            };
            for (i, seq) in seqs.iter_mut().enumerate() {
                assert!(kv.reserve(seq, PROMPT + STEPS + 1));
                for chunk in prompt.chunks(budget) {
                    batch.clear();
                    batch.push_with(&seq.blocks, seq.len, chunk, Logits::None, adapter(i));
                    seq.len += chunk.len();
                    m.backend.forward(&batch);
                }
            }
            let mut best = f64::MAX;
            for step in 0..STEPS {
                batch.clear();
                for (i, seq) in seqs.iter_mut().enumerate() {
                    batch.push_with(
                        &seq.blocks,
                        seq.len,
                        &[prompt[step % prompt.len()]],
                        Logits::Last,
                        adapter(i),
                    );
                    seq.len += 1;
                }
                let t0 = Instant::now();
                m.backend.forward(&batch);
                best = best.min(t0.elapsed().as_secs_f64());
            }
            eprintln!(
                "{name}: batch {b:>2}: {:.1} ms/step, {:.0} tok/s",
                best * 1e3,
                b as f64 / best
            );
            rows.push((name.clone(), b, best, b as f64 / best));
        }
    }
    println!("\n| backend | batch | step | throughput |");
    println!("|---|---:|---:|---:|");
    for (name, b, step, tps) in rows {
        println!("| {name} | {b} | {:.1} ms | {tps:.0} tok/s |", step * 1e3);
    }
    Ok(())
}

/// Runs calibration text through the model in bf16 on the CPU, recording
/// each linear layer's mean squared input per channel, and saves it.
fn calibrate(a: &Args) -> Result<()> {
    use tachyon::imatrix::Importance;
    use tachyon::kv::{Batch, Logits};
    let file = if a.file_set {
        a.file.clone()
    } else {
        PathBuf::from("bench/calib.txt")
    };
    let out = a.out.clone().unwrap_or_else(|| a.model.join("imatrix.bin"));
    let cfg = Config::load(&a.model)?;
    let tok = Tokenizer::load(&a.model.join("tokenizer.json"))?;
    let text = std::fs::read_to_string(&file)
        .map_err(|e| Error(format!("cannot read {}: {e}", file.display())))?;
    let tokens = tok.encode(&text, false);
    let ctx = tokens.len().min(cfg.max_positions);
    let blocks = ctx.div_ceil(BLOCK);
    let pool = Arc::new(Pool::new(a.threads));
    let mut backend = CpuBackend::new(
        cfg.clone(),
        Weights::load(&a.model, &cfg, tachyon::weights::Format::Bf16)?,
        pool,
        blocks,
    );
    backend.calibration = Some(Importance::new(&cfg));
    let started = Instant::now();
    let table: Vec<u32> = (0..blocks as u32).collect();
    let mut batch = Batch::default();
    for (i, chunk) in tokens[..ctx].chunks(PREFILL_CHUNK).enumerate() {
        batch.clear();
        batch.push(&table, i * PREFILL_CHUNK, chunk, Logits::None);
        backend.forward(&batch);
    }
    let importance = backend.calibration.take().unwrap();
    importance.save(&out)?;
    eprintln!(
        "measured {} tokens of {} in {:.1}s; wrote {}",
        importance.tokens,
        file.display(),
        started.elapsed().as_secs_f64(),
        out.display()
    );
    Ok(())
}

/// `--schema` or `--json`, as an output constraint.
fn output_format(a: &Args) -> Result<OutputFormat> {
    if let Some(path) = &a.schema {
        let text = std::fs::read_to_string(path)
            .map_err(|e| Error(format!("cannot read {}: {e}", path.display())))?;
        let schema = tachyon::schema::Schema::parse(&Json::parse(&text)?)?;
        return Ok(OutputFormat::JsonSchema(tachyon::schema::JsonSchema::new(
            schema,
        )));
    }
    Ok(if a.json {
        OutputFormat::JsonObject
    } else {
        OutputFormat::Text
    })
}
