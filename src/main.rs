use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use tachyon::chat::{ChatTemplate, Message, DEFAULT_SYSTEM};
use tachyon::config::Config;
use tachyon::engine::Engine;
use tachyon::generate::{generate, GenOptions, GenStats, PREFILL_CHUNK};
use tachyon::model::{CpuEngine, Weights};
use tachyon::pool::Pool;
use tachyon::sampler::{argmax, SamplerConfig};
use tachyon::tokenizer::Tokenizer;
use tachyon::weights::Format;
use tachyon::{bail, Error, Result};

const USAGE: &str = "\
tachyon: an LLM inference engine written from scratch

usage: tachyon <command> [options]

commands:
  run        generate a reply to --prompt
  chat       interactive multi-turn chat
  serve      OpenAI-compatible API and chat page (--port, default 8080)
  speculate  plain vs speculative decoding of --prompt, side by side
  bench      speed and quality of each weight format
  ppl        perplexity of the model on --file

options:
  --model DIR        model folder (default models/qwen2.5-0.5b-instruct)
  --quant F          bf16 | q8 | q4 (default q8)
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
  --formats LIST     formats for `bench` (default bf16,q8,q4)
  --backends LIST    backends for `bench` (default cpu,gpu)
  --draft F          speculative decoding with a draft model in format F
  --draft-model DIR  draft model folder (default: the same model)
  --draft-k K        tokens drafted per step (default 4)
  --port P           port for `serve` (default 8080)
  --host H           address for `serve` (default 127.0.0.1)";

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
        formats: vec![Format::Bf16, Format::Q8, Format::Q4],
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
            Format::parse(v).ok_or_else(|| Error(format!("unknown format {v:?} (bf16, q8, q4)")))
        };
        match flag.as_str() {
            "--model" | "-m" => a.model = value()?.into(),
            "--quant" | "-q" => a.format = format(&value()?)?,
            "--prompt" | "-p" => a.prompt = Some(value()?),
            "--system" => a.system = Some(value()?),
            "--raw" => a.raw = true,
            "--max-tokens" | "-n" => a.max_tokens = num(&flag, value()?)?,
            "--temp" => a.sampler.temperature = num(&flag, value()?)?,
            "--top-k" => a.sampler.top_k = num(&flag, value()?)?,
            "--top-p" => a.sampler.top_p = num(&flag, value()?)?,
            "--repeat-penalty" => a.sampler.repetition_penalty = num(&flag, value()?)?,
            "--seed" => a.sampler.seed = num(&flag, value()?)?,
            "--threads" | "-t" => a.threads = num(&flag, value()?)?,
            "--ctx" => a.ctx = num(&flag, value()?)?,
            "--file" => a.file = value()?.into(),
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

fn load_from(
    a: &Args,
    dir: &std::path::Path,
    format: Format,
    gpu: bool,
    pool: &Arc<Pool>,
) -> Result<Loaded> {
    let cfg = Config::load(dir)?;
    let tok = Tokenizer::load(&dir.join("tokenizer.json"))?;
    let started = Instant::now();
    let weights = Weights::load(dir, &cfg, format)?;
    let bytes = weights.bytes();
    let engine: Box<dyn Engine> = if gpu {
        gpu_engine(cfg.clone(), weights, a.ctx)?
    } else {
        Box::new(CpuEngine::new(cfg.clone(), weights, pool.clone(), a.ctx))
    };
    eprintln!(
        "loaded {} ({:.0}M params, {}, {:.0} MB) in {:.2}s on {}",
        a.model
            .file_name()
            .map_or("model".into(), |n| n.to_string_lossy()),
        cfg.params() as f64 / 1e6,
        format.name(),
        bytes as f64 / 1e6,
        started.elapsed().as_secs_f64(),
        engine.name()
    );
    Ok(Loaded {
        engine,
        tok,
        cfg,
        bytes,
    })
}

#[cfg(target_os = "macos")]
fn gpu_engine(cfg: Config, weights: Weights, ctx: usize) -> Result<Box<dyn Engine>> {
    Ok(Box::new(tachyon::gpu::GpuEngine::new(cfg, weights, ctx)?))
}

#[cfg(not(target_os = "macos"))]
fn gpu_engine(_: Config, _: Weights, _: usize) -> Result<Box<dyn Engine>> {
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
        let tpl = ChatTemplate::new(&m.tok)?;
        tpl.encode(
            &m.tok,
            &[Message::new("user", prompt)],
            Some(a.system.as_deref().unwrap_or(DEFAULT_SYSTEM)),
        )
    };
    let opts = GenOptions {
        max_tokens: a.max_tokens,
        sampler: a.sampler.clone(),
        stop_tokens: stop_tokens(&m.cfg, &m.tok),
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
    };

    // The conversation as tokens. Each turn appends to it, and generate()
    // finds everything but the new user message already in the KV cache.
    let mut convo = Vec::new();
    tpl.push(
        &m.tok,
        &mut convo,
        "system",
        a.system.as_deref().unwrap_or(DEFAULT_SYSTEM),
    );
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
            tpl.push(
                &m.tok,
                &mut convo,
                "system",
                a.system.as_deref().unwrap_or(DEFAULT_SYSTEM),
            );
            eprintln!("(conversation cleared)");
            continue;
        }
        let mut prompt = convo.clone();
        tpl.push(&m.tok, &mut prompt, "user", line);
        tpl.open_assistant(&m.tok, &mut prompt);
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
    const PREFILL: usize = 256;
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
                for chunk in tokens[..PREFILL].chunks(PREFILL_CHUNK) {
                    next = argmax(e.forward(chunk, false));
                }
                prefill = prefill.max(PREFILL as f64 / t0.elapsed().as_secs_f64());

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
            "\nprefill {PREFILL} tokens, decode {DECODE} tokens (best of {RUNS} runs), perplexity over {} tokens of {}",
            r.6,
            a.file.display()
        );
    }
    Ok(())
}

fn chat_prompt(a: &Args, tok: &Tokenizer, prompt: &str) -> Result<Vec<u32>> {
    if a.raw {
        return Ok(tok.encode(prompt, false));
    }
    let tpl = ChatTemplate::new(tok)?;
    let system = a.system.as_deref().unwrap_or(DEFAULT_SYSTEM);
    Ok(tpl.encode(tok, &[Message::new("user", prompt)], Some(system)))
}

/// Runs the same greedy generation plainly and speculatively, checks the
/// outputs match, and reports the speedup.
fn speculate(a: &Args) -> Result<()> {
    let prompt = a
        .prompt
        .as_deref()
        .ok_or_else(|| Error::msg("speculate needs --prompt"))?;
    let pool = Arc::new(Pool::new(a.threads));
    let mut target = load(a, a.format, a.gpu, &pool)?;
    let draft_dir = a.draft_model.clone().unwrap_or_else(|| a.model.clone());
    let mut draft = load_from(a, &draft_dir, a.draft.unwrap_or(Format::Q4), a.gpu, &pool)?;
    let tokens = chat_prompt(a, &target.tok, prompt)?;
    let stop = stop_tokens(&target.cfg, &target.tok);
    let opts = GenOptions {
        max_tokens: a.max_tokens,
        sampler: SamplerConfig::greedy(),
        stop_tokens: stop.clone(),
    };

    eprintln!("\n--- plain greedy decoding on {}", target.engine.name());
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

    eprintln!(
        "\n--- speculative: {} drafts {} tokens per step",
        draft.engine.name(),
        a.draft_k
    );
    let spec = tachyon::speculative::generate_speculative(
        target.engine.as_mut(),
        draft.engine.as_mut(),
        &target.tok,
        &tokens,
        a.draft_k,
        a.max_tokens,
        &stop,
        &mut |_| true,
    )?;
    let same = spec.tokens == plain.tokens;
    eprintln!(
        "{} tokens at {:.1} tok/s ({:.2}x) · {:.0}% of drafts accepted · {:.2} tokens per target pass · output {}",
        spec.tokens.len(),
        spec.decode_tps(),
        spec.decode_tps() / plain.decode_tps(),
        spec.acceptance() * 100.0,
        spec.tokens_per_pass(),
        if same { "identical to plain decoding" } else { "DIFFERS from plain decoding" }
    );
    if !same {
        bail!("speculative output differs from plain greedy output");
    }
    Ok(())
}

fn serve(a: &Args) -> Result<()> {
    let pool = Arc::new(Pool::new(a.threads));
    let m = load(a, a.format, a.gpu, &pool)?;
    let template = ChatTemplate::new(&m.tok)?;
    let stop = stop_tokens(&m.cfg, &m.tok);
    let id = a
        .model
        .file_name()
        .map_or("model".into(), |n| n.to_string_lossy().into_owned());
    let mut server = tachyon::server::Server::new(m.engine, m.tok, template, stop, id);
    server.defaults = a.sampler.clone();
    server.max_tokens = a.max_tokens;
    tachyon::server::serve(server, &format!("{}:{}", a.host, a.port))
}
