//! End-to-end checks against Granite 3.1 1B-A400M Instruct, a mixture of
//! experts (32 per layer, 8 per token). They need the model in
//! models/granite-3.1-1b-a400m-instruct (see the README) and are skipped
//! without it.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use tachyon::chat::{ChatTemplate, Message};
use tachyon::config::Config;
use tachyon::engine::{Engine, Session};
use tachyon::model::{CpuBackend, Weights};
use tachyon::pool::Pool;
use tachyon::sampler::argmax;
use tachyon::tokenizer::Tokenizer;
use tachyon::weights::Format;

static SERIAL: Mutex<()> = Mutex::new(());

fn setup() -> Option<(MutexGuard<'static, ()>, PathBuf)> {
    let guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/granite-3.1-1b-a400m-instruct");
    if dir.join("model.safetensors").exists() {
        Some((guard, dir))
    } else {
        eprintln!("skipping: {} not found", dir.display());
        None
    }
}

fn greedy(e: &mut dyn Engine, prompt: &[u32], n: usize, stop: u32) -> Vec<u32> {
    let vocab = e.config().vocab;
    e.truncate(0);
    let mut out = vec![argmax(&e.forward(prompt, false)[..vocab])];
    while out.len() < n && *out.last().unwrap() != stop {
        let next = argmax(&e.forward(&out[out.len() - 1..], false)[..vocab]);
        out.push(next);
    }
    out
}

#[test]
fn tokenizer_and_template() {
    let Some((_g, dir)) = setup() else { return };
    let tok = Tokenizer::load(&dir.join("tokenizer.json")).unwrap();
    let text = std::fs::read_to_string(dir.join("../../bench/eval.txt")).unwrap();
    assert_eq!(tok.decode(&tok.encode(&text, false), false), text);
    let tpl = ChatTemplate::new(&tok).unwrap();
    let p = tpl.encode(&tok, &[Message::new("user", "Hi!")], None);
    let shown = tok.decode(&p, false);
    assert!(shown.starts_with(
        "<|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.\nToday's Date: "
    ));
    assert!(shown.ends_with(
        "You are Granite, developed by IBM. You are a helpful AI assistant.<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>Hi!<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
    ));
    assert_eq!(tpl.end_of_turn(), 0);
}

/// The experts are routed the same on both backends: identical greedy
/// tokens, through the per-(token, expert) path for decode and the
/// per-expert GEMMs for the prompt.
#[test]
fn answers_and_gpu_matches_cpu() {
    let Some((_g, dir)) = setup() else { return };
    let tok = Tokenizer::load(&dir.join("tokenizer.json")).unwrap();
    let cfg = Config::load(&dir).unwrap();
    assert_eq!((cfg.experts, cfg.experts_per_token), (32, 8));
    let tpl = ChatTemplate::new(&tok).unwrap();
    let p = tpl.encode(
        &tok,
        &[Message::new(
            "user",
            "What is the capital of France? Answer in one sentence.",
        )],
        None,
    );
    let pool = Arc::new(Pool::new(Pool::default_threads()));
    let w = Weights::load(&dir, &cfg, Format::Bf16).unwrap();
    let mut cpu = Session::new(Box::new(CpuBackend::new(cfg.clone(), w, pool, 16)));
    let on_cpu = greedy(&mut cpu, &p, 24, tpl.end_of_turn());
    let reply = tok.decode(&on_cpu, true);
    assert!(reply.contains("Paris"), "{reply:?}");

    #[cfg(target_os = "macos")]
    {
        let w = Weights::load(&dir, &cfg, Format::Bf16).unwrap();
        match tachyon::gpu::GpuBackend::new(cfg.clone(), w, 16, 256, false) {
            Ok(g) => {
                let on_gpu = greedy(&mut Session::new(Box::new(g)), &p, 24, tpl.end_of_turn());
                assert_eq!(on_gpu, on_cpu);
            }
            Err(e) => eprintln!("skipping the GPU: {e}"),
        }
    }
}

/// Granite's tool format: a `tools` turn of indented JSON, the template's
/// sentence about `<|tool_call|>` (the token) in the system message, and a
/// call as `<|tool_call|>` and a JSON list. Asked plainly, the model calls
/// the tool; forced, its arguments fit the schema.
#[test]
fn calls_tools_in_granite_format() {
    use tachyon::constrain::OutputFormat;
    use tachyon::generate::{generate, GenOptions};
    use tachyon::json::Json;
    use tachyon::sampler::SamplerConfig;
    use tachyon::schema::{JsonSchema, Schema};
    let Some((_g, dir)) = setup() else { return };
    let tok = Tokenizer::load(&dir.join("tokenizer.json")).unwrap();
    let cfg = Config::load(&dir).unwrap();
    let tpl = ChatTemplate::new(&tok).unwrap();
    assert!(tpl.supports_tools() && tpl.calls_as_list());
    let params = r#"{"type":"object","properties":{"city":{"type":"string"},"unit":{"enum":["celsius","fahrenheit"]}},"required":["city"]}"#;
    let tool = Json::parse(&format!(
        r#"{{"type":"function","function":{{"name":"get_weather","description":"Get the current weather in a city","parameters":{params}}}}}"#
    ))
    .unwrap();
    let mut p = tpl.encode_with_tools(
        &tok,
        &[Message::new(
            "user",
            "What is the weather in Tokyo right now?",
        )],
        None,
        std::slice::from_ref(&tool),
    );
    let shown = tok.decode(&p, false);
    assert!(shown.contains(
        "You are Granite, developed by IBM. You are a helpful AI assistant with access to the following tools. When a tool is required to answer the user's query, respond with <|tool_call|> followed by"
    ));
    let call_token = tok.added_token("<|tool_call|>").unwrap();
    assert!(p.contains(&call_token));
    assert!(shown.contains(
        "<|end_of_text|>\n<|start_of_role|>tools<|end_of_role|>[\n    {\n        \"type\": \"function\",\n        \"function\": {\n            \"name\": \"get_weather\","
    ));
    assert!(shown.ends_with(
        "]<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>What is the weather in Tokyo right now?<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
    ));

    let pool = Arc::new(Pool::new(Pool::default_threads()));
    let w = Weights::load(&dir, &cfg, Format::Q8).unwrap();
    let mut e = Session::new(Box::new(CpuBackend::new(cfg.clone(), w, pool, 32)));

    // Unforced: the reply starts with the call token, then the list.
    let out = greedy(&mut e, &p, 60, tpl.end_of_turn());
    let text = tok.decode(&out[..out.len() - 1], false);
    let (content, calls) = tpl.parse_calls(&text, false);
    assert_eq!(calls.len(), 1, "{text}");
    assert_eq!(calls[0].name, "get_weather");
    assert_eq!(
        calls[0].arguments.get("city").and_then(Json::as_str),
        Some("Tokyo")
    );
    assert_eq!(content, "");

    // Forced: the prompt opens the call, the schema shapes the list.
    tpl.open_call(&tok, &mut p);
    let schema = Json::parse(&format!(
        r#"{{"type":"array","minItems":1,"maxItems":1,"items":{{"type":"object","properties":{{"name":{{"const":"get_weather"}},"arguments":{params}}},"required":["name","arguments"]}}}}"#
    ))
    .unwrap();
    for seed in 1..=3 {
        let opts = GenOptions {
            max_tokens: 60,
            sampler: SamplerConfig {
                temperature: 1.0,
                seed,
                ..SamplerConfig::default()
            },
            stop_tokens: cfg.eos_ids.clone(),
            format: OutputFormat::JsonSchema(JsonSchema::new(Schema::parse(&schema).unwrap())),
        };
        let mut text = String::new();
        generate(&mut e, &tok, &mut Vec::new(), &p, &opts, &mut |s| {
            text.push_str(s);
            true
        })
        .unwrap();
        let (_, calls) = tpl.parse_calls(&text, true);
        assert_eq!(calls.len(), 1, "{text}");
        assert!(
            calls[0]
                .arguments
                .get("city")
                .and_then(Json::as_str)
                .is_some(),
            "{text}"
        );
    }
}
