//! The generation loop: prefill the prompt, then decode one token at a time.

use std::time::{Duration, Instant};

use crate::constrain::{Constraint, OutputFormat};
use crate::engine::Engine;
use crate::sampler::{Sampler, SamplerConfig};
use crate::tokenizer::{Tokenizer, Utf8Stream};
use crate::{bail, Result};

/// Prompt tokens per forward pass during prefill.
pub const PREFILL_CHUNK: usize = 128;

#[derive(Debug, Clone)]
pub struct GenOptions {
    pub max_tokens: usize,
    pub sampler: SamplerConfig,
    pub stop_tokens: Vec<u32>,
    /// Constrain the output's shape (JSON, a schema).
    pub format: OutputFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finish {
    /// The model ended its turn.
    Stop,
    /// Hit `max_tokens` or the context window.
    Length,
    /// The caller asked to stop.
    Cancelled,
}

impl Finish {
    pub fn as_str(self) -> &'static str {
        match self {
            Finish::Stop => "stop",
            Finish::Length => "length",
            Finish::Cancelled => "cancelled",
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenStats {
    pub prompt_tokens: usize,
    /// Prompt tokens skipped because the KV cache already held them.
    pub cached_tokens: usize,
    pub tokens: Vec<u32>,
    pub prefill: Duration,
    pub decode: Duration,
    pub finish: Finish,
}

impl GenStats {
    pub fn prefill_tps(&self) -> f64 {
        (self.prompt_tokens - self.cached_tokens) as f64 / self.prefill.as_secs_f64().max(1e-9)
    }

    /// Decode speed, counting the forward passes the decode phase ran.
    pub fn decode_tps(&self) -> f64 {
        self.tokens.len().saturating_sub(1) as f64 / self.decode.as_secs_f64().max(1e-9)
    }
}

/// Generates a reply to `prompt`.
///
/// `cache` lists the tokens currently in the engine's KV cache. The longest
/// prefix it shares with `prompt` is reused instead of recomputed (a repeated
/// system prompt or the earlier turns of a chat), and on return it lists
/// the new cache contents.
///
/// `emit` gets each new piece of text; returning `false` stops generation.
pub fn generate(
    engine: &mut dyn Engine,
    tok: &Tokenizer,
    cache: &mut Vec<u32>,
    prompt: &[u32],
    opts: &GenOptions,
    emit: &mut dyn FnMut(&str) -> bool,
) -> Result<GenStats> {
    if prompt.is_empty() {
        bail!("empty prompt");
    }
    if prompt.len() >= engine.max_seq() {
        bail!(
            "prompt is {} tokens but the context window is {}",
            prompt.len(),
            engine.max_seq()
        );
    }
    if cache.len() != engine.pos() {
        cache.clear();
    }
    // Always recompute at least the last prompt token: we need its logits.
    let reuse = cache
        .iter()
        .zip(prompt)
        .take_while(|(a, b)| a == b)
        .count()
        .min(prompt.len() - 1);
    engine.truncate(reuse);
    cache.truncate(reuse);

    let vocab = tok.len();
    let mut sampler = Sampler::new(opts.sampler.clone());
    let mut constraint = opts.format.constraint();
    let pick = |sampler: &mut Sampler,
                logits: &[f32],
                recent: &[u32],
                c: &Option<Box<dyn Constraint>>| match c {
        Some(c) => sampler.sample_with(logits, vocab, recent, &|t| c.allows(tok, t)),
        None => sampler.sample(logits, vocab, recent),
    };
    let started = Instant::now();
    let pending = &prompt[reuse..];
    let chunks = pending.chunks(PREFILL_CHUNK).count();
    let mut next = 0;
    for (i, chunk) in pending.chunks(PREFILL_CHUNK).enumerate() {
        let logits = engine.forward(chunk, false);
        if i + 1 == chunks {
            next = pick(&mut sampler, logits, &[], &constraint);
        }
    }
    cache.extend_from_slice(pending);
    let prefill = started.elapsed();

    let started = Instant::now();
    let mut tokens = Vec::new();
    let mut text = Utf8Stream::default();
    let finish = loop {
        if opts.stop_tokens.contains(&next) || constraint.as_ref().is_some_and(|c| c.is_complete())
        {
            break Finish::Stop;
        }
        tokens.push(next);
        if let Some(c) = &mut constraint {
            c.advance(tok.token_bytes(next));
        }
        if !tok.is_special(next) {
            let piece = text.push(tok.token_bytes(next));
            if !piece.is_empty() && !emit(&piece) {
                break Finish::Cancelled;
            }
        }
        if tokens.len() >= opts.max_tokens || engine.pos() >= engine.max_seq() {
            break Finish::Length;
        }
        let logits = engine.forward(&[next], false);
        cache.push(next);
        let recent = &tokens[tokens.len().saturating_sub(64)..];
        next = pick(&mut sampler, logits, recent, &constraint);
    };
    let rest = text.finish();
    if !rest.is_empty() {
        emit(&rest);
    }

    Ok(GenStats {
        prompt_tokens: prompt.len(),
        cached_tokens: reuse,
        tokens,
        prefill,
        decode: started.elapsed(),
        finish,
    })
}
