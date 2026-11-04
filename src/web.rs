//! The browser build's interface: plain exported functions over WebAssembly
//! memory, driven by `web/worker.js`. There is no wasm-bindgen (this crate
//! has no dependencies): the page copies each model file into memory it
//! gets from `tachyon_alloc`, and reads text back through a pointer and a
//! length.
//!
//! ```text
//! cargo rustc --release --lib --target wasm32-unknown-unknown --crate-type cdylib
//! ```

use std::cell::RefCell;
use std::sync::Arc;

use crate::chat::{default_system, ChatTemplate, Message};
use crate::config::Config;
use crate::engine::{Engine, Session};
use crate::generate::PREFILL_CHUNK;
use crate::model::{CpuBackend, Weights};
use crate::pool::Pool;
use crate::safetensors::SafeTensors;
use crate::sampler::{Sampler, SamplerConfig};
use crate::tokenizer::{Tokenizer, Utf8Stream};
use crate::weights::Format;
use crate::Result;

/// Positions of context the browser keeps (KV cache size).
const CONTEXT: usize = 2048;

struct Chat {
    cfg: Config,
    tok: Tokenizer,
    tpl: ChatTemplate,
    engine: Session,
    /// The conversation so far, and the tokens now in the KV cache.
    messages: Vec<Message>,
    cached: Vec<u32>,
    reply: Option<Reply>,
}

struct Reply {
    sampler: Sampler,
    next: u32,
    tokens: Vec<u32>,
    left: usize,
    text: String,
    stream: Utf8Stream,
}

#[derive(Default)]
struct State {
    chat: Option<Chat>,
    /// The latest piece of text or error message, for the page to read.
    out: Vec<u8>,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::default();
}

/// A panic's message: in the browser a panic traps, and the page reads
/// this afterwards to say why.
static PANIC: std::sync::Mutex<Vec<u8>> = std::sync::Mutex::new(Vec::new());

#[no_mangle]
pub extern "C" fn tachyon_panic() -> *const u8 {
    PANIC.lock().map_or(std::ptr::null(), |p| p.as_ptr())
}

#[no_mangle]
pub extern "C" fn tachyon_panic_len() -> usize {
    PANIC.lock().map_or(0, |p| p.len())
}

/// Takes ownership of `len` bytes the page wrote at `ptr`.
unsafe fn take(ptr: *mut u8, len: usize) -> Vec<u8> {
    if ptr.is_null() {
        Vec::new()
    } else {
        Vec::from_raw_parts(ptr, len, len)
    }
}

fn text(bytes: Vec<u8>) -> Result<String> {
    String::from_utf8(bytes).map_err(|_| crate::Error::msg("a model file is not UTF-8"))
}

/// Memory for the page to write `len` bytes into; ownership passes back
/// with the call that receives it.
#[no_mangle]
pub extern "C" fn tachyon_alloc(len: usize) -> *mut u8 {
    static HOOK: std::sync::Once = std::sync::Once::new();
    HOOK.call_once(|| {
        std::panic::set_hook(Box::new(|info| {
            if let Ok(mut p) = PANIC.lock() {
                *p = info.to_string().into_bytes();
            }
        }))
    });
    let mut v = Vec::<u8>::with_capacity(len);
    let ptr = v.as_mut_ptr();
    std::mem::forget(v);
    ptr
}

/// The text of the last call's output or error.
#[no_mangle]
pub extern "C" fn tachyon_out() -> *const u8 {
    STATE.with(|s| s.borrow().out.as_ptr())
}

#[no_mangle]
pub extern "C" fn tachyon_out_len() -> usize {
    STATE.with(|s| s.borrow().out.len())
}

fn fail(e: crate::Error) -> i32 {
    STATE.with(|s| s.borrow_mut().out = e.to_string().into_bytes());
    -1
}

/// Loads a model from its files: `config.json`, `generation_config.json`
/// (or null), `tokenizer.json` and `model.safetensors`, converted to
/// `format` (0 bf16, 1 8-bit, 2 4-bit). 0 on success; -1 with the error at
/// `tachyon_out`.
///
/// # Safety
/// Each pointer must come from `tachyon_alloc` with its length written.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn tachyon_load(
    config: *mut u8,
    config_len: usize,
    generation: *mut u8,
    generation_len: usize,
    tokenizer: *mut u8,
    tokenizer_len: usize,
    weights: *mut u8,
    weights_len: usize,
    format: u32,
) -> i32 {
    let files = (
        take(config, config_len),
        take(generation, generation_len),
        take(tokenizer, tokenizer_len),
        take(weights, weights_len),
    );
    match load(files, format) {
        Ok(chat) => {
            STATE.with(|s| s.borrow_mut().chat = Some(chat));
            0
        }
        Err(e) => fail(e),
    }
}

fn load(
    (config, generation, tokenizer, weights): (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>),
    format: u32,
) -> Result<Chat> {
    let generation = (!generation.is_empty())
        .then(|| text(generation))
        .transpose()?;
    let cfg = Config::from_json(&text(config)?, generation.as_deref())?;
    let tok = Tokenizer::from_json(&text(tokenizer)?)?;
    let tpl = ChatTemplate::new(&tok)?;
    let format = [Format::Bf16, Format::Q8, Format::Q4M][format.min(2) as usize];
    let st = SafeTensors::from_bytes(weights)?;
    let w = Weights::from_safetensors(&st, &cfg, format, None)?;
    drop(st); // the checkpoint's bytes; the converted weights are separate
    let pool = Arc::new(Pool::new(1));
    let engine = Session::new(Box::new(CpuBackend::new(
        cfg.clone(),
        w,
        pool,
        CONTEXT / 16,
    )));
    Ok(Chat {
        cfg,
        tok,
        tpl,
        engine,
        messages: Vec::new(),
        cached: Vec::new(),
        reply: None,
    })
}

/// Starts the assistant's reply to a new user message (UTF-8 at `ptr`),
/// after the conversation so far. 0, or -1 with an error.
///
/// # Safety
/// `ptr` must come from `tachyon_alloc` with `len` bytes written.
#[no_mangle]
pub unsafe extern "C" fn tachyon_start(
    ptr: *mut u8,
    len: usize,
    max_tokens: u32,
    temperature: f32,
    seed: u32,
) -> i32 {
    let message = match text(take(ptr, len)) {
        Ok(m) => m,
        Err(e) => return fail(e),
    };
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        let Some(chat) = s.chat.as_mut() else {
            drop(s);
            return fail(crate::Error::msg("no model loaded"));
        };
        chat.messages.push(Message::new("user", &message));
        let system = default_system(&chat.cfg.model_type);
        let mut prompt = chat.tpl.encode(&chat.tok, &chat.messages, system);
        if prompt.len() + max_tokens as usize > CONTEXT {
            // Keep the context: forget the oldest turns.
            chat.messages.drain(..chat.messages.len().saturating_sub(1));
            prompt = chat.tpl.encode(&chat.tok, &chat.messages, system);
        }
        // Reuse the cache for the part of the prompt it already holds.
        let same = chat
            .cached
            .iter()
            .zip(&prompt)
            .take_while(|(a, b)| a == b)
            .count();
        let same = same.min(prompt.len() - 1);
        chat.engine.truncate(same);
        let vocab = chat.cfg.vocab;
        let mut logits = Vec::new();
        for piece in prompt[same..].chunks(PREFILL_CHUNK) {
            logits = chat.engine.forward(piece, false)[..vocab].to_vec();
        }
        let mut sampler = Sampler::new(SamplerConfig {
            temperature,
            seed: seed as u64,
            ..SamplerConfig::default()
        });
        let next = sampler.sample(&logits, vocab, &[]);
        chat.cached = prompt;
        chat.reply = Some(Reply {
            sampler,
            next,
            tokens: Vec::new(),
            left: max_tokens as usize,
            text: String::new(),
            stream: Utf8Stream::default(),
        });
        0
    })
}

/// Emits the next token of the reply: its text's byte length (the text at
/// `tachyon_out`, possibly empty mid-character), or -1 once the reply is
/// over (it is then part of the conversation).
#[no_mangle]
pub extern "C" fn tachyon_next() -> i32 {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        let State { chat, out } = &mut *s;
        let Some(chat) = chat.as_mut() else {
            return -1;
        };
        let Some(r) = chat.reply.as_mut() else {
            return -1;
        };
        let stop = chat.cfg.eos_ids.contains(&r.next) || r.next == chat.tpl.end_of_turn();
        if stop || r.left == 0 {
            let reply = chat.reply.take().unwrap();
            chat.messages.push(Message::new("assistant", &reply.text));
            out.clear();
            return -1;
        }
        let token = r.next;
        let piece = r.stream.push(chat.tok.token_bytes(token));
        r.text.push_str(&piece);
        r.tokens.push(token);
        r.left -= 1;
        chat.cached.push(token);
        let vocab = chat.cfg.vocab;
        let logits = chat.engine.forward(&[token], false);
        r.next = r.sampler.sample(&logits[..vocab], vocab, &r.tokens);
        *out = piece.into_bytes();
        out.len() as i32
    })
}

/// Forgets the conversation.
#[no_mangle]
pub extern "C" fn tachyon_reset() {
    STATE.with(|s| {
        if let Some(chat) = s.borrow_mut().chat.as_mut() {
            chat.messages.clear();
            chat.reply = None;
        }
    });
}
