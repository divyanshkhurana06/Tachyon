//! An OpenAI-compatible HTTP server on `std::net`, with a minimal hand-written
//! HTTP/1.1 implementation.
//!
//! - `POST /v1/chat/completions`: chat, optionally streamed as server-sent
//!   events, so the official `openai` clients and chat UIs work unchanged
//! - `GET /v1/models`, `GET /health`, `GET /info`
//! - `GET /`: a small chat page
//!
//! Requests run one at a time against a single KV cache. Consecutive
//! requests that share a prefix (the same system prompt, the earlier turns
//! of a chat) skip recomputing it.

use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::chat::{ChatTemplate, Message, DEFAULT_SYSTEM};
use crate::engine::Engine;
use crate::generate::{generate, Finish, GenOptions};
use crate::json::{obj, Json};
use crate::sampler::SamplerConfig;
use crate::tokenizer::Tokenizer;
use crate::Result;

const UI: &str = include_str!("ui.html");
const MAX_HEADER: usize = 64 * 1024;
const MAX_BODY: usize = 8 * 1024 * 1024;

pub struct Server {
    pub engine: Box<dyn Engine>,
    pub tok: Tokenizer,
    pub template: ChatTemplate,
    pub stop: Vec<u32>,
    pub model_id: String,
    pub defaults: SamplerConfig,
    pub max_tokens: usize,
    cache: Vec<u32>,
}

impl Server {
    pub fn new(
        engine: Box<dyn Engine>,
        tok: Tokenizer,
        template: ChatTemplate,
        stop: Vec<u32>,
        model_id: String,
    ) -> Server {
        Server {
            engine,
            tok,
            template,
            stop,
            model_id,
            defaults: SamplerConfig::default(),
            max_tokens: 1024,
            cache: Vec::new(),
        }
    }

    fn info(&self) -> Json {
        let c = self.engine.config();
        obj([
            ("model", self.model_id.as_str().into()),
            ("backend", self.engine.name().into()),
            ("params", c.params().into()),
            ("context", self.engine.max_seq().into()),
        ])
    }
}

pub fn serve(server: Server, addr: &str) -> Result<()> {
    let listener = TcpListener::bind(addr)?;
    eprintln!(
        "serving {} on http://{addr} (chat UI at /, API at /v1)",
        server.model_id
    );
    let shared = Arc::new(Mutex::new(server));
    for conn in listener.incoming() {
        let Ok(conn) = conn else { continue };
        let shared = shared.clone();
        thread::spawn(move || {
            if let Err(e) = handle(conn, &shared) {
                if e.kind() != io::ErrorKind::BrokenPipe {
                    eprintln!("connection error: {e}");
                }
            }
        });
    }
    Ok(())
}

struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn read_request(conn: &mut TcpStream) -> io::Result<Option<Request>> {
    conn.set_read_timeout(Some(Duration::from_secs(30)))?;
    let bad = |msg: &str| io::Error::new(io::ErrorKind::InvalidData, msg.to_string());
    let mut buf = Vec::new();
    let mut chunk = [0u8; 8192];
    let head_end = loop {
        if let Some(i) = find(&buf, b"\r\n\r\n") {
            break i;
        }
        if buf.len() > MAX_HEADER {
            return Err(bad("headers too large"));
        }
        let n = conn.read(&mut chunk)?;
        if n == 0 {
            return Ok(None);
        }
        buf.extend_from_slice(&chunk[..n]);
    };
    let head = std::str::from_utf8(&buf[..head_end]).map_err(|_| bad("headers are not UTF-8"))?;
    let mut lines = head.split("\r\n");
    let mut first = lines.next().unwrap_or("").split(' ');
    let method = first.next().unwrap_or("").to_string();
    let path = first
        .next()
        .unwrap_or("/")
        .split('?')
        .next()
        .unwrap_or("/")
        .to_string();
    let mut length = 0;
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            if name.trim().eq_ignore_ascii_case("content-length") {
                length = value
                    .trim()
                    .parse()
                    .map_err(|_| bad("bad content-length"))?;
            }
        }
    }
    if length > MAX_BODY {
        return Err(bad("body too large"));
    }
    let mut body = buf[head_end + 4..].to_vec();
    while body.len() < length {
        let n = conn.read(&mut chunk)?;
        if n == 0 {
            return Err(bad("body cut short"));
        }
        body.extend_from_slice(&chunk[..n]);
    }
    body.truncate(length);
    Ok(Some(Request { method, path, body }))
}

const CORS: &str = "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n";

fn respond(conn: &mut TcpStream, status: &str, content_type: &str, body: &[u8]) -> io::Result<()> {
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n{CORS}Connection: close\r\n\r\n",
        body.len()
    );
    conn.write_all(head.as_bytes())?;
    conn.write_all(body)
}

fn respond_json(conn: &mut TcpStream, status: &str, body: &Json) -> io::Result<()> {
    respond(
        conn,
        status,
        "application/json",
        body.to_string().as_bytes(),
    )
}

fn error(conn: &mut TcpStream, status: &str, message: &str) -> io::Result<()> {
    let body = obj([(
        "error",
        obj([
            ("message", message.into()),
            ("type", "invalid_request_error".into()),
        ]),
    )]);
    respond_json(conn, status, &body)
}

fn handle(mut conn: TcpStream, shared: &Mutex<Server>) -> io::Result<()> {
    let Some(req) = read_request(&mut conn)? else {
        return Ok(());
    };
    let lock = || {
        shared
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    };
    match (req.method.as_str(), req.path.as_str()) {
        ("OPTIONS", _) => respond(&mut conn, "204 No Content", "text/plain", b""),
        ("GET", "/") => respond(
            &mut conn,
            "200 OK",
            "text/html; charset=utf-8",
            UI.as_bytes(),
        ),
        ("GET", "/health") => respond_json(&mut conn, "200 OK", &obj([("status", "ok".into())])),
        ("GET", "/info") => {
            let info = lock().info();
            respond_json(&mut conn, "200 OK", &info)
        }
        ("GET", "/v1/models") => {
            let id = lock().model_id.clone();
            let model = obj([
                ("id", id.into()),
                ("object", "model".into()),
                ("owned_by", "tachyon".into()),
            ]);
            respond_json(
                &mut conn,
                "200 OK",
                &obj([("object", "list".into()), ("data", Json::Arr(vec![model]))]),
            )
        }
        ("POST", "/v1/chat/completions") => chat_completion(&mut conn, shared, &req.body),
        _ => error(
            &mut conn,
            "404 Not Found",
            &format!("no route for {} {}", req.method, req.path),
        ),
    }
}

struct ChatRequest {
    messages: Vec<Message>,
    stream: bool,
    max_tokens: Option<usize>,
    sampler: SamplerConfig,
}

fn parse_chat(body: &[u8], defaults: &SamplerConfig) -> std::result::Result<ChatRequest, String> {
    let text = std::str::from_utf8(body).map_err(|_| "body is not UTF-8".to_string())?;
    let j = Json::parse(text)?;
    let list = j
        .get("messages")
        .and_then(Json::as_array)
        .ok_or("`messages` must be an array")?;
    let mut messages = Vec::new();
    for m in list {
        let role = m
            .get("role")
            .and_then(Json::as_str)
            .ok_or("each message needs a `role`")?;
        // `content` is a string, or a list of parts of which we use the text ones.
        let content = match m.get("content") {
            Some(Json::Str(s)) => s.clone(),
            Some(Json::Arr(parts)) => parts
                .iter()
                .filter_map(|p| p.get("text").and_then(Json::as_str))
                .collect::<Vec<_>>()
                .join(""),
            Some(Json::Null) | None => String::new(),
            Some(_) => return Err("`content` must be a string or a list of parts".into()),
        };
        messages.push(Message::new(role, &content));
    }
    if messages.is_empty() {
        return Err("`messages` is empty".into());
    }
    let num = |k: &str| j.get(k).and_then(Json::as_f64);
    let mut sampler = defaults.clone();
    if let Some(t) = num("temperature") {
        sampler.temperature = t as f32;
    }
    if let Some(p) = num("top_p") {
        sampler.top_p = p as f32;
    }
    if let Some(k) = num("top_k") {
        sampler.top_k = k as usize;
    }
    if let Some(r) = num("repetition_penalty") {
        sampler.repetition_penalty = r as f32;
    }
    if let Some(s) = j.get("seed").and_then(Json::as_u64) {
        sampler.seed = s;
    }
    Ok(ChatRequest {
        messages,
        stream: j.get("stream").and_then(Json::as_bool).unwrap_or(false),
        max_tokens: j
            .get("max_tokens")
            .or(j.get("max_completion_tokens"))
            .and_then(Json::as_usize),
        sampler,
    })
}

fn chat_completion(conn: &mut TcpStream, shared: &Mutex<Server>, body: &[u8]) -> io::Result<()> {
    let mut guard = shared
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let s = &mut *guard;
    let req = match parse_chat(body, &s.defaults) {
        Ok(r) => r,
        Err(e) => return error(conn, "400 Bad Request", &e),
    };
    let prompt = s
        .template
        .encode(&s.tok, &req.messages, Some(DEFAULT_SYSTEM));
    let room = s.engine.max_seq().saturating_sub(prompt.len());
    if room == 0 {
        let msg = format!(
            "the prompt is {} tokens; the context window is {}",
            prompt.len(),
            s.engine.max_seq()
        );
        return error(conn, "400 Bad Request", &msg);
    }
    let opts = GenOptions {
        max_tokens: req.max_tokens.unwrap_or(s.max_tokens).min(room),
        sampler: req.sampler,
        stop_tokens: s.stop.clone(),
    };
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let id = format!("chatcmpl-{created:x}{:04x}", s.cache.len() % 0xFFFF);
    let model = s.model_id.clone();
    let chunk = |delta: Json, finish: Option<&str>| {
        obj([
            ("id", id.as_str().into()),
            ("object", "chat.completion.chunk".into()),
            ("created", created.into()),
            ("model", model.as_str().into()),
            (
                "choices",
                Json::Arr(vec![obj([
                    ("index", 0usize.into()),
                    ("delta", delta),
                    ("finish_reason", finish.map_or(Json::Null, Json::from)),
                ])]),
            ),
        ])
    };

    let mut text = String::new();
    let mut write_failed = false;
    if req.stream {
        conn.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n{CORS}Connection: close\r\n\r\n").as_bytes())?;
        let first = chunk(
            obj([("role", "assistant".into()), ("content", "".into())]),
            None,
        );
        conn.write_all(format!("data: {first}\n\n").as_bytes())?;
    }
    let stats = {
        let Server {
            engine, tok, cache, ..
        } = s;
        let mut emit = |piece: &str| {
            if !req.stream {
                text.push_str(piece);
                return true;
            }
            let event = chunk(obj([("content", piece.into())]), None);
            write_failed = conn
                .write_all(format!("data: {event}\n\n").as_bytes())
                .is_err();
            !write_failed // a closed connection stops generation
        };
        generate(engine.as_mut(), tok, cache, &prompt, &opts, &mut emit)
    };
    let stats = match stats {
        Ok(stats) => stats,
        Err(e) if !req.stream => return error(conn, "400 Bad Request", &e.0),
        Err(e) => {
            return conn.write_all(format!("data: {}\n\n", obj([("error", e.0.into())])).as_bytes())
        }
    };
    if write_failed {
        return Ok(());
    }

    let finish = if stats.finish == Finish::Length {
        "length"
    } else {
        "stop"
    };
    let usage = obj([
        ("prompt_tokens", stats.prompt_tokens.into()),
        ("completion_tokens", stats.tokens.len().into()),
        (
            "total_tokens",
            (stats.prompt_tokens + stats.tokens.len()).into(),
        ),
    ]);
    let timings = obj([
        ("cached_tokens", stats.cached_tokens.into()),
        ("prefill_tokens_per_second", stats.prefill_tps().into()),
        ("decode_tokens_per_second", stats.decode_tps().into()),
        ("backend", s.engine.name().into()),
    ]);
    if req.stream {
        let Json::Obj(mut last) = chunk(obj([]), Some(finish)) else {
            unreachable!()
        };
        last.push(("usage".into(), usage));
        last.push(("timings".into(), timings));
        conn.write_all(format!("data: {}\n\ndata: [DONE]\n\n", Json::Obj(last)).as_bytes())
    } else {
        let message = obj([("role", "assistant".into()), ("content", text.into())]);
        let body = obj([
            ("id", id.as_str().into()),
            ("object", "chat.completion".into()),
            ("created", created.into()),
            ("model", s.model_id.as_str().into()),
            (
                "choices",
                Json::Arr(vec![obj([
                    ("index", 0usize.into()),
                    ("message", message),
                    ("finish_reason", finish.into()),
                ])]),
            ),
            ("usage", usage),
            ("timings", timings),
        ]);
        respond_json(conn, "200 OK", &body)
    }
}
