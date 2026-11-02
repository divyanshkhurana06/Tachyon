//! An OpenAI-compatible HTTP server on `std::net`, with a minimal hand-written
//! HTTP/1.1 implementation.
//!
//! - `POST /v1/chat/completions`: chat, optionally streamed as server-sent
//!   events, so the official `openai` clients and chat UIs work unchanged.
//!   Supports `stop` strings, `response_format: {"type": "json_object"}`
//!   (constrained decoding: always valid JSON) and vLLM's `ignore_eos`.
//! - `GET /v1/models`, `GET /health`, `GET /info`
//! - `GET /`: a small chat page
//!
//! Each connection gets a thread that submits its request to the
//! [`Scheduler`], which batches every in-flight request into shared forward
//! passes, and relays tokens back as they come.

use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::chat::{ChatTemplate, Message};
use crate::constrain::JsonObject;
use crate::generate::Finish;
use crate::json::{obj, Json};
use crate::sampler::SamplerConfig;
use crate::scheduler::{Event, Request, Scheduler, Summary};
use crate::tokenizer::Tokenizer;
use crate::Result;

const UI: &str = include_str!("ui.html");
const MAX_HEADER: usize = 64 * 1024;
const MAX_BODY: usize = 8 * 1024 * 1024;
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

pub struct Server {
    pub scheduler: Scheduler,
    pub tok: Arc<Tokenizer>,
    pub template: ChatTemplate,
    pub stop: Vec<u32>,
    pub model_id: String,
    pub backend: String,
    pub params: usize,
    pub context: usize,
    /// Added when a request has no system message.
    pub system: Option<String>,
    pub kv_tokens: usize,
    pub defaults: SamplerConfig,
    pub max_tokens: usize,
}

impl Server {
    fn info(&self) -> Json {
        let st = &self.scheduler.stats;
        obj([
            ("model", self.model_id.as_str().into()),
            ("backend", self.backend.as_str().into()),
            ("params", self.params.into()),
            ("context", self.context.into()),
            ("kv_cache_tokens", self.kv_tokens.into()),
            ("running", st.running.load(Ordering::Relaxed).into()),
            ("waiting", st.waiting.load(Ordering::Relaxed).into()),
            ("steps", st.steps.load(Ordering::Relaxed).into()),
            ("tokens_generated", st.tokens.load(Ordering::Relaxed).into()),
            (
                "forward_seconds",
                (st.forward_ns.load(Ordering::Relaxed) as f64 / 1e9).into(),
            ),
            (
                "step_seconds",
                (st.step_ns.load(Ordering::Relaxed) as f64 / 1e9).into(),
            ),
        ])
    }
}

pub fn serve(server: Server, addr: &str) -> Result<()> {
    let listener = TcpListener::bind(addr)?;
    eprintln!(
        "serving {} on http://{addr} (chat UI at /, API at /v1)",
        server.model_id
    );
    let server = Arc::new(server);
    for conn in listener.incoming() {
        let Ok(conn) = conn else { continue };
        let server = server.clone();
        thread::spawn(move || {
            if let Err(e) = handle(conn, &server) {
                if !matches!(
                    e.kind(),
                    io::ErrorKind::BrokenPipe | io::ErrorKind::ConnectionReset
                ) {
                    eprintln!("connection error: {e}");
                }
            }
        });
    }
    Ok(())
}

struct HttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn read_request(conn: &mut TcpStream) -> io::Result<Option<HttpRequest>> {
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
    Ok(Some(HttpRequest { method, path, body }))
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

fn handle(mut conn: TcpStream, server: &Server) -> io::Result<()> {
    let Some(req) = read_request(&mut conn)? else {
        return Ok(());
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
        ("GET", "/info") => respond_json(&mut conn, "200 OK", &server.info()),
        ("GET", "/v1/models") => {
            let model = obj([
                ("id", server.model_id.as_str().into()),
                ("object", "model".into()),
                ("owned_by", "tachyon".into()),
            ]);
            respond_json(
                &mut conn,
                "200 OK",
                &obj([("object", "list".into()), ("data", Json::Arr(vec![model]))]),
            )
        }
        ("POST", "/v1/chat/completions") => chat_completion(&mut conn, server, &req.body),
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
    stop: Vec<String>,
    json: bool,
    ignore_eos: bool,
    /// Qwen3's `enable_thinking` (top level, or in `chat_template_kwargs`
    /// as vLLM takes it).
    thinking: bool,
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
    let stop = match j.get("stop") {
        Some(Json::Str(s)) => vec![s.clone()],
        Some(Json::Arr(a)) => a
            .iter()
            .filter_map(Json::as_str)
            .map(str::to_string)
            .collect(),
        _ => Vec::new(),
    };
    let json = match j
        .get("response_format")
        .and_then(|f| f.get("type"))
        .and_then(Json::as_str)
    {
        None | Some("text") => false,
        Some("json_object") => true,
        Some(other) => {
            return Err(format!(
                "response_format type {other:?} is not supported (text, json_object)"
            ))
        }
    };
    Ok(ChatRequest {
        messages,
        stream: j.get("stream").and_then(Json::as_bool).unwrap_or(false),
        max_tokens: j
            .get("max_tokens")
            .or(j.get("max_completion_tokens"))
            .and_then(Json::as_usize),
        sampler,
        stop: stop.into_iter().filter(|s| !s.is_empty()).collect(),
        json,
        ignore_eos: j.get("ignore_eos").and_then(Json::as_bool).unwrap_or(false),
        thinking: j
            .get("enable_thinking")
            .or(j
                .get("chat_template_kwargs")
                .and_then(|k| k.get("enable_thinking")))
            .and_then(Json::as_bool)
            .unwrap_or(true),
    })
}

/// Cuts the output at the first stop string. While streaming, holds back
/// any tail that could still turn out to be the start of one.
pub struct StopStrings {
    stops: Vec<String>,
    held: String,
}

impl StopStrings {
    pub fn new(stops: Vec<String>) -> StopStrings {
        StopStrings {
            stops,
            held: String::new(),
        }
    }

    /// Text safe to emit now, and whether a stop string was hit.
    pub fn push(&mut self, piece: &str) -> (String, bool) {
        self.held.push_str(piece);
        if let Some(at) = self
            .stops
            .iter()
            .filter_map(|s| self.held.find(s.as_str()))
            .min()
        {
            let out = self.held[..at].to_string();
            self.held.clear();
            return (out, true);
        }
        let keep = self
            .stops
            .iter()
            .flat_map(|s| s.char_indices().skip(1).map(move |(i, _)| &s[..i]))
            .filter(|prefix| self.held.ends_with(prefix))
            .map(str::len)
            .max()
            .unwrap_or(0);
        let out = self.held[..self.held.len() - keep].to_string();
        self.held.drain(..self.held.len() - keep);
        (out, false)
    }

    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.held)
    }
}

fn chat_completion(conn: &mut TcpStream, server: &Server, body: &[u8]) -> io::Result<()> {
    let req = match parse_chat(body, &server.defaults) {
        Ok(r) => r,
        Err(e) => return error(conn, "400 Bad Request", &e),
    };
    let mut prompt = server
        .template
        .encode(&server.tok, &req.messages, server.system.as_deref());
    if !req.thinking {
        server.template.skip_thinking(&server.tok, &mut prompt);
    }
    let room = server.context.saturating_sub(prompt.len());
    if room == 0 {
        let msg = format!(
            "the prompt is {} tokens; the context window is {}",
            prompt.len(),
            server.context
        );
        return error(conn, "400 Bad Request", &msg);
    }
    let (tx, rx) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    server.scheduler.submit(Request {
        prompt,
        max_tokens: req.max_tokens.unwrap_or(server.max_tokens).min(room),
        sampler: req.sampler,
        stop_tokens: if req.ignore_eos {
            Vec::new()
        } else {
            server.stop.clone()
        },
        constraint: req.json.then(|| Box::new(JsonObject::default()) as _),
        events: tx,
        cancel: cancel.clone(),
    });

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let id = format!(
        "chatcmpl-{created:x}{:06x}",
        NEXT_ID.fetch_add(1, Ordering::Relaxed)
    );
    let chunk = |delta: Json, finish: Option<&str>| {
        obj([
            ("id", id.as_str().into()),
            ("object", "chat.completion.chunk".into()),
            ("created", created.into()),
            ("model", server.model_id.as_str().into()),
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
    // Any write error means the client left: stop generating for it.
    let send = |conn: &mut TcpStream, data: &str| -> io::Result<()> {
        conn.write_all(data.as_bytes())
            .inspect_err(|_| cancel.store(true, Ordering::Relaxed))
    };

    if req.stream {
        send(conn, &format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n{CORS}Connection: close\r\n\r\n"))?;
        let first = chunk(
            obj([("role", "assistant".into()), ("content", "".into())]),
            None,
        );
        send(conn, &format!("data: {first}\n\n"))?;
    }
    let mut stops = StopStrings::new(req.stop);
    let mut text = String::new();
    let mut hit_stop = false;
    let summary: Summary = loop {
        match rx.recv() {
            Ok(Event::Text(piece)) if !hit_stop => {
                let (out, hit) = stops.push(&piece);
                if hit {
                    hit_stop = true;
                    cancel.store(true, Ordering::Relaxed);
                }
                if out.is_empty() {
                    continue;
                }
                if req.stream {
                    send(
                        conn,
                        &format!("data: {}\n\n", chunk(obj([("content", out.into())]), None)),
                    )?;
                } else {
                    text.push_str(&out);
                }
            }
            Ok(Event::Text(_)) => {}
            Ok(Event::Done(s)) => break s,
            Ok(Event::Error(e)) if req.stream => {
                return send(conn, &format!("data: {}\n\n", obj([("error", e.into())])))
            }
            Ok(Event::Error(e)) => return error(conn, "400 Bad Request", &e),
            Err(_) => return error(conn, "500 Internal Server Error", "the scheduler stopped"),
        }
    };
    if !hit_stop {
        let rest = stops.finish();
        if req.stream && !rest.is_empty() {
            send(
                conn,
                &format!("data: {}\n\n", chunk(obj([("content", rest.into())]), None)),
            )?;
        } else {
            text.push_str(&rest);
        }
    }

    let finish = if summary.finish == Finish::Length && !hit_stop {
        "length"
    } else {
        "stop"
    };
    let usage = obj([
        ("prompt_tokens", summary.prompt_tokens.into()),
        ("completion_tokens", summary.completion_tokens.into()),
        (
            "total_tokens",
            (summary.prompt_tokens + summary.completion_tokens).into(),
        ),
    ]);
    let timings = obj([
        ("cached_tokens", summary.cached_tokens.into()),
        ("time_to_first_token", summary.ttft_secs.into()),
        ("decode_tokens_per_second", summary.decode_tps.into()),
        ("mean_batch", summary.mean_batch.into()),
        ("backend", server.backend.as_str().into()),
    ]);
    if req.stream {
        let Json::Obj(mut last) = chunk(obj([]), Some(finish)) else {
            unreachable!()
        };
        last.push(("usage".into(), usage));
        last.push(("timings".into(), timings));
        send(
            conn,
            &format!("data: {}\n\ndata: [DONE]\n\n", Json::Obj(last)),
        )
    } else {
        let message = obj([("role", "assistant".into()), ("content", text.into())]);
        let body = obj([
            ("id", id.as_str().into()),
            ("object", "chat.completion".into()),
            ("created", created.into()),
            ("model", server.model_id.as_str().into()),
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

#[cfg(test)]
mod tests {
    use super::StopStrings;

    #[test]
    fn stop_strings_cut_and_hold_back() {
        let mut s = StopStrings::new(vec!["END".into(), "\n\n".into()]);
        assert_eq!(s.push("hello E"), ("hello ".into(), false)); // "E" might start "END"
        assert_eq!(s.push("x"), ("Ex".into(), false));
        assert_eq!(s.push(" fin"), (" fin".into(), false));
        assert_eq!(s.push("al EN"), ("al ".into(), false));
        assert_eq!(s.push("D more"), ("".into(), true));
        let mut s = StopStrings::new(vec!["\n\n".into()]);
        assert_eq!(s.push("a\n"), ("a".into(), false));
        assert_eq!(s.push("b"), ("\nb".into(), false));
        assert_eq!(s.finish(), "");
    }
}
