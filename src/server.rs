//! An OpenAI-compatible HTTP server on `std::net`, with a minimal hand-written
//! HTTP/1.1 implementation.
//!
//! - `POST /v1/chat/completions`: chat, optionally streamed as server-sent
//!   events, so the official `openai` clients and chat UIs work unchanged.
//!   Supports `stop` strings, `response_format` (`json_object`, or
//!   `json_schema` for output matching a schema)
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

use crate::chat::{ChatTemplate, Message, ToolCall};
use crate::constrain::OutputFormat;
use crate::generate::Finish;
use crate::json::{obj, Json};
use crate::sampler::SamplerConfig;
use crate::sampler::TokenLogprob;
use crate::scheduler::{Event, Request, Scheduler, Summary};
use crate::schema::{JsonSchema, Schema};
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
    /// LoRA adapters a request can name as its `model` (ids from 1).
    pub adapters: Vec<String>,
}

impl Server {
    fn info(&self) -> Json {
        let st = &self.scheduler.stats;
        obj([
            ("model", self.model_id.as_str().into()),
            (
                "adapters",
                Json::Arr(self.adapters.iter().map(|a| a.as_str().into()).collect()),
            ),
            ("backend", self.backend.as_str().into()),
            ("params", self.params.into()),
            ("context", self.context.into()),
            ("kv_cache_tokens", self.kv_tokens.into()),
            ("running", st.running.load(Ordering::Relaxed).into()),
            ("waiting", st.waiting.load(Ordering::Relaxed).into()),
            ("steps", st.steps.load(Ordering::Relaxed).into()),
            ("tokens_generated", st.tokens.load(Ordering::Relaxed).into()),
            ("preemptions", st.preemptions.load(Ordering::Relaxed).into()),
            ("swaps", st.swaps.load(Ordering::Relaxed).into()),
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
            let mut models = vec![obj([
                ("id", server.model_id.as_str().into()),
                ("object", "model".into()),
                ("owned_by", "tachyon".into()),
            ])];
            // Adapters are listed as models of their own, as vLLM does.
            models.extend(server.adapters.iter().map(|a| {
                obj([
                    ("id", a.as_str().into()),
                    ("object", "model".into()),
                    ("owned_by", "tachyon".into()),
                    ("parent", server.model_id.as_str().into()),
                ])
            }));
            respond_json(
                &mut conn,
                "200 OK",
                &obj([("object", "list".into()), ("data", Json::Arr(models))]),
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
    /// Selects a LoRA adapter by name; anything else means the base model.
    model: Option<String>,
    /// vLLM's `priority`: lower is served first.
    priority: i32,
    messages: Vec<Message>,
    stream: bool,
    max_tokens: Option<usize>,
    sampler: SamplerConfig,
    stop: Vec<String>,
    format: OutputFormat,
    ignore_eos: bool,
    /// `logprobs`: report each token's log-probability and this many alternatives.
    logprobs: Option<usize>,
    /// Qwen3's `enable_thinking` (top level, or in `chat_template_kwargs`
    /// as vLLM takes it).
    thinking: bool,
    /// OpenAI-style tool definitions, as given.
    tools: Vec<Json>,
    tool_choice: ToolChoice,
}

#[derive(Debug, Clone, PartialEq)]
enum ToolChoice {
    Auto,
    None,
    /// Must call some tool.
    Required,
    /// Must call this one.
    Named(String),
}

fn tool_name(tool: &Json) -> Option<&str> {
    tool.get("function")
        .and_then(|f| f.get("name"))
        .and_then(Json::as_str)
}

/// The schema a forced call must match: `{"name": ..., <key>: ...}`, with
/// the named function's own parameter schema for the arguments (`key` is
/// "arguments" for Qwen, "parameters" for Llama 3).
fn call_schema(
    tools: &[Json],
    choice: &ToolChoice,
    key: &str,
    as_list: bool,
) -> std::result::Result<Schema, String> {
    let any_object = obj([("type", "object".into())]);
    let (name, arguments) = match choice {
        ToolChoice::Named(n) => {
            let tool = tools
                .iter()
                .find(|t| tool_name(t) == Some(n.as_str()))
                .ok_or(format!("tool_choice names unknown function {n:?}"))?;
            let params = tool
                .get("function")
                .and_then(|f| f.get("parameters"))
                .cloned()
                .unwrap_or(any_object);
            (obj([("const", n.as_str().into())]), params)
        }
        _ => {
            let names = tools.iter().filter_map(tool_name).map(Json::from).collect();
            (obj([("enum", Json::Arr(names))]), any_object)
        }
    };
    let schema = obj([
        ("type", "object".into()),
        (
            "properties",
            Json::Obj(vec![("name".into(), name), (key.into(), arguments)]),
        ),
        ("required", Json::Arr(vec!["name".into(), key.into()])),
    ]);
    // Granite calls are a list: `[{"name": ..., "arguments": ...}]`.
    let schema = if as_list {
        obj([
            ("type", "array".into()),
            ("items", schema),
            ("minItems", 1usize.into()),
            ("maxItems", 1usize.into()),
        ])
    } else {
        schema
    };
    Schema::parse(&schema).map_err(|e| format!("bad tool parameters: {e}"))
}

/// One `logprobs.content` entry, as OpenAI shapes it.
fn logprob_json(tok: &Tokenizer, lp: &TokenLogprob) -> Json {
    let entry = |t: u32, logprob: f32| {
        let bytes = tok.token_bytes(t);
        vec![
            (
                "token".to_string(),
                Json::from(String::from_utf8_lossy(bytes).into_owned()),
            ),
            ("logprob".to_string(), Json::Num(logprob as f64)),
            (
                "bytes".to_string(),
                Json::Arr(bytes.iter().map(|&b| Json::from(b as usize)).collect()),
            ),
        ]
    };
    let mut fields = entry(lp.token, lp.logprob);
    fields.push((
        "top_logprobs".to_string(),
        Json::Arr(
            lp.top
                .iter()
                .map(|&(t, l)| Json::Obj(entry(t, l)))
                .collect(),
        ),
    ));
    Json::Obj(fields)
}

/// Adds `logprobs: {"content": entries}` to a chunk's choice, when asked for.
fn with_logprobs(mut chunk: Json, entries: Option<&[Json]>) -> Json {
    if let (Some(entries), Json::Obj(fields)) = (entries, &mut chunk) {
        if let Some((_, Json::Arr(choices))) = fields.iter_mut().find(|(k, _)| k == "choices") {
            if let Some(Json::Obj(choice)) = choices.first_mut() {
                choice.push((
                    "logprobs".to_string(),
                    obj([("content", Json::Arr(entries.to_vec()))]),
                ));
            }
        }
    }
    chunk
}

fn tool_call_json(c: &ToolCall, index: Option<usize>) -> Json {
    let mut fields = Vec::new();
    if let Some(i) = index {
        fields.push(("index".to_string(), i.into()));
    }
    fields.push(("id".to_string(), c.id.as_str().into()));
    fields.push(("type".to_string(), "function".into()));
    fields.push((
        "function".to_string(),
        obj([
            ("name", c.name.as_str().into()),
            ("arguments", c.arguments.to_string().into()),
        ]),
    ));
    Json::Obj(fields)
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
        let mut message = Message::new(role, &content);
        for (i, call) in m
            .get("tool_calls")
            .and_then(Json::as_array)
            .unwrap_or(&[])
            .iter()
            .enumerate()
        {
            let f = call
                .get("function")
                .ok_or("each tool call needs a `function`")?;
            let name = f
                .get("name")
                .and_then(Json::as_str)
                .ok_or("each tool call needs a function name")?;
            let arguments = match f.get("arguments") {
                Some(Json::Str(a)) => {
                    Json::parse(a).map_err(|e| format!("tool call arguments are not JSON: {e}"))?
                }
                Some(a) => a.clone(),
                None => Json::Obj(Vec::new()),
            };
            let id = call
                .get("id")
                .and_then(Json::as_str)
                .map_or(format!("call_{i}"), str::to_string);
            message.tool_calls.push(ToolCall {
                id,
                name: name.to_string(),
                arguments,
            });
        }
        messages.push(message);
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
    let rf = j.get("response_format");
    let format = match rf.and_then(|f| f.get("type")).and_then(Json::as_str) {
        None | Some("text") => OutputFormat::Text,
        Some("json_object") => OutputFormat::JsonObject,
        Some("json_schema") => {
            let schema = rf
                .and_then(|f| f.get("json_schema"))
                .and_then(|s| s.get("schema"))
                .ok_or("response_format json_schema needs json_schema.schema")?;
            OutputFormat::JsonSchema(JsonSchema::new(
                Schema::parse(schema).map_err(|e| format!("bad schema: {e}"))?,
            ))
        }
        Some(other) => {
            return Err(format!(
                "response_format type {other:?} is not supported (text, json_object, json_schema)"
            ))
        }
    };
    let tools: Vec<Json> = j
        .get("tools")
        .and_then(Json::as_array)
        .unwrap_or(&[])
        .to_vec();
    if tools.iter().any(|t| tool_name(t).is_none()) {
        return Err("each tool needs `function.name`".into());
    }
    let tool_choice = match j.get("tool_choice") {
        None | Some(Json::Null) => ToolChoice::Auto,
        Some(Json::Str(c)) if c == "auto" => ToolChoice::Auto,
        Some(Json::Str(c)) if c == "none" => ToolChoice::None,
        Some(Json::Str(c)) if c == "required" => ToolChoice::Required,
        Some(c) => match c.get("function").and_then(|f| f.get("name")).and_then(Json::as_str) {
            Some(n) => ToolChoice::Named(n.to_string()),
            None => return Err("tool_choice must be auto, none, required or {\"type\": \"function\", \"function\": {\"name\": ...}}".into()),
        },
    };
    if tool_choice != ToolChoice::Auto && tool_choice != ToolChoice::None && tools.is_empty() {
        return Err("tool_choice needs `tools`".into());
    }
    Ok(ChatRequest {
        model: j.get("model").and_then(Json::as_str).map(str::to_string),
        priority: j
            .get("priority")
            .and_then(Json::as_f64)
            .map_or(0, |p| p.clamp(i32::MIN as f64, i32::MAX as f64) as i32),
        messages,
        stream: j.get("stream").and_then(Json::as_bool).unwrap_or(false),
        max_tokens: j
            .get("max_tokens")
            .or(j.get("max_completion_tokens"))
            .and_then(Json::as_usize),
        sampler,
        stop: stop.into_iter().filter(|s| !s.is_empty()).collect(),
        format,
        ignore_eos: j.get("ignore_eos").and_then(Json::as_bool).unwrap_or(false),
        thinking: j
            .get("enable_thinking")
            .or(j
                .get("chat_template_kwargs")
                .and_then(|k| k.get("enable_thinking")))
            .and_then(Json::as_bool)
            .unwrap_or(true),
        tools,
        tool_choice,
        logprobs: j
            .get("logprobs")
            .and_then(Json::as_bool)
            .unwrap_or(false)
            .then(|| {
                j.get("top_logprobs")
                    .and_then(Json::as_usize)
                    .unwrap_or(0)
                    .min(20)
            }),
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
    let uses_tools = !req.tools.is_empty()
        || req
            .messages
            .iter()
            .any(|m| m.role == "tool" || !m.tool_calls.is_empty());
    if uses_tools && !server.template.supports_tools() {
        let msg = "this model's chat format has no tool calling here yet";
        return error(conn, "400 Bad Request", msg);
    }
    let shown: &[Json] = if req.tool_choice == ToolChoice::None {
        &[]
    } else {
        &req.tools
    };
    let mut prompt = server.template.encode_with_tools(
        &server.tok,
        &req.messages,
        server.system.as_deref(),
        shown,
    );
    if !req.thinking {
        server.template.skip_thinking(&server.tok, &mut prompt);
    }
    // A forced call: open the call ourselves (Qwen's <tool_call>) and
    // constrain what follows to {"name": ..., "arguments": <its schema>}.
    let forced = matches!(req.tool_choice, ToolChoice::Required | ToolChoice::Named(_));
    let format = if forced {
        server.template.open_call(&server.tok, &mut prompt);
        let key = server.template.arguments_key();
        match call_schema(
            &req.tools,
            &req.tool_choice,
            key,
            server.template.calls_as_list(),
        ) {
            Ok(schema) => OutputFormat::JsonSchema(JsonSchema::new(schema)),
            Err(e) => return error(conn, "400 Bad Request", &e),
        }
    } else {
        req.format.clone()
    };
    let room = server.context.saturating_sub(prompt.len());
    if room == 0 {
        let msg = format!(
            "the prompt is {} tokens; the context window is {}",
            prompt.len(),
            server.context
        );
        return error(conn, "400 Bad Request", &msg);
    }
    let adapter = server
        .adapters
        .iter()
        .position(|a| Some(a) == req.model.as_ref());
    let model_id = adapter.map_or(server.model_id.as_str(), |i| &server.adapters[i]);
    let (tx, rx) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    server.scheduler.submit(Request {
        adapter: adapter.map_or(0, |i| i as u32 + 1),
        visible: server.template.visible_tokens(),
        priority: req.priority,
        prompt,
        max_tokens: req.max_tokens.unwrap_or(server.max_tokens).min(room),
        sampler: req.sampler,
        stop_tokens: if req.ignore_eos {
            Vec::new()
        } else {
            server.stop.clone()
        },
        constraint: format.constraint(),
        events: tx,
        cancel: cancel.clone(),
        logprobs: req.logprobs,
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
            ("model", model_id.into()),
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
    // With tools, hold the reply back until it is clearly text or a call.
    let tools_on = !shown.is_empty();
    let mut held: Option<String> = (tools_on || forced).then(String::new);
    let mut sent = 0; // bytes of `text` already streamed
    let mut lps: Vec<Json> = Vec::new();
    let mut lps_sent = 0;
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
                text.push_str(&out);
                let mut piece = out;
                if let Some(buf) = held.as_mut() {
                    buf.push_str(&piece);
                    let start = buf.trim_start();
                    let openers = server.template.call_openers();
                    let maybe = |o: &&str| start.starts_with(o) || o.starts_with(start);
                    if forced || openers.iter().any(maybe) {
                        continue; // a call, or still might be
                    }
                    piece = held.take().unwrap();
                }
                if req.stream {
                    let c = with_logprobs(
                        chunk(obj([("content", piece.into())]), None),
                        req.logprobs.map(|_| &lps[lps_sent..]),
                    );
                    lps_sent = lps.len();
                    send(conn, &format!("data: {c}\n\n"))?;
                    sent = text.len();
                }
            }
            Ok(Event::Text(_)) => {}
            Ok(Event::Logprob(lp)) => {
                if !hit_stop {
                    lps.push(logprob_json(&server.tok, &lp));
                }
            }
            Ok(Event::Done(s)) => break s,
            Ok(Event::Error(e)) if req.stream => {
                return send(conn, &format!("data: {}\n\n", obj([("error", e.into())])))
            }
            Ok(Event::Error(e)) => return error(conn, "400 Bad Request", &e),
            Err(_) => return error(conn, "500 Internal Server Error", "the scheduler stopped"),
        }
    };
    if !hit_stop {
        text.push_str(&stops.finish());
    }
    let (content, mut calls) = if forced || tools_on {
        server.template.parse_calls(&text, forced)
    } else {
        (text.clone(), Vec::new())
    };
    for (i, c) in calls.iter_mut().enumerate() {
        c.id = format!("call_{}_{i}", &id[9..]);
    }
    if req.stream {
        // Whatever is still held back (or was never streamed) goes out now.
        let unsent = if calls.is_empty() {
            text[sent..].to_string()
        } else if held.is_some() {
            content.clone() // a call came first; its preamble was never sent
        } else {
            String::new()
        };
        if !unsent.is_empty() || (req.logprobs.is_some() && lps_sent < lps.len()) {
            let c = with_logprobs(
                chunk(obj([("content", unsent.into())]), None),
                req.logprobs.map(|_| &lps[lps_sent..]),
            );
            send(conn, &format!("data: {c}\n\n"))?;
        }
        if !calls.is_empty() {
            let deltas = calls
                .iter()
                .enumerate()
                .map(|(i, c)| tool_call_json(c, Some(i)))
                .collect();
            send(
                conn,
                &format!(
                    "data: {}\n\n",
                    chunk(obj([("tool_calls", Json::Arr(deltas))]), None)
                ),
            )?;
        }
    }

    let finish = if !calls.is_empty() {
        "tool_calls"
    } else if summary.finish == Finish::Length && !hit_stop {
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
        let mut message = vec![
            ("role".to_string(), Json::from("assistant")),
            (
                "content".to_string(),
                if content.is_empty() && !calls.is_empty() {
                    Json::Null
                } else {
                    content.into()
                },
            ),
        ];
        if !calls.is_empty() {
            message.push((
                "tool_calls".to_string(),
                Json::Arr(calls.iter().map(|c| tool_call_json(c, None)).collect()),
            ));
        }
        let message = Json::Obj(message);
        let body = obj([
            ("id", id.as_str().into()),
            ("object", "chat.completion".into()),
            ("created", created.into()),
            ("model", model_id.into()),
            (
                "choices",
                Json::Arr(vec![obj([
                    ("index", 0usize.into()),
                    ("message", message),
                    ("finish_reason", finish.into()),
                    (
                        "logprobs",
                        if req.logprobs.is_some() {
                            obj([("content", Json::Arr(lps))])
                        } else {
                            Json::Null
                        },
                    ),
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
