//! Chat prompt formats. ChatML, which Qwen was trained on:
//!
//! ```text
//! <|im_start|>system
//! You are a helpful assistant.<|im_end|>
//! <|im_start|>user
//! Hi!<|im_end|>
//! <|im_start|>assistant
//! ```
//!
//! and Llama 3's, whose system turn always carries the date:
//!
//! ```text
//! <|begin_of_text|><|start_header_id|>system<|end_header_id|>
//!
//! Cutting Knowledge Date: December 2023
//! Today Date: 26 Jul 2024
//!
//! <|eot_id|><|start_header_id|>user<|end_header_id|>
//!
//! Hi!<|eot_id|><|start_header_id|>assistant<|end_header_id|>
//!
//! ```
//!
//! and Granite's, `<|start_of_role|>user<|end_of_role|>Hi!<|end_of_text|>`
//! with a newline after each turn and a default system message. Each
//! family also has its own way of offering tools and of calling them.
//!
//! Control tokens are inserted by id; message text is encoded with added
//! tokens disabled, so a user typing `<|im_start|>` cannot forge a turn.

use std::time::{SystemTime, UNIX_EPOCH};

use crate::json::Json;
use crate::tokenizer::Tokenizer;
use crate::{Error, Result};

pub const DEFAULT_SYSTEM: &str =
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

/// The system prompt a model's own chat template adds when none is given:
/// Qwen2.5's names itself; Qwen3's adds none.
pub fn default_system(model_type: &str) -> Option<&'static str> {
    (model_type == "qwen2").then_some(DEFAULT_SYSTEM)
}

#[derive(Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
    /// Calls an assistant message made (OpenAI's `tool_calls`).
    pub tool_calls: Vec<ToolCall>,
}

impl Message {
    pub fn new(role: &str, content: &str) -> Message {
        Message {
            role: role.to_string(),
            content: content.to_string(),
            tool_calls: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// The arguments as a JSON object.
    pub arguments: Json,
}

const TOOLS_HEADER: &str = "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>";
const TOOLS_FOOTER: &str = "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>";

/// The tool calls in a reply: each `<tool_call>{"name": .., "arguments": ..}</tool_call>`
/// block, and the text before the first one.
pub fn parse_tool_calls(text: &str) -> (String, Vec<ToolCall>) {
    let mut calls = Vec::new();
    let content = text
        .split("<tool_call>")
        .next()
        .unwrap_or("")
        .trim()
        .to_string();
    for (i, block) in text.split("<tool_call>").skip(1).enumerate() {
        let body = block.split("</tool_call>").next().unwrap_or("").trim();
        if let Some(call) = Json::parse(body).ok().and_then(|v| call_from_json(&v, i)) {
            calls.push(call);
        }
    }
    (content, calls)
}

/// `{"name": .., "arguments" or "parameters": ..}` as a call; arguments may
/// also come as a JSON string.
fn call_from_json(v: &Json, i: usize) -> Option<ToolCall> {
    let name = v.get("name")?.as_str()?;
    let arguments = match v.get("arguments").or(v.get("parameters")) {
        Some(Json::Str(s)) => Json::parse(s).unwrap_or(Json::Obj(Vec::new())),
        Some(a @ Json::Obj(_)) => a.clone(),
        Some(_) => return None,
        None => Json::Obj(Vec::new()),
    };
    Some(ToolCall {
        id: format!("call_{i}"),
        name: name.to_string(),
        arguments,
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Style {
    ChatMl,
    Llama3,
    Granite,
}

pub struct ChatTemplate {
    style: Style,
    /// `<|im_start|>`, Llama 3's `<|start_header_id|>` or Granite's
    /// `<|start_of_role|>`.
    start: u32,
    /// After the role: Llama 3's `<|end_header_id|>`, Granite's
    /// `<|end_of_role|>`.
    end_header: u32,
    /// `<|im_end|>`, Llama 3's `<|eot_id|>` or Granite's `<|end_of_text|>`.
    end: u32,
    /// What follows `end`: a newline in ChatML and Granite.
    after_end: Vec<u32>,
    /// Llama 3's `<|begin_of_text|>`.
    bos: Option<u32>,
    /// Granite's `<|tool_call|>`, which starts a reply's list of calls.
    tool_call: Option<u32>,
}

impl ChatTemplate {
    /// The format the tokenizer's control tokens belong to.
    pub fn new(tok: &Tokenizer) -> Result<ChatTemplate> {
        let id = |s: &str| tok.added_token(s);
        if let (Some(start), Some(end)) = (id("<|im_start|>"), id("<|im_end|>")) {
            return Ok(ChatTemplate {
                style: Style::ChatMl,
                start,
                end_header: start,
                end,
                after_end: tok.encode("\n", false),
                bos: None,
                tool_call: None,
            });
        }
        if let (Some(start), Some(end_header), Some(end)) = (
            id("<|start_header_id|>"),
            id("<|end_header_id|>"),
            id("<|eot_id|>"),
        ) {
            return Ok(ChatTemplate {
                style: Style::Llama3,
                start,
                end_header,
                end,
                after_end: Vec::new(),
                bos: id("<|begin_of_text|>"),
                tool_call: None,
            });
        }
        if let (Some(start), Some(end_header), Some(end)) = (
            id("<|start_of_role|>"),
            id("<|end_of_role|>"),
            id("<|end_of_text|>"),
        ) {
            return Ok(ChatTemplate {
                style: Style::Granite,
                start,
                end_header,
                end,
                after_end: tok.encode("\n", false),
                bos: None,
                tool_call: id("<|tool_call|>"),
            });
        }
        Err(Error::msg(
            "the tokenizer has no ChatML, Llama 3 or Granite chat tokens; use --raw",
        ))
    }

    /// The token that ends a turn.
    pub fn end_of_turn(&self) -> u32 {
        self.end
    }

    /// Whether tool calling is implemented for this format.
    pub fn supports_tools(&self) -> bool {
        self.style != Style::Granite || self.tool_call.is_some()
    }

    /// What a reply that calls a tool starts with: Qwen's `<tool_call>`
    /// block, the bare JSON object Llama 3 answers with, or Granite's
    /// `<|tool_call|>` and a list (which its models also write out as plain
    /// `<tool_call>`, as vLLM's parser allows).
    pub fn call_openers(&self) -> &'static [&'static str] {
        match self.style {
            Style::ChatMl => &["<tool_call>"],
            Style::Llama3 => &["{"],
            Style::Granite => &["<|tool_call|>", "<tool_call>"],
        }
    }

    /// Special tokens whose text belongs in a reply: Granite's
    /// `<|tool_call|>`, which marks a reply as calls.
    pub fn visible_tokens(&self) -> Vec<u32> {
        self.tool_call.into_iter().collect()
    }

    /// Whether calls come as a JSON list (Granite) rather than one object.
    pub fn calls_as_list(&self) -> bool {
        self.style == Style::Granite
    }

    /// The key a call's arguments go under.
    pub fn arguments_key(&self) -> &'static str {
        match self.style {
            Style::ChatMl | Style::Granite => "arguments",
            Style::Llama3 => "parameters",
        }
    }

    /// Starts a reply that must be a call; the rest is constrained to the
    /// call's JSON.
    pub fn open_call(&self, tok: &Tokenizer, out: &mut Vec<u32>) {
        match self.style {
            Style::ChatMl => out.extend(tok.encode("<tool_call>\n", true)),
            Style::Granite => out.extend(self.tool_call),
            Style::Llama3 => {}
        }
    }

    /// A reply's text and the tool calls in it. `opened`: the reply began
    /// inside a call that `open_call` started.
    pub fn parse_calls(&self, text: &str, opened: bool) -> (String, Vec<ToolCall>) {
        match self.style {
            Style::ChatMl if opened => (
                String::new(),
                parse_tool_calls(&format!("<tool_call>{text}</tool_call>")).1,
            ),
            Style::ChatMl => parse_tool_calls(text),
            Style::Granite => {
                let found = ["<|tool_call|>", "<tool_call>"]
                    .iter()
                    .filter_map(|m| text.split_once(m))
                    .min_by_key(|(before, _)| before.len());
                let (content, list) = match found {
                    _ if opened => ("", text),
                    Some(split) => split,
                    None => return (text.to_string(), Vec::new()),
                };
                let calls: Vec<ToolCall> = match Json::parse(list.trim()) {
                    Ok(Json::Arr(items)) => items
                        .iter()
                        .enumerate()
                        .filter_map(|(i, v)| call_from_json(v, i))
                        .collect(),
                    Ok(v) => call_from_json(&v, 0).into_iter().collect(),
                    Err(_) => Vec::new(),
                };
                if calls.is_empty() {
                    (text.to_string(), calls)
                } else {
                    (content.trim().to_string(), calls)
                }
            }
            Style::Llama3 => match Json::parse(text.trim())
                .ok()
                .and_then(|v| call_from_json(&v, 0))
            {
                Some(call) => (String::new(), vec![call]),
                None => (text.to_string(), Vec::new()),
            },
        }
    }

    /// A whole conversation, ending with an open assistant turn.
    pub fn encode(&self, tok: &Tokenizer, messages: &[Message], system: Option<&str>) -> Vec<u32> {
        let (system, rest) = match messages.first() {
            Some(m) if m.role == "system" => (Some(m.content.as_str()), &messages[1..]),
            _ => (system, messages),
        };
        let mut out = Vec::new();
        self.begin(tok, &mut out, system);
        for m in rest {
            self.push(tok, &mut out, &m.role, &m.content);
        }
        self.open_assistant(tok, &mut out);
        out
    }

    /// What a conversation starts with: the system turn, if any (Llama 3's
    /// always has one, with the date).
    pub fn begin(&self, tok: &Tokenizer, out: &mut Vec<u32>, system: Option<&str>) {
        match self.style {
            Style::ChatMl => {
                if let Some(system) = system {
                    self.push(tok, out, "system", system);
                }
            }
            Style::Llama3 => {
                out.extend(self.bos);
                self.header(tok, out, "system");
                let text = format!(
                    "\n\nCutting Knowledge Date: December 2023\nToday Date: {}\n\n{}",
                    date(today()),
                    system.unwrap_or("").trim()
                );
                out.extend(tok.encode(&text, false));
                self.close_turn(out);
            }
            Style::Granite => {
                let default = granite_system(" You are a helpful AI assistant.");
                self.push(tok, out, "system", system.unwrap_or(&default));
            }
        }
    }

    pub fn push(&self, tok: &Tokenizer, out: &mut Vec<u32>, role: &str, content: &str) {
        match self.style {
            Style::ChatMl => {
                out.push(self.start);
                out.extend(tok.encode(&format!("{role}\n{content}"), false));
            }
            Style::Llama3 => {
                self.header(tok, out, role);
                out.extend(tok.encode(&format!("\n\n{}", content.trim()), false));
            }
            Style::Granite => {
                self.header(tok, out, role);
                out.extend(tok.encode(content, false));
            }
        }
        self.close_turn(out);
    }

    fn header(&self, tok: &Tokenizer, out: &mut Vec<u32>, role: &str) {
        out.push(self.start);
        out.extend(tok.encode(role, false));
        out.push(self.end_header);
    }

    /// A conversation with tools available, as Qwen's templates render it:
    /// the tool signatures go in the system message, an assistant's calls
    /// are `<tool_call>` blocks, and consecutive tool results become one
    /// user turn of `<tool_response>` blocks. Only the template's own markup
    /// is encoded with added tokens; message text and tool definitions
    /// can't inject control tokens.
    pub fn encode_with_tools(
        &self,
        tok: &Tokenizer,
        messages: &[Message],
        system: Option<&str>,
        tools: &[Json],
    ) -> Vec<u32> {
        if tools.is_empty()
            && messages
                .iter()
                .all(|m| m.tool_calls.is_empty() && m.role != "tool")
        {
            return self.encode(tok, messages, system);
        }
        match self.style {
            Style::Llama3 => return self.encode_llama_tools(tok, messages, system, tools),
            Style::Granite => return self.encode_granite_tools(tok, messages, system, tools),
            Style::ChatMl => {}
        }
        let mut out = Vec::new();
        let (first_system, rest) = match messages.first() {
            Some(m) if m.role == "system" => (Some(m.content.as_str()), &messages[1..]),
            _ => (system, messages),
        };
        let markup = |out: &mut Vec<u32>, text: &str| out.extend(tok.encode(text, true));
        let text = |out: &mut Vec<u32>, text: &str| out.extend(tok.encode(text, false));

        out.push(self.start);
        text(&mut out, "system\n");
        if let Some(s) = first_system {
            text(&mut out, s);
            if !tools.is_empty() {
                text(&mut out, "\n\n");
            }
        }
        if !tools.is_empty() {
            markup(&mut out, TOOLS_HEADER);
            for t in tools {
                text(&mut out, &format!("\n{}", t.to_spaced_string()));
            }
            markup(&mut out, TOOLS_FOOTER);
        }
        self.close_turn(&mut out);

        let mut i = 0;
        while i < rest.len() {
            let m = &rest[i];
            match m.role.as_str() {
                "assistant" if !m.tool_calls.is_empty() => {
                    out.push(self.start);
                    text(&mut out, "assistant");
                    if !m.content.is_empty() {
                        text(&mut out, &format!("\n{}", m.content));
                    }
                    for c in &m.tool_calls {
                        markup(&mut out, "\n<tool_call>\n");
                        let call = format!(
                            "{{\"name\": {}, \"arguments\": {}}}",
                            Json::Str(c.name.clone()),
                            c.arguments.to_spaced_string()
                        );
                        text(&mut out, &call);
                        markup(&mut out, "\n</tool_call>");
                    }
                    self.close_turn(&mut out);
                }
                "tool" => {
                    out.push(self.start);
                    text(&mut out, "user");
                    while i < rest.len() && rest[i].role == "tool" {
                        markup(&mut out, "\n<tool_response>\n");
                        text(&mut out, &rest[i].content);
                        markup(&mut out, "\n</tool_response>");
                        i += 1;
                    }
                    self.close_turn(&mut out);
                    continue;
                }
                role => self.push(tok, &mut out, role, &m.content),
            }
            i += 1;
        }
        self.open_assistant(tok, &mut out);
        out
    }

    /// Tools as Llama 3.2's template renders them: an `ipython` environment
    /// line in the system turn, the tool definitions (as indented JSON) at
    /// the top of the first user turn, a call as the assistant's whole reply
    /// (`{"name": .., "parameters": ..}`, one per turn), and results in
    /// `ipython` turns, JSON-quoted as its template quotes them.
    fn encode_llama_tools(
        &self,
        tok: &Tokenizer,
        messages: &[Message],
        system: Option<&str>,
        tools: &[Json],
    ) -> Vec<u32> {
        let text = |out: &mut Vec<u32>, text: &str| out.extend(tok.encode(text, false));
        let (system, mut rest) = match messages.first() {
            Some(m) if m.role == "system" => (Some(m.content.as_str()), &messages[1..]),
            _ => (system, messages),
        };
        let mut out = Vec::new();
        out.extend(self.bos);
        self.header(tok, &mut out, "system");
        let env = if tools.is_empty() {
            ""
        } else {
            "Environment: ipython\n"
        };
        text(
            &mut out,
            &format!(
                "\n\n{env}Cutting Knowledge Date: December 2023\nToday Date: {}\n\n{}",
                date(today()),
                system.unwrap_or("").trim()
            ),
        );
        self.close_turn(&mut out);
        if let (false, Some(first)) = (tools.is_empty(), rest.first()) {
            self.header(tok, &mut out, "user");
            let mut intro = String::from(
                "\n\nGiven the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.\n\nRespond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.Do not use variables.\n\n",
            );
            for t in tools {
                intro.push_str(&t.to_indented_string(4));
                intro.push_str("\n\n");
            }
            intro.push_str(first.content.trim());
            text(&mut out, &intro);
            self.close_turn(&mut out);
            rest = &rest[1..];
        }
        for m in rest {
            match (m.role.as_str(), m.tool_calls.first()) {
                (_, Some(c)) => {
                    self.header(tok, &mut out, "assistant");
                    let call = format!(
                        "\n\n{{\"name\": {}, \"parameters\": {}}}",
                        Json::Str(c.name.clone()),
                        c.arguments.to_spaced_string()
                    );
                    text(&mut out, &call);
                    self.close_turn(&mut out);
                }
                ("tool" | "ipython", None) => {
                    self.header(tok, &mut out, "ipython");
                    text(&mut out, &format!("\n\n{}", Json::Str(m.content.clone())));
                    self.close_turn(&mut out);
                }
                (role, None) => self.push(tok, &mut out, role, &m.content),
            }
        }
        self.open_assistant(tok, &mut out);
        out
    }

    /// Tools as Granite 3.1's template renders them: a sentence about them
    /// in the default system message, their definitions (indented JSON) in
    /// a `tools` turn, and calls as `<|tool_call|>` and a JSON list. Its
    /// template writes no assistant calls back, so they are written as the
    /// model made them; results go in `tool_response` turns.
    fn encode_granite_tools(
        &self,
        tok: &Tokenizer,
        messages: &[Message],
        system: Option<&str>,
        tools: &[Json],
    ) -> Vec<u32> {
        let (system, rest) = match messages.first() {
            Some(m) if m.role == "system" => (Some(m.content.as_str()), &messages[1..]),
            _ => (system, messages),
        };
        let mut out = Vec::new();
        self.header(tok, &mut out, "system");
        match system {
            Some(s) => out.extend(tok.encode(s, false)),
            // The template's own text, which names <|tool_call|> (the token).
            None if !tools.is_empty() => {
                out.extend(tok.encode(&granite_system(GRANITE_TOOLS), true))
            }
            None => {
                out.extend(tok.encode(&granite_system(" You are a helpful AI assistant."), false))
            }
        }
        self.close_turn(&mut out);
        if !tools.is_empty() {
            self.header(tok, &mut out, "tools");
            out.extend(tok.encode(&Json::Arr(tools.to_vec()).to_indented_string(4), false));
            self.close_turn(&mut out);
        }
        for m in rest {
            match m.role.as_str() {
                "assistant" if !m.tool_calls.is_empty() => {
                    self.header(tok, &mut out, "assistant");
                    out.extend(tok.encode(&m.content, false));
                    out.extend(self.tool_call);
                    let calls: Vec<Json> = m
                        .tool_calls
                        .iter()
                        .map(|c| {
                            Json::Obj(vec![
                                ("name".into(), Json::Str(c.name.clone())),
                                ("arguments".into(), c.arguments.clone()),
                            ])
                        })
                        .collect();
                    out.extend(tok.encode(&Json::Arr(calls).to_spaced_string(), false));
                    self.close_turn(&mut out);
                }
                "tool" => self.push(tok, &mut out, "tool_response", &m.content),
                role => self.push(tok, &mut out, role, &m.content),
            }
        }
        self.open_assistant(tok, &mut out);
        out
    }

    pub fn open_assistant(&self, tok: &Tokenizer, out: &mut Vec<u32>) {
        match self.style {
            Style::ChatMl => {
                out.push(self.start);
                out.extend(tok.encode("assistant\n", false));
            }
            Style::Llama3 => {
                self.header(tok, out, "assistant");
                out.extend(tok.encode("\n\n", false));
            }
            Style::Granite => self.header(tok, out, "assistant"),
        }
    }

    /// Qwen3 thinks aloud in `<think>...</think>` before answering. Its
    /// template turns that off by starting the reply with an empty block.
    pub fn skip_thinking(&self, tok: &Tokenizer, out: &mut Vec<u32>) {
        if tok.added_token("<think>").is_some() {
            out.extend(tok.encode("<think>\n\n</think>\n\n", true));
        }
    }

    pub fn close_turn(&self, out: &mut Vec<u32>) {
        out.push(self.end);
        out.extend_from_slice(&self.after_end);
    }
}

const GRANITE_TOOLS: &str = " You are a helpful AI assistant with access to the following tools. When a tool is required to answer the user's query, respond with <|tool_call|> followed by a JSON list of tools used. If a tool does not exist in the provided list of tools, notify the user that you do not have the ability to fulfill the request.";

/// Granite's default system message, dated as its template dates it, then
/// `tail`.
fn granite_system(tail: &str) -> String {
    let (day, month, year) = civil(today());
    format!(
        "Knowledge Cutoff Date: April 2024.\nToday's Date: {} {day:02}, {year}.\nYou are Granite, developed by IBM.{tail}",
        MONTH_NAMES[month - 1]
    )
}

/// Days since 1970-01-01, UTC. The browser build has no clock, so it uses
/// the date Llama 3's template falls back to (26 Jul 2024).
fn today() -> i64 {
    if cfg!(target_arch = "wasm32") {
        return 19_930;
    }
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() / 86_400) as i64
}

const MONTH_NAMES: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];

/// A day number as Llama 3's template writes dates ("26 Jul 2024").
fn date(days: i64) -> String {
    let (day, month, year) = civil(days);
    format!("{day:02} {} {year}", &MONTH_NAMES[month - 1][..3])
}

/// (day, month 1-12, year) of a day number, by Howard Hinnant's
/// civil-from-days algorithm.
fn civil(days: i64) -> (i64, usize, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    (day, month as usize, year)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_like_llama_templates() {
        assert_eq!(date(0), "01 Jan 1970");
        assert_eq!(date(11_016), "29 Feb 2000");
        assert_eq!(date(19_930), "26 Jul 2024");
        assert_eq!(date(20_760), "03 Nov 2026");
    }

    #[test]
    fn parses_tool_calls_out_of_a_reply() {
        let reply = "Let me check.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"time\", \"arguments\": \"{}\"}\n</tool_call>";
        let (content, calls) = parse_tool_calls(reply);
        assert_eq!(content, "Let me check.");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(
            calls[0].arguments.get("city").and_then(Json::as_str),
            Some("Paris")
        );
        assert_eq!(calls[1].arguments, Json::Obj(vec![]));
        assert_eq!(parse_tool_calls("no calls here").1, vec![]);
    }
}
