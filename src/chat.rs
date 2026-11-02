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
        let Ok(v) = Json::parse(body) else { continue };
        let Some(name) = v.get("name").and_then(Json::as_str) else {
            continue;
        };
        let arguments = match v.get("arguments") {
            Some(Json::Str(s)) => Json::parse(s).unwrap_or(Json::Obj(Vec::new())),
            Some(a) => a.clone(),
            None => Json::Obj(Vec::new()),
        };
        calls.push(ToolCall {
            id: format!("call_{i}"),
            name: name.to_string(),
            arguments,
        });
    }
    (content, calls)
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Style {
    ChatMl,
    Llama3,
}

pub struct ChatTemplate {
    style: Style,
    /// `<|im_start|>`, or Llama 3's `<|start_header_id|>`.
    start: u32,
    /// Llama 3's `<|end_header_id|>` after the role.
    end_header: u32,
    /// `<|im_end|>`, or Llama 3's `<|eot_id|>`.
    end: u32,
    /// What follows `end`: a newline in ChatML.
    after_end: Vec<u32>,
    /// Llama 3's `<|begin_of_text|>`.
    bos: Option<u32>,
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
            });
        }
        Err(Error::msg(
            "the tokenizer has no ChatML or Llama 3 chat tokens; use --raw",
        ))
    }

    /// The token that ends a turn.
    pub fn end_of_turn(&self) -> u32 {
        self.end
    }

    /// Whether this format has Qwen-style `<tool_call>` tool calling.
    pub fn supports_tools(&self) -> bool {
        self.style == Style::ChatMl
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
        if !self.supports_tools()
            || tools.is_empty()
                && messages
                    .iter()
                    .all(|m| m.tool_calls.is_empty() && m.role != "tool")
        {
            return self.encode(tok, messages, system);
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

/// Days since 1970-01-01, UTC.
fn today() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() / 86_400) as i64
}

/// A day number as Llama 3's template writes dates ("26 Jul 2024"), by
/// Howard Hinnant's civil-from-days algorithm.
fn date(days: i64) -> String {
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{day:02} {} {year}", MONTHS[month as usize - 1])
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
