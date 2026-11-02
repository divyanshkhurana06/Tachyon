//! ChatML, the prompt format Qwen2.5 was trained on:
//!
//! ```text
//! <|im_start|>system
//! You are a helpful assistant.<|im_end|>
//! <|im_start|>user
//! Hi!<|im_end|>
//! <|im_start|>assistant
//! ```
//!
//! Control tokens are inserted by id; message text is encoded with added
//! tokens disabled, so a user typing `<|im_start|>` cannot forge a turn.

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

pub struct ChatTemplate {
    im_start: u32,
    im_end: u32,
    newline: Vec<u32>,
}

impl ChatTemplate {
    pub fn new(tok: &Tokenizer) -> Result<ChatTemplate> {
        let find = |s: &str| {
            tok.added_token(s)
                .ok_or_else(|| Error(format!("tokenizer has no {s}; use --raw")))
        };
        Ok(ChatTemplate {
            im_start: find("<|im_start|>")?,
            im_end: find("<|im_end|>")?,
            newline: tok.encode("\n", false),
        })
    }

    pub fn im_end(&self) -> u32 {
        self.im_end
    }

    /// A whole conversation, ending with an open assistant turn.
    pub fn encode(&self, tok: &Tokenizer, messages: &[Message], system: Option<&str>) -> Vec<u32> {
        let mut out = Vec::new();
        if let Some(system) = system.filter(|_| messages.first().is_none_or(|m| m.role != "system"))
        {
            self.push(tok, &mut out, "system", system);
        }
        for m in messages {
            self.push(tok, &mut out, &m.role, &m.content);
        }
        self.open_assistant(tok, &mut out);
        out
    }

    pub fn push(&self, tok: &Tokenizer, out: &mut Vec<u32>, role: &str, content: &str) {
        out.push(self.im_start);
        out.extend(tok.encode(&format!("{role}\n{content}"), false));
        self.close_turn(out);
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
        let mut out = Vec::new();
        let (first_system, rest) = match messages.first() {
            Some(m) if m.role == "system" => (Some(m.content.as_str()), &messages[1..]),
            _ => (system, messages),
        };
        let markup = |out: &mut Vec<u32>, text: &str| out.extend(tok.encode(text, true));
        let text = |out: &mut Vec<u32>, text: &str| out.extend(tok.encode(text, false));

        out.push(self.im_start);
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
                    out.push(self.im_start);
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
                    out.push(self.im_start);
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
        out.push(self.im_start);
        out.extend(tok.encode("assistant\n", false));
    }

    /// Qwen3 thinks aloud in `<think>...</think>` before answering. Its
    /// template turns that off by starting the reply with an empty block.
    pub fn skip_thinking(&self, tok: &Tokenizer, out: &mut Vec<u32>) {
        if tok.added_token("<think>").is_some() {
            out.extend(tok.encode("<think>\n\n</think>\n\n", true));
        }
    }

    pub fn close_turn(&self, out: &mut Vec<u32>) {
        out.push(self.im_end);
        out.extend_from_slice(&self.newline);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
