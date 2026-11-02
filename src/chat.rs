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
}

impl Message {
    pub fn new(role: &str, content: &str) -> Message {
        Message {
            role: role.to_string(),
            content: content.to_string(),
        }
    }
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
