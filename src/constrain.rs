//! Constrained decoding: restrict sampling to tokens that keep the output
//! valid, so it is guaranteed to parse.
//!
//! [`JsonObject`] is a byte-level pushdown automaton for JSON (RFC 8259)
//! whose top-level value must be an object, as in OpenAI's
//! `response_format: {"type": "json_object"}`. Before a token is sampled,
//! its bytes are run through a copy of the automaton; tokens that would make
//! the output invalid are skipped. Once the object closes, generation stops.

use crate::tokenizer::Tokenizer;

pub trait Constraint: Send {
    /// May `token` come next?
    fn allows(&self, tok: &Tokenizer, token: u32) -> bool;
    /// Feeds the bytes of the token that was chosen.
    fn advance(&mut self, bytes: &[u8]);
    /// The output is complete and generation should end.
    fn is_complete(&self) -> bool;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Num {
    Minus,
    Zero,
    Int,
    Dot,
    Frac,
    Exp,
    ExpSign,
    ExpDigits,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// A value must start here (the top level accepts only `{`).
    Value,
    /// Just after `[`: a value or `]`.
    ArrayStart,
    /// Just after `{`: a key or `}`.
    ObjectStart,
    /// After `,` in an object: a key.
    Key,
    /// After a key: `:`.
    Colon,
    /// After a value inside a container: `,` or the closing bracket.
    After,
    /// Inside a string. `hex` counts the digits left of a `\u` escape.
    Str {
        key: bool,
        escape: bool,
        hex: u8,
    },
    Number(Num, u8),
    /// Partway through `true`, `false` or `null`.
    Literal(&'static [u8], u8),
    Done,
}

/// Generous but finite, so the model can't stall forever on padding.
const MAX_WHITESPACE: u8 = 16;
const MAX_NUMBER: u8 = 40;
const MAX_DEPTH: usize = 32;

#[derive(Debug, Clone)]
pub struct JsonObject {
    stack: Vec<bool>, // true = object, false = array
    state: State,
    whitespace: u8,
}

impl Default for JsonObject {
    fn default() -> Self {
        JsonObject {
            stack: Vec::new(),
            state: State::Value,
            whitespace: 0,
        }
    }
}

impl JsonObject {
    fn value_done(&mut self) {
        self.state = if self.stack.is_empty() {
            State::Done
        } else {
            State::After
        };
    }

    fn open(&mut self, object: bool) -> bool {
        if self.stack.len() >= MAX_DEPTH {
            return false;
        }
        self.stack.push(object);
        self.state = if object {
            State::ObjectStart
        } else {
            State::ArrayStart
        };
        true
    }

    fn close(&mut self, object: bool) -> bool {
        if self.stack.last() != Some(&object) {
            return false;
        }
        self.stack.pop();
        self.value_done();
        true
    }

    fn start_value(&mut self, b: u8) -> bool {
        if self.stack.is_empty() {
            return b == b'{' && self.open(true);
        }
        self.state = match b {
            b'{' => return self.open(true),
            b'[' => return self.open(false),
            b'"' => State::Str {
                key: false,
                escape: false,
                hex: 0,
            },
            b'-' => State::Number(Num::Minus, 1),
            b'0' => State::Number(Num::Zero, 1),
            b'1'..=b'9' => State::Number(Num::Int, 1),
            b't' => State::Literal(b"true", 1),
            b'f' => State::Literal(b"false", 1),
            b'n' => State::Literal(b"null", 1),
            _ => return false,
        };
        true
    }

    /// Feeds one byte; false if it cannot appear here.
    fn step(&mut self, b: u8) -> bool {
        let ws = matches!(b, b' ' | b'\t' | b'\n' | b'\r');
        match self.state {
            State::Str { key, escape, hex } => {
                return match (escape, hex) {
                    (_, n) if n > 0 => {
                        let ok = b.is_ascii_hexdigit();
                        self.state = State::Str {
                            key,
                            escape: false,
                            hex: n - 1,
                        };
                        ok
                    }
                    (true, _) => {
                        self.state = State::Str {
                            key,
                            escape: false,
                            hex: if b == b'u' { 4 } else { 0 },
                        };
                        matches!(
                            b,
                            b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' | b'u'
                        )
                    }
                    _ if b == b'"' => {
                        if key {
                            self.state = State::Colon;
                        } else {
                            self.value_done();
                        }
                        true
                    }
                    _ if b == b'\\' => {
                        self.state = State::Str {
                            key,
                            escape: true,
                            hex: 0,
                        };
                        true
                    }
                    _ => b >= 0x20,
                };
            }
            State::Number(n, len) => {
                let digit = b.is_ascii_digit();
                let next = match (n, b) {
                    (Num::Minus, b'0') => Some(Num::Zero),
                    (Num::Minus, _) if digit => Some(Num::Int),
                    (Num::Zero | Num::Int, b'.') => Some(Num::Dot),
                    (Num::Int, _) if digit => Some(Num::Int),
                    (Num::Dot | Num::Frac, _) if digit => Some(Num::Frac),
                    (Num::Zero | Num::Int | Num::Frac, b'e' | b'E') => Some(Num::Exp),
                    (Num::Exp, b'+' | b'-') => Some(Num::ExpSign),
                    (Num::Exp | Num::ExpSign | Num::ExpDigits, _) if digit => Some(Num::ExpDigits),
                    _ => None,
                };
                if let Some(next) = next {
                    self.state = State::Number(next, len + 1);
                    return len < MAX_NUMBER;
                }
                // The number ends here if it can; the byte then belongs to what follows.
                if !matches!(n, Num::Zero | Num::Int | Num::Frac | Num::ExpDigits) {
                    return false;
                }
                self.value_done();
            }
            State::Literal(word, at) => {
                if word[at as usize] != b {
                    return false;
                }
                if at as usize + 1 == word.len() {
                    self.value_done();
                } else {
                    self.state = State::Literal(word, at + 1);
                }
                return true;
            }
            _ => {}
        }

        if ws {
            self.whitespace += 1;
            return self.whitespace <= MAX_WHITESPACE && self.state != State::Done;
        }
        self.whitespace = 0;
        match self.state {
            State::Value => self.start_value(b),
            State::ArrayStart if b == b']' => self.close(false),
            State::ArrayStart => self.start_value(b),
            State::ObjectStart if b == b'}' => self.close(true),
            State::ObjectStart | State::Key if b == b'"' => {
                self.state = State::Str {
                    key: true,
                    escape: false,
                    hex: 0,
                };
                true
            }
            State::Colon if b == b':' => {
                self.state = State::Value;
                true
            }
            State::After => match (b, self.stack.last()) {
                (b',', Some(&true)) => {
                    self.state = State::Key;
                    true
                }
                (b',', Some(&false)) => {
                    self.state = State::Value;
                    true
                }
                (b'}', _) => self.close(true),
                (b']', _) => self.close(false),
                _ => false,
            },
            _ => false,
        }
    }

    /// Whether these bytes may come next.
    pub fn accepts_bytes(&self, bytes: &[u8]) -> bool {
        let mut probe = self.clone();
        bytes.iter().all(|&b| probe.step(b))
    }
}

impl Constraint for JsonObject {
    fn allows(&self, tok: &Tokenizer, token: u32) -> bool {
        let bytes = tok.token_bytes(token);
        !tok.is_special(token) && !bytes.is_empty() && self.accepts_bytes(bytes)
    }

    fn advance(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.step(b);
        }
    }

    fn is_complete(&self) -> bool {
        self.state == State::Done
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(text: &str) -> (bool, bool) {
        let mut j = JsonObject::default();
        let ok = j.accepts_bytes(text.as_bytes());
        j.advance(text.as_bytes());
        (ok, j.is_complete())
    }

    #[test]
    fn accepts_valid_objects() {
        for text in [
            r#"{}"#,
            r#"{"a": 1}"#,
            r#"{"name": "Ada", "age": 36, "langs": ["en", "fr"], "ok": true, "x": null}"#,
            r#"{"n": -0.5e+10, "z": 0, "e": "é\n\"q\""}"#,
            "{\n  \"nested\": {\"deep\": [[1, 2], {\"k\": false}]}\n}",
        ] {
            assert_eq!(check(text), (true, true), "{text}");
        }
    }

    #[test]
    fn accepts_prefixes_but_not_completion() {
        for text in [
            r#"{"a": "unfinished"#,
            r#"{"a": [1, 2"#,
            r#"{"a": tr"#,
            r#"{"a": 12"#,
        ] {
            assert_eq!(check(text), (true, false), "{text}");
        }
    }

    #[test]
    fn rejects_invalid_json() {
        for text in [
            "[1]",
            "\"top\"",
            "{a: 1}",
            r#"{"a" 1}"#,
            r#"{"a": 01}"#,
            r#"{"a": 1,}"#,
            r#"{"a": tru}"#,
            r#"{"a": "\x"}"#,
            r#"{"a": [1}"#,
            r#"{"a": 1}}"#,
            "{\"a\": \"line\nbreak\"}",
            r#"{"a": .5}"#,
        ] {
            assert!(!check(text).0, "accepted {text}");
        }
    }

    #[test]
    fn whitespace_is_bounded() {
        assert!(!check(&format!("{{{}", " ".repeat(40))).0);
    }
}
