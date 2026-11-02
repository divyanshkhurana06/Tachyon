//! Byte-level BPE tokenizer that reads Hugging Face `tokenizer.json` files
//! for GPT-4-style vocabularies (Qwen2, Llama 3).
//!
//! Encoding takes three steps:
//! 1. split out added tokens such as `<|im_start|>` (only when allowed, so
//!    user text can never smuggle in control tokens);
//! 2. split the rest into word-like chunks, following the model's
//!    pre-tokenizer regex, hand-written here instead of using a regex engine;
//! 3. inside each chunk, start from single bytes and repeatedly apply the
//!    lowest-ranked merge until none applies.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::json::Json;
use crate::{bail, Error, Result};

pub struct Tokenizer {
    /// id -> raw bytes (added tokens: their UTF-8 content).
    tokens: Vec<Vec<u8>>,
    special: Vec<bool>,
    byte_ids: [u32; 256],
    /// (left, right) -> (rank, merged id)
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// Added tokens, matched literally before pre-tokenization.
    added: Vec<(String, u32)>,
    /// Llama 3 groups digits in threes; Qwen2 splits every digit.
    max_digits: usize,
    /// Llama 3's `ignore_merges`: a chunk that is itself in the vocabulary
    /// becomes that token, whatever the merges would have made of it.
    whole: Option<HashMap<Vec<u8>, u32>>,
}

impl Tokenizer {
    pub fn load(path: &Path) -> Result<Tokenizer> {
        let text = fs::read_to_string(path)
            .map_err(|e| Error(format!("cannot read {}: {e}", path.display())))?;
        let j = Json::parse(&text)?;
        let model = j
            .get("model")
            .ok_or_else(|| Error::msg("tokenizer.json: no model"))?;
        if model.get("type").and_then(Json::as_str) != Some("BPE") {
            bail!("only BPE tokenizers are supported");
        }
        let max_digits = if text.contains(r"\\p{N}{1,3}") { 3 } else { 1 };
        let ignore_merges = model
            .get("ignore_merges")
            .and_then(Json::as_bool)
            .unwrap_or(false);

        let unbyte = byte_decoder();
        let to_bytes = |s: &str| {
            s.chars()
                .map(|c| unbyte.get(&c).copied())
                .collect::<Option<Vec<u8>>>()
        };

        let vocab = model
            .get("vocab")
            .and_then(Json::as_object)
            .ok_or_else(|| Error::msg("tokenizer.json: no vocab"))?;
        let added_json = j
            .get("added_tokens")
            .and_then(Json::as_array)
            .unwrap_or(&[]);
        let max_id = vocab
            .iter()
            .filter_map(|(_, id)| id.as_usize())
            .chain(added_json.iter().filter_map(|t| t.get("id")?.as_usize()))
            .max()
            .ok_or_else(|| Error::msg("tokenizer.json: empty vocab"))?;

        let mut tokens = vec![Vec::new(); max_id + 1];
        let mut by_bytes: HashMap<Vec<u8>, u32> = HashMap::with_capacity(vocab.len());
        for (s, id) in vocab {
            let id = id
                .as_usize()
                .ok_or_else(|| Error::msg("tokenizer.json: bad id"))?;
            let bytes =
                to_bytes(s).ok_or_else(|| Error(format!("token {s:?} is not byte-level")))?;
            by_bytes.insert(bytes.clone(), id as u32);
            tokens[id] = bytes;
        }

        let mut byte_ids = [0u32; 256];
        for b in 0..=255u8 {
            byte_ids[b as usize] = *by_bytes
                .get(&vec![b])
                .ok_or_else(|| Error(format!("vocab lacks byte {b}")))?;
        }

        let merge_list = model
            .get("merges")
            .and_then(Json::as_array)
            .ok_or_else(|| Error::msg("tokenizer.json: no merges"))?;
        let mut merges = HashMap::with_capacity(merge_list.len());
        for (rank, m) in merge_list.iter().enumerate() {
            // Older files store "a b", newer ones ["a", "b"].
            let pair = match m {
                Json::Str(s) => s.split_once(' '),
                Json::Arr(a) => a
                    .first()
                    .and_then(Json::as_str)
                    .zip(a.get(1).and_then(Json::as_str)),
                _ => None,
            };
            let (a, b) = pair.ok_or_else(|| Error(format!("bad merge {m}")))?;
            let (Some(a), Some(b)) = (to_bytes(a), to_bytes(b)) else {
                bail!("bad merge {m}")
            };
            let merged = [a.as_slice(), b.as_slice()].concat();
            if let (Some(&ia), Some(&ib), Some(&im)) =
                (by_bytes.get(&a), by_bytes.get(&b), by_bytes.get(&merged))
            {
                merges.entry((ia, ib)).or_insert((rank as u32, im));
            }
        }

        let mut special = vec![false; tokens.len()];
        let mut added = Vec::new();
        for t in added_json {
            let (Some(content), Some(id)) = (
                t.get("content").and_then(Json::as_str),
                t.get("id").and_then(Json::as_usize),
            ) else {
                continue;
            };
            tokens[id] = content.as_bytes().to_vec();
            special[id] = t.get("special").and_then(Json::as_bool).unwrap_or(false);
            added.push((content.to_string(), id as u32));
        }
        // Longest first, so a token that is a prefix of another loses ties.
        added.sort_by_key(|(s, _)| std::cmp::Reverse(s.len()));

        Ok(Tokenizer {
            tokens,
            special,
            byte_ids,
            merges,
            added,
            max_digits,
            whole: ignore_merges.then_some(by_bytes),
        })
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Id of an added token such as `<|im_end|>`.
    pub fn added_token(&self, content: &str) -> Option<u32> {
        self.added
            .iter()
            .find(|(s, _)| s == content)
            .map(|(_, id)| *id)
    }

    pub fn is_special(&self, id: u32) -> bool {
        self.special.get(id as usize).copied().unwrap_or(false)
    }

    pub fn token_bytes(&self, id: u32) -> &[u8] {
        self.tokens.get(id as usize).map_or(&[], Vec::as_slice)
    }

    /// `allow_added`: treat special added-token text (control tokens such
    /// as `<|im_start|>`) as the token itself. Leave it off for untrusted
    /// text. Added tokens that aren't special (Qwen's `<tool_call>`) are
    /// always matched, as Hugging Face's tokenizers and llama.cpp do.
    pub fn encode(&self, text: &str, allow_added: bool) -> Vec<u32> {
        let mut out = Vec::new();
        let mut rest = text;
        loop {
            let found = self.find_added(rest, allow_added);
            let plain = found.map_or(rest, |(at, _, _)| &rest[..at]);
            for chunk in pre_tokenize(plain, self.max_digits) {
                self.bpe(chunk.as_bytes(), &mut out);
            }
            match found {
                Some((at, len, id)) => {
                    out.push(id);
                    rest = &rest[at + len..];
                }
                None => return out,
            }
        }
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .filter(|&&id| !(skip_special && self.is_special(id)))
            .flat_map(|&id| self.token_bytes(id).iter().copied())
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Earliest added token in `s` (special ones only if `special_too`):
    /// (byte offset, length, id).
    fn find_added(&self, s: &str, special_too: bool) -> Option<(usize, usize, u32)> {
        let mut best: Option<(usize, usize, u32)> = None;
        for (content, id) in &self.added {
            if !special_too && self.is_special(*id) {
                continue;
            }
            if let Some(at) = s.find(content.as_str()) {
                if best.is_none_or(|(b, _, _)| at < b) {
                    best = Some((at, content.len(), *id));
                }
            }
        }
        best
    }

    fn bpe(&self, bytes: &[u8], out: &mut Vec<u32>) {
        if let Some(&id) = self.whole.as_ref().and_then(|w| w.get(bytes)) {
            out.push(id);
            return;
        }
        let mut ids: Vec<u32> = bytes.iter().map(|&b| self.byte_ids[b as usize]).collect();
        while ids.len() > 1 {
            let mut best: Option<(u32, usize, u32)> = None;
            for p in 0..ids.len() - 1 {
                if let Some(&(rank, merged)) = self.merges.get(&(ids[p], ids[p + 1])) {
                    if best.is_none_or(|(r, _, _)| rank < r) {
                        best = Some((rank, p, merged));
                    }
                }
            }
            let Some((_, p, merged)) = best else { break };
            ids[p] = merged;
            ids.remove(p + 1);
        }
        out.extend(ids);
    }
}

/// GPT-2's byte-to-character table: printable bytes map to themselves, the
/// rest to code points from U+0100 up, so every token is a visible string.
fn byte_encoder() -> [char; 256] {
    let mut map = ['\0'; 256];
    let mut n = 0;
    for b in 0..256u32 {
        let printable =
            (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
        let code = if printable {
            b
        } else {
            n += 1;
            255 + n
        };
        map[b as usize] = char::from_u32(code).unwrap();
    }
    map
}

fn byte_decoder() -> HashMap<char, u8> {
    byte_encoder()
        .iter()
        .enumerate()
        .map(|(b, &c)| (c, b as u8))
        .collect()
}

fn is_letter(c: char) -> bool {
    // Close to \p{L}; differs only for combining marks some scripts use.
    c.is_alphabetic()
}

fn is_number(c: char) -> bool {
    c.is_numeric() // exactly \p{N}
}

fn is_newline(c: char) -> bool {
    c == '\r' || c == '\n'
}

fn is_other(c: char) -> bool {
    !c.is_whitespace() && !is_letter(c) && !is_number(c)
}

/// Splits text the way this pattern does (leftmost alternative wins):
///
/// ```text
/// (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}{1,max_digits}
///   | ` ?[^\s\p{L}\p{N}]+[\r\n]*` | \s*[\r\n]+ | \s+(?!\S) | \s+
/// ```
pub fn pre_tokenize(text: &str, max_digits: usize) -> Vec<&str> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let offset = |i: usize| chars.get(i).map_or(text.len(), |&(o, _)| o);
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let end = match_at(&chars, i, max_digits);
        out.push(&text[offset(i)..offset(end)]);
        i = end;
    }
    out
}

fn match_at(c: &[(usize, char)], i: usize, max_digits: usize) -> usize {
    let at = |j: usize| c.get(j).map(|&(_, ch)| ch);
    let run = |mut j: usize, pred: fn(char) -> bool| {
        while at(j).is_some_and(pred) {
            j += 1;
        }
        j
    };
    let c0 = at(i).unwrap();

    // 's 't 're 've 'm 'll 'd, any case
    if c0 == '\'' {
        if let Some(a) = at(i + 1).map(|a| a.to_ascii_lowercase()) {
            if matches!(a, 's' | 't' | 'm' | 'd') {
                return i + 2;
            }
            let b = at(i + 2).map(|b| b.to_ascii_lowercase());
            if matches!(
                (a, b),
                ('r', Some('e')) | ('v', Some('e')) | ('l', Some('l'))
            ) {
                return i + 3;
            }
        }
    }
    // a word, optionally led by one non-letter, non-digit, non-newline char
    if is_letter(c0) {
        return run(i, is_letter);
    }
    if !is_newline(c0) && !is_number(c0) && at(i + 1).is_some_and(is_letter) {
        return run(i + 1, is_letter);
    }
    // digits
    if is_number(c0) {
        let mut j = i + 1;
        while j - i < max_digits && at(j).is_some_and(is_number) {
            j += 1;
        }
        return j;
    }
    // punctuation, optionally led by a space, swallowing trailing newlines
    let start = if c0 == ' ' && at(i + 1).is_some_and(is_other) {
        i + 1
    } else {
        i
    };
    if at(start).is_some_and(is_other) {
        return run(run(start, is_other), is_newline);
    }
    // whitespace
    let end = run(i, char::is_whitespace);
    if let Some(last_newline) = (i..end).rev().find(|&j| is_newline(c[j].1)) {
        return last_newline + 1; // \s*[\r\n]+
    }
    if end == c.len() || end - i == 1 {
        return end; // \s+ at the end, or a lone space before a word
    }
    end - 1 // \s+(?!\S): leave the last space to lead the next word
}

/// Turns a stream of token bytes into text without splitting a multi-byte
/// character that spans two tokens.
#[derive(Default)]
pub struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    out.push_str(s);
                    self.pending.clear();
                    return out;
                }
                Err(e) => {
                    let good = e.valid_up_to();
                    out.push_str(std::str::from_utf8(&self.pending[..good]).unwrap());
                    match e.error_len() {
                        Some(bad) => {
                            out.push(char::REPLACEMENT_CHARACTER);
                            self.pending.drain(..good + bad);
                        }
                        None => {
                            self.pending.drain(..good); // incomplete: wait for more
                            return out;
                        }
                    }
                }
            }
        }
    }

    pub fn finish(&mut self) -> String {
        let out = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_like_the_regex() {
        let cases: &[(&str, &[&str])] = &[
            ("Hello world", &["Hello", " world"]),
            (
                "I'm here, it's 2024!",
                &[
                    "I", "'m", " here", ",", " it", "'s", " ", "2", "0", "2", "4", "!",
                ],
            ),
            ("a  b", &["a", " ", " b"]),
            ("x\n\n  y", &["x", "\n\n", " ", " y"]),
            ("end   ", &["end", "   "]),
            ("(hello) ...ok", &["(hello", ")", " ...", "ok"]),
            ("fn main() {\n", &["fn", " main", "()", " {\n"]),
            ("café über", &["café", " über"]),
        ];
        for (text, expected) in cases {
            assert_eq!(&pre_tokenize(text, 1), expected, "for {text:?}");
        }
        assert_eq!(pre_tokenize("12345", 3), ["123", "45"]);
    }

    #[test]
    fn utf8_stream_joins_split_characters() {
        let mut s = Utf8Stream::default();
        let bytes = "é😀".as_bytes();
        let mut out = String::new();
        for b in bytes {
            out += &s.push(std::slice::from_ref(b));
        }
        assert_eq!(out, "é😀");
    }

    #[test]
    fn byte_table_is_a_bijection() {
        let enc = byte_encoder();
        let dec = byte_decoder();
        assert_eq!(dec.len(), 256);
        assert_eq!(enc[b' ' as usize], 'Ġ');
        assert_eq!(enc[b'\n' as usize], 'Ċ');
        assert_eq!(enc[b'A' as usize], 'A');
    }
}
