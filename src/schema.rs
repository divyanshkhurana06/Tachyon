//! JSON Schema constrained decoding, as in OpenAI's structured outputs
//! (`response_format: {"type": "json_schema", ...}`).
//!
//! The schema is compiled into a small tree of nodes, and a byte-level
//! pushdown matcher walks it: a stack of frames for the objects, arrays,
//! strings, numbers and literals being written. The sampler only picks
//! tokens whose bytes the matcher accepts, so the output both parses and
//! has the requested shape.
//!
//! Supported: `type` (object, array, string, integer, number, boolean,
//! null, or a list of them), `properties` (written in schema order; those
//! not listed in `required` may be left out), `items`, `minItems`, `maxItems`,
//! `enum`, `const`, `anyOf`/`oneOf`, and `$ref` into `$defs`; for strings
//! `minLength`, `maxLength`, `pattern` (matching the whole string, see
//! `regex.rs`) and the formats date, time, date-time, uuid, email and
//! ipv4; for numbers `minimum`, `maximum` and their exclusive forms
//! (bounded numbers are written without an exponent). Alternatives in a
//! union are told apart by their first byte, so "string or null" works but
//! "integer or number" commits to the first.

use std::sync::Arc;

use crate::constrain::Constraint;
use crate::json::Json;
use crate::regex::{format_pattern, Live, Regex};
use crate::tokenizer::Tokenizer;
use crate::{bail, Error, Result};

#[derive(Debug, Clone)]
enum Node {
    /// Any JSON value.
    Any,
    /// An object with any keys and any values.
    AnyObject,
    /// Fields in order: key (JSON-encoded, with quotes), value node, and
    /// whether it is required.
    Object(Vec<(Vec<u8>, usize, bool)>),
    Array {
        items: usize,
        min: usize,
        max: Option<usize>,
    },
    String {
        /// Length limits, in characters.
        min: usize,
        max: Option<usize>,
        pattern: Option<Arc<Regex>>,
    },
    Number {
        integer: bool,
        bounds: Option<Bounds>,
    },
    /// One of these exact JSON texts (enum, const, booleans, null).
    Literals(Vec<Vec<u8>>),
    Union(Vec<usize>),
}

/// A number's allowed range; an exclusive end is not itself allowed.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Bounds {
    lo: f64,
    hi: f64,
    lo_open: bool,
    hi_open: bool,
}

impl Bounds {
    /// `minimum`, `maximum` and `exclusiveMinimum`/`exclusiveMaximum`, as
    /// numbers (draft 6 on) or as flags on the others (draft 4).
    fn of(s: &Json) -> Result<Option<Bounds>> {
        let num = |k: &str| s.get(k).and_then(Json::as_f64);
        let mut b = Bounds {
            lo: num("minimum").unwrap_or(f64::NEG_INFINITY),
            hi: num("maximum").unwrap_or(f64::INFINITY),
            lo_open: false,
            hi_open: false,
        };
        match s.get("exclusiveMinimum") {
            Some(Json::Bool(open)) => b.lo_open = *open,
            Some(Json::Num(x)) if *x >= b.lo => (b.lo, b.lo_open) = (*x, true),
            _ => {}
        }
        match s.get("exclusiveMaximum") {
            Some(Json::Bool(open)) => b.hi_open = *open,
            Some(Json::Num(x)) if *x <= b.hi => (b.hi, b.hi_open) = (*x, true),
            _ => {}
        }
        if b.lo > b.hi || (b.lo == b.hi && (b.lo_open || b.hi_open)) {
            bail!("the number range is empty");
        }
        Ok((b.lo.is_finite() || b.hi.is_finite()).then_some(b))
    }

    fn contains(&self, v: f64) -> bool {
        (if self.lo_open {
            v > self.lo
        } else {
            v >= self.lo
        }) && (if self.hi_open {
            v < self.hi
        } else {
            v <= self.hi
        })
    }

    /// Whether a number written so far as `text` (in state `state`) can
    /// still end inside the range: the magnitudes its completions can take
    /// are intervals, checked against the range (taking ends as closed).
    fn reachable(&self, text: &[u8], state: Num, integer: bool) -> bool {
        let neg = text.first() == Some(&b'-');
        let (lo, hi) = if neg {
            ((-self.hi).max(0.0), -self.lo)
        } else {
            (self.lo.max(0.0), self.hi)
        };
        if lo > hi {
            return false;
        }
        let digits = std::str::from_utf8(&text[neg as usize..]).unwrap_or("");
        let (int, frac) = digits.split_once('.').unwrap_or((digits, ""));
        let d: f64 = int.parse().unwrap_or(0.0);
        // Completions span [a, z): open at z unless integer.
        let meets = |a: f64, z: f64| a <= hi && (z > lo || a >= lo);
        match state {
            Num::Minus => true,
            Num::Zero if integer => lo <= 0.0,
            Num::Zero => meets(0.0, 1.0),
            Num::Int => {
                // More integer digits: [d * 10^j, (d + 1) * 10^j) for each j.
                let mut p = 1.0;
                while d * p <= hi && p < 1e40 {
                    let z = (d + 1.0) * p;
                    if if integer { z - 1.0 >= lo } else { z > lo } {
                        return true;
                    }
                    p *= 10.0;
                }
                false
            }
            Num::Dot => meets(d, d + 1.0),
            Num::Frac => {
                let v: f64 = format!("{int}.{frac}").parse().unwrap_or(0.0);
                meets(v, v + 10f64.powi(-(frac.len() as i32)))
            }
            _ => true,
        }
    }
}

/// Node 0 is always `Any` and node 1 an array of anything.
const ANY: usize = 0;
const ANY_ARRAY: usize = 1;

#[derive(Debug)]
pub struct Schema {
    nodes: Vec<Node>,
    root: usize,
}

impl Schema {
    pub fn parse(schema: &Json) -> Result<Schema> {
        let mut b = Builder {
            nodes: vec![
                Node::Any,
                Node::Array {
                    items: ANY,
                    min: 0,
                    max: None,
                },
            ],
            root: schema,
        };
        let root = b.node(schema, 0)?;
        Ok(Schema {
            nodes: b.nodes,
            root,
        })
    }
}

struct Builder<'a> {
    nodes: Vec<Node>,
    root: &'a Json,
}

fn resolve<'a>(root: &'a Json, reference: &str) -> Result<&'a Json> {
    let path = reference
        .strip_prefix("#/")
        .ok_or_else(|| Error(format!("only local $refs are supported, not {reference}")))?;
    let mut at = root;
    for part in path.split('/') {
        at = at
            .get(part)
            .ok_or_else(|| Error(format!("$ref {reference} not found")))?;
    }
    Ok(at)
}

impl Builder<'_> {
    fn push(&mut self, n: Node) -> usize {
        self.nodes.push(n);
        self.nodes.len() - 1
    }

    fn node(&mut self, s: &Json, depth: usize) -> Result<usize> {
        if depth > 32 {
            bail!("schema nests too deeply (a recursive $ref?)");
        }
        if let Some(r) = s.get("$ref").and_then(Json::as_str) {
            let target = resolve(self.root, r)?;
            return self.node(target, depth + 1);
        }
        if let Some(v) = s.get("const") {
            return Ok(self.push(Node::Literals(vec![v.to_string().into_bytes()])));
        }
        if let Some(values) = s.get("enum").and_then(Json::as_array) {
            if values.is_empty() {
                bail!("enum must not be empty");
            }
            return Ok(self.push(Node::Literals(
                values.iter().map(|v| v.to_string().into_bytes()).collect(),
            )));
        }
        for key in ["anyOf", "oneOf"] {
            if let Some(alts) = s.get(key).and_then(Json::as_array) {
                let ids = alts
                    .iter()
                    .map(|a| self.node(a, depth + 1))
                    .collect::<Result<Vec<_>>>()?;
                return Ok(self.push(Node::Union(ids)));
            }
        }
        match s.get("type") {
            Some(Json::Str(t)) => self.typed(s, t, depth),
            Some(Json::Arr(types)) => {
                let ids = types
                    .iter()
                    .map(|t| {
                        self.typed(
                            s,
                            t.as_str()
                                .ok_or_else(|| Error::msg("type names must be strings"))?,
                            depth,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(self.push(Node::Union(ids)))
            }
            None if s.get("properties").is_some() => self.typed(s, "object", depth),
            None => Ok(ANY),
            Some(other) => bail!("bad type {other}"),
        }
    }

    fn typed(&mut self, s: &Json, t: &str, depth: usize) -> Result<usize> {
        let node = match t {
            "object" => match s.get("properties").and_then(Json::as_object) {
                Some(props) if !props.is_empty() => {
                    let required: Vec<&str> = s
                        .get("required")
                        .and_then(Json::as_array)
                        .unwrap_or(&[])
                        .iter()
                        .filter_map(Json::as_str)
                        .collect();
                    let mut fields = Vec::with_capacity(props.len());
                    for (key, value) in props {
                        let id = self.node(value, depth + 1)?;
                        let key_json = Json::Str(key.clone()).to_string().into_bytes();
                        fields.push((key_json, id, required.contains(&key.as_str())));
                    }
                    Node::Object(fields)
                }
                _ => Node::AnyObject,
            },
            "array" => {
                let items = match s.get("items") {
                    Some(i) => self.node(i, depth + 1)?,
                    None => ANY,
                };
                let min = s.get("minItems").and_then(Json::as_usize).unwrap_or(0);
                let max = s.get("maxItems").and_then(Json::as_usize);
                if max.is_some_and(|m| m < min) {
                    bail!("maxItems is below minItems");
                }
                Node::Array { items, min, max }
            }
            "string" => {
                let min = s.get("minLength").and_then(Json::as_usize).unwrap_or(0);
                let max = s.get("maxLength").and_then(Json::as_usize);
                if max.is_some_and(|m| m < min) {
                    bail!("maxLength is below minLength");
                }
                let pattern = match s.get("pattern").and_then(Json::as_str) {
                    Some(p) => Some(p),
                    None => s
                        .get("format")
                        .and_then(Json::as_str)
                        .and_then(format_pattern),
                };
                Node::String {
                    min,
                    max,
                    pattern: pattern.map(Regex::new).transpose()?.map(Arc::new),
                }
            }
            "integer" | "number" => Node::Number {
                integer: t == "integer",
                bounds: Bounds::of(s)?,
            },
            "boolean" => Node::Literals(vec![b"true".to_vec(), b"false".to_vec()]),
            "null" => Node::Literals(vec![b"null".to_vec()]),
            other => bail!("unsupported type {other:?}"),
        };
        Ok(self.push(node))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
    Open,
    Key,
    /// Inside a key, `pos` bytes in (objects with a schema only).
    InKey,
    Colon,
    After,
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

#[derive(Debug, Clone)]
enum Frame {
    /// A value of this node starts here.
    Value(usize),
    Object {
        node: usize,
        /// The first field that may still be written.
        next: usize,
        stage: Stage,
        /// While a key is written: the fields it may still be, and how
        /// many of its bytes have been seen.
        keys: Vec<usize>,
        pos: usize,
    },
    AnyObject {
        stage: Stage,
    },
    Array {
        node: usize,
        count: usize,
        stage: Stage,
    },
    Str {
        escape: bool,
        hex: u8,
        /// For strings with a length or pattern to keep to.
        check: Option<Box<StrCheck>>,
    },
    Num {
        state: Num,
        len: u8,
        integer: bool,
        /// A bounded number, and its text so far.
        bounds: Option<Bounds>,
        text: Vec<u8>,
    },
    Lit {
        options: Vec<Vec<u8>>,
        pos: usize,
    },
}

/// A checked string's progress: characters so far, the pattern's live
/// states, and a character being assembled from UTF-8 bytes or a `\u`
/// escape.
#[derive(Debug, Clone)]
struct StrCheck {
    node: usize,
    chars: usize,
    live: Live,
    code: u32,
    utf8: u8,
}

const MAX_WHITESPACE: u8 = 16;
const MAX_NUMBER: u8 = 40;
const MAX_DEPTH: usize = 64;

/// The matcher: a [`Constraint`] that admits exactly the JSON texts the
/// schema describes.
#[derive(Debug, Clone)]
pub struct JsonSchema {
    schema: Arc<Schema>,
    stack: Vec<Frame>,
    whitespace: u8,
}

impl JsonSchema {
    pub fn new(schema: Schema) -> JsonSchema {
        let root = schema.root;
        JsonSchema {
            schema: Arc::new(schema),
            stack: vec![Frame::Value(root)],
            whitespace: 0,
        }
    }

    /// Whether these bytes may come next.
    pub fn accepts_bytes(&self, bytes: &[u8]) -> bool {
        let mut probe = self.clone();
        bytes.iter().all(|&b| probe.step(b))
    }

    fn push(&mut self, f: Frame) -> bool {
        if self.stack.len() >= MAX_DEPTH {
            return false;
        }
        self.stack.push(f);
        true
    }

    /// Begins a value of `node` with byte `b`.
    fn start(&mut self, node: usize, b: u8) -> bool {
        let n = &self.schema.nodes[node];
        match n {
            Node::Any => match b {
                b'{' => self.push(Frame::AnyObject { stage: Stage::Open }),
                b'[' => self.push(Frame::Array {
                    node: ANY_ARRAY,
                    count: 0,
                    stage: Stage::Open,
                }),
                b'"' => self.push(Frame::Str {
                    escape: false,
                    hex: 0,
                    check: None,
                }),
                b'-' | b'0'..=b'9' => self.start_number(false, None, b),
                _ => self.start_literals(
                    vec![b"true".to_vec(), b"false".to_vec(), b"null".to_vec()],
                    b,
                ),
            },
            Node::AnyObject => b == b'{' && self.push(Frame::AnyObject { stage: Stage::Open }),
            Node::Object(_) => {
                b == b'{'
                    && self.push(Frame::Object {
                        node,
                        next: 0,
                        stage: Stage::Open,
                        keys: Vec::new(),
                        pos: 0,
                    })
            }
            Node::Array { .. } => {
                b == b'['
                    && self.push(Frame::Array {
                        node,
                        count: 0,
                        stage: Stage::Open,
                    })
            }
            Node::String { min, max, pattern } => {
                let checked = *min > 0 || max.is_some() || pattern.is_some();
                let check = checked.then(|| {
                    Box::new(StrCheck {
                        node,
                        chars: 0,
                        live: pattern.as_ref().map(|r| r.start()).unwrap_or_default(),
                        code: 0,
                        utf8: 0,
                    })
                });
                b == b'"'
                    && self.push(Frame::Str {
                        escape: false,
                        hex: 0,
                        check,
                    })
            }
            Node::Number { integer, bounds } => {
                let (integer, bounds) = (*integer, *bounds);
                matches!(b, b'-' | b'0'..=b'9') && self.start_number(integer, bounds, b)
            }
            Node::Literals(options) => {
                let options = options.clone();
                self.start_literals(options, b)
            }
            Node::Union(alts) => {
                for &alt in alts.clone().iter() {
                    let mut probe = self.clone();
                    if probe.start(alt, b) {
                        *self = probe;
                        return true;
                    }
                }
                false
            }
        }
    }

    fn start_number(&mut self, integer: bool, bounds: Option<Bounds>, b: u8) -> bool {
        let state = match b {
            b'-' => Num::Minus,
            b'0' => Num::Zero,
            _ => Num::Int,
        };
        let text = if bounds.is_some() {
            vec![b]
        } else {
            Vec::new()
        };
        if bounds.is_some_and(|r| !r.reachable(&text, state, integer)) {
            return false;
        }
        self.push(Frame::Num {
            state,
            len: 1,
            integer,
            bounds,
            text,
        })
    }

    fn start_literals(&mut self, options: Vec<Vec<u8>>, b: u8) -> bool {
        self.push(Frame::Lit { options, pos: 0 }) && self.step(b)
    }

    /// The frame on top finished; the byte that ended it, if any, belongs
    /// to the parent.
    fn pop(&mut self) {
        self.stack.pop();
    }

    /// Feeds one byte; false if it cannot appear here.
    fn step(&mut self, b: u8) -> bool {
        let ws = matches!(b, b' ' | b'\t' | b'\n' | b'\r');
        let schema = self.schema.clone();
        loop {
            let Some(top) = self.stack.last_mut() else {
                return false;
            };
            match top {
                Frame::Object {
                    node,
                    next,
                    stage: stage @ Stage::InKey,
                    keys,
                    pos,
                } => {
                    // Keys are whole JSON strings, so none is a prefix of
                    // another: the first to end is the one written.
                    let Node::Object(fields) = &schema.nodes[*node] else {
                        unreachable!()
                    };
                    keys.retain(|&f| fields[f].0.get(*pos) == Some(&b));
                    *pos += 1;
                    if let Some(&f) = keys.iter().find(|&&f| fields[f].0.len() == *pos) {
                        *next = f + 1;
                        *stage = Stage::Colon;
                    }
                    return !keys.is_empty();
                }
                Frame::Str {
                    escape,
                    hex,
                    check: Some(ck),
                } => {
                    let Node::String { min, max, pattern } = &schema.nodes[ck.node] else {
                        unreachable!()
                    };
                    let room = max.is_none_or(|m| ck.chars < m);
                    // One more character: within the length, and the
                    // pattern still alive.
                    let feed = |ck: &mut StrCheck, c: Option<char>| {
                        let Some(c) = c else { return false };
                        ck.chars += 1;
                        if max.is_some_and(|m| ck.chars > m) {
                            return false;
                        }
                        match pattern {
                            Some(r) => {
                                ck.live = r.step(&ck.live, c);
                                !ck.live.is_empty()
                            }
                            None => true,
                        }
                    };
                    if ck.utf8 > 0 {
                        if b & 0xC0 != 0x80 {
                            return false;
                        }
                        ck.code = ck.code << 6 | (b & 0x3F) as u32;
                        ck.utf8 -= 1;
                        return ck.utf8 > 0 || feed(ck, char::from_u32(ck.code));
                    }
                    if *hex > 0 {
                        let Some(d) = (b as char).to_digit(16) else {
                            return false;
                        };
                        ck.code = ck.code * 16 + d;
                        *hex -= 1;
                        return *hex > 0 || feed(ck, char::from_u32(ck.code));
                    }
                    if *escape {
                        *escape = false;
                        let c = match b {
                            b'"' => '"',
                            b'\\' => '\\',
                            b'/' => '/',
                            b'b' => '\x08',
                            b'f' => '\x0c',
                            b'n' => '\n',
                            b'r' => '\r',
                            b't' => '\t',
                            b'u' => {
                                (*hex, ck.code) = (4, 0);
                                return true;
                            }
                            _ => return false,
                        };
                        return feed(ck, Some(c));
                    }
                    return match b {
                        b'"' => {
                            let done = ck.chars >= *min
                                && pattern.as_ref().is_none_or(|r| r.accepts(&ck.live));
                            if done {
                                self.pop();
                            }
                            done
                        }
                        b'\\' => {
                            *escape = true;
                            room
                        }
                        0x20..=0x7F => feed(ck, Some(b as char)),
                        0xC2..=0xF4 => {
                            (ck.utf8, ck.code) = match b {
                                0xF0.. => (3, (b & 0x07) as u32),
                                0xE0.. => (2, (b & 0x0F) as u32),
                                _ => (1, (b & 0x1F) as u32),
                            };
                            room && pattern.as_ref().is_none_or(|r| r.wants_non_ascii(&ck.live))
                        }
                        _ => false,
                    };
                }
                Frame::Str { escape, hex, .. } => {
                    return match (*escape, *hex) {
                        (_, n) if n > 0 => {
                            *hex = n - 1;
                            b.is_ascii_hexdigit()
                        }
                        (true, _) => {
                            *escape = false;
                            *hex = if b == b'u' { 4 } else { 0 };
                            matches!(
                                b,
                                b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' | b'u'
                            )
                        }
                        _ if b == b'"' => {
                            self.pop();
                            true
                        }
                        _ if b == b'\\' => {
                            *escape = true;
                            true
                        }
                        _ => b >= 0x20,
                    };
                }
                Frame::Num {
                    state,
                    len,
                    integer,
                    bounds,
                    text,
                } => {
                    let digit = b.is_ascii_digit();
                    let next = match (*state, b) {
                        (Num::Minus, b'0') => Some(Num::Zero),
                        (Num::Minus, _) if digit => Some(Num::Int),
                        (Num::Zero | Num::Int, b'.') if !*integer => Some(Num::Dot),
                        (Num::Int, _) if digit => Some(Num::Int),
                        (Num::Dot | Num::Frac, _) if digit => Some(Num::Frac),
                        (Num::Zero | Num::Int | Num::Frac, b'e' | b'E')
                            if !*integer && bounds.is_none() =>
                        {
                            Some(Num::Exp)
                        }
                        (Num::Exp, b'+' | b'-') => Some(Num::ExpSign),
                        (Num::Exp | Num::ExpSign | Num::ExpDigits, _) if digit => {
                            Some(Num::ExpDigits)
                        }
                        _ => None,
                    };
                    if let Some(next) = next {
                        *state = next;
                        *len += 1;
                        if let Some(r) = bounds {
                            text.push(b);
                            if !r.reachable(text, next, *integer) {
                                return false;
                            }
                        }
                        return *len <= MAX_NUMBER;
                    }
                    if !matches!(state, Num::Zero | Num::Int | Num::Frac | Num::ExpDigits) {
                        return false;
                    }
                    if let Some(r) = bounds {
                        let v = std::str::from_utf8(text).ok().and_then(|t| t.parse().ok());
                        if !v.is_some_and(|v| r.contains(v)) {
                            return false;
                        }
                    }
                    self.pop(); // the number ended; `b` belongs to the parent
                    continue;
                }
                Frame::Lit { options, pos } => {
                    let p = *pos;
                    let cont: Vec<Vec<u8>> = options
                        .iter()
                        .filter(|o| o.get(p) == Some(&b))
                        .cloned()
                        .collect();
                    if !cont.is_empty() {
                        let done = cont.iter().all(|o| o.len() == p + 1);
                        *options = cont;
                        *pos = p + 1;
                        if done {
                            self.pop();
                        }
                        return true;
                    }
                    if options.iter().any(|o| o.len() == p) {
                        self.pop(); // a shorter option ended here
                        continue;
                    }
                    return false;
                }
                _ => {}
            }

            if ws {
                self.whitespace += 1;
                return self.whitespace <= MAX_WHITESPACE;
            }
            self.whitespace = 0;
            let top = self.stack.last_mut().unwrap();
            return match top {
                Frame::Value(node) => {
                    let node = *node;
                    self.pop();
                    self.start(node, b)
                }
                Frame::Object {
                    node,
                    next,
                    stage,
                    keys,
                    pos,
                } => {
                    let Node::Object(fields) = &schema.nodes[*node] else {
                        unreachable!()
                    };
                    let rest_optional = fields[*next..].iter().all(|f| !f.2);
                    match (*stage, b) {
                        (Stage::Open | Stage::Key, b'"') => {
                            // The next field, or a later one past optional ones.
                            let mut can = Vec::new();
                            for (f, field) in fields.iter().enumerate().skip(*next) {
                                can.push(f);
                                if field.2 {
                                    break;
                                }
                            }
                            *keys = can;
                            *pos = 1;
                            *stage = Stage::InKey;
                            !keys.is_empty()
                        }
                        (Stage::Open | Stage::After, b'}') if rest_optional => {
                            self.pop();
                            true
                        }
                        (Stage::Colon, b':') => {
                            let value = fields[*next - 1].1;
                            *stage = Stage::After;
                            self.push(Frame::Value(value))
                        }
                        (Stage::After, b',') if *next < fields.len() => {
                            *stage = Stage::Key;
                            true
                        }
                        _ => false,
                    }
                }
                Frame::AnyObject { stage } => match (*stage, b) {
                    (Stage::Open, b'}') | (Stage::After, b'}') => {
                        self.pop();
                        true
                    }
                    (Stage::Open | Stage::Key, b'"') => {
                        *stage = Stage::Colon;
                        self.push(Frame::Str {
                            escape: false,
                            hex: 0,
                            check: None,
                        })
                    }
                    (Stage::Colon, b':') => {
                        *stage = Stage::After;
                        self.push(Frame::Value(ANY))
                    }
                    (Stage::After, b',') => {
                        *stage = Stage::Key;
                        true
                    }
                    _ => false,
                },
                Frame::Array { node, count, stage } => {
                    let Node::Array { items, min, max } = schema.nodes[*node] else {
                        unreachable!()
                    };
                    match (*stage, b) {
                        (Stage::Open, b']') if min == 0 => {
                            self.pop();
                            true
                        }
                        (Stage::Open, _) => {
                            *stage = Stage::After;
                            *count = 1;
                            self.push(Frame::Value(items)) && self.step(b)
                        }
                        (Stage::After, b',') if max.is_none_or(|m| *count < m) => {
                            *count += 1;
                            self.push(Frame::Value(items))
                        }
                        (Stage::After, b']') if *count >= min => {
                            self.pop();
                            true
                        }
                        _ => false,
                    }
                }
                _ => unreachable!(),
            };
        }
    }
}

impl Constraint for JsonSchema {
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
        self.stack.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matcher(schema: &str) -> JsonSchema {
        JsonSchema::new(Schema::parse(&Json::parse(schema).unwrap()).unwrap())
    }

    fn check(m: &JsonSchema, text: &str) -> (bool, bool) {
        let mut m = m.clone();
        let ok = m.accepts_bytes(text.as_bytes());
        m.advance(text.as_bytes());
        (ok, m.is_complete())
    }

    const PERSON: &str = r##"{
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "role": {"enum": ["admin", "user"]},
            "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
            "manager": {"type": ["string", "null"]},
            "address": {"$ref": "#/$defs/address"}
        },
        "required": ["name", "age", "role", "tags", "manager", "address"],
        "$defs": {"address": {"type": "object", "properties": {"city": {"type": "string"}, "zip": {"type": "number"}}, "required": ["city", "zip"]}}
    }"##;

    #[test]
    fn accepts_instances_of_the_schema() {
        let m = matcher(PERSON);
        for text in [
            r#"{"name":"Ada","age":36,"role":"admin","tags":[],"manager":null,"address":{"city":"London","zip":1.5e3}}"#,
            "{\n  \"name\": \"B\\\"ob\",\n  \"age\": -7,\n  \"role\": \"user\",\n  \"tags\": [\"a\", \"b\", \"c\"],\n  \"manager\": \"Ada\",\n  \"address\": {\"city\": \"\", \"zip\": 0}\n}",
        ] {
            assert_eq!(check(&m, text), (true, true), "{text}");
        }
    }

    #[test]
    fn rejects_what_the_schema_forbids() {
        let m = matcher(PERSON);
        for text in [
            r#"{"age":1"#,                                                    // wrong first key
            r#"{"name":5"#,                         // name must be a string
            r#"{"name":"A","age":1.5"#,             // age must be an integer
            r#"{"name":"A","age":1,"role":"boss""#, // not in the enum
            r#"{"name":"A","age":1,"role":"user","tags":["a","b","c","d"]"#, // maxItems
            r#"{"name":"A","age":1,"role":"user","tags":[],"manager":true"#, // string or null
            r#"{"name":"A","age":1,"role":"user","tags":[],"manager":null}"#, // missing address
        ] {
            assert!(!check(&m, text).0, "accepted {text}");
        }
    }

    #[test]
    fn optional_properties_may_be_left_out_in_order() {
        let m = matcher(
            r#"{"type":"object","properties":{"a":{"type":"integer"},"ab":{"type":"integer"},"b":{"type":"string"},"c":{"type":"boolean"}},"required":["b"]}"#,
        );
        for text in [
            r#"{"b":"x"}"#,
            r#"{"a":1,"b":"x"}"#,
            r#"{"ab":2,"b":"x","c":true}"#,
            r#"{"a":1,"ab":2,"b":"x","c":false}"#,
            r#"{ "b" : "x" , "c" : true }"#,
        ] {
            assert_eq!(check(&m, text), (true, true), "{text}");
        }
        for text in [
            r#"{}"#,                 // b is required
            r#"{"a":1}"#,            // b is required
            r#"{"b":"x","a":1}"#,    // out of order
            r#"{"c":true}"#,         // can't skip the required b
            r#"{"b":"x","c":true,"#, // nothing may follow c
            r#"{"a":1,"a":2"#,       // no repeats
        ] {
            assert!(!check(&m, text).0 || !check(&m, text).1, "accepted {text}");
        }
        let all_optional = matcher(r#"{"type":"object","properties":{"k v":{"type":"null"}}}"#);
        assert_eq!(check(&all_optional, "{}"), (true, true));
        assert_eq!(check(&all_optional, r#"{"k v":null}"#), (true, true));
    }

    #[test]
    fn prefixes_are_accepted_but_not_complete() {
        let m = matcher(PERSON);
        assert_eq!(check(&m, r#"{"name": "Ad"#), (true, false));
        assert_eq!(check(&m, r#"{"name": "Ada", "age": 3"#), (true, false));
    }

    #[test]
    fn unconstrained_parts_take_any_json() {
        let m = matcher(
            r#"{"type":"object","properties":{"data":{},"list":{"type":"array","minItems":1}}}"#,
        );
        assert_eq!(
            check(&m, r#"{"data":{"x":[1,{"y":null}]},"list":[true,"s",-2]}"#),
            (true, true)
        );
        assert!(!check(&m, r#"{"data":1,"list":[]}"#).0); // minItems
    }

    #[test]
    fn bad_schemas_are_errors() {
        for s in [
            r#"{"type":"date"}"#,
            r##"{"$ref":"#/$defs/missing"}"##,
            r#"{"enum":[]}"#,
        ] {
            assert!(Schema::parse(&Json::parse(s).unwrap()).is_err(), "{s}");
        }
    }

    #[test]
    fn string_lengths_patterns_and_formats() {
        let m = matcher(
            r#"{"type":"object","properties":{
                "code":{"type":"string","pattern":"^[A-Z]{3}-\\d{2}$"},
                "name":{"type":"string","minLength":2,"maxLength":4},
                "day":{"type":"string","format":"date"}},
              "required":["code","name","day"]}"#,
        );
        let ok = r#"{"code":"ABC-12","name":"Ada","day":"2026-11-05"}"#;
        assert_eq!(check(&m, ok), (true, true));
        for bad in [
            r#"{"code":"A1"#,                                  // a digit where a letter must be
            r#"{"code":"abc"#,                                 // lower case
            r#"{"code":"ABC-123"#,                             // too long for the pattern
            r#"{"code":"ABC-1""#,                              // ends early
            r#"{"code":"ABC-12","name":"A""#,                  // shorter than minLength
            r#"{"code":"ABC-12","name":"Grace"#,               // longer than maxLength
            r#"{"code":"ABC-12","name":"Ada","day":"2026-13"#, // month 13
            r#"{"code":"ABC-12","name":"Ada","day":"Nov"#,
        ] {
            assert!(!check(&m, bad).0, "{bad}");
        }
        // Escapes and multi-byte characters count as one character each.
        let m = matcher(r#"{"type":"string","maxLength":3,"pattern":"[^x]*"}"#);
        assert_eq!(check(&m, r#""a\né""#), (true, true));
        assert_eq!(check(&m, r#""é\"b""#), (true, true));
        assert!(!check(&m, r#""abcd"#).0);
        assert!(!check(&m, r#""x"#).0); // an escaped x is still x
                                        // An ASCII-only pattern refuses a multi-byte character at its first byte.
        let m = matcher(r#"{"type":"string","pattern":"[a-z]+"}"#);
        assert!(!m.accepts_bytes("\"\u{e9}".as_bytes()[..2].as_ref()));
    }

    #[test]
    fn number_ranges() {
        let m = matcher(
            r#"{"type":"object","properties":{
                "age":{"type":"integer","minimum":0,"maximum":120},
                "score":{"type":"number","exclusiveMinimum":-1,"maximum":1}},
              "required":["age","score"]}"#,
        );
        for ok in [
            r#"{"age":0,"score":-0.5}"#,
            r#"{"age":120,"score":1}"#,
            r#"{"age":37,"score":0.999}"#,
            r#"{"age":7,"score":-0.0001}"#,
        ] {
            assert_eq!(check(&m, ok), (true, true), "{ok}");
        }
        for bad in [
            r#"{"age":-1"#,            // below the minimum ("-0" is 0, so allowed)
            r#"{"age":121"#,           // above the maximum
            r#"{"age":1000"#,          // can only grow
            r#"{"age":12.5"#,          // an integer
            r#"{"age":5,"score":-1,"#, // the exclusive minimum itself
            r#"{"age":5,"score":2"#,
            r#"{"age":5,"score":1.5"#,
            r#"{"age":5,"score":1e"#, // no exponents when bounded
        ] {
            assert!(!check(&m, bad).0, "{bad}");
        }
        // Prefixes that can still reach the range are allowed.
        let m = matcher(
            r#"{"type":"object","properties":{"n":{"type":"integer","minimum":500,"maximum":600}},"required":["n"]}"#,
        );
        assert!(m.accepts_bytes(br#"{"n":5"#));
        assert!(!m.accepts_bytes(br#"{"n":7"#));
        assert!(!m.accepts_bytes(br#"{"n":5}"#));
        assert!(check(&m, r#"{"n":512}"#).1);
        assert!(Schema::parse(
            &Json::parse(r#"{"type":"integer","minimum":3,"maximum":2}"#).unwrap()
        )
        .is_err());
    }
}
