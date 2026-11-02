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
//! `enum`, `const`, `anyOf`/`oneOf`, and `$ref` into `$defs`. Alternatives
//! in a union are told apart by their first byte, so "string or null"
//! works but "integer or number" commits to the first.

use std::sync::Arc;

use crate::constrain::Constraint;
use crate::json::Json;
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
    String,
    Number {
        integer: bool,
    },
    /// One of these exact JSON texts (enum, const, booleans, null).
    Literals(Vec<Vec<u8>>),
    Union(Vec<usize>),
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
            "string" => Node::String,
            "integer" => Node::Number { integer: true },
            "number" => Node::Number { integer: false },
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
    },
    Num {
        state: Num,
        len: u8,
        integer: bool,
    },
    Lit {
        options: Vec<Vec<u8>>,
        pos: usize,
    },
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
                }),
                b'-' | b'0'..=b'9' => self.start_number(false, b),
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
            Node::String => {
                b == b'"'
                    && self.push(Frame::Str {
                        escape: false,
                        hex: 0,
                    })
            }
            Node::Number { integer } => {
                let integer = *integer;
                matches!(b, b'-' | b'0'..=b'9') && self.start_number(integer, b)
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

    fn start_number(&mut self, integer: bool, b: u8) -> bool {
        let state = match b {
            b'-' => Num::Minus,
            b'0' => Num::Zero,
            _ => Num::Int,
        };
        self.push(Frame::Num {
            state,
            len: 1,
            integer,
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
                Frame::Str { escape, hex } => {
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
                } => {
                    let digit = b.is_ascii_digit();
                    let next = match (*state, b) {
                        (Num::Minus, b'0') => Some(Num::Zero),
                        (Num::Minus, _) if digit => Some(Num::Int),
                        (Num::Zero | Num::Int, b'.') if !*integer => Some(Num::Dot),
                        (Num::Int, _) if digit => Some(Num::Int),
                        (Num::Dot | Num::Frac, _) if digit => Some(Num::Frac),
                        (Num::Zero | Num::Int | Num::Frac, b'e' | b'E') if !*integer => {
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
                        return *len <= MAX_NUMBER;
                    }
                    if !matches!(state, Num::Zero | Num::Int | Num::Frac | Num::ExpDigits) {
                        return false;
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
}
