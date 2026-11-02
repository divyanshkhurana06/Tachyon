//! Property tests on hostile input, with a small deterministic generator
//! instead of a fuzzing crate: the parsers must never panic, and two
//! independent JSON implementations (the parser and the JSON-mode
//! automaton) must agree.

use std::fs;
use std::path::PathBuf;

use tachyon::constrain::{Constraint, JsonObject};
use tachyon::json::Json;
use tachyon::safetensors::SafeTensors;
use tachyon::tokenizer::pre_tokenize;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len())]
    }
}

/// A random JSON value, nested up to `depth`.
fn value(rng: &mut Rng, depth: usize) -> Json {
    let kind = if depth == 0 {
        rng.below(4)
    } else {
        rng.below(6)
    };
    match kind {
        0 => Json::Null,
        1 => Json::Bool(rng.below(2) == 1),
        2 => Json::Num(*rng.pick(&[0.0, -1.5, 42.0, 3.25e-7, -1e21, 123456.0])),
        3 => Json::Str(
            rng.pick(&[
                "",
                "a",
                "quote \" and \\ slash",
                "tab\tnew\nline",
                "é😀日本",
                "\u{1}",
            ])
            .to_string(),
        ),
        4 => Json::Arr((0..rng.below(4)).map(|_| value(rng, depth - 1)).collect()),
        _ => object(rng, depth - 1),
    }
}

fn object(rng: &mut Rng, depth: usize) -> Json {
    Json::Obj(
        (0..rng.below(4))
            .map(|i| (format!("k{i}"), value(rng, depth)))
            .collect(),
    )
}

/// Random bytes biased toward JSON punctuation.
fn noise(rng: &mut Rng, len: usize) -> Vec<u8> {
    let alphabet = b"{}[]\":,0123456789.eE+-tfnrul \n\\abc\x00\xff";
    (0..len)
        .map(|_| {
            if rng.below(8) == 0 {
                rng.next() as u8
            } else {
                *rng.pick(alphabet)
            }
        })
        .collect()
}

#[test]
fn json_parser_round_trips_and_never_panics() {
    let mut rng = Rng(0x1234_5678);
    for _ in 0..2000 {
        let v = value(&mut rng, 4);
        let text = v.to_string();
        assert_eq!(Json::parse(&text).unwrap(), v, "{text}");

        // Mutate it: drop, duplicate or replace a byte.
        let mut bytes = text.into_bytes();
        let i = rng.below(bytes.len().max(1));
        match rng.below(3) {
            0 if !bytes.is_empty() => {
                bytes.remove(i);
            }
            1 if !bytes.is_empty() => bytes.insert(i, bytes[i]),
            _ => bytes.insert(i.min(bytes.len()), *rng.pick(b"{}[],:\"\\x")),
        }
        if let Ok(s) = String::from_utf8(bytes) {
            let _ = Json::parse(&s);
        }
    }
    for len in 0..300 {
        if let Ok(s) = String::from_utf8(noise(&mut rng, len)) {
            let _ = Json::parse(&s);
        }
    }
}

#[test]
fn json_mode_accepts_exactly_what_parses() {
    let mut rng = Rng(0x9E37_79B9);
    // Every object the generator makes is accepted and completes.
    for _ in 0..1000 {
        let text = object(&mut rng, 3).to_string();
        let mut c = JsonObject::default();
        c.advance(text.as_bytes());
        assert!(c.is_complete(), "rejected valid JSON {text}");
    }
    // Anything the automaton calls complete must parse.
    for _ in 0..20000 {
        let len = 1 + rng.below(40);
        let mut bytes = b"{".to_vec();
        bytes.extend(noise(&mut rng, len));
        let mut c = JsonObject::default();
        let mut accepted = 0;
        for (i, &b) in bytes.iter().enumerate() {
            let mut probe = c.clone();
            probe.advance(&[b]);
            // `advance` ignores validity; check with a one-byte token instead.
            if !accepts(&c, b) {
                break;
            }
            c = probe;
            accepted = i + 1;
            if c.is_complete() {
                break;
            }
        }
        if c.is_complete() {
            let text = String::from_utf8_lossy(&bytes[..accepted]);
            assert!(
                Json::parse(&text).is_ok(),
                "automaton accepted invalid JSON {text:?}"
            );
        }
    }
}

/// Whether the automaton allows byte `b` next.
fn accepts(c: &JsonObject, b: u8) -> bool {
    // Build a throwaway one-token tokenizer view through the public API.
    c.accepts_bytes(&[b])
}

#[test]
fn pre_tokenizer_chunks_cover_the_input_exactly() {
    let mut rng = Rng(42);
    let pieces = [
        "Hello",
        " world",
        "  ",
        "\n",
        "\r\n",
        "'s",
        "'LL",
        "123",
        "４５",
        "!?",
        " ...",
        "é",
        "日本語",
        "😀",
        "\t",
        "x",
        "_",
        "<|im_end|>",
    ];
    for _ in 0..2000 {
        let text: String = (0..rng.below(20)).map(|_| *rng.pick(&pieces)).collect();
        for digits in [1, 3] {
            let chunks = pre_tokenize(&text, digits);
            assert_eq!(chunks.concat(), text);
            assert!(chunks.iter().all(|c| !c.is_empty()));
        }
    }
}

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tachyon-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn safetensors_file(header: &str, data: &[u8]) -> Vec<u8> {
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(data);
    out
}

#[test]
fn safetensors_reader_rejects_corrupt_files() {
    let dir = temp_dir("st");
    let path = dir.join("model.safetensors");
    let header = r#"{"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"__metadata__":{"format":"pt"}}"#;
    let data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let good = safetensors_file(header, &data);
    fs::write(&path, &good).unwrap();
    let st = SafeTensors::open_dir(&dir).unwrap();
    assert_eq!(st.tensor("w").unwrap().to_f32(), [1.0, 2.0, 3.0, 4.0]);
    assert!(st.tensor("missing").is_err());
    drop(st);

    let bad_headers = [
        r#"{"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,64]}}"#,
        r#"{"w":{"dtype":"F32","shape":[3,2],"data_offsets":[0,16]}}"#,
        r#"{"w":{"dtype":"I64","shape":[2],"data_offsets":[0,16]}}"#,
        r#"{"w":{"dtype":"F32","shape":"x","data_offsets":[0,16]}}"#,
        r#"{"w":{"dtype":"F32","shape":[2,2],"data_offsets":[16,0]}}"#,
        r#"[1,2]"#,
        "not json",
    ];
    for h in bad_headers {
        fs::write(&path, safetensors_file(h, &data)).unwrap();
        assert!(SafeTensors::open_dir(&dir).is_err(), "accepted header {h}");
    }

    // Truncations and random corruption: errors, never panics.
    let mut rng = Rng(7);
    for cut in [0, 3, 8, 20, good.len() - 1] {
        fs::write(&path, &good[..cut]).unwrap();
        assert!(
            SafeTensors::open_dir(&dir).is_err(),
            "accepted a file cut at {cut}"
        );
    }
    for _ in 0..300 {
        let mut bytes = good.clone();
        for _ in 0..1 + rng.below(4) {
            let i = rng.below(bytes.len());
            bytes[i] = rng.next() as u8;
        }
        fs::write(&path, &bytes).unwrap();
        if let Ok(st) = SafeTensors::open_dir(&dir) {
            let _ = st.tensor("w").map(|t| t.to_f32());
        }
    }
    let _ = fs::remove_dir_all(&dir);
}

/// A schema describing `v`'s exact shape.
fn schema_of(v: &Json) -> Json {
    let ty = |t: &str| Json::Obj(vec![("type".into(), Json::Str(t.into()))]);
    match v {
        Json::Null => ty("null"),
        Json::Bool(_) => ty("boolean"),
        Json::Num(n) if n.fract() == 0.0 => ty("integer"),
        Json::Num(_) => ty("number"),
        Json::Str(_) => ty("string"),
        Json::Arr(items) => {
            // Items share a schema only if every element has the same one.
            let schemas: Vec<String> = items.iter().map(|i| schema_of(i).to_string()).collect();
            let items = if schemas.windows(2).all(|w| w[0] == w[1]) {
                items.first().map(schema_of).unwrap_or(Json::Obj(vec![]))
            } else {
                Json::Obj(vec![])
            };
            Json::Obj(vec![
                ("type".into(), Json::Str("array".into())),
                ("items".into(), items),
            ])
        }
        Json::Obj(fields) => Json::Obj(vec![
            ("type".into(), Json::Str("object".into())),
            (
                "properties".into(),
                Json::Obj(
                    fields
                        .iter()
                        .map(|(k, v)| (k.clone(), schema_of(v)))
                        .collect(),
                ),
            ),
        ]),
    }
}

#[test]
fn json_schema_matcher_accepts_its_instances_and_only_valid_json() {
    use tachyon::schema::{JsonSchema, Schema};
    let mut rng = Rng(0xC0FF_EE11);
    for _ in 0..1500 {
        let v = object(&mut rng, 3);
        let m = JsonSchema::new(Schema::parse(&schema_of(&v)).unwrap());
        let compact = v.to_string();
        // Our strings contain no ',' or ':', so this only adds whitespace.
        let spaced = compact.replace(',', ", ").replace(':', ": ");
        for text in [&compact, &spaced] {
            let mut c = m.clone();
            assert!(c.accepts_bytes(text.as_bytes()), "rejected {text}");
            c.advance(text.as_bytes());
            assert!(c.is_complete(), "incomplete {text}");
        }
    }
    // Anything an any-JSON schema calls complete must parse.
    let any = JsonSchema::new(Schema::parse(&Json::parse("{}").unwrap()).unwrap());
    for _ in 0..20000 {
        let len = 1 + rng.below(30);
        let bytes = noise(&mut rng, len);
        let mut c = any.clone();
        let mut accepted = 0;
        for (i, &b) in bytes.iter().enumerate() {
            if !c.accepts_bytes(&[b]) {
                break;
            }
            c.advance(&[b]);
            accepted = i + 1;
            if c.is_complete() {
                break;
            }
        }
        if c.is_complete() {
            let text = String::from_utf8_lossy(&bytes[..accepted]);
            assert!(
                Json::parse(&text).is_ok(),
                "matcher accepted invalid JSON {text:?}"
            );
        }
    }
}
