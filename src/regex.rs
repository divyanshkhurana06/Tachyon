//! A small regular-expression engine, for JSON Schema's `pattern` and
//! `format` in constrained decoding.
//!
//! Patterns compile to a Thompson NFA over characters, and a match is a set
//! of live states advanced one character at a time. That is what decoding
//! needs: after every character the matcher can tell whether the string so
//! far can still match (some state is live) and whether it may end here
//! (the accepting state is live), with no backtracking.
//!
//! Supported: literals, `.`, classes (`[a-z_]`, `[^...]`, `\d \w \s` and
//! their negations), groups (`(...)`, `(?:...)`, named), `|`, and the
//! quantifiers `* + ? {n} {n,} {n,m}` (lazy forms match the same strings).
//! A pattern must match the whole string, as constrained decoders use it;
//! `^` and `$` are accepted at its ends. Backreferences, lookaround and
//! `\b` are not regular and are rejected.

use crate::{bail, Error, Result};

/// Most NFA states a pattern may expand to (`x{1000}` copies `x`).
const MAX_STATES: usize = 20_000;

#[derive(Debug, Clone, PartialEq)]
struct Class {
    ranges: Vec<(char, char)>,
    negated: bool,
}

impl Class {
    fn one(c: char) -> Class {
        Class {
            ranges: vec![(c, c)],
            negated: false,
        }
    }

    fn matches(&self, c: char) -> bool {
        self.ranges.iter().any(|&(a, b)| a <= c && c <= b) != self.negated
    }

    /// Whether it matches any character outside ASCII.
    fn any_non_ascii(&self) -> bool {
        self.negated || self.ranges.iter().any(|&(_, b)| b > '\x7f')
    }
}

#[derive(Debug, Clone)]
enum Ast {
    Empty,
    Class(Class),
    Concat(Vec<Ast>),
    Alt(Vec<Ast>),
    Repeat(Box<Ast>, usize, Option<usize>),
}

#[derive(Debug, Clone)]
enum State {
    /// Consumes one character in the class, then goes on.
    Char(Class, usize),
    /// Goes on to any of these without consuming anything.
    Split(Vec<usize>),
    Match,
}

#[derive(Debug, Clone)]
pub struct Regex {
    states: Vec<State>,
    start: usize,
}

/// Where a match stands: the live states, sorted.
pub type Live = Vec<usize>;

impl Regex {
    pub fn new(pattern: &str) -> Result<Regex> {
        let chars: Vec<char> = pattern.chars().collect();
        let mut p = Parser { s: &chars, i: 0 };
        p.eat('^');
        let ast = p.alternation()?;
        if p.i < chars.len() {
            bail!("pattern {pattern:?}: unexpected {:?}", chars[p.i]);
        }
        let mut r = Regex {
            states: vec![State::Match],
            start: 0,
        };
        r.start = r.compile(&ast, 0)?;
        Ok(r)
    }

    fn push(&mut self, s: State) -> Result<usize> {
        if self.states.len() >= MAX_STATES {
            bail!("pattern is too large");
        }
        self.states.push(s);
        Ok(self.states.len() - 1)
    }

    /// Compiles `ast` to states that continue to `next`; returns its start.
    fn compile(&mut self, ast: &Ast, next: usize) -> Result<usize> {
        Ok(match ast {
            Ast::Empty => next,
            Ast::Class(c) => self.push(State::Char(c.clone(), next))?,
            Ast::Concat(parts) => {
                let mut at = next;
                for part in parts.iter().rev() {
                    at = self.compile(part, at)?;
                }
                at
            }
            Ast::Alt(alts) => {
                let starts = alts
                    .iter()
                    .map(|a| self.compile(a, next))
                    .collect::<Result<Vec<_>>>()?;
                self.push(State::Split(starts))?
            }
            Ast::Repeat(inner, min, max) => {
                let mut at = match max {
                    // Up to max - min optional copies, each skippable to `next`.
                    Some(max) => {
                        let mut at = next;
                        for _ in *min..*max {
                            let body = self.compile(inner, at)?;
                            at = self.push(State::Split(vec![body, next]))?;
                        }
                        at
                    }
                    None => {
                        let lp = self.push(State::Split(Vec::new()))?;
                        let body = self.compile(inner, lp)?;
                        self.states[lp] = State::Split(vec![body, next]);
                        lp
                    }
                };
                for _ in 0..*min {
                    at = self.compile(inner, at)?;
                }
                at
            }
        })
    }

    fn close(&self, live: &mut Live, s: usize) {
        if live.contains(&s) {
            return;
        }
        live.push(s);
        if let State::Split(next) = &self.states[s] {
            for &n in next {
                self.close(live, n);
            }
        }
    }

    /// The states live before any character.
    pub fn start(&self) -> Live {
        let mut live = Vec::new();
        self.close(&mut live, self.start);
        live.sort_unstable();
        live
    }

    /// The states live after `c`; empty if the match is dead.
    pub fn step(&self, live: &Live, c: char) -> Live {
        let mut next = Vec::new();
        for &s in live {
            if let State::Char(class, to) = &self.states[s] {
                if class.matches(c) {
                    self.close(&mut next, *to);
                }
            }
        }
        next.sort_unstable();
        next
    }

    /// Whether the string may end here.
    pub fn accepts(&self, live: &Live) -> bool {
        live.contains(&0)
    }

    /// Whether some character outside ASCII may come next.
    pub fn wants_non_ascii(&self, live: &Live) -> bool {
        live.iter()
            .any(|&s| matches!(&self.states[s], State::Char(c, _) if c.any_non_ascii()))
    }

    /// Whether all of `text` matches.
    pub fn is_match(&self, text: &str) -> bool {
        let mut live = self.start();
        for c in text.chars() {
            live = self.step(&live, c);
            if live.is_empty() {
                return false;
            }
        }
        self.accepts(&live)
    }
}

struct Parser<'a> {
    s: &'a [char],
    i: usize,
}

fn class(ranges: &[(char, char)], negated: bool) -> Class {
    Class {
        ranges: ranges.to_vec(),
        negated,
    }
}

const DIGIT: [(char, char); 1] = [('0', '9')];
const WORD: [(char, char); 4] = [('0', '9'), ('A', 'Z'), ('_', '_'), ('a', 'z')];
const SPACE: [(char, char); 3] = [('\t', '\r'), (' ', ' '), ('\u{a0}', '\u{a0}')];

impl Parser<'_> {
    fn peek(&self) -> Option<char> {
        self.s.get(self.i).copied()
    }

    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(c) {
            self.i += 1;
            true
        } else {
            false
        }
    }

    fn alternation(&mut self) -> Result<Ast> {
        let mut alts = vec![self.concat()?];
        while self.eat('|') {
            alts.push(self.concat()?);
        }
        Ok(if alts.len() == 1 {
            alts.pop().unwrap()
        } else {
            Ast::Alt(alts)
        })
    }

    fn concat(&mut self) -> Result<Ast> {
        let mut parts = Vec::new();
        while let Some(c) = self.peek() {
            if c == '|' || c == ')' {
                break;
            }
            if c == '$' && self.i + 1 == self.s.len() {
                self.i += 1; // the pattern's end, which every match reaches
                break;
            }
            let atom = self.atom()?;
            parts.push(self.quantified(atom)?);
        }
        Ok(match parts.len() {
            0 => Ast::Empty,
            1 => parts.pop().unwrap(),
            _ => Ast::Concat(parts),
        })
    }

    fn number(&mut self) -> Option<usize> {
        let start = self.i;
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.i += 1;
        }
        self.s[start..self.i]
            .iter()
            .collect::<String>()
            .parse()
            .ok()
    }

    /// `{n}`, `{n,}` or `{n,m}` here, if that is what follows.
    fn braces(&mut self) -> Result<Option<(usize, Option<usize>)>> {
        let save = self.i;
        self.i += 1;
        let Some(min) = self.number() else {
            self.i = save; // a literal '{'
            return Ok(None);
        };
        let max = if self.eat(',') {
            self.number()
        } else {
            Some(min)
        };
        if !self.eat('}') {
            self.i = save;
            return Ok(None);
        }
        if let Some(m) = max.filter(|&m| m < min) {
            bail!("bad repetition {{{min},{m}}}");
        }
        Ok(Some((min, max)))
    }

    fn quantified(&mut self, atom: Ast) -> Result<Ast> {
        let (min, max) = match self.peek() {
            Some('*') => (0, None),
            Some('+') => (1, None),
            Some('?') => (0, Some(1)),
            Some('{') => match self.braces()? {
                Some(q) => {
                    self.i -= 1; // so the step below lands after the '}'
                    q
                }
                None => return Ok(atom),
            },
            _ => return Ok(atom),
        };
        self.i += 1;
        self.eat('?'); // lazy: the same strings
        if matches!(self.peek(), Some('*' | '+' | '?')) {
            bail!("nested quantifier");
        }
        Ok(Ast::Repeat(Box::new(atom), min, max))
    }

    fn atom(&mut self) -> Result<Ast> {
        let c = self.peek().unwrap();
        self.i += 1;
        Ok(match c {
            '(' => {
                if self.eat('?') {
                    if self.eat(':') {
                    } else if self.eat('<') || (self.eat('P') && self.eat('<')) {
                        if matches!(self.peek(), Some('=' | '!')) {
                            bail!("lookbehind is not supported");
                        }
                        while self.peek().is_some_and(|c| c != '>') {
                            self.i += 1;
                        }
                        if !self.eat('>') {
                            bail!("unterminated group name");
                        }
                    } else {
                        bail!("lookaround and inline flags are not supported");
                    }
                }
                let inner = self.alternation()?;
                if !self.eat(')') {
                    bail!("missing )");
                }
                inner
            }
            '[' => Ast::Class(self.bracket()?),
            '.' => Ast::Class(class(&[('\n', '\n')], true)),
            '\\' => Ast::Class(self.escape(false)?),
            '^' => bail!("^ is only supported at the start"),
            '$' => bail!("$ is only supported at the end"),
            '*' | '+' | '?' => bail!("nothing to repeat before {c:?}"),
            c => Ast::Class(Class::one(c)),
        })
    }

    /// After a backslash.
    fn escape(&mut self, in_class: bool) -> Result<Class> {
        let Some(c) = self.peek() else {
            bail!("pattern ends with a backslash");
        };
        self.i += 1;
        Ok(match c {
            'd' => class(&DIGIT, false),
            'D' => class(&DIGIT, true),
            'w' => class(&WORD, false),
            'W' => class(&WORD, true),
            's' => class(&SPACE, false),
            'S' => class(&SPACE, true),
            'n' => Class::one('\n'),
            't' => Class::one('\t'),
            'r' => Class::one('\r'),
            'f' => Class::one('\x0c'),
            'v' => Class::one('\x0b'),
            '0' => Class::one('\0'),
            'b' if in_class => Class::one('\x08'),
            'x' | 'u' => {
                let n = if c == 'x' { 2 } else { 4 };
                let hex: String = self
                    .s
                    .get(self.i..self.i + n)
                    .unwrap_or(&[])
                    .iter()
                    .collect();
                let v = u32::from_str_radix(&hex, 16)
                    .ok()
                    .filter(|_| hex.len() == n)
                    .and_then(char::from_u32)
                    .ok_or_else(|| Error(format!("bad \\{c} escape")))?;
                self.i += n;
                Class::one(v)
            }
            c if c.is_ascii_alphanumeric() => {
                bail!("\\{c} is not supported (backreferences and \\b aren't regular)")
            }
            c => Class::one(c),
        })
    }

    /// After `[`.
    fn bracket(&mut self) -> Result<Class> {
        let negated = self.eat('^');
        let mut ranges = Vec::new();
        let mut first = true;
        loop {
            let Some(c) = self.peek() else {
                bail!("missing ]");
            };
            if c == ']' && !first {
                self.i += 1;
                break;
            }
            first = false;
            self.i += 1;
            let lo = if c == '\\' {
                let e = self.escape(true)?;
                if e.negated || e.ranges.len() != 1 || e.ranges[0].0 != e.ranges[0].1 {
                    if e.negated {
                        bail!("negated classes inside [...] are not supported");
                    }
                    ranges.extend(e.ranges);
                    continue;
                }
                e.ranges[0].0
            } else {
                c
            };
            if self.peek() == Some('-') && self.s.get(self.i + 1).is_some_and(|&n| n != ']') {
                self.i += 1;
                let hi = match self.peek().unwrap() {
                    '\\' => {
                        self.i += 1;
                        let e = self.escape(true)?;
                        if e.ranges.len() != 1 || e.ranges[0].0 != e.ranges[0].1 {
                            bail!("bad range in [...]");
                        }
                        e.ranges[0].0
                    }
                    h => {
                        self.i += 1;
                        h
                    }
                };
                if hi < lo {
                    bail!("bad range {lo}-{hi}");
                }
                ranges.push((lo, hi));
            } else {
                ranges.push((lo, lo));
            }
        }
        Ok(Class { ranges, negated })
    }
}

/// A pattern for one of JSON Schema's string formats, where one is known.
pub fn format_pattern(format: &str) -> Option<&'static str> {
    macro_rules! date {
        () => {
            r"\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])"
        };
    }
    macro_rules! time {
        () => {
            r"([01]\d|2[0-3]):[0-5]\d:[0-5]\d(\.\d{1,9})?(Z|[+-]([01]\d|2[0-3]):[0-5]\d)"
        };
    }
    macro_rules! octet {
        () => {
            r"(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
        };
    }
    Some(match format {
        "date" => date!(),
        "time" => time!(),
        "date-time" => concat!(date!(), "T", time!()),
        "uuid" => r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        "email" => {
            r"[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9-]{1,63}(\.[A-Za-z0-9-]{1,63})*\.[A-Za-z]{2,24}"
        }
        "ipv4" => concat!("(", octet!(), r"\.){3}", octet!()),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(p: &str, s: &str) -> bool {
        Regex::new(p).unwrap().is_match(s)
    }

    #[test]
    fn matches_whole_strings() {
        assert!(m("abc", "abc"));
        assert!(!m("abc", "abcd"));
        assert!(!m("abc", "ab"));
        assert!(m("^a.c$", "a-c"));
        assert!(m("a|bc|", ""));
        assert!(m("a|bc|", "bc"));
        assert!(!m("a|bc", "ac"));
    }

    #[test]
    fn classes_and_escapes() {
        assert!(m(r"\d{3}-\d{4}", "555-1234"));
        assert!(!m(r"\d{3}-\d{4}", "555-12345"));
        assert!(m(r"[A-Z][a-z_]*", "Hello_world"));
        assert!(!m(r"[A-Z][a-z_]*", "hello"));
        assert!(m(r"[^0-9]+", "abc"));
        assert!(!m(r"[^0-9]+", "a1"));
        assert!(m(r"\w+\s\W", "ab_1 !"));
        assert!(m(r"[\d.-]+", "3.14-2"));
        assert!(m(r"a\.b\\c", r"a.b\c"));
        assert!(m(r"\u00e9t\x65", "éte"));
        assert!(m(".", "é"));
        assert!(!m(".", "\n"));
        assert!(m("[-a]", "-") && m("[a-]", "-") && m("[]a]", "]"));
    }

    #[test]
    fn quantifiers_and_groups() {
        assert!(m("(ab)+", "ababab"));
        assert!(!m("(ab)+", "aba"));
        assert!(m("(?:a|b){2,3}", "aba"));
        assert!(!m("(?:a|b){2,3}", "abab"));
        assert!(!m("(?:a|b){2,3}", "a"));
        assert!(m("a{2,}", "aaaaa") && !m("a{2,}", "a"));
        assert!(m("(?<year>\\d{4})", "2026"));
        assert!(m("x*?y+?z??", "xyy"));
        assert!(m("a{,2}", "a{,2}")); // not a quantifier: literal braces
        assert!(m("colou?r", "color") && m("colou?r", "colour"));
    }

    #[test]
    fn rejects_what_is_not_regular() {
        for p in [
            r"(a)\1", r"\bword", "(?=a)", "(?<=a)b", "a**", "*a", "(a", "[a",
        ] {
            assert!(Regex::new(p).is_err(), "{p}");
        }
        assert!(Regex::new("(a{200}){200}").is_err());
    }

    #[test]
    fn steps_tell_live_and_accepting() {
        let r = Regex::new(r"\d{2}").unwrap();
        let s = r.start();
        assert!(!r.accepts(&s));
        let s = r.step(&s, '4');
        assert!(!s.is_empty() && !r.accepts(&s));
        let s2 = r.step(&s, '2');
        assert!(r.accepts(&s2));
        assert!(r.step(&s, 'x').is_empty());
        assert!(!r.wants_non_ascii(&r.start()));
        assert!(Regex::new(".")
            .unwrap()
            .wants_non_ascii(&Regex::new(".").unwrap().start()));
    }

    #[test]
    fn formats() {
        let f = |name: &str, s: &str| {
            Regex::new(format_pattern(name).unwrap())
                .unwrap()
                .is_match(s)
        };
        assert!(f("date", "2026-11-05") && !f("date", "2026-13-05"));
        assert!(f("date-time", "2026-11-05T09:30:00Z"));
        assert!(f("date-time", "2026-11-05T09:30:00.123+05:30"));
        assert!(!f("date-time", "2026-11-05 09:30:00"));
        assert!(f("time", "23:59:59Z") && !f("time", "24:00:00Z"));
        assert!(f("uuid", "123e4567-e89b-12d3-a456-426614174000"));
        assert!(f("email", "ada.lovelace@example.co.uk") && !f("email", "nope@"));
        assert!(f("ipv4", "192.168.0.255") && !f("ipv4", "256.1.1.1"));
        assert!(format_pattern("color").is_none());
    }
}
