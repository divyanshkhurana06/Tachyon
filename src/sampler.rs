//! Picks the next token from the logits: greedy, or temperature sampling
//! with top-k, top-p (nucleus) and a repetition penalty.

use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// 0 means greedy (always the most likely token).
    pub temperature: f32,
    /// Keep only the k most likely tokens (0 = no limit).
    pub top_k: usize,
    /// Keep the smallest set of tokens whose probability reaches p.
    pub top_p: f32,
    /// Above 1 discourages tokens that appeared recently.
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl Default for SamplerConfig {
    /// Qwen2.5's recommended settings.
    fn default() -> Self {
        SamplerConfig {
            temperature: 0.7,
            top_k: 20,
            top_p: 0.8,
            repetition_penalty: 1.05,
            seed: clock_seed(),
        }
    }
}

impl SamplerConfig {
    pub fn greedy() -> Self {
        SamplerConfig {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: 0,
        }
    }
}

fn clock_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0x9E37_79B9, |d| d.as_nanos() as u64)
}

pub struct Sampler {
    cfg: SamplerConfig,
    rng: u64,
    buf: Vec<f32>,
    ranked: Vec<(f32, u32)>,
    idx: Vec<u32>,
    probs: Vec<f32>,
}

impl Sampler {
    pub fn new(cfg: SamplerConfig) -> Sampler {
        let rng = cfg.seed.wrapping_mul(0x2545_F491_4F6C_DD1D) | 1;
        Sampler {
            cfg,
            rng,
            buf: Vec::new(),
            ranked: Vec::new(),
            idx: Vec::new(),
            probs: Vec::new(),
        }
    }

    /// `vocab` caps the ids considered: checkpoints pad the output layer
    /// with rows the tokenizer has no token for.
    pub fn sample(&mut self, logits: &[f32], vocab: usize, recent: &[u32]) -> u32 {
        let logits = &logits[..vocab.min(logits.len())];
        let penalize = self.cfg.repetition_penalty != 1.0 && !recent.is_empty();
        if self.cfg.temperature <= 0.0 && !penalize {
            return argmax(logits);
        }
        self.pick(logits, recent, None)
    }

    /// Like [`Sampler::sample`], but only tokens for which `allowed` holds
    /// can be picked (constrained decoding).
    pub fn sample_with(
        &mut self,
        logits: &[f32],
        vocab: usize,
        recent: &[u32],
        allowed: &dyn Fn(u32) -> bool,
    ) -> u32 {
        let logits = &logits[..vocab.min(logits.len())];
        self.pick(logits, recent, Some(allowed))
    }

    /// The probabilities the sampler would draw from, after every setting
    /// is applied, as (token, probability) pairs summing to 1. Greedy
    /// sampling gives the single best token probability 1.
    pub fn distribution(
        &mut self,
        logits: &[f32],
        vocab: usize,
        recent: &[u32],
    ) -> Vec<(u32, f32)> {
        let logits = &logits[..vocab.min(logits.len())];
        let keep = self.candidates(logits, recent, None);
        let mass: f32 = self.probs[..keep].iter().sum();
        self.idx[..keep]
            .iter()
            .zip(&self.probs)
            .map(|(&t, &p)| (t, p / mass))
            .collect()
    }

    /// Draws a token from the output of [`Sampler::distribution`] (or any
    /// weights; they need not sum to 1).
    pub fn draw(&mut self, weighted: &[(u32, f32)]) -> u32 {
        let total: f32 = weighted.iter().map(|(_, p)| p).sum();
        let mut r = self.next_f32() * total;
        for &(t, p) in weighted {
            r -= p;
            if r <= 0.0 {
                return t;
            }
        }
        weighted.last().map_or(0, |&(t, _)| t)
    }

    fn pick(
        &mut self,
        logits: &[f32],
        recent: &[u32],
        allowed: Option<&dyn Fn(u32) -> bool>,
    ) -> u32 {
        let keep = self.candidates(logits, recent, allowed);
        if keep == 0 {
            return argmax(logits); // nothing allowed: cannot happen for JSON
        }
        let mass: f32 = self.probs[..keep].iter().sum();
        let mut r = self.next_f32() * mass;
        for j in 0..keep {
            r -= self.probs[j];
            if r <= 0.0 {
                return self.idx[j];
            }
        }
        self.idx[keep - 1]
    }

    /// Fills `idx` with candidate tokens, best first, and `probs` with
    /// their unnormalized probabilities; returns how many survive top-p.
    fn candidates(
        &mut self,
        logits: &[f32],
        recent: &[u32],
        allowed: Option<&dyn Fn(u32) -> bool>,
    ) -> usize {
        let greedy = self.cfg.temperature <= 0.0;
        let t = if greedy { 1.0 } else { self.cfg.temperature };
        self.buf.clear();
        self.buf.extend_from_slice(logits);
        if self.cfg.repetition_penalty != 1.0 && !recent.is_empty() {
            let mut seen: Vec<u32> = recent.to_vec();
            seen.sort_unstable();
            seen.dedup();
            let p = self.cfg.repetition_penalty;
            for id in seen {
                if let Some(l) = self.buf.get_mut(id as usize) {
                    *l = if *l > 0.0 { *l / p } else { *l * p };
                }
            }
        }
        let n = self.buf.len();
        let want = if greedy {
            1
        } else if self.cfg.top_k == 0 {
            n
        } else {
            self.cfg.top_k.min(n)
        };
        // With a constraint, rank a wider window so some allowed token is
        // likely to be in it.
        let window = if allowed.is_some() {
            want.max(256).min(n)
        } else {
            want
        };

        self.ranked.clear();
        if window <= 1024 {
            top_n(&self.buf, window, &mut self.ranked);
        } else {
            // No top-k: everything within 30 temperatures of the best (any
            // other token has relative probability under e^-30), sorted.
            let max = self.buf.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let cutoff = max - 30.0 * t;
            self.ranked.extend(
                self.buf
                    .iter()
                    .enumerate()
                    .filter(|(_, &l)| l >= cutoff)
                    .map(|(i, &l)| (l, i as u32)),
            );
            self.ranked.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        }
        if let Some(ok) = allowed {
            self.ranked.retain(|&(_, id)| ok(id));
            if self.ranked.is_empty() {
                // Every likely token is ruled out: look at the whole vocabulary.
                self.ranked.extend(
                    self.buf
                        .iter()
                        .enumerate()
                        .filter(|&(i, _)| ok(i as u32))
                        .map(|(i, &l)| (l, i as u32)),
                );
                self.ranked.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            }
        }
        self.ranked.truncate(want);
        self.idx.clear();
        self.idx.extend(self.ranked.iter().map(|&(_, id)| id));
        if self.idx.is_empty() {
            self.probs.clear();
            return 0;
        }

        let best = self.ranked[0].0;
        self.probs.clear();
        self.probs
            .extend(self.ranked.iter().map(|&(l, _)| ((l - best) / t).exp()));
        if greedy || self.cfg.top_p >= 1.0 {
            return self.probs.len();
        }
        let total: f32 = self.probs.iter().sum();
        let mut acc = 0.0;
        for (j, p) in self.probs.iter().enumerate() {
            acc += p / total;
            if acc >= self.cfg.top_p {
                return j + 1;
            }
        }
        self.probs.len()
    }

    /// A uniform random number in [0, 1).
    pub fn uniform(&mut self) -> f32 {
        self.next_f32()
    }

    /// xorshift64*: fast, and plenty random for picking tokens.
    fn next_f32(&mut self) -> f32 {
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        let x = self.rng.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (x >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// The `n` largest values of `x` with their indices, largest first, in one
/// pass: a value is only inserted if it beats the current n-th best, which
/// after the first few thousand elements almost never happens.
pub fn top_n(x: &[f32], n: usize, out: &mut Vec<(f32, u32)>) {
    out.clear();
    if n == 0 {
        return;
    }
    let mut floor = f32::NEG_INFINITY;
    for (i, &v) in x.iter().enumerate() {
        if out.len() < n || v > floor {
            let at = out.partition_point(|&(w, _)| w >= v);
            if out.len() == n {
                out.pop();
            }
            out.insert(at, (v, i as u32));
            if out.len() == n {
                floor = out[n - 1].0;
            }
        }
    }
}

/// A token's log-probability under the model, and the `n` most likely
/// alternatives', from raw logits (before temperature or penalties, as
/// OpenAI reports them).
#[derive(Debug, Clone, PartialEq)]
pub struct TokenLogprob {
    pub token: u32,
    pub logprob: f32,
    pub top: Vec<(u32, f32)>,
}

pub fn logprobs(logits: &[f32], token: u32, n: usize) -> TokenLogprob {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse = max
        + logits
            .iter()
            .map(|&l| ((l - max) as f64).exp())
            .sum::<f64>()
            .ln() as f32;
    let mut best = Vec::new();
    top_n(logits, n, &mut best);
    TokenLogprob {
        token,
        logprob: logits[token as usize] - lse,
        top: best.into_iter().map(|(l, t)| (t, l - lse)).collect(),
    }
}

pub fn argmax(x: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in x.iter().enumerate() {
        if v > x[best] {
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_and_top_k() {
        let logits = [0.1, 3.0, 2.9, -1.0];
        assert_eq!(
            Sampler::new(SamplerConfig::greedy()).sample(&logits, 4, &[]),
            1
        );
        let mut s = Sampler::new(SamplerConfig {
            temperature: 1.0,
            top_k: 2,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: 3,
        });
        for _ in 0..200 {
            assert!(matches!(s.sample(&logits, 4, &[]), 1 | 2));
        }
    }

    #[test]
    fn constraint_and_distribution() {
        let logits = [0.1, 3.0, 2.9, -1.0];
        let mut s = Sampler::new(SamplerConfig::greedy());
        assert_eq!(s.sample_with(&logits, 4, &[], &|t| t != 1), 2);
        let mut s = Sampler::new(SamplerConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: 1,
        });
        let d = s.distribution(&logits, 4, &[]);
        assert_eq!(d[0].0, 1);
        assert!((d.iter().map(|x| x.1).sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn top_n_matches_a_full_sort() {
        let x: Vec<f32> = (0..5000u32)
            .map(|i| ((i.wrapping_mul(2_654_435_761)) % 10007) as f32)
            .collect();
        let mut sorted: Vec<(f32, u32)> =
            x.iter().enumerate().map(|(i, &v)| (v, i as u32)).collect();
        sorted.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut out = Vec::new();
        top_n(&x, 37, &mut out);
        assert_eq!(
            out.iter().map(|p| p.0).collect::<Vec<_>>(),
            sorted[..37].iter().map(|p| p.0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn logprobs_are_log_softmax() {
        let lp = logprobs(&[1.0, 2.0, 3.0], 1, 2);
        let z = (1f32.exp() + 2f32.exp() + 3f32.exp()).ln();
        assert!((lp.logprob - (2.0 - z)).abs() < 1e-5);
        assert_eq!(lp.top.iter().map(|t| t.0).collect::<Vec<_>>(), [2, 1]);
    }

    #[test]
    fn penalty_can_flip_the_choice() {
        let mut s = Sampler::new(SamplerConfig {
            repetition_penalty: 2.0,
            ..SamplerConfig::greedy()
        });
        assert_eq!(s.sample(&[0.0, 3.0, 2.0], 3, &[1]), 2);
    }
}
