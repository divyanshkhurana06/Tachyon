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

        self.buf.clear();
        self.buf.extend_from_slice(logits);
        if penalize {
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
        if self.cfg.temperature <= 0.0 {
            return argmax(&self.buf);
        }

        let n = self.buf.len();
        let k = if self.cfg.top_k == 0 {
            n
        } else {
            self.cfg.top_k.min(n)
        };
        let buf = &self.buf;
        let by_logit = |a: &u32, b: &u32| buf[*b as usize].total_cmp(&buf[*a as usize]);
        self.idx.clear();
        self.idx.extend(0..n as u32);
        if k < n {
            self.idx.select_nth_unstable_by(k - 1, by_logit);
            self.idx.truncate(k);
        }
        self.idx.sort_unstable_by(by_logit);

        let max = buf[self.idx[0] as usize];
        let t = self.cfg.temperature;
        self.probs.clear();
        self.probs.extend(
            self.idx
                .iter()
                .map(|&i| ((buf[i as usize] - max) / t).exp()),
        );
        let total: f32 = self.probs.iter().sum();

        let mut keep = self.probs.len();
        if self.cfg.top_p < 1.0 {
            let mut acc = 0.0;
            for (j, p) in self.probs.iter().enumerate() {
                acc += p / total;
                if acc >= self.cfg.top_p {
                    keep = j + 1;
                    break;
                }
            }
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

    /// xorshift64*: fast, and plenty random for picking tokens.
    fn next_f32(&mut self) -> f32 {
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        let x = self.rng.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (x >> 40) as f32 / (1u64 << 24) as f32
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
    fn penalty_can_flip_the_choice() {
        let mut s = Sampler::new(SamplerConfig {
            repetition_penalty: 2.0,
            ..SamplerConfig::greedy()
        });
        assert_eq!(s.sample(&[0.0, 3.0, 2.0], 3, &[1]), 2);
    }
}
