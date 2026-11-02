//! Importance matrices for activation-aware quantization.
//!
//! Rounding a weight to 4 bits costs as much as the error times the input
//! it multiplies, and some input channels carry far larger activations than
//! others. Running calibration text through the model and recording each
//! linear layer's mean squared input per channel tells the quantizer which
//! columns' errors matter, so it can spend its precision there (llama.cpp's
//! imatrix, in the spirit of AWQ). Fitting scales by weight error alone,
//! without this, made perplexity worse.

use std::fs;
use std::path::Path;

use crate::config::Config;
use crate::{bail, Result};

const MAGIC: &[u8; 4] = b"TQIM";

/// Sums of squared inputs, per input channel, for each matrix of a layer.
#[derive(Debug, Clone)]
pub struct LayerImportance {
    pub qkv: Vec<f32>,
    pub o: Vec<f32>,
    pub gate_up: Vec<f32>,
    pub down: Vec<f32>,
}

impl LayerImportance {
    /// The vector for a matrix kind (`qkv`, `o`, `gate_up`, `down`).
    pub fn get(&self, which: &str) -> Option<&[f32]> {
        match which {
            "qkv" => Some(&self.qkv),
            "o" => Some(&self.o),
            "gate_up" => Some(&self.gate_up),
            "down" => Some(&self.down),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Importance {
    /// Tokens the sums cover.
    pub tokens: u64,
    pub layers: Vec<LayerImportance>,
}

impl Importance {
    pub fn new(c: &Config) -> Importance {
        let layer = LayerImportance {
            qkv: vec![0.0; c.hidden],
            o: vec![0.0; c.q_dim()],
            gate_up: vec![0.0; c.hidden],
            down: vec![0.0; c.intermediate],
        };
        Importance {
            tokens: 0,
            layers: vec![layer; c.layers],
        }
    }

    /// Adds the squares of each row of `x` (rows of `sums.len()` values).
    pub fn accumulate(sums: &mut [f32], x: &[f32]) {
        for row in x.chunks_exact(sums.len()) {
            for (s, &v) in sums.iter_mut().zip(row) {
                *s += v * v;
            }
        }
    }

    /// Divides the sums by the token count, giving mean squared inputs.
    pub fn normalized(&self) -> Importance {
        let n = self.tokens.max(1) as f32;
        let scale = |v: &Vec<f32>| v.iter().map(|x| x / n).collect();
        Importance {
            tokens: self.tokens,
            layers: self
                .layers
                .iter()
                .map(|l| LayerImportance {
                    qkv: scale(&l.qkv),
                    o: scale(&l.o),
                    gate_up: scale(&l.gate_up),
                    down: scale(&l.down),
                })
                .collect(),
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut out = MAGIC.to_vec();
        out.extend_from_slice(&self.tokens.to_le_bytes());
        out.extend_from_slice(&(self.layers.len() as u32).to_le_bytes());
        for l in &self.layers {
            for v in [&l.qkv, &l.o, &l.gate_up, &l.down] {
                out.extend_from_slice(&(v.len() as u32).to_le_bytes());
                out.extend(v.iter().flat_map(|x| x.to_le_bytes()));
            }
        }
        fs::write(path, out)?;
        Ok(())
    }

    /// Loads a saved matrix and checks it fits `c`.
    pub fn load(path: &Path, c: &Config) -> Result<Importance> {
        let bytes = fs::read(path)
            .map_err(|e| crate::Error(format!("cannot read {}: {e}", path.display())))?;
        let mut at = 0;
        let mut take = |n: usize| -> Result<&[u8]> {
            let s = bytes
                .get(at..at + n)
                .ok_or_else(|| crate::Error::msg("importance file is truncated"))?;
            at += n;
            Ok(s)
        };
        if take(4)? != MAGIC {
            bail!("{} is not an importance matrix", path.display());
        }
        let tokens = u64::from_le_bytes(take(8)?.try_into().unwrap());
        let layers = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
        let expected = Importance::new(c);
        if layers != expected.layers.len() {
            bail!(
                "importance matrix has {layers} layers; the model has {}",
                expected.layers.len()
            );
        }
        let mut out = Vec::with_capacity(layers);
        for want in &expected.layers {
            let mut read = |len: usize| -> Result<Vec<f32>> {
                let n = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
                if n != len {
                    bail!("importance matrix does not match this model's shape");
                }
                Ok(take(n * 4)?
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect())
            };
            out.push(LayerImportance {
                qkv: read(want.qkv.len())?,
                o: read(want.o.len())?,
                gate_up: read(want.gate_up.len())?,
                down: read(want.down.len())?,
            });
        }
        Ok(Importance {
            tokens,
            layers: out,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saves_and_loads() {
        let c = Config {
            model_type: "qwen2".into(),
            hidden: 4,
            intermediate: 6,
            layers: 2,
            heads: 2,
            kv_heads: 1,
            head_dim: 2,
            vocab: 10,
            max_positions: 16,
            rope_theta: 1e4,
            rope_scaling: None,
            norm_eps: 1e-6,
            tied_embeddings: true,
            eos_ids: vec![],
            embedding_multiplier: 1.0,
            attention_multiplier: None,
            residual_multiplier: 1.0,
            logits_scaling: 1.0,
            experts: 0,
            experts_per_token: 0,
        };
        let mut imp = Importance::new(&c);
        Importance::accumulate(
            &mut imp.layers[1].down,
            &[1.0, 2.0, 0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        imp.tokens = 2;
        let path = std::env::temp_dir().join(format!("tachyon-imatrix-{}.bin", std::process::id()));
        imp.save(&path).unwrap();
        let back = Importance::load(&path, &c).unwrap().normalized();
        assert_eq!(back.layers[1].down, [1.0, 2.0, 0.0, 0.0, 0.0, 4.5]);
        let mut other = c.clone();
        other.layers = 3;
        assert!(Importance::load(&path, &other).is_err());
        let _ = std::fs::remove_file(path);
    }
}
