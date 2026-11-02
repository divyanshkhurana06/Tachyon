use std::fs;
use std::path::Path;

use crate::json::Json;
use crate::{bail, Error, Result};

/// Model shape, read from a Hugging Face `config.json`.
#[derive(Debug, Clone)]
pub struct Config {
    pub model_type: String,
    pub hidden: usize,
    pub intermediate: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    pub max_positions: usize,
    pub rope_theta: f64,
    /// Llama 3's stretch of the low RoPE frequencies for long context.
    pub rope_scaling: Option<RopeScaling>,
    pub norm_eps: f32,
    pub tied_embeddings: bool,
    pub eos_ids: Vec<u32>,
}

/// Llama 3.1+ RoPE scaling: wavelengths shorter than the original context
/// over `high_freq_factor` keep their frequency, those longer than it over
/// `low_freq_factor` are slowed by `factor`, and the band between blends
/// the two.
#[derive(Debug, Clone, PartialEq)]
pub struct RopeScaling {
    pub factor: f64,
    pub low_freq_factor: f64,
    pub high_freq_factor: f64,
    pub original_max_positions: f64,
}

impl RopeScaling {
    pub fn apply(&self, freq: f64) -> f64 {
        let wavelen = 2.0 * std::f64::consts::PI / freq;
        let (low, high) = (
            self.original_max_positions / self.low_freq_factor,
            self.original_max_positions / self.high_freq_factor,
        );
        if wavelen < high {
            freq
        } else if wavelen > low {
            freq / self.factor
        } else {
            let smooth = (self.original_max_positions / wavelen - self.low_freq_factor)
                / (self.high_freq_factor - self.low_freq_factor);
            (1.0 - smooth) * freq / self.factor + smooth * freq
        }
    }
}

impl Config {
    pub fn load(dir: &Path) -> Result<Config> {
        let path = dir.join("config.json");
        let text = fs::read_to_string(&path)
            .map_err(|e| Error(format!("cannot read {}: {e}", path.display())))?;
        let j = Json::parse(&text)?;
        let num = |k: &str| {
            j.get(k)
                .and_then(Json::as_usize)
                .ok_or_else(|| Error(format!("config.json: missing {k}")))
        };

        let model_type = j
            .get("model_type")
            .and_then(Json::as_str)
            .unwrap_or("")
            .to_string();
        if !matches!(model_type.as_str(), "qwen2" | "qwen3" | "llama" | "mistral") {
            bail!(
                "model_type {model_type:?} is not supported (qwen2, qwen3, llama and mistral are)"
            );
        }
        let rope_scaling = match j.get("rope_scaling") {
            None | Some(Json::Null) => None,
            Some(r) if r.get("rope_type").and_then(Json::as_str) == Some("llama3") => {
                let f = |k: &str| {
                    r.get(k)
                        .and_then(Json::as_f64)
                        .ok_or_else(|| Error(format!("config.json: rope_scaling lacks {k}")))
                };
                Some(RopeScaling {
                    factor: f("factor")?,
                    low_freq_factor: f("low_freq_factor")?,
                    high_freq_factor: f("high_freq_factor")?,
                    original_max_positions: f("original_max_position_embeddings")?,
                })
            }
            Some(_) => bail!("only Llama 3's rope_scaling is supported"),
        };
        if j.get("hidden_act")
            .and_then(Json::as_str)
            .is_some_and(|a| a != "silu")
        {
            bail!("only SiLU-gated MLPs are supported");
        }

        let hidden = num("hidden_size")?;
        let heads = num("num_attention_heads")?;
        let kv_heads = j
            .get("num_key_value_heads")
            .and_then(Json::as_usize)
            .unwrap_or(heads);
        let head_dim = j
            .get("head_dim")
            .and_then(Json::as_usize)
            .unwrap_or(hidden / heads);
        if heads % kv_heads != 0 || head_dim % 2 != 0 {
            bail!("config.json: unsupported head layout");
        }

        // generation_config.json often adds stop tokens (e.g. <|endoftext|>).
        let mut eos_ids = token_ids(j.get("eos_token_id"));
        if let Some(g) = fs::read_to_string(dir.join("generation_config.json"))
            .ok()
            .and_then(|t| Json::parse(&t).ok())
        {
            for id in token_ids(g.get("eos_token_id")) {
                if !eos_ids.contains(&id) {
                    eos_ids.push(id);
                }
            }
        }

        Ok(Config {
            model_type,
            hidden,
            intermediate: num("intermediate_size")?,
            layers: num("num_hidden_layers")?,
            heads,
            kv_heads,
            head_dim,
            vocab: num("vocab_size")?,
            max_positions: j
                .get("max_position_embeddings")
                .and_then(Json::as_usize)
                .unwrap_or(4096),
            rope_theta: j
                .get("rope_theta")
                .and_then(Json::as_f64)
                .unwrap_or(10_000.0),
            rope_scaling,
            norm_eps: j.get("rms_norm_eps").and_then(Json::as_f64).unwrap_or(1e-6) as f32,
            tied_embeddings: j
                .get("tie_word_embeddings")
                .and_then(Json::as_bool)
                .unwrap_or(false),
            eos_ids,
        })
    }

    pub fn q_dim(&self) -> usize {
        self.heads * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// Parameter count, for display.
    pub fn params(&self) -> usize {
        let attn = self.hidden * (self.q_dim() + 2 * self.kv_dim()) + self.q_dim() * self.hidden;
        let mlp = 3 * self.hidden * self.intermediate;
        let embed = self.vocab * self.hidden * if self.tied_embeddings { 1 } else { 2 };
        self.layers * (attn + mlp) + embed
    }
}

fn token_ids(v: Option<&Json>) -> Vec<u32> {
    match v {
        Some(Json::Arr(a)) => a
            .iter()
            .filter_map(Json::as_u64)
            .map(|n| n as u32)
            .collect(),
        Some(v) => v.as_u64().map(|n| vec![n as u32]).unwrap_or_default(),
        None => Vec::new(),
    }
}
