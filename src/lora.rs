//! LoRA adapters, many served at once from one base model.
//!
//! A LoRA fine-tune changes a weight matrix W to W + s B A, where A is
//! r x in, B is out x r, r is small (8 to 64) and s = alpha / r. Instead of
//! merging that into a copy of the model per fine-tune, the product is
//! applied at run time beside the base matmul, y += s B (A x). Each token of
//! a batch names its own adapter or none, so requests for different
//! fine-tunes share every forward pass (as in S-LoRA and Punica). A rank-16
//! adapter adds about 2% to the arithmetic of the tokens that use it and
//! nothing to those that don't.
//!
//! Adapters load from PEFT's format: `adapter_config.json` (`r`,
//! `lora_alpha`, `use_rslora`, `target_modules`) and
//! `adapter_model.safetensors`.

use std::path::Path;

use crate::config::Config;
use crate::json::Json;
use crate::kernels::{dot_bf16, f32_to_bf16};
use crate::pool::{Pool, SyncPtr};
use crate::safetensors::SafeTensors;
use crate::{bail, Error, Result};

/// Every adapter's low-rank update to one of the model's (stacked) matrices.
/// The stacked matrices hold several projections one after another (q, k
/// and v; gate and up), each with its own A and B, so A is kept as
/// `segments * rank` rows and row `r` of the output reads only its own
/// segment's `rank` values.
pub struct Lora {
    pub rows: usize,
    pub cols: usize,
    pub rank: usize,
    /// Where each segment's output rows end.
    pub ends: Vec<usize>,
    /// `[adapter][segment * rank + j][cols]`, bf16 (adapters are read whole
    /// for every step that uses them, so their size is their cost).
    pub a: Vec<u16>,
    /// `[adapter][row][rank]`, bf16, with the scale folded in.
    pub b: Vec<u16>,
}

impl Lora {
    /// Rows of A per adapter.
    pub fn inner(&self) -> usize {
        self.ends.len() * self.rank
    }

    /// Adds every token's update to `y` (`ids.len()` rows of `rows`) on the
    /// pool, skipping tokens with no adapter (id 0). Tokens are grouped by
    /// adapter so each adapter's A and B are read once per batch, not once
    /// per token. Each token's arithmetic is the same whatever else is in
    /// the batch.
    pub fn apply_batch(&self, pool: &Pool, ids: &[u32], x: &[f32], y: &mut [f32]) {
        let groups = group(ids);
        if groups.is_empty() {
            return;
        }
        let (inner, r, cols, rows) = (self.inner(), self.rank, self.cols, self.rows);
        let mut h = vec![0.0f32; ids.len() * inner];
        let hp = SyncPtr::new(&mut h);
        // h = A x: each row of A against its group's tokens.
        pool.parallel_for(inner, 4, |js| {
            for j in js {
                for (id, toks) in &groups {
                    let a = &self.a[((id - 1) * inner + j) * cols..][..cols];
                    for &tok in toks {
                        let v = dot_bf16(a, &x[tok * cols..(tok + 1) * cols]);
                        // Each (token, j) is written once.
                        unsafe { hp.write(tok * inner + j, v) };
                    }
                }
            }
        });
        // y += B h, each output row reading its own segment of h.
        let out = SyncPtr::new(y);
        pool.parallel_for(rows, 64, |range| {
            for row in range {
                let seg = self.ends.iter().position(|&e| row < e).unwrap();
                for (id, toks) in &groups {
                    let b = &self.b[((id - 1) * rows + row) * r..][..r];
                    for &tok in toks {
                        let hs = &h[tok * inner + seg * r..][..r];
                        // Rows are split between threads, so no two touch
                        // the same element.
                        unsafe { out.slice(tok * rows + row, 1)[0] += dot_bf16(b, hs) };
                    }
                }
            }
        });
    }
}

/// The batch's tokens grouped by adapter, without those that have none.
pub fn group(ids: &[u32]) -> Vec<(usize, Vec<usize>)> {
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    for (tok, &id) in ids.iter().enumerate().filter(|(_, &id)| id != 0) {
        match groups.iter_mut().find(|(g, _)| *g == id as usize) {
            Some((_, toks)) => toks.push(tok),
            None => groups.push((id as usize, vec![tok])),
        }
    }
    groups
}

/// One layer's updates; `None` where no adapter touches the matrix.
#[derive(Default)]
pub struct LayerLora {
    pub qkv: Option<Lora>,
    pub o: Option<Lora>,
    pub gate_up: Option<Lora>,
    pub down: Option<Lora>,
}

/// A set of adapters for one base model. Adapter ids start at 1 (0 is the
/// base model) in the order given; ranks are padded with zeros to the
/// largest.
pub struct Adapters {
    pub names: Vec<String>,
    pub rank: usize,
    pub layers: Vec<LayerLora>,
}

/// One adapter as read from disk.
pub struct Adapter {
    pub name: String,
    pub rank: usize,
    pub scale: f32,
    pub weights: Vec<Projection>,
}

/// An adapter's A (`rank x in`) and B (`out x rank`, scale folded in) for
/// one projection: `module` indexes [`MODULES`].
pub struct Projection {
    pub layer: usize,
    pub module: usize,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
}

/// The projections an adapter may target, in the order they are stacked.
pub const MODULES: [&str; 7] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
];

impl Adapter {
    /// Reads a PEFT adapter directory.
    pub fn load(name: &str, dir: &Path) -> Result<Adapter> {
        let path = dir.join("adapter_config.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|e| Error(format!("cannot read {}: {e}", path.display())))?;
        let cfg = Json::parse(&text).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        let st = SafeTensors::open_dir(dir)?;
        Adapter::from_parts(name, &cfg, &st)
    }

    pub fn from_parts(name: &str, cfg: &Json, st: &SafeTensors) -> Result<Adapter> {
        let rank = cfg
            .get("r")
            .and_then(Json::as_usize)
            .ok_or_else(|| Error(format!("adapter {name}: no rank (r)")))?;
        let alpha = cfg
            .get("lora_alpha")
            .and_then(Json::as_f64)
            .unwrap_or(rank as f64);
        let rs = cfg.get("use_rslora").and_then(Json::as_bool) == Some(true);
        let scale = if rs {
            alpha / (rank as f64).sqrt()
        } else {
            alpha / rank as f64
        } as f32;
        if cfg.get("fan_in_fan_out").and_then(Json::as_bool) == Some(true) {
            bail!("adapter {name}: fan_in_fan_out is not supported");
        }
        if cfg
            .get("modules_to_save")
            .is_some_and(|m| m.as_array().is_some_and(|a| !a.is_empty()))
        {
            bail!("adapter {name}: modules_to_save (fully trained layers) is not supported");
        }

        // Tensors are named like
        // base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight.
        let mut weights = Vec::new();
        for full in st.names() {
            let Some(rest) = full.split("layers.").nth(1) else {
                continue;
            };
            let parts: Vec<&str> = rest.split('.').collect();
            let (Some(layer), Some(module), Some(which)) = (
                parts.first().and_then(|l| l.parse::<usize>().ok()),
                parts.get(2),
                parts.get(3),
            ) else {
                continue;
            };
            if *which != "lora_A" {
                continue;
            }
            let Some(m) = MODULES.iter().position(|x| x == module) else {
                bail!("adapter {name}: {full} targets a module this engine does not adapt");
            };
            let b_name = full.replacen("lora_A", "lora_B", 1);
            let (a, b) = (st.tensor(full)?, st.tensor(&b_name)?);
            if a.shape.len() != 2 || b.shape.len() != 2 || a.shape[0] != rank || b.shape[1] != rank
            {
                bail!(
                    "adapter {name}: {full} is {:?} and its B {:?}, expected rank {rank}",
                    a.shape,
                    b.shape
                );
            }
            let mut bv = b.to_f32();
            bv.iter_mut().for_each(|v| *v *= scale);
            weights.push(Projection {
                layer,
                module: m,
                a: a.to_f32(),
                b: bv,
            });
        }
        if weights.is_empty() {
            bail!("adapter {name}: no LoRA weights found");
        }
        weights.sort_by_key(|p| (p.layer, p.module));
        Ok(Adapter {
            name: name.to_string(),
            rank,
            scale,
            weights,
        })
    }
}

impl Adapters {
    /// Loads `name=dir` pairs.
    pub fn load(c: &Config, specs: &[(String, std::path::PathBuf)]) -> Result<Adapters> {
        let list = specs
            .iter()
            .map(|(name, dir)| Adapter::load(name, dir))
            .collect::<Result<Vec<_>>>()?;
        Adapters::new(c, list)
    }

    /// Packs adapters for `c`'s shapes.
    pub fn new(c: &Config, list: Vec<Adapter>) -> Result<Adapters> {
        if list.is_empty() {
            bail!("no adapters");
        }
        for (i, a) in list.iter().enumerate() {
            if list[..i].iter().any(|b| b.name == a.name) || a.name.is_empty() {
                bail!(
                    "adapter names must be distinct and non-empty ({:?})",
                    a.name
                );
            }
        }
        // Padded to a multiple of 4 for the GPU's vector loads.
        let rank = list
            .iter()
            .map(|a| a.rank)
            .max()
            .unwrap()
            .next_multiple_of(4);
        let (q, kv, inter, dim) = (c.q_dim(), c.kv_dim(), c.intermediate, c.hidden);
        // (modules, their output rows, input columns) per stacked matrix.
        let shapes: [(&[usize], Vec<usize>, usize); 4] = [
            (&[0, 1, 2], vec![q, kv, kv], dim),
            (&[3], vec![dim], q),
            (&[4, 5], vec![inter, inter], dim),
            (&[6], vec![dim], inter),
        ];
        let n = list.len();
        let mut layers = Vec::with_capacity(c.layers);
        for l in 0..c.layers {
            let mut out: [Option<Lora>; 4] = Default::default();
            for (slot, (modules, seg_rows, cols)) in shapes.iter().enumerate() {
                let used = list.iter().any(|a| {
                    a.weights
                        .iter()
                        .any(|p| p.layer == l && modules.contains(&p.module))
                });
                if !used {
                    continue;
                }
                if slot >= 2 && c.experts > 0 {
                    bail!("LoRA on mixture-of-experts MLPs is not supported");
                }
                let rows: usize = seg_rows.iter().sum();
                let inner = modules.len() * rank;
                let mut a_all = vec![0.0; n * inner * cols];
                let mut b_all = vec![0.0; n * rows * rank];
                // Granite's residual multiplier is folded into o and down;
                // their updates scale the same way.
                let mult = if slot % 2 == 1 {
                    c.residual_multiplier
                } else {
                    1.0
                };
                for (id, ad) in list.iter().enumerate() {
                    let mut row0 = 0;
                    for (s, (&m, &seg)) in modules.iter().zip(seg_rows).enumerate() {
                        let found = ad.weights.iter().find(|p| p.layer == l && p.module == m);
                        if let Some(Projection { a, b, .. }) = found {
                            if a.len() != ad.rank * cols || b.len() != seg * ad.rank {
                                bail!(
                                    "adapter {}: layer {l} {} does not fit this model",
                                    ad.name,
                                    MODULES[m]
                                );
                            }
                            for j in 0..ad.rank {
                                let dst = (id * inner + s * rank + j) * cols;
                                a_all[dst..dst + cols]
                                    .copy_from_slice(&a[j * cols..(j + 1) * cols]);
                            }
                            for r in 0..seg {
                                for j in 0..ad.rank {
                                    b_all[(id * rows + row0 + r) * rank + j] =
                                        b[r * ad.rank + j] * mult;
                                }
                            }
                        }
                        row0 += seg;
                    }
                }
                out[slot] = Some(Lora {
                    rows,
                    cols: *cols,
                    rank,
                    ends: seg_rows
                        .iter()
                        .scan(0, |end, &r| {
                            *end += r;
                            Some(*end)
                        })
                        .collect(),
                    a: a_all.iter().map(|&v| f32_to_bf16(v)).collect(),
                    b: b_all.iter().map(|&v| f32_to_bf16(v)).collect(),
                });
            }
            let [qkv, o, gate_up, down] = out;
            layers.push(LayerLora {
                qkv,
                o,
                gate_up,
                down,
            });
        }
        if let Some(a) = list
            .iter()
            .find(|a| a.weights.iter().any(|p| p.layer >= c.layers))
        {
            bail!("adapter {} has more layers than the model", a.name);
        }
        Ok(Adapters {
            names: list.into_iter().map(|a| a.name).collect(),
            rank,
            layers,
        })
    }

    /// The id of the adapter called `name` (from 1).
    pub fn id(&self, name: &str) -> Option<u32> {
        self.names
            .iter()
            .position(|n| n == name)
            .map(|i| i as u32 + 1)
    }

    pub fn bytes(&self) -> usize {
        self.layers
            .iter()
            .flat_map(|l| [&l.qkv, &l.o, &l.gate_up, &l.down])
            .flatten()
            .map(|m| (m.a.len() + m.b.len()) * 2)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applies_each_segment_with_its_own_b() {
        // Two segments of 2 and 1 rows, rank 1, 2 columns; two adapters.
        let m = Lora {
            rows: 3,
            cols: 2,
            rank: 1,
            ends: vec![2, 3],
            a: [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]
                .map(f32_to_bf16)
                .to_vec(),
            b: [1.0, 10.0, 100.0, 9.0, 9.0, 9.0].map(f32_to_bf16).to_vec(),
        };
        let x = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let mut y = [0.5; 9];
        m.apply_batch(&Pool::new(1), &[1, 0, 2], &x, &mut y);
        // Token 0: h = [A0 x, A1 x] = [3, 7]; rows 0 and 1 read h0, row 2
        // reads h1. Token 1 has no adapter; token 2's A is zero.
        assert_eq!(y, [3.5, 30.5, 700.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
    }
}
