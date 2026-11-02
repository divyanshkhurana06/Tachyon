use crate::config::Config;
use crate::kv::{Batch, Logits, BLOCK};

/// A model on some hardware, with a paged KV cache. Runs batches of tokens
/// from any number of sequences in one forward pass.
pub trait Backend: Send {
    fn config(&self) -> &Config;

    /// e.g. "cpu q8 x6" or "metal bf16 (Apple M3)".
    fn name(&self) -> String;

    /// KV cache size, in blocks of [`BLOCK`] positions.
    fn num_blocks(&self) -> usize;

    /// Most tokens one forward pass takes.
    fn max_batch(&self) -> usize;

    /// Runs `batch` and returns logits for its `logit_rows`, one row of
    /// `config().vocab` values each.
    fn forward(&mut self, batch: &Batch) -> &[f32];
}

/// A single sequence: everything generation needs.
pub trait Engine: Send {
    fn config(&self) -> &Config;

    fn name(&self) -> String;

    /// Tokens currently held in the KV cache.
    fn pos(&self) -> usize;

    fn max_seq(&self) -> usize;

    /// Forgets everything after the first `pos` tokens. The cache entries
    /// beyond it are simply overwritten later, so this is free.
    fn truncate(&mut self, pos: usize);

    /// Runs `tokens` at positions `pos()..` and returns the logits for the
    /// last one, or for every one (`tokens.len() x vocab`) when `all`.
    fn forward(&mut self, tokens: &[u32], all: bool) -> &[f32];
}

/// One sequence that owns a whole backend's cache, laid out in order.
pub struct Session {
    backend: Box<dyn Backend>,
    table: Vec<u32>,
    max_seq: usize,
    pos: usize,
    batch: Batch,
}

impl Session {
    pub fn new(backend: Box<dyn Backend>) -> Session {
        let blocks = backend.num_blocks();
        let max_seq = (blocks * BLOCK).min(backend.config().max_positions);
        Session {
            table: (0..blocks as u32).collect(),
            backend,
            max_seq,
            pos: 0,
            batch: Batch::default(),
        }
    }
}

impl Engine for Session {
    fn config(&self) -> &Config {
        self.backend.config()
    }

    fn name(&self) -> String {
        self.backend.name()
    }

    fn pos(&self) -> usize {
        self.pos
    }

    fn max_seq(&self) -> usize {
        self.max_seq
    }

    fn truncate(&mut self, pos: usize) {
        self.pos = self.pos.min(pos);
    }

    fn forward(&mut self, tokens: &[u32], all: bool) -> &[f32] {
        assert!(!tokens.is_empty(), "forward needs at least one token");
        assert!(
            self.pos + tokens.len() <= self.max_seq,
            "context window full"
        );
        let step = self.backend.max_batch();
        let logits = if all { Logits::All } else { Logits::Last };
        // Split oversized calls; only the final chunk's logits are kept,
        // so `all` must fit in one batch.
        assert!(
            !all || tokens.len() <= step,
            "at most {step} tokens when asking for all logits"
        );
        let chunks = tokens.len().div_ceil(step);
        for (i, chunk) in tokens.chunks(step).enumerate() {
            self.batch.clear();
            let want = if i + 1 == chunks {
                logits
            } else {
                Logits::None
            };
            self.batch.push(&self.table, self.pos, chunk, want);
            self.pos += chunk.len();
            if i + 1 < chunks {
                self.backend.forward(&self.batch);
            }
        }
        self.backend.forward(&self.batch)
    }
}
