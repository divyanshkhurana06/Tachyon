use crate::config::Config;

/// A model plus one KV cache: everything generation needs from a backend.
pub trait Engine: Send {
    fn config(&self) -> &Config;

    /// e.g. "cpu q8 x8" or "metal bf16".
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
