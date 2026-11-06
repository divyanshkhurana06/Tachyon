//! Continuous batching (Orca, vLLM): many requests share one model.
//!
//! A single thread owns the backend and loops: every step it packs one
//! pending token from each generating sequence, then chunks of new prompts
//! up to the batch budget, into one forward pass. Requests join and leave
//! between steps, so a long reply never holds up a short one, and the
//! weights read for one step serve every sequence in it. Decoding is bound
//! by reading weights, so batching N sequences costs far less than N times
//! one.
//!
//! Prefill and decode are the same thing here: a sequence has tokens that
//! are not in the cache yet (the whole prompt at first, then the last
//! sampled token) and feeds as many as the budget allows.
//!
//! When the KV cache runs out, the least urgent sequence not yet in the step
//! (lowest priority, then latest to arrive) is preempted: its keys and
//! values are copied out to host memory, its blocks released, and it
//! re-queues ahead of its priority class, to be copied back in when there is
//! room. Without swap space left it is recomputed instead; its full blocks
//! stay in the prefix cache, so that is often nearly free too.
//!
//! LoRA adapters are loaded and unloaded through the same channel as
//! requests, between steps, and requests name their adapter, resolved when
//! they arrive: so a request always runs with the adapter it named, or is
//! refused.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use crate::config::Config;
use crate::constrain::Constraint;
use crate::engine::{Backend, SKIP};
use crate::generate::Finish;
use crate::kv::{Batch, BlockPool, Logits, Seq};
use crate::lora::{Adapter, Adapters};
use crate::sampler::{Sampler, SamplerConfig, TokenLogprob};
use crate::tokenizer::{Tokenizer, Utf8Stream};

/// Threads used to sample a big batch's next tokens.
const SAMPLE_THREADS: usize = 4;

pub struct Request {
    pub prompt: Vec<u32>,
    pub max_tokens: usize,
    pub sampler: SamplerConfig,
    pub stop_tokens: Vec<u32>,
    /// Restricts which tokens may be sampled (e.g. JSON mode).
    pub constraint: Option<Box<dyn Constraint>>,
    pub events: Sender<Event>,
    /// Set by the caller to abandon the request (e.g. the client hung up).
    pub cancel: Arc<AtomicBool>,
    /// Report each token's log-probability and this many alternatives.
    pub logprobs: Option<usize>,
    /// The LoRA adapter to run with, by name (see `lora.rs`); `None` for
    /// the base model.
    pub adapter: Option<String>,
    /// Lower runs first, as in vLLM; ties go to the earlier request.
    pub priority: i32,
    /// Special tokens whose text is part of the reply (Granite's
    /// `<|tool_call|>`); others are left out.
    pub visible: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct Summary {
    pub prompt_tokens: usize,
    /// Prompt tokens found in the prefix cache.
    pub cached_tokens: usize,
    pub completion_tokens: usize,
    pub finish: Finish,
    /// Arrival to first generated token.
    pub ttft_secs: f64,
    /// Generation speed this request saw after its first token.
    pub decode_tps: f64,
    /// Sequences in the batch, on average, while this request was decoding.
    pub mean_batch: f64,
}

#[derive(Debug)]
pub enum Event {
    /// The log-probability of the next generated token (only when the
    /// request asked), sent just before that token's text.
    Logprob(TokenLogprob),
    Text(String),
    Done(Summary),
    Error(String),
}

#[derive(Debug, Clone)]
pub struct SchedulerOptions {
    /// Most sequences in flight at once.
    pub max_seqs: usize,
    /// Longest a sequence (prompt plus reply) may get.
    pub max_seq_len: usize,
    /// Host memory for preempted sequences' caches, in tokens; 0 always
    /// recomputes them instead.
    pub swap_tokens: usize,
}

/// Live numbers, for `/info` and the load test.
#[derive(Debug, Default)]
pub struct Stats {
    pub running: std::sync::atomic::AtomicUsize,
    pub waiting: std::sync::atomic::AtomicUsize,
    pub steps: std::sync::atomic::AtomicU64,
    pub tokens: std::sync::atomic::AtomicU64,
    /// Time inside the model's forward pass, and in whole steps.
    pub forward_ns: std::sync::atomic::AtomicU64,
    pub step_ns: std::sync::atomic::AtomicU64,
    /// Sequences preempted, and of those, swapped out rather than recomputed.
    pub preemptions: std::sync::atomic::AtomicU64,
    pub swaps: std::sync::atomic::AtomicU64,
}

/// What the scheduler thread receives.
enum Message {
    Request(Request),
    Load(Adapter, Sender<Result<(), String>>),
    Unload(String, Sender<Result<(), String>>),
}

pub struct Scheduler {
    tx: Sender<Message>,
    pub stats: Arc<Stats>,
    /// The adapters loaded now, in id order.
    adapters: Arc<std::sync::Mutex<Vec<String>>>,
}

impl Scheduler {
    pub fn spawn(
        backend: Box<dyn Backend>,
        tok: Arc<Tokenizer>,
        opts: SchedulerOptions,
    ) -> Scheduler {
        let (tx, rx) = mpsc::channel();
        let stats = Arc::new(Stats::default());
        let adapters = Arc::default();
        let (st, names) = (stats.clone(), Arc::clone(&adapters));
        thread::Builder::new()
            .name("scheduler".into())
            .spawn(move || Worker::new(backend, tok, opts, st, names).run(rx))
            .expect("spawn scheduler");
        Scheduler {
            tx,
            stats,
            adapters,
        }
    }

    pub fn submit(&self, req: Request) {
        if let Err(mpsc::SendError(Message::Request(req))) = self.tx.send(Message::Request(req)) {
            let _ = req
                .events
                .send(Event::Error("the scheduler has stopped".into()));
        }
    }

    /// Adds an adapter that requests may then name, between two steps.
    pub fn load_adapter(&self, adapter: Adapter) -> Result<(), String> {
        let (tx, rx) = mpsc::channel();
        let _ = self.tx.send(Message::Load(adapter, tx));
        rx.recv()
            .unwrap_or_else(|_| Err("the scheduler has stopped".into()))
    }

    /// Removes an adapter no queued or running request uses.
    pub fn unload_adapter(&self, name: &str) -> Result<(), String> {
        let (tx, rx) = mpsc::channel();
        let _ = self.tx.send(Message::Unload(name.to_string(), tx));
        rx.recv()
            .unwrap_or_else(|_| Err("the scheduler has stopped".into()))
    }

    /// The names of the adapters loaded now.
    pub fn adapters(&self) -> Vec<String> {
        self.adapters.lock().unwrap().clone()
    }
}

struct Sequence {
    req: Request,
    seq: Option<Seq>,
    /// Prompt plus everything generated: what must be in the cache before
    /// the next token can be sampled.
    tokens: Vec<u32>,
    prompt_len: usize,
    cached: Option<usize>,
    sampler: Sampler,
    text: Utf8Stream,
    arrived: Instant,
    first_token: Option<Instant>,
    batch_sum: usize,
    batch_steps: usize,
    /// The token just sampled, before it is checked and appended.
    picked: u32,
    picked_logprob: Option<TokenLogprob>,
    /// A preempted sequence's cache, while it waits to be resumed.
    swapped: Option<Swapped>,
    /// Its adapter's id in the batch (0: none), and the salt that keeps its
    /// prefix cache apart (unique to each adapter ever loaded, so a reused
    /// id never meets another adapter's cached blocks).
    adapter: u32,
    salt: u32,
}

/// The keys and values of a sequence's first `len` positions, out of the
/// cache.
struct Swapped {
    data: Vec<u8>,
    len: usize,
}

impl Sequence {
    fn generated(&self) -> usize {
        self.tokens.len() - self.prompt_len
    }

    fn pending(&self) -> usize {
        self.tokens.len() - self.seq.as_ref().map_or(0, |s| s.len)
    }

    fn send(&self, e: Event) -> bool {
        self.req.events.send(e).is_ok()
    }
}

struct Worker {
    backend: Box<dyn Backend>,
    tok: Arc<Tokenizer>,
    opts: SchedulerOptions,
    stats: Arc<Stats>,
    pool: BlockPool,
    running: Vec<Sequence>,
    waiting: VecDeque<Sequence>,
    batch: Batch,
    /// Tokens' worth of swapped-out cache held now.
    swapped_tokens: usize,
    cfg: Config,
    /// Loaded adapters by id - 1 (a slot emptied by unloading is reused by
    /// the next load), and each one's prefix-cache salt.
    adapters: Vec<Option<Adapter>>,
    salts: Vec<u32>,
    next_salt: u32,
    names: Arc<std::sync::Mutex<Vec<String>>>,
}

impl Worker {
    fn new(
        backend: Box<dyn Backend>,
        tok: Arc<Tokenizer>,
        opts: SchedulerOptions,
        stats: Arc<Stats>,
        names: Arc<std::sync::Mutex<Vec<String>>>,
    ) -> Worker {
        let pool = BlockPool::new(backend.num_blocks());
        let cfg = backend.config().clone();
        Worker {
            backend,
            tok,
            opts,
            stats,
            pool,
            running: Vec::new(),
            waiting: VecDeque::new(),
            batch: Batch::default(),
            swapped_tokens: 0,
            cfg,
            adapters: Vec::new(),
            salts: Vec::new(),
            next_salt: 1,
            names,
        }
    }

    fn run(mut self, rx: Receiver<Message>) {
        loop {
            // Sleep only when there is nothing at all to do.
            if self.running.is_empty() && self.waiting.is_empty() {
                match rx.recv() {
                    Ok(m) => self.receive(m),
                    Err(_) => return,
                }
            }
            while let Ok(m) = rx.try_recv() {
                self.receive(m);
            }
            if self.running.is_empty() && self.waiting.is_empty() {
                continue;
            }
            let started = Instant::now();
            self.drop_cancelled();
            self.admit();
            self.step();
            self.stats
                .step_ns
                .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
            self.stats
                .running
                .store(self.running.len(), Ordering::Relaxed);
            self.stats
                .waiting
                .store(self.waiting.len(), Ordering::Relaxed);
        }
    }

    fn receive(&mut self, m: Message) {
        match m {
            Message::Request(req) => self.enqueue(req),
            Message::Load(adapter, reply) => {
                let _ = reply.send(self.load(adapter));
            }
            Message::Unload(name, reply) => {
                let _ = reply.send(self.unload(&name));
            }
        }
    }

    fn slot(&self, name: &str) -> Option<usize> {
        self.adapters
            .iter()
            .position(|a| a.as_ref().is_some_and(|a| a.name == name))
    }

    /// Rebuilds the backend's adapters from the slots (ranks are padded to
    /// the largest, so all of them change together), and the list of names.
    fn install(&mut self) -> Result<(), String> {
        let built = Adapters::build(&self.cfg, &self.adapters).map_err(|e| e.to_string())?;
        self.backend
            .load_adapters(built)
            .map_err(|e| e.to_string())?;
        *self.names.lock().unwrap() = self
            .adapters
            .iter()
            .flatten()
            .map(|a| a.name.clone())
            .collect();
        Ok(())
    }

    fn load(&mut self, adapter: Adapter) -> Result<(), String> {
        if self.slot(&adapter.name).is_some() {
            return Err(format!("adapter {:?} is already loaded", adapter.name));
        }
        let i = match self.adapters.iter().position(Option::is_none) {
            Some(i) => i,
            None => {
                self.adapters.push(None);
                self.salts.push(0);
                self.adapters.len() - 1
            }
        };
        self.adapters[i] = Some(adapter);
        self.salts[i] = self.next_salt;
        self.next_salt += 1;
        self.install().inspect_err(|_| {
            // Back to what was there; that built before.
            self.adapters[i] = None;
            let _ = self.install();
        })
    }

    fn unload(&mut self, name: &str) -> Result<(), String> {
        let i = self
            .slot(name)
            .ok_or_else(|| format!("adapter {name:?} is not loaded"))?;
        let id = i as u32 + 1;
        let users = self
            .running
            .iter()
            .chain(&self.waiting)
            .filter(|s| s.adapter == id)
            .count();
        if users > 0 {
            return Err(format!("adapter {name:?} is in use by {users} requests"));
        }
        self.adapters[i] = None;
        while self.adapters.last().is_some_and(Option::is_none) {
            self.adapters.pop();
            self.salts.pop();
        }
        self.install()
    }

    fn enqueue(&mut self, req: Request) {
        let limit = self
            .opts
            .max_seq_len
            .min(self.backend.num_blocks() * crate::kv::BLOCK);
        if req.prompt.is_empty() || req.prompt.len() >= limit {
            let _ = req.events.send(Event::Error(format!(
                "the prompt is {} tokens; the limit is {}",
                req.prompt.len(),
                limit - 1
            )));
            return;
        }
        let (adapter, salt) = match &req.adapter {
            None => (0, 0),
            Some(name) => match self.slot(name) {
                Some(i) => (i as u32 + 1, self.salts[i]),
                None => {
                    let msg = format!("adapter {name:?} is not loaded");
                    let _ = req.events.send(Event::Error(msg));
                    return;
                }
            },
        };
        let sampler = Sampler::new(req.sampler.clone());
        let s = Sequence {
            tokens: req.prompt.clone(),
            prompt_len: req.prompt.len(),
            req,
            seq: None,
            cached: None,
            sampler,
            text: Utf8Stream::default(),
            arrived: Instant::now(),
            first_token: None,
            batch_sum: 0,
            batch_steps: 0,
            picked: 0,
            picked_logprob: None,
            swapped: None,
            adapter,
            salt,
        };
        self.queue(s, false);
    }

    /// Queues a sequence by priority: behind the others of its priority, or
    /// with `first` (a preempted one), ahead of them.
    fn queue(&mut self, s: Sequence, first: bool) {
        let p = s.req.priority;
        let at = self
            .waiting
            .iter()
            .position(|w| w.req.priority > p || (first && w.req.priority == p))
            .unwrap_or(self.waiting.len());
        self.waiting.insert(at, s);
    }

    fn drop_cancelled(&mut self) {
        let mut freed = 0;
        self.waiting.retain(|s| {
            let keep = !s.req.cancel.load(Ordering::Relaxed);
            if !keep {
                freed += s.swapped.as_ref().map_or(0, |w| w.len);
            }
            keep
        });
        self.swapped_tokens -= freed;
        let mut i = 0;
        while i < self.running.len() {
            if self.running[i].req.cancel.load(Ordering::Relaxed) {
                let s = self.running.swap_remove(i);
                self.finish(s, Finish::Cancelled);
            } else {
                i += 1;
            }
        }
    }

    /// Moves queued requests into the running set while there is room for
    /// their next chunk, so a freshly preempted sequence isn't readmitted
    /// only to be preempted again.
    fn admit(&mut self) {
        while self.running.len() < self.opts.max_seqs {
            let Some(s) = self.waiting.front() else { break };
            let restore = s.swapped.as_ref().map_or(0, |w| w.len);
            let chunk = (s.tokens.len() - restore).min(self.backend.max_batch());
            if !self.running.is_empty()
                && self.pool.available() < (restore + chunk).div_ceil(crate::kv::BLOCK) + 1
            {
                break;
            }
            let mut s = self.waiting.pop_front().unwrap();
            let seq = match s.swapped.take() {
                // Swapped out: blocks still in the prefix cache are taken
                // from there, and the rest copied back in.
                Some(w) => {
                    let mut seq = self.pool.start_with(&s.tokens[..w.len], s.salt);
                    let cached = seq.blocks.len();
                    if !self.pool.reserve(&mut seq, w.len) {
                        self.pool.release(seq);
                        s.swapped = Some(w);
                        self.waiting.push_front(s);
                        break;
                    }
                    let mut blocks = seq.blocks[..w.len.div_ceil(crate::kv::BLOCK)].to_vec();
                    blocks[..cached].fill(SKIP);
                    self.backend.load_blocks(&blocks, &w.data);
                    seq.len = w.len;
                    self.swapped_tokens -= w.len;
                    seq
                }
                None => {
                    let seq = self.pool.start_with(&s.tokens, s.salt);
                    s.cached.get_or_insert(seq.len);
                    seq
                }
            };
            s.seq = Some(seq);
            self.running.push(s);
        }
    }

    /// Releases a sequence's blocks; full ones stay in the prefix cache.
    fn release(&mut self, s: &mut Sequence) {
        if let Some(mut seq) = s.seq.take() {
            self.pool.commit(&mut seq, &s.tokens);
            self.pool.release(seq);
        }
    }

    /// Frees room by sending a sequence after `after` back to the queue:
    /// only those not yet in this step's batch are candidates, and of them
    /// the least urgent goes. Its cache is swapped out if there is room for
    /// it, or else recomputed later.
    fn preempt(&mut self, after: usize) -> bool {
        if self.running.len() <= after + 1 {
            return false;
        }
        let victim = (after + 1..self.running.len())
            .max_by_key(|&j| (self.running[j].req.priority, self.running[j].arrived))
            .unwrap();
        let mut s = self.running.remove(victim);
        self.stats.preemptions.fetch_add(1, Ordering::Relaxed);
        let len = s.seq.as_ref().map_or(0, |q| q.len);
        if len > 0 && self.swapped_tokens + len <= self.opts.swap_tokens {
            let seq = s.seq.as_ref().unwrap();
            let blocks = &seq.blocks[..len.div_ceil(crate::kv::BLOCK)];
            if let Some(data) = self.backend.save_blocks(blocks) {
                self.swapped_tokens += len;
                s.swapped = Some(Swapped { data, len });
                self.stats.swaps.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.release(&mut s);
        self.queue(s, true);
        true
    }

    fn step(&mut self) {
        // By priority; within one, decoding sequences (one pending token)
        // go first, to keep replies flowing, and prompts fill the rest of
        // the budget.
        self.running
            .sort_by_key(|s| (s.req.priority, s.pending() > 1));
        let budget = self.backend.max_batch();
        self.batch.clear();
        let mut rows: Vec<usize> = Vec::new(); // running index per logits row
        let mut i = 0;
        while i < self.running.len() && self.batch.len() < budget {
            let s = &self.running[i];
            let fed = s.seq.as_ref().map_or(0, |q| q.len);
            let take = s.pending().min(budget - self.batch.len());
            let mut seq = self.running[i]
                .seq
                .take()
                .expect("running sequences hold blocks");
            if !self.pool.reserve(&mut seq, fed + take) {
                self.running[i].seq = Some(seq);
                if self.preempt(i) {
                    continue; // retry this sequence with the freed blocks
                }
                if !self.batch.is_empty() {
                    break; // earlier sequences hold the cache; this one waits a step
                }
                // Alone and still too big for the cache.
                let mut s = self.running.remove(i);
                self.release(&mut s);
                let _ = s.send(Event::Error(
                    "the KV cache is too small for this request".into(),
                ));
                continue;
            }
            let s = &mut self.running[i];
            let done_prompt = fed + take == s.tokens.len();
            self.batch.push_with(
                &seq.blocks,
                fed,
                &s.tokens[fed..fed + take],
                if done_prompt {
                    Logits::Last
                } else {
                    Logits::None
                },
                s.adapter,
            );
            seq.len = fed + take;
            s.seq = Some(seq);
            if done_prompt {
                rows.push(i);
            }
            i += 1;
        }
        if self.batch.is_empty() {
            return;
        }

        let vocab = self.backend.config().vocab;
        let usable = self.tok.len().min(vocab);
        let started = Instant::now();
        let logits = self.backend.forward(&self.batch);
        self.stats
            .forward_ns
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        self.stats.steps.fetch_add(1, Ordering::Relaxed);
        self.stats
            .tokens
            .fetch_add(rows.len() as u64, Ordering::Relaxed);
        let decoding = rows.len();

        // Pick every sequence's next token, spread over a few threads: each
        // pick scans 150k logits, which adds up across a big batch.
        let tok = &*self.tok;
        let mut work: Vec<(usize, &mut Sequence)> = Vec::with_capacity(rows.len());
        let mut by_index: Vec<(usize, usize)> =
            rows.iter().enumerate().map(|(row, &i)| (i, row)).collect();
        by_index.sort_unstable();
        let mut wanted = by_index.into_iter().peekable();
        for (i, s) in self.running.iter_mut().enumerate() {
            if wanted.peek().is_some_and(|&(j, _)| j == i) {
                work.push((wanted.next().unwrap().1, s));
            }
        }
        let sample = |(row, s): &mut (usize, &mut Sequence)| {
            let logits = &logits[*row * vocab..*row * vocab + usable];
            let recent = &s.tokens[s.tokens.len().saturating_sub(64).max(s.prompt_len)..];
            s.picked = match &s.req.constraint {
                Some(c) => s
                    .sampler
                    .sample_with(logits, usable, recent, &|t| c.allows(tok, t)),
                None => s.sampler.sample(logits, usable, recent),
            };
            s.picked_logprob = s
                .req
                .logprobs
                .map(|n| crate::sampler::logprobs(logits, s.picked, n));
        };
        let threads = work.len().div_ceil(4).min(SAMPLE_THREADS);
        if threads <= 1 {
            work.iter_mut().for_each(sample);
        } else {
            let per = work.len().div_ceil(threads);
            thread::scope(|scope| {
                for part in work.chunks_mut(per) {
                    scope.spawn(|| part.iter_mut().for_each(sample));
                }
            });
        }

        let mut finished: Vec<(usize, Finish)> = Vec::new();
        for &idx in &rows {
            let s = &mut self.running[idx];
            let next = s.picked;
            s.first_token.get_or_insert_with(Instant::now);
            s.batch_sum += decoding;
            s.batch_steps += 1;

            let ended = s.req.constraint.as_ref().is_some_and(|c| c.is_complete());
            if s.req.stop_tokens.contains(&next) || ended {
                finished.push((idx, Finish::Stop));
                continue;
            }
            s.tokens.push(next);
            if let Some(lp) = s.picked_logprob.take() {
                if !s.send(Event::Logprob(lp)) {
                    finished.push((idx, Finish::Cancelled));
                    continue;
                }
            }
            if let Some(c) = &mut s.req.constraint {
                c.advance(self.tok.token_bytes(next));
            }
            if !self.tok.is_special(next) || s.req.visible.contains(&next) {
                let piece = s.text.push(self.tok.token_bytes(next));
                if !piece.is_empty() && !s.send(Event::Text(piece)) {
                    finished.push((idx, Finish::Cancelled));
                    continue;
                }
            }
            if s.generated() >= s.req.max_tokens || s.tokens.len() >= self.opts.max_seq_len {
                finished.push((idx, Finish::Length));
            }
        }
        for s in self.running.iter_mut() {
            if let Some(seq) = s.seq.as_mut() {
                self.pool.commit(seq, &s.tokens);
            }
        }
        // Remove from the back so earlier indices stay valid.
        finished.sort_by_key(|&(i, _)| std::cmp::Reverse(i));
        for (idx, why) in finished {
            let s = self.running.swap_remove(idx);
            self.finish(s, why);
        }
    }

    fn finish(&mut self, mut s: Sequence, why: Finish) {
        let rest = s.text.finish();
        if !rest.is_empty() {
            s.send(Event::Text(rest));
        }
        self.release(&mut s);
        let now = Instant::now();
        let first = s.first_token.unwrap_or(now);
        let gen = s.generated();
        let summary = Summary {
            prompt_tokens: s.prompt_len,
            cached_tokens: s.cached.unwrap_or(0),
            completion_tokens: gen,
            finish: why,
            ttft_secs: (first - s.arrived).as_secs_f64(),
            decode_tps: gen.saturating_sub(1) as f64 / (now - first).as_secs_f64().max(1e-9),
            mean_batch: s.batch_sum as f64 / s.batch_steps.max(1) as f64,
        };
        s.send(Event::Done(summary));
    }
}
