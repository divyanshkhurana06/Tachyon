//! A persistent thread pool for many tiny parallel jobs.
//!
//! One token through a 24-layer model is ~150 parallel matrix multiplies,
//! each taking tens of microseconds. Spawning threads or parking on a
//! condvar per job would cost about as much as the work, so idle workers
//! spin briefly before sleeping and a dispatch usually starts within a
//! microsecond.

use std::cell::UnsafeCell;
use std::ops::Range;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::*};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

type Job = dyn Fn(usize) + Sync;

/// Spin iterations (a few hundred microseconds) before a worker sleeps.
const SPIN: u32 = 1 << 17;

struct Shared {
    /// The running job. Written only while no worker is reading it.
    job: UnsafeCell<Option<&'static Job>>,
    generation: AtomicUsize,
    remaining: AtomicUsize,
    sleepers: AtomicUsize,
    shutdown: AtomicBool,
    lock: Mutex<()>,
    wake: Condvar,
}

unsafe impl Sync for Shared {}

pub struct Pool {
    shared: Arc<Shared>,
    workers: Vec<thread::JoinHandle<()>>,
    busy: Mutex<()>,
}

impl Pool {
    pub fn new(threads: usize) -> Pool {
        let shared = Arc::new(Shared {
            job: UnsafeCell::new(None),
            generation: AtomicUsize::new(0),
            remaining: AtomicUsize::new(0),
            sleepers: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            lock: Mutex::new(()),
            wake: Condvar::new(),
        });
        let workers = (1..threads.max(1))
            .map(|idx| {
                let s = shared.clone();
                thread::Builder::new()
                    .name(format!("tachyon-{idx}"))
                    .spawn(move || worker(&s, idx))
                    .unwrap()
            })
            .collect();
        Pool {
            shared,
            workers,
            busy: Mutex::new(()),
        }
    }

    /// Leaves some cores free. Workers spin while waiting, so if every core
    /// is busy, any other process preempts one and each multiply then waits
    /// for it: on an M3, 8 threads decode at half the speed of 6.
    /// On Apple Silicon: all performance cores plus half the efficiency cores.
    pub fn default_threads() -> usize {
        if cfg!(target_arch = "wasm32") {
            return 1; // the browser build has no threads
        }
        let all = thread::available_parallelism().map_or(4, |n| n.get());
        #[cfg(target_os = "macos")]
        if let (Some(p), Some(e)) = (
            sysctl("hw.perflevel0.physicalcpu"),
            sysctl("hw.perflevel1.physicalcpu"),
        ) {
            return (p + e / 2).clamp(1, all);
        }
        if all > 4 {
            all - 2
        } else {
            all
        }
    }

    pub fn threads(&self) -> usize {
        self.workers.len() + 1
    }

    /// Runs `f(i)` once on each thread `i` in `0..threads()`; the caller is 0.
    pub fn broadcast<F: Fn(usize) + Sync>(&self, f: F) {
        if self.workers.is_empty() {
            return f(0);
        }
        let _busy = self.busy.lock().unwrap();
        let s = &*self.shared;
        let job: &(dyn Fn(usize) + Sync + '_) = &f;
        // SAFETY: `Wait` below blocks until every worker has finished with
        // `job`, so it never outlives `f` even if `f` panics on this thread.
        let job: &'static Job = unsafe { std::mem::transmute(job) };
        unsafe { *s.job.get() = Some(job) };
        s.remaining.store(self.workers.len(), SeqCst);
        s.generation.fetch_add(1, SeqCst);
        if s.sleepers.load(SeqCst) > 0 {
            let _g = s.lock.lock().unwrap();
            s.wake.notify_all();
        }

        struct Wait<'a>(&'a Shared);
        impl Drop for Wait<'_> {
            fn drop(&mut self) {
                while self.0.remaining.load(Acquire) != 0 {
                    std::hint::spin_loop();
                }
                unsafe { *self.0.job.get() = None };
            }
        }
        let _wait = Wait(s);
        f(0);
    }

    /// Calls `f` on chunks of `0..n` of about `grain` items, spread over the
    /// threads. Chunks are handed out dynamically, so slower efficiency cores
    /// simply take fewer of them.
    pub fn parallel_for(&self, n: usize, grain: usize, f: impl Fn(Range<usize>) + Sync) {
        let grain = grain.max(1);
        if n <= grain || self.workers.is_empty() {
            if n > 0 {
                f(0..n);
            }
            return;
        }
        let next = AtomicUsize::new(0);
        self.broadcast(|_| loop {
            let start = next.fetch_add(grain, Relaxed);
            if start >= n {
                break;
            }
            f(start..(start + grain).min(n));
        });
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, SeqCst);
        {
            let _g = self.shared.lock.lock().unwrap();
            self.shared.wake.notify_all();
        }
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

#[cfg(target_os = "macos")]
fn sysctl(name: &str) -> Option<usize> {
    use std::ffi::{c_char, c_int, c_void, CString};
    extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            old: *mut c_void,
            oldlen: *mut usize,
            new: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }
    let name = CString::new(name).ok()?;
    let mut value: i32 = 0;
    let mut len = std::mem::size_of::<i32>();
    let r = unsafe {
        sysctlbyname(
            name.as_ptr(),
            &mut value as *mut i32 as *mut c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (r == 0 && value > 0).then_some(value as usize)
}

fn worker(s: &Shared, idx: usize) {
    let mut seen = 0;
    loop {
        let mut spins = 0;
        loop {
            let g = s.generation.load(Acquire);
            if g != seen {
                seen = g;
                break;
            }
            if s.shutdown.load(Relaxed) {
                return;
            }
            if spins < SPIN {
                spins += 1;
                std::hint::spin_loop();
                continue;
            }
            let mut guard = s.lock.lock().unwrap();
            s.sleepers.fetch_add(1, SeqCst);
            while s.generation.load(SeqCst) == seen && !s.shutdown.load(SeqCst) {
                guard = s.wake.wait(guard).unwrap();
            }
            s.sleepers.fetch_sub(1, SeqCst);
            drop(guard);
            spins = 0;
        }
        let job = unsafe { (*s.job.get()).expect("job is set while a generation runs") };
        if panic::catch_unwind(AssertUnwindSafe(|| job(idx))).is_err() {
            // The caller is spinning on `remaining`; there is no safe way on.
            eprintln!("tachyon: worker thread panicked");
            std::process::abort();
        }
        s.remaining.fetch_sub(1, Release);
    }
}

/// A raw pointer that may be shared across the pool, for jobs that write
/// disjoint parts of one output buffer.
pub struct SyncPtr<T>(*mut T);

impl<T> Clone for SyncPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SyncPtr<T> {}
unsafe impl<T: Send> Send for SyncPtr<T> {}
unsafe impl<T: Send> Sync for SyncPtr<T> {}

impl<T> SyncPtr<T> {
    pub fn new(s: &mut [T]) -> Self {
        SyncPtr(s.as_mut_ptr())
    }

    /// # Safety
    /// No two threads may touch the same elements, and the range must lie
    /// inside the slice this pointer came from.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn slice(&self, start: usize, len: usize) -> &mut [T] {
        std::slice::from_raw_parts_mut(self.0.add(start), len)
    }

    /// # Safety
    /// As for [`SyncPtr::slice`].
    pub unsafe fn write(&self, i: usize, v: T) {
        *self.0.add(i) = v;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_item_is_visited_once() {
        let pool = Pool::new(4);
        for n in [0, 1, 7, 1000, 4097] {
            let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            pool.parallel_for(n, 13, |r| {
                for i in r {
                    hits[i].fetch_add(1, Relaxed);
                }
            });
            assert!(hits.iter().all(|h| h.load(Relaxed) == 1));
        }
    }

    #[test]
    fn survives_sleeping_workers() {
        let pool = Pool::new(3);
        let count = AtomicUsize::new(0);
        for _ in 0..3 {
            pool.broadcast(|_| {
                count.fetch_add(1, Relaxed);
            });
            thread::sleep(std::time::Duration::from_millis(30)); // let workers park
        }
        assert_eq!(count.load(Relaxed), 9);
    }
}
