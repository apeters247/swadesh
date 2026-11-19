//! Parallel-or-sequential helpers, so the core also builds without threads
//! (e.g. `wasm32-unknown-unknown`) via `--no-default-features`.
//!
//! Fork safety: rayon's *global* pool cannot survive `fork()`: a child process
//! inherits the pool's bookkeeping but not its worker threads, so the first
//! parallel call in the child waits forever. (Python's `multiprocessing` with
//! the `fork` start method, which sacreBLEU's `PairedTest(n_jobs>1)` uses, hits
//! this.) We therefore run all parallel work in a pool owned by the current
//! process id and build a fresh one after a fork.

#[cfg(feature = "parallel")]
mod pool {
    use std::sync::{Arc, Mutex};

    static POOL: Mutex<Option<(u32, Arc<rayon::ThreadPool>)>> = Mutex::new(None);

    pub fn get() -> Arc<rayon::ThreadPool> {
        let pid = std::process::id();
        // The lock is held only to clone an Arc, so a fork can only inherit it
        // locked in a vanishingly small window; poisoning is recovered from.
        let mut guard = POOL.lock().unwrap_or_else(|p| p.into_inner());
        match guard.as_ref() {
            Some((p, pool)) if *p == pid => pool.clone(),
            _ => {
                // First use in this process (or first use after a fork).
                let pool = Arc::new(rayon::ThreadPoolBuilder::new().build().expect("rayon pool"));
                *guard = Some((pid, pool.clone()));
                pool
            }
        }
    }
}

/// Run `f` inside this process's thread pool (no-op without `parallel`).
#[inline]
pub fn install<R: Send, F: FnOnce() -> R + Send>(f: F) -> R {
    #[cfg(feature = "parallel")]
    {
        pool::get().install(f)
    }
    #[cfg(not(feature = "parallel"))]
    {
        f()
    }
}

/// Number of worker threads that parallel helpers will use.
#[inline]
pub fn num_threads() -> usize {
    #[cfg(feature = "parallel")]
    {
        pool::get().current_num_threads()
    }
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
}

/// `slice.(par_)chunks_mut(size).enumerate().for_each(f)`.
#[inline]
pub fn for_each_chunk<T, F>(slice: &mut [T], size: usize, f: F)
where
    T: Send,
    F: Fn((usize, &mut [T])) + Sync + Send,
{
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        install(|| slice.par_chunks_mut(size).enumerate().for_each(f));
    }
    #[cfg(not(feature = "parallel"))]
    {
        slice.chunks_mut(size).enumerate().for_each(f);
    }
}
