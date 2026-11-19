//! Parallel-or-sequential helpers, so the core also builds without threads
//! (e.g. `wasm32-unknown-unknown`) via `--no-default-features`.

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
        slice.par_chunks_mut(size).enumerate().for_each(f);
    }
    #[cfg(not(feature = "parallel"))]
    {
        slice.chunks_mut(size).enumerate().for_each(f);
    }
}
