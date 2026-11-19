//! Runs in its own process with a one-thread rayon pool, so the single-threaded
//! bootstrap path (taken when `num_threads() <= 1`) is the one under test.

use swadesh_core::nprng::default_rng;
use swadesh_core::resample::{bootstrap_sums_f32, bootstrap_sums_f32_seeded};

#[test]
fn single_thread_bootstrap_equals_explicit_index_draws() {
    // Must be set before the pool is first built in this process.
    std::env::set_var("RAYON_NUM_THREADS", "1");
    assert_eq!(swadesh_core::par::num_threads(), 1);

    let (n, n_samples, width, seed) = (5usize, 7usize, 3usize, 12345u128);
    let stats: Vec<f32> = (0..n * width).map(|i| i as f32 * 0.5 + 1.0).collect();

    // The seeded path draws `next_below(n)` per index, row-major. Reproduce that stream
    // explicitly and sum the rows with the index-based kernel.
    let mut g = default_rng(seed);
    let idxs: Vec<i64> = (0..n_samples * n).map(|_| g.next_below(n as u64) as i64).collect();
    let expected = bootstrap_sums_f32(&stats, width, &idxs, n);

    assert_eq!(bootstrap_sums_f32_seeded(&stats, width, n, n_samples, seed), expected);
}
