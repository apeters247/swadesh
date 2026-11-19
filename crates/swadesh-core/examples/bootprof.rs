use std::time::Instant;
use swadesh_core::{nprng, resample};
fn main() {
    let (n, w, s) = (5000usize, 10usize, 1000usize);
    let stats: Vec<f32> = (0..n * w).map(|i| (i % 37) as f32).collect();
    let t = Instant::now();
    let mut g = nprng::default_rng(12345);
    let mut acc = 0u64;
    for _ in 0..n * s { acc = acc.wrapping_add(g.next_below(n as u64)); }
    println!("draw only   {:6.1} ms ({acc})", t.elapsed().as_secs_f64() * 1e3);
    let idx: Vec<i64> = nprng::default_rng(12345).integers(0, n as i64, n * s);
    let t = Instant::now();
    let out = resample::bootstrap_sums_f32(&stats, w, &idx, n);
    println!("sum only    {:6.1} ms ({})", t.elapsed().as_secs_f64() * 1e3, out[0]);
    let t = Instant::now();
    let out = resample::bootstrap_sums_f32_seeded(&stats, w, n, s, 12345);
    println!("fused       {:6.1} ms ({})", t.elapsed().as_secs_f64() * 1e3, out[0]);
}
