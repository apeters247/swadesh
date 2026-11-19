//! Bit-exact checks of the NumPy-2 float32 bootstrap scorers against sacreBLEU 2.6.0.
//! Fixtures: `tools/gen_np2_fixtures.py` (CPython 3.11, NumPy 2), which applies sacreBLEU's own
//! `_compute_score_from_stats` to random float32 rows, including both float32 rounding regimes
//! (values above 2^24), zero totals, every smoothing method and the chrF eps paths.

use swadesh_core::score::{chrf_f_score_np2_f32, compute_bleu_np2_f32, NpVal};
use swadesh_core::{BleuScoreConfig, Smooth, SumMode};

fn fixture(name: &str) -> String {
    std::fs::read_to_string(format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap()
}

#[test]
fn bleu_np2_f32_matches_sacrebleu_bitwise() {
    let mut n = 0;
    for line in fixture("np2_bleu.txt").lines() {
        let f: Vec<&str> = line.split(' ').collect();
        let order: usize = f[0].parse().unwrap();
        let smooth = Smooth::from_name(f[1]).unwrap();
        let sv = if f[2] == "none" { smooth.default_value() } else { f[2].parse().unwrap() };
        let cfg = BleuScoreConfig {
            smooth,
            smooth_value: sv,
            effective_order: f[3] == "1",
            max_order: order,
            sum_mode: SumMode::for_python(3, 11),
        };
        let expected = f64::from_bits(f[4].parse().unwrap());
        let row: Vec<f32> = f[5..].iter().map(|x| f32::from_bits(x.parse().unwrap())).collect();
        let got = compute_bleu_np2_f32(&row, &cfg);
        assert_eq!(got.to_bits(), expected.to_bits(), "line {}: {line}\n got {got:e} want {expected:e}", n + 1);
        n += 1;
    }
    assert!(n >= 1000);
}

#[test]
fn chrf_np2_f32_matches_sacrebleu_bitwise_and_type() {
    let mut n = 0;
    for line in fixture("np2_chrf.txt").lines() {
        let f: Vec<&str> = line.split(' ').collect();
        let order: usize = f[0].parse().unwrap();
        let beta: i64 = f[1].parse().unwrap();
        let eps = f[2] == "1";
        let want_f32 = f[3] == "1";
        let expected = f64::from_bits(f[4].parse().unwrap());
        let row: Vec<f32> = f[5..].iter().map(|x| f32::from_bits(x.parse().unwrap())).collect();
        let got = chrf_f_score_np2_f32(&row, order, beta, eps);
        let (val, is_f32) = match got {
            NpVal::F32(v) => (v as f64, true),
            NpVal::Py(v) => (v, false),
        };
        assert_eq!(is_f32, want_f32, "result type, line {}: {line}", n + 1);
        assert_eq!(val.to_bits(), expected.to_bits(), "line {}: {line}\n got {val:e} want {expected:e}", n + 1);
        n += 1;
    }
    assert!(n >= 1000);
}

#[test]
fn npval_promotion_rules() {
    use NpVal::{Py, F32};
    // Python-with-Python stays float64; anything with a float32 is computed in float32.
    assert_eq!(Py(0.1).plus(Py(0.2)), Py(0.1 + 0.2));
    assert_eq!(F32(0.1).plus(Py(0.2)), F32(0.1f32 + 0.2f64 as f32));
    assert_eq!(Py(0.2).plus(F32(0.1)), F32(0.2f64 as f32 + 0.1f32));
    assert_eq!(F32(3.0).times(Py(1.0 / 3.0)), F32(3.0f32 * (1.0f64 / 3.0) as f32));
    assert_eq!(Py(1.0).over(Py(3.0)), Py(1.0 / 3.0));
    assert_eq!(F32(1.0).over(F32(3.0)), F32(1.0f32 / 3.0f32));
    assert_eq!(Py(7.0).times(Py(3.0)), Py(21.0));
    assert_eq!(F32(1.5).as_f64(), 1.5);
    assert_eq!(Py(-2.25).as_f64(), -2.25);
}
