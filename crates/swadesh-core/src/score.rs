//! Final score computation from sufficient statistics, replicating CPython's
//! floating point semantics *operation by operation* so results are bit-identical.

/// Which CPython `sum()` algorithm to emulate for float summation.
///
/// Verified empirically against builtin `sum()` (see `tools/check_sum_semantics.py`)
/// and against the CPython source `Python/bltinmodule.c` at v3.12.12, v3.13.12, v3.14.3.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SumMode {
    /// CPython <= 3.11: naive left-to-right `f += x`, ints included.
    Naive,
    /// CPython 3.12 and 3.13: Neumaier compensation for float items. Int items
    /// (in the float phase) are added naively, and the compensation is folded in
    /// once at the end.
    Neumaier,
    /// CPython >= 3.14: the float phase keeps a compensated (hi, lo) accumulator
    /// and int items go through the same compensated add as floats.
    NeumaierInts,
}

impl SumMode {
    /// Maps a CPython `major.minor` to the summation algorithm it uses.
    /// Versions newer than the last verified one (3.14) use `NeumaierInts`.
    pub fn for_python(major: u32, minor: u32) -> Self {
        match (major, minor) {
            (m, n) if (m, n) >= (3, 14) => SumMode::NeumaierInts,
            (m, n) if (m, n) >= (3, 12) => SumMode::Neumaier,
            _ => SumMode::Naive,
        }
    }
}

/// One step of Neumaier's compensated summation (CPython `cs_add` / the float path).
#[inline]
fn neumaier_add(sum: &mut f64, comp: &mut f64, x: f64) {
    let t = *sum + x;
    if sum.abs() >= x.abs() {
        *comp += (*sum - t) + x;
    } else {
        *comp += (x - t) + *sum;
    }
    *sum = t;
}

/// A Python number that is either an `int` or a `float`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PyNum {
    Int(i64),
    Float(f64),
}

impl PyNum {
    #[inline]
    pub fn as_f64(self) -> f64 {
        match self {
            PyNum::Int(i) => i as f64,
            PyNum::Float(f) => f,
        }
    }
}

/// Emulates the builtin `sum(items)` (start = int 0).
pub fn py_sum(items: &[PyNum], mode: SumMode) -> PyNum {
    let mut it = items.iter();
    // Integer fast path.
    let mut i_result: i64 = 0;
    let mut f_result: f64;
    loop {
        match it.next() {
            None => return PyNum::Int(i_result),
            Some(PyNum::Int(v)) => i_result += v,
            Some(PyNum::Float(x)) => {
                f_result = (i_result as f64) + x;
                break;
            }
        }
    }
    let mut c = 0.0f64;
    for item in it {
        match (*item, mode) {
            (PyNum::Float(x), SumMode::Naive) => f_result += x,
            (PyNum::Float(x), _) => neumaier_add(&mut f_result, &mut c, x),
            // Ints in the float phase: naive on 3.12/3.13, compensated on 3.14+.
            (PyNum::Int(v), SumMode::NeumaierInts) => neumaier_add(&mut f_result, &mut c, v as f64),
            (PyNum::Int(v), _) => f_result += v as f64,
        }
    }
    if mode != SumMode::Naive && c != 0.0 && c.is_finite() {
        f_result += c;
    }
    PyNum::Float(f_result)
}

/// sacreBLEU `my_log`.
#[inline]
pub fn my_log(p: f64) -> PyNum {
    if p == 0.0 {
        PyNum::Int(-9999999999)
    } else {
        PyNum::Float(p.ln())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Smooth {
    None,
    Floor,
    AddK,
    Exp,
}

impl Smooth {
    pub fn from_name(s: &str) -> Option<Self> {
        Some(match s {
            "none" => Smooth::None,
            "floor" => Smooth::Floor,
            "add-k" => Smooth::AddK,
            "exp" => Smooth::Exp,
            _ => return None,
        })
    }
    pub fn default_value(self) -> f64 {
        match self {
            Smooth::Floor => 0.1,
            Smooth::AddK => 1.0,
            _ => 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BleuScoreConfig {
    pub smooth: Smooth,
    pub smooth_value: f64,
    pub effective_order: bool,
    pub max_order: usize,
    pub sum_mode: SumMode,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BleuResult {
    pub score: f64,
    pub bp: f64,
    pub precisions: Vec<f64>,
}

/// sacreBLEU `BLEU.compute_bleu`. `stats` = `[sys_len, ref_len, correct.., total..]`.
pub fn compute_bleu(stats: &[f64], cfg: &BleuScoreConfig) -> BleuResult {
    let n = cfg.max_order;
    let sys_len = stats[0];
    let ref_len = stats[1];
    let mut correct: Vec<f64> = stats[2..2 + n].to_vec();
    let mut total: Vec<f64> = stats[2 + n..2 + 2 * n].to_vec();
    let mut bp = 1.0f64;
    if sys_len < ref_len {
        bp = if sys_len > 0.0 { (1.0 - ref_len / sys_len).exp() } else { 0.0 };
    }
    let mut precisions = vec![0.0f64; n];
    if !correct.iter().any(|&c| c != 0.0) {
        return BleuResult { score: 0.0, bp, precisions };
    }
    let mut smooth_mteval = 1.0f64;
    let mut eff_order = n;
    let sv = cfg.smooth_value;
    for k in 1..=n {
        if cfg.smooth == Smooth::AddK && k > 1 {
            correct[k - 1] += sv;
            total[k - 1] += sv;
        }
        if total[k - 1] == 0.0 {
            break;
        }
        if cfg.effective_order {
            eff_order = k;
        }
        if correct[k - 1] == 0.0 {
            match cfg.smooth {
                Smooth::Exp => {
                    smooth_mteval *= 2.0;
                    precisions[k - 1] = 100.0 / (smooth_mteval * total[k - 1]);
                }
                Smooth::Floor => {
                    precisions[k - 1] = 100.0 * sv / total[k - 1];
                }
                _ => {}
            }
        } else {
            precisions[k - 1] = 100.0 * correct[k - 1] / total[k - 1];
        }
    }
    let logs: Vec<PyNum> = precisions[..eff_order].iter().map(|&p| my_log(p)).collect();
    let s = py_sum(&logs, cfg.sum_mode);
    let mean = match s {
        PyNum::Int(i) => (i as f64) / (eff_order as f64),
        PyNum::Float(f) => f / (eff_order as f64),
    };
    BleuResult { score: bp * mean.exp(), bp, precisions }
}

/// sacreBLEU `CHRF._compute_f_score`.
pub fn chrf_f_score(stats: &[i64], order: usize, beta: i64, eps_smoothing: bool) -> f64 {
    let eps = 1e-16f64;
    let mut score = 0.0f64;
    let mut effective_order = 0i64;
    // beta^2 in i128: an i64 product overflows (a panic in debug builds) for
    // |beta| > 3.0e9. Identical to the i64 product wherever that did not overflow.
    let factor = ((beta as i128) * (beta as i128)) as f64;
    let (mut avg_prec, mut avg_rec) = (0.0f64, 0.0f64);
    for i in 0..order {
        let n_hyp = stats[3 * i];
        let n_ref = stats[3 * i + 1];
        let n_match = stats[3 * i + 2] as f64;
        let prec = if n_hyp > 0 { n_match / n_hyp as f64 } else { eps };
        let rec = if n_ref > 0 { n_match / n_ref as f64 } else { eps };
        let denom = factor * prec + rec;
        score += if denom > 0.0 { (1.0 + factor) * prec * rec / denom } else { eps };
        if n_hyp > 0 && n_ref > 0 {
            avg_prec += prec;
            avg_rec += rec;
            effective_order += 1;
        }
    }
    if eps_smoothing {
        return 100.0 * score / order as f64;
    }
    if effective_order == 0 {
        avg_prec = 0.0;
        avg_rec = 0.0;
    } else {
        avg_prec /= effective_order as f64;
        avg_rec /= effective_order as f64;
    }
    if avg_prec + avg_rec != 0.0 {
        let mut s = (1.0 + factor) * avg_prec * avg_rec;
        s /= (factor * avg_prec) + avg_rec;
        100.0 * s
    } else {
        0.0
    }
}

/// sacreBLEU's `BLEU._compute_score_from_stats` applied to a **float32 numpy
/// row** (as in bootstrap resampling), under NumPy >= 2 promotion rules
/// (NEP 50: a Python scalar combined with a `np.float32` is cast to float32 and
/// the operation happens in float32). Returns the score (a Python float).
///
/// Mirrors `BLEU.compute_bleu` line by line:
/// * `sys_len = int(stats[0])`, `ref_len = int(stats[1])`, BP in float64;
/// * add-k: `correct[n] += k` and `total[n] += k` in float32;
/// * exp:   `100. / (smooth_mteval * total[n])`   -> float32 ops;
/// * floor: `100. * k / total[n]` -> `(100.*k)` in float64, then float32 division;
/// * else:  `100. * correct[n] / total[n]`        -> float32 ops;
/// * `my_log(p)` = `math.log(float(p))` in float64, summed with CPython `sum()`.
pub fn compute_bleu_np2_f32(stats: &[f32], cfg: &BleuScoreConfig) -> f64 {
    let n = cfg.max_order;
    let sys_len = stats[0].trunc() as f64;
    let ref_len = stats[1].trunc() as f64;
    let mut correct: Vec<f32> = stats[2..2 + n].to_vec();
    let mut total: Vec<f32> = stats[2 + n..2 + 2 * n].to_vec();
    let mut bp = 1.0f64;
    if sys_len < ref_len {
        bp = if sys_len > 0.0 { (1.0 - ref_len / sys_len).exp() } else { 0.0 };
    }
    if !correct.iter().any(|&c| c != 0.0) {
        return 0.0;
    }
    // precisions start as Python 0.0 floats; set entries become float32.
    let mut precisions = vec![0.0f64; n];
    let mut smooth_mteval = 1.0f64;
    let mut eff_order = n;
    let sv = cfg.smooth_value;
    for k in 1..=n {
        if cfg.smooth == Smooth::AddK && k > 1 {
            correct[k - 1] += sv as f32;
            total[k - 1] += sv as f32;
        }
        if total[k - 1] == 0.0 {
            break;
        }
        if cfg.effective_order {
            eff_order = k;
        }
        if correct[k - 1] == 0.0 {
            match cfg.smooth {
                Smooth::Exp => {
                    smooth_mteval *= 2.0;
                    let denom = (smooth_mteval as f32) * total[k - 1];
                    precisions[k - 1] = (100.0f32 / denom) as f64;
                }
                Smooth::Floor => {
                    let num = (100.0 * sv) as f32;
                    precisions[k - 1] = (num / total[k - 1]) as f64;
                }
                _ => {}
            }
        } else {
            precisions[k - 1] = ((100.0f32 * correct[k - 1]) / total[k - 1]) as f64;
        }
    }
    let logs: Vec<PyNum> = precisions[..eff_order].iter().map(|&p| my_log(p)).collect();
    let s = py_sum(&logs, cfg.sum_mode);
    let mean = match s {
        PyNum::Int(i) => (i as f64) / (eff_order as f64),
        PyNum::Float(f) => f / (eff_order as f64),
    };
    bp * mean.exp()
}

/// A Python number as it flows through sacreBLEU's chrF arithmetic on a
/// float32 numpy row: either a `np.float32` scalar or a Python `float`/`int`.
/// NumPy >= 2 (NEP 50) promotion: Python scalars are "weak", so any operation
/// involving a float32 is done in float32 (the Python operand is first cast to
/// float32); Python-with-Python stays float64.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NpVal {
    F32(f32),
    Py(f64),
}

impl NpVal {
    #[inline]
    fn op(self, o: NpVal, f32op: fn(f32, f32) -> f32, f64op: fn(f64, f64) -> f64) -> NpVal {
        match (self, o) {
            (NpVal::F32(a), NpVal::F32(b)) => NpVal::F32(f32op(a, b)),
            (NpVal::F32(a), NpVal::Py(b)) => NpVal::F32(f32op(a, b as f32)),
            (NpVal::Py(a), NpVal::F32(b)) => NpVal::F32(f32op(a as f32, b)),
            (NpVal::Py(a), NpVal::Py(b)) => NpVal::Py(f64op(a, b)),
        }
    }
    #[inline]
    pub fn plus(self, o: NpVal) -> NpVal {
        self.op(o, |a, b| a + b, |a, b| a + b)
    }
    #[inline]
    pub fn times(self, o: NpVal) -> NpVal {
        self.op(o, |a, b| a * b, |a, b| a * b)
    }
    #[inline]
    pub fn over(self, o: NpVal) -> NpVal {
        self.op(o, |a, b| a / b, |a, b| a / b)
    }
    #[inline]
    pub fn as_f64(self) -> f64 {
        match self {
            NpVal::F32(v) => v as f64,
            NpVal::Py(v) => v,
        }
    }
}

/// sacreBLEU `CHRF._compute_f_score` on a float32 numpy row under NumPy >= 2
/// semantics, operation by operation (see [`NpVal`]). Returns the value and
/// whether sacreBLEU's result is a `np.float32` (vs a Python float).
pub fn chrf_f_score_np2_f32(stats: &[f32], order: usize, beta: i64, eps_smoothing: bool) -> NpVal {
    use NpVal::{Py, F32};
    let eps = Py(1e-16);
    let factor = Py((beta * beta) as f64); // Python int (weak, exact)
    let one_plus = Py((1 + beta * beta) as f64);
    let mut score = Py(0.0);
    let mut effective_order: i64 = 0;
    let (mut avg_prec, mut avg_rec) = (Py(0.0), Py(0.0));
    for i in 0..order {
        let (n_hyp, n_ref, n_match) = (stats[3 * i], stats[3 * i + 1], stats[3 * i + 2]);
        let prec = if n_hyp > 0.0 { F32(n_match / n_hyp) } else { eps };
        let rec = if n_ref > 0.0 { F32(n_match / n_ref) } else { eps };
        let denom = factor.times(prec).plus(rec);
        let term = if denom.as_f64() > 0.0 { one_plus.times(prec).times(rec).over(denom) } else { eps };
        score = score.plus(term);
        if n_hyp > 0.0 && n_ref > 0.0 {
            avg_prec = avg_prec.plus(prec);
            avg_rec = avg_rec.plus(rec);
            effective_order += 1;
        }
    }
    if eps_smoothing {
        return Py(100.0).times(score).over(Py(order as f64));
    }
    if effective_order == 0 {
        avg_prec = Py(0.0);
        avg_rec = Py(0.0);
    } else {
        avg_prec = avg_prec.over(Py(effective_order as f64));
        avg_rec = avg_rec.over(Py(effective_order as f64));
    }
    if avg_prec.plus(avg_rec).as_f64() != 0.0 {
        let mut s = one_plus.times(avg_prec).times(avg_rec);
        s = s.over(factor.times(avg_prec).plus(avg_rec));
        Py(100.0).times(s)
    } else {
        Py(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex_f64(s: &str) -> f64 {
        f64::from_bits(u64::from_str_radix(s, 16).unwrap())
    }

    fn item(s: &str) -> PyNum {
        match s.as_bytes()[0] {
            b'i' => PyNum::Int(s[1..].parse().unwrap()),
            _ => PyNum::Float(hex_f64(&s[1..])),
        }
    }

    fn same(a: PyNum, b: PyNum) -> bool {
        match (a, b) {
            (PyNum::Int(x), PyNum::Int(y)) => x == y,
            (PyNum::Float(x), PyNum::Float(y)) => x.to_bits() == y.to_bits(),
            _ => false,
        }
    }

    #[test]
    fn sum_mode_follows_cpython_version() {
        assert_eq!(SumMode::for_python(2, 7), SumMode::Naive);
        assert_eq!(SumMode::for_python(3, 10), SumMode::Naive);
        assert_eq!(SumMode::for_python(3, 11), SumMode::Naive);
        assert_eq!(SumMode::for_python(3, 12), SumMode::Neumaier);
        assert_eq!(SumMode::for_python(3, 13), SumMode::Neumaier);
        assert_eq!(SumMode::for_python(3, 14), SumMode::NeumaierInts);
        assert_eq!(SumMode::for_python(3, 15), SumMode::NeumaierInts);
        assert_eq!(SumMode::for_python(4, 0), SumMode::NeumaierInts);
    }

    #[test]
    fn as_f64_converts_both_variants() {
        assert_eq!(PyNum::Int(-3).as_f64(), -3.0);
        assert_eq!(PyNum::Int(7).as_f64(), 7.0);
        assert_eq!(PyNum::Float(2.5).as_f64(), 2.5);
        assert_eq!(PyNum::Float(-0.25).as_f64(), -0.25);
    }

    /// Builtin `sum()` on CPython 3.12 (Neumaier), 3.14 (NeumaierInts) and a plain left-to-right
    /// reduction for the Naive mode (CPython <= 3.11). Fixtures come from the interpreters.
    #[test]
    fn py_sum_matches_cpython_builtin_sum() {
        for line in include_str!("../tests/fixtures/mut_py_sum.txt").lines() {
            let c: Vec<&str> = line.split('\t').collect();
            let items: Vec<PyNum> = c[3].split_whitespace().map(item).collect();
            for (mode, want) in [(SumMode::Naive, c[0]), (SumMode::Neumaier, c[1]), (SumMode::NeumaierInts, c[2])] {
                let got = py_sum(&items, mode);
                assert!(same(got, item(want)), "mode {mode:?} items {items:?}: got {got:?} want {want}");
            }
        }
    }

    #[test]
    fn bleu_scores_match_sacrebleu_grid() {
        for line in include_str!("../tests/fixtures/mut_bleu_score.txt").lines() {
            let t: Vec<&str> = line.split_whitespace().collect();
            let smooth = Smooth::from_name(t[0]).unwrap();
            let smooth_value = if t[1] == "none" { smooth.default_value() } else { t[1].parse().unwrap() };
            let n: usize = t[3].parse().unwrap();
            let stats: Vec<f64> = t[4..4 + 2 + 2 * n].iter().map(|s| s.parse::<i64>().unwrap() as f64).collect();
            let want = hex_f64(t[4 + 2 + 2 * n]);
            let cfg = BleuScoreConfig {
                smooth,
                smooth_value,
                effective_order: t[2] == "1",
                max_order: n,
                sum_mode: SumMode::NeumaierInts,
            };
            let got = compute_bleu(&stats, &cfg).score;
            assert_eq!(got.to_bits(), want.to_bits(), "{line}");
        }
    }

    #[test]
    fn chrf_scores_match_sacrebleu_grid() {
        for line in include_str!("../tests/fixtures/mut_chrf_score.txt").lines() {
            let t: Vec<&str> = line.split_whitespace().collect();
            let order = t[0].parse::<usize>().unwrap() + t[1].parse::<usize>().unwrap();
            let beta: i64 = t[2].parse().unwrap();
            let eps = t[3] == "1";
            let stats: Vec<i64> = t[4..4 + 3 * order].iter().map(|s| s.parse().unwrap()).collect();
            let want = hex_f64(t[4 + 3 * order]);
            let got = chrf_f_score(&stats, order, beta, eps);
            assert_eq!(got.to_bits(), want.to_bits(), "{line}");
        }
    }

    #[test]
    fn bleu_zero_hypothesis_and_reference_has_bp_one() {
        // sys == ref == 0: sacreBLEU's `if sys_len < ref_len` is false, so BP stays 1.
        let cfg = BleuScoreConfig {
            smooth: Smooth::None,
            smooth_value: 0.0,
            effective_order: false,
            max_order: 2,
            sum_mode: SumMode::NeumaierInts,
        };
        let r = compute_bleu(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &cfg);
        assert_eq!(r.bp, 1.0);
        assert_eq!(r.score, 0.0);
        // Perfect precisions at equal length: BP is 1 and the score is 100.
        let r = compute_bleu(&[4.0, 4.0, 4.0, 3.0, 4.0, 3.0], &cfg);
        assert_eq!(r.bp, 1.0);
        assert!((r.score - 100.0).abs() < 1e-9, "perfect precisions give 100, got {}", r.score);
    }
}
