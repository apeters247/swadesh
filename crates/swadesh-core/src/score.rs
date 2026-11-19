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
    let factor = (beta * beta) as f64;
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
    pub fn add(self, o: NpVal) -> NpVal {
        self.op(o, |a, b| a + b, |a, b| a + b)
    }
    #[inline]
    pub fn mul(self, o: NpVal) -> NpVal {
        self.op(o, |a, b| a * b, |a, b| a * b)
    }
    #[inline]
    pub fn div(self, o: NpVal) -> NpVal {
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
        let denom = factor.mul(prec).add(rec);
        let term = if denom.as_f64() > 0.0 { one_plus.mul(prec).mul(rec).div(denom) } else { eps };
        score = score.add(term);
        if n_hyp > 0.0 && n_ref > 0.0 {
            avg_prec = avg_prec.add(prec);
            avg_rec = avg_rec.add(rec);
            effective_order += 1;
        }
    }
    if eps_smoothing {
        return Py(100.0).mul(score).div(Py(order as f64));
    }
    if effective_order == 0 {
        avg_prec = Py(0.0);
        avg_rec = Py(0.0);
    } else {
        avg_prec = avg_prec.div(Py(effective_order as f64));
        avg_rec = avg_rec.div(Py(effective_order as f64));
    }
    if avg_prec.add(avg_rec).as_f64() != 0.0 {
        let mut s = one_plus.mul(avg_prec).mul(avg_rec);
        s = s.div(factor.mul(avg_prec).add(avg_rec));
        Py(100.0).mul(s)
    } else {
        Py(0.0)
    }
}
