//! Final score computation from sufficient statistics, replicating CPython's
//! floating point semantics *operation by operation* so results are bit-identical.

/// Which CPython `sum()` algorithm to emulate for float summation.
/// CPython 3.12 switched `sum()` of floats to Neumaier compensated summation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SumMode {
    /// CPython <= 3.11: naive left-to-right.
    Naive,
    /// CPython >= 3.12: Neumaier compensation for float items.
    Neumaier,
}

impl SumMode {
    pub fn for_python(major: u32, minor: u32) -> Self {
        if (major, minor) >= (3, 12) {
            SumMode::Neumaier
        } else {
            SumMode::Naive
        }
    }
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
        match *item {
            PyNum::Float(x) => match mode {
                SumMode::Naive => f_result += x,
                SumMode::Neumaier => {
                    let t = f_result + x;
                    if f_result.abs() >= x.abs() {
                        c += (f_result - t) + x;
                    } else {
                        c += (x - t) + f_result;
                    }
                    f_result = t;
                }
            },
            PyNum::Int(v) => f_result += v as f64,
        }
    }
    if mode == SumMode::Neumaier && c != 0.0 && c.is_finite() {
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
