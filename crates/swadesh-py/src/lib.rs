//! Python bindings for swadesh-core.

use std::borrow::Cow;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyString};

use swadesh_core::{
    bleu_corpus_stats, chrf_corpus_stats, chrf_f_score, compute_bleu, BleuConfig, BleuScoreConfig,
    ChrfConfig, Smooth, SumMode, Tokenizer,
};

/// Strings borrowed from Python objects (or owned when lowercasing was needed).
struct Corpus<'py> {
    hyps: Vec<Cow<'py, str>>,
    refs: Vec<Vec<Cow<'py, str>>>,
    nrefs_min: usize,
    nrefs_max: usize,
    tok_count: usize,
}

/// Characters whose `str.lower()` result is identical in CPython (any Unicode
/// version >= 13) and Rust's `str::to_lowercase`. Excludes U+03A3 (Σ), whose
/// lowercase depends on context (Final_Sigma) and could differ.
#[inline]
fn lower_is_stable(c: char) -> bool {
    let cp = c as u32;
    (cp < 0x0530 && cp != 0x03A3)
        || (0x1E00..=0x1EFF).contains(&cp) // Latin Extended Additional (Vietnamese)
        || (0x2000..=0x206F).contains(&cp) // General Punctuation (no case)
        || (0x3000..=0x9FFF).contains(&cp) // CJK, kana (no case)
        || (0xAC00..=0xD7A3).contains(&cp) // Hangul syllables (no case)
        || (0xFF00..=0xFFEF).contains(&cp) // half/full-width forms
}

fn text<'py>(obj: &'py Bound<'py, PyAny>, lowercase: bool) -> PyResult<Cow<'py, str>> {
    let s = obj.cast::<PyString>()?;
    let raw: &str = s.to_str()?;
    if !lowercase {
        return Ok(Cow::Borrowed(raw));
    }
    if raw.is_ascii() {
        if raw.bytes().any(|b| b.is_ascii_uppercase()) {
            Ok(Cow::Owned(raw.to_ascii_lowercase()))
        } else {
            Ok(Cow::Borrowed(raw))
        }
    } else if raw.chars().all(lower_is_stable) {
        // Case mappings in these blocks have been stable across all Unicode
        // versions any supported CPython uses, so Rust's lowercasing is exact.
        Ok(Cow::Owned(raw.to_lowercase()))
    } else {
        // Defer to CPython's own str.lower() for exact Unicode semantics.
        let lowered = s.call_method0("lower")?;
        Ok(Cow::Owned(lowered.cast::<PyString>()?.to_str()?.to_owned()))
    }
}

fn load<'py>(
    hyp_objs: &'py [Bound<'py, PyAny>],
    ref_objs: &'py [Vec<Bound<'py, PyAny>>],
    lowercase: bool,
) -> PyResult<Corpus<'py>> {
    let n = hyp_objs.len();
    let mut hyps = Vec::with_capacity(n);
    let mut tok_count = 0usize;
    for h in hyp_objs {
        let t = text(h, lowercase)?;
        if t.ends_with(" .") {
            tok_count += 1;
        }
        hyps.push(t);
    }
    let mut refs: Vec<Vec<Cow<'py, str>>> = (0..n).map(|_| Vec::with_capacity(ref_objs.len())).collect();
    for stream in ref_objs {
        for (i, r) in stream.iter().enumerate().take(n) {
            if r.is_none() {
                continue;
            }
            refs[i].push(text(r, lowercase)?);
        }
    }
    let nrefs_min = refs.iter().map(|r| r.len()).min().unwrap_or(0);
    let nrefs_max = refs.iter().map(|r| r.len()).max().unwrap_or(0);
    Ok(Corpus { hyps, refs, nrefs_min, nrefs_max, tok_count })
}

fn as_objs<'py>(list: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    list.try_iter()?.collect()
}

fn ref_streams<'py>(refs: &Bound<'py, PyAny>) -> PyResult<Vec<Vec<Bound<'py, PyAny>>>> {
    refs.try_iter()?.map(|s| as_objs(&s?)).collect()
}

fn i64_bytes<'py>(py: Python<'py>, v: &[i64]) -> Bound<'py, PyBytes> {
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 8) };
    PyBytes::new(py, bytes)
}

fn tokenizer(name: &str) -> PyResult<Tokenizer> {
    Tokenizer::from_name(name).ok_or_else(|| PyValueError::new_err(format!("unsupported native tokenizer {name:?}")))
}

fn sum_mode(py: Python<'_>) -> SumMode {
    let v = py.version_info();
    SumMode::for_python(v.major as u32, v.minor as u32)
}

/// Returns (stats_bytes, n_rows, width, nrefs_min, nrefs_max, tok_count).
#[pyfunction]
#[pyo3(signature = (hyps, refs, tokenize, lowercase, max_order))]
fn bleu_stats<'py>(
    py: Python<'py>,
    hyps: &Bound<'py, PyAny>,
    refs: &Bound<'py, PyAny>,
    tokenize: &str,
    lowercase: bool,
    max_order: usize,
) -> PyResult<(Bound<'py, PyBytes>, usize, usize, usize, usize, usize)> {
    let tok = tokenizer(tokenize)?;
    let hyp_objs = as_objs(hyps)?;
    let ref_objs = ref_streams(refs)?;
    let c = load(&hyp_objs, &ref_objs, lowercase)?;
    let hs: Vec<&str> = c.hyps.iter().map(|s| s.as_ref()).collect();
    let rs: Vec<Vec<&str>> = c.refs.iter().map(|v| v.iter().map(|s| s.as_ref()).collect()).collect();
    let cfg = BleuConfig { tokenizer: tok, max_order };
    let stats = py.detach(|| bleu_corpus_stats(&cfg, &hs, &rs));
    let width = 2 + 2 * max_order;
    Ok((i64_bytes(py, &stats), hs.len(), width, c.nrefs_min, c.nrefs_max, c.tok_count))
}

/// Returns (stats_bytes, n_rows, width, nrefs_min, nrefs_max).
#[pyfunction]
#[pyo3(signature = (hyps, refs, char_order, word_order, beta, lowercase, whitespace, eps_smoothing))]
#[allow(clippy::too_many_arguments)]
fn chrf_stats<'py>(
    py: Python<'py>,
    hyps: &Bound<'py, PyAny>,
    refs: &Bound<'py, PyAny>,
    char_order: usize,
    word_order: usize,
    beta: i64,
    lowercase: bool,
    whitespace: bool,
    eps_smoothing: bool,
) -> PyResult<(Bound<'py, PyBytes>, usize, usize, usize, usize)> {
    let hyp_objs = as_objs(hyps)?;
    let ref_objs = ref_streams(refs)?;
    let c = load(&hyp_objs, &ref_objs, lowercase)?;
    let hs: Vec<&str> = c.hyps.iter().map(|s| s.as_ref()).collect();
    let rs: Vec<Vec<&str>> = c.refs.iter().map(|v| v.iter().map(|s| s.as_ref()).collect()).collect();
    let cfg = ChrfConfig { char_order, word_order, beta, whitespace, eps_smoothing };
    let stats = py.detach(|| chrf_corpus_stats(&cfg, &hs, &rs));
    let width = 3 * (char_order + word_order);
    Ok((i64_bytes(py, &stats), hs.len(), width, c.nrefs_min, c.nrefs_max))
}

/// Segment-level BLEU for many (hyp, refs) pairs at once. Bit-identical to
/// calling `BLEU(...).sentence_score(h, refs).score` in a Python loop.
#[pyfunction]
#[pyo3(signature = (hyps, refs, tokenize, lowercase, max_order, smooth_method, smooth_value, effective_order))]
#[allow(clippy::too_many_arguments)]
fn bleu_sentence_scores<'py>(
    py: Python<'py>,
    hyps: &Bound<'py, PyAny>,
    refs: &Bound<'py, PyAny>,
    tokenize: &str,
    lowercase: bool,
    max_order: usize,
    smooth_method: &str,
    smooth_value: Option<f64>,
    effective_order: bool,
) -> PyResult<Bound<'py, PyList>> {
    let tok = tokenizer(tokenize)?;
    let smooth = Smooth::from_name(smooth_method)
        .ok_or_else(|| PyValueError::new_err(format!("unknown smooth_method {smooth_method:?}")))?;
    let hyp_objs = as_objs(hyps)?;
    let ref_objs = ref_streams(refs)?;
    let c = load(&hyp_objs, &ref_objs, lowercase)?;
    let hs: Vec<&str> = c.hyps.iter().map(|s| s.as_ref()).collect();
    let rs: Vec<Vec<&str>> = c.refs.iter().map(|v| v.iter().map(|s| s.as_ref()).collect()).collect();
    let cfg = BleuConfig { tokenizer: tok, max_order };
    let scfg = BleuScoreConfig {
        smooth,
        smooth_value: smooth_value.unwrap_or(smooth.default_value()),
        effective_order,
        max_order,
        sum_mode: sum_mode(py),
    };
    let width = 2 + 2 * max_order;
    let scores: Vec<f64> = py.detach(|| {
        let stats = bleu_corpus_stats(&cfg, &hs, &rs);
        stats
            .chunks(width)
            .map(|row| {
                let f: Vec<f64> = row.iter().map(|&x| x as f64).collect();
                compute_bleu(&f, &scfg).score
            })
            .collect()
    });
    PyList::new(py, scores)
}

/// Segment-level chrF for many pairs at once.
#[pyfunction]
#[pyo3(signature = (hyps, refs, char_order, word_order, beta, lowercase, whitespace, eps_smoothing))]
#[allow(clippy::too_many_arguments)]
fn chrf_sentence_scores<'py>(
    py: Python<'py>,
    hyps: &Bound<'py, PyAny>,
    refs: &Bound<'py, PyAny>,
    char_order: usize,
    word_order: usize,
    beta: i64,
    lowercase: bool,
    whitespace: bool,
    eps_smoothing: bool,
) -> PyResult<Bound<'py, PyList>> {
    let hyp_objs = as_objs(hyps)?;
    let ref_objs = ref_streams(refs)?;
    let c = load(&hyp_objs, &ref_objs, lowercase)?;
    let hs: Vec<&str> = c.hyps.iter().map(|s| s.as_ref()).collect();
    let rs: Vec<Vec<&str>> = c.refs.iter().map(|v| v.iter().map(|s| s.as_ref()).collect()).collect();
    let cfg = ChrfConfig { char_order, word_order, beta, whitespace, eps_smoothing };
    let order = char_order + word_order;
    let scores: Vec<f64> = py.detach(|| {
        chrf_corpus_stats(&cfg, &hs, &rs)
            .chunks(3 * order)
            .map(|row| chrf_f_score(row, order, beta, eps_smoothing))
            .collect()
    });
    PyList::new(py, scores)
}

fn f32_vec(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn i64_vec(b: &[u8]) -> Vec<i64> {
    b.chunks_exact(8).map(|c| i64::from_ne_bytes(c.try_into().unwrap())).collect()
}

/// Bootstrap row sums: `stats_f32[idxs].sum(1)` with numpy's exact float32
/// summation order, without materializing the 3-D tensor. Returns f32 bytes.
#[pyfunction]
fn bootstrap_sums_f32<'py>(py: Python<'py>, stats: &[u8], width: usize, idxs: &[u8], n: usize) -> Bound<'py, PyBytes> {
    let s = f32_vec(stats);
    let ix = i64_vec(idxs);
    let out = py.detach(|| swadesh_core::resample::bootstrap_sums_f32(&s, width, &ix, n));
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const u8, out.len() * 4) };
    PyBytes::new(py, bytes)
}

/// Paired approximate randomization sums (exact int64). Returns (a, b) bytes.
#[pyfunction]
fn ar_sums_i64<'py>(
    py: Python<'py>,
    bl: &[u8],
    sys: &[u8],
    width: usize,
    sel: &[u8],
    n: usize,
) -> (Bound<'py, PyBytes>, Bound<'py, PyBytes>) {
    let (b1, s1) = (i64_vec(bl), i64_vec(sys));
    let (a, b) = py.detach(|| swadesh_core::resample::ar_sums_i64(&b1, &s1, width, sel, n));
    (i64_bytes(py, &a), i64_bytes(py, &b))
}

#[pymodule]
fn _swadesh(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bleu_stats, m)?)?;
    m.add_function(wrap_pyfunction!(chrf_stats, m)?)?;
    m.add_function(wrap_pyfunction!(bleu_sentence_scores, m)?)?;
    m.add_function(wrap_pyfunction!(chrf_sentence_scores, m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_sums_f32, m)?)?;
    m.add_function(wrap_pyfunction!(ar_sums_i64, m)?)?;
    m.add("GENERATED_PYTHON", swadesh_core::tables::GENERATED_PYTHON)?;
    m.add("GENERATED_REGEX", swadesh_core::tables::GENERATED_REGEX)?;
    Ok(())
}
