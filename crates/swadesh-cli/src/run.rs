//! The sacreBLEU CLI pipeline (`sacrebleu.py::main`) for single-system scoring.

use std::borrow::Cow;
use std::io::Write;

use swadesh_core::significance::{bootstrap_ci, paired_ar, paired_ar_f64, paired_bs, ScoreKind};
use swadesh_core::stats::{BleuConfig, ChrfConfig};
use swadesh_core::stats_dispatch::{bleu_stats as bleu_segment, chrf_stats as chrf_segment};
use swadesh_core::ter_clean::ter_segment as ter_segment_clean;
use swadesh_core::Tokenizer;
use swadesh_core::{
    bleu_corpus_stats, chrf_corpus_stats, chrf_f_score, compute_bleu, ter_corpus_stats,
    BleuScoreConfig, Smooth, SumMode, TerConfig, TerError,
};

use crate::args::{Cli, TOKENIZE_CHOICES};
use crate::input;
use crate::lower::py_lower;
use crate::pyfmt::fixed;
use crate::pyfmt::json_float;
use crate::report::{json_str, print_single, Score, Signature};
use crate::{fail, Fail};


/// Tokenizers built into swadesh.
const NATIVE_TOKENIZERS: &[&str] = &["none", "13a", "intl", "zh", "char"];

pub fn execute(cli: &Cli, out: &mut dyn Write) -> Result<(), Fail> {
    if cli.sentence_level && cli.metrics.len() > 1 {
        return fail(1, "Only one metric can be used in sentence-level mode.");
    }
    if cli.num_refs != 1 && cli.refs.len() > 1 {
        return fail(
            1,
            "The --num-refs argument allows you to provide any number of tab-delimited references \
             in a single file. You cannot then provide multiple reference files.",
        );
    }
    if cli.refs.is_empty() {
        return fail(
            1,
            "If manual references given, make sure to provide them before the -i/--input argument \
             to avoid confusion. (swadesh needs reference files; test sets need pip install swadesh)",
        );
    }
    if cli.width < 0 {
        return fail(1, "Format specifier missing precision");
    }
    if let Some(pair) = &cli.langpair {
        if pair.split('-').nth(1).is_none() {
            return fail(1, "IndexError: list index out of range (-l expects SRC-TRG, e.g. en-de)");
        }
    }
    if cli.max_ngram_order < 1 && cli.metrics.iter().any(|m| m == "bleu") {
        return fail(2, "--max-ngram-order must be at least 1");
    }
    // Unsupported options fail before any file is opened.
    input::check_encoding(&cli.encoding)?;
    if cli.metrics.iter().any(|m| m == "bleu") {
        bleu_tokenizer(cli)?;
    }

    let refs = read_refs(cli)?;
    let (systems, sys_names) = read_systems(cli)?;
    if systems.iter().any(|s| s.is_empty()) {
        return fail(1, "Test set None contains no sentence");
    }
    if systems.iter().any(|s| refs.iter().any(|r| r.len() != s.len())) {
        return fail(1, "System and reference streams have different lengths.");
    }
    let nrefs = refs.len();
    let width = cli.width as usize;
    let json = cli.format == "json";

    // The `-m` names become upper-case metric names; duplicates collapse, output is sorted.
    let mut names: Vec<String> = cli.metrics.iter().map(|m| m.to_uppercase()).collect();
    names.sort();
    names.dedup();

    // sacreBLEU refuses `-ci` with more than one system: the interval is a single-system report and
    // the paired tests are the multi-system one.
    if cli.confidence && systems.len() > 1 {
        eprintln!("sacreBLEU: Use paired tests (--paired) for multiple systems.");
        return Err(Fail { code: 1, msg: String::new() });
    }

    // Paired significance testing: the first system is the baseline, the others are compared to it.
    if cli.paired_ar || cli.paired_bs {
        if cli.sentence_level {
            return fail(1, "sentence-level output is not available for paired significance testing");
        }
        if !json {
            return fail(1, "the paired report is implemented for --format json so far");
        }
        let bs = cli.paired_bs;
        let n_samples = if bs { cli.paired_bs_n } else { cli.paired_ar_n };
        if n_samples <= 0 {
            return fail(1, "the paired test needs at least one sample");
        }
        let n_samples = n_samples as usize;
        let (seed, _) = bootstrap_seed()?;
        let n_ar_conf = 0usize; // `-ci` with several systems is refused above, as in sacreBLEU

        // A system named like the baseline is dropped, with a note, before anything is counted.
        let mut keep: Vec<usize> = vec![0];
        for i in 1..systems.len() {
            if sys_names[i] == sys_names[0] {
                eprintln!(
                    "sacreBLEU: Ignoring '{}' as it was also given as the baseline.",
                    sys_names[i]
                );
            } else {
                keep.push(i);
            }
        }
        // Any other repeated name is an error.
        let mut seen = std::collections::HashSet::new();
        for i in keep.iter().skip(1) {
            if !seen.insert(sys_names[*i].as_str()) {
                eprintln!("sacreBLEU: '{}' already used to name a system.", sys_names[*i]);
                eprintln!("sacreBLEU: Make sure to have a different basename for each system.");
                return Err(Fail { code: 1, msg: String::new() });
            }
        }
        if keep.len() < 2 {
            return fail(1, "You must provide at least 2 systems for paired testing.");
        }
        eprintln!("sacreBLEU: Found {} systems.", keep.len());
        let bl_display = format!("Baseline: {}", sys_names[0]);

        // sacreBLEU pre-computes the baseline for *every* metric first, then walks the systems, so
        // the status lines come out in two passes.
        let mut baselines: Vec<CorpusRows> = Vec::with_capacity(names.len());
        for metric in &names {
            let bl = corpus_rows(cli, metric, &systems[0], &refs, nrefs)?;
            eprintln!("sacreBLEU: Pre-computing {} statistics for '{}'", metric, bl_display);
            baselines.push(bl);
        }

        // Per system, the per-metric result: (name, score, p_value, mean, ci).
        type Entry = (String, f64, Option<f64>, Option<f64>, Option<f64>);
        let mut table: Vec<Vec<Entry>> = (0..keep.len()).map(|_| Vec::new()).collect();

        for (mi, metric) in names.iter().enumerate() {
            let bl = &baselines[mi];
            let (bl_mean, bl_ci) = if bs {
                let (m, c) = bootstrap_ci(&bl.f64_rows, bl.width, &bl.kind, n_samples, seed);
                (Some(m), Some(c))
            } else if n_ar_conf > 0 {
                let (m, c) = bootstrap_ci(&bl.f64_rows, bl.width, &bl.kind, n_ar_conf, seed);
                (Some(m), Some(c))
            } else {
                (None, None)
            };
            table[0].push((bl.name.clone(), bl.score, None, bl_mean, bl_ci));

            for (k, i) in keep.iter().enumerate().skip(1) {
                let sys = corpus_rows(cli, metric, &systems[*i], &refs, nrefs)?;
                eprintln!(
                    "sacreBLEU: Computing {} for '{}' and extracting sufficient statistics",
                    sys.name, sys_names[*i]
                );
                // sacreBLEU says "paired bootstrap resampling test" but plain "approximate
                // randomization test": the word "paired" is only in the bootstrap wording.
                if bs {
                    eprintln!("sacreBLEU:  > Performing paired bootstrap resampling test (# resamples: {n_samples})");
                } else {
                    eprintln!("sacreBLEU:  > Performing approximate randomization test (# trials: {n_samples})");
                }
                let diff = (bl.score - sys.score).abs();
                let (p, mean, ci) = if bs {
                    let (p, mean, ci) = paired_bs(
                        &bl.f64_rows, &sys.f64_rows, sys.width, &sys.kind, n_samples, seed, diff,
                    );
                    (p, Some(mean), Some(ci))
                } else if bl.i64_rows.is_empty() {
                    // TER's statistics are not integral: sacreBLEU shuffles the floats instead.
                    let p = paired_ar_f64(
                        &bl.f64_rows, &sys.f64_rows, sys.width, &sys.kind, n_samples, seed, diff,
                    );
                    (p, None, None)
                } else {
                    paired_ar(
                        &bl.i64_rows, &sys.i64_rows, sys.width, &sys.kind, n_samples, n_ar_conf, seed, diff,
                    )
                };
                table[k].push((sys.name.clone(), sys.score, Some(p), mean, ci));
            }
        }

        let mut rows_out: Vec<String> = Vec::with_capacity(keep.len());
        for (k, entry) in table.iter().enumerate() {
            let shown = if k == 0 { bl_display.clone() } else { sys_names[keep[k]].clone() };
            let mut lines: Vec<String> = vec![format!("        \"system\": {}", json_str(&shown))];
            for (name, score, p, mean, ci) in entry {
                let opt = |v: &Option<f64>| match v {
                    Some(x) => json_float(*x),
                    None => "null".to_string(),
                };
                let fields = vec![
                    format!("            \"score\": {}", json_float(*score)),
                    format!("            \"p_value\": {}", opt(p)),
                    format!("            \"mean\": {}", opt(mean)),
                    format!("            \"ci\": {}", opt(ci)),
                ];
                lines.push(format!("        {}: {{\n{}\n        }}", json_str(name), fields.join(",\n")));
            }
            rows_out.push(format!("    {{\n{}\n    }}", lines.join(",\n")));
        }
        writeln!(out, "[\n{}\n]", rows_out.join(",\n")).map_err(io_err)?;
        return Ok(());
    }

    // Multi-system mode: sacreBLEU reports every system side by side instead of one score. Only the
    // JSON form is implemented so far; the text/latex forms need a faithful `tabulate` port.
    if systems.len() > 1 {
        // sacreBLEU refuses two systems with the same name, printing both lines itself and
        // exiting 1 before any output.
        let mut seen = std::collections::HashSet::new();
        for name in &sys_names {
            if !seen.insert(name.as_str()) {
                eprintln!("sacreBLEU: '{name}' already used to name a system.");
                eprintln!("sacreBLEU: Make sure to have a different basename for each system.");
                return Err(Fail { code: 1, msg: String::new() });
            }
        }
        eprintln!("sacreBLEU: Found {} systems.", systems.len());
        if cli.sentence_level {
            return fail(1, "sentence-level output for several systems is not implemented yet");
        }
        if !json {
            return fail(1, "the multi-system report is implemented for --format json so far");
        }
        let mut rows: Vec<String> = Vec::with_capacity(systems.len());
        for (system, name) in systems.iter().zip(&sys_names) {
            let mut entries: Vec<String> =
                vec![format!("        \"system\": {}", json_str(name))];
            for metric_name in &names {
                let (score, _sig) = match metric_name.as_str() {
                    "BLEU" => bleu_corpus(cli, bleu_tokenizer(cli)?, system, &refs, nrefs)?,
                    "TER" => ter_corpus(cli, system, &refs, nrefs)?,
                    _ => chrf_corpus(cli, system, &refs, nrefs)?,
                };
                entries.push(format!(
                    "        {}: {}",
                    json_str(&score.name),
                    json_str(&fixed(score.score, width))
                ));
            }
            rows.push(format!("    {{\n{}\n    }}", entries.join(",\n")));
        }
        writeln!(out, "[\n{}\n]", rows.join(",\n")).map_err(io_err)?;
        return Ok(());
    }

    if cli.sentence_level {
        let metric = &names[0];
        let bleu_tok = if metric == "BLEU" { Some(bleu_tokenizer(cli)?) } else { None };
        if metric == "CHRF" {
            check_chrf_orders(cli)?;
        }
        let mut lines = Vec::with_capacity(systems[0].len());
        let sig_short = cli.short;
        for i in 0..systems[0].len() {
            let seg_refs: Vec<&str> = refs.iter().map(|r| r[i].as_str()).collect();
            let (score, sig) = match bleu_tok {
                Some(tok) => {
                    let hyp = maybe_lower(&systems[0][i], cli.lowercase);
                    let hyp_refs: Vec<Cow<str>> = seg_refs.iter().map(|r| maybe_lower(r, cli.lowercase)).collect();
                    let refs_view: Vec<&str> = hyp_refs.iter().map(|c| c.as_ref()).collect();
                    bleu_sentence(cli, tok, &hyp, &refs_view, nrefs)
                }
                None if metric == "TER" => ter_sentence(cli, &systems[0][i], &seg_refs, nrefs)?,
                None => {
                    let hyp = maybe_lower(&systems[0][i], cli.chrf_lowercase);
                    let hyp_refs: Vec<Cow<str>> = seg_refs.iter().map(|r| maybe_lower(r, cli.chrf_lowercase)).collect();
                    let refs_view: Vec<&str> = hyp_refs.iter().map(|c| c.as_ref()).collect();
                    chrf_sentence(cli, &hyp, &refs_view, nrefs)?
                }
            };
            lines.push(score.format(width, cli.score_only, &sig, sig_short, false));
        }
        for line in lines {
            writeln!(out, "{line}").map_err(io_err)?;
        }
        return Ok(());
    }

    let mut results: Vec<String> = Vec::new();
    for name in &names {
        let (score, sig) = match name.as_str() {
            "BLEU" => bleu_corpus(cli, bleu_tokenizer(cli)?, &systems[0], &refs, nrefs)?,
            "TER" => ter_corpus(cli, &systems[0], &refs, nrefs)?,
            _ => chrf_corpus(cli, &systems[0], &refs, nrefs)?,
        };
        // sacreBLEU: the short signature only applies outside JSON output.
        let short = !json && cli.short;
        results.push(score.format(width, cli.score_only, &sig, short, json));
    }
    print_single(&results, json, out).map_err(io_err)?;
    Ok(())
}

/// `-ci` needs a reproducible draw: sacreBLEU reads `SACREBLEU_SEED` (default `12345`) and accepts
/// `none` for an entropy-seeded draw. The entropy draw cannot be reproduced, so it is refused with
/// a message instead of quietly reporting a different interval.
fn bootstrap_seed() -> Result<(u128, String), Fail> {
    let raw = std::env::var("SACREBLEU_SEED").unwrap_or_else(|_| "12345".to_string());
    if raw.eq_ignore_ascii_case("none") {
        return Err(Fail {
            code: 1,
            msg: "SACREBLEU_SEED=none is not supported: swadesh draws the bootstrap indices deterministically".into(),
        });
    }
    let seed: u128 = raw.parse().map_err(|_| Fail {
        code: 1,
        msg: format!("ValueError: invalid literal for int() with base 10: {raw:?}"),
    })?;
    Ok((seed, raw.to_lowercase()))
}

/// `--confidence-n`, refused when it could not produce an interval.
fn confidence_samples(cli: &Cli) -> Result<usize, Fail> {
    if cli.confidence_n <= 0 {
        return Err(Fail { code: 1, msg: "StatisticsError: mean requires at least one data point".into() });
    }
    Ok(cli.confidence_n as usize)
}

/// The bootstrap interval when `-ci` was given, and the seed string the signature reports.
fn confidence(
    cli: &Cli,
    rows: &[f64],
    width: usize,
    kind: &ScoreKind,
) -> Result<(Option<(f64, f64)>, Option<String>), Fail> {
    if !cli.confidence {
        return Ok((None, None));
    }
    let n = confidence_samples(cli)?;
    let (seed, seed_str) = bootstrap_seed()?;
    Ok((Some(bootstrap_ci(rows, width, kind, n, seed)), Some(seed_str)))
}

fn io_err(e: std::io::Error) -> Fail {
    Fail { code: 1, msg: format!("writing output: {e}") }
}

fn maybe_lower(s: &str, lower: bool) -> Cow<'_, str> {
    if lower {
        py_lower(s)
    } else {
        Cow::Borrowed(s)
    }
}

/// BLEU tokenizer: `-tok`, else from the target language of `-l`, else 13a.
fn bleu_tokenizer(cli: &Cli) -> Result<Tokenizer, Fail> {
    let name: String = match &cli.tokenize {
        Some(t) => t.clone(),
        None => match &cli.langpair {
            Some(pair) => {
                let Some(trg) = pair.split('-').nth(1) else {
                    return fail(1, "IndexError: list index out of range (-l expects SRC-TRG, e.g. en-de)");
                };
                match trg {
                    "zh" => "zh".to_string(),
                    "ja" => "ja-mecab".to_string(),
                    "ko" => "ko-mecab".to_string(),
                    _ => "13a".to_string(),
                }
            }
            None => "13a".to_string(),
        },
    };
    if !NATIVE_TOKENIZERS.contains(&name.as_str()) {
        debug_assert!(TOKENIZE_CHOICES.contains(&name.as_str()));
        return fail(
            2,
            format!("the '{name}' tokenizer is not built into swadesh; use the Python package: pip install swadesh"),
        );
    }
    Tokenizer::from_name(&name).ok_or_else(|| Fail { code: 2, msg: format!("unknown tokenizer {name}") })
}

/// Reference streams, one `Vec<String>` per reference (`read_refs` in sacrebleu.py).
fn read_refs(cli: &Cli) -> Result<Vec<Vec<String>>, Fail> {
    let nlists = (cli.refs.len() as i64).max(cli.num_refs).max(0) as usize;
    let mut full: Vec<Vec<String>> = vec![Vec::new(); nlists];
    for (refno, path) in cli.refs.iter().enumerate() {
        let lines = input::file_lines(path, &cli.encoding)?;
        for (idx, line) in lines.into_iter().enumerate() {
            if cli.num_refs == 1 {
                full[refno].push(line);
                continue;
            }
            // Python: line.split('\t', maxsplit=num_refs-1); a negative maxsplit splits everything.
            let parts: Vec<&str> = if cli.num_refs >= 1 {
                line.splitn(cli.num_refs as usize, '\t').collect()
            } else {
                line.split('\t').collect()
            };
            if parts.len() as i64 != cli.num_refs {
                return fail(
                    17,
                    format!(
                        "FATAL: line {}: expected {} fields, but found {}.",
                        idx + 1,
                        cli.num_refs,
                        parts.len()
                    ),
                );
            }
            for (r, part) in parts.iter().enumerate() {
                full[r].push(part.to_string());
            }
        }
    }
    Ok(full)
}

/// The single hypothesis stream (`-i FILE` or stdin).
/// The systems to score, with the names sacreBLEU reports for them.
///
/// One system per `-i` file (named by the path as given), or the tab-separated columns of stdin
/// (named `System 1`, `System 2`, ...). A single file or tab-free stdin is one system, which keeps
/// the ordinary single-score output.
fn read_systems(cli: &Cli) -> Result<(Vec<Vec<String>>, Vec<String>), Fail> {
    match &cli.input {
        None => {
            let lines = input::stdin_lines(&cli.encoding)?;
            if lines.is_empty() {
                return Ok((vec![lines], vec!["System 1".to_string()]));
            }
            if !lines.iter().any(|l| l.contains('\t')) {
                return Ok((vec![lines], vec!["System 1".to_string()]));
            }
            // Tab-separated hypotheses: each line holds one segment per system.
            let n = lines[0].split('\t').count();
            if lines.iter().any(|l| l.split('\t').count() != n) {
                return fail(17, "FATAL: the number of tab-delimited fields in the input stream differ across lines.");
            }
            let mut systems: Vec<Vec<String>> = vec![Vec::with_capacity(lines.len()); n];
            for line in &lines {
                for (k, part) in line.split('\t').enumerate() {
                    systems[k].push(part.to_string());
                }
            }
            let names = (1..=n).map(|i| format!("System {i}")).collect();
            Ok((systems, names))
        }
        Some(files) if files.is_empty() => fail(2, "-i/--input needs a file name"),
        Some(files) => {
            let mut systems = Vec::with_capacity(files.len());
            for f in files {
                systems.push(input::file_lines(f, &cli.encoding)?);
            }
            Ok((systems, files.clone()))
        }
    }
}


/// The per-segment statistics of one system for one metric: the rows (as `f64`, and as `i64` for the
/// metrics whose statistics are integral), the metric's kind and the corpus score.
struct CorpusRows {
    f64_rows: Vec<f64>,
    i64_rows: Vec<i64>,
    width: usize,
    kind: ScoreKind,
    score: f64,
    name: String,
}

fn corpus_rows(
    cli: &Cli,
    metric: &str,
    system: &[String],
    refs: &[Vec<String>],
    nrefs: usize,
) -> Result<CorpusRows, Fail> {
    let _ = nrefs;
    match metric {
        "BLEU" => {
            let tok = bleu_tokenizer(cli)?;
            let hyps: Vec<Cow<str>> = system.iter().map(|h| maybe_lower(h, cli.lowercase)).collect();
            let hyp_view: Vec<&str> = hyps.iter().map(|c| c.as_ref()).collect();
            let seg_owned = segment_refs(refs, cli.lowercase);
            let seg_refs = as_views(&seg_owned);
            let (bcfg, scfg) = bleu_config(cli, tok, false);
            let width = 2 + 2 * bcfg.max_order;
            let rows = bleu_corpus_stats(&bcfg, &hyp_view, &seg_refs);
            let mut total = vec![0i64; width];
            for row in rows.chunks(width) {
                for (t, v) in total.iter_mut().zip(row) {
                    *t += v;
                }
            }
            let stats: Vec<f64> = total.iter().map(|&v| v as f64).collect();
            let score = compute_bleu(&stats, &scfg).score;
            Ok(CorpusRows {
                f64_rows: rows.iter().map(|&v| v as f64).collect(),
                i64_rows: rows,
                width,
                kind: ScoreKind::Bleu(scfg),
                score,
                name: "BLEU".into(),
            })
        }
        "TER" => {
            let cfg = ter_config(cli);
            let lower = !cli.ter_case_sensitive;
            let hyps: Vec<Cow<str>> = system.iter().map(|h| maybe_lower(h, lower)).collect();
            let hyp_view: Vec<&str> = hyps.iter().map(|c| c.as_ref()).collect();
            let seg_owned = segment_refs(refs, lower);
            let seg_refs = as_views(&seg_owned);
            let rows = ter_corpus_stats(&cfg, &hyp_view, &seg_refs).map_err(ter_err)?;
            let mut total = [0.0f64; 2];
            for row in rows.chunks(2) {
                total[0] += row[0];
                total[1] += row[1];
            }
            Ok(CorpusRows {
                score: ter_score_value(total[0], total[1]),
                f64_rows: rows,
                i64_rows: Vec::new(),
                width: 2,
                kind: ScoreKind::Ter,
                name: "TER".into(),
            })
        }
        _ => {
            check_chrf_orders(cli)?;
            let cfg = chrf_config(cli);
            let order = cfg.char_order + cfg.word_order;
            if order == 0 {
                return fail(1, "IndexError: list index out of range (chrF with no n-gram orders)");
            }
            let hyps: Vec<Cow<str>> = system.iter().map(|h| maybe_lower(h, cli.chrf_lowercase)).collect();
            let hyp_view: Vec<&str> = hyps.iter().map(|c| c.as_ref()).collect();
            let seg_owned = segment_refs(refs, cli.chrf_lowercase);
            let seg_refs = as_views(&seg_owned);
            let width = 3 * order;
            let rows = chrf_corpus_stats(&cfg, &hyp_view, &seg_refs);
            let mut total = vec![0i64; width];
            for row in rows.chunks(width.max(1)) {
                for (t, v) in total.iter_mut().zip(row) {
                    *t += v;
                }
            }
            let score = chrf_f_score(&total, order, cfg.beta, cfg.eps_smoothing);
            Ok(CorpusRows {
                f64_rows: rows.iter().map(|&v| v as f64).collect(),
                i64_rows: rows,
                width,
                kind: ScoreKind::Chrf { order, beta: cfg.beta, eps_smoothing: cfg.eps_smoothing },
                score,
                name: chrf_name(cli),
            })
        }
    }
}

fn bleu_tokenizer_sig(tok: Tokenizer) -> &'static str {
    match tok {
        Tokenizer::None => "none",
        Tokenizer::T13a => "13a",
        Tokenizer::Intl => "intl",
        Tokenizer::Zh => "zh",
        Tokenizer::Char => "char",
    }
}

/// Smoothing setting as sacreBLEU prints it: `floor[0.10]`, `add-k[1.00]`, `exp`, `none`.
fn bleu_smooth(cli: &Cli) -> (Smooth, f64, String) {
    let method = Smooth::from_name(&cli.smooth).unwrap_or(Smooth::Exp);
    let value = cli.smooth_value.unwrap_or_else(|| method.default_value());
    let text = match method {
        Smooth::Floor | Smooth::AddK => format!("{}[{}]", cli.smooth, fixed(value, 2)),
        _ => cli.smooth.clone(),
    };
    (method, value, text)
}

fn bleu_signature(cli: &Cli, tok: Tokenizer, nrefs: usize, eff: bool) -> Signature {
    let (_, _, smooth_text) = bleu_smooth(cli);
    Signature::new(
        nrefs,
        vec![
            ("case", "c", if cli.lowercase { "lc" } else { "mixed" }.to_string()),
            ("eff", "e", if eff { "yes" } else { "no" }.to_string()),
            ("tok", "tok", bleu_tokenizer_sig(tok).to_string()),
            ("smooth", "s", smooth_text),
        ],
    )
}

fn bleu_verbose(sys_len: i64, ref_len: i64, bp: f64, precisions: &[f64]) -> String {
    let prec: Vec<String> = precisions.iter().map(|p| fixed(*p, 1)).collect();
    let ratio = if ref_len != 0 { sys_len as f64 / ref_len as f64 } else { 0.0 };
    format!(
        "{} (BP = {} ratio = {} hyp_len = {} ref_len = {})",
        prec.join("/"),
        fixed(bp, 3),
        fixed(ratio, 3),
        sys_len,
        ref_len
    )
}

fn bleu_config(cli: &Cli, tok: Tokenizer, eff: bool) -> (BleuConfig, BleuScoreConfig) {
    let (smooth, smooth_value, _) = bleu_smooth(cli);
    let max_order = cli.max_ngram_order as usize;
    (
        BleuConfig { tokenizer: tok, max_order },
        BleuScoreConfig { smooth, smooth_value, effective_order: eff, max_order, sum_mode: SumMode::for_python(cli.python_version.0, cli.python_version.1) },
    )
}

fn bleu_corpus(
    cli: &Cli,
    tok: Tokenizer,
    system: &[String],
    refs: &[Vec<String>],
    nrefs: usize,
) -> Result<(Score, Signature), Fail> {
    if !cli.force {
        let tokenized = system.iter().filter(|h| h.ends_with(" .")).count();
        if tokenized >= 100 {
            eprintln!("sacreBLEU: That's 100 lines that end in a tokenized period ('.')");
            eprintln!("sacreBLEU: It looks like you forgot to detokenize your test data, which may hurt your score.");
        }
    }
    let hyps: Vec<Cow<str>> = system.iter().map(|h| maybe_lower(h, cli.lowercase)).collect();
    let hyp_view: Vec<&str> = hyps.iter().map(|c| c.as_ref()).collect();
    let seg_owned = segment_refs(refs, cli.lowercase);
    let seg_refs = as_views(&seg_owned);
    let (bcfg, scfg) = bleu_config(cli, tok, false);
    let width = 2 + 2 * bcfg.max_order;
    let rows = bleu_corpus_stats(&bcfg, &hyp_view, &seg_refs);
    let mut total = vec![0i64; width];
    for row in rows.chunks(width) {
        for (t, v) in total.iter_mut().zip(row) {
            *t += v;
        }
    }
    let stats: Vec<f64> = total.iter().map(|&v| v as f64).collect();
    let res = compute_bleu(&stats, &scfg);
    let verbose = bleu_verbose(total[0], total[1], res.bp, &res.precisions);
    let rows_f: Vec<f64> = rows.iter().map(|&v| v as f64).collect();
    let (conf, seed_str) = confidence(cli, &rows_f, width, &ScoreKind::Bleu(scfg))?;
    let mut sig = bleu_signature(cli, tok, nrefs, false);
    if let Some(seed) = &seed_str {
        sig = sig.with_confidence(confidence_samples(cli)?, seed);
    }
    Ok((Score { name: "BLEU".into(), score: res.score, verbose: Some(verbose), confidence: conf }, sig))
}

fn bleu_sentence(
    cli: &Cli,
    tok: Tokenizer,
    hyp: &str,
    refs: &[&str],
    nrefs: usize,
) -> (Score, Signature) {
    let (bcfg, scfg) = bleu_config(cli, tok, true);
    let mut row = Vec::new();
    bleu_segment(&bcfg, hyp, refs, &mut row);
    let stats: Vec<f64> = row.iter().map(|&v| v as f64).collect();
    let res = compute_bleu(&stats, &scfg);
    let verbose = bleu_verbose(row[0], row[1], res.bp, &res.precisions);
    let sig = bleu_signature(cli, tok, nrefs, true);
    (Score { name: "BLEU".into(), score: res.score, verbose: Some(verbose), confidence: None }, sig)
}

fn chrf_config(cli: &Cli) -> ChrfConfig {
    ChrfConfig {
        char_order: cli.chrf_char_order as usize,
        word_order: cli.chrf_word_order as usize,
        beta: cli.chrf_beta,
        whitespace: cli.chrf_whitespace,
        eps_smoothing: cli.chrf_eps_smoothing,
    }
}

fn chrf_name(cli: &Cli) -> String {
    format!("chrF{}{}", cli.chrf_beta, "+".repeat(cli.chrf_word_order as usize))
}

fn chrf_signature(cli: &Cli, nrefs: usize) -> Signature {
    Signature::new(
        nrefs,
        vec![
            ("case", "c", if cli.chrf_lowercase { "lc" } else { "mixed" }.to_string()),
            ("eff", "e", if cli.chrf_eps_smoothing { "no" } else { "yes" }.to_string()),
            ("nc", "nc", cli.chrf_char_order.to_string()),
            ("nw", "nw", cli.chrf_word_order.to_string()),
            ("space", "s", if cli.chrf_whitespace { "yes" } else { "no" }.to_string()),
        ],
    )
}

/// sacreBLEU fails on chrF orders that make no sense: negative orders, and an
/// order of zero in corpus mode (`sum_of_lists` indexes an empty row).
fn check_chrf_orders(cli: &Cli) -> Result<(), Fail> {
    if cli.chrf_char_order < 0 {
        return fail(1, "chrF character order must not be negative");
    }
    if cli.chrf_word_order < 0 {
        return fail(
            2,
            "negative chrF word order (-cw) is not supported by swadesh; pip install swadesh",
        );
    }
    Ok(())
}

fn chrf_corpus(cli: &Cli, system: &[String], refs: &[Vec<String>], nrefs: usize) -> Result<(Score, Signature), Fail> {
    check_chrf_orders(cli)?;
    let cfg = chrf_config(cli);
    let order = cfg.char_order + cfg.word_order;
    if order == 0 {
        return fail(1, "IndexError: list index out of range (chrF with no n-gram orders)");
    }
    let hyps: Vec<Cow<str>> = system.iter().map(|h| maybe_lower(h, cli.chrf_lowercase)).collect();
    let hyp_view: Vec<&str> = hyps.iter().map(|c| c.as_ref()).collect();
    let seg_owned = segment_refs(refs, cli.chrf_lowercase);
    let seg_refs = as_views(&seg_owned);
    let width = 3 * order;
    let rows = chrf_corpus_stats(&cfg, &hyp_view, &seg_refs);
    let mut total = vec![0i64; width];
    for row in rows.chunks(width.max(1)) {
        for (t, v) in total.iter_mut().zip(row) {
            *t += v;
        }
    }
    let score = chrf_f_score(&total, order, cfg.beta, cfg.eps_smoothing);
    let rows_f: Vec<f64> = rows.iter().map(|&v| v as f64).collect();
    let kind = ScoreKind::Chrf { order, beta: cfg.beta, eps_smoothing: cfg.eps_smoothing };
    let (conf, seed_str) = confidence(cli, &rows_f, width, &kind)?;
    let mut sig = chrf_signature(cli, nrefs);
    if let Some(seed) = &seed_str {
        sig = sig.with_confidence(confidence_samples(cli)?, seed);
    }
    Ok((Score { name: chrf_name(cli), score, verbose: None, confidence: conf }, sig))
}

fn chrf_sentence(cli: &Cli, hyp: &str, refs: &[&str], nrefs: usize) -> Result<(Score, Signature), Fail> {
    check_chrf_orders(cli)?;
    let cfg = chrf_config(cli);
    let order = cfg.char_order + cfg.word_order;
    if order == 0 && cfg.eps_smoothing {
        return fail(1, "ZeroDivisionError: division by zero (chrF eps smoothing with no n-gram orders)");
    }
    let mut row = Vec::new();
    chrf_segment(&cfg, hyp, refs, &mut row);
    let score = chrf_f_score(&row, order, cfg.beta, cfg.eps_smoothing);
    Ok((Score { name: chrf_name(cli), score, verbose: None, confidence: None }, chrf_signature(cli, nrefs)))
}

// ------------------------------------------------------------------ TER

fn ter_config(cli: &Cli) -> TerConfig {
    TerConfig {
        normalized: cli.ter_normalized,
        no_punct: cli.ter_no_punct,
        asian_support: cli.ter_asian_support,
        case_sensitive: cli.ter_case_sensitive,
    }
}

/// `TERSignature`: nrefs, case, tok (always `tercom`), norm, punct (yes unless
/// --ter-no-punct), asian, version. The text is not lowercased here: TER's
/// tokenizer does its own lowercasing.
fn ter_signature(cli: &Cli, nrefs: usize) -> Signature {
    Signature::new(
        nrefs,
        vec![
            ("case", "c", if cli.ter_case_sensitive { "mixed" } else { "lc" }.to_string()),
            ("tok", "t", "tercom".to_string()),
            ("norm", "nr", if cli.ter_normalized { "yes" } else { "no" }.to_string()),
            ("punct", "pn", if cli.ter_no_punct { "no" } else { "yes" }.to_string()),
            ("asian", "as", if cli.ter_asian_support { "yes" } else { "no" }.to_string()),
        ],
    )
}

/// `TER._compute_score_from_stats`: `100 * edits / ref_len`, with the
/// empty-reference rules (1.0 for edits without reference words, 0.0 for nothing).
fn ter_score_value(edits: f64, ref_len: f64) -> f64 {
    let ratio = if ref_len > 0.0 {
        edits / ref_len
    } else if edits > 0.0 {
        1.0
    } else {
        0.0
    };
    100.0 * ratio
}

fn ter_err(_e: TerError) -> Fail {
    // The clean-room TER reports one condition: a segment with no references.
    Fail { code: 1, msg: "ZeroDivisionError: division by zero (TER segment without references)".into() }
}

fn ter_corpus(cli: &Cli, system: &[String], refs: &[Vec<String>], nrefs: usize) -> Result<(Score, Signature), Fail> {
    let cfg = ter_config(cli);
    // The core tokenizer does not lowercase: sacreBLEU's TER lowercases the
    // hypothesis and references first unless --ter-case-sensitive.
    let lower = !cli.ter_case_sensitive;
    let hyps: Vec<Cow<str>> = system.iter().map(|h| maybe_lower(h, lower)).collect();
    let hyp_view: Vec<&str> = hyps.iter().map(|c| c.as_ref()).collect();
    let seg_owned = segment_refs(refs, lower);
    let seg_refs = as_views(&seg_owned);
    let rows = ter_corpus_stats(&cfg, &hyp_view, &seg_refs).map_err(ter_err)?;
    // `sum_of_lists`: sequential per-column sums of [num_edits, avg_ref_len].
    let mut total = [0.0f64; 2];
    for row in rows.chunks(2) {
        total[0] += row[0];
        total[1] += row[1];
    }
    let score = ter_score_value(total[0], total[1]);
    let (conf, seed_str) = confidence(cli, &rows, 2, &ScoreKind::Ter)?;
    let mut sig = ter_signature(cli, nrefs);
    if let Some(seed) = &seed_str {
        sig = sig.with_confidence(confidence_samples(cli)?, seed);
    }
    Ok((Score { name: "TER".into(), score, verbose: None, confidence: conf }, sig))
}

fn ter_sentence(cli: &Cli, hyp: &str, refs: &[&str], nrefs: usize) -> Result<(Score, Signature), Fail> {
    let cfg = ter_config(cli);
    let lower = !cli.ter_case_sensitive;
    let hyp = maybe_lower(hyp, lower);
    let refs_lower: Vec<Cow<str>> = refs.iter().map(|r| maybe_lower(r, lower)).collect();
    let refs_view: Vec<&str> = refs_lower.iter().map(|c| c.as_ref()).collect();
    let [edits, ref_len] = ter_segment_clean(&cfg, &hyp, &refs_view).map_err(ter_err)?;
    let score = ter_score_value(edits, ref_len);
    Ok((Score { name: "TER".into(), score, verbose: None, confidence: None }, ter_signature(cli, nrefs)))
}

/// Per-segment reference lists (transposed from per-reference streams), lowercased if asked.
fn segment_refs(refs: &[Vec<String>], lower: bool) -> Vec<Vec<Cow<'_, str>>> {
    let n = refs.first().map_or(0, |r| r.len());
    (0..n).map(|i| refs.iter().map(|r| maybe_lower(&r[i], lower)).collect()).collect()
}

fn as_views<'a>(segs: &'a [Vec<Cow<'a, str>>]) -> Vec<Vec<&'a str>> {
    segs.iter().map(|v| v.iter().map(|c| c.as_ref()).collect()).collect()
}
