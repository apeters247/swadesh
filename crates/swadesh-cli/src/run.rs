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
use crate::dataset;
use crate::input;
use crate::lower::py_lower;
use crate::pyfmt::fixed;
use crate::pyfmt::json_float;
use crate::report::{json_str, print_single, Score, Signature};
use crate::tabulate::{self, Align};
use crate::{fail, Fail};


/// Tokenizers built into swadesh.
const NATIVE_TOKENIZERS: &[&str] = &["none", "13a", "intl", "zh", "char"];

pub fn execute(cli: &Cli, out: &mut dyn Write) -> Result<(), Fail> {
    // `--list` short-circuits before anything is read, as sacreBLEU does.
    if cli.list {
        // With `-t`, list that set's pairs and fields; an unknown set prints nothing at all and
        // exits 0, which is what the oracle does (measured).
        if let Some(name) = &cli.test_set {
            if let Some(found) = dataset::lookup(name) {
                for (pair, fields) in &found.pairs {
                    // With `-l` only that pair is listed (measured); a pair the set does not have
                    // lists nothing, like an unknown set (the oracle crashes on that combination).
                    if let Some(wanted) = &cli.langpair {
                        if wanted != pair {
                            continue;
                        }
                    }
                    writeln!(out, "{pair}: {}", fields.join(", ")).map_err(io_err)?;
                }
            }
            return Ok(());
        }
        writeln!(out, "The available test sets are:").map_err(io_err)?;
        for ds in crate::datasets::DATASETS {
            writeln!(out, "{:<30}: {}", ds.name, ds.description).map_err(io_err)?;
        }
        return Ok(());
    }
    if cli.sentence_level && cli.metrics.len() > 1 {
        eprintln!("sacreBLEU: Only one metric can be used in sentence-level mode.");
        return Err(Fail { code: 1, msg: String::new() });
    }
    // `--download NAME -l PAIR` fetches the set's archive and quits. The oracle raises when `-l` is
    // missing (registered divergence: swadesh refuses in one line, same exit code, because the
    // archive is per set).
    if let Some(name) = &cli.download {
        if cli.langpair.is_none() {
            eprintln!("sacreBLEU: I need a language pair (-l). Use --list to see available language pairs for this test set.");
            return Err(Fail { code: 1, msg: String::new() });
        }
        #[cfg(feature = "downloader")]
        {
            if !crate::fetch::download_set(name)? {
                eprintln!("sacreBLEU: Unknown test set '{name}'");
                eprintln!("sacreBLEU: Please run with --list to see the available test sets.");
                return Err(Fail { code: 1, msg: String::new() });
            }
        }
        #[cfg(not(feature = "downloader"))]
        {
            let _ = name;
            eprintln!("swadesh: error: this build has no downloader (--no-default-features)");
            return Err(Fail { code: 1, msg: String::new() });
        }
        return Ok(());
    }

    // `--citation` needs a test set: sacreBLEU has no default citation, and the check comes before
    // the reference checks but after the sentence-level one (both measured). With `-t` the oracle
    // crashes (`TypeError: argument of type 'WMTXMLDataset' is not iterable`), so printing the
    // catalogue's own citation is a documented divergence rather than a reproduction of the crash.
    if cli.citation && cli.test_set.is_none() {
        eprintln!("sacreBLEU: I need a test set (-t).");
        return Err(Fail { code: 1, msg: String::new() });
    }

    // The test set, when `-t` names one: the references come from it, so the reference checks
    // below do not apply. Measured order: the `-t`-versus-positional conflict, then the missing
    // language pair, then the set lookup, then the pair.
    let mut test_set: Option<(dataset::Found, String)> = None;
    if let Some(name) = cli.test_set.clone() {
        if !cli.refs.is_empty() {
            eprintln!("sacreBLEU: I need exactly one of (a) a predefined test set (-t) or (b) a list of references");
            return Err(Fail { code: 1, msg: String::new() });
        }
        let found = dataset::lookup(&name).ok_or_else(|| dataset::unknown_set(&name))?;
        if cli.citation {
            // The citation of a built-in set is in the catalogue; a user set may carry `.citation`.
            let citation = dataset::citation_of(&name)
                .or_else(|| dataset::read_field(&found.dir.join("citation")).ok().map(|l| l.join("\n")));
            match citation {
                Some(text) => writeln!(out, "{text}").map_err(io_err)?,
                None => {
                    eprintln!("sacreBLEU: No citation for test set '{name}'.");
                    return Err(Fail { code: 1, msg: String::new() });
                }
            }
            return Ok(());
        }
        let Some(pair) = cli.langpair.clone() else {
            eprintln!("sacreBLEU: I need a language pair (-l). Use --list to see available language pairs for this test set.");
            return Err(Fail { code: 1, msg: String::new() });
        };
        if found.fields_of(&pair).is_none() {
            let available: Vec<String> = found.pairs.iter().map(|(p, _)| p.clone()).collect();
            return Err(dataset::unknown_pair(&name, &pair, &available));
        }
        test_set = Some((found, pair));
    }

    // A cached set whose processed files are missing is rebuilt from its raw tree when the family
    // is the measured SGML one - sacreBLEU regenerates them the same way, which is why removing a
    // sidecar brings it back. A family the measured rule does not cover keeps the usual refusal.
    if let Some((found, pair)) = &test_set {
        let fields = found.fields_of(pair).cloned().unwrap_or_default();
        if fields.iter().any(|field| !dataset::field_path(found, pair, field).exists()) {
            // Each family says whether it recognises the layout. A raw tree may already be there; if
            // not, the downloader fetches it (when the build has one and the set is single-archive),
            // and then the parsers run again. An unknown layout or format leaves the refusal below
            // intact rather than producing plausible-looking wrong data.
            // Each family answers only for the layouts it measured; the order does not matter
            // because each refuses anything it does not recognise.
            let rebuild = |found: &dataset::Found, pair: &str| -> Result<bool, Fail> {
                for step in [crate::sgml::rebuild, crate::wmtxml::rebuild, crate::plain::rebuild] {
                    if step(found, pair)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            };
            if !rebuild(found, pair)? {
                #[cfg(feature = "downloader")]
                if crate::fetch::ensure_archive(found)? {
                    rebuild(found, pair)?;
                }
            }
        }
    }

    // `-ol`/`--subset` are segment filters, and they apply to the whole segment set: the reference,
    // the system output *by position*, and what `--echo` prints. Both need a test set - without one
    // the oracle raises a traceback, so swadesh reports a clean error instead (registered).
    let mut filter_without_set = false;
    let mask: Option<Vec<usize>> = match (&test_set, cli.origlang.as_deref(), cli.subset.as_deref()) {
        (None, None, None) => None,
        (None, _, _) => {
            filter_without_set = true;
            None
        }
        (Some((found, pair)), origlang, subset) => {
            if origlang.is_none() && subset.is_none() {
                None
            } else {
                Some(dataset::segment_filter(found, pair, origlang, subset)?)
            }
        }
    };
    if let Some(kept) = &mask {
        // Nothing matched: `--echo` prints nothing and exits 0 (measured), scoring refuses with the
        // oracle's line - including its doubled subset value.
        if kept.is_empty() && cli.echo.is_empty() {
            let name = &test_set.as_ref().expect("a test set implies a mask").0.name;
            return Err(dataset::no_sentence(name, cli.origlang.as_deref(), cli.subset.as_deref()));
        }
    }

    // `--echo FIELD...`: print the set's text and quit. `all` means every field, in the set's own
    // order; anything else must be a field the pair has.
    if let Some((found, pair)) = &test_set {
        if !cli.echo.is_empty() {
            let fields = found.fields_of(pair).expect("the pair was checked above");
            let default_ref = found.default_reference(pair);
            let wanted: Vec<String> = if cli.echo.iter().any(|e| e == "all") {
                fields.clone()
            } else {
                let mut wanted = Vec::with_capacity(cli.echo.len());
                for field in &cli.echo {
                    if fields.contains(field) {
                        wanted.push(field.clone());
                    } else if field == "ref" && default_ref.is_some() {
                        // `ref` stays accepted as an alias for the default reference, which the
                        // oracle documents in its own field error ("will return ref:A").
                        wanted.push(default_ref.clone().expect("checked"));
                    } else {
                        return Err(dataset::unknown_field(
                            &found.name,
                            pair,
                            field,
                            fields,
                            default_ref.as_deref(),
                        ));
                    }
                }
                wanted
            };
            let mut columns: Vec<Vec<String>> = Vec::with_capacity(wanted.len());
            for field in &wanted {
                let path = dataset::field_path(found, pair, field);
                if !path.exists() {
                    return Err(dataset::not_cached(found, pair, field, &path));
                }
                columns.push(dataset::read_field(&path)?);
            }
            let rows: Vec<usize> = match &mask {
                Some(kept) => kept.clone(),
                None => (0..columns.iter().map(|c| c.len()).max().unwrap_or(0)).collect(),
            };
            for i in rows {
                let line: Vec<&str> =
                    columns.iter().map(|c| c.get(i).map(String::as_str).unwrap_or("")).collect();
                writeln!(out, "{}", line.join("\t")).map_err(io_err)?;
            }
            return Ok(());
        }
    }
    if cli.num_refs != 1 && cli.refs.len() > 1 {
        return fail(
            1,
            "The --num-refs argument allows you to provide any number of tab-delimited references \
             in a single file. You cannot then provide multiple reference files.",
        );
    }
    if cli.refs.is_empty() && test_set.is_none() {
        eprintln!(
            "sacreBLEU: If manual references given, make sure to provide them before the \
             -i/--input argument to avoid confusion."
        );
        eprintln!("sacreBLEU: Otherwise, I need a predefined test set (-t) from the following list:");
        let names: Vec<String> =
            crate::datasets::DATASETS.iter().rev().map(|d| format!("'{}'", d.name)).collect();
        eprintln!("sacreBLEU: [{}]", names.join(", "));
        return Err(Fail { code: 1, msg: String::new() });
    }
    // A segment filter needs a test set. With no references at all the oracle prints its own
    // catalogue message (which the check above already did); with references it raises a traceback,
    // so swadesh refuses in one line instead (registered).
    if filter_without_set {
        eprintln!("sacreBLEU: I need a test set (-t) to filter segments with -ol/--subset.");
        return Err(Fail { code: 1, msg: String::new() });
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

    // The references: the set's default reference when `-t` was used, else the positional files.
    let refs: Vec<Vec<String>> = match &test_set {
        Some((found, pair)) => {
            let field = found.default_reference(pair).unwrap_or_else(|| "ref".to_string());
            let path = dataset::field_path(found, pair, &field);
            if !path.exists() {
                return Err(dataset::not_cached(found, pair, &field, &path));
            }
            vec![dataset::read_field(&path)?]
        }
        None => read_refs(cli)?,
    };
    let (systems, sys_names) = read_systems(cli)?;
    // A segment filter keeps the same indices in the reference and in every system, which is why a
    // full-length system output scores against a filtered reference (measured).
    let select = |lines: &[String]| -> Vec<String> {
        match &mask {
            Some(kept) => kept.iter().filter_map(|&i| lines.get(i).cloned()).collect(),
            None => lines.to_vec(),
        }
    };
    let refs: Vec<Vec<String>> = refs.iter().map(|r| select(r)).collect();
    let systems: Vec<Vec<String>> = systems.iter().map(|s| select(s)).collect();

    // sacreBLEU's order of checks is measured, not chosen (docs/lab-notebook.md). The systems are
    // named first: the paired tests drop a later system that shares the baseline's name, with a
    // note. Duplicate names and the sentence-level refusals then fire, and only after them are the
    // streams validated (so `-i h1 h2 -ci` reports the paired-test hint even when the lengths
    // differ).
    let paired = cli.paired_ar || cli.paired_bs;
    let keep: Vec<usize> = if paired {
        let mut keep = vec![0];
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
        keep
    } else {
        (0..systems.len()).collect()
    };
    // Two systems with the same name: sacreBLEU prints both lines itself and exits 1 before any
    // output. The paired tests run their own pass (they tolerate a repeated baseline).
    if !paired && systems.len() > 1 {
        let mut seen = std::collections::HashSet::new();
        for name in &sys_names {
            if !seen.insert(name.as_str()) {
                eprintln!("sacreBLEU: '{name}' already used to name a system.");
                eprintln!("sacreBLEU: Make sure to have a different basename for each system.");
                return Err(Fail { code: 1, msg: String::new() });
            }
        }
    }
    if keep.len() > 1 && cli.sentence_level {
        eprintln!("sacreBLEU: Only one system can be evaluated in sentence-level mode.");
        return Err(Fail { code: 1, msg: String::new() });
    }
    if paired {
        if cli.sentence_level {
            eprintln!("sacreBLEU: Statistical tests are unavailable in sentence-level mode.");
            return Err(Fail { code: 1, msg: String::new() });
        }
        if keep.len() < 2 {
            eprintln!("sacreBLEU: Paired tests require multiple input systems given to --input (-i).");
            return Err(Fail { code: 1, msg: String::new() });
        }
        // Any repeated name other than the baseline is an error.
        let mut seen = std::collections::HashSet::new();
        for i in keep.iter().skip(1) {
            if !seen.insert(sys_names[*i].as_str()) {
                eprintln!("sacreBLEU: '{}' already used to name a system.", sys_names[*i]);
                eprintln!("sacreBLEU: Make sure to have a different basename for each system.");
                return Err(Fail { code: 1, msg: String::new() });
            }
        }
    }
    // `-ci` is a single-system report: sacreBLEU points at the paired tests for several systems,
    // even when the paired tests were also asked for.
    if cli.confidence && systems.len() > 1 {
        eprintln!("sacreBLEU: Use paired tests (--paired) for multiple systems.");
        return Err(Fail { code: 1, msg: String::new() });
    }
    if systems.iter().any(|s| s.is_empty()) {
        eprintln!("sacreBLEU: Test set None contains no sentence");
        return Err(Fail { code: 1, msg: String::new() });
    }
    if systems.iter().any(|s| refs.iter().any(|r| r.len() != s.len())) {
        eprintln!("sacreBLEU: System and reference streams have different lengths.");
        // With a test set the oracle adds a cache hint naming the set's directory (measured).
        if let Some((found, _)) = &test_set {
            eprintln!("sacreBLEU: This could be an issue with your system output or with sacreBLEU's reference database if -t is given.");
            eprintln!("sacreBLEU: For the latter, try cleaning out the cache by typing:");
            eprintln!();
            eprintln!("sacreBLEU:   rm -r {}", found.dir.display());
            eprintln!();
            eprintln!("sacreBLEU: The test sets will be re-downloaded the next time you run sacreBLEU.");
        }
        return Err(Fail { code: 1, msg: String::new() });
    }
    let nrefs = refs.len();
    let width = cli.width as usize;
    let json = cli.format == "json";

    // The `-m` names become upper-case metric names; duplicates collapse, output is sorted.
    let mut names: Vec<String> = cli.metrics.iter().map(|m| m.to_uppercase()).collect();
    names.sort();
    names.dedup();

    // Paired significance testing: the first system is the baseline, the others are compared to it.
    if paired {
        let bs = cli.paired_bs;
        // `-pbsn` and `-parn` fall back to their defaults when not positive, as in sacreBLEU.
        let n_samples = if bs {
            if cli.paired_bs_n <= 0 { 1000 } else { cli.paired_bs_n }
        } else if cli.paired_ar_n <= 0 {
            10000
        } else {
            cli.paired_ar_n
        };
        let n_samples = n_samples as usize;
        let (seed, seed_str) = bootstrap_seed()?;
        let n_ar_conf = 0usize; // `-ci` with several systems is refused above, as in sacreBLEU
        eprintln!("sacreBLEU: Found {} systems.", keep.len());
        // The name the report gives each system. sacreBLEU prefixes the baseline's own name with
        // `Baseline: ` when the systems came from files, but labels it plain `Baseline` for stdin
        // systems, where the others are then renumbered from 1.
        let from_stdin = cli.input.is_none();
        let display: Vec<String> = (0..systems.len())
            .map(|k| {
                if from_stdin {
                    if k == 0 {
                        "Baseline".to_string()
                    } else {
                        format!("System {k}")
                    }
                } else if k == 0 {
                    format!("Baseline: {}", sys_names[0])
                } else {
                    sys_names[k].clone()
                }
            })
            .collect();
        let bl_display = display[0].clone();
        // The footer names the baseline with the file-derived label, but with nothing at all for
        // stdin systems -- sacreBLEU prints `compared to .` there, which is kept as measured.
        let footer_baseline = if from_stdin { String::new() } else { bl_display.clone() };

        // The paired signature carries `bs`/`ar` and the seed between `nrefs` and the
        // metric-specific pairs; the text/latex footer prints it per metric.
        let mut sigs: Vec<Signature> = Vec::with_capacity(names.len());
        for metric in &names {
            let base = match metric.as_str() {
                "BLEU" => bleu_signature(cli, bleu_tokenizer(cli)?, nrefs, false),
                "TER" => ter_signature(cli, nrefs),
                _ => chrf_signature(cli, nrefs),
            };
            sigs.push(base.with_paired(bs, n_samples, &seed_str));
        }

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
                    sys.name, display[*i]
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

        if json {
            let mut rows_out: Vec<String> = Vec::with_capacity(keep.len());
            for (k, entry) in table.iter().enumerate() {
                let shown = display[keep[k]].clone();
                let mut lines: Vec<String> =
                    vec![format!("        \"system\": {}", json_str(&shown))];
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
                    lines.push(format!(
                        "        {}: {{\n{}\n        }}",
                        json_str(name),
                        fields.join(",\n")
                    ));
                }
                rows_out.push(format!("    {{\n{}\n    }}", lines.join(",\n")));
            }
            writeln!(out, "[\n{}\n]", rows_out.join(",\n")).map_err(io_err)?;
            return Ok(());
        }

        // Text and latex: the same grid as the multi-system report. The heading carries the
        // interval for `--paired-bs`, the baseline row is the score alone, and every other system
        // takes a second grid line (a newline inside its cell) holding its p-value -- in latex
        // tabulate joins that line onto the score with a space instead.
        let mut header: Vec<String> = vec!["System".to_string()];
        for (name, ..) in &table[0] {
            header.push(if bs {
                format!("{name} (\u{3bc} \u{b1} 95% CI)")
            } else {
                name.clone()
            });
        }
        let mut body: Vec<Vec<String>> = Vec::with_capacity(keep.len());
        for (k, entry) in table.iter().enumerate() {
            let mut cells: Vec<String> = vec![display[keep[k]].clone()];
            for (_, score, p, mean, ci) in entry {
                let mut cell = match (mean, ci) {
                    (Some(m), Some(c)) => format!(
                        "{} ({} \u{b1} {})",
                        fixed(*score, width),
                        fixed(*m, width),
                        fixed(*c, width)
                    ),
                    _ => fixed(*score, width),
                };
                if let Some(p) = p {
                    let star = if *p < 0.05 { "*" } else { "" };
                    cell.push_str(&format!("\n(p = {p:.4}){star}"));
                }
                cells.push(cell);
            }
            body.push(cells);
        }
        let mut aligns: Vec<Align> = vec![Align::Right];
        aligns.extend(std::iter::repeat_n(Align::Center, header.len() - 1));
        if cli.format == "latex" {
            write!(out, "{}", tabulate::latex_table(&header, &body, &aligns)).map_err(io_err)?;
        } else {
            write!(out, "{}", tabulate::fancy_grid(&header, &body, &aligns)).map_err(io_err)?;
        }
        writeln!(out).map_err(io_err)?;
        write!(out, "{}", paired_footer(bs, n_samples, &footer_baseline)).map_err(io_err)?;
        write!(out, "-----------------\nMetric signatures\n-----------------\n").map_err(io_err)?;
        for (metric, sig) in table[0].iter().zip(&sigs) {
            writeln!(out, " - {:<11}{}", metric.0, sig.format(cli.short)).map_err(io_err)?;
        }
        return Ok(());
    }

    // Multi-system mode: sacreBLEU reports every system side by side instead of one score. The
    // text and latex forms are `tabulate`'s `fancy_grid` / `latex_booktabs`; the port lives in
    // `crate::tabulate` and its geometry is measured from sacreBLEU (docs/lab-notebook.md).
    if systems.len() > 1 {
        // Duplicate names and `-sl` were refused above, in sacreBLEU's own order.
        eprintln!("sacreBLEU: Found {} systems.", systems.len());

        // Score every system for every metric. The column headings are the score objects' own
        // names (BLEU / chrF2 / chrF2++ / TER) and the signatures are identical across systems,
        // so they are taken from the first.
        let mut header: Vec<String> = vec!["System".to_string()];
        let mut sigs: Vec<(String, String)> = Vec::new();
        let mut body: Vec<Vec<String>> = Vec::with_capacity(systems.len());
        for (i, (system, name)) in systems.iter().zip(&sys_names).enumerate() {
            let mut cells: Vec<String> = vec![name.clone()];
            for metric_name in &names {
                let (score, sig) = match metric_name.as_str() {
                    "BLEU" => bleu_corpus(cli, bleu_tokenizer(cli)?, system, &refs, nrefs)?,
                    "TER" => ter_corpus(cli, system, &refs, nrefs)?,
                    _ => chrf_corpus(cli, system, &refs, nrefs)?,
                };
                if i == 0 {
                    header.push(score.name.clone());
                    sigs.push((score.name.clone(), sig.format(cli.short)));
                }
                cells.push(fixed(score.score, width));
            }
            body.push(cells);
        }

        if json {
            let mut rows: Vec<String> = Vec::with_capacity(body.len());
            for cells in &body {
                let mut entries: Vec<String> =
                    vec![format!("        \"system\": {}", json_str(&cells[0]))];
                for (j, metric_name) in header.iter().skip(1).enumerate() {
                    entries.push(format!(
                        "        {}: {}",
                        json_str(metric_name),
                        json_str(&cells[j + 1])
                    ));
                }
                rows.push(format!("    {{\n{}\n    }}", entries.join(",\n")));
            }
            writeln!(out, "[\n{}\n]", rows.join(",\n")).map_err(io_err)?;
            return Ok(());
        }

        let mut aligns: Vec<Align> = vec![Align::Right];
        aligns.extend(std::iter::repeat_n(Align::Center, header.len() - 1));
        if cli.format == "latex" {
            write!(out, "{}", tabulate::latex_table(&header, &body, &aligns)).map_err(io_err)?;
        } else {
            write!(out, "{}", tabulate::fancy_grid(&header, &body, &aligns)).map_err(io_err)?;
        }
        writeln!(out).map_err(io_err)?;
        write!(out, "-----------------\nMetric signatures\n-----------------\n").map_err(io_err)?;
        for (name, sig) in &sigs {
            writeln!(out, " - {:<11}{}", name, sig).map_err(io_err)?;
        }
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

    // `-d/--detail` breaks the corpus score down by group, and it forces the text form: measured,
    // `-d -f json` prints the text score line followed by the detail lines.
    let detail = cli.detail && !cli.sentence_level;
    let json = json && !detail;
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
    if detail {
        match &test_set {
            // Without a test set there is nothing to group by, and the oracle says so on stdout.
            None => writeln!(out, "No subset information found. Consider using --origlang argument.")
                .map_err(io_err)?,
            Some((found, pair)) => write_detail(
                cli, out, found, pair, &names, &systems[0], &refs, nrefs, width, &mask,
            )?,
        }
    }
    Ok(())
}

/// The `-d/--detail` block: one line per group per metric, in sacreBLEU's measured layout. The
/// columns are padded to their widest value - the domain name (so `ALL` lines up with
/// `conversation`), the sentence count, and the metric's display name (so `BLEU` lines up with
/// `chrF2`).
#[allow(clippy::too_many_arguments)]
fn write_detail(
    cli: &Cli,
    out: &mut dyn Write,
    found: &dataset::Found,
    pair: &str,
    names: &[String],
    system: &[String],
    refs: &[Vec<String>],
    nrefs: usize,
    width: usize,
    mask: &Option<Vec<usize>>,
) -> Result<(), Fail> {
    let groups = dataset::groups(found, pair)?;
    // `groups` indexes the unfiltered set; `system`/`refs` are already masked, so the group is
    // translated into positions within the mask. (A segment filter and `-d` together make the
    // oracle raise a traceback; swadesh reports the groups of the filtered set instead - registered.)
    let positions_of = |idx: &Vec<usize>| -> Vec<usize> {
        match mask {
            None => idx.clone(),
            Some(kept) => kept
                .iter()
                .enumerate()
                .filter(|(_, original)| idx.contains(original))
                .map(|(position, _)| position)
                .collect(),
        }
    };
    struct Row {
        lang: String,
        domain: String,
        sentences: usize,
        metric: String,
        score: f64,
    }
    let mut rows: Vec<Row> = Vec::new();
    for (lang, domain, idx) in &groups {
        let idx = positions_of(idx);
        if idx.is_empty() {
            continue;
        }
        let sub_system: Vec<String> = idx.iter().map(|&i| system[i].clone()).collect();
        let sub_refs: Vec<Vec<String>> =
            refs.iter().map(|r| idx.iter().map(|&i| r[i].clone()).collect()).collect();
        for name in names {
            let (score, _sig) = match name.as_str() {
                "BLEU" => bleu_corpus(cli, bleu_tokenizer(cli)?, &sub_system, &sub_refs, nrefs)?,
                "TER" => ter_corpus(cli, &sub_system, &sub_refs, nrefs)?,
                _ => chrf_corpus(cli, &sub_system, &sub_refs, nrefs)?,
            };
            rows.push(Row {
                lang: lang.clone(),
                domain: domain.clone(),
                sentences: idx.len(),
                metric: score.name.clone(),
                score: score.score,
            });
        }
    }
    let w_domain = rows.iter().map(|r| r.domain.chars().count()).max().unwrap_or(0);
    // The sentence count is left-justified in a field of at least four characters: wmt24's counts
    // are all three digits (997/206/149/531/111) and the oracle still pads them to four. No cached
    // set has a five-digit group, so the upper behaviour is unmeasured; `max(4, digits)` is the
    // reading that keeps every measured case exact.
    let w_sentences = rows.iter().map(|r| r.sentences.to_string().len()).max().unwrap_or(1).max(4);
    let w_metric = rows.iter().map(|r| r.metric.chars().count()).max().unwrap_or(0);
    for r in &rows {
        writeln!(
            out,
            "origlang={} domain={:<w_domain$} : sentences={:<w_sentences$}   {:<w_metric$} = {}",
            r.lang,
            r.domain,
            r.sentences,
            r.metric,
            fixed(r.score, width)
        )
        .map_err(io_err)?;
    }
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

/// The bootstrap interval when `-ci` was given, with the `(samples, seed)` the signature reports.
///
/// Two measured quirks of sacreBLEU 2.6.0:
///   * `--confidence-n` of 1 or less drops `-ci` silently: no interval, and no `bs`/`seed` pair.
///   * a resampled mean of exactly 0.0 makes its interval undefined, so it reports the score alone
///     -- but the `bs`/`seed` pair stays in the signature.
fn confidence(
    cli: &Cli,
    rows: &[f64],
    width: usize,
    kind: &ScoreKind,
) -> Result<(Option<(f64, f64)>, Option<(usize, String)>), Fail> {
    if !cli.confidence || cli.confidence_n <= 1 {
        return Ok((None, None));
    }
    let n = cli.confidence_n as usize;
    let (seed, seed_str) = bootstrap_seed()?;
    let (mean, half_width) = bootstrap_ci(rows, width, kind, n, seed);
    let interval = if mean == 0.0 { None } else { Some((mean, half_width)) };
    Ok((interval, Some((n, seed_str))))
}

/// The explanatory block under the paired table. Every line is measured: the rule is as long as
/// the title, and the sentence about the baseline names it with an empty string, as sacreBLEU does.
fn paired_footer(bootstrap: bool, n_samples: usize, baseline: &str) -> String {
    let title = if bootstrap {
        format!("Paired bootstrap resampling test with {n_samples} resampling trials")
    } else {
        format!("Paired approximate randomization test with {n_samples} trials")
    };
    let detail = if bootstrap {
        "   Actual system score / bootstrap estimated true mean / 95% CI are provided for each metric."
    } else {
        "   Actual system score is provided for each metric."
    };
    let rule = "-".repeat(title.chars().count());
    let mut s = String::new();
    s.push_str(&format!("{rule}\n{title}\n{rule}\n"));
    s.push_str(&format!(" - Each system is pairwise compared to {baseline}.\n"));
    s.push_str(detail);
    s.push_str("\n\n");
    s.push_str(" - Null hypothesis: the system and the baseline translations are essentially\n");
    s.push_str("   generated by the same underlying process. For a given system and the baseline,\n");
    s.push_str("   the p-value is roughly the probability of the absolute score difference (delta)\n");
    s.push_str(
        "   or higher occurring due to chance, under the assumption that the null hypothesis is correct.\n",
    );
    s.push('\n');
    s.push_str(" - Assuming a significance threshold of 0.05, the null hypothesis can be rejected\n");
    s.push_str(
        "   for p-values < 0.05 (marked with \"*\"). This means that the delta is unlikely to be attributed\n",
    );
    s.push_str("   to chance, hence the system is significantly \"different\" than the baseline.\n");
    s.push_str("   Otherwise, the p-values are highlighted in red.\n");
    s.push('\n');
    s.push_str(" - NOTE: Significance does not tell whether a system is \"better\" than the baseline but rather\n");
    s.push_str(
        "   emphasizes the \"difference\" of the systems in terms of the replicability of the delta.\n",
    );
    s.push('\n');
    s
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
    let (conf, bootstrap) = confidence(cli, &rows_f, width, &ScoreKind::Bleu(scfg))?;
    let mut sig = bleu_signature(cli, tok, nrefs, false);
    if let Some((samples, seed)) = &bootstrap {
        sig = sig.with_confidence(*samples, seed);
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
    let (conf, bootstrap) = confidence(cli, &rows_f, width, &kind)?;
    let mut sig = chrf_signature(cli, nrefs);
    if let Some((samples, seed)) = &bootstrap {
        sig = sig.with_confidence(*samples, seed);
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
    let (conf, bootstrap) = confidence(cli, &rows, 2, &ScoreKind::Ter)?;
    let mut sig = ter_signature(cli, nrefs);
    if let Some((samples, seed)) = &bootstrap {
        sig = sig.with_confidence(*samples, seed);
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
