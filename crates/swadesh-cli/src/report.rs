//! Signatures and score lines, mirroring sacreBLEU's `Signature.format`,
//! `Score.format` and `print_single_results`.

use std::io::{self, Write};

use crate::pyfmt::{fixed, json_float};

/// sacreBLEU version that signatures claim, so they are interchangeable with it.
pub const SACREBLEU_VERSION: &str = "2.6.0";

/// sacreBLEU's signature: `(long key, short key, value)` triples in print order.
#[derive(Debug, Clone)]
pub struct Signature {
    pairs: Vec<(&'static str, &'static str, String)>,
}

impl Signature {
    /// `middle` holds the metric-specific pairs; `nrefs` comes first and `version` last.
    pub fn new(nrefs: usize, middle: Vec<(&'static str, &'static str, String)>) -> Self {
        let mut pairs = vec![("nrefs", "#", nrefs.to_string())];
        pairs.extend(middle);
        pairs.push(("version", "v", SACREBLEU_VERSION.to_string()));
        Signature { pairs }
    }

    /// Add the bootstrap pairs sacreBLEU puts between `nrefs` and the metric-specific ones.
    pub fn with_confidence(mut self, n_samples: usize, seed: &str) -> Self {
        self.pairs.insert(1, ("bs", "bs", n_samples.to_string()));
        self.pairs.insert(2, ("seed", "rs", seed.to_string()));
        self
    }

    /// `Signature.format(short)`.
    pub fn format(&self, short: bool) -> String {
        self.pairs
            .iter()
            .map(|(long, abbr, value)| format!("{}:{}", if short { abbr } else { long }, value))
            .collect::<Vec<_>>()
            .join("|")
    }
}

/// A computed metric value, before formatting.
#[derive(Debug, Clone)]
pub struct Score {
    pub name: String,
    pub score: f64,
    /// BLEU's verbose part, e.g. "46.3/16.0/... (BP = 1.000 ratio = ...)".
    pub verbose: Option<String>,
    /// `(mean, half_width)` when `-ci` asked for a bootstrap confidence interval.
    pub confidence: Option<(f64, f64)>,
}

impl Score {
    /// `Score.format(width, score_only, signature, is_json)`. The signature is
    /// printed short when `short` (text only); JSON always carries the long form.
    pub fn format(&self, width: usize, score_only: bool, sig: &Signature, short: bool, is_json: bool) -> String {
        // sacreBLEU appends the confidence to the score string before the `score_only` check,
        // so `-b -ci` prints "37.9 (μ = 37.1 ± 14.3)".
        let sc_plain = fixed(self.score, width);
        let conf: Option<(String, String)> = self
            .confidence
            .map(|(m, c)| (fixed(m, width), fixed(c, width)));
        let mut sc = sc_plain.clone();
        if let Some((m, c)) = &conf {
            sc.push_str(&format!(" (\u{3bc} = {} \u{b1} {})", m, c));
        }
        if score_only {
            return sc;
        }
        if is_json {
            return self.format_json(&sc_plain, &conf, sig);
        }
        let mut line = format!("{}|{} = {}", self.name, sig.format(short), sc);
        if let Some(v) = &self.verbose {
            line.push(' ');
            line.push_str(v);
        }
        line
    }

    fn format_json(&self, sc: &str, conf: &Option<(String, String)>, sig: &Signature) -> String {
        let score_value: f64 = sc.parse().unwrap_or(f64::NAN);
        let mut entries: Vec<(String, String)> = vec![
            ("name".into(), json_str(&self.name)),
            ("score".into(), json_float(score_value)),
            ("signature".into(), json_str(&sig.format(false))),
        ];
        if let Some((m, c)) = conf {
            entries.push(("confidence_mean".into(), json_float(m.parse().unwrap_or(f64::NAN))));
            entries.push(("confidence_var".into(), json_float(c.parse().unwrap_or(f64::NAN))));
            entries.push(("confidence".into(), json_str(&format!("\u{3bc} = {} \u{b1} {}", m, c))));
        }
        if let Some(v) = &self.verbose {
            entries.push(("verbose_score".into(), json_str(v)));
        }
        for (long, _, value) in &sig.pairs {
            entries.push(((*long).to_string(), json_str(value)));
        }
        let body: Vec<String> = entries.iter().map(|(k, v)| format!(" {}: {}", json_str(k), v)).collect();
        format!("{{\n{}\n}}", body.join(",\n"))
    }
}

/// `json.dumps` for a string with `ensure_ascii=False`.
fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// `print_single_results`: one or more formatted metric results.
pub fn print_single(results: &[String], json: bool, out: &mut dyn Write) -> io::Result<()> {
    if json {
        if results.len() > 1 {
            writeln!(out, "[\n{}\n]", results.join(",\n"))?;
        } else {
            writeln!(out, "{}", results[0])?;
        }
        return Ok(());
    }
    if results.len() == 1 {
        return writeln!(out, "{}", results[0]);
    }
    // Align the "=" signs of several metrics, as sacreBLEU does.
    let mut lens: Vec<usize> = Vec::new();
    for line in results {
        match line.find('=') {
            Some(pos) => lens.push(pos.saturating_sub(1)),
            None => writeln!(out, "{}", line)?,
        }
    }
    if let Some(&w) = lens.iter().max() {
        for (len, line) in lens.iter().zip(results) {
            let (left, right) = line.split_at(*len);
            writeln!(out, "{:>w$}{}", left, right, w = w)?;
        }
    }
    Ok(())
}
