//! Command-line parsing. It follows argparse's rules for the options swadesh
//! understands (unambiguous long-option prefixes, `--opt=value`, attached short
//! values, `-qb` flag bundling, negative numbers as values, `--`, choices).
//! Options that swadesh does not implement are rejected with a pointer to the
//! Python package.

use std::collections::VecDeque;

use crate::{fail, Fail};

pub const METRIC_CHOICES: &[&str] = &["bleu", "chrf", "ter"];
pub const SMOOTH_CHOICES: &[&str] = &["none", "floor", "add-k", "exp"];
pub const FORMAT_CHOICES: &[&str] = &["json", "text", "latex"];
/// sacreBLEU's full tokenizer list; swadesh builds in only the first five.
pub const TOKENIZE_CHOICES: &[&str] = &[
    "none", "zh", "13a", "intl", "char", "ja-mecab", "ko-mecab", "spm", "flores101", "flores200",
    "spBLEU-1K",
];

pub const USAGE: &str = "\
usage: swadesh [-h] [-V] [-m {bleu,chrf,ter} ...] [-i INPUT] [options] [REF ...]

Compute BLEU, chrF and/or TER. Hypotheses are read from stdin unless -i is given.
Output is byte-identical to `sacrebleu REF -i HYP` (sacreBLEU 2.6.0).

positional arguments:
  REF                   reference file(s), one line per segment

options:
  -i, --input FILE      hypothesis file (default: stdin)
  -m, --metrics {bleu,chrf,ter} ...
                        metrics to compute (default: bleu)
  -sl, --sentence-level
                        score each segment (text output, one metric)
  -s, --smooth-method {none,floor,add-k,exp}
                        BLEU smoothing (default: exp)
  -sv, --smooth-value X
                        value for floor / add-k smoothing
  -tok, --tokenize {none,zh,13a,intl,char}
                        BLEU tokenizer (default: 13a)
  -lc, --lowercase      BLEU: lowercase input
  --force               BLEU: suppress the tokenized-input warning
  -ci, --confidence     report a bootstrap confidence interval
  -cin, --confidence-n N
                        bootstrap resamples for the confidence interval (default: 1000)
  -par, --paired-ar     paired approximate-randomisation test between systems
  -pbs, --paired-bs     paired bootstrap resampling test between systems
  -parn, --paired-ar-n N
                        trials for --paired-ar (default: 10000)
  -pbsn, --paired-bs-n N
                        resamples for --paired-bs (default: 1000)
  -pj, --paired-jobs N  worker processes for the paired test (default: 1)
  --list                print the available test sets and quit
  --citation            print the citation for a test set (-t)
  --max-ngram-order N   BLEU maximum n-gram order (default: 4)
  --python-version X.Y  emulate sacreBLEU running on this CPython (float sum()
                        semantics changed in 3.12 and 3.14; default: 3.12)
  -cc, --chrf-char-order N
                        chrF character n-gram order (default: 6)
  -cw, --chrf-word-order N
                        chrF word n-gram order (2 gives chrF++; default: 0)
  --chrf-beta N         chrF beta (default: 2)
  --chrf-whitespace     chrF: keep whitespace in character n-grams
  --chrf-lowercase      chrF: lowercase input
  --chrf-eps-smoothing  chrF: epsilon smoothing instead of effective order
  -l, --language-pair XX-YY
                        target language, only used to pick the default tokenizer
  -e, --encoding ENC    input encoding (only utf-8)
  -nr, --num-refs N     split each reference line on tabs into N references
  -f, --format {json,text,latex}
                        output format (default: json; env SACREBLEU_FORMAT)
  -b, --score-only      print only the score
  -w, --width N         decimal places (default: 1)
  -sh, --short          shorter signature (text output)
  --ter-case-sensitive  TER: case sensitive (default: lowercase)
  --ter-asian-support   TER: special treatment of Asian characters
  --ter-no-punct        TER: remove punctuation
  --ter-normalized      TER: basic normalization and tokenization
  -q, --quiet           suppress informational messages
  -nc, --no-color       accepted for compatibility (no colors are used)
  -V, --version         print swadesh and sacreBLEU versions
  -h, --help            show this help
";


/// The usage block argparse prints above an argument error, transcribed from sacreBLEU 2.6.0 with
/// the program name substituted (its only intentional difference). argparse wraps it to the
/// terminal width; those are the lines for the default 80 columns.
pub const USAGE_BLOCK: &str = "\
usage: swadesh [-h] [--citation] [--list] [--test-set TEST_SET]
                 [--language-pair LANGPAIR] [--origlang ORIGLANG]
                 [--subset SUBSET] [--download DOWNLOAD]
                 [--echo ECHO [ECHO ...]] [--input [INPUT ...]]
                 [--num-refs NUM_REFS] [--encoding ENCODING]
                 [--metrics {bleu,chrf,ter} [{bleu,chrf,ter} ...]]
                 [--sentence-level] [--smooth-method {none,floor,add-k,exp}]
                 [--smooth-value BLEU_SMOOTH_VALUE]
                 [--tokenize {none,zh,13a,intl,char,ja-mecab,ko-mecab,spm,flores101,flores200,spBLEU-1K}]
                 [--lowercase] [--force] [--chrf-char-order CHRF_CHAR_ORDER]
                 [--chrf-word-order CHRF_WORD_ORDER] [--chrf-beta CHRF_BETA]
                 [--chrf-whitespace] [--chrf-lowercase] [--chrf-eps-smoothing]
                 [--ter-case-sensitive] [--ter-asian-support] [--ter-no-punct]
                 [--ter-normalized] [--confidence]
                 [--confidence-n CONFIDENCE_N] [--paired-ar | --paired-bs]
                 [--paired-ar-n PAIRED_AR_N] [--paired-bs-n PAIRED_BS_N]
                 [--paired-jobs PAIRED_JOBS] [--quiet] [--short]
                 [--score-only] [--width WIDTH] [--detail] [--no-color]
                 [--format {json,text,latex}] [--version]
                 [refs ...]
";

/// Whether an error message is one of argparse's own, which are printed under the usage block.
pub fn is_argparse_error(msg: &str) -> bool {
    msg.starts_with("argument ")
        || msg.starts_with("unrecognized arguments:")
        || msg.starts_with("ambiguous option:")
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Nargs {
    Zero,
    One,
    Star,
    Plus,
}

struct Opt {
    long: &'static str,
    short: &'static str,
    nargs: Nargs,
    /// Dispatch key; empty for options swadesh does not implement.
    id: &'static str,
    /// Error message for options swadesh does not implement.
    unsupported: &'static str,
}

const fn o(long: &'static str, short: &'static str, nargs: Nargs, id: &'static str) -> Opt {
    Opt { long, short, nargs, id, unsupported: "" }
}

const fn u(long: &'static str, short: &'static str, nargs: Nargs, unsupported: &'static str) -> Opt {
    Opt { long, short, nargs, id: "", unsupported }
}

const OPTS: &[Opt] = &[
    o("--help", "-h", Nargs::Zero, "help"),
    o("--version", "-V", Nargs::Zero, "version"),
    o("--metrics", "-m", Nargs::Plus, "metrics"),
    o("--sentence-level", "-sl", Nargs::Zero, "sentence_level"),
    o("--input", "-i", Nargs::Star, "input"),
    o("--num-refs", "-nr", Nargs::One, "num_refs"),
    o("--encoding", "-e", Nargs::One, "encoding"),
    o("--language-pair", "-l", Nargs::One, "langpair"),
    o("--smooth-method", "-s", Nargs::One, "smooth"),
    o("--smooth-value", "-sv", Nargs::One, "smooth_value"),
    o("--tokenize", "-tok", Nargs::One, "tokenize"),
    o("--lowercase", "-lc", Nargs::Zero, "lowercase"),
    o("--force", "", Nargs::Zero, "force"),
    o("--confidence", "-ci", Nargs::Zero, "confidence"),
    o("--confidence-n", "-cin", Nargs::One, "confidence_n"),
    o("--paired-ar", "-par", Nargs::Zero, "paired_ar"),
    o("--paired-bs", "-pbs", Nargs::Zero, "paired_bs"),
    o("--paired-ar-n", "-parn", Nargs::One, "paired_ar_n"),
    o("--paired-bs-n", "-pbsn", Nargs::One, "paired_bs_n"),
    o("--paired-jobs", "-j", Nargs::One, "paired_jobs"),
    o("--list", "", Nargs::Zero, "list"),
    o("--citation", "", Nargs::Zero, "citation"),
    o("--cite", "", Nargs::Zero, "citation"),
    o("--max-ngram-order", "", Nargs::One, "max_order"),
    o("--python-version", "", Nargs::One, "pyver"),
    o("--chrf-char-order", "-cc", Nargs::One, "cc"),
    o("--chrf-word-order", "-cw", Nargs::One, "cw"),
    o("--chrf-beta", "", Nargs::One, "beta"),
    o("--chrf-whitespace", "", Nargs::Zero, "whitespace"),
    o("--chrf-lowercase", "", Nargs::Zero, "chrf_lowercase"),
    o("--chrf-eps-smoothing", "", Nargs::Zero, "eps"),
    o("--quiet", "-q", Nargs::Zero, "quiet"),
    o("--short", "-sh", Nargs::Zero, "short"),
    o("--score-only", "-b", Nargs::Zero, "score_only"),
    o("--width", "-w", Nargs::One, "width"),
    o("--no-color", "-nc", Nargs::Zero, "no_color"),
    o("--format", "-f", Nargs::One, "format"),
    // Recognised so that they fail with a helpful message instead of "unknown option".
    u("--test-set", "-t", Nargs::One, "-t/--test-set (built-in test sets) is not supported by swadesh; pip install swadesh"),
    u("--download", "", Nargs::One, "--download (test sets) is not supported by swadesh; pip install swadesh"),
    u("--echo", "", Nargs::Plus, "--echo (test sets) is not supported by swadesh; pip install swadesh"),
    u("--origlang", "-ol", Nargs::One, "--origlang (test-set subsets) is not supported by swadesh; pip install swadesh"),
    u("--subset", "", Nargs::One, "--subset (test-set subsets) is not supported by swadesh; pip install swadesh"),
    u("--detail", "-d", Nargs::Zero, "--detail (translationese breakdown) is not supported by swadesh; pip install swadesh"),
    o("--ter-case-sensitive", "", Nargs::Zero, "ter_case_sensitive"),
    o("--ter-asian-support", "", Nargs::Zero, "ter_asian_support"),
    o("--ter-no-punct", "", Nargs::Zero, "ter_no_punct"),
    o("--ter-normalized", "", Nargs::Zero, "ter_normalized"),
];

/// Parsed settings, with argparse defaults already applied.
#[derive(Debug)]
pub struct Cli {
    pub refs: Vec<String>,
    /// `None`: read stdin.
    pub input: Option<Vec<String>>,
    pub num_refs: i64,
    pub encoding: String,
    pub metrics: Vec<String>,
    pub sentence_level: bool,
    pub smooth: String,
    pub smooth_value: Option<f64>,
    pub tokenize: Option<String>,
    pub lowercase: bool,
    pub force: bool,
    /// `-ci`: report a bootstrap confidence interval.
    pub confidence: bool,
    pub confidence_n: i64,
    /// `--paired-ar` / `--paired-bs`: a paired significance test between systems.
    pub paired_ar: bool,
    pub paired_bs: bool,
    pub paired_ar_n: i64,
    pub paired_bs_n: i64,
    pub paired_jobs: i64,
    /// `--list`: print the test-set catalogue and quit.
    pub list: bool,
    /// `--citation`: print the citation of the test set named by `-t`.
    pub citation: bool,
    pub max_ngram_order: i64,
    pub python_version: (u32, u32),
    pub chrf_char_order: i64,
    pub chrf_word_order: i64,
    pub chrf_beta: i64,
    pub chrf_whitespace: bool,
    pub chrf_lowercase: bool,
    pub chrf_eps_smoothing: bool,
    pub langpair: Option<String>,
    pub quiet: bool,
    pub short: bool,
    pub score_only: bool,
    pub width: i64,
    pub format: String,
    pub ter_case_sensitive: bool,
    pub ter_asian_support: bool,
    pub ter_no_punct: bool,
    pub ter_normalized: bool,
}

pub enum Action {
    Run(Box<Cli>),
    Help,
    Version,
}

/// argparse's notion of "this token is an option", as opposed to a value.
fn looks_like_option(tok: &str) -> bool {
    if tok.len() < 2 || !tok.starts_with('-') {
        return false;
    }
    if is_negative_number(tok) {
        return false;
    }
    // Something with spaces is a value, unless it names an option exactly.
    !tok.contains(' ') || find_exact(tok).is_some()
}

fn is_negative_number(tok: &str) -> bool {
    // argparse's `^-\d+$|^-\d*\.\d+$`
    let body = match tok.strip_prefix('-') {
        Some(b) if !b.is_empty() => b,
        _ => return false,
    };
    let all_digits = |s: &str| s.bytes().all(|b| b.is_ascii_digit());
    if all_digits(body) {
        return true;
    }
    match body.split_once('.') {
        Some((a, b)) => all_digits(a) && !b.is_empty() && all_digits(b),
        None => false,
    }
}

fn find_exact(tok: &str) -> Option<&'static Opt> {
    OPTS.iter().find(|o| o.long == tok || (!o.short.is_empty() && o.short == tok))
}

/// Resolve an option token to its spec and an attached value, if any.
fn find_opt(tok: &str) -> Result<(&'static Opt, Option<String>), Fail> {
    if tok.starts_with("--") {
        let (name, inline) = match tok.split_once('=') {
            Some((n, v)) => (n, Some(v.to_string())),
            None => (tok, None),
        };
        if let Some(opt) = OPTS.iter().find(|o| o.long == name) {
            return Ok((opt, inline));
        }
        let matches: Vec<&Opt> = OPTS.iter().filter(|o| o.long.starts_with(name)).collect();
        return match matches.len() {
            1 => Ok((matches[0], inline)),
            0 => fail(2, format!("unrecognized arguments: {tok}")),
            _ => {
                let names: Vec<&str> = matches.iter().map(|o| o.long).collect();
                fail(2, format!("ambiguous option: {name} could match {}", names.join(", ")))
            }
        };
    }
    if let Some(opt) = find_exact(tok) {
        return Ok((opt, None));
    }
    // Attached value or bundled flags: "-sv0.5", "-qb". `get` rather than
    // `split_at`: a multi-byte character straddling byte 2 ("-é") is an
    // unrecognized argument, not a panic.
    if tok.len() > 2 {
        if let (Some(head), Some(tail)) = (tok.get(..2), tok.get(2..)) {
            if let Some(opt) = OPTS.iter().find(|o| o.short == head) {
                return Ok((opt, Some(tail.to_string())));
            }
        }
    }
    fail(2, format!("unrecognized arguments: {tok}"))
}

fn parse_int(opt: &str, v: &str) -> Result<i64, Fail> {
    v.trim()
        .parse::<i64>()
        .or_else(|_| fail(2, format!("argument {opt}: invalid int value: '{v}'")))
}

fn parse_float(opt: &str, v: &str) -> Result<f64, Fail> {
    v.trim()
        .parse::<f64>()
        .or_else(|_| fail(2, format!("argument {opt}: invalid float value: '{v}'")))
}

/// The name argparse prints for an option in an error: `--long/-short`, long form first even when
/// the short one was typed. Measured for every option that can fail to parse.
fn opt_display(opt: &Opt) -> String {
    if opt.short.is_empty() {
        opt.long.to_string()
    } else {
        format!("{}/{}", opt.long, opt.short)
    }
}

fn choice(opt: &str, v: &str, choices: &[&str]) -> Result<String, Fail> {
    if choices.contains(&v) {
        Ok(v.to_string())
    } else {
        let list: Vec<String> = choices.iter().map(|c| format!("'{c}'")).collect();
        fail(2, format!("argument {opt}: invalid choice: '{v}' (choose from {})", list.join(", ")))
    }
}

pub fn parse(argv: &[String]) -> Result<Action, Fail> {
    let mut cli = Cli {
        refs: Vec::new(),
        input: None,
        num_refs: 1,
        encoding: "utf-8".to_string(),
        metrics: Vec::new(),
        sentence_level: false,
        smooth: "exp".to_string(),
        smooth_value: None,
        tokenize: None,
        lowercase: false,
        force: false,
        confidence: false,
        confidence_n: 1000,
        paired_ar: false,
        paired_bs: false,
        paired_ar_n: 10000,
        paired_bs_n: 1000,
        paired_jobs: 1,
        list: false,
        citation: false,
        max_ngram_order: 4,
        python_version: (3, 12),
        chrf_char_order: 6,
        chrf_word_order: 0,
        chrf_beta: 2,
        chrf_whitespace: false,
        chrf_lowercase: false,
        chrf_eps_smoothing: false,
        langpair: None,
        quiet: false,
        short: false,
        score_only: false,
        width: 1,
        format: "json".to_string(),
        ter_case_sensitive: false,
        ter_asian_support: false,
        ter_no_punct: false,
        ter_normalized: false,
    };

    let mut queue: VecDeque<String> = argv.iter().cloned().collect();
    let mut positional: Vec<String> = Vec::new();
    let mut only_positional = false;
    // argparse consumes the positional group once: REF... -opt X REF2 is an error.
    let mut positional_closed = false;

    while let Some(tok) = queue.pop_front() {
        if only_positional || !looks_like_option(&tok) {
            if positional_closed {
                return fail(2, format!("unrecognized arguments: {tok}"));
            }
            positional.push(tok);
            continue;
        }
        if tok == "--" {
            only_positional = true;
            continue;
        }
        let (opt, inline) = find_opt(&tok)?;
        if !opt.unsupported.is_empty() {
            return fail(2, opt.unsupported);
        }
        let values: Vec<String> = match opt.nargs {
            Nargs::Zero => match inline {
                None => Vec::new(),
                Some(s) if s.is_empty() => Vec::new(),
                Some(s) if tok.starts_with("--") => {
                    return fail(2, format!("argument {}: ignored explicit argument '{s}'", opt_display(opt)))
                }
                // Bundled flags: "-qb" means "-q -b".
                Some(s) => {
                    queue.push_front(format!("-{s}"));
                    Vec::new()
                }
            },
            Nargs::One => match inline {
                Some(s) => vec![s],
                None => match queue.front() {
                    Some(next) if !looks_like_option(next) => vec![queue.pop_front().unwrap()],
                    _ => return fail(2, format!("argument {}: expected one argument", opt_display(opt))),
                },
            },
            Nargs::Star | Nargs::Plus => {
                let mut vals: Vec<String> = inline.into_iter().collect();
                while let Some(next) = queue.front() {
                    if looks_like_option(next) {
                        break;
                    }
                    vals.push(queue.pop_front().unwrap());
                }
                if opt.nargs == Nargs::Plus && vals.is_empty() {
                    return fail(2, format!("argument {}: expected at least one argument", opt_display(opt)));
                }
                vals
            }
        };
        if !positional.is_empty() {
            positional_closed = true;
        }
        let one = || values.first().cloned().unwrap_or_default();
        match opt.id {
            "help" => return Ok(Action::Help),
            "version" => return Ok(Action::Version),
            "metrics" => {
                let mut ms = Vec::new();
                for v in &values {
                    ms.push(choice(&opt_display(opt), v, METRIC_CHOICES)?);
                }
                cli.metrics = ms;
            }
            "sentence_level" => cli.sentence_level = true,
            "input" => cli.input = Some(values),
            "num_refs" => cli.num_refs = parse_int(&opt_display(opt), &one())?,
            "encoding" => cli.encoding = one(),
            "langpair" => cli.langpair = Some(one()),
            "smooth" => cli.smooth = choice(&opt_display(opt), &one(), SMOOTH_CHOICES)?,
            "smooth_value" => cli.smooth_value = Some(parse_float(&opt_display(opt), &one())?),
            "tokenize" => cli.tokenize = Some(choice(&opt_display(opt), &one(), TOKENIZE_CHOICES)?),
            "lowercase" => cli.lowercase = true,
            "force" => cli.force = true,
            "confidence" => cli.confidence = true,
            "confidence_n" => cli.confidence_n = parse_int(&opt_display(opt), &one())?,
            // The two paired tests are mutually exclusive, and argparse names the one that
            // came *second* as the argument that is not allowed (measured in both orders).
            "paired_ar" if cli.paired_bs => {
                return fail(
                    2,
                    "argument --paired-ar/-par: not allowed with argument --paired-bs/-pbs",
                )
            }
            "paired_ar" => cli.paired_ar = true,
            "paired_bs" if cli.paired_ar => {
                return fail(
                    2,
                    "argument --paired-bs/-pbs: not allowed with argument --paired-ar/-par",
                )
            }
            "paired_bs" => cli.paired_bs = true,
            "paired_ar_n" => cli.paired_ar_n = parse_int(&opt_display(opt), &one())?,
            "paired_bs_n" => cli.paired_bs_n = parse_int(&opt_display(opt), &one())?,
            "paired_jobs" => cli.paired_jobs = parse_int(&opt_display(opt), &one())?,
            "list" => cli.list = true,
            "citation" => cli.citation = true,
            "max_order" => cli.max_ngram_order = parse_int("--max-ngram-order", &one())?,
            "pyver" => {
                let v = one();
                let parsed = v.split_once('.').and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?)));
                cli.python_version = match parsed {
                    Some(x) => x,
                    None => return fail(2, format!("argument --python-version: expected X.Y, got '{v}'")),
                };
            }
            "cc" => cli.chrf_char_order = parse_int(&opt_display(opt), &one())?,
            "cw" => cli.chrf_word_order = parse_int(&opt_display(opt), &one())?,
            "beta" => cli.chrf_beta = parse_int(&opt_display(opt), &one())?,
            "whitespace" => cli.chrf_whitespace = true,
            "chrf_lowercase" => cli.chrf_lowercase = true,
            "eps" => cli.chrf_eps_smoothing = true,
            "quiet" => cli.quiet = true,
            "short" => cli.short = true,
            "score_only" => cli.score_only = true,
            "width" => cli.width = parse_int(&opt_display(opt), &one())?,
            "no_color" => {}
            "format" => cli.format = choice(&opt_display(opt), &one(), FORMAT_CHOICES)?,
            "ter_case_sensitive" => cli.ter_case_sensitive = true,
            "ter_asian_support" => cli.ter_asian_support = true,
            "ter_no_punct" => cli.ter_no_punct = true,
            "ter_normalized" => cli.ter_normalized = true,
            _ => {}
        }
    }

    cli.refs = positional;
    if cli.metrics.is_empty() {
        cli.metrics = vec!["bleu".to_string()];
    }
    // sacreBLEU lets SACREBLEU_FORMAT override -f when it names a valid format.
    if let Ok(env) = std::env::var("SACREBLEU_FORMAT") {
        let lower = env.to_lowercase();
        if FORMAT_CHOICES.contains(&lower.as_str()) {
            cli.format = lower;
        }
    }
    Ok(Action::Run(Box::new(cli)))
}
