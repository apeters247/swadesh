//! Tests for public API paths that the Python-driven differential tests never reach.
//!
//! Expected values come from sacreBLEU 2.6.0 or numpy (noted per test), or from the
//! arithmetic the function is documented to perform.

use swadesh_core::corpus_stats;
use swadesh_core::ngram::{count, max_merge, AnyRuns};
use swadesh_core::nprng::{default_rng, mean_f64};
use swadesh_core::resample::bootstrap_sums_f32_seeded;
use swadesh_core::score::{compute_bleu, py_sum, BleuScoreConfig, PyNum, SumMode};
use swadesh_core::unicode::Interner;
use swadesh_core::table::CountTable;
use swadesh_core::ter::{ter_segment, tercom_tokenize, TerConfig};
use swadesh_core::{Smooth, Tokenizer};

fn ter_cfg(normalized: bool) -> TerConfig {
    TerConfig { normalized, no_punct: false, asian_support: false, case_sensitive: false }
}

// ------------------------------------------------------------------ ngram.rs

#[test]
fn wide_ngram_runs_report_total_and_emptiness() {
    // 5 words x 32 bits = 160 bits > 128: keys fall back to Vec<u32>.
    let seq = [1u32, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6];
    let runs = count(&seq, 5, 32);
    assert!(matches!(runs, AnyRuns::Wide(_)));
    assert_eq!(runs.total(), (seq.len() - 4) as u64);
    assert!(!runs.is_empty());
    assert!(AnyRuns::Wide(Vec::new()).is_empty());
    assert_eq!(AnyRuns::Wide(Vec::new()).total(), 0);
    assert!(AnyRuns::Packed(Vec::new()).is_empty());
    assert_eq!(AnyRuns::Packed(Vec::new()).total(), 0);
}

#[test]
fn max_merge_takes_per_key_maximum_over_packed_parts() {
    let a = AnyRuns::Packed(vec![(1u128, 2), (5, 1)]);
    let b = AnyRuns::Packed(vec![(1u128, 3), (2, 4)]);
    match max_merge(vec![a, AnyRuns::Packed(Vec::new()), b]) {
        AnyRuns::Packed(r) => assert_eq!(r, vec![(1, 3), (2, 4), (5, 1)]),
        other => panic!("expected packed runs, got {other:?}"),
    }
    match max_merge(vec![AnyRuns::Packed(vec![(7u128, 1)])]) {
        AnyRuns::Packed(r) => assert_eq!(r, vec![(7, 1)]),
        other => panic!("expected packed runs, got {other:?}"),
    }
    assert!(matches!(max_merge(vec![AnyRuns::Packed(Vec::new())]), AnyRuns::Packed(r) if r.is_empty()));
}

// ------------------------------------------------------------------ score.rs

#[test]
fn py_sum_naive_mode_and_integer_items_in_float_phase() {
    // CPython <= 3.11 sum(): plain left-to-right float addition (values are exact here).
    assert_eq!(py_sum(&[PyNum::Int(1), PyNum::Float(0.5), PyNum::Float(0.25)], SumMode::Naive), PyNum::Float(1.75));
    assert_eq!(py_sum(&[PyNum::Int(1), PyNum::Float(0.5), PyNum::Int(2)], SumMode::Naive), PyNum::Float(3.5));
    assert_eq!(py_sum(&[], SumMode::Naive), PyNum::Int(0));
}

#[test]
fn sum_mode_per_cpython_version() {
    assert_eq!(SumMode::for_python(2, 7), SumMode::Naive);
    assert_eq!(SumMode::for_python(3, 11), SumMode::Naive);
    assert_eq!(SumMode::for_python(3, 12), SumMode::Neumaier);
    assert_eq!(SumMode::for_python(3, 14), SumMode::NeumaierInts);
}

#[test]
fn pynum_as_f64_converts_both_variants() {
    assert_eq!(PyNum::Int(3).as_f64(), 3.0);
    assert_eq!(PyNum::Float(0.5).as_f64(), 0.5);
}

#[test]
fn smooth_from_name_rejects_unknown_names() {
    assert_eq!(Smooth::from_name("floor"), Some(Smooth::Floor));
    assert_eq!(Smooth::from_name("bogus"), None);
}

#[test]
fn bleu_with_only_zero_precision_scores_zero() {
    // Order-1 precision is 0 and only order 1 is effective: the mean is the
    // integer sentinel from my_log(0), so exp(sentinel) underflows to 0.0.
    let cfg = BleuScoreConfig {
        smooth: Smooth::None,
        smooth_value: 0.0,
        effective_order: true,
        max_order: 2,
        sum_mode: SumMode::Neumaier,
    };
    // [sys_len, ref_len, correct1, correct2, total1, total2]
    let stats = [4.0, 4.0, 0.0, 1.0, 4.0, 0.0];
    assert_eq!(compute_bleu(&stats, &cfg).score, 0.0);
}

// ------------------------------------------------------------------ nprng.rs

#[test]
fn mean_of_empty_slice_is_nan_like_numpy() {
    assert!(mean_f64(&[]).is_nan());
    assert_eq!(mean_f64(&[1.0, 2.0]), 1.5);
}

#[test]
fn bit_generator_accessor_is_deterministic() {
    let a = default_rng(12345).bit_generator().next_u64();
    let b = default_rng(12345).bit_generator().next_u64();
    assert_eq!(a, b);
}

#[test]
fn large_range_integers_hit_lemire_rejection_like_numpy() {
    // numpy: np.random.default_rng(12345).integers(0, 2**63-1, size=8).
    // A range of 2^63-1 rejects about half the 64-bit products, so the rejection loop runs.
    let got = default_rng(12345).integers(0, i64::MAX, 8);
    assert_eq!(
        got,
        vec![
            2096804712593481934, 2921580012919480942, 7354398262316660715, 6237348419996972167,
            3607348892368485766, 3069666675758614433, 5518424227241521869, 1722318865822139695,
        ]
    );
}

#[test]
fn rejection_heavy_range_matches_numpy() {
    // numpy: np.random.default_rng(12345).integers(0, 2**62+1, size=16). With excl = 2^62+1 the
    // threshold (2^64 + 1) mod excl is about 2^62, so the Lemire rejection loop repeats often.
    let got = default_rng(12345).integers(0, (1i64 << 62) + 1, 16);
    assert_eq!(
        got,
        vec![
            1048402356296740967, 1460790006459740471, 3118674209998486084, 1803674446184242883,
            2759212113620760935, 861159432911069847, 3102539641994749835, 4343299105880222464,
            1144831291291707959, 4375941941058989476, 3077089633434087669, 442251168775416978,
            2037625810852727241, 4088167049579352133, 1015193296656253866, 737388307885840079,
        ]
    );
}

// ------------------------------------------------------------------ resample.rs

#[test]
fn bootstrap_with_no_draws_or_no_samples_is_empty() {
    assert!(bootstrap_sums_f32_seeded(&[1.0, 2.0], 2, 0, 5, 1).is_empty());
    assert!(bootstrap_sums_f32_seeded(&[1.0, 2.0], 2, 2, 0, 1).is_empty());
}

// ------------------------------------------------------------------ stats.rs / table.rs / lib.rs

#[test]
fn interner_clear_restarts_ids() {
    let mut i = Interner::default();
    assert_eq!(i.id("a"), 0);
    assert_eq!(i.id("b"), 1);
    i.clear();
    assert_eq!(i.id("b"), 0);
}

#[test]
fn count_table_default_is_usable_and_survives_generation_wraparound() {
    let mut t = CountTable::default();
    t.add(7);
    assert_eq!(t.get(7), 1);
    assert_eq!(t.get(8), 0); // absent key
    // Each reset bumps the generation stamp; the stamp wraps to 0 after u32::MAX resets,
    // at which point every entry is restamped and the table is logically empty.
    for _ in 0..u32::MAX {
        t.reset(1);
    }
    assert_eq!(t.get(7), 0);
    t.add(7);
    assert_eq!(t.get(7), 1);
}

#[test]
fn corpus_stats_leaves_row_zero_when_segment_writes_wrong_width() {
    let out = corpus_stats(&["a", "b"], &[vec!["a"], vec!["b"]], 3, |_, _, o| o.push(1));
    assert_eq!(out, vec![0; 6]);
}

#[test]
fn tokenizer_from_name_rejects_unknown_names() {
    assert_eq!(Tokenizer::from_name("13a"), Some(Tokenizer::T13a));
    assert_eq!(Tokenizer::from_name("bogus"), None);
}

// ------------------------------------------------------------------ ter.rs (expected values from sacreBLEU 2.6.0)

#[test]
fn tercom_normalized_splits_final_possessive_like_sacrebleu() {
    // sacrebleu TercomTokenizer(normalized=True)("the dog's") == "the dog 's"
    let mut out = String::new();
    tercom_tokenize(&ter_cfg(true), "the dog's", &mut out);
    assert_eq!(out.split_whitespace().collect::<Vec<_>>(), vec!["the", "dog", "'s"]);
}

#[test]
fn ter_normalized_asian_support_matches_sacrebleu() {
    // sacrebleu TER(normalized=True, asian_support=True).sentence_score("你好世界", ["你好世界吗"]) -> 1 edit, ref_length 5.0
    let cfg = TerConfig { normalized: true, no_punct: false, asian_support: true, case_sensitive: false };
    assert_eq!(ter_segment(&cfg, "你好世界", &["你好世界吗"]).unwrap(), [1.0, 5.0]);
}

#[test]
fn ter_shifted_words_match_sacrebleu() {
    // sacrebleu TER().sentence_score(hyp, [ref]) values; these need block shifts.
    let cases: [(&str, &str, [f64; 2]); 4] = [
        ("d a b c e f g h", "a b c d e f g h", [1.0, 8.0]),
        ("x y the cat sat on a mat", "the cat sat on a mat x y", [1.0, 8.0]),
        ("b c a", "a b c", [1.0, 3.0]),
        ("the cat the mat the dog the rug", "the rug the dog the mat the cat", [3.0, 8.0]),
    ];
    for (hyp, reference, expected) in cases {
        assert_eq!(ter_segment(&ter_cfg(false), hyp, &[reference]).unwrap(), expected, "hyp {hyp:?}");
    }
}

#[test]
fn ter_multiple_references_take_the_best_and_average_lengths_like_sacrebleu() {
    // sacrebleu: 1 edit against the closest reference; ref_length is the mean of the three (7, 7, 6).
    let refs = ["the cat sat on a mat today", "the cat sat on the rug today", "the cat sat on the mat"];
    let r = ter_segment(&ter_cfg(false), "the cat sat on the mat today", &refs).unwrap();
    assert_eq!(r[0], 1.0);
    assert_eq!(r[1], 20.0 / 3.0);
}

#[test]
fn ter_short_hypothesis_against_long_reference_matches_sacrebleu() {
    // sacrebleu TER().sentence_score("a", [60 words]) -> num_edits 60, ref_length 60.0
    let long_ref = (0..60).map(|i| format!("w{i}")).collect::<Vec<_>>().join(" ");
    let r = ter_segment(&ter_cfg(false), "a", &[long_ref.as_str()]).unwrap();
    assert_eq!(r, [60.0, 60.0]);
}

#[test]
fn ter_repeated_word_prefixes_match_sacrebleu() {
    // sacrebleu TER().sentence_score(hyp_i, [ref_i]) -> [2, 8.0] for every i
    for i in 0..12 {
        let hyp = format!("the cat sat on the mat number {i} today");
        let r = format!("the cat sat on a mat number {i}");
        assert_eq!(ter_segment(&ter_cfg(false), &hyp, &[r.as_str()]).unwrap(), [2.0, 8.0], "segment {i}");
    }
}

#[test]
fn ter_random_small_vocab_pairs_match_sacrebleu() {
    // 40 pairs from random.Random(20261119) over the words a..e (6-12 words each), expected values from
    // sacrebleu TER().sentence_score(hyp, [ref]) -> (num_edits, ref_length). Repetition gives many shift
    // candidates, which reach the edit cache and the shift-bounds paths.
    let rows: [(&str, &str, f64, f64); 40] = [
        ("d e e b a c e d c a c e", "b c a e d e", 7.0, 6.0),
        ("c e e b a a b b", "a b c d a a b e c", 6.0, 9.0),
        ("b e c a d d e", "e c a e b c a", 4.0, 7.0),
        ("a d b d d a", "e c a e b e b d d", 6.0, 9.0),
        ("b d d b d a c a d", "e d b c e a e", 6.0, 7.0),
        ("d e d d c d d d c e", "c c d e e b b d c e", 5.0, 10.0),
        ("e c a a d c c c e e", "a a b e d b a d e a b e", 8.0, 12.0),
        ("c d a b a a d a e", "e b d d e e c a b", 6.0, 9.0),
        ("a a b a c d d e a", "a d e e e b a d b a d", 6.0, 11.0),
        ("c a e a d c", "a c c d d d b b", 6.0, 8.0),
        ("e d e d b d a", "b c b e c e e b b d b d", 8.0, 12.0),
        ("b c c d b e a d a b", "d b b e c b b a c", 6.0, 9.0),
        ("a d e d d e e b", "a c d b e d d e a e d c", 5.0, 12.0),
        ("c a b a e b d c d d c d", "a e e e b c e e b e d c", 8.0, 12.0),
        ("d d b b b b b", "d d e b a c c e e d", 7.0, 10.0),
        ("c e e c b c e c d d", "b e a b a d e a a c", 7.0, 10.0),
        ("b e d b e b c d", "b e d c a d e e c", 4.0, 9.0),
        ("e d e b e b", "c d b a c b a", 5.0, 7.0),
        ("b c b c c b c e", "a d b d a a d a", 7.0, 8.0),
        ("d b c e d c b d", "c d c c c d d a a a b", 7.0, 11.0),
        ("d a d e b a", "e c d c a a d c e d b b", 7.0, 12.0),
        ("b a b b a c", "c e d a b b", 4.0, 6.0),
        ("d d c e d c d d d c e d", "b c d e b c c", 9.0, 7.0),
        ("d b b b e d d", "d c b b c b d a", 4.0, 8.0),
        ("e b d d e d a b", "b c d b b b b", 5.0, 7.0),
        ("c c e e d c b b", "c b c d e e c d b a b", 4.0, 11.0),
        ("b e e e d c e c e e", "b b c d a e d", 7.0, 7.0),
        ("e e d d b a c a d a e a", "c a b d a d a b", 7.0, 8.0),
        ("a e a a d a c d e a", "d e d c b d b e", 7.0, 8.0),
        ("a a a e c a b a", "b a d a a e a a", 3.0, 8.0),
        ("a b e c d e a a c d a a", "d a b c e d c d c b b a", 7.0, 12.0),
        ("a d a d d e c a", "c a a e d b e c e b d e", 7.0, 12.0),
        ("e e e d d c e b d b", "e d a a a d d e d a", 7.0, 10.0),
        ("d c d a a a d d d a", "d a b a c a d d", 5.0, 8.0),
        ("b e e d c c b c c", "b e b b c a c b c", 4.0, 9.0),
        ("a c d d b b e d d c", "b e c c e c b c", 6.0, 8.0),
        ("c e a d c e c c d d", "a e d c a c a", 6.0, 7.0),
        ("c b b a b c a e a", "c a d c a b d e a d e e", 7.0, 12.0),
        ("a d a c c b c c", "e b b d d c", 6.0, 6.0),
        ("e e c b b a e", "b e d d d b d a c b d e", 8.0, 12.0),
    ];
    for (h, r, edits, reflen) in rows {
        assert_eq!(ter_segment(&ter_cfg(false), h, &[r]).unwrap(), [edits, reflen], "hyp {h:?} ref {r:?}");
    }
}

#[test]
fn max_merge_skips_empty_wide_parts() {
    let wide = AnyRuns::Wide(vec![(vec![1u32, 2], 1)]);
    match max_merge(vec![AnyRuns::Wide(Vec::new()), wide]) {
        AnyRuns::Wide(r) => assert_eq!(r, vec![(vec![1u32, 2], 1)]),
        other => panic!("expected wide runs, got {other:?}"),
    }
}

#[test]
fn ter_unbalanced_length_pairs_match_sacrebleu() {
    // 60 pairs from random.Random(7331) over one to four words, alternating a long hypothesis with a long
    // reference. Expected values from sacrebleu TER().sentence_score(hyp, [ref]) -> (num_edits, ref_length).
    let rows: [(&str, &str, f64, f64); 60] = [
        ("c a a a c", "b b a c b a c b c c a", 8.0, 11.0),
        ("b b c c b c a b c a a c c a b a c b b", "b b a b", 15.0, 4.0),
        ("b a a a", "b a a b b a c c c a a b", 8.0, 12.0),
        ("a d a a c b a b c d c b", "c b d a b d", 7.0, 6.0),
        ("a c a b b c a", "b c b b b c b b a b b a a a c", 10.0, 15.0),
        ("b c a a b b c c c b b c c c c a b", "a a a a b a", 13.0, 6.0),
        ("c a d", "c d d b d c a a b a b a c d b", 12.0, 15.0),
        ("a a a a a a a b a a a a a a a a a b a", "b a a b", 15.0, 4.0),
        ("a a b b b a a", "b a b a a a a a a a a a a a a b a a b", 13.0, 19.0),
        ("d a c c a b c d d d a d a b c c", "b a c", 13.0, 3.0),
        ("b c b b c b b", "b c c b c c a c c c a a", 8.0, 12.0),
        ("c c c c b b b a b a b a a", "c c a b b c a", 7.0, 7.0),
        ("c a a", "d d d d b c a b c b c", 9.0, 11.0),
        ("a d b a d a c a d d b d", "c c c d d a a", 9.0, 7.0),
        ("a a b", "b b a a a b a b a a a a b b b a a", 14.0, 17.0),
        ("a b b c a b b c a a b a a", "a a a a a", 8.0, 5.0),
        ("c b b c a c b", "b a c c b b b b a a a a b b b c b b b", 13.0, 19.0),
        ("a a b b a b b b b a b b b a a b b a", "a a b a b a", 12.0, 6.0),
        ("c d a b b", "a c b a a c d a a d b d d a d d d", 13.0, 17.0),
        ("c b c c b b c c c a c c b b a b c b b", "b c a b", 15.0, 4.0),
        ("a b a c b a a", "b b b a a a a b a a a b c a a b a a a c", 13.0, 20.0),
        ("b b b b a b b a c b b c c", "b c a", 11.0, 3.0),
        ("a a b a c b a", "a a a c a a a c a a b", 6.0, 11.0),
        ("b d d c c d a b c a c d", "b b c d b a", 7.0, 6.0),
        ("c a a b b a", "b b b c a a b a c a b c", 7.0, 12.0),
        ("b b a a c b c d c c c a c a b d c a d", "b a a d a d", 13.0, 6.0),
        ("b a a b a", "a a a b a b a b b b a b b a b a", 11.0, 16.0),
        ("b a b b a b a a a b a a a b a b a a a b", "a a a b a b", 14.0, 6.0),
        ("d a a d a c c", "c a a a c c d b d b b b", 7.0, 12.0),
        ("d c a b d a b d b b", "d c b a b b c", 4.0, 7.0),
        ("c b b c c b d", "c a c d a d a c a a a d c c b b", 11.0, 16.0),
        ("a c d c c b b c b d d d c", "a d b a", 10.0, 4.0),
        ("c a b", "c a b b c c a a a b c b b a", 11.0, 14.0),
        ("d b b c a b c b c d c c d a b b b", "d a d", 14.0, 3.0),
        ("a a c c", "a b a b b b b c a a a a a a b a a c b", 15.0, 19.0),
        ("d b b b d b c a a b d d c c b", "b d a b", 11.0, 4.0),
        ("b b a b b a a", "b b a a b b a b a a a b a", 6.0, 13.0),
        ("c b a b d b a c b a b", "d a d d b", 8.0, 5.0),
        ("a a c c a c", "a b a a a c b b a a a c", 7.0, 12.0),
        ("a a b a a b a b a b b b b a a b b a a", "a a b", 16.0, 3.0),
        ("b d b c d", "b a b b a a a c a a b c a a a", 12.0, 15.0),
        ("c b c b a b a c a a", "c c b a b", 5.0, 5.0),
        ("a a a a", "b b a b b b b a a b a a a a", 10.0, 14.0),
        ("d c c b d b d b b a", "b c b", 8.0, 3.0),
        ("c b b d d c", "d a c d b c b d d c b a d d d", 9.0, 15.0),
        ("c a d c c a a a a d a d d", "b a d d b d", 9.0, 6.0),
        ("b b b b", "a a b b b a a a b b b a", 8.0, 12.0),
        ("b b a a b a a b b b b a a b b a a b", "b b a b a b b", 11.0, 7.0),
        ("c b c c", "c c c a b c a a c b", 6.0, 10.0),
        ("a d c a d b b b d c c b d", "c c a", 11.0, 3.0),
        ("b a b", "a b b b b b a b b b a b a b a a b a", 15.0, 18.0),
        ("a b b b a a a b a b b a a a a a b b a", "b b a a b a b", 12.0, 7.0),
        ("b a b a b a b", "b a a a b a b a a a a b a b", 7.0, 14.0),
        ("a c b c c b b a a c d d d c a d a b", "d d b", 15.0, 3.0),
        ("c c a b a", "c a c a b c b b b c a b c c b b b b b c", 15.0, 20.0),
        ("b b b a a b b a a a a", "b b b a a b b", 4.0, 7.0),
        ("c b a c c", "c c a c c c c b b a c c c c a b b a c", 14.0, 19.0),
        ("c a b b b a a a b b c a b c b", "a b b c a a a", 9.0, 7.0),
        ("b a b", "b b b b b a b a a a b b a b b", 12.0, 15.0),
        ("b b a c c a b c c b a b b a b a b a a", "b c c c c c c", 14.0, 7.0),
    ];
    for (h, r, edits, reflen) in rows {
        assert_eq!(ter_segment(&ter_cfg(false), h, &[r]).unwrap(), [edits, reflen], "hyp {h:?} ref {r:?}");
    }
}

#[test]
fn preprocess_bleu_rstrips_then_tokenizes_like_sacrebleu() {
    // sacrebleu _preprocess_segment: tokenizer(sent.rstrip()); 13a output from sacrebleu Tokenizer13a.
    let mut out = String::new();
    swadesh_core::preprocess_bleu(Tokenizer::T13a, "Hello, world!   ", &mut out);
    assert_eq!(out, "Hello , world !");
    swadesh_core::preprocess_bleu(Tokenizer::None, "keep  this \t", &mut out);
    assert_eq!(out, "keep  this");
}

#[test]
fn tokens_splits_on_python_whitespace() {
    // str.split() semantics: runs of any Unicode whitespace separate tokens.
    let toks: Vec<&str> = swadesh_core::tokens("a  b\tc\u{a0}d").collect();
    assert_eq!(toks, vec!["a", "b", "c", "d"]);
}
