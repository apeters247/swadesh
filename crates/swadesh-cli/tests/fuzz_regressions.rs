//! Regression tests for crashes found by the `fuzz/` cargo-fuzz targets.

use swadesh_cli::run_argv;

/// fuzz target `cli_argv`: `find_opt` sliced the argument at byte 2, which
/// panics when that is inside a multi-byte character.
#[test]
fn non_ascii_short_option_is_an_error_not_a_panic() {
    for tok in ["-é", "-Ւ", "-Ւx", "-aé", "-\u{1F600}"] {
        let mut out = Vec::new();
        let err = run_argv(&[tok.to_string()], &mut out).expect_err(tok);
        assert_eq!(err.code, 2, "{tok}: {}", err.msg);
    }
}
