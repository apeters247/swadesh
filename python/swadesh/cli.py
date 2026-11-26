"""`swadesh` command line: the Rust CLI, in process.

The work is done by the compiled code (`swadesh._swadesh.run_cli`, the same entry point the
standalone `swadesh` binary uses), not by sacreBLEU's parser: swadesh must work with sacreBLEU
absent. The one intentional difference from sacreBLEU is the program name in usage and error lines,
which is ``swadesh``.

``--python-version`` is added automatically unless the user passed one, so the Rust float-summation
semantics match the interpreter running this script. It is not added for ``-h``/``--help``/
``-V``/``--version``, where it would be invisible anyway.

If the extension is not importable (a source checkout without a built Rust module), the old adapter
path is used instead and the reason is printed once to stderr.
"""
import sys


def main(argv=None):
    args = sys.argv[1:] if argv is None else list(argv)
    runner = _binary_runner()
    if runner is None:
        return _sacrebleu_adapter(args)
    return runner(args)


def _binary_runner():
    """A callable that runs the compiled CLI, or None when the extension is unavailable."""
    try:
        from swadesh import _swadesh
    except Exception:  # noqa: BLE001 - any import failure means "no extension"
        return None
    if not hasattr(_swadesh, "run_cli"):
        return None

    help_or_version = ("-h", "--help", "-V", "--version")

    def run(args):
        argv = list(args)
        has_pyver = any(a == "--python-version" or a.startswith("--python-version=") for a in argv)
        if not has_pyver and not any(a in help_or_version for a in argv):
            argv += ["--python-version", "%d.%d" % sys.version_info[:2]]
        code, out, err = _swadesh.run_cli(argv)
        if out:
            sys.stdout.write(out)
            sys.stdout.flush()
        if err:
            sys.stderr.write(err)
            sys.stderr.flush()
        return code

    return run


def _sacrebleu_adapter(args):
    """The fallback: run sacreBLEU's own parser with swadesh's metrics patched in."""
    import swadesh

    sys.stderr.write(
        "swadesh: the compiled extension is not available; falling back to sacreBLEU's CLI\n"
    )
    try:
        swadesh.patch()
        from sacrebleu.sacrebleu import main as _main
    except ImportError as e:
        # A real CLI error, not a traceback: the library API has no such requirement.
        raise SystemExit(f"swadesh: {e}") from e
    sys.argv = ["swadesh"] + list(args)
    return _main()


if __name__ == "__main__":
    sys.exit(main())
