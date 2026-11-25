"""`swadesh` command line: the sacreBLEU CLI, accelerated. Same flags, same output.

The only intentional difference is the program name in usage and error lines,
which is ``swadesh`` (sacreBLEU's is ``sacrebleu``).

This is an adapter, not an independent CLI: it installs swadesh's metrics into sacreBLEU with
``patch()`` and then runs sacreBLEU's own argument parser, so every flag sacreBLEU has works
unchanged. The Rust binary carries a CLI of its own (``cargo build -p swadesh-cli``) but it is a
different, smaller flag set, so it is not what this entry point uses.
"""
import sys


def main(argv=None):
    import swadesh

    try:
        swadesh.patch()
        from sacrebleu.sacrebleu import main as _main
    except ImportError as e:
        # A real CLI error, not a traceback: the library API has no such requirement.
        raise SystemExit(f"swadesh: {e}") from e
    args = sys.argv[1:] if argv is None else list(argv)
    sys.argv = ["swadesh"] + args
    return _main()


if __name__ == "__main__":
    sys.exit(main())
