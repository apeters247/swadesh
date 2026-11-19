"""`swadesh` command line: the sacreBLEU CLI, accelerated. Same flags, same output.

The only intentional difference is the program name in usage and error lines,
which is ``swadesh`` (sacreBLEU's is ``sacrebleu``).
"""
import sys


def main(argv=None):
    import swadesh
    swadesh.patch()
    from sacrebleu.sacrebleu import main as _main
    args = sys.argv[1:] if argv is None else list(argv)
    sys.argv = ["swadesh"] + args
    return _main()


if __name__ == "__main__":
    sys.exit(main())
