"""`swadesh` command line: the sacreBLEU CLI, accelerated. Same flags, same output."""
import sys


def main(argv=None):
    import swadesh
    swadesh.patch()
    from sacrebleu.sacrebleu import main as _main
    if argv is not None:
        sys.argv = [sys.argv[0]] + list(argv)
    return _main()


if __name__ == "__main__":
    sys.exit(main())
