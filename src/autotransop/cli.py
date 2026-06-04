from __future__ import annotations

import argparse

from . import __version__


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="autotransop")
    parser.add_argument("--version", action="store_true", help="Print package version and exit.")
    args = parser.parse_args(argv)
    if args.version:
        print(__version__)
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
