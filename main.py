"""Project command-line entry point."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    regression = subparsers.add_parser("regression")
    regression.add_argument("--version", choices=("a", "b", "both"), required=True)
    args = parser.parse_args()
    if args.command == "regression":
        if args.version in ("a", "both"):
            from src.regression.version_a_strict import main as run_a
            run_a()
        if args.version in ("b", "both"):
            from src.regression.version_b_borderline import main as run_b
            run_b()


if __name__ == "__main__":
    main()