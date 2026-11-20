import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Hotel booking ML project")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preprocess", help="Prepare the shared cleaned dataset")
    args = parser.parse_args()
    if args.command == "preprocess":
        from src.data.prepare_data import main as prepare_data

        prepare_data()


if __name__ == "__main__":
    main()