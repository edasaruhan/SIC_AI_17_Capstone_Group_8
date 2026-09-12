"""Explicit offline training and prediction entry point."""

import argparse
import json
from pathlib import Path

from evidence_eval.io import clean, read_json

from .pipeline import plan, predict, status, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "train", "status", "predict"])
    parser.add_argument("--root", type=Path, default=Path("data/processed/evidence_v1"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/final_models_v1"))
    parser.add_argument("--release", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.command == "plan":
        result = plan(args.root)
    elif args.command == "train":
        result = train(args.root, args.output)
    elif args.command == "status":
        result = status(args.output)
    else:
        if not args.release or not args.request:
            parser.error("predict requires --release and --request")
        result = predict(args.root, args.release, read_json(args.request), args.device)
    print(json.dumps(clean(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
