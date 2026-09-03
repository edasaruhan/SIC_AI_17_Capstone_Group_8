"""Command-line entry point. Subcommands are registered by their owning modules."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bias-eval")
    parser.add_argument("--config", default="configs/evaluation/suite.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan", help="Print the frozen 400-cell plan without API calls")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        from .planner import render_plan

        print(render_plan(args.config))
        return 0
    return 2
