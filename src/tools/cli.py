"""Common CLI helpers for arena tools."""
from __future__ import annotations

import argparse
from typing import Callable


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--bundle",
        dest="bundle",
        help="Path to arena bundle JSON (defaults to data/<derived>.json)",
    )
    return parser


def parse_and_run(
    parser: argparse.ArgumentParser,
    handler: Callable[[argparse.Namespace], int],
) -> int:
    args = parser.parse_args()
    try:
        return handler(args)
    except KeyboardInterrupt:
        print("Interrupted by user.")
        return 130
