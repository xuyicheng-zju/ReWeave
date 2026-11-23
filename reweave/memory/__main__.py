# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RPent Contributors
# Modified for ReWeave.
"""Offline, explicit local text import; no online learning or network access."""

from __future__ import annotations

import argparse
from pathlib import Path

from reweave.memory.import_bundle import publish_import
from reweave.utils.logging import get_logger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m reweave.memory")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser(
        "import-bundle", help="Freeze explicit local text into a new Bundle's C skills."
    )
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--bundle", type=Path, required=True)
    command.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New Bundle JSON file; its parent must exist. A .receipt.json audit is published beside it.",
    )
    args = parser.parse_args(argv)
    try:
        receipt = publish_import(args.manifest, args.bundle, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"text import failed: {exc}\n")
    get_logger("memory").info(
        "Published C text import: %s", receipt["output_bundle"]["record_id"]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
