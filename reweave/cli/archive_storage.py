"""Finalize exact original media after episode exit, before evidence publication."""

import argparse
import json
from pathlib import Path

from reweave.evolution.evidence.media_storage import finalize_media


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(finalize_media(args.attempt_directory), indent=2))


if __name__ == "__main__":
    main()
