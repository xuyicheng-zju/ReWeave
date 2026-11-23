"""Keep internal run labels out of the public project page and its player."""

from pathlib import Path
import re


def test_public_demo_copy():
    site = Path(__file__).resolve().parents[1] / "website"
    internal_label = re.compile(
        r"\bwitness(?:es|ed)?\b|\barchive(?:d)?\s+(?:development|witness)\b"
        r"|\bfrozen bundle identifiers\b|\brecording details\b|\b[a-f0-9]{32}\b",
        re.IGNORECASE,
    )
    for name in ("index.html", "script.js"):
        match = internal_label.search((site / name).read_text(encoding="utf-8"))
        assert match is None, f"Internal run label in {name}: {match.group()}"
