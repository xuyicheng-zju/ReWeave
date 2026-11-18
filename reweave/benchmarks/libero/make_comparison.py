# Copyright (c) 2026 RPent Contributors
# Modified for the ReWeave namespace.
"""Build a labelled side-by-side video from two LIBERO run directories.

The common case compares a pure-VLA baseline (from ``baseline_vla.py``) with
an agent run (from ``harness --robot libero``). Explicit labels also support
paired frozen agent bundles. The script finds each side's rollout video and burns in the task,
condition label, and recorded outcome.

    python -m reweave.benchmarks.libero.make_comparison \
        --baseline logs/baseline_10_t0_s0 \
        --agent logs/20260811_libero_10_t0_s0 \
        --out demos/libero_10_t0_s0.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

# Font is optional — labels degrade to nothing rather than failing the render.
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    Image = None


def _find_video(run_dir: Path) -> Path:
    """Locate the full-episode rollout video in a baseline or agent run dir.

    ``baseline_vla.py`` writes ``rollout.mp4``; the agent toolkit writes
    ``episode.mp4`` for the whole episode plus one ``action_<tool>.mp4``
    per primitive call — so prefer the known names over "largest file",
    which would otherwise pick a single primitive's clip.
    """
    for name in ("rollout.mp4", "episode.mp4"):
        direct = run_dir / name
        if direct.exists():
            return direct
        nested = sorted(run_dir.rglob(name))
        if nested:
            return nested[0]
    candidates = sorted(
        (p for p in run_dir.rglob("*.mp4")),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no rollout/episode mp4 found under {run_dir}")
    return candidates[0]


def _read_frames(path: Path) -> list[np.ndarray]:
    reader = imageio.get_reader(path)
    try:
        return [np.asarray(f) for f in reader]
    finally:
        reader.close()


def _outcome(run_dir: Path) -> tuple[bool | None, str]:
    """Read only the evaluator verdict; Planner finish claims are not success."""
    result = run_dir / "result.json"
    if result.exists():
        data = json.loads(result.read_text())
        verdict = data.get("success")
        if verdict is not None and type(verdict) is not bool:
            raise ValueError("baseline success must be a boolean or absent")
        return verdict, "pure VLA (pi0.5)"

    transcripts = list(run_dir.glob("transcript_*.json"))
    if len(transcripts) > 1:
        raise ValueError("comparison requires one episode, not multiple transcripts")
    if transcripts:
        data = json.loads(transcripts[0].read_text())
        verdict = data.get("official_success")
        if verdict is not None and type(verdict) is not bool:
            raise ValueError("official_success must be a boolean or absent")
        return verdict, "Agentic VLA"

    return None, run_dir.name


def _pad_to(frames: list[np.ndarray], length: int) -> list[np.ndarray]:
    """Hold the last frame so both sides run the same duration."""
    if not frames:
        raise ValueError("empty frame list")
    if len(frames) >= length:
        return frames[:length]
    return frames + [frames[-1]] * (length - len(frames))


def _resize(frame: np.ndarray, height: int) -> np.ndarray:
    if frame.shape[0] == height:
        return frame
    if Image is None:
        return frame
    width = int(round(frame.shape[1] * height / frame.shape[0]))
    return np.asarray(Image.fromarray(frame).resize((width, height), Image.BILINEAR))


def _font(size: int):
    """Best-effort truetype lookup; falls back to PIL's bitmap default."""
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _label(
    frame: np.ndarray, title: str, verdict: str, colour: tuple[int, int, int]
) -> np.ndarray:
    """Burn one side's name and outcome onto the top of its frame."""
    if Image is None:
        return frame
    image = Image.fromarray(frame.copy())
    draw = ImageDraw.Draw(image)
    band_h = max(26, frame.shape[0] // 9)
    draw.rectangle([0, 0, image.width, band_h], fill=(0, 0, 0))
    font = _font(max(11, frame.shape[0] // 20))
    draw.text((6, 2), title, fill=(235, 235, 235), font=font)
    # Right-align the verdict so SUCCESS/FAILURE reads at a glance.
    vfont = _font(max(12, frame.shape[0] // 18))
    w = draw.textlength(verdict, font=vfont)
    draw.text((image.width - w - 6, 1), verdict, fill=colour, font=vfont)
    return np.asarray(image)


def _header(width: int, text: str, height: int) -> np.ndarray:
    """A full-width caption strip naming the task, drawn once above both sides."""
    if Image is None:
        return np.zeros((height, width, 3), dtype=np.uint8)
    image = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    font = _font(max(11, height // 2))
    # Shrink until it fits rather than letting the text run off the edge.
    size = max(11, height // 2)
    while size > 8 and draw.textlength(text, font=font) > width - 12:
        size -= 1
        font = _font(size)
    w = draw.textlength(text, font=font)
    draw.text(
        ((width - w) / 2, (height - size) / 2 - 1),
        text,
        fill=(255, 255, 255),
        font=font,
    )
    return np.asarray(image)


def _pad_even(frame: np.ndarray) -> np.ndarray:
    """Pad odd dimensions for libx264's yuv420p pixel format."""
    pad_height = frame.shape[0] % 2
    pad_width = frame.shape[1] % 2
    if not pad_height and not pad_width:
        return frame
    return np.pad(
        frame,
        ((0, pad_height), (0, pad_width), (0, 0)),
        mode="constant",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", required=True, type=Path, help="pure-VLA baseline run dir"
    )
    parser.add_argument(
        "--agent", required=True, type=Path, help="agent harness run dir"
    )
    parser.add_argument(
        "--baseline-label",
        default=None,
        help="Override the inferred label for the baseline side.",
    )
    parser.add_argument(
        "--agent-label",
        default=None,
        help="Override the inferred label for the agent side.",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--task-name",
        default=None,
        help="Caption for the clip (defaults to the run dir name).",
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--height", type=int, default=256)
    args = parser.parse_args()

    left = _read_frames(_find_video(args.baseline))
    right = _read_frames(_find_video(args.agent))
    left_ok, left_name = _outcome(args.baseline)
    right_ok, right_name = _outcome(args.agent)
    left_name = args.baseline_label or left_name
    right_name = args.agent_label or right_name

    def verdict(ok: bool | None) -> str:
        return {True: "SUCCESS", False: "FAILURE", None: "?"}[ok]

    length = max(len(left), len(right))
    left = _pad_to(left, length)
    right = _pad_to(right, length)

    task_name = args.task_name or args.agent.name
    green, red = (90, 230, 120), (250, 110, 110)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps, macro_block_size=1)
    header = None
    try:
        for lf, rf in zip(left, right):
            lf = _label(
                _resize(lf, args.height),
                left_name,
                verdict(left_ok),
                green if left_ok else red,
            )
            rf = _label(
                _resize(rf, args.height),
                right_name,
                verdict(right_ok),
                green if right_ok else red,
            )
            gap = np.zeros((lf.shape[0], 4, 3), dtype=lf.dtype)
            body = np.hstack([lf, gap, rf])
            if header is None:
                header = _header(body.shape[1], task_name, max(20, args.height // 10))
            writer.append_data(_pad_even(np.vstack([header, body])))
    finally:
        writer.close()

    print(
        json.dumps(
            {
                "out": str(args.out),
                "frames": length,
                "baseline": {"dir": str(args.baseline), "success": left_ok},
                "agent": {"dir": str(args.agent), "success": right_ok},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
