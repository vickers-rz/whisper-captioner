#!/usr/bin/env python3
"""Extract planned non-contiguous video frames and write their true timestamps."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="/opt/homebrew/bin/ffmpeg")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamps: dict[str, float] = {}
    for window in plan.get("windows", []):
        window_id = int(window["id"])
        start = float(window["start"])
        end = float(window["end"])
        fps = float(window["fps"])
        pattern = args.output_dir / f"target-{window_id:03d}-%05d.jpg"
        command = [args.ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(args.video), "-vf", f"fps={fps}", "-q:v", "3", str(pattern)]
        subprocess.run(command, check=True)
        for offset, frame in enumerate(sorted(args.output_dir.glob(f"target-{window_id:03d}-*.jpg"))):
            timestamps[frame.name] = round(start + offset / fps, 6)
    (args.output_dir / "timestamps.json").write_text(json.dumps(timestamps, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"frames": len(timestamps), "timestamps": str(args.output_dir / 'timestamps.json')}, ensure_ascii=False))
    return 0
