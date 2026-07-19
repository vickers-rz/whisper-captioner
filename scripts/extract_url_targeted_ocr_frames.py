#!/usr/bin/env python3
"""Extract targeted OCR frames from a local video or a remote yt-dlp URL.

For a remote source, yt-dlp resolves a temporary low-bandwidth video URL and
FFmpeg requests only the planned time ranges. The complete video is not saved.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.chrome_cookie_snapshot import yt_dlp_cookie_session


def is_remote(value: str) -> bool:
    return urlparse(value).scheme in {"http", "https"}


def resolve_video_input(
    source: str,
    yt_dlp: str,
    cookie_args: list[str],
    height: int,
) -> str:
    if not is_remote(source):
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"Local video not found: {path}")
        return str(path)
    command = [
        yt_dlp,
        "--no-playlist",
        "--get-url",
        "-f",
        f"bv*[height<={height}]/b[height<={height}]",
    ]
    command.extend(cookie_args)
    command.append(source)
    completed = subprocess.run(command, text=True, capture_output=True, check=True)
    urls = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not urls:
        raise RuntimeError("yt-dlp did not resolve a video stream URL")
    return urls[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cookies-from-chrome", action="store_true")
    parser.add_argument("--chrome-profile", default="Default")
    parser.add_argument("--cookies-browser-spec", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--ffmpeg", default="/opt/homebrew/bin/ffmpeg")
    parser.add_argument("--yt-dlp", default="/opt/homebrew/bin/yt-dlp")
    args = parser.parse_args()

    with yt_dlp_cookie_session(
        enabled=args.cookies_from_chrome,
        chrome_profile=args.chrome_profile,
        forwarded_browser_spec=args.cookies_browser_spec,
    ) as cookie_session:
        cookie_args = cookie_session.arguments
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        remote = is_remote(args.source)
        video_input = resolve_video_input(
            args.source, args.yt_dlp, cookie_args, args.height
        )
        timestamps: dict[str, float] = {}
        windows = plan.get("windows", [])
        for position, window in enumerate(windows):
            # YouTube signed media URLs can expire during a large disagreement job.
            # Refresh periodically while still requesting only selected ranges.
            if remote and position and position % 40 == 0:
                video_input = resolve_video_input(
                    args.source, args.yt_dlp, cookie_args, args.height
                )
            window_id = int(window["id"])
            start = float(window["start"])
            end = float(window["end"])
            fps = float(window["fps"])
            pattern = args.output_dir / f"target-{window_id:04d}-%05d.jpg"
            command = [
                args.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{start:.3f}",
                "-t",
                f"{max(0.05, end - start):.3f}",
                "-i",
                video_input,
                "-vf",
                f"fps={fps}",
                "-q:v",
                "3",
                str(pattern),
            ]
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0 and remote:
                video_input = resolve_video_input(
                    args.source, args.yt_dlp, cookie_args, args.height
                )
                command[command.index("-i") + 1] = video_input
                completed = subprocess.run(command, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"FFmpeg failed for OCR window {window_id}")
            matches = sorted(args.output_dir.glob(f"target-{window_id:04d}-*.jpg"))
            for offset, frame in enumerate(matches):
                timestamps[frame.name] = round(start + offset / fps, 6)
            print(
                f"OCR window {window_id}/{len(windows)}: "
                f"{start:.2f}-{end:.2f}s, {len(matches)} frames",
                flush=True,
            )

        timestamp_path = args.output_dir / "timestamps.json"
        timestamp_path.write_text(
            json.dumps(timestamps, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"frames": len(timestamps), "timestamps": str(timestamp_path)}))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
