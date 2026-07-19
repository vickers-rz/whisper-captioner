#!/usr/bin/env python3
"""Run audio-only ASR against a public YouTube URL through Gemini's video input."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROMPT = """
仅将此公开视频中的实际语音转写为连续、完整的简体中文全文。
所有中文内容必须使用中国大陆规范简体字；如果听到或识别出繁体字词，必须转换为对应简体字形。
禁止输出繁体中文字符，例如將、這、個、臺、為、與、後、說、時、會、應、國、學、開、關、過、還。
把视频作为音频源：忽略所有画面、烧录字幕、标题、缩略图、简介、章节和其他视觉信息。
不要总结、不要分析、不要添加时间戳、不要标注说话人、不要说明你的处理过程。
不要根据画面或常识补写没有听到的内容；保留重复、过渡句、广告口播和结尾。
只输出简体中文转写正文。
""".strip()


@dataclass(frozen=True)
class GeminiUrlAsrResult:
    text: str
    metadata: dict[str, Any]
    raw_response: dict[str, Any]


def dump_model(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def extract_text(data: Any) -> str:
    if isinstance(data, dict):
        output_text = data.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()
        texts: list[str] = []

        def visit(item: Any) -> None:
            if isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    texts.append(item["text"])
                    return
                for child in item.values():
                    visit(child)
            elif isinstance(item, list):
                for child in item:
                    visit(child)

        visit(data)
        return "\n".join(texts).strip()
    return ""


def transcribe_youtube_url(
    *,
    url: str,
    api_key: str,
    model: str = "gemini-2.5-flash",
    timeout: float = 1200.0,
    client: Any | None = None,
) -> GeminiUrlAsrResult:
    """Transcribe a public YouTube URL with an audio-only output contract."""
    if not api_key.strip():
        raise RuntimeError("GEMINI_API_KEY is required")
    if client is None:
        from google import genai

        client = genai.Client(api_key=api_key)

    started = time.monotonic()
    response = client.interactions.create(
        model=model,
        input=[
            {"type": "video", "uri": url, "resolution": "low"},
            {"type": "text", "text": PROMPT},
        ],
        generation_config={"temperature": 0.0, "max_output_tokens": 60000},
        response_modalities=["text"],
        store=False,
        timeout=timeout,
    )
    raw = dump_model(response)
    text = extract_text(raw)
    if not text:
        raise RuntimeError("Gemini returned no transcript text")
    metadata = {
        "input": "public-youtube-url-video-part",
        "prompt_mode": "audio-only-asr-ignore-visuals",
        "visual_analysis_requested": False,
        "timestamps_requested": False,
        "model": model,
        "url": url,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "characters": len(text),
    }
    return GeminiUrlAsrResult(text=text, metadata=metadata, raw_response=raw)


def save_result(result: GeminiUrlAsrResult, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    transcript = output_dir / "gemini-youtube-url-audio-only-transcript.txt"
    metadata = output_dir / "gemini-youtube-url-audio-only-metadata.json"
    transcript.write_text(result.text.rstrip() + "\n", encoding="utf-8")
    metadata.write_text(
        json.dumps(
            {**result.metadata, "response": result.raw_response},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"transcript": str(transcript), "metadata": str(metadata)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is required")

    result = transcribe_youtube_url(
        url=args.url,
        api_key=api_key,
        model=args.model,
    )
    save_result(result, args.output_dir)
    print(json.dumps(result.metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
