#!/usr/bin/env python3
"""Use raw Apple Vision OCR cues to adjudicate OGG-vs-URL transcript differences."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def normalized(text: str) -> str:
    return "".join(re.findall(r"[0-9A-Za-z\u3400-\u9fff]+", text.lower()))


def exact_support(probe: str, ocr: str) -> bool:
    return bool(probe) and probe in ocr


def evidence_cues(cues: list[dict[str, Any]], start: int, end: int) -> list[dict[str, Any]]:
    # The character position is from the OGG/Gemini template, not a visual claim.
    # Add a small context margin so edits at a cue boundary remain inspectable.
    left, right = max(0, start - 10), end + 10
    return [
        cue
        for cue in cues
        if int(cue.get("matched_template_character_start", -1)) < right
        and int(cue.get("matched_template_character_end", -1)) > left
        and str(cue.get("ocr_text", "")).strip()
    ]


def verdict(left_probe: str, right_probe: str, ocr: str) -> str:
    left_supported = exact_support(left_probe, ocr)
    right_supported = exact_support(right_probe, ocr)
    if left_supported and not right_supported:
        return "ocr_supports_ogg"
    if right_supported and not left_supported:
        return "ocr_supports_url"
    if left_supported and right_supported:
        return "ocr_supports_both_or_shared_context"
    return "ocr_insufficient_or_boundary_split"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ogg", type=Path, required=True)
    parser.add_argument("--url", type=Path, required=True)
    parser.add_argument("--difference-json", type=Path, required=True)
    parser.add_argument("--ocr-fusion-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    ogg = normalized(args.ogg.read_text(encoding="utf-8"))
    url = normalized(args.url.read_text(encoding="utf-8"))
    differences = json.loads(args.difference_json.read_text(encoding="utf-8"))["ogg_vs_url"]["differences"]
    cues = json.loads(args.ocr_fusion_json.read_text(encoding="utf-8")).get("cues", [])
    decisions: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for item in differences:
        left_start, left_end = map(int, item["left_range"])
        right_start, right_end = map(int, item["right_range"])
        left_probe = ogg[max(0, left_start - 6):min(len(ogg), left_end + 7)]
        right_probe = url[max(0, right_start - 6):min(len(url), right_end + 7)]
        matched = evidence_cues(cues, left_start, left_end)
        raw_ocr = normalized("".join(str(cue.get("ocr_text", "")) for cue in matched))
        decision = verdict(left_probe, right_probe, raw_ocr)
        counts[decision] = counts.get(decision, 0) + 1
        decisions.append(
            {
                **item,
                "verdict": decision,
                "ogg_probe": left_probe,
                "url_probe": right_probe,
                "raw_ocr_evidence": raw_ocr,
                "ocr_cues": [
                    {
                        "start": cue.get("start"),
                        "end": cue.get("end"),
                        "ocr_text": cue.get("ocr_text"),
                        "confidence": cue.get("ocr_confidence"),
                        "source": cue.get("source"),
                    }
                    for cue in matched
                ],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "raw_apple_vision_ocr_exact_probe",
        "caveat": "OCR text is raw Vision evidence. Its cue-to-template position was established by OGG/NUC alignment, so unsupported items are not proof that URL is correct.",
        "counts": counts,
        "decisions": decisions,
    }
    (args.output_dir / "ocr-adjudicated-transcript-differences.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# OCR 裁决 OGG 与 URL 转写差异",
        "",
        "原始 Apple Vision OCR 用作文字证据；OCR cue 的定位使用既有 OGG/NUC 对齐，因此“支持 OGG”是强画面支持，`ocr_insufficient_or_boundary_split` 不构成对 URL 的支持。",
        "",
        "## 汇总",
        "",
    ]
    lines.extend(f"- `{key}`：{value}" for key, value in sorted(counts.items()))
    for item in decisions:
        cue_summary = ", ".join(
            f"{float(cue['start']):.1f}-{float(cue['end']):.1f}s {cue['ocr_text']}"
            for cue in item["ocr_cues"]
        ) or "∅"
        lines.extend(
            [
                "",
                f"## {item['index']}. {item['verdict']}",
                "",
                f"- OGG：`{item['ogg_changed'] or '∅'}`；URL：`{item['url_changed'] or '∅'}`",
                f"- OGG 探针：`{item['ogg_probe']}`",
                f"- URL 探针：`{item['url_probe']}`",
                f"- 原始 OCR：`{item['raw_ocr_evidence'] or '∅'}`",
                f"- OCR cue：{cue_summary}",
            ]
        )
    (args.output_dir / "ocr-adjudicated-transcript-differences.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
