#!/usr/bin/env python3
"""Use targeted Apple Vision OCR to adjudicate NUC-vs-Gemini text disputes.

This script never changes timestamps. It applies only conservative, equal-length
replacement corrections where OCR context supports NUC more strongly than Gemini.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


def normalized_with_indices(text: str) -> tuple[list[str], list[int]]:
    chars: list[str] = []
    indices: list[int] = []
    for index, char in enumerate(text):
        for item in unicodedata.normalize("NFKC", char).lower():
            if item.isalnum() or "\u3400" <= item <= "\u9fff":
                chars.append(item)
                indices.append(index)
    return chars, indices


def normalize(text: str) -> str:
    return "".join(normalized_with_indices(text)[0])


def local_score(observed: str, candidate: str) -> float:
    if not observed or not candidate:
        return 0.0
    matcher = difflib.SequenceMatcher(None, observed, candidate, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return matched / max(1, min(len(observed), len(candidate)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemini", type=Path, required=True)
    parser.add_argument("--differences", type=Path, required=True)
    parser.add_argument("--ocr", type=Path, required=True)
    parser.add_argument("--output-transcript", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--context", type=int, default=12)
    parser.add_argument("--minimum-score", type=float, default=0.68)
    parser.add_argument("--minimum-margin", type=float, default=0.12)
    args = parser.parse_args()

    gemini_text = args.gemini.read_text(encoding="utf-8").strip()
    gemini_chars, original_indices = normalized_with_indices(gemini_text)
    differences = json.loads(args.differences.read_text(encoding="utf-8"))["difference_groups"]
    ocr_cues = json.loads(args.ocr.read_text(encoding="utf-8")).get("cues", [])
    adjudications: list[dict[str, Any]] = []
    edits: list[tuple[int, int, str, int]] = []

    for difference in differences:
        g_start, g_end = map(int, difference["gemini_range"])
        n_start, n_end = map(int, difference["nuc_range"])
        gemini_variant = str(difference.get("gemini_text", ""))
        nuc_variant = str(difference.get("nuc_text", ""))
        time_window = difference.get("time_window") or [None, None]
        record: dict[str, Any] = {
            "index": difference["index"],
            "operation": difference["operation"],
            "time_window": time_window,
            "gemini_text": gemini_variant,
            "nuc_text": nuc_variant,
            "decision": "insufficient_evidence",
            "applied": False,
        }
        if (
            difference["operation"] != "replace"
            or not gemini_variant
            or not nuc_variant
            or len(gemini_variant) != len(nuc_variant)
            or g_start >= g_end
            or n_start >= n_end
            or time_window[0] is None
        ):
            record["reason"] = "only_equal_length_replacements_are_auto-applicable"
            adjudications.append(record)
            continue

        window_start = float(time_window[0]) - 1.2
        window_end = float(time_window[1]) + 1.2
        observed = [
            normalize(str(cue.get("text", "")))
            for cue in ocr_cues
            if float(cue.get("end", 0.0)) >= window_start
            and float(cue.get("start", 0.0)) <= window_end
        ]
        observed = [item for item in observed if item]
        if not observed:
            record["reason"] = "no_ocr_text_in_dispute_window"
            adjudications.append(record)
            continue

        left = max(0, g_start - args.context)
        right = min(len(gemini_chars), g_end + args.context)
        gemini_context = "".join(gemini_chars[left:right])
        relative_start = g_start - left
        relative_end = g_end - left
        nuc_context = (
            gemini_context[:relative_start]
            + nuc_variant
            + gemini_context[relative_end:]
        )
        gemini_score = max(local_score(item, gemini_context) for item in observed)
        nuc_score = max(local_score(item, nuc_context) for item in observed)
        record.update(
            ocr_text=observed,
            gemini_context=gemini_context,
            nuc_context=nuc_context,
            gemini_score=round(gemini_score, 6),
            nuc_score=round(nuc_score, 6),
        )
        if nuc_score >= args.minimum_score and nuc_score - gemini_score >= args.minimum_margin:
            original_start = original_indices[g_start]
            original_end = original_indices[g_end - 1] + 1
            edits.append((original_start, original_end, nuc_variant, int(difference["index"])))
            record["decision"] = "ocr_supports_nuc"
            record["applied"] = True
        elif gemini_score >= args.minimum_score and gemini_score - nuc_score >= args.minimum_margin:
            record["decision"] = "ocr_supports_gemini"
            record["reason"] = "gemini_retained"
        else:
            record["reason"] = "ocr_scores_not_decisive"
        adjudications.append(record)

    corrected = gemini_text
    for start, end, replacement, _index in sorted(edits, reverse=True):
        corrected = corrected[:start] + replacement + corrected[end:]
    args.output_transcript.parent.mkdir(parents=True, exist_ok=True)
    args.output_transcript.write_text(corrected + "\n", encoding="utf-8")
    report = {
        "policy": "OCR text only; timestamps unchanged; conservative equal-length replacements",
        "difference_groups": len(differences),
        "applied_corrections": len(edits),
        "adjudications": adjudications,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"difference_groups": len(differences), "applied_corrections": len(edits)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
