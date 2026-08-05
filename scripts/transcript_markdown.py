"""Markdown formatting helpers for standalone ASR transcripts."""

from __future__ import annotations

import re
from pathlib import Path

CJK = r"\u2e80-\u2eff\u2f00-\u2fdf\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"

RICH_MARKDOWN_STYLE = """<style>
  body {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
    font-size: 17px;
    line-height: 1.78;
    color: #202124;
    max-width: 920px;
    margin: 0 auto;
    padding: 24px 28px 56px;
  }
  h1 {
    font-size: 2rem;
    line-height: 1.25;
    margin: 0 0 1rem;
  }
  h2 {
    font-size: 1.42rem;
    margin-top: 2.2rem;
    padding-bottom: 0.35rem;
    border-bottom: 2px solid #dfe3ea;
  }
  h3 {
    font-size: 1.12rem;
    margin-top: 1.4rem;
  }
  strong {
    font-weight: 700;
    color: #111827;
  }
  u {
    text-decoration-thickness: 0.12em;
    text-underline-offset: 0.18em;
    text-decoration-color: #e11d48;
  }
  code {
    font-family: "SF Mono", Menlo, Consolas, monospace;
    font-size: 0.92em;
    padding: 0.12em 0.34em;
    border-radius: 4px;
    background: #f1f5f9;
  }
  blockquote {
    margin: 1rem 0;
    padding: 0.7rem 1rem;
    border-left: 4px solid #64748b;
    background: #f8fafc;
    color: #334155;
  }
  table {
    border-collapse: collapse;
    width: 100%;
    margin: 1rem 0;
    font-size: 0.95em;
  }
  th, td {
    border: 1px solid #d6dae1;
    padding: 0.52rem 0.64rem;
    vertical-align: top;
  }
  th {
    background: #f3f6fa;
    font-weight: 700;
  }
  hr {
    border: 0;
    border-top: 1px solid #e5e7eb;
    margin: 2rem 0;
  }
</style>"""


def markdown_escape_inline(value: str) -> str:
    return value.replace("\\", "\\\\").replace("`", "\\`")


def format_cjk_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(rf"([{CJK}])([A-Za-z0-9@#&%])", r"\1 \2", text)
    text = re.sub(rf"([A-Za-z0-9@#&%])([{CJK}])", r"\1 \2", text)
    text = re.sub(rf"([{CJK}]),", r"\1，", text)
    text = re.sub(rf"([{CJK}])\.", r"\1。", text)
    text = re.sub(rf"([{CJK}])\?", r"\1？", text)
    text = re.sub(rf"([{CJK}])!", r"\1！", text)
    text = re.sub(rf"([{CJK}]):", r"\1：", text)
    text = re.sub(rf"([{CJK}]);", r"\1；", text)
    return text


def sentence_paragraphs(text: str, *, max_sentences: int = 4, max_chars: int = 520) -> list[str]:
    existing = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    if len(existing) > 1:
        return existing

    sentences = [
        item.strip()
        for item in re.findall(r".+?(?:[。！？!?]+[」』”’）)]*|$)", text, flags=re.S)
        if item.strip()
    ]
    paragraphs: list[str] = []
    current: list[str] = []
    current_chars = 0
    for sentence in sentences:
        current.append(sentence)
        current_chars += len(sentence)
        if len(current) >= max_sentences or current_chars >= max_chars:
            paragraphs.append("".join(current).strip())
            current = []
            current_chars = 0
    if current:
        paragraphs.append("".join(current).strip())
    return paragraphs or ([text] if text else [])


def transcript_markdown(
    text: str,
    *,
    title: str,
    source: str,
    model: str,
    audio_path: Path | None = None,
    input_label: str | None = None,
) -> str:
    formatted = format_cjk_text(text)
    lines = [
        RICH_MARKDOWN_STYLE,
        "",
        f"# {title}",
        "",
        "## Metadata",
        "",
        f"- Source: `{markdown_escape_inline(source)}`",
        f"- Model: `{markdown_escape_inline(model)}`",
    ]
    if input_label is not None:
        lines.append(f"- Input: `{markdown_escape_inline(input_label)}`")
    if audio_path is not None:
        lines.append(f"- Audio: `{markdown_escape_inline(str(audio_path))}`")
    lines.extend(["", "## Transcript", ""])
    for paragraph in sentence_paragraphs(formatted):
        lines.extend([paragraph, ""])
    lines.append("")
    return "\n".join(lines)


def segmented_markdown(
    body: str,
    *,
    title: str,
    source: str,
    model: str,
    input_path: Path,
    input_label: str = "Input",
) -> str:
    formatted = format_cjk_text(body)
    lines = [
        RICH_MARKDOWN_STYLE,
        "",
        f"# {title}",
        "",
        "## Metadata",
        "",
        f"- Source: `{markdown_escape_inline(source)}`",
        f"- Model: `{markdown_escape_inline(model)}`",
        f"- {input_label}: `{markdown_escape_inline(str(input_path))}`",
        "",
        "## Segmented Transcript",
        "",
        formatted,
        "",
    ]
    return "\n".join(lines)
