from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request


MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
RAPID_MLX_BASE_URL = os.environ.get("RAPID_MLX_BASE_URL", "http://127.0.0.1:8765/v1")
RAPID_MLX_MODEL = os.environ.get("RAPID_MLX_MODEL", "qwen2.5-3b-mlx")
VALID_TYPES = {"person", "brand", "product", "model", "acronym", "term"}
SYSTEM_PROMPT = (
    "你是一个术语抽取函数。严格输出一个 JSON object，不要输出 Markdown。"
    "不要输出分析、解释、<think>、内部 reasoning 或多余文本。"
    "输出必须符合 schema: "
    "{\"terms\":[{\"term\":\"string\",\"type\":\"person|brand|product|model|acronym|term\"}]}。"
    "term 必须逐字保留原文大小写、连字符、点号和数字。"
)


def build_messages(text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                "从文本中抽取专有名词、模型名、产品名、品牌名、英文缩写和英文术语。"
                "只返回符合 schema 的 JSON object。\n"
                f"文本：\n{text}"
            ),
        },
    ]


def parse_json_object(output: str) -> dict:
    output = output.replace("<think>", "").replace("</think>", "")
    first = output.find("{")
    last = output.rfind("}")
    if first >= 0 and last > first:
        output = output[first : last + 1]
    return normalize_terms(json.loads(output))


def normalize_terms(data: dict) -> dict:
    terms = []
    seen = set()
    for item in data.get("terms", []):
        if isinstance(item, str):
            term = item.strip()
            term_type = "term"
        elif isinstance(item, dict):
            term = str(item.get("term", "")).strip()
            term_type = str(item.get("type", "term")).strip()
        else:
            continue
        if not term or term in seen:
            continue
        seen.add(term)
        if term_type not in VALID_TYPES:
            term_type = "term"
        terms.append({"term": term, "type": term_type})
    return {"terms": terms}


def generate_with_rapid_mlx(messages: list[dict[str, str]], max_tokens: int) -> dict:
    payload = {
        "model": RAPID_MLX_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.1,
        # Rapid-MLX accepts this metadata in newer builds; older OpenAI-compatible
        # handlers ignore it, and the server can also be launched with --no-thinking.
        "no_thinking": True,
    }
    req = urllib.request.Request(
        f"{RAPID_MLX_BASE_URL.rstrip('/')}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=8) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    content = data["choices"][0]["message"].get("content", "")
    return parse_json_object(content)


def generate_with_mlx_lm(model_name: str, messages: list[dict[str, str]], max_tokens: int) -> dict:
    from mlx_lm import generate, load

    model, tokenizer = load(model_name)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    output = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    return parse_json_object(output)


def main() -> int:
    payload = json.loads(sys.stdin.read())
    text = str(payload.get("text", "")).strip()
    if not text:
        print(json.dumps({"terms": []}, ensure_ascii=False))
        return 0

    started = time.monotonic()
    max_tokens = int(payload.get("max_tokens", 192))
    messages = build_messages(text)
    backend = "rapid-mlx"
    try:
        data = generate_with_rapid_mlx(messages, max_tokens)
    except (OSError, urllib.error.URLError, KeyError, ValueError, json.JSONDecodeError):
        backend = "mlx-lm"
        data = generate_with_mlx_lm(payload.get("model") or MODEL, messages, max_tokens)
    data["backend"] = backend
    data["elapsed_sec"] = round(time.monotonic() - started, 3)
    print(json.dumps(data, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
