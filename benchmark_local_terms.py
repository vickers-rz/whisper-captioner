from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


ROOT = Path("/Users/vickers/whisper-captioner")
INPUT_PATH = ROOT / "benchmark_terms_input.txt"
GGUF_MODEL = Path("/Users/vickers/local-models/gguf/qwen2.5-3b-instruct-q4_k_m.gguf")
MLX_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"

PROMPT = """你是一个术语抽取助手。请从下面文本中抽取专有名词、模型名、产品名、品牌名、英文缩写和英文术语。
输出严格为 JSON，格式如下：
{"terms":[{"term":"...", "type":"person|brand|product|model|acronym|term"}]}
不要输出解释。

文本：
"""


def run_gguf(text: str) -> dict:
    prompt = PROMPT + text
    cmd = [
        "/opt/homebrew/bin/llama-cli",
        "-m",
        str(GGUF_MODEL),
        "-ngl",
        "999",
        "-c",
        "4096",
        "-n",
        "256",
        "--temp",
        "0.1",
        "-no-cnv",
        "-p",
        prompt,
    ]
    started = time.monotonic()
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    elapsed = time.monotonic() - started
    return {
        "name": "gguf_llama_cpp",
        "elapsed_sec": round(elapsed, 3),
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip()[-4000:],
        "stderr": proc.stderr.strip()[-4000:],
    }


def run_mlx(text: str) -> dict:
    code = f"""
import time
from mlx_lm import load, generate
model, tokenizer = load("{MLX_MODEL}")
prompt = {PROMPT!r} + {text!r}
started = time.monotonic()
output = generate(model, tokenizer, prompt=prompt, max_tokens=256, temp=0.1, verbose=False)
elapsed = time.monotonic() - started
print("ELAPSED=" + str(round(elapsed, 3)))
print(output)
"""
    cmd = ["conda", "run", "-n", "rapidmlx", "python", "-c", code]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    elapsed = None
    stdout = proc.stdout.strip()
    if "ELAPSED=" in stdout:
        first, _, rest = stdout.partition("\n")
        try:
            elapsed = float(first.replace("ELAPSED=", "").strip())
        except ValueError:
            elapsed = None
        stdout = rest.strip()
    return {
        "name": "mlx_qwen",
        "elapsed_sec": elapsed,
        "returncode": proc.returncode,
        "stdout": stdout[-4000:],
        "stderr": proc.stderr.strip()[-4000:],
    }


def main() -> None:
    text = INPUT_PATH.read_text(encoding="utf-8")
    results = {
        "input_path": str(INPUT_PATH),
        "input_text": text,
        "results": [
            run_gguf(text),
            run_mlx(text),
        ],
    }
    out = ROOT / "benchmark_local_terms_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
