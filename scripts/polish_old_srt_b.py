#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
旧 SRT 文件纯文本二次规整优化脚本 (方案 B)
不重新做 ASR，直接读取已有的 SRT，使用 Gemini 2.5 Flash 对文本进行拼写、专业术语和错别字修正。
"""

import os
import json
import time
import re
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── 配置与全局变量 ───────────────────────────────────────────────────────────

STAGING_DIR = Path("~/Documents/temp/batch_asr_staging").expanduser()
MANIFEST_PATH = STAGING_DIR / "manifest.json"

GEMINI_API_KEY = "AIzaSyADl6hpoxdZUVdEqvLylzEwV7lvdr93Jdk"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
GEMINI_MODEL = "gemini-2.5-flash"

LLM_BATCH_SIZE = 1500
MAX_CONCURRENT_WORKERS = 8

LLM_SYSTEM_PROMPT = (
    "You are a Chinese subtitle corrector. Your task is to proofread and improve subtitle text "
    "while preserving original meaning. The subtitle index and the corrected text should be in format: "
    "\"序号: 修正后的文本\". Only output corrected lines, no explanations."
)

_LLM_LINE_RE = re.compile(r"^(\d+):\s*(.+)$")

# ─── 辅助函数 ─────────────────────────────────────────────────────────────────

def parse_llm_lines(reply: str, expected_count: int) -> dict:
    corrected = {}
    fallback_lines = []
    for line in reply.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = _LLM_LINE_RE.match(line)
        if m:
            corrected[int(m.group(1)) - 1] = m.group(2).strip()
        elif not re.match(r"^(Whisper|Native|Final|Output)\s*[:=]", line, re.I):
            fallback_lines.append(line)
    if not corrected and len(fallback_lines) == expected_count:
        corrected = {i: text for i, text in enumerate(fallback_lines)}
    return corrected

def build_system_prompt_for_video(mp4_name: str) -> str:
    terms = ["FastAPI", "Python", "LLM", "API", "RAG", "Prompt", "Agent", "JSON", "Docker", "GPU", "CUDA"]
    name_lower = mp4_name.lower()
    if "gradio" in name_lower:
        terms.extend(["Gradio", "WebUI", "Interface"])
    if "langchain" in name_lower:
        terms.extend(["LangChain", "RetrievalQA", "Memory", "Chain", "OutputParser"])
    if "faiss" in name_lower:
        terms.extend(["FAISS", "VectorStore", "Index", "Similarity"])
    if "chroma" in name_lower:
        terms.extend(["Chroma", "ChromaDB", "Client", "Collection"])
    if "milvus" in name_lower:
        terms.extend(["Milvus", "Connections", "Schema", "Utility"])
    if "pinecone" in name_lower:
        terms.extend(["Pinecone", "Dimension", "Index", "Upsert"])
    if "ragas" in name_lower:
        terms.extend(["Ragas", "Evaluation", "Faithfulness", "Answer Relevance"])
    if "langsmith" in name_lower:
        terms.extend(["LangSmith", "Tracer", "Dataset", "Run"])
    if "vllm" in name_lower:
        terms.extend(["vLLM", "Offline Inference", "Server"])
    if "ollama" in name_lower:
        terms.extend(["Ollama", "Model"])
    if "chatglm" in name_lower or "qwen" in name_lower:
        terms.extend(["ChatGLM", "Qwen", "HuggingFace", "ModelScope", "Transformers"])
    if "parser" in name_lower or "解析器" in name_lower:
        terms.extend(["Pydantic", "Enum", "Structure", "OutputParser"])
        
    terms = list(dict.fromkeys(terms))
    terms_str = ", ".join(terms)
    
    custom_prompt = (
        "You are a Chinese subtitle corrector. Your task is to proofread and improve subtitle text "
        "while preserving original meaning. The subtitle index and the corrected text should be in format: "
        "\"序号: 修正后的文本\". Only output corrected lines, no explanations.\n"
        "Important: This video is about AI and software development. Correct phonetically misrecognized terms "
        f"and output them in their standard form. Key terms expected in this video include: [{terms_str}]."
    )
    return custom_prompt

def load_srt_segments(srt_path: Path) -> list:
    """解析已有 SRT 文件提取段落"""
    if not srt_path.exists():
        return []
    content = srt_path.read_text(encoding="utf-8")
    
    # 简单的 SRT 解析正则
    # 匹配: 序号 \n 时间戳 \n 文本
    pattern = re.compile(r"(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n((?:[^\n]+\n*)+)")
    segments = []
    for match in pattern.finditer(content):
        idx = int(match.group(1))
        start_str = match.group(2)
        end_str = match.group(3)
        text = match.group(4).strip()
        segments.append({
            "index": idx,
            "start_str": start_str,
            "end_str": end_str,
            "text": text
        })
    return segments

def save_srt_segments(srt_path: Path, segments: list) -> None:
    with open(srt_path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"{seg['index']}\n")
            f.write(f"{seg['start_str']} --> {seg['end_str']}\n")
            f.write(f"{seg['text']}\n\n")

def polish_batch(batch_idx: int, batch: list, system_prompt: str) -> dict:
    lines = [f"{i + 1}: {seg['text']}" for i, seg in enumerate(batch)]
    user_text = "\n".join(lines)
    body = {
        "model": GEMINI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.2,
        "max_tokens": 8192,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GEMINI_API_KEY}",
    }
    
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                GEMINI_API_URL,
                data=json.dumps(body).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            reply = data["choices"][0]["message"]["content"]
            return parse_llm_lines(reply, len(batch))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace").strip()
            print(f"      ⚠ Gemini HTTP {exc.code} (Batch {batch_idx}): {detail[:100]}", flush=True)
            time.sleep(3 * (attempt + 1))
        except Exception as exc:
            print(f"      ⚠ Gemini 异常 (Batch {batch_idx}): {exc}", flush=True)
            time.sleep(3 * (attempt + 1))
    return {}

def polish_old_srt(mp4_path: Path) -> tuple[bool, str]:
    srt_path = mp4_path.with_suffix(".srt")
    if not srt_path.exists():
        return False, "SRT 文件不存在"
        
    segments = load_srt_segments(srt_path)
    if not segments:
        return False, "SRT 文件内容解析为空"
        
    result = list(segments)
    batches = []
    for batch_start in range(0, len(segments), LLM_BATCH_SIZE):
        batch = segments[batch_start : batch_start + LLM_BATCH_SIZE]
        batches.append((batch_start, batch))
        
    system_prompt = build_system_prompt_for_video(mp4_path.name)
    corrected_results = {}
    
    with ThreadPoolExecutor(max_workers=min(len(batches), 4)) as executor:
        future_to_batch = {
            executor.submit(polish_batch, idx, batch, system_prompt): (idx, start_idx)
            for idx, (start_idx, batch) in enumerate(batches)
        }
        for future in as_completed(future_to_batch):
            idx, start_idx = future_to_batch[future]
            try:
                batch_corrected = future.result()
                for local_i, corrected_text in batch_corrected.items():
                    corrected_results[start_idx + local_i] = corrected_text
            except Exception as e:
                print(f"    ⚠ Batch {idx} 发生异常: {e}", flush=True)

    # 用规整后的文本更新 segments 缓存
    changed = False
    for i in range(len(result)):
        if i in corrected_results and corrected_results[i] != result[i]["text"]:
            result[i]["text"] = corrected_results[i]
            changed = True
            
    if changed:
        save_srt_segments(srt_path, result)
        return True, "成功更新规整"
    else:
        return True, "文本无需修改，保持原样"

# ─── 主入口 ───────────────────────────────────────────────────────────────────

def main():
    if not MANIFEST_PATH.exists():
        print(f"未找到 manifest.json: {MANIFEST_PATH}，无法识别旧文件清单。")
        return
        
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"读取 manifest.json 失败: {e}")
        return

    # 找出没有 JSON ASR 缓存的旧 SRT 视频 (即在我们的主程序跑之前就存在的)
    old_videos = []
    for slug, info in manifest.items():
        json_path = Path(info["json"])
        if not json_path.exists():
            mp4_path = Path(info["mp4"])
            if mp4_path.exists():
                old_videos.append(mp4_path)

    # 排序
    old_videos.sort(key=lambda p: str(p))

    if not old_videos:
        print("未检测到属于上一代有缺陷断句的 46 个旧 SRT 文件，无需进行方案 B 纯文本规整。")
        return

    print("=" * 60)
    print("方案 B：直接对 46 个旧 SRT 进行 Gemini 并发文本规整")
    print(f"待处理旧字幕视频数: {len(old_videos)}")
    print("=" * 60, flush=True)

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS) as executor:
        future_to_srt = {
            executor.submit(polish_old_srt, mp4_path): mp4_path
            for mp4_path in old_videos
        }
        
        for future in as_completed(future_to_srt):
            mp4_path = future_to_srt[future]
            try:
                success, msg = future.result()
                if success:
                    print(f"  ✓ 【处理成功】 {mp4_path.name} : {msg}", flush=True)
                else:
                    print(f"  ✗ 【处理失败】 {mp4_path.name} : {msg}", flush=True)
            except Exception as e:
                print(f"  ✗ 【处理异常】 {mp4_path.name} : {e}", flush=True)

    print("\n" + "=" * 60)
    print("方案 B 纯文本二次规整全部完成！")
    print("=" * 60, flush=True)

if __name__ == "__main__":
    main()
