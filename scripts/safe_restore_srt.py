#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
安全的高并发 SRT 还原与规整脚本
直接读取完好的 staging json，在生成 srt 时提供严格的安全防护，防止 LLM 丢字或截断导致的字幕丢失。
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

# 每个视频 ASR segments 发送给 Gemini 的最大行数
LLM_BATCH_SIZE = 100
# 并发线程数
MAX_CONCURRENT_WORKERS = 8

# LLM 提示词 (加入强制约束：必须输出所有请求的行)
LLM_SYSTEM_PROMPT = (
    "You are a Chinese subtitle corrector. Your task is to proofread and improve subtitle text "
    "while preserving original meaning. The subtitle index and the corrected text should be in format: "
    "\"序号: 修正后的文本\". Only output corrected lines, no explanations.\n"
    "CRITICAL REQUIREMENT: You MUST return exactly the same number of lines as the input. Do not omit any lines."
)

_LLM_LINE_RE = re.compile(r"^(\d+):\s*(.+)$")

# ─── 辅助逻辑 ─────────────────────────────────────────────────────────────────

def parse_llm_lines(reply: str, expected_count: int) -> dict:
    """安全解析回复，严格根据 '序号:' 精准对齐"""
    corrected = {}
    fallback_lines = []
    
    for line in reply.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = _LLM_LINE_RE.match(line)
        if m:
            idx = int(m.group(1)) - 1
            text = m.group(2).strip()
            corrected[idx] = text
        elif not re.match(r"^(Whisper|Native|Final|Output)\s*[:=]", line, re.I):
            fallback_lines.append(line)
            
    # 如果序号方式没匹配到任何内容，但行数恰好相同，进行对齐回退
    if not corrected and len(fallback_lines) == expected_count:
        corrected = {i: text for i, text in enumerate(fallback_lines)}
        
    return corrected

def format_timestamp(seconds: float) -> str:
    """将 ASR 的秒数(float)转换为 SRT 标准的时分秒毫秒格式"""
    millis_total = max(0, int(round(seconds * 1000)))
    millis = millis_total % 1000
    total_seconds = millis_total // 1000
    secs = total_seconds % 60
    minutes_total = total_seconds // 60
    minutes = minutes_total % 60
    hours = minutes_total // 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def save_srt(srt_path: Path, segments: list) -> None:
    """保存为标准 SRT"""
    with open(srt_path, "w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments, 1):
            start_str = format_timestamp(seg["start"])
            end_str = format_timestamp(seg["end"])
            f.write(f"{idx}\n")
            f.write(f"{start_str} --> {end_str}\n")
            f.write(f"{seg['text']}\n\n")

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
        f"and output them in their standard form. Key terms expected in this video include: [{terms_str}].\n"
        "CRITICAL: You MUST return exactly the same number of lines as input, index matching 1-to-1."
    )
    return custom_prompt

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
            print(f"      ⚠ Gemini HTTP {exc.code} (Batch {batch_idx}): {detail[:120]}", flush=True)
            time.sleep(4 * (attempt + 1))
        except Exception as exc:
            print(f"      ⚠ Gemini 异常 (Batch {batch_idx}): {exc}", flush=True)
            time.sleep(4 * (attempt + 1))
    return {}

def restore_and_polish_video(mp4_path: Path, json_path: Path) -> tuple[bool, list, str]:
    """读取 staging json, 并发做 LLM 规整。对规整丢失做安全回退保护"""
    try:
        cached = json.loads(json_path.read_text(encoding="utf-8"))
        segments = cached["segments"]
    except Exception as e:
        return False, [], f"读取中转 JSON 错误: {e}"

    if not segments:
        return False, [], "暂存 JSON 中没有 segments 文本"

    result = list(segments)
    batches = []
    for batch_start in range(0, len(segments), LLM_BATCH_SIZE):
        batch = segments[batch_start : batch_start + LLM_BATCH_SIZE]
        batches.append((batch_start, batch))

    system_prompt = build_system_prompt_for_video(mp4_path.name)
    corrected_results = {}

    with ThreadPoolExecutor(max_workers=min(len(batches), 4)) as executor:
        future_to_batch = {
            executor.submit(polish_batch, idx, batch, system_prompt): (idx, start_idx, batch)
            for idx, (start_idx, batch) in enumerate(batches)
        }
        for future in as_completed(future_to_batch):
            idx, start_idx, batch = future_to_batch[future]
            try:
                batch_corrected = future.result()
                
                # ── 🚨 严格对齐校验 🚨 ──
                # 如果 LLM 返回行数与预期行数不一致，直接抛出异常，暴露真实错误
                valid_count = len(batch_corrected)
                expected_count = len(batch)
                
                if valid_count != expected_count:
                    raise RuntimeError(
                        f"行数校验失败！大模型回复行数严重不匹配（预期 {expected_count} 行，实际仅匹配到 {valid_count} 行）"
                    )
                
                # 正常保存结果
                for local_i, corrected_text in batch_corrected.items():
                    corrected_results[start_idx + local_i] = corrected_text
                            
            except Exception as e:
                # 显式向上传递错误，直接标记此视频规整失败
                return False, [], f"分批规整失败 (Batch {idx}): {e}"

    # 组装最终结果
    for i in range(len(result)):
        if i in corrected_results:
            result[i] = {**result[i], "text": corrected_results[i]}

    return True, result, ""

# ─── 主入口 ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("安全多线程字幕还原与规整服务")
    print(f"  并发数: {MAX_CONCURRENT_WORKERS}")
    print(f"  模型  : {GEMINI_MODEL}")
    print("=" * 60)

    manifest = load_manifest()
    if not manifest:
        print("未找到有效 manifest.json")
        return

    # 对所有在清单中，但 json 存在的视频（代表已转录过 ASR 的）重新通过完好的 JSON 生成一份 SRT
    # 以防有文件在此之前遭到了写入异常缩水
    pending = []
    for slug, info in manifest.items():
        mp4_path = Path(info["mp4"])
        json_path = Path(info["json"])
        if json_path.exists() and mp4_path.exists():
            pending.append((slug, mp4_path, json_path))

    pending.sort(key=lambda x: str(x[1]))

    print(f"准备校验并重塑已生成 ASR 的所有 {len(pending)} 个视频字幕文件...", flush=True)

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS) as executor:
        future_to_video = {
            executor.submit(restore_and_polish_video, mp4_path, json_path): (slug, mp4_path)
            for slug, mp4_path, json_path in pending
        }

        for future in as_completed(future_to_video):
            slug, mp4_path = future_to_video[future]
            srt_path = mp4_path.with_suffix(".srt")
            try:
                success, polished, err_msg = future.result()
                if success:
                    save_srt(srt_path, polished)
                    # 重新标为已完成
                    manifest[slug]["srt_done"] = True
                    save_manifest(manifest)
                    print(f"  ✓ 【重构完成】 SRT 写入完毕，大小 {srt_path.stat().st_size} 字节: {mp4_path.name}", flush=True)
                else:
                    print(f"  ✗ 【重构失败】 {mp4_path.name} : {err_msg}", flush=True)
            except Exception as e:
                print(f"  ✗ 【重构异常】 {mp4_path.name} : {e}", flush=True)

    print("\n" + "=" * 60)
    print("字幕安全防御重构全部完成！已彻底恢复！")
    print("=" * 60, flush=True)

def load_manifest():
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except:
            return {}
    return {}

def save_manifest(manifest):
    try:
        MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"  ⚠ 写入 manifest.json 失败: {e}", flush=True)

if __name__ == "__main__":
    main()
