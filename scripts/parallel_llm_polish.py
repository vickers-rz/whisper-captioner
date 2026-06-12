#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量 LLM 规整与 SRT 生成脚本（并发版）
使用已跑完的 ASR JSON 暂存数据，并发调用 Gemini 2.5 Flash 进行规整并生成 SRT。
"""

import os
import json
import time
import re
import urllib.request
import urllib.error
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from google import genai
    from google.genai import types as genai_types
    from pydantic import BaseModel
    HAS_GOOGLE_GENAI = True

    class RefinedChapter(BaseModel):
        text: str
except ImportError:
    genai = None
    genai_types = None
    BaseModel = object
    HAS_GOOGLE_GENAI = False

# ─── 配置与全局变量 ───────────────────────────────────────────────────────────

OSRT_ROOT = Path.home() / "Movies" / "OSRT"
MANIFEST_PATH = OSRT_ROOT / "manifest.json"

FALLBACK_THRESHOLD = 0.80

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
GEMINI_MODEL = "gemini-2.5-flash"

# 每个视频 ASR segments 切片发送给 Gemini 的 batch 大小
LLM_BATCH_SIZE = 120
# 最大并发请求数。同一个 API Key 建议在 5-10 之间，以防触及每分钟请求数 (RPM) 限制。
MAX_CONCURRENT_WORKERS = 8


def require_gemini_api_key() -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("缺少环境变量 GEMINI_API_KEY")
    return GEMINI_API_KEY

# LLM 提示词 (参考 whisper_captioner 原版提示词进行规整，不带 timestamps 干扰)
LLM_SYSTEM_PROMPT = (
    "你是中文视频字幕的严格校订编辑。你的任务是对字幕进行忠实去噪和规整。\n\n"
    "这是“忠实校订”，不是摘要、改写、提纲或出版式压缩。\n\n"
    "只允许：\n"
    "1. 删除“嗯、啊、呃”等没有语义的语气词。\n"
    "2. 删除结巴造成的连续重复，如“这个这个这个”。\n"
    "3. 删除完全相同且相邻的重复句。\n"
    "4. 修正明显病句、标点、错别字和技术术语。\n\n"
    "不得删除：\n"
    "1. 包含任何新增信息的近义重复。\n"
    "2. 教学或讲解中的强调、解释、因果关系和过渡。\n"
    "3. 例子、数字、代码标识、条件、否定表达和操作步骤。\n"
    "4. 任何不能确定是冗余的内容。\n\n"
    "不得总结、扩写、改变观点、重排论证或添加原文没有的知识。\n"
    "输出长度原则上不得低于输入正文的 80%。不确定是否应删除时，保留原文。\n"
    "输出格式必须为 \"序号: 规整后的文本\"，每行一个，不能包含多余的解释。如果不需要修改，直接输出原句。"
)

_LLM_LINE_RE = re.compile(r"^(\d+):\s*(.+)$")

# ─── 辅助函数 ─────────────────────────────────────────────────────────────────

def load_manifest():
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_manifest(manifest):
    try:
        OSRT_ROOT.mkdir(parents=True, exist_ok=True)
        tmp = MANIFEST_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(MANIFEST_PATH)
    except Exception as e:
        print(f"  ⚠ 写入 manifest 失败: {e}", flush=True)

def parse_llm_lines(reply: str, expected_count: int) -> dict:
    """解析 LLM 返回的 "序号: 文本" 格式内容"""
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
    """将 segments 保存为标准 SRT 文件"""
    with open(srt_path, "w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments, 1):
            start_str = format_timestamp(seg["start"])
            end_str = format_timestamp(seg["end"])
            f.write(f"{idx}\n")
            f.write(f"{start_str} --> {end_str}\n")
            f.write(f"{seg['text']}\n\n")

def polish_batch(batch_idx: int, batch: list, total_segments: int, system_prompt: str) -> dict:
    """对单个 batch 的字幕切片调用 Gemini API，进行局部重试"""
    lines = [f"{i + 1}: {s['text']}" for i, s in enumerate(batch)]
    user_text = "\n".join(lines)
    source_chars = len(user_text)

    if HAS_GOOGLE_GENAI:
        try:
            client = genai.Client(api_key=require_gemini_api_key())
            is_pro = "pro" in GEMINI_MODEL.lower()
            thinking_config = genai_types.ThinkingConfig(thinking_budget=1024) if is_pro else None
            
            prior_feedback = ""
            for attempt in range(1, 4):
                prompt = f"{prior_feedback}\n\n待校订正文：\n{user_text}"
                try:
                    response = client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=prompt,
                        config=genai_types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=0.1,
                            max_output_tokens=8192,
                            thinking_config=thinking_config,
                            response_mime_type="application/json",
                            response_schema=RefinedChapter,
                        ),
                    )
                    output_text = response.parsed.text.strip() if response.parsed else ""
                    if not output_text and response.text:
                        output_text = response.text.strip()
                        
                    ratio = len(output_text) / max(1, source_chars)
                    if ratio >= FALLBACK_THRESHOLD:
                        parsed = parse_llm_lines(output_text, len(batch))
                        if len(parsed) >= len(batch) * FALLBACK_THRESHOLD:
                            return parsed
                    
                    prior_feedback = (
                        f"上一版删减过度（保留率仅 {ratio:.2f}）。"
                        "请严格按照一对一原则恢复所有被遗漏的句子和短语。"
                        f"本次输出不得少于输入正文的 {FALLBACK_THRESHOLD*100:.0f}%，不能通过概括来缩短内容。"
                    )
                    print(f"      ⚠ Gemini Native SDK 保留率过低 {ratio:.2f} (Batch {batch_idx}, 尝试 {attempt}): 退避重试", flush=True)
                    time.sleep(attempt * 2)
                except Exception as exc:
                    print(f"      ⚠ Gemini Native API 异常 (Batch {batch_idx}, 尝试 {attempt}): {exc}", flush=True)
                    time.sleep(attempt * 2)
            return {}
        except Exception as e:
            print(f"      ⚠ 无法初始化 Gemini Native Client，降级到 urllib: {e}", flush=True)

    # Fallback to original urllib implementation
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
        "Authorization": f"Bearer {require_gemini_api_key()}",
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
            parsed = parse_llm_lines(reply, len(batch))
            if len(parsed) < len(batch) * FALLBACK_THRESHOLD:
                raise ValueError(f"返回行数({len(parsed)})低于阈值(需>={int(len(batch)*FALLBACK_THRESHOLD)})，疑似总结行为")
            return parsed
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace").strip()
            print(f"      ⚠ Gemini API HTTP {exc.code} (Batch {batch_idx}, 尝试 {attempt+1}): {detail[:150]}", flush=True)
            time.sleep(5 * (attempt + 1))
        except Exception as exc:
            print(f"      ⚠ Gemini API 异常 (Batch {batch_idx}, 尝试 {attempt+1}): {exc}", flush=True)
            time.sleep(5 * (attempt + 1))
            
    # 重试全部失败后，返回空表示不作修改
    return {}

def build_system_prompt_for_video(mp4_name: str) -> str:
    """根据视频文件名动态识别可能出现的技术名词，组装定制化的提示词"""
    terms = ["FastAPI", "Python", "LLM", "API", "RAG", "Prompt", "Agent", "JSON", "Docker", "GPU", "CUDA"]
    
    # 提取视频标题中的术语并补充
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
        
    # 去重
    terms = list(dict.fromkeys(terms))
    terms_str = ", ".join(terms)
    
    custom_prompt = (
        "你是中文视频字幕的严格校订编辑。你的任务是对字幕进行忠实去噪和规整。\n\n"
        "这是“忠实校订”，不是摘要、改写、提纲或出版式压缩。\n\n"
        "只允许：\n"
        "1. 删除“嗯、啊、呃”等没有语义的语气词。\n"
        "2. 删除结巴造成的连续重复，如“这个这个这个”。\n"
        "3. 删除完全相同且相邻的重复句。\n"
        "4. 修正明显病句、标点、错别字和技术术语。\n\n"
        "不得删除：\n"
        "1. 包含任何新增信息的近义重复。\n"
        "2. 教学或讲解中的强调、解释、因果关系和过渡。\n"
        "3. 例子、数字、代码标识、条件、否定表达和操作步骤。\n"
        "4. 任何不能确定是冗余的内容。\n\n"
        "不得总结、扩写、改变观点、重排论证或添加原文没有的知识。\n"
        "输出长度原则上不得低于输入正文的 80%。不确定是否应删除时，保留原文。\n"
        "输出格式必须为 \"序号: 规整后的文本\"，每行一个，不能包含多余的解释。如果不需要修改，直接输出原句。\n\n"
        "Important: This video is about AI and software development. Correct phonetically misrecognized terms "
        f"and output them in their standard form. Key terms expected in this video include: [{terms_str}]."
    )
    return custom_prompt

def polish_single_video(mp4_path: Path, json_path: Path) -> tuple[bool, list, str]:
    """对单个视频的所有 ASR 文本执行分批并发 LLM 规整"""
    system_prompt = build_system_prompt_for_video(mp4_path.name)
    try:
        cached = json.loads(json_path.read_text(encoding="utf-8"))
        segments = cached["segments"]
    except Exception as e:
        return False, [], f"读取 JSON 失败: {e}"

    if not segments:
        return False, [], "ASR 结果为空"

    result = list(segments)
    batches = []
    for batch_start in range(0, len(segments), LLM_BATCH_SIZE):
        batch = segments[batch_start : batch_start + LLM_BATCH_SIZE]
        batches.append((batch_start, batch))

    # 并发处理各个段落的 LLM 规整
    corrected_results = {}
    with ThreadPoolExecutor(max_workers=min(len(batches), 4)) as executor:
        future_to_batch = {
            executor.submit(polish_batch, idx, batch, len(segments), system_prompt): (idx, start_idx, batch)
            for idx, (start_idx, batch) in enumerate(batches)
        }
        for future in as_completed(future_to_batch):
            idx, start_idx, batch = future_to_batch[future]
            try:
                batch_corrected = future.result()
                for local_i, corrected_text in batch_corrected.items():
                    corrected_results[start_idx + local_i] = corrected_text
            except Exception as e:
                print(f"    ⚠ Batch {idx} 并发执行抛出异常: {e}", flush=True)

    # 用规整后的文本更新 segments
    for i in range(len(result)):
        if i in corrected_results:
            result[i] = {**result[i], "text": corrected_results[i]}

    return True, result, ""

# ─── 主入口 ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("并行 LLM 实时规整与 SRT 生成服务 (常驻监听模式)")
    print(f"  并发数: {MAX_CONCURRENT_WORKERS}")
    print(f"  模型  : {GEMINI_MODEL}")
    print(f"  监听路径: {MANIFEST_PATH}")
    print("=" * 60)

    # 循环监控，每隔 5 秒检查一次是否有新增 ASR JSON 完成
    active_workers = {}
    
    print("开始实时监控并规整字幕... 按 Ctrl+C 退出。\n", flush=True)

    try:
        while True:
            manifest = load_manifest()
            if not manifest:
                time.sleep(5)
                continue

            # 筛选已完成 ASR（即 json 存在），但是未标记 polished_done 的项目
            pending = []
            for slug, info in manifest.items():
                if info.get("polished_done"):
                    continue
                if not info.get("asr_done"):
                    continue
                json_path = Path(info["json"])
                if json_path.exists():
                    pending.append((slug, info))

            # 按原视频路径名字母排序（按章节顺序）
            pending = sorted(pending, key=lambda x: x[1].get("mp4", ""))

            if pending:
                print(f"⏳ 检测到 {len(pending)} 个新 ASR 视频已就绪，启动高并发规整...", flush=True)
                
                with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS) as executor:
                    future_to_video = {}
                    for slug, info in pending:
                        mp4_path = Path(info["mp4"])
                        json_path = Path(info["json"])
                        future = executor.submit(polish_single_video, mp4_path, json_path)
                        future_to_video[future] = (slug, mp4_path)

                    for future in as_completed(future_to_video):
                        slug, mp4_path = future_to_video[future]
                        srt_path = mp4_path.with_suffix(".polished.srt")
                        try:
                            success, polished, err_msg = future.result()
                            if success:
                                save_srt(srt_path, polished)
                                # 立即写 manifest 更新状态
                                # 重新载入以防止并发覆写其他项
                                cur_manifest = load_manifest()
                                if slug in cur_manifest:
                                    cur_manifest[slug]["polished_done"] = True
                                    save_manifest(cur_manifest)
                                print(f"  ✓ 【规整完成】 SRT 生成成功: {mp4_path.name}", flush=True)
                            else:
                                print(f"  ✗ 【规整失败】 {mp4_path.name}: {err_msg}", flush=True)
                        except Exception as exc:
                            print(f"  ✗ 【处理异常】 {mp4_path.name}: {exc}", flush=True)
                
                print("✨ 批次规整结束，继续监听...\n", flush=True)

            time.sleep(6)
            
    except KeyboardInterrupt:
        print("\n已收到退出指令，实时监控服务停止。", flush=True)

if __name__ == "__main__":
    main()
