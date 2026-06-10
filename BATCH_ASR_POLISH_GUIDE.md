# 批量 ASR 与高并发实时 LLM 规整字幕工具

本项目包含两个专门为大批量视频设计的字幕转录与规整工具。通过主副程序解耦，实现 **ASR 识别** 与 **多线程并发大模型规整** 的实时流水线式协作。

## 🛠️ 工具组成

### 1. 主程序：`batch_transcribe_02_03.py`
- **位置**：`batch_transcribe_02_03.py`
- **职责**：
  - 扫描指定章节目录下的所有 MP4 文件，并以严格字母序（章节顺序）推进。
  - 提取 16k 单声道 WAV，自动将大音频切割为 `30秒分块 + 2秒重叠` 的安全切片，避开局域网 NUC ASR（Qwen3-ASR 1.7B 服务）的 30MB 载荷上限。
  - **只进行 Phase 1 ASR 转录**。转录完成后，将半成品段落暂存至本地磁盘 `~/Documents/temp/batch_asr_staging/` 作为 JSON 缓存，并更新清单，不作任何 LLM 处理。

### 2. 副服务（实时监听守护进程）：`parallel_llm_polish.py`
- **位置**：`scripts/parallel_llm_polish.py`
- **职责**：
  - 常驻后台，以每 6 秒为周期实时轮询监听 `manifest.json` 清单。
  - 只要发现有任何视频刚完成了 ASR（JSON 暂存已就绪，但尚未标记 `srt_done`），便**立即并发**调起 Google Gemini 2.5 Flash API 进行文本纠错和断句。
  - **动态术语字典**：根据当前处理的视频文件名，动态装配对应的技术名词表（例如 RAG, LangChain, Milvus 等），指导大模型准确校正谐音错别字。
  - 支持最多 **8 个视频并发 + 视频内分批并发**，并在检测到 API 限频时进行自动退避和重试。
  - 实时生成标准的 `.srt` 字幕文件并存放到**原视频所在的同级文件夹**下。

---

## 🚀 启动与使用指南

### 1. 运行主 ASR 进程
在终端中执行以开始提取并转录音频：
```bash
cd /Users/vickers/Documents/whisper-captioner
python3 batch_transcribe_02_03.py >> ~/Documents/temp/batch_asr_staging/run.log 2>&1 &
```
*(进度日志会输出在 `~/Documents/temp/batch_asr_staging/run.log` 中。)*

### 2. 运行实时并发规整服务（副程序）
开启另一个终端窗口，或者在后台常驻运行副程序以实时处理生成的 JSON：
```bash
python3 /Users/vickers/Documents/whisper-captioner/scripts/parallel_llm_polish.py
```
副服务启动后会输出类似如下日志并持续监听：
```text
============================================================
并行 LLM 实时规整与 SRT 生成服务 (常驻监听模式)
  并发数: 8
  模型  : gemini-2.5-flash
  监听路径: /Users/vickers/Documents/temp/batch_asr_staging/manifest.json
============================================================
开始实时监控并规整字幕... 按 Ctrl+C 退出。

⏳ 检测到 3 个新 ASR 视频已就绪，启动高并发规整...
  ✓ 【规整完成】 SRT 生成成功: 01 uv的介绍与安装.mp4
  ✓ 【规整完成】 SRT 生成成功: 02 MCP的介绍.mp4
...
```

---

## 📂 路径规范
- **临时及清单目录**：`~/Documents/temp/batch_asr_staging/`
- **输出字幕路径**：与各 `.mp4` 视频文件保持**完全同名并处于同目录下**。
