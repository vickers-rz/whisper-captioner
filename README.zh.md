# Whisper Captioner 中文版

本项目是一个本地 macOS 字幕助手，用于 Loopback 音频实时字幕，以及对网页视频进行可控播放和字幕处理。

> 说明：本文是 [README.md](/Users/vickers/Documents/whisper-captioner/README.md) 的中文版本。命令、路径、环境变量、端口、模型名和文件名尽量保持原样，方便和英文 README 对照。

## 运行

```bash
conda run -n whishperapp_pyside6 python /Users/vickers/Documents/whisper-captioner/whisper_captioner/app.py
```

或者：

```bash
bash /Users/vickers/Documents/whisper-captioner/run.sh
```

## 取证转写 TUI

双击 [ForensicSubtitle.command](/Users/vickers/Documents/whisper-captioner/ForensicSubtitle.command)，或在终端中运行它，即可打开可恢复的取证字幕菜单：

```bash
/Users/vickers/Documents/whisper-captioner/ForensicSubtitle.command
```

菜单前两项独立于完整取证流水线：

- `Gemini URL -> full transcript`：用 `yt-dlp` 下载公开 YouTube URL 的最佳音频，优先使用 WebM，把首个音频流转为 OGG/Opus，然后通过 Gemini File API 做仅音频 ASR。它不会请求摘要、时间戳或视觉分析结果。如需旧的“直接把 URL 交给 Gemini”路径，可调用 `scripts/asr_entrypoints.py gemini-url --direct-url`。
  默认 OGG 保存为 `/Volumes/T7_APFS/MacBackup/Movies/WhisperCaptioner/artifacts/generated/Gemini-URL-ASR [VIDEO_ID]/work/gemini-audio.ogg`，排版后的 Markdown ASR 文稿保存为 `/Volumes/T7_APFS/MacBackup/Movies/WhisperCaptioner/artifacts/generated/Gemini-URL-ASR [VIDEO_ID]/gemini-local-audio-asr-transcript.md`；如果设置了 `WHISPER_CAPTIONER_OUTPUT_DIR`，则位于该目录下的 `artifacts/generated/` 中。
  Markdown 文稿使用内置轻量中文排版：metadata 标题区、正文区、中英文/数字间距、常见中文标点清理，以及按句子自动分段。
  TUI 还提供 `ASR Markdown 文稿 -> jieba/Ollama embedding RAG -> 本机 Qwen 全文语义分段`。它现在使用面向长视频字幕的四层流水线：中文断句、Paragraph/Topic 语义窗口、TF-IDF 加 `qwen3-embedding:0.6b` 双索引与 RRF 融合，最后交给本机 `qwen3.5:4b` 做结构分析。本地索引包含 Sentence、Paragraph、Topic 三层 embedding，输出 `*-ollama-segmented.md`。CLI 可用 `--embedding-backend ollama`、`tfidf` 或 `hybrid` 对比不同方案。
  Command TUI 已把四种分段方案拆成四个独立菜单项：纯 Qwen、TF-IDF RAG、Ollama embedding RAG 和 Hybrid RAG。
  另外还新增第 5 种 ASR 后处理模式：Gemini 2.5 Flash/Pro 富 Markdown 规整，适合已配置 Gemini API Key 时使用。
- `Local OGG/audio -> NUC ASR`：把本地文件标准化为 `16 kHz / mono / WAV`，并提供 Qwen3-ASR 1.7B（优先文本准确性，显式伪时间戳）、faster-whisper large-v3（词级时间戳）或两者顺序执行。

同样的入口也可以从终端直接调用：

```bash
/Users/vickers/Documents/whisper-captioner/ForensicSubtitle.command gemini-url \
  'https://www.youtube.com/watch?v=VIDEO_ID'

/Users/vickers/Documents/whisper-captioner/ForensicSubtitle.command nuc-local \
  '/path/to/audio.ogg'
```

程序在读取 Chrome cookies 前会先询问；Gemini key 只保存在环境变量或 macOS Keychain 中；每个任务旁边都会写入 `pipeline-manifest.json`。OCR 只从短视频片段采样，用来裁决有争议文本；NUC 词级时间戳仍是唯一的时间轴权威。当前工作流见 `docs/final_forensic_subtitle_pipeline.md`。

启用 Chrome cookies 时，命令只会把所选配置文件的主 `Cookies` 数据库快照到私有临时目录，避免 yt-dlp 把较新的嵌套 Chrome/Glic 扩展数据库误认为主 cookie 库。流水线进程退出时会删除该快照。默认配置文件是 `Default`；可设置 `FORENSIC_CHROME_PROFILE`，也可在 TUI 提示中选择其它配置文件。

## 确定性取证命令

完整的双击 TUI 入口：

```text
/Users/vickers/Documents/whisper-captioner/ForensicSubtitle.command
```

该菜单可以运行或恢复 `docs/final_forensic_subtitle_pipeline.md` 中描述的完整流水线，显示最近任务状态，打开产物目录，查看上次运行日志尾部，并执行环境诊断。Gemini key 从 `GEMINI_API_KEY` 或现有 macOS Keychain 项读取；新输入的 key 不会写入日志或 manifest。

当前最终工作流也提供非交互命令。先在不下载完整视频的情况下探测硬字幕：

```bash
python /Users/vickers/Documents/whisper-captioner/scripts/forensic_subtitle_command.py \
  probe-hard-subs 'https://www.youtube.com/watch?v=VIDEO_ID' \
  --output-dir artifacts/probe \
  --cookies-from-chrome
```

生成 NUC 词级 ASR JSON 和 Gemini OGG 转写后，可以创建保留初始本地 SRT 时间戳的 Gemini 文本回填、按时间窗口分组的差异报告，以及确定性的最终 SRT：

```bash
python /Users/vickers/Documents/whisper-captioner/scripts/forensic_subtitle_command.py \
  finalize \
  --nuc-asr /path/to/nuc-word-asr.json \
  --gemini /path/to/gemini-transcript.txt \
  --output-dir artifacts/final
```

`finalize` 不会修改 Gemini 文本；它只根据标点、字幕长度、阅读时长和 NUC 词级时间选择展示断句。输出包括 `gemini-backfilled-local-timeline.srt`、`transcript-differences.json`、`final-timeline.json` 和 `final.srt`。

如果要用 shell 编排 Python 阶段，可使用 `scripts/forensic_subtitle_pipeline.sh`。它暴露 `probe`、`finalize` 和 `targeted-ocr`；不带参数运行可查看完整参数列表。`targeted-ocr` 只采样 `transcript-differences.json` 指定的本地视频窗口，并保存 OCR 证据，不改变 NUC 时间轴。

## 当前架构

项目已经从最初的单文件原型拆分出来：

- `whisper_captioner/app.py`：Qt 主窗口、托盘菜单、高层编排和信号连接。
- `whisper_captioner/config.py`：路径、模型名、工具路径、流水线版本。
- `whisper_captioner/models.py`：共享 dataclass，以及字幕/LLM 模式配置。
- `whisper_captioner/subtitle_io.py`：SRT/VTT 解析、JSON 分段缓存、SRT/TXT 导出。
- `whisper_captioner/cache.py`：规范化媒体 URL、面向 Bilibili 分 P 的缓存键，以及 URL 校验。
- `whisper_captioner/chrome_control.py`：Chrome AppleScript 视频控制辅助函数。轮询函数读取视频时间时不会激活 Chrome；播放控制函数可按需激活目标标签页。
- `whisper_captioner/overlay.py`：悬浮字幕层、固定按钮、拖拽/缩放、字体、透明度和播放控制。
- `whisper_captioner/workers.py`：实时 Loopback 采集、NUC 实时 worker、实时会话润色/重识别 worker、受控 URL 字幕处理、本地队列 worker，以及基于外部 blocklist 的幻觉短语过滤。
- `whisper_captioner/llm_handler.py`：Native Ollama API、Gemini、OpenAI-compatible、Anthropic 和 Rapid-MLX 字幕校对调用。
- `whisper_captioner/mlx_terms.py`：本地 Rapid-MLX/MLX 术语抽取辅助工具，目前不属于主 Gemini 全文档流水线。
- `whisper_captioner/qwen_chat_service.py`：本地字幕后处理 Web 工作区，支持 Qwen3-8B / Gemini 聊天、手动字幕上传，以及二阶段清理或改写文章。

架构注意事项：

- `app.py`、`workers.py` 和 `qwen_chat_service.py` 是主要增长热点。
- 新转写后端逻辑应尽量进入共享 transcriber service，而不是在队列 worker 和受控播放 worker 之间重复实现。
- 新的 NUC 生命周期或优先级决策应放在 scheduler 层，代理保持为薄的 job/admission client。
- 新字幕后处理工作区功能应避免继续膨胀单文件 Web service；应先拆分存储、资源索引、prompt 和 HTTP 路由。
- 当前重构地图见 `ARCHITECTURE_AUDIT_v2026-05-07.md`，其中也包含 2026-05-14 架构审查更新。

运行时目录：

- `/Volumes/T7_APFS/MacBackup/Movies/WhisperCaptioner/artifacts/generated/`：按来源标题分组的字幕、转写和共享 Markdown 输出。
- `/Volumes/T7_APFS/MacBackup/Movies/WhisperCaptioner/artifacts/logs/`：应用日志。
- `/Volumes/T7_APFS/MacBackup/Movies/WhisperCaptioner/artifacts/notes/`：不绑定到生成字幕目录的独立笔记导出。
- `/Volumes/T7_APFS/MacBackup/Movies/WhisperCaptioner/cache/`：按来源划分的处理缓存和最终 segment JSON。
- `/Volumes/T7_APFS/MacBackup/Movies/WhisperCaptioner/cache/local-audio/`：本地媒体文件提取出的 `16 kHz / mono / wav` 缓存。相同源文件会在重试间复用，直到手动清理或源文件变化。
- `/Volumes/T7_APFS/MacBackup/Movies/WhisperCaptioner/qwen-chat/`：Web 工作区上传、存储和操作导出。
- `/Volumes/T7_APFS/MacBackup/Movies/WhisperCaptioner/realtime/`：持久化的实时会话音频和审阅 manifest。
- `/Volumes/T7_APFS/MacBackup/Movies/whisper-captioner_APP_Resource/whisper-models/`：本地 Whisper 模型二进制文件。
- `/Volumes/T7_APFS/MacBackup/Movies/whisper-captioner_APP_Resource/third_party/`：可选后端使用的本地第三方源码 checkout 和构建产物。
- `/Volumes/T7_APFS/MacBackup/Movies/whisper-captioner_APP_Resource/huggingface-cache/`：`huggingface_hub`、`mlx-audio` 等使用的 Hugging Face / MLX 模型缓存。
- `/Volumes/T7_APFS/MacBackup/Movies/whisper-captioner_APP_Resource/local-models/SenseVoice.cpp/`：默认 SenseVoice.cpp FP16 运行时路径。

路径覆盖：

- `WHISPER_CAPTIONER_OUTPUT_DIR`：运行时输出根目录，默认 `/Volumes/T7_APFS/MacBackup/Movies/WhisperCaptioner`。
- `WHISPER_CAPTIONER_RESOURCE_DIR`：模型/资源根目录，默认 `/Volumes/T7_APFS/MacBackup/Movies/whisper-captioner_APP_Resource`。
- `WHISPER_CAPTIONER_LOCAL_MODELS_DIR`：本地模型根目录，默认 `/Volumes/T7_APFS/MacBackup/Movies/whisper-captioner_APP_Resource/local-models`。
- `WHISPER_CAPTIONER_SENSEVOICE_DIR`：SenseVoice.cpp 运行时根目录，默认 `/Volumes/T7_APFS/MacBackup/Movies/whisper-captioner_APP_Resource/local-models/SenseVoice.cpp`。

准备本地 SenseVoice.cpp 运行时：

```bash
bash scripts/migrate_local_sensevoice_runtime.sh
```

## 当前稳定性说明

近期维护主要集中在受控 URL 路径和小屏可用性：

- 主窗口默认尺寸更小，并使用可滚动的中央布局，因此在 24 英寸 1080p 副屏上也能访问完整 GUI。
- 手动提供的 `zh.*` 字幕被发现时，受控 URL 播放不会退出应用。应用会直接加载这些字幕分段、刷新转写列表，并在不运行 Whisper 或 LLM 的情况下开始受控播放。
- Qwen3-ASR 伪时间戳在本地队列处理和受控 URL 处理间共享，避免选择 Qwen3-ASR 后端时受控模式崩溃。
- 受控字幕查找使用缓存的当前索引，以及用于 `bisect` fallback 的字幕开始索引缓存，避免长视频每 250 ms 全量扫描字幕。
- 受控 SenseVoice.cpp 分块现在会调用正确的 `RollingPrefetchWorker` 命令辅助函数。

## 模式

- `NUC faster-whisper large-v3-turbo（远程 CUDA，快速）`：最快的远程批处理选项。
- `NUC faster-whisper large-v3（远程 CUDA，高质量）`：较慢的远程选项，保留更完整的 large-v3 转写。
- `NUC Qwen3-ASR 1.7B（远程高质量离线）`：远程高质量离线模式，适合长音频，在 NUC 上排队，和实时 ASR 分离。
- `实时字幕 NUC large-v3（远程 CUDA，3s延迟）`：低延迟实时模式，转移到 NUC 执行，并完整持久化会话。
- `实时字幕 whisper.cpp small（SoundSource/Loopback）`：最低延迟的本地实时模式，用于经 Loopback 路由的 Chrome 或本地播放器音频。
- `实时字幕 whisper.cpp q5_0（large-v3-turbo）`：可接受稍高延迟时的更高质量本地实时模式。
- `Qwen3-ASR 0.6B 4bit（默认）`：推荐的 MLX-Audio 工作流，优先转写文本并做轻量规范化。
- `Qwen3-ASR 1.7B 8bit（高质量）`：需要更好转写润色时的高质量 MLX-Audio 工作流。
- `MLX-Audio 5bit（whisper-large-v3-turbo-asr-5bit）`：实验性 MLX-Audio Whisper 后端。
- `SenseVoice-Small-mlx`：基于 `mlx-community/SenseVoiceSmall` 的实验性 MLX-Audio 模式。
- `SenseVoice.cpp FP16`：由 SenseVoice.cpp 驱动的本地 GGUF/Metal 后端。速度快、流畅，但长文件的分块边界处理仍在调优。
- `MLX Whisper FP16（whisper-large-v3-turbo）`：使用 `mlx-whisper` 的更高精度 MLX fallback。
- `whisper.cpp q5_0（large-v3-turbo）`：当前默认受控 URL 后端，也是最强的字幕风格基线。
- `whisper.cpp small`：最快的本地 whisper.cpp 批处理模式。
- `whisper.cpp 高精度 q5_0（large-v3）`：更慢、更高准确率的 whisper.cpp 批处理模式。

## 受控播放流水线

对于可由 `yt-dlp` 处理的网页视频，使用 `Controlled URL captions`。

当前流程：

1. 规范化媒体 URL。例如 Bilibili URL 会变为 `https://www.bilibili.com/video/<BV...>`，多 P 视频保留 `?p=`。
2. 检查手动提供的 `zh.*` 内置字幕。如果存在，直接加载字幕，跳过本地 Whisper 和 LLM，并开始受控播放。自动字幕不被视为安全的提前退出来源。
3. 暂停 Chrome，并查找与当前流水线签名匹配的最终字幕缓存。
4. 如果没有有效最终缓存，则用 `yt-dlp` 下载音频。
5. 用 `ffmpeg` 把音频切成 30 秒分块。
6. 使用当前选择的后端转写每个分块，并把分块内时间戳平移到完整视频时间轴。
7. 在启用且所选 provider 就绪时运行全文档 LLM 校对。默认配置 provider 是 Gemini 2.5 Flash，但流水线使用 UI 中选择的 provider。
8. 保存 `final-subtitles-current.json`，并导出 `.srt` 和 `.txt`。
9. 从 0 秒启动 Chrome，并通过轮询受控视频的 `currentTime` 渲染字幕，避免反复抢走用户焦点。

缓存身份由规范化媒体 URL、Whisper 模型、分块时长、LLM provider/model 和 `SUBTITLE_PIPELINE_VERSION` 组成。缓存键也包含 Whisper 后端，因此 `mlx-audio`、`mlx-whisper` 和 `whisper.cpp` 输出不会互相覆盖。

已知缓存后续事项：

- `b23.tv` 短链接尚未在生成缓存键前展开。
- 原生字幕缓存仍是普通 segment JSON 文件，还没有自己的 metadata/signature。

后处理输出保存在当前视频缓存旁边：

- `video-summary-analysis.md`：视频摘要、结构、论证分析、关键词和一句话结论。
- `video-article.md`：根据转写改写出的润色长文。

## 字幕后处理工作区

本地 Qwen Web 入口已经扩展为字幕后处理工作区。

它可以：

- 上传第三方 `.srt`、`.vtt` 或 `.txt` 文件，而不运行转写流水线。
- 扫描 `/Volumes/T7_APFS/MacBackup/Movies/WhisperCaptioner/artifacts/generated/` 下之前生成的字幕文件，并在历史侧边栏中展示。
- 把任意上传或历史字幕文件附加到工作对话。
- 基于附加字幕内容与 LLM 聊天。
- 对附加字幕执行一键 `语句规整` 或 `转写成文稿`。
- 在 `NUC Ollama Gemma 4 E4B (16K)`、`NUC Ollama Qwen3-14B`、`Local Rapid-MLX Qwen3-8B`、`Gemini 2.5 Flash` 和 `Gemini 2.5 Pro` 之间切换。

注意：

- 长字幕内容可能超过本地 Qwen3-8B 更舒适的单次处理范围时，会显示明确警告。
- NUC Gemma 4 provider 会显式请求 16K 上下文窗口，关闭 thinking，把输出限制为 8192 token，并在请求间保持模型加载 10 分钟。
- 出现该警告时，如果已在桌面应用 Settings 面板配置 API key，建议切换到 `Gemini 2.5 Pro`。
- 操作导出写入 `/Volumes/T7_APFS/MacBackup/Movies/WhisperCaptioner/qwen-chat/exports/`。

同步控制：

- `Sub -0.5s` / `Sub +0.5s`：调整并持久化当前规范化视频缓存的字幕偏移。
- `Sync line`：把当前显示的字幕行对齐到受控 Chrome 视频的当前时间。

## 幻觉 Blocklist

为了抑制反复出现且明显无关的字幕幻觉，应用会在写入原始缓存或导出字幕前过滤一小组已知坏短语。

- 内置默认值已经覆盖本项目中观察到的几个反复短语，例如 `优优独播剧场——YoYo Television Series Exclusive`。
- 不改 Python 代码也可以扩展过滤器，把短语追加到：
  `/Volumes/T7_APFS/MacBackup/Movies/WhisperCaptioner/hallucination_blocklist.txt`
- 每行添加一个短语。
- 以 `#` 开头的行会被视为注释。

该 blocklist 是有意保守的文本层保护。它不会改变模型解码参数；只会在转写之后、缓存/导出持久化之前删除精确匹配的重复垃圾短语。

## 运行时产物

某些第三方语音工具可能会生成临时 sidecar 文件，例如 `fbank_lfr_cmvn_feature.json`。

- 这些运行时产物不视为源文件。
- 仓库会忽略已知生成产物。
- 部分子进程以 `cwd=/Volumes/T7_APFS/MacBackup/Movies/WhisperCaptioner` 启动，因此这些文件会落到输出区域，而不是污染源码树。

## 产物迁移

生成的字幕和转写输出现在默认写入 `/Volumes/T7_APFS/MacBackup/Movies/WhisperCaptioner/artifacts/generated/`。

- 新的队列导出、受控字幕导出和按视频共享的 Markdown 副本会写到这里。
- Web 工作区直接扫描此位置来发现历史生成字幕资源。
- `hallucination_blocklist.txt`、`cache/`、`qwen-chat/` 和 `realtime/` 仍保留在顶层，因为它们是运行时状态，不是导出交付物。
- 可选本地模型、第三方后端和 Hugging Face 缓存现在位于 `/Volumes/T7_APFS/MacBackup/Movies/whisper-captioner_APP_Resource/`。

## 本地基准说明

在这台 Apple M2 Mac mini 上：

- `large-v3-turbo-q5_0`：基准总耗时约 7.6s。
- `large-v3-q5_0`：基准总耗时约 16.1s。

推荐默认值：

- 默认字幕：`whisper.cpp large-v3-turbo-q5_0`。
- 可接受延迟的非直播视频：`Controlled URL captions`。
- 最高准确率批处理：`whisper.cpp 高精度 q5_0（large-v3）`。

后端基准：

- 30 秒样本中，`whisper.cpp q5_0` 平均约 `4.64s`，`mlx-whisper FP16` 约 `6.42s`，`mlx-audio q5` 约 `7.82s`。
- 70 秒混合技术样本中，`whisper.cpp q5_0` 平均约 `8.18s`，`mlx-whisper FP16` 约 `9.35s`，`mlx-audio q5` 约 `10.25s`。
- 由于 `whisper.cpp q5_0` 在这些本地样本上更快，且分块时间戳更干净，它仍是默认值。
- 在这台 Apple M2 上使用 `whisper-cpp 1.8.6` 时，一个 45 秒真实样本在 6 线程加 Flash Attention 下耗时 `7.35s`，8 线程为 `7.60s`，4 线程为 `8.71s`。
- 关闭 Flash Attention 后耗时 `12.61-13.18s`，且输出从 22 段变为 18 段，因此 APP 默认启用 Flash Attention 并使用 6 线程。

受控播放现在会在开始 ASR 前下载嵌入的中文字幕。手动字幕优先，自动字幕作为 fallback；选中的轨道会在播放前缓存并导出为标准 SRT/TXT。

分析标签页可以基于带时间戳的转写生成 LLM 视频章节。章节包含标题和描述，会缓存为 JSON、导出为 Markdown，并可点击跳转受控 Chrome 视频。

章节按钮是上下文敏感且始终手动触发的。受控 URL 播放期间，它会生成并显示章节浮层。对于本地字幕或已有 SRT，必要时会要求选择 SRT，并写入单独的 `*-带章节.srt` 文件，把章节标题和描述插入对应字幕位置；源 SRT 不会被覆盖。

批量 Gemini 脚本只从环境变量读取凭据：

```bash
export GEMINI_API_KEY="your-new-key"
```

桌面 APP 和 Web 字幕工作区也会优先使用 `GEMINI_API_KEY`，再使用 APP 设置中较旧的 key。其它支持的环境变量包括 `OPENAI_API_KEY`、`DEEPSEEK_API_KEY`、`MINIMAX_API_KEY` 和 `ANTHROPIC_API_KEY`。仓库不保存任何 API key。

设置页会展示两个值。环境变量会覆盖已保存设置：

- `WHISPER_CAPTIONER_CPP_THREADS=6`
- `WHISPER_CAPTIONER_CPP_FLASH_ATTN=1`

### 30s 对比

样本：

- `/tmp/sensevoice-test-30s.wav`

本机结果：

- `whisper.cpp large-v3-turbo-q5_0`：`4.22s`
- `SenseVoice.cpp FP16`：`5.37s`
- `mlx-community/Qwen3-ASR-0.6B-4bit`：`6.46s`
- `SenseVoice.cpp q8_0`：`17.43s`
- `mlx-community/Qwen3-ASR-1.7B-8bit`：`9.50s`
- `SenseVoiceSmall via mlx-audio`：`20.43s`

### NUC 3080 Ti 远程推理性能

对于 5.5 秒的中英技术混合短音频块，也就是典型实时分块大小：

- `NUC faster-whisper large-v3 CUDA`：`0.37s`（RTF `0.067x`），最快，比实时快 15 倍，准确率 100%。
- `Mac M2 SenseVoice.cpp FP16`：`0.46s`（RTF `0.083x`），第二快，但技术术语上有同音错误。
- `Mac M2 whisper.cpp turbo-q5_0`：`2.43s`（RTF `0.438x`）。

结论：NUC 后端在速度和上下文感知能力（large-v3）上都明显强于本地 M2 推理。

## 远程 NUC 推理

应用原生集成局域网中的外部 Intel NUC，配备 NVIDIA RTX 3080 Ti，例如 `192.168.31.196`，用于大幅加速推理。

- **NUC ASR**：面向 APP 的 `:8000` 仍是 OpenAI-compatible `/v1/audio/transcriptions` 端点。当前 NUC 布局中，它前面有一个感知 busy 状态的薄代理，真实转写转发到内部 `:18000` 的 `faster-whisper-server` 后端。
- **NUC High-Quality ASR**：可选 `Qwen3-ASR 1.7B` 路径，通过端口 `8001` 的独立代理暴露，面向单并发长音频离线任务，而不是实时分块。
- **NUC LLM**：由 native `Ollama` 提供，端口 `11434` 暴露 `/api/chat`。`qwen3:14b` 等模型在 NUC 上比 Mac 本地 Rapid-MLX 快 6.5 倍。

使用时只需在应用 UI 下拉框中选择 `nuc_asr` 或 `nuc_ollama` provider。应用实现了自动超时和 fallback，保证 NUC 离线时本地 Mac 仍可用。

当前 NUC 端口：

- `:8000`：面向 APP 的实时/默认 ASR 通道；常驻代理，暴露 `/health`、`/busy`、`/v1/models` 以及转写/job 端点。
- `:18000`：`:8000` 代理使用的内部 `faster-whisper-server` 后端。
- `:8001`：面向 APP 的 `Qwen3-ASR 1.7B` 高质量离线代理。
- `:8002`：可调试访问的 `Qwen3-ASR 1.7B` 后端，由官方 `qwen-asr-serve` 提供。
- `:11434`：Ollama LLM API。

### 本地文件远程 ASR 缓存

对于本地媒体文件，应用现在避免每次重试都重复执行 `ffmpeg` 提取：

- Mac 会先把源文件转换一次，写入 `/Volumes/T7_APFS/MacBackup/Movies/WhisperCaptioner/cache/local-audio/<cache-key>/audio-16k-mono.wav`。
- 重试会复用缓存 WAV，而不是再次提取音频。
- UI 提供 `删除本地音频缓存`，可以为当前本地文件手动清理缓存。
- 不同源文件天然拥有不同缓存键，因为缓存键包含解析后的路径、文件大小和 mtime。

当源是本地文件时，这同时适用于 `NUC faster-whisper large-v3（远程 CUDA）` 和 `NUC Qwen3-ASR 1.7B（远程高质量离线）`。

### NUC 运行时部署

使用仓库内同步脚本，不要编辑 `/srv` 下的匿名副本：

```bash
bash /Users/vickers/Documents/whisper-captioner/scripts/sync_nuc_runtime.sh --sync-only
bash /Users/vickers/Documents/whisper-captioner/scripts/sync_nuc_runtime.sh --deploy
```

`--sync-only` 是默认值。它会校验 SHA-256、备份当前 NUC 源码和容器定义，并安装候选脚本，但不重建服务。`--deploy` 还会重建 scheduler、代理和两个 GPU 后端。它不会删除 staging、results 或模型缓存。

部署后的布局包括：

- 一个只在本机访问的 service scheduler sidecar，可按需启动/停止 `Qwen3-ASR 1.7B` 后端。
- 官方 `qwen-asr-serve` 后端，端口 `8002`。
- 轻量串行化代理，端口 `8001`。
- 单并发 admission，带 GPU 空闲显存保护和 `faster-whisper` busy 感知。
- 请求前自动 warm-start 后端，请求窗口过期后自动关闭空闲后端。
- 通过 `POST :8010/release/asr` 启动的 `180s` faster-whisper 空闲计时器。
- 持久化宿主机 staging/results 目录：`/srv/qwen3-asr-1p7b/qwen-asr-staging` 和 `/srv/qwen3-asr-1p7b/qwen-asr-results`。

`:8000` 仍是面向 APP 的端点，但变成一个很薄的前置代理：

- 统计活跃 `faster-whisper` 请求数。
- 为 scheduler 暴露 `/busy`。
- 为客户端探测暴露 OpenAI-compatible 静态 `/v1/models` 响应。
- 把真实转写工作转发到 `:18000` 的内部后端。
- 在最后一个活跃请求结束后要求 scheduler 释放后端。

Admission 行为：

- 活跃 Qwen 离线任务会阻止新的 faster-whisper admission。
- `realtime_asr` 优先级值标识被允许进入的请求通道；它不会抢占正在运行的 Qwen 任务。
- 两个 GPU 后端互斥：切换通道时会停止空闲后端，并等待 VRAM 释放后再启动另一个。
- 已经活跃的 faster-whisper 请求允许完成，然后排队中的 Qwen 任务再切换 GPU 通道。
- Qwen 空闲时，`faster-whisper` 正常启动。
- `:8001` 代理会在内部重试 Qwen admission，而不是立即把临时 scheduler `429` 或 `503` 暴露给客户端。
- Mac 应用的本地文件 NUC 路径优先使用 `POST /jobs/upload` 加 `GET /jobs/{id}` 轮询，因此长任务即使原始上传请求不稳定，也能持续报告进度。
- busy endpoint 不可用时，scheduler 会把 GPU 利用率作为保守信号。
- `nvidia-smi` 不可用或空闲显存低于阈值时，admission 会以 HTTP `503` 失败。
- `Qwen` 请求完成后，后端会在空闲超时后停止，避免占用 VRAM。
- 最后一个 faster-whisper 请求结束后，它的后端会在 `180s` 后停止；`:8000` 代理保持可用。
- 队列和受控字幕 worker 在实际使用 NUC faster-whisper 后调用 `POST :8000/release/asr`。ASR 代理只把这个安全操作转发给仅 localhost 可访问的 scheduler；busy、offline 和 timeout 响应只记录日志，不改变已完成结果。

### ASR 历史和恢复分块处理

**ASR 历史** 标签页由 `CACHE_DIR / "asr-history.json"` 支撑。它支持原始文件缺失时用缓存 WAV 重跑、原子写入、损坏文件保留，以及旧输出路径迁移。删除历史行不会删除其 WAV、字幕缓存或输出文件。

恢复后的 Qwen3-ASR 控制包括 1-4 个本地进程副本、默认 `45s` 根分块、一层自适应拆分（`max(10.0, fastest_of_first_3 * 1.5)`），以及 FFmpeg 远程分块 VAD（`-35dB`、`0.3s`）。机器验收后，默认启用 2 副本本地 Qwen 处理、自适应拆分和远程 VAD。即使还没有时间基线，根分块在一次重试后仍失败时也会拆分一次，拆分后的子缓存可独立恢复。

环境变量会覆盖 `QSettings`：

```text
WHISPER_CAPTIONER_QWEN_PARALLEL=1
WHISPER_CAPTIONER_QWEN_REPLICAS=2
WHISPER_CAPTIONER_QWEN_CHUNK_SECONDS=45
WHISPER_CAPTIONER_ADAPTIVE_SPLIT=1
WHISPER_CAPTIONER_REMOTE_VAD=1
```

当前 `NUC Qwen3-ASR 1.7B` 的大文件本地文件流程：

1. Mac 先提取并缓存完整 `audio-16k-mono.wav`。
2. Mac 把完整缓存 WAV 上传到 `http://<NUC>:8001/jobs/upload`。
3. 代理把原始上传写入 `/srv/qwen3-asr-1p7b/qwen-asr-staging/<task-id>/`。
4. 如果 WAV 足够小，代理直接上游发送，并把返回文本按真实 WAV 时长摊分为近似句级时间戳。
5. 如果 WAV 大于直接上传阈值，代理创建名义 `30s` 分块，并在可用边缘加入 `2s` 上下文。
6. 相邻文本使用中文前缀/后缀模糊匹配合并。长重叠允许最多两次插入、删除或替换，不依赖 Qwen 伪时间戳。
7. 非静音分块返回空结果时，会作为两个重叠半块重试。静音保持为合法空结果。
8. 合并前，代理会过滤明显重复幻觉分块，例如低信息音频窗口中一个短语重复数百次的长分块。
9. 代理把 `metadata.json`、`response.json` 和 `chunks.json` 写入 `/srv/qwen3-asr-1p7b/qwen-asr-results/<task-id>/`。失败任务还会写入 `error.json`。

2026-05-13 验证结果：

- 合成测试文件：`test-huge.wav`
- NUC 上大小：约 `68 MB`
- 时长：`2200s`（`36m40s`）
- 任务 ID：原始验证运行为 `20260513-143855-test-huge.wav`。新的任务 ID 会包含短随机后缀，例如 `YYYYMMDD-HHMMSS-<8hex>-filename.wav`，避免同一秒内多个上传发生碰撞。
- 结果：NUC 侧分块后成功完成。
- 分块数：`74`
- 从保存的 metadata 计算的墙钟耗时：约 `4m34s`

NUC 上有用的检查路径：

- faster-whisper staging/results：`/srv/qwen3-asr-1p7b/asr-staging`、`/srv/qwen3-asr-1p7b/asr-results`
- faster-whisper home 快捷路径：`/home/jack/whisper-captioner-asr-files/staging`、`/home/jack/whisper-captioner-asr-files/results`
- Qwen staging/results：`/srv/qwen3-asr-1p7b/qwen-asr-staging`、`/srv/qwen3-asr-1p7b/qwen-asr-results`

当前保留策略：

- 这些 staging/results 文件会一直保留，直到手动删除。
- 目前没有 TTL 或自动清理任务。
- 对 Qwen 来说，NUC 当前持久化 JSON metadata/results；最终 `.srt` 和 `.txt` 仍由 Mac 应用导出。
- 当 Qwen 分块被识别为明显重复幻觉并过滤时，`chunks.json` 会保留该分块记录并记录 `filtered_reason`；合并转写会把该分块视为空，而不是导出重复垃圾文本。
- `chunks.json` 还记录请求上下文窗口、检测到的 dBFS、重叠字符/错误、新文本和空结果重试诊断。

### NUC ASR 优化基准（2026-06-11）

同一个 `227.718s` 中文 WAV 被发送到 NUC 面向 APP 的代理：

- `deepdml/faster-whisper-large-v3-turbo-ct2`，warm：`3.68s`，57 段。
- `large-v3`，warm/cached：`15.54s`，84 段。
- Turbo 约快 `4.2x`，但在该样本上输出文本更少，因此两个模式都保留。
- 第一次 Turbo 请求耗时 `82.01s`，因为它下载并加载了 `1.6 GB` 模型。这个一次性成本不代表 warm inference。

scheduler 和 faster-whisper 模型 TTL 都是 `900s`，因此活跃工作会话可以避免之前 3-5 分钟的重复加载周期，同时结束后仍会让 GPU 回到冷基线。

共存检查：

```bash
curl -fsS http://127.0.0.1:8000/busy
curl -fsS http://127.0.0.1:8000/v1/models
curl -fsS http://127.0.0.1:8001/healthz
curl -fsS http://127.0.0.1:8010/status
```

`GET :8000/health` 即使后端有意处于冷状态，也会以 HTTP 200 报告代理健康。查看 `upstream` 字段可区分 `healthy` 和 `stopped_or_unhealthy`。`:8000` 转写活跃时，`/busy` 应报告 `active_requests: 1`。Qwen 任务活跃时，scheduler 会推迟新的 faster-whisper admission。

### NUC GPU Guard Helper

当 NUC GPU 被 `Qwen3-ASR 1.7B` 或 `faster-whisper` 占住时，使用：

```bash
bash /Users/vickers/Documents/whisper-captioner/scripts/nuc_gpu_memory_guard.sh status
bash /Users/vickers/Documents/whisper-captioner/scripts/nuc_gpu_memory_guard.sh auto-clean
bash /Users/vickers/Documents/whisper-captioner/scripts/nuc_gpu_memory_guard.sh prep-asr
bash /Users/vickers/Documents/whisper-captioner/scripts/nuc_gpu_memory_guard.sh release-asr
bash /Users/vickers/Documents/whisper-captioner/scripts/nuc_gpu_memory_guard.sh idle-watch
bash /Users/vickers/Documents/whisper-captioner/scripts/nuc_gpu_memory_guard.sh unload-all
bash /Users/vickers/Documents/whisper-captioner/scripts/nuc_gpu_memory_guard.sh start-qwen
```

说明：

- `auto-clean` 只会在空闲 GPU 显存低于阈值时停止 `Qwen3-ASR 1.7B` 代理/后端容器。
- `prep-asr` 会在需要时先释放 Qwen GPU 占用，再通过 scheduler 启动 faster-whisper 后端。
- `release-asr` 启动 scheduler 的 faster-whisper 空闲计时器。
- `idle-watch` 停止空闲 Qwen 占用，并推动 scheduler 的 faster-whisper release 计时器。
- `unload-all` 停止两个 ASR 通道并释放 GPU 显存，不删除容器。
- `start-qwen` 先尝试对现有 `Qwen3-ASR 1.7B` 容器执行 `docker start`；只有容器不存在时才 fallback 到重新部署。
