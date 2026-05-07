# Whisper Captioner - 重构审核记录

## 当前状态

Claude 已经把原来的单文件原型拆成多个模块，当前代码可以通过 Python 编译检查：

```bash
/opt/anaconda3/envs/pyside6/bin/python -m compileall -q /Users/vickers/whisper-captioner/whisper_captioner
```

当前核心模块：

- `whisper_captioner/app.py`: Qt 主窗口、托盘菜单、信号连接、播放编排。
- `whisper_captioner/overlay.py`: 字幕浮窗、置顶图钉、拖动缩放、字体和透明度设置、播放控制按钮。
- `whisper_captioner/workers.py`: 实时音频、延迟音频、队列任务、Controlled URL 字幕生成 worker。
- `whisper_captioner/chrome_control.py`: Chrome AppleScript 控制。字幕计时轮询现在应使用无激活副作用的读取路径。
- `whisper_captioner/llm_handler.py`: Gemini/OpenAI-compatible/Anthropic/Rapid-MLX 校正调用。
- `whisper_captioner/cache.py`: canonical URL、缓存 key、URL 验证。
- `whisper_captioner/subtitle_io.py`: SRT/VTT/JSON 字幕读写。
- `whisper_captioner/config.py` 和 `whisper_captioner/models.py`: 配置常量、模式和数据类。

## 本轮修复

- 修复 Chrome 字幕轮询抢焦点问题：`chrome_current_time_url()` 通过 `activate_tab=False` 静默读取目标视频时间，不再每 120ms 激活 Chrome。
- 新增 URL 定向恢复播放：`chrome_resume_url()`，缓冲恢复优先恢复目标 URL 对应的视频。
- 受控播放中的缓冲暂停/恢复改为优先使用目标 URL，避免多个视频标签页时误控当前激活标签。
- Bilibili canonical URL 保留 `?p=` 参数，避免多 P/分集视频共用错误缓存。
- `zh.*` 内置字幕检测不再传 `--write-auto-subs`，避免把自动字幕误判为可直接跳过 Whisper/LLM 的人工中文字幕。
- 清理 `app.py` 中重构后遗留的未使用导入，降低后续维护噪音。
- 更新 `README.md`，让架构说明和当前代码一致。
- 新增 Analysis 标签页：复用当前视频字幕转写稿，生成“视频总结与分析”或“完整文章”，并缓存为 Markdown。
- 新增 MLX-Audio q5 后端：`mlx-community/whisper-large-v3-turbo-asr-5bit` 作为实验模式保留；缓存签名包含 backend/model，避免和 whisper.cpp q5_0 混用。
- 根据 30 秒和 70 秒本地样本 benchmark，Controlled URL captions 默认切回 `whisper.cpp large-v3-turbo-q5_0`。
- 为 MLX Whisper 的 chunk SRT 输出增加时间轴 clamp，避免单个 30 秒 chunk 被模型扩展到 chunk 外导致整体字幕漂移。
- 修复主窗口在 24 寸 1080p 副屏上底部不可见的问题：默认窗口尺寸调小，中央布局改为 `QScrollArea`，底部日志区限制最大高度。
- 修复 Controlled URL + SenseVoice.cpp 路径中 `RollingPrefetchWorker` 误调用 `QueueWorker` 方法的问题：`_run()` / `_run_capture()` 改为 `_run_cmd()` / `_run_cmd_capture()`。
- 修复 Controlled URL 命中人工 `zh.*` 字幕后直接退出整个 App 的行为：现在会加载内置字幕、刷新全文字幕列表、从 0 秒启动受控播放，并跳过 Whisper/LLM。
- 优化受控字幕 tick 查找：从每 250ms 线性扫描全部字幕，改为当前 index / 邻近 index 快路径加缓存 start-time 列表二分查找。
- 修复 Controlled URL + Qwen3-ASR 路径中缺少 `_pseudo_timestamp_qwen3_text` 的运行期错误：Qwen3 伪时间戳和短段合并逻辑已提升为模块级共享 helper，供 `QueueWorker` 和 `RollingPrefetchWorker` 共用。

## 仍需关注

- `run_chrome_script_for_url()` 仍然通过 URL 前缀匹配目标标签；如果同一个视频开了多个标签页，仍可能选择第一个匹配项。后续更稳的方案是记录 Chrome window/tab id 或通过 CDP 绑定目标页。
- `native-subtitles-zh.json` 仍是简单 segment list，没有独立的来源/版本 metadata。后续如果再次调整内置字幕策略，建议给 native subtitle cache 单独加 signature。
- 多行上下文/卡拉 OK 红字展示目前尚未真正实现，`SubtitleOverlay.set_caption_context()` 当前仍只显示当前句。
- Analysis 输出缓存目前只按当前视频 cache 目录区分，没有单独写入 LLM provider/model signature；如果切换模型后想强制重生成，需要删除对应 Markdown 缓存文件。
- `mlx-community/whisper-large-v3-turbo-asr-5bit` 必须走 `mlx-audio`，不能走 `mlx-whisper` 或 Rapid-MLX；当前已验证可生成 SRT，但本机样本速度和时间轴表现不如 whisper.cpp q5_0。
- `app.py` 仍承担 UI 状态机和播放编排，长期可以继续拆出 `playback_controller.py` 或 `controlled_session.py`。
- `cache.py` 的 canonical URL 仍可增强：YouTube Shorts (`/shorts/<id>`) 尚未统一为 `watch?v=<id>`；`b23.tv` 短链也尚未展开，会造成缓存复用不足。
- `subtitle_io.load_segments()` 目前假定 JSON schema 正确。坏 cache 可能触发较晚的异常；建议增加字段校验和更清楚的错误信息。
- `workers.py` 仍有临时文件清理问题。`yt-dlp` 下载文件、chunk wav、subchunk wav 长期使用会累积在系统临时目录。
- `workers.py` 的 subprocess helper 在异常路径上没有统一 `finally: self.proc = None`，通常不致命，但会留下已结束进程引用。
- `_repair_sparse_chunk_with_subchunks()` 里的 `int(f"{chunk_index}{part_index}")` 只用于日志/标签，但编号语义不准确，建议改为清晰的 label 生成方式。
- `QueueWorker` 和 `RollingPrefetchWorker` 仍有多处重复转写逻辑。后续应把命令执行、backend transcriber、cache pipeline 拆开。

## 新发现但暂不建议直接按单行修改

- `QueueWorker._transcribe_sense_voice_cpp_chunk_series()` 中 `step = max(1.0, chunk_duration)` 看起来和 `RollingPrefetchWorker` 的 `chunk_duration - overlap` 不一致，但当前 QueueWorker 同时会对非最后 chunk 使用 `trailing_trim=overlap`。如果只把 step 改成 `chunk_duration - overlap`，可能引入重复字幕区间。若要统一 SenseVoice overlap 策略，需要一起设计 `step`、`leading_trim` 和 `trailing_trim`。

## 建议的下一步小修顺序

1. 修 `cache.py` 的 YouTube Shorts canonical URL，顺手考虑 `youtu.be` 和 `youtube.com/live/<id>`。
2. 给 `subtitle_io.load_segments()` 加基础 schema 校验，坏 cache 报错要能定位文件和字段。
3. 修 `_repair_sparse_chunk_with_subchunks()` 的 subchunk label，避免拼接编号误导日志。
4. 整理 `QueueWorker` / `RollingPrefetchWorker` 的 `_run*` helper，确保 `self.proc` 在成功、失败、中止路径都会清理。
5. 增加临时文件清理，只清理当前 job 的 `whisper-rolling-<stamp>*`、chunk/subchunk 文件，避免误删用户文件。
6. 单独设计 Chrome tab identity：短期记录 AppleScript window/tab index，长期考虑 CDP。

## 架构优化方向

- `config.py`: 路径和工具位置高度本机硬编码。短期可增加环境变量覆盖和启动时路径检查；长期可放入设置页。
- `models.py`: 模式表和 README 重复维护。建议增加模式诊断/导出函数，减少文档漂移。
- `cache.py`: 继续增强 Bilibili、YouTube Shorts/live/playlist、短链 canonical 处理。
- `subtitle_io.py`: 增加 cache schema 校验和更明确的错误类型。
- `chrome_control.py`: 抽 AppleScript/JavaScript escaping，补 tab identity，减少误控标签页。
- `llm_handler.py`: 全片字幕一次性送 LLM 有 token 风险，后续需要按长度分批和缓存分片结果。
- `workers.py`: 最值得拆，建议拆成 `process_runner.py`、`transcribers.py`、`controlled_pipeline.py`、`native_subtitles.py`。
- `app.py`: 建议抽 `ControlledPlaybackController` 和 `AnalysisController`，让 `MainWindow` 少承担业务状态机。
- `overlay.py`: 当前可用；后续可把文本/emoji 控制按钮换成一致的图标按钮，并让上一句/当前句上下文展示更可配置。

## 当前主线

当前推荐先稳定 Controlled URL captions：

1. URL 输入或自动读取 Chrome 当前视频 URL。
2. canonical 化并建立缓存目录。
3. 如果有人工 `zh.*` 字幕，则直接加载字幕并启动受控播放，避免浪费 Whisper/LLM 资源。
4. 否则暂停目标 Chrome 视频，检查当前 pipeline 的 final cache。
5. 没有缓存时下载音频、分块 Whisper、整片 Gemini 2.5 Flash 校正。
6. 写入 final cache、SRT、TXT。
7. 启动目标 Chrome 视频，从视频 `currentTime` 静默轮询并驱动浮窗字幕。
