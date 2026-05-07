# 字幕同步架构修复与当前状态

## 当前状态

项目已经从早期的“边缓冲边播放”方案，调整为“整片生成后播放”的方案：

1. canonicalize URL，避免同一视频被 query 参数切成多个缓存。
2. 检查 `zh.*` 内置字幕，存在则跳过 Whisper/LLM。
3. 无正式中文字幕时，暂停 Chrome。
4. 下载音频并按 30 秒分块。
5. `whisper-cli` 转写每个 chunk，并把块内时间戳转换为全局时间戳。
6. 可选使用 Gemini 2.5 Flash 做全文规整。
7. 写入 `final-subtitles-current.json`。
8. 从 Chrome 视频 `0s` 开始播放，并按 controlled video 的 `currentTime` 显示字幕。

## 已修复：分块时间戳计算错误

早期问题：

```python
offset += self.chunk_seconds
```

当前已修复为：

```python
offset += remaining  # Use actual chunk duration, not fixed chunk_seconds
```

这样最后一个不足 30 秒的 chunk 不会污染后续全局时间轴。

## 已修复：URL 与缓存碎片化

同一个 Bilibili 视频可能带不同 `spm_id_from`、`trackid`、`vd_source` 等参数。之前这些 URL 会生成不同 cache key，导致：

- 最终字幕缓存无法复用。
- `subtitle-sync.json` 偏移无法复用。
- Chrome tab 匹配不稳定。

当前使用 `canonical_media_url()`：

- Bilibili: `https://www.bilibili.com/video/<BV...>`
- YouTube: `https://www.youtube.com/watch?v=<id>`

## 已修复：controlled seek/sync 使用统一视频源

controlled 模式下：

- cache key 使用 canonical URL。
- Chrome `currentTime` 读取使用 canonical URL 定位目标 tab。
- seek 和 `Sync line` 也使用 controlled URL，而不是 active tab。

## 当前模块拆分进度

已从 `app.py` 抽出：

- `config.py`: 路径、工具、模型、pipeline 常量。
- `models.py`: `CaptionMode`、`LLMProvider`、`SubtitleSegment`。
- `subtitle_io.py`: SRT/VTT/JSON 字幕读写。
- `cache.py`: canonical media URL 和 cache slug。
- `chrome_control.py`: Chrome AppleScript 控制。

后续建议继续抽：

- `llm.py`
- `workers.py`
- `overlay.py`
- `main_window.py`

## 验证方法

1. 打开同一个 Bilibili 视频，但使用不同 query 参数的 URL。
2. 第一次运行 controlled captions，生成 final cache。
3. 第二次运行，应复用同一个 canonical cache。
4. 使用 `Sub +/-0.5s` 或 `Sync line` 调整偏移。
5. 再次运行同视频，应加载同一个 `subtitle-sync.json`。

如果仍不同步，下一步应记录 controlled video 的 `currentTime`、字幕 `caption_time`、当前字幕 index，并比较导出的 SRT 是否本身对齐音频。
