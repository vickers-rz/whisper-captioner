# YouTube 字幕取证与时间轴融合流程

## 目的

将公开视频处理为可人工审核、可追溯的字幕与文本分析底座。该流程将文本正确性、声学时间和画面证据分层处理，不允许任一模型的生成结果无证据地覆盖另一层。

## 最佳方案

| 层 | 主要来源 | 职责 | 不承担的职责 |
| --- | --- | --- | --- |
| 长上下文文本 | Gemini 音频 ASR | 全文、专名、标点、内容分析 | 最终逐词时间戳 |
| 声学时间 | NUC faster-whisper words | 连续词级时间骨架 | 长文本纠错权威 |
| 画面证据 | Apple Vision 硬字幕 OCR | 实际可见文字、字幕起止边界 | 不可见语音和完整段落补写 |

推荐最终交付为 Gemini 全文加 NUC 时间轴，并由高置信 OCR 硬字幕锚点进行单调校正。

## 固定步骤

1. 使用 yt-dlp 下载音频；只有确认存在烧录字幕时下载视频。YouTube 需要 Chrome cookie 时，必须先取得用户明确授权。
2. 通过 Gemini API 对标准化 OGG/WAV 取得完整原文，保存原始文本和模型/提示词元数据。
3. 使用 NUC faster-whisper 输出含 `words` 的 `*-asr.json`，这是唯一的声学词级时间基线。
4. 对硬字幕视频以 2 FPS 抽帧，在样本联系表确认 ROI 后运行 Apple Vision OCR。
5. 运行 `scripts/evaluate_burned_subtitle_ocr.py`，保存原始 JSONL、OCR cues、SRT 与质量报告。
6. 运行 `scripts/fuse_burned_ocr_with_gemini_template.py`：
   - Gemini 与 NUC 做单调字符对齐；
   - OCR 仅在 Gemini 邻域内精确规范化匹配时恢复标点；
   - 精确匹配的 OCR cue 起止时间作为视觉锚点；
   - 用锚点对 NUC 时间做分段偏移校正，空档保留 NUC 的时间形状。
7. 输出三种字幕并保留 JSON：视觉 OCR、视觉 OCR 安全增强、Gemini OCR 锚定完整时间轴。

## 差异驱动 OCR 复核

当两份 ASR 文本存在争议时，不必对全片持续高频 OCR。使用
`scripts/plan_targeted_ocr_frames.py` 将 OGG/Gemini 模板字符位置通过 NUC
词级时间反推为目标时间窗：短窗以 6 FPS 复核，长缺失区在两端以 6 FPS 复核、
中段以 2 FPS 覆盖。`scripts/extract_targeted_ocr_frames.py` 会生成非连续帧及
真实时间戳映射；`apple_vision_ocr.swift --timestamps` 可保留这些时间戳。

再用 `scripts/adjudicate_transcript_differences_with_ocr.py` 对原始 Vision OCR
裁决 OGG/URL 候选。只将 `ocr_supports_url` 的单字符安全改动通过
`scripts/apply_ocr_adjudications.py` 写入新的裁决版 SRT；`OCR 证据不足` 不是对
URL 的支持，必须保持未决或人工复听。

## 已验证经验

- Gemini 的直连 YouTube 多模态能力适合内容理解，但其生成时间戳不能作为逐词事实依据；直接音频 ASR 更可复现。
- Apple Vision 在 M 系列 Mac 上可可靠识别该视频居中、底部的中文烧录字幕。对 `1280x640`、2 FPS 的 48 分钟视频，5,810 帧均成功处理。
- 本视频的 OCR 相对 Gemini 全文规范化字符覆盖为 98.7908%；2,106 条精确 OCR/Gemini cue 形成 4,212 个时间锚点，NUC 时间轴平均修正 0.205 秒。
- 对高质量 OCR，不应将 Gemini 按时间窗重切并直接替换 cue。这会造成相邻条目丢首字、重复和不可读字幕。仅允许局部精确匹配内的标点/书名号恢复。
- OCR 不可见、片头文字、广告叠层和未匹配 cue 必须保留其来源状态，不能伪装成视觉确认的语音文本。

## 代码与输出

- [apple_vision_ocr.swift](../scripts/apple_vision_ocr.swift)：逐帧 Apple Vision OCR。
- [evaluate_burned_subtitle_ocr.py](../scripts/evaluate_burned_subtitle_ocr.py)：OCR cue 聚合和质量审计。
- [fuse_burned_ocr_with_gemini_template.py](../scripts/fuse_burned_ocr_with_gemini_template.py)：Gemini/NUC/OCR 融合与 OCR 锚定时间轴。
- Codex skill：`~/.codex/skills/youtube-forensic-transcript`。

## 最低验收条件

- NUC `words` 时间单调、无空文本或负时长。
- OCR 已先做 ROI 抽样验证，且原始 JSONL 无未解释错误。
- OCR/Gemini 锚点仅使用精确规范化匹配且保持时间顺序。
- 报告包含覆盖率、置信度、锚点接受/拒绝数量、平均和最大偏移。
- 最终交付目录同时保存原始证据、各版本 SRT、JSON 和报告。
