# Gemini 模板引导硬字幕 OCR 实验（前 5 分钟）

## 目标

以最初的 Gemini ASR 全文为内容基准，改进 Apple Vision 对画面烧录字幕的识别，同时不把推断文本伪装成逐帧视觉证据。

## 输入与对齐

- Gemini 原始全文：`gemini-2.5-flash-transcript.txt`。
- 时间锚点：NUC Whisper 融合文件的词级时间戳。
- 视觉证据：Apple Vision 每秒 2 帧的字幕区域 OCR。
- Gemini 到 NUC 的单调字符对齐：14,177 / 14,638 个 Gemini 规范化字符匹配，覆盖率 96.8507%。

## 结果

### 视觉 OCR

- 233 条 OCR cue，其中 223 条能在 Gemini 模板邻域精确验证。
- OCR 相对 Gemini 前五分钟基准：1,421 / 1,432 个字符匹配，Gemini 覆盖率 99.2318%，OCR 精度 98.7491%。
- 余下差异包含片头残留字幕“的入场券”和少量两套文稿的措辞边界差异，不能安全地视为 OCR 漏字。

### 安全模板补丁

只在 OCR 去除标点后与 Gemini 片段完全一致时，恢复该片段内部、成对完整的标点。共修复 3 条：

- `路德维希冯米塞斯` -> `路德维希·冯·米塞斯`
- `马丁路德的` -> `马丁·路德的`
- `未竟的明史手稿` -> `未竟的《明史》手稿`

不会补入额外正文，不会拆分或重切 OCR cue；书名号不完整的候选会保留 OCR 原文。

### 完整 Gemini 时间轴

以 Gemini 为主体、用 NUC 词级锚点插值生成了独立 SRT。它相对 Gemini 基准的字符覆盖率为 100%，用于全文阅读、内容分析和后续结构化；其时间边界为推断结果，不是逐帧硬字幕验证结果。

## 交付物说明

- `burned-subtitle-gemini-template.srt/json`：推荐用于审阅画面烧录字幕。JSON 同时保留 `ocr_text`、补丁来源和模板时间窗。
- `gemini-template-full-timeline.srt/json`：推荐用于以 Gemini 文本为准的完整转写与内容分析。
- `burned-subtitle-gemini-template-report.json`：所有覆盖率、精度与对齐指标。
