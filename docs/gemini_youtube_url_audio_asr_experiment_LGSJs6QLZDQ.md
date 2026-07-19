# Gemini 公开 YouTube URL 音频 ASR 实验

## 目的

验证 Gemini API 能否直接接收公开视频 URL，只输出音轨全文转写，从而跳过本地音频下载、WebM/OGG 转码和 Gemini File API 上传。

## 方法

- 输入：`https://www.youtube.com/watch?v=LGSJs6QLZDQ`。
- 模型：`gemini-3.5-flash`。
- API：`client.interactions.create`，`video` URL 输入，`resolution=low`。
- 提示词：明确要求仅转写实际语音，忽略画面、烧录字幕、标题、简介和章节；不生成时间戳、总结或视频分析。

## 结果

- 成功，无本地音频下载、转码或 File API 上传。
- Gemini 处理耗时：109.051 秒。
- 输出：15,076 个原始字符，13,925 个规范化字符。
- 与此前 `gemini-2.5-flash` File API 音频转写比较：
  - File API 文本规范化字符：14,638。
  - 匹配字符：13,703。
  - URL 输出对 File API 的覆盖：93.6125%。
  - URL 输出精度：98.4057%。
  - 全局序列相似度：95.9493%。

## 结论

公开 URL 输入可作为 Gemini 的快速全文候选和视频理解入口，能消除 Gemini 分支的本地下载/转码/上传步骤。

本次结果仍比 File API 音频原文少约 6.4% 的规范化字符，且两次使用的模型不同，不能据此宣称 URL 输入整体更准确。因此：

- 用作快速候选或 URL 无障碍时的默认入口。
- 对最终可审核字幕，继续由 NUC words 提供时间骨架，并以 OCR 锚点校正。
- 在没有独立质量对照时，保留 File API 音频 ASR 作为可复现回退，而不完全替换。
