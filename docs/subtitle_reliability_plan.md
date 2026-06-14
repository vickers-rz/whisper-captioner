# 单模型字幕可靠性闭环

## 范围

本阶段只实现单模型字幕可靠性闭环。JSON 是无损事实来源，SRT 是由确定性
规则生成的渲染产物。双模型融合与说话人分离不在本阶段实现。

实施内容：

- 版本化 `ASRResult`、逐词时间戳和旧 segment 缓存兼容读取。
- NUC faster-whisper 逐词参数转发、完整响应留存和 capability fallback。
- 确定性 cue builder，以及单调、非重叠、非空时间轴约束。
- ffmpeg speech regions、质量审计、局部补录和 `large-v3` 复核。
- detect-once-then-pin 语言策略。
- OmniVAD shadow 采集，不参与生产决策。
- 可选 CLI 强制对齐，失败时回退原结果。
- 分阶段缓存、质量报告、UI 和日志状态。

## 默认参数

| 项目 | 中文 | 拉丁文本 |
| --- | ---: | ---: |
| 单条最大字符数 | 22 | 42 |
| 目标 CPS | 12 | 17 |

通用默认值：

- cue 最长 5 秒，最短 0.4 秒。
- 词间停顿超过 0.8 秒优先断句。
- 句号、问号和感叹号是硬边界。
- 逗号和分号在达到最小长度后可作为软边界。
- 可疑区间增加 2 秒 guard，相距不足 0.5 秒时合并。
- 局部补录使用 8 至 15 秒窗口，最多两轮。
- ffmpeg VAD 是生产默认，OmniVAD 只运行 shadow。

## 缓存

新结构使用 `schema_version=2`。原始 ASR、cues、质量报告、OmniVAD shadow
和对齐结果分别缓存。旧 segment 数组继续可读，但不能绕过新版质量审计。

缓存失效边界：

- cue 参数变化只重建 cues。
- VAD 参数变化重建 speech regions、质量报告和后续阶段。
- ASR 模型、语言或分块变化才重跑 ASR。
- 对齐缓存绑定音频哈希、文本哈希、适配器名称和版本。
- 最终 SRT 或最终文本缓存存在时，仍必须存在可接受的新版质量报告。

## 质量状态

- `passed`
- `passed_with_warnings`
- `incomplete_speech_coverage`
- `failed`

不完整结果仍允许导出 SRT，但日志、UI 和 `quality-report.json` 必须列出具体
区间、原因、补录次数和采用模型。

## 实施进度

- [x] 建立本阶段与下一阶段边界文档。
- [x] v2 数据结构和旧缓存兼容。
- [x] NUC 逐词参数与响应解析。
- [x] 确定性 cue builder。
- [x] 统一质量审计和最多两轮局部补录。
- [x] detect-once-then-pin。
- [x] OmniVAD shadow。
- [x] 可选 CLI 强制对齐。
- [x] UI、日志、缓存签名和恢复流程。
- [x] 单元测试及真实素材回归。

## 验收记录

验收目标：

- 默认 speech coverage 至少 98.5%。
- 连续未解释人声缺口不超过 0.6 秒，或在报告中明确列出。
- 不存在未报告的低密度长段。
- `04:17-04:20` 恢复“航道实业高中”“仁川云峰工业高中”“云山机械工高中”。
- OmniVAD 和对齐 CLI 均缺失时仍能正常生成字幕。
- Chrome 多窗口选择、yt-dlp 重试和现有 NUC 修复不回退。

最终测试命令、素材路径、质量指标和结果在实现完成后补入本节。

### 2026-06-14 实施结果

测试：

```text
/Users/vickers/miniforge3/envs/whishperapp_pyside6/bin/python \
  -m unittest discover -s tests
```

- 完整测试通过，静态编译和 `git diff --check` 通过。
- NUC proxy 已部署逐词参数修复，真实响应同时包含顶层和 segment 内 words。
- 真实响应保存在 NUC
  `/app/asr-results/20260614-062306-319db533-school-names-240-265.wav`。
- OmniVAD `0.2.12` 已安装并完成真实音频冒烟，25 秒音频 RTF 约 `0.0115`。
- LattifAI `1.5.16` 已安装，MPS/CoreML 和依赖自检正常；当前缺少
  `LATTIFAI_API_KEY`，因此 UI/质量报告显示 `unauthenticated`，普通字幕不受影响。

真实素材：

```text
/Volumes/T7_APFS/MacBackup/Movies/WhisperCaptioner/cache/local-audio/
70bb48dae97aa3ec665138ef/audio-16k-mono.wav
```

- 30 秒坏块使用两个 15 秒 turbo 窗口补录，并执行 large-v3 第二轮复核。
- 复核选择保留质量更好的 turbo 候选，恢复：
  - 航道实业高中
  - 仁川云峰工业高中
  - 云山机械工高中
- 修复后 segment 时长均为正、时间单调且无重叠。
- guard 前后的可靠原字幕得到保留，补录结果只替换问题区间。

覆盖审计的保守结果：

- `240-265s` 中 `244-252s` 被 ffmpeg 和 OmniVAD 同时判为非静音/voice-like，
  但 turbo 与 large-v3 局部复核均无可转写文本。
- 本阶段不允许 OmniVAD shadow 改变生产决策，因此该区间会明确报告为
  `incomplete_speech_coverage`，不会伪报达到 98.5%。
- 后续若要把影视背景声从 coverage 分母排除，需要经过独立基准后将音频事件
  分类器升级为生产决策组件；不在本阶段静默放宽审计。
