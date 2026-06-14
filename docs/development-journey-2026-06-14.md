# 2026-06-14 开发心路历程

本次会话从阅读 Codex 会话记录开始，逐步完成项目架构梳理、字幕可靠性管道提交、LattifAI 放弃、NUC 代理修复、SRT 质量对比、Gemini+Whisper 双模型融合的完整实现与接入。

---

## 1. 项目架构全量阅读

**用户提示词**：
> 请你完整阅读此项目的文档及代码，捋清此APP的架构和逻辑

**工作内容**：
- 通读 `README.md`、`ARCHITECTURE_AUDIT_v2026-05-07.md`、`CODE_REVIEW.md`
- 通读 `docs/subtitle_reliability_plan.md`、`docs/dual_asr_future_plan.md`、`docs/reference_projects_review_2026-06-13.md`
- 通读所有核心源码模块：`models.py`、`config.py`、`cache.py`、`subtitle_io.py`、`chrome_control.py`、`llm_handler.py`、`subtitle_reliability.py`、`external_backends.py`、`overlay.py`、`app.py`、`workers.py`

**产出**：完整的四层架构分析（UI 层 → 编排层 → 服务层 → 数据层），14 种 ASR 后端矩阵，13 种 LLM 提供商矩阵，NUC 远程推理架构图，四大运行时流程（Controlled URL / Realtime / QueueWorker / Local File），字幕可靠性管道详细说明。

---

## 2. Git 动向分析

**用户提示词**：
> 请你再通过git查看下此项目最近的动向

**工作内容**：
- 分析最近 40 条提交，划分为三阶段演进：
  - 阶段 1（6月初）：功能恢复与加固
  - 阶段 2（6月12日）：LLM 管线强化 + 安全加固
  - 阶段 3（6月13日）：NUC ASR 分块 + 语言自动检测
- 识别当前未提交工作：10 个文件修改（+915/-114 行）、8 个新文件未跟踪

**产出**：清晰的提交历史总结，确认未提交工作为 Codex 会话中完成的字幕可靠性闭环。

---

## 3. 未提交内容测试与提交

**用户提示词**：
> 请对尚未提交的内容进行完整测试，如果测试可以通过，就请提交git

**工作内容**：
- 运行完整测试套件：93 项全部通过，8.6s
- 编译检查 12 个源文件通过
- `git diff --check` 通过
- 提交 `e98c4a8`：18 文件，+2325/-114 行

**提交内容**：v2 ASR 数据模型、确定性 cue builder、质量审计、两轮局部补录、Chrome 多窗口选择、yt-dlp bot 检测重试、NUC 逐词参数转发、OmniVAD shadow、LattifAI 对齐适配器。

---

## 4. 下一步目标分析

**用户提示词**：
> 请你看下项目里的文档所描述的项目下一步的目标

**工作内容**：
- 分析四层次下步目标：
  - 短期：OmniVAD 基准测试 → 升级生产 VAD、LattifAI API key 验证
  - 中期：Whisper + Gemini 双模型融合（最大价值项）、说话人分离
  - 长期：workers.py / app.py 拆分、NUC 调度集中化、qwen_chat_service.py 拆分

---

## 5. 放弃 LattifAI

**用户提示词**：
> LattifAI我测试过了，不太行，请放弃，写在文档里

**工作内容**：
- 从 `external_backends.py` 移除 `CliAlignmentBackend`、`AlignmentBackend` protocol、`alignment_cache_key`、`optional_alignment_backend`、`save_alignment_result`
- 从 `workers.py` 移除对齐调用块（~25 行），替换为 `"status": "abandoned"`
- 从 `test_external_backends.py` 移除对齐 CLI 测试
- 更新三份文档（`subtitle_reliability_plan.md`、`dual_asr_future_plan.md`、`reference_projects_review_2026-06-13.md`）
- 提交 `f788536`：6 文件，+13/-130 行

---

## 6. APP 日志分析 + NUC Qwen 代理修复

**用户提示词**：
> 请看看APP日志，看现在正在跑的这个为啥这么慢

**问题诊断**：
- 音频 771 秒（12.9 分钟）、24MB，NUC Qwen3-ASR 1.7B 耗时 100 秒
- 根因：`MAX_DIRECT_UPLOAD_MB=64`，24MB < 64MB 被当作小文件整段发送
- Qwen3-ASR 一次性处理 771 秒音频，无分块无进度反馈

**修复**：
- 在 `nuc_qwen3_asr_1p7b_proxy.py` 新增 `MAX_DIRECT_UPLOAD_SECONDS=120`
- 直接上传条件改为：文件大小 < 64MB **且** 时长 < 120 秒
- 超过 2 分钟的音频自动走 30s/chunk 分块路径
- 通过 `docker cp` + `docker restart` 部署到 NUC
- 提交 `a607bfb`：本地 1 文件，+3/-1 行

---

## 7. SRT 质量跨版本对比

**用户提示词**：
> 请你对比/Volumes/T7_APFS/.../1729839784568919/路径下的所有srt，根据文件的时间戳以及此项目的git历史来判断下srt的质量

**分析结果**：

| 版本 | Git 基线 | 段数 | p50 时长 | 评价 |
|---|---|---|---|---|
| 172322 (6/13 17:24) | `cbf264d` | 139 | 5.6s | 正常，forced zh 导致英文混合 |
| 175635 (6/13 17:58) | `a0aa0f1` | 425 | 1.6s | **崩溃**: LLM 润色幻觉，13 种中文垃圾文本 |
| 172322-手规 | 手动 | 139 | 5.6s | 手动修复版 |
| auto-language-fixed (6/13 18:49) | `2210767` | 198 | 3.1s | **最佳**: language auto-detect 修复 |
| 155938 (6/14 16:01) | `e98c4a8` | 40 | 19.3s | ⚠️ v2 管道 cue builder 合并过度，`<asr_text>` 泄漏 |
| 160843 (6/14 16:08) | `a607bfb` | 40 | 19.3s | 同上，NUC 分块对文本影响极小 |

**结论**：最大单次质量提升是 language auto-detect。v2 管道有 cue builder 过度合并和标记泄漏两个 bug。

**用户关注点**：
> 我关注的是时间轴的粒度问题

进一步分析：v5/v6 粒度退化是因为用户用了 NUC Qwen3-ASR（整段上传 771s），而 v4 用的是 NUC faster-whisper（60s 分块）。两者不是同一后端。

---

## 8. Gemini + Whisper 双模型融合原型

**用户提示词**：
> 你先试试文档里的gemini的超高精度ASR识别率 加 whisper单词级时间戳 两者互补的那个方案，看看会不会有改善。Gemini的提示词及别人的测评报告请参考：...

**参考资料**（网络不可达，基于项目自有文档 `dual_asr_future_plan.md` 和 `reference_projects_review_2026-06-13.md`）：
- Gemini 2.5 Flash WER ~4%（极高文本准确率）
- Gemini 自报时间戳 DER 73%–324%（完全不可用）
- 因此：Gemini 负责文本 → Whisper 负责时间轴 → 字符级对齐融合

**实现**：
- `gemini_transcribe_audio()`：调用 Gemini API，text-only 输出（不要求时间戳）
- `fuse_gemini_with_whisper()`：difflib.SequenceMatcher 字符级对齐

**实测结果**（772s 英语 IELTS 音频，前 180s 切片）：

| 指标 | Gemini+Whisper | Qwen v4 | Qwen v5/v6 |
|---|---|---|---|
| 段数 | **44** | 41 | ~9 |
| p50 时长 | **0.8s** | 3.6s | 12.9s |
| 粒度改善 | — | **4.5×** | **16×** |
| Gemini 耗时 | 9.8s | — | — |

提交 `ca5bd8a`：2 文件，+252/-3 行。

---

## 9. Gemini 融合 GUI 开关

**用户提示词**：
> 此功能是否在APP的GUI里可以进行手动开启？

> 要的

**实现**：
- `ui_builder.py`：新增 `「启用 Gemini + Whisper 双模型融合」` 复选框
- `app.py`：QSettings 持久化存取
- `RollingPrefetchWorker`：融合在 ASR 完成后、质量审计前自动触发
- 提交 `6ca3bfb`：3 文件，+50/-1 行

---

## 10. 失败弹窗替代静默跳过

**用户提示词**：
> 关于你上述的："无 API key 时静默跳过，无 word timestamps 时跳过并记录原因"，请不要这样做，你这样我根本就不能及时发现实情，请加个失败后的弹窗可以手动选择跳过或取消

**实现**：
- 预检（启动前）：无 `GEMINI_API_KEY` → QMessageBox：「跳过融合，继续」/「取消任务」
- 运行时：`gemini_fusion_blocked` 信号 → `_on_gemini_fusion_blocked()` 弹窗
- 提交 `fa4eb30`：2 文件，+39/-1 行

---

## 11. QueueWorker 接入 + ASR 模式自动跳转

**用户提示词**：
> 你看下日志，为啥我明明已经手动勾选了双模型，现在为啥模型还没调好

**问题**：用户用本地文件队列模式（QueueWorker + NUC Qwen3-ASR），Gemini 融合只接入了 RollingPrefetchWorker（受控字幕模式）。

**用户提示词**：
> 要的，还要在我勾选好后，ASR模型选择框那里要自动跳好

**实现**：
- `QueueWorker` 构造器新增 `gemini_fusion_enabled` + `gemini_api_key` 参数
- `_transcribe_file_via_nuc_chunks()` 返回类型改为 `ASRResult`（含 words），替代 `list[SubtitleSegment]`
- `_process()` 中 nuc_asr 分支加入融合逻辑
- `_on_gemini_fusion_toggled()`：勾选时自动切换 ASR 模式到 `nuc_asr_turbo`
- 提交 `202d930`：2 文件，+71/-7 行

---

## 12. Qwen3.5-4B 时间轴仲裁者

**用户提示词**：
> 如果NUC的词级时间戳与Gemini的ASR的句子对不上，你是咋处理的？...有没有仲裁者机制？仲裁者能否用NUC上的qwen3.5-4B？

**架构澄清（用户）**：
> 你可能理解有偏差，咱们本来就是以Gemini的文本为唯一信任文本的，NUC上的词级时间戳只是为了补齐Gemini的时间轴短板的，所以NUC上的仲裁者的仲裁内容只是对其时间轴而已，不可对Gemini的文本进行修改

**最终架构**：
```
Gemini 2.5 Flash (云端)          NUC faster-whisper (本地)
      │                                  │
  文本转写（唯一真相源）              词级时间戳（时间轴）
      │                                  │
      └──────────┬───────────────────────┘
                 │
        fuse_gemini_with_whisper_arbitrated()
                 │
         ┌───────┴────────┐
         │ ① difflib 对齐  │  → 每段算置信度 (0.0-1.0)
         │ ② 置信度 < 0.4  │  → NUC Qwen3.5-4B 仲裁时间轴
         │ ③ 规范化时间轴   │  → 单调、非重叠、≥0.4s
         └───────┴────────┘
                 │
           最终 SRT（Gemini 文本 + 可靠时间戳）
```

**仲裁者角色**：
- 只输出 `start_ms end_ms` 两个整数
- Gemini 文本永不被修改
- 仅在 difflib 对齐置信度 < 0.4 时调用

提交 `2f0b151`：3 文件，+278/-3 行。

---

## 13. 全部提交记录

```
2f0b151 feat: add Qwen3.5-4B timing arbiter for Gemini+Whisper fusion
202d930 feat: wire Gemini fusion into QueueWorker and auto-select ASR mode
fa4eb30 fix: replace silent Gemini fusion skip with explicit dialogs
6ca3bfb feat: wire Gemini + Whisper fusion into GUI and pipeline
ca5bd8a feat: add Gemini + Whisper dual-model transcription fusion
a607bfb fix: add duration threshold to NUC Qwen direct-upload decision
f788536 refactor: remove LattifAI alignment integration
e98c4a8 feat: implement single-model subtitle reliability pipeline
```

## 14. 当前测试状态

102 项单元测试全部通过，耗时 ~15s。

覆盖模块：`asr_history`、`chaptering`、`chrome_control`、`external_backends`、`llm_handler`、`mac_gpu_monitor`、`nuc_faster_whisper_proxy`、`nuc_qwen_proxy`、`nuc_scheduler`、`overlay_chapters`、`parallel_llm_polish`、`qwen_chat_service`、`subtitle_reliability`、`worker_recovery`。

---

## 15. 未完成事项

- [ ] GitHub push（网络阻断，SSH/HTTPS 均不可达）
- [ ] OmniVAD shadow → 生产 VAD 升级（需独立基准测试）
- [ ] `dual_asr_future_plan.md` 中描述的完整双模型融合评分器 + UI 控件
- [ ] 说话人分离（diarization）
- [ ] workers.py / app.py 架构拆分
- [ ] NUC 调度集中化
