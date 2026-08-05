#!/bin/bash
set -uo pipefail

SCRIPT_ROOT="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_PYTHON="${HOME}/miniforge3/envs/whishperapp_pyside6/bin/python"
PYTHON_BIN="${FORENSIC_PYTHON:-$DEFAULT_PYTHON}"
OUTPUT_ROOT="${WHISPER_CAPTIONER_OUTPUT_DIR:-/Volumes/T7_APFS/MacBackup/Movies/WhisperCaptioner}/artifacts/generated"
LOG_DIR="${HOME}/Library/Logs/WhisperCaptioner"
LAST_LOG="$LOG_DIR/forensic-subtitle-last.log"
KEYCHAIN_SERVICE="WhisperCaptioner"
GEMINI_KEYCHAIN_ACCOUNT="gemini-api-key"

if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python3 || true)"
fi

usage() {
  cat <<'EOF'
Forensic Subtitle TUI

Usage:
  ./ForensicSubtitle.command menu       Interactive TUI (default on double-click)
  ./ForensicSubtitle.command gemini-url [URL]
                                       URL -> yt-dlp webm/bestaudio -> OGG/Opus -> Gemini ASR
  ./ForensicSubtitle.command gemini-url-direct [URL]
                                       Gemini URL direct -> audio-only full transcript
  ./ForensicSubtitle.command native-subtitles [URL]
                                       YouTube/Bilibili URL -> detect/download native subtitles
  ./ForensicSubtitle.command gemini-local [MEDIA]
                                       Local audio/video media -> Gemini OGG/File API transcript
  ./ForensicSubtitle.command segment-md [TRANSCRIPT_MD]
                                       Hybrid RAG -> local Qwen rich Markdown segmentation
  ./ForensicSubtitle.command segment-pure [TRANSCRIPT]
                                       Pure local Qwen rich Markdown segmentation
  ./ForensicSubtitle.command segment-tfidf [TRANSCRIPT]
                                       TF-IDF RAG -> local Qwen rich Markdown segmentation
  ./ForensicSubtitle.command segment-ollama [TRANSCRIPT]
                                       Ollama embedding RAG -> local Qwen rich Markdown segmentation
  ./ForensicSubtitle.command segment-hybrid [TRANSCRIPT]
                                       Hybrid RAG -> local Qwen rich Markdown segmentation
  ./ForensicSubtitle.command segment-gemini [TRANSCRIPT]
                                       Gemini 2.5 Flash/Pro -> rich Markdown segmentation
  ./ForensicSubtitle.command nuc-local [MEDIA]
                                       Local audio/video media -> selectable NUC ASR
  ./ForensicSubtitle.command run [URL]  Run/resume the complete pipeline
  ./ForensicSubtitle.command status     Show recent jobs
  ./ForensicSubtitle.command doctor     Check dependencies and NUC connectivity
  ./ForensicSubtitle.command outputs    Open generated artifacts in Finder
  ./ForensicSubtitle.command tail       Show the last pipeline log

Pipeline: hard-subtitle probe -> NUC word ASR -> Gemini OGG ASR -> fixed-time
text backfill -> optional targeted OCR text adjudication -> deterministic final SRT.
EOF
}

pause_menu() {
  echo
  printf "按 Enter 返回菜单..."
  read -r _unused || true
}

python_worker() {
  "$PYTHON_BIN" -u "$SCRIPT_ROOT/scripts/forensic_tui.py" "$@"
}

asr_worker() {
  "$PYTHON_BIN" -u "$SCRIPT_ROOT/scripts/asr_entrypoints.py" "$@"
}

format_elapsed() {
  elapsed_seconds="$1"
  hours=$((elapsed_seconds / 3600))
  minutes=$(((elapsed_seconds % 3600) / 60))
  seconds=$((elapsed_seconds % 60))
  if [ "$hours" -gt 0 ]; then
    printf "%02d:%02d:%02d" "$hours" "$minutes" "$seconds"
  else
    printf "%02d:%02d" "$minutes" "$seconds"
  fi
}

run_logged() {
  job_label="$1"
  shift
  mkdir -p "$LOG_DIR"
  job_started_at="$(date +%s)"
  echo
  echo "$job_label"
  echo "日志：$LAST_LOG"
  echo
  set +e
  "$@" 2>&1 | tee "$LAST_LOG"
  job_status=${PIPESTATUS[0]}
  set -e
  job_finished_at="$(date +%s)"
  job_elapsed=$((job_finished_at - job_started_at))
  if [ "${job_status}" -eq 0 ]; then
    echo
    echo "任务完成。处理耗时：$(format_elapsed "$job_elapsed")"
  else
    echo
    echo "任务退出，状态码：${job_status}；处理耗时：$(format_elapsed "$job_elapsed")"
  fi
  return "${job_status}"
}

strip_outer_quotes() {
  stripped_value="$1"
  case "$stripped_value" in
    \"*\") stripped_value="${stripped_value#\"}"; stripped_value="${stripped_value%\"}" ;;
    \'*\') stripped_value="${stripped_value#\'}"; stripped_value="${stripped_value%\'}" ;;
  esac
  printf "%s" "$stripped_value"
}

gemini_url_identity() {
  "$PYTHON_BIN" - "$1" <<'PY'
import re
import sys
from urllib.parse import parse_qs, urlparse

url = sys.argv[1]
parsed = urlparse(url)
query = parse_qs(parsed.query)
identity = (query.get("v") or [""])[0]
if not identity:
    parts = [part for part in parsed.path.split("/") if part]
    identity = parts[-1] if parts else "youtube"
identity = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "-", identity).strip(" .-") or "youtube"
print(identity[:120])
PY
}

mask_secret() {
  secret_value="$1"
  secret_length=${#secret_value}
  if [ "$secret_length" -le 24 ]; then
    printf "%s" "已保存（长度 ${secret_length}）"
    return
  fi
  printf "%s...%s" "${secret_value:0:10}" "${secret_value: -10}"
}

saved_gemini_key() {
  if [ -n "${GEMINI_API_KEY:-}" ]; then
    printf "%s" "$GEMINI_API_KEY"
    return
  fi
  /usr/bin/security find-generic-password \
    -s "$KEYCHAIN_SERVICE" \
    -a "$GEMINI_KEYCHAIN_ACCOUNT" \
    -w 2>/dev/null || true
}

save_gemini_key() {
  key_value="$1"
  if [ -z "$key_value" ]; then
    return 0
  fi
  /usr/bin/security add-generic-password \
    -U \
    -s "$KEYCHAIN_SERVICE" \
    -a "$GEMINI_KEYCHAIN_ACCOUNT" \
    -w "$key_value" >/dev/null 2>&1 || true
}

choose_gemini_model() {
  default_model="${GEMINI_URL_ASR_MODEL:-gemini-2.5-flash}"
  echo
  echo "Gemini 模型："
  echo "  1) gemini-2.5-flash（推荐）"
  echo "  2) gemini-2.5-pro"
  echo "  3) gemini-2.0-flash"
  printf "> [1] "
  read -r requested_model_choice
  case "${requested_model_choice:-1}" in
    2) model_value="gemini-2.5-pro" ;;
    3) model_value="gemini-2.0-flash" ;;
    1|"") model_value="$default_model" ;;
    *)
      echo "未知模型选项：$requested_model_choice；使用默认模型：$default_model"
      model_value="$default_model"
      ;;
  esac
  GEMINI_MODEL_VALUE="$model_value"
}

prompt_gemini_key() {
  echo
  existing_api_key="$(saved_gemini_key)"
  if [ -n "$existing_api_key" ]; then
    echo "Gemini API Key：已保存 $(mask_secret "$existing_api_key")"
    printf "Gemini API Key（按 Enter 沿用；输入新 Key 将更新保存，输入不显示）: "
  else
    printf "Gemini API Key（未保存；请输入，输入不显示）: "
  fi
  stty -echo 2>/dev/null || true
  read -r api_key_value
  stty echo 2>/dev/null || true
  echo
  if [ -n "$api_key_value" ]; then
    save_gemini_key "$api_key_value"
  fi
  GEMINI_API_KEY_VALUE="$api_key_value"
}

run_gemini_url_asr() {
  supplied_url="${1:-}"
  direct_url="${2:-0}"
  if [ -n "$supplied_url" ]; then
    url_value="$supplied_url"
  else
    if [ "$direct_url" = "1" ]; then
      echo "输入公开 YouTube URL（直接交给 Gemini，仅请求音频转写）："
    else
      echo "输入公开 YouTube URL（将用 yt-dlp 下载 webm/bestaudio，再转为 OGG/Opus 发给 Gemini）："
    fi
    printf "> "
    read -r url_value
  fi
  if [ -z "$url_value" ]; then
    echo "未输入 YouTube URL。"
    return 2
  fi

  choose_gemini_model
  model_value="$GEMINI_MODEL_VALUE"
  prompt_gemini_key
  api_key_value="$GEMINI_API_KEY_VALUE"

  echo
  echo "产物目录留空时，将按 YouTube video ID 保存到："
  echo "  $OUTPUT_ROOT"
  if [ "$direct_url" != "1" ]; then
    default_identity="$(gemini_url_identity "$url_value")"
    default_job_dir="$OUTPUT_ROOT/Gemini-URL-ASR [$default_identity]"
    echo "默认 OGG/Opus 保存路径："
    echo "  $default_job_dir/work/gemini-audio.ogg"
    echo "默认 yt-dlp 原始音频保存路径："
    echo "  $default_job_dir/work/source-audio.<ext>"
    echo "默认 ASR 文稿保存路径："
    echo "  $default_job_dir/gemini-local-audio-asr-transcript.md"
  else
    default_identity="$(gemini_url_identity "$url_value")"
    default_job_dir="$OUTPUT_ROOT/Gemini-URL-ASR [$default_identity]"
    echo "当前为直接 URL 模式：不会生成本地 OGG。"
    echo "默认 ASR 文稿保存路径："
    echo "  $default_job_dir/gemini-youtube-url-audio-only-transcript.md"
  fi
  printf "自定义本次作业目录（可留空）: "
  read -r custom_output

  command_args=(gemini-url "$url_value" --model "$model_value")
  if [ "$direct_url" = "1" ]; then
    command_args+=(--direct-url)
  fi
  if [ -n "$custom_output" ]; then
    custom_output="$(strip_outer_quotes "$custom_output")"
    command_args+=(--output-dir "$custom_output")
    if [ "$direct_url" != "1" ]; then
      echo "本次 OGG/Opus 将保存到："
      echo "  $custom_output/work/gemini-audio.ogg"
      echo "本次 ASR 文稿将保存到："
      echo "  $custom_output/gemini-local-audio-asr-transcript.md"
    else
      echo "本次 ASR 文稿将保存到："
      echo "  $custom_output/gemini-youtube-url-audio-only-transcript.md"
    fi
  fi
  if [ "$direct_url" = "1" ]; then
    job_label="Gemini URL -> 直接提交 URL -> Gemini 全文转写"
  else
    job_label="Gemini URL -> yt-dlp 音频下载 -> OGG/Opus -> Gemini 全文转写"
  fi
  if [ -n "$api_key_value" ]; then
    GEMINI_API_KEY="$api_key_value" run_logged \
      "$job_label" \
      asr_worker "${command_args[@]}"
  else
    run_logged \
      "$job_label" \
      asr_worker "${command_args[@]}"
  fi
}

run_native_subtitles() {
  supplied_url="${1:-}"
  if [ -n "$supplied_url" ]; then
    url_value="$supplied_url"
  else
    echo "输入 YouTube/Bilibili 视频 URL（检测是否有自带字幕；有则直接下载）："
    printf "> "
    read -r url_value
  fi
  if [ -z "$url_value" ]; then
    echo "未输入视频 URL。"
    return 2
  fi

  echo
  printf "允许 yt-dlp 读取 Chrome Cookie？Bilibili/会员/地区限制视频建议启用 [y/N]: "
  read -r cookie_choice
  cookie_choice="${cookie_choice:-N}"
  chrome_profile="${FORENSIC_CHROME_PROFILE:-Default}"
  case "$cookie_choice" in
    y|Y|yes|YES)
      printf "Chrome Profile 名称或路径 [%s]: " "$chrome_profile"
      read -r requested_chrome_profile
      chrome_profile="${requested_chrome_profile:-$chrome_profile}"
      ;;
  esac

  echo
  echo "产物目录留空时，将按视频标题保存到："
  echo "  $OUTPUT_ROOT"
  printf "自定义本次作业目录（可留空）: "
  read -r custom_output

  command_args=(native-subtitles "$url_value")
  case "$cookie_choice" in
    y|Y|yes|YES)
      command_args+=(--cookies-from-chrome --chrome-profile "$chrome_profile")
      ;;
  esac
  if [ -n "$custom_output" ]; then
    custom_output="$(strip_outer_quotes "$custom_output")"
    command_args+=(--output-dir "$custom_output")
  fi

  run_logged \
    "视频链接 -> 检测/下载自带字幕（优先中文字幕；无中文字幕则尝试任意字幕）" \
    asr_worker "${command_args[@]}"
}

run_gemini_local_asr() {
  supplied_media="${1:-}"
  if [ -n "$supplied_media" ]; then
    audio_value="$supplied_media"
  else
    echo "输入本地音频/视频媒体文件路径（支持 OGG、WebM、MP4、MKV 等，只提取音频流）："
    printf "> "
    read -r audio_value
  fi
  audio_value="$(strip_outer_quotes "$audio_value")"
  if [ -z "$audio_value" ]; then
    echo "未输入本地媒体路径。"
    return 2
  fi

  choose_gemini_model
  model_value="$GEMINI_MODEL_VALUE"
  prompt_gemini_key
  api_key_value="$GEMINI_API_KEY_VALUE"

  echo
  echo "产物目录留空时，将按媒体文件名保存到："
  echo "  $OUTPUT_ROOT"
  printf "自定义本次作业目录（可留空）: "
  read -r custom_output

  command_args=(gemini-local "$audio_value" --model "$model_value")
  if [ -n "$custom_output" ]; then
    custom_output="$(strip_outer_quotes "$custom_output")"
    command_args+=(--output-dir "$custom_output")
  fi
  if [ -n "$api_key_value" ]; then
    GEMINI_API_KEY="$api_key_value" run_logged \
      "本地媒体 -> Gemini OGG/File API 全文转写" \
      asr_worker "${command_args[@]}"
  else
    run_logged \
      "本地媒体 -> Gemini OGG/File API 全文转写" \
      asr_worker "${command_args[@]}"
  fi
}

run_segment_markdown() {
  supplied_markdown="${1:-}"
  segmentation_mode="${2:-hybrid}"
  if [ -n "$supplied_markdown" ]; then
    markdown_value="$supplied_markdown"
  else
    echo "输入 ASR 文稿路径（支持 .md 或 .txt）："
    printf "> "
    read -r markdown_value
  fi
  markdown_value="$(strip_outer_quotes "$markdown_value")"
  if [ -z "$markdown_value" ]; then
    echo "未输入 Markdown 文稿路径。"
    return 2
  fi
  case "$segmentation_mode" in
    pure)
      output_suffix="01-pure-qwen35-4b-rich"
      job_label="纯 Qwen3.5-4B -> 富 Markdown 全文规整"
      ;;
    tfidf)
      output_suffix="02-tfidf-rag-qwen35-4b-rich"
      job_label="TF-IDF RAG -> Qwen3.5-4B 富 Markdown 全文规整"
      ;;
    ollama)
      output_suffix="03-ollama-embedding-rag-qwen35-4b-rich"
      job_label="Ollama embedding RAG -> Qwen3.5-4B 富 Markdown 全文规整"
      ;;
    hybrid|*)
      output_suffix="04-hybrid-rag-qwen35-4b-rich"
      job_label="Hybrid RAG -> Qwen3.5-4B 富 Markdown 全文规整"
      segmentation_mode="hybrid"
      ;;
  esac
  input_dir="$(dirname "$markdown_value")"
  input_base="$(basename "$markdown_value")"
  input_stem="${input_base%.*}"
  output_value="$input_dir/$input_stem-$output_suffix.md"
  echo
  echo "默认分段文稿保存路径："
  echo "  $output_value"
  printf "自定义本次分段文稿路径（可留空）: "
  read -r custom_output

  command_args=(segment-md "$markdown_value")
  case "$segmentation_mode" in
    pure)
      command_args+=(--no-rag)
      ;;
    tfidf)
      command_args+=(--embedding-backend tfidf)
      ;;
    ollama)
      command_args+=(--embedding-backend ollama)
      ;;
    hybrid)
      command_args+=(--embedding-backend hybrid)
      ;;
  esac
  if [ -n "$custom_output" ]; then
    custom_output="$(strip_outer_quotes "$custom_output")"
    command_args+=(--output "$custom_output")
  else
    command_args+=(--output "$output_value")
  fi
  run_logged \
    "$job_label" \
    asr_worker "${command_args[@]}"
}

run_segment_gemini() {
  supplied_markdown="${1:-}"
  if [ -n "$supplied_markdown" ]; then
    markdown_value="$supplied_markdown"
  else
    echo "输入 ASR 文稿路径（支持 .md 或 .txt）："
    printf "> "
    read -r markdown_value
  fi
  markdown_value="$(strip_outer_quotes "$markdown_value")"
  if [ -z "$markdown_value" ]; then
    echo "未输入 ASR 文稿路径。"
    return 2
  fi

  choose_gemini_model
  model_value="$GEMINI_MODEL_VALUE"
  prompt_gemini_key
  api_key_value="$GEMINI_API_KEY_VALUE"

  model_slug="$(printf "%s" "$model_value" | tr '/:' '--')"
  input_dir="$(dirname "$markdown_value")"
  input_base="$(basename "$markdown_value")"
  input_stem="${input_base%.*}"
  output_value="$input_dir/$input_stem-05-gemini-$model_slug-rich.md"

  echo
  echo "默认 Gemini 富 Markdown 文稿保存路径："
  echo "  $output_value"
  printf "自定义本次 Gemini 文稿路径（可留空）: "
  read -r custom_output

  command_args=(segment-gemini "$markdown_value" --model "$model_value")
  if [ -n "$custom_output" ]; then
    custom_output="$(strip_outer_quotes "$custom_output")"
    command_args+=(--output "$custom_output")
  else
    command_args+=(--output "$output_value")
  fi
  if [ -n "$api_key_value" ]; then
    GEMINI_API_KEY="$api_key_value" run_logged \
      "Gemini $model_value -> 富 Markdown 全文规整" \
      asr_worker "${command_args[@]}"
  else
    run_logged \
      "Gemini $model_value -> 富 Markdown 全文规整" \
      asr_worker "${command_args[@]}"
  fi
}

run_nuc_local_asr() {
  supplied_media="${1:-}"
  if [ -n "$supplied_media" ]; then
    audio_value="$supplied_media"
  else
    echo "输入本地音频/视频媒体文件路径（支持 OGG、WebM、MP4、MKV 等，只提取音频流）："
    printf "> "
    read -r audio_value
  fi
  audio_value="$(strip_outer_quotes "$audio_value")"
  if [ -z "$audio_value" ]; then
    echo "未输入本地媒体路径。"
    return 2
  fi

  echo
  echo "NUC ASR 模式："
  echo "  1) Qwen3-ASR 1.7B：文字准确优先；时间轴仅为伪时间轴"
  echo "  2) faster-whisper large-v3-turbo：词级时间轴精度优先（推荐作时间依据）"
  echo "  3) 两者依次运行：同时保留全文与词级时间轴"
  printf "> [2] "
  read -r backend_choice
  case "${backend_choice:-2}" in
    1|qwen) backend="qwen" ;;
    3|both) backend="both" ;;
    *) backend="whisper" ;;
  esac

  echo
  printf "同时运行 Gemini OGG/File API 全文 ASR 作为文字覆盖对照？[y/N]: "
  read -r gemini_choice

  echo
  echo "产物目录留空时，将按媒体文件名保存到："
  echo "  $OUTPUT_ROOT"
  printf "自定义本次作业目录（可留空）: "
  read -r custom_output

  command_args=(nuc-local "$audio_value" --backend "$backend")
  case "${gemini_choice:-N}" in
    y|Y|yes|YES)
      choose_gemini_model
      model_value="$GEMINI_MODEL_VALUE"
      prompt_gemini_key
      api_key_value="$GEMINI_API_KEY_VALUE"
      command_args+=(--gemini-asr --gemini-model "$model_value")
      ;;
    *) api_key_value="" ;;
  esac
  if [ -n "$custom_output" ]; then
    custom_output="$(strip_outer_quotes "$custom_output")"
    command_args+=(--output-dir "$custom_output")
  fi
  if [ -n "$api_key_value" ]; then
    GEMINI_API_KEY="$api_key_value" run_logged "本地媒体 -> NUC ASR" asr_worker "${command_args[@]}"
  else
    run_logged "本地媒体 -> NUC ASR" asr_worker "${command_args[@]}"
  fi
}

run_pipeline() {
  supplied_source="${1:-}"
  if [ -n "$supplied_source" ]; then
    source_value="$supplied_source"
  else
    echo "输入公开 YouTube URL 或本地媒体路径："
    printf "> "
    read -r source_value
  fi
  if [ -z "$source_value" ]; then
    echo "未输入视频来源。"
    return 2
  fi

  echo
  printf "允许 yt-dlp 读取 Chrome Cookie？[y/N]: "
  read -r cookie_choice
  cookie_choice="${cookie_choice:-N}"
  chrome_profile="${FORENSIC_CHROME_PROFILE:-Default}"
  case "$cookie_choice" in
    y|Y|yes|YES)
      printf "Chrome Profile 名称或路径 [%s]: " "$chrome_profile"
      read -r requested_chrome_profile
      chrome_profile="${requested_chrome_profile:-$chrome_profile}"
      ;;
  esac

  echo
  echo "OCR 模式："
  echo "  1) auto：先下载低清短片段预检（推荐）"
  echo "  2) on：已确认有硬字幕，直接进入争议 OCR"
  echo "  3) off：完全跳过 OCR"
  printf "> [1] "
  read -r ocr_choice
  case "${ocr_choice:-1}" in
    2|on) ocr_mode="on" ;;
    3|off) ocr_mode="off" ;;
    *) ocr_mode="auto" ;;
  esac

  echo
  echo "产物目录留空时，按视频标题保存到："
  echo "  $OUTPUT_ROOT"
  printf "自定义本次作业目录（可留空）: "
  read -r custom_output

  command_args=(run "$source_value" --ocr "$ocr_mode")
  case "$cookie_choice" in
    y|Y|yes|YES)
      command_args+=(--cookies-from-chrome --chrome-profile "$chrome_profile")
      ;;
  esac
  if [ -n "$custom_output" ]; then
    command_args+=(--output-dir "$custom_output")
  fi

  run_logged \
    "开始运行完整取证流水线；同一 URL 与目录再次运行会自动续跑。" \
    python_worker "${command_args[@]}"
}

show_status() {
  python_worker status
}

doctor() {
  echo "Python: $PYTHON_BIN"
  if [ -z "$PYTHON_BIN" ] || [ ! -x "$PYTHON_BIN" ]; then
    echo "MISSING Python runtime"
    return 1
  fi
  python_worker doctor
}

open_outputs() {
  mkdir -p "$OUTPUT_ROOT"
  open "$OUTPUT_ROOT"
}

tail_log() {
  if [ -f "$LAST_LOG" ]; then
    tail -n 200 "$LAST_LOG"
  else
    echo "暂无流水线日志：$LAST_LOG"
  fi
}

interactive_menu() {
  while true; do
    clear 2>/dev/null || true
    cat <<'EOF'
============================================================
  Whisper Captioner · 视频字幕取证流水线
============================================================

  1) Gemini URL -> yt-dlp 下载音频 -> OGG/Opus -> Gemini 全文转写
  2) Gemini URL -> 直接交给 Gemini（不下载音频）
  3) YouTube/Bilibili 链接 -> 检测/下载视频自带字幕
  4) 本地音频/视频媒体 -> Gemini OGG/File API 全文转写
  5) ASR 文稿 -> 纯 Qwen3.5-4B 富 Markdown 规整
  6) ASR 文稿 -> TF-IDF RAG + Qwen3.5-4B 富 Markdown 规整
  7) ASR 文稿 -> Ollama Embedding RAG + Qwen3.5-4B 富 Markdown 规整
  8) ASR 文稿 -> Hybrid RAG + Qwen3.5-4B 富 Markdown 规整
  9) ASR 文稿 -> Gemini 2.5 Flash/Pro 富 Markdown 规整
  10) 本地音频/视频媒体 -> NUC ASR（Qwen / large-v3-turbo）
  11) 一键运行 / 续跑完整取证 Pipeline
  12) 查看近期作业状态
  13) 打开产物目录
  14) 查看最近日志
  15) 环境诊断
  h) 查看 Pipeline 说明
  q) 退出
EOF
    echo
    printf "> "
    read -r choice || exit 0
    set +e
    case "$choice" in
      1) run_gemini_url_asr; action_status=$? ;;
      2) run_gemini_url_asr "" 1; action_status=$? ;;
      3) run_native_subtitles; action_status=$? ;;
      4) run_gemini_local_asr; action_status=$? ;;
      5) run_segment_markdown "" pure; action_status=$? ;;
      6) run_segment_markdown "" tfidf; action_status=$? ;;
      7) run_segment_markdown "" ollama; action_status=$? ;;
      8) run_segment_markdown "" hybrid; action_status=$? ;;
      9) run_segment_gemini; action_status=$? ;;
      10) run_nuc_local_asr; action_status=$? ;;
      11) run_pipeline; action_status=$? ;;
      12) show_status; action_status=$? ;;
      13) open_outputs; action_status=$? ;;
      14) tail_log; action_status=$? ;;
      15) doctor; action_status=$? ;;
      h|H) less "$SCRIPT_ROOT/docs/final_forensic_subtitle_pipeline.md"; action_status=$? ;;
      q|Q) exit 0 ;;
      *) echo "未知选项：$choice"; action_status=2 ;;
    esac
    set -e
    if [ "$action_status" -ne 0 ]; then
      echo
      echo "操作结束，状态码：$action_status"
    fi
    pause_menu
  done
}

command_name="${1:-menu}"
case "$command_name" in
  menu) interactive_menu ;;
  gemini-url) shift; run_gemini_url_asr "${1:-}" ;;
  gemini-url-direct) shift; run_gemini_url_asr "${1:-}" 1 ;;
  native-subtitles) shift; run_native_subtitles "${1:-}" ;;
  gemini-local) shift; run_gemini_local_asr "${1:-}" ;;
  segment-md) shift; run_segment_markdown "${1:-}" hybrid ;;
  segment-pure) shift; run_segment_markdown "${1:-}" pure ;;
  segment-tfidf) shift; run_segment_markdown "${1:-}" tfidf ;;
  segment-ollama) shift; run_segment_markdown "${1:-}" ollama ;;
  segment-hybrid) shift; run_segment_markdown "${1:-}" hybrid ;;
  segment-gemini) shift; run_segment_gemini "${1:-}" ;;
  nuc-local) shift; run_nuc_local_asr "${1:-}" ;;
  run) shift; run_pipeline "${1:-}" ;;
  status) show_status ;;
  doctor) doctor ;;
  outputs) open_outputs ;;
  tail) tail_log ;;
  -h|--help|help) usage ;;
  *)
    echo "未知命令：$command_name"
    echo
    usage
    exit 2
    ;;
esac
