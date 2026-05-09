from __future__ import annotations

import bisect
import json
import re
import subprocess
import shutil
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse

from PySide6.QtCore import QSettings, QThread, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QSystemTrayIcon,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from whisper_captioner.cache import cache_slug, canonical_media_url, validate_url_for_yt_dlp
from whisper_captioner.chrome_control import (
    chrome_current_time,
    chrome_current_time_url,
    chrome_get_url,
    chrome_pause,
    chrome_pause_url,
    chrome_play_from,
    chrome_play_url_from,
    chrome_resume,
    chrome_resume_url,
    chrome_seek_relative,
    chrome_seek_url_relative,
    chrome_toggle_playback,
)
from whisper_captioner.llm_handler import (
    llm_provider_ready,
    test_llm_connection,
)
from whisper_captioner.overlay import SubtitleOverlay
from whisper_captioner.qwen_chat_service import QwenChatServiceManager
from whisper_captioner.ui_builder import build_main_window_ui
from whisper_captioner.ui_shell import WINDOW_STYLESHEET
from whisper_captioner.workers import (
    LLMTextWorker,
    NUCRealtimeWorker,
    QueueWorker,
    RealtimeWorker,
    RealtimePolishWorker,
    RealtimeReRecognizeWorker,
    RollingPrefetchWorker,
    clean_title_for_filename,
    infer_source_title,
    source_output_dir,
)
from whisper_captioner.config import (
    BUFFER_PAUSE_MARGIN,
    BUFFER_RESUME_MARGIN,
    CACHE_DIR,
    DEFAULT_SUBTITLE_OFFSET,
    FFMPEG,
    LOG_DIR,
    NOTES_DIR,
    WHISPER_STREAM,
    MODELS_DIR,
    OUTPUT_DIR,
    REALTIME_DIR,
    SUBTITLE_PIPELINE_VERSION,
)
from whisper_captioner.models import CaptionMode, LLM_PROVIDERS, MODES, SubtitleSegment
from whisper_captioner.subtitle_io import (
    load_segments,
    overlapping_segments,
    parse_srt,
    save_segments,
    save_segments_as_srt,
    save_segments_as_txt,
)

LOG_LEVELS = {
    "quiet": 0,
    "normal": 1,
    "debug": 2,
    "trace": 3,
}

class MainWindow(QMainWindow):
    def __init__(self, overlay: SubtitleOverlay) -> None:
        super().__init__()
        self.overlay = overlay
        self.setWindowTitle("Whisper Captioner")
        self.resize(1000, 680)
        self.realtime_thread: Optional[QThread] = None
        self.realtime_worker: Optional[Union[RealtimeWorker, NUCRealtimeWorker]] = None
        self.realtime_polish_thread: Optional[QThread] = None
        self.realtime_polish_worker: Optional[RealtimePolishWorker] = None
        self.realtime_re_recognize_thread: Optional[QThread] = None
        self.realtime_re_recognize_worker: Optional[RealtimeReRecognizeWorker] = None
        
        # Audio capturing and control state
        self.queue_thread: Optional[QThread] = None
        self.queue_worker: Optional[QueueWorker] = None
        self.controlled_thread: Optional[QThread] = None
        self.controlled_worker: Optional[RollingPrefetchWorker] = None
        self.llm_text_thread: Optional[QThread] = None
        self.llm_text_worker: Optional[LLMTextWorker] = None
        self.controlled_segments: list[SubtitleSegment] = []
        self._controlled_segment_starts: list[float] = []
        self.controlled_timer = QTimer(self)
        self.controlled_timer.setInterval(250)
        self.controlled_timer.timeout.connect(self._tick_controlled_captions)
        self._rolling_all_done = False
        self._buffering_paused = False
        self._shutting_down = False
        self._last_auto_url = ""
        self._controlled_url = ""
        self._controlled_paused = False
        self._controlled_resume_time = 0.0
        self._controlled_cache_dir: Optional[Path] = None
        self._latest_export_base: Optional[Path] = None
        self._current_caption_index = -1
        self._last_caption_debug_second = -1
        self._last_chrome_probe_monotonic = 0.0
        self._last_chrome_probe_time: Optional[float] = None
        self._playback_anchor_monotonic = 0.0
        self._playback_anchor_time = 0.0
        self.subtitle_offset = DEFAULT_SUBTITLE_OFFSET
        self._log_level = LOG_LEVELS["normal"]
        self._log_file_path: Optional[Path] = None
        self._pending_ui_logs: list[str] = []
        self._pending_file_logs: list[str] = []
        self._max_status_lines = 1200
        self._status_summary = "就绪"
        self._current_progress: tuple[int, int] | None = None
        self.qwen_chat_service = QwenChatServiceManager()

        self._log_flush_timer = QTimer(self)
        self._log_flush_timer.setInterval(120)
        self._log_flush_timer.timeout.connect(self._flush_ui_logs)
        self._log_flush_timer.start()
        self._file_log_flush_timer = QTimer(self)
        self._file_log_flush_timer.setInterval(500)
        self._file_log_flush_timer.timeout.connect(self._flush_file_logs)
        self._file_log_flush_timer.start()
        self._apply_window_style()
        build_main_window_ui(self)
        self.llm_provider_combo.currentIndexChanged.connect(self._on_llm_provider_changed)
        self.llm_test_button.clicked.connect(self._test_llm)
        self.qwen_chat_start_button.clicked.connect(self.start_qwen_chat_service)
        self.qwen_chat_open_button.clicked.connect(self.open_qwen_chat)
        self.stop_button.clicked.connect(self.stop_all)
        self.add_button.clicked.connect(self.add_queue_item)
        self.up_button.clicked.connect(lambda: self.move_item(-1))
        self.down_button.clicked.connect(lambda: self.move_item(1))
        self.process_button.clicked.connect(self.process_queue)
        self.process_local_button.clicked.connect(self.process_local_media)
        self.realtime_button.clicked.connect(self.start_realtime)
        self.list_audio_devices_button.clicked.connect(self.list_audio_input_devices)
        self.capture_id_input.editingFinished.connect(self._save_capture_id)
        self.controlled_button.clicked.connect(self.start_controlled_url)
        self.controlled_button_2.clicked.connect(self.start_controlled_url)
        self.overlay_show_button.clicked.connect(self.overlay.show)
        self.overlay_font_button.clicked.connect(self.overlay._choose_font)
        self.overlay_bigger_button.clicked.connect(lambda: self.overlay._adjust_font_size(2))
        self.overlay_smaller_button.clicked.connect(lambda: self.overlay._adjust_font_size(-2))
        self.overlay_more_opacity_button.clicked.connect(lambda: self.overlay.adjust_opacity(0.05))
        self.overlay_less_opacity_button.clicked.connect(lambda: self.overlay.adjust_opacity(-0.05))
        self.overlay_reset_button.clicked.connect(self.overlay._reset_position)

        # Realtime Session UI Connections
        self.session_list.itemSelectionChanged.connect(self._on_session_selected)
        self.session_view_combo.currentIndexChanged.connect(self._on_session_selected)
        self.session_polish_button.clicked.connect(self._polish_session)
        self.session_rerecognize_button.clicked.connect(self._re_recognize_session)
        self.session_open_button.clicked.connect(self._open_session_dir)
        self.session_delete_button.clicked.connect(self._delete_session)

        QTimer.singleShot(100, self._load_realtime_sessions)
        
        # Initial status update
        self.clear_cache_button.clicked.connect(self.clear_current_video_cache)
        self.open_cache_button.clicked.connect(self.open_current_cache_in_finder)
        self.open_outputs_button.clicked.connect(self.reveal_current_outputs_in_finder)
        self.rewind_5_button.clicked.connect(self.rewind_5s)
        self.rewind_button.clicked.connect(self.rewind_10s)
        self.play_pause_button.clicked.connect(self.toggle_playback)
        self.forward_5_button.clicked.connect(self.forward_5s)
        self.forward_button.clicked.connect(self.forward_10s)
        self.subtitle_earlier_button.clicked.connect(lambda: self.adjust_subtitle_offset(-0.5))
        self.subtitle_later_button.clicked.connect(lambda: self.adjust_subtitle_offset(0.5))
        self.subtitle_sync_button.clicked.connect(self.sync_current_subtitle_line)
        self.summarize_button.clicked.connect(lambda: self.start_llm_text_task("summary"))
        self.article_button.clicked.connect(lambda: self.start_llm_text_task("article"))
        self.ask_button.clicked.connect(lambda: self.start_llm_text_task("qa"))
        self.transcript_list.itemClicked.connect(self.jump_to_subtitle_index)
        self.overlay.rewind_5_requested.connect(self.rewind_5s)
        self.overlay.rewind_requested.connect(self.rewind_10s)
        self.overlay.play_pause_requested.connect(self.toggle_playback)
        self.overlay.forward_5_requested.connect(self.forward_5s)
        self.overlay.forward_requested.connect(self.forward_10s)
        self._load_settings()
        self._init_log_file()

    def _load_settings(self) -> None:
        settings = QSettings("WhisperCaptioner", "App")
        saved_mode = str(settings.value("mode/key", "hq_turbo"))
        mode_idx = self.mode_combo.findData(saved_mode)
        if mode_idx >= 0:
            self.mode_combo.setCurrentIndex(mode_idx)
        elif self.mode_combo.count():
            self.mode_combo.setCurrentIndex(0)
        self.llm_group.setChecked(settings.value("llm/enabled", True, type=bool))
        saved_provider = str(settings.value("llm/provider", "gemini_flash"))
        idx = self.llm_provider_combo.findData(saved_provider)
        if idx >= 0:
            self.llm_provider_combo.setCurrentIndex(idx)
        saved_log_level = str(settings.value("log/level", "normal"))
        log_idx = self.log_level_combo.findData(saved_log_level)
        if log_idx >= 0:
            self.log_level_combo.setCurrentIndex(log_idx)
        self.capture_id_input.setText(str(settings.value("audio/capture_id", 0, type=int)))
        self._on_log_level_changed()
        self._on_llm_provider_changed()

    def _apply_window_style(self) -> None:
        self.setStyleSheet(WINDOW_STYLESHEET)

    def _on_llm_provider_changed(self) -> None:
        key = self.llm_provider_combo.currentData()
        provider = next(p for p in LLM_PROVIDERS if p.key == key)
        settings = QSettings("WhisperCaptioner", "App")
        self.llm_api_key_input.setText(settings.value(f"llm/apikey/{key}", ""))
        is_custom = (key == "custom")
        self.llm_custom_url_input.setVisible(is_custom)
        self.llm_custom_model_input.setVisible(is_custom)
        self.llm_api_key_input.setVisible(provider.requires_api_key)
        if is_custom:
            self.llm_custom_url_input.setText(settings.value("llm/custom_url", ""))
            self.llm_custom_model_input.setText(settings.value("llm/custom_model", ""))
        settings.setValue("llm/provider", key)

    def _save_current_llm_key(self) -> None:
        key = self.llm_provider_combo.currentData()
        settings = QSettings("WhisperCaptioner", "App")
        settings.setValue("llm/enabled", self.llm_group.isChecked())
        settings.setValue(f"llm/apikey/{key}", self.llm_api_key_input.text())
        if key == "custom":
            settings.setValue("llm/custom_url", self.llm_custom_url_input.text().strip())
            settings.setValue("llm/custom_model", self.llm_custom_model_input.text().strip())

    def _save_mode_selection(self) -> None:
        settings = QSettings("WhisperCaptioner", "App")
        settings.setValue("mode/key", self.mode_combo.currentData())

    def _save_capture_id(self, *_args) -> None:
        settings = QSettings("WhisperCaptioner", "App")
        settings.setValue("audio/capture_id", self._capture_id())

    def _capture_id(self) -> int:
        text = self.capture_id_input.text().strip()
        match = re.match(r"^-?\d+", text)
        if match:
            return int(match.group(0))
        return 0

    def _init_log_file(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self._log_file_path = LOG_DIR / f"whisper-captioner-{stamp}.log"
        self._write_log_line("normal", f"Log file started: {self._log_file_path}")

    def _write_log_line(self, level_key: str, message: str) -> None:
        if not self._log_file_path:
            return
        line = f"[{time.strftime('%H:%M:%S')}] [{level_key.upper()}] {message}\n"
        self._pending_file_logs.append(line)

    def _on_log_level_changed(self) -> None:
        key = self.log_level_combo.currentData()
        self._log_level = LOG_LEVELS.get(str(key), LOG_LEVELS["normal"])
        settings = QSettings("WhisperCaptioner", "App")
        settings.setValue("log/level", key)
        self._write_log_line("normal", f"Log level set to {key}")

    def _test_llm(self) -> None:
        self._save_current_llm_key()
        key = self.llm_provider_combo.currentData()
        provider = next(p for p in LLM_PROVIDERS if p.key == key)
        api_key = self.llm_api_key_input.text().strip()
        if provider.requires_api_key and not api_key:
            QMessageBox.warning(self, "测试", "请输入 API key。")
            return
        self.llm_test_button.setEnabled(False)
        self.log(f"Testing connection to {provider.label}...")
        
        url_override = self.llm_custom_url_input.text().strip() if key == "custom" else ""
        model_override = self.llm_custom_model_input.text().strip() if key == "custom" else ""

        ok, msg = test_llm_connection(provider, api_key, url_override, model_override)
        if ok:
            QMessageBox.information(self, "测试", msg)
            self.log("LLM API test passed.")
        else:
            QMessageBox.critical(self, "测试失败", msg)
            self.log(f"LLM API test failed: {msg}")
        self.llm_test_button.setEnabled(True)

    def start_qwen_chat_service(self) -> None:
        try:
            url = self.qwen_chat_service.start()
        except Exception as exc:
            self.log(f"Qwen3-8B chat service failed to start: {exc}")
            QMessageBox.critical(self, "Qwen3-8B 聊天", f"启动本地聊天服务失败：\n\n{exc}")
            return
        self.log(f"Qwen3-8B chat service ready at {url}")
        QMessageBox.information(
            self,
            "Qwen3-8B 聊天",
            f"本地聊天服务已启动：\n\n{url}\n\n这条链路与当前转录功能无关，可单独测试 8B 模型能力。",
        )

    def open_qwen_chat(self) -> None:
        try:
            url = self.qwen_chat_service.start()
        except Exception as exc:
            self.log(f"Qwen3-8B chat service failed to start: {exc}")
            QMessageBox.critical(self, "Qwen3-8B 聊天", f"启动本地聊天服务失败：\n\n{exc}")
            return
        subprocess.run(["open", url], check=False)
        self.log(f"Opened Qwen3-8B chat at {url}")

    def _current_llm_provider_config(self):
        self._save_current_llm_key()
        key = self.llm_provider_combo.currentData()
        provider = next(p for p in LLM_PROVIDERS if p.key == key)
        api_key = self.llm_api_key_input.text().strip()
        api_url = self.llm_custom_url_input.text().strip() if key == "custom" else ""
        model_id = self.llm_custom_model_input.text().strip() if key == "custom" else ""
        return provider, api_key, api_url, model_id

    def _transcript_for_llm(self) -> str:
        return "\n".join(
            f"[{self._format_seconds(segment.start)} - {self._format_seconds(segment.end)}] {segment.text}"
            for segment in self.controlled_segments
            if segment.text.strip()
        )

    def _note_base_name(self) -> str:
        source = self.url_input.text().strip() or self._controlled_url
        if source:
            return clean_title_for_filename(infer_source_title(source))
        if self._controlled_cache_dir:
            return self._controlled_cache_dir.name
        return time.strftime("%Y%m%d-%H%M%S")

    def _video_output_dir(self) -> Path:
        source = self.url_input.text().strip() or self._controlled_url or self._note_base_name()
        return source_output_dir(NOTES_DIR.parent, clean_title_for_filename(infer_source_title(source)))

    def _save_shared_note_copy(self, task_key: str, text: str) -> Optional[Path]:
        note_dir = self._video_output_dir()
        suffix = {
            "summary": "总结分析.md",
            "article": "改写文章.md",
            "qa": "字幕问答.md",
        }.get(task_key, f"{task_key}.md")
        model_suffix = self._current_output_variant_suffix()
        path = note_dir / f"{self._note_base_name()}-{model_suffix}-{suffix}"
        path.write_text(text, encoding="utf-8")
        return path

    def _reveal_in_finder(self, path: Path) -> None:
        subprocess.run(["open", "-R", str(path)], check=False)

    def _current_output_files(self) -> list[Path]:
        files: list[Path] = []
        if self._latest_export_base:
            for suffix in (".srt", ".txt"):
                candidate = self._latest_export_base.with_suffix(suffix)
                if candidate.exists():
                    files.append(candidate)
        for task_key in ("summary", "article", "qa"):
            cache_path = self._postprocess_output_path(task_key)
            if cache_path and cache_path.exists():
                files.append(cache_path)
            suffix = {
                "summary": "总结分析.md",
                "article": "改写文章.md",
                "qa": "字幕问答.md",
            }[task_key]
            shared = self._video_output_dir() / f"{self._note_base_name()}-{suffix}"
            if shared.exists():
                files.append(shared)
        seen: set[str] = set()
        unique_files: list[Path] = []
        for path in files:
            key = str(path)
            if key not in seen:
                seen.add(key)
                unique_files.append(path)
        return unique_files

    def _subtitle_windows_for_rag(self, window_size: int = 6, stride: int = 3) -> list[tuple[tuple[float, float], str]]:
        windows: list[tuple[tuple[float, float], str]] = []
        segments = [segment for segment in self.controlled_segments if segment.text.strip()]
        if not segments:
            return windows
        for i in range(0, len(segments), stride):
            chunk = segments[i : i + window_size]
            if not chunk:
                continue
            start = chunk[0].start
            end = chunk[-1].end
            text = " ".join(segment.text for segment in chunk)
            windows.append(((start, end), text))
        return windows

    def _rag_candidates(self, question: str, top_k: int = 4) -> list[tuple[tuple[float, float], str]]:
        tokens = [token.lower() for token in question.replace("/", " ").replace("-", " ").split() if token.strip()]
        scored: list[tuple[int, tuple[float, float], str]] = []
        for span, text in self._subtitle_windows_for_rag():
            lowered = text.lower()
            score = sum(lowered.count(token) for token in tokens)
            if score or not tokens:
                scored.append((score, span, text))
        scored.sort(key=lambda item: (item[0], item[1][0]), reverse=True)
        best = [(span, text) for score, span, text in scored[:top_k]]
        if best:
            return best
        return self._subtitle_windows_for_rag(window_size=4, stride=4)[:top_k]

    def _rag_context_text(self, question: str) -> str:
        snippets = []
        for index, (span, text) in enumerate(self._rag_candidates(question), 1):
            snippets.append(
                f"[证据{index} {self._format_seconds(span[0])}-{self._format_seconds(span[1])}] {text}"
            )
        return "\n".join(snippets)

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        total = max(0, int(seconds))
        return f"{total // 60:02d}:{total % 60:02d}"

    def _format_transcript_timestamp(self, seconds: float) -> str:
        millis_total = max(0, int(round(seconds * 1000)))
        secs = (millis_total // 1000) % 60
        minutes_total = millis_total // 60000
        mins = minutes_total % 60
        hours = minutes_total // 60
        return f"{hours:02d}:{mins:02d}:{secs:02d}"

    def _refresh_transcript_list(self) -> None:
        if not hasattr(self, "transcript_list"):
            return
        self.transcript_list.clear()
        for index, segment in enumerate(self.controlled_segments):
            item = QListWidgetItem(
                f"[{self._format_transcript_timestamp(segment.start)} - {self._format_transcript_timestamp(segment.end)}] {segment.text}"
            )
            item.setData(256, index)
            self.transcript_list.addItem(item)

    def _set_controlled_segments(self, segments: list[SubtitleSegment]) -> None:
        self.controlled_segments = list(segments)
        self._controlled_segment_starts = [segment.start for segment in self.controlled_segments]

    def _extend_controlled_segments(self, segments: list[SubtitleSegment]) -> None:
        self.controlled_segments.extend(segments)
        self._controlled_segment_starts.extend(segment.start for segment in segments)

    def jump_to_subtitle_index(self, item: QListWidgetItem) -> None:
        index = item.data(256)
        if index is None or not (0 <= int(index) < len(self.controlled_segments)):
            return
        segment = self.controlled_segments[int(index)]
        target = max(0.0, segment.start - self.subtitle_offset)
        ok = False
        if self._controlled_url:
            ok = chrome_play_url_from(self._controlled_url, target)
        if not ok:
            chrome_play_from(target)
        self._controlled_resume_time = target
        self._set_playback_anchor(target)
        self._controlled_paused = False
        self._buffering_paused = False
        self.controlled_timer.start()
        self._tick_controlled_captions()

    def start_llm_text_task(self, task_key: str) -> None:
        if self.llm_text_thread:
            self.log("LLM post-process is already running")
            return
        if not self.controlled_segments:
            QMessageBox.information(
                self,
                "还没有字幕文本",
                "请先运行“网址受控字幕”，让程序先生成当前视频的完整字幕。",
            )
            return

        provider, api_key, api_url, model_id = self._current_llm_provider_config()
        if not llm_provider_ready(provider, api_key):
            QMessageBox.warning(self, "需要 LLM", "请先在“设置”里启用并配置一个 LLM 提供商。")
            return

        transcript = self._transcript_for_llm()
        cached_path = self._postprocess_output_path(task_key)
        if task_key != "qa" and cached_path and cached_path.exists():
            cached_text = cached_path.read_text(encoding="utf-8", errors="ignore")
            self.analysis_output.setPlainText(cached_text)
            self._save_shared_note_copy(task_key, cached_text)
            self.log(f"Loaded cached LLM {task_key} output: {cached_path}")
            return

        if task_key == "qa":
            question = self.analysis_question_input.text().strip()
            if not question:
                QMessageBox.information(self, "需要问题", "请先输入一个基于当前字幕的问题。")
                return
            title = "字幕问答"
            context_text = self._rag_context_text(question)
            self.analysis_context_output.setPlainText(context_text)
            system_prompt = "你是一位严谨的字幕内容研究助手。请严格依据提供的字幕证据作答，不要编造字幕中没有的信息。"
            user_text = (
                f"用户问题：{question}\n\n"
                "下面是从当前字幕中检索出来的相关证据片段，请优先依据这些证据回答；若证据不足，请明确说明。\n\n"
                f"{context_text}\n\n"
                "输出要求：\n"
                "1. 先直接回答问题。\n"
                "2. 再给出依据了哪些证据片段。\n"
                "3. 如果字幕证据不足，请明确说“证据不足”，不要脑补。\n"
            )
            max_tokens = 8000
        elif task_key == "article":
            title = "完整文章"
            system_prompt = "你是一位中文长文编辑，擅长把视频转写稿整理为可阅读的完整文章。"
            user_text = (
                "请把下面的视频字幕转写稿改写成一篇完整、自然、结构清晰的中文文章。\n"
                "要求：\n"
                "1. 不要逐句罗列字幕，不要保留字幕编号。\n"
                "2. 保留原视频的核心观点、术语、例子和逻辑顺序。\n"
                "3. 可以合并重复口语、修正病句、补足自然衔接，但不要虚构原文没有的信息。\n"
                "4. 使用标题、小标题和自然段。\n\n"
                f"字幕转写稿：\n{transcript}"
            )
            max_tokens = 24000
        else:
            title = "视频总结与分析"
            system_prompt = "你是一位严谨的视频内容分析助手，擅长从转写稿中提炼结构、观点和可执行洞察。"
            user_text = (
                "请基于下面的视频字幕转写稿输出一份中文总结与分析。\n"
                "要求：\n"
                "1. 先给出 5-10 条要点摘要。\n"
                "2. 再给出视频结构梳理，尽量引用时间点。\n"
                "3. 分析作者/讲者的核心论点、论证方式、重要例子和潜在不足。\n"
                "4. 最后给出适合复习的关键词和一句话结论。\n"
                "5. 不要虚构字幕里没有的信息。\n\n"
                f"字幕转写稿：\n{transcript}"
            )
            max_tokens = 16000

        self.analysis_output.setPlainText(f"{title}生成中...")
        self.summarize_button.setEnabled(False)
        self.article_button.setEnabled(False)
        self.llm_text_thread = QThread()
        self.llm_text_worker = LLMTextWorker(
            task_key,
            user_text,
            provider,
            api_key,
            api_url,
            model_id,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
        )
        self.llm_text_worker.moveToThread(self.llm_text_thread)
        self.llm_text_thread.started.connect(self.llm_text_worker.run)
        self.llm_text_worker.status.connect(self.log)
        self.llm_text_worker.result.connect(self._handle_llm_text_result)
        self.llm_text_worker.finished.connect(self.llm_text_thread.quit)
        self.llm_text_thread.finished.connect(self._clear_llm_text_worker)
        self.llm_text_thread.start()

    def _handle_llm_text_result(self, task_key: str, text: str) -> None:
        self.analysis_output.setPlainText(text)
        path = self._postprocess_output_path(task_key)
        if path:
            path.write_text(text, encoding="utf-8")
            self.log(f"Saved LLM {task_key} output: {path}")
        shared_path = self._save_shared_note_copy(task_key, text)
        if shared_path:
            self.log(f"Saved shared note copy for {task_key}: {shared_path}")

    def _postprocess_output_path(self, task_key: str) -> Optional[Path]:
        if not self._controlled_cache_dir:
            return None
        name = {
            "summary": "总结分析.md",
            "article": "改写文章.md",
            "qa": "字幕问答.md",
        }.get(task_key, f"video-{task_key}.md")
        return self._controlled_cache_dir / f"{self._current_output_variant_suffix()}-{name}"

    def _current_output_variant_suffix(self) -> str:
        mode = self.current_mode()
        parts = [clean_title_for_filename(mode.key, fallback="mode")]
        if self.llm_group.isChecked():
            key = self.llm_provider_combo.currentData()
            provider = clean_title_for_filename(str(key), fallback="llm")
            parts.append(provider)
            if key == "custom":
                custom_model = clean_title_for_filename(self.llm_custom_model_input.text().strip(), fallback="custom")
                parts.append(custom_model)
        else:
            parts.append("raw")
        return "-".join(parts)

    def _resolved_cache_dir_for_current_mode(self) -> Optional[Path]:
        source = self.url_input.text().strip() or self._controlled_url
        if not source.startswith(("http://", "https://")):
            chrome_url = chrome_get_url()
            if chrome_url.startswith(("http://", "https://")):
                source = chrome_url
        if not source.startswith(("http://", "https://")):
            return self._controlled_cache_dir
        mode = self.current_mode()
        canonical = canonical_media_url(source)
        job_key = cache_slug(canonical, mode.backend, mode.model_name, 30)
        return CACHE_DIR / job_key

    def clear_current_video_cache(self) -> None:
        cache_dir = self._resolved_cache_dir_for_current_mode()
        if not cache_dir or not cache_dir.exists():
            QMessageBox.information(self, "缓存", "当前视频 + 当前模式还没有可删除的缓存。")
            return
        label = self.current_mode().label
        answer = QMessageBox.question(
            self,
            "删除缓存",
            f"确认删除当前视频在此模式下的缓存吗？\n\n{label}\n\n{cache_dir}",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        shutil.rmtree(cache_dir, ignore_errors=True)
        if self._controlled_cache_dir and self._controlled_cache_dir == cache_dir:
            self._controlled_cache_dir = None
            self._set_controlled_segments([])
            self._refresh_transcript_list()
            self.analysis_output.clear()
            self.analysis_context_output.clear()
        self.log(f"Deleted current video cache: {cache_dir}")
        self.overlay.set_caption("Current video cache deleted")

    def open_current_cache_in_finder(self) -> None:
        cache_dir = self._resolved_cache_dir_for_current_mode()
        if not cache_dir or not cache_dir.exists():
            QMessageBox.information(self, "缓存", "当前视频 + 当前模式还没有缓存。")
            return
        subprocess.run(["open", str(cache_dir)], check=False)
        self.log(f"Opened current cache in Finder: {cache_dir}")

    def reveal_current_outputs_in_finder(self) -> None:
        files = self._current_output_files()
        if not files:
            QMessageBox.information(
                self,
                "文件",
                "当前视频还没有生成可定位的字幕或文本文件。",
            )
            return
        for path in files:
            self._reveal_in_finder(path)
        self.log(f"Revealed {len(files)} current output file(s) in Finder")

    def current_mode(self) -> CaptionMode:
        key = self.mode_combo.currentData()
        return next(mode for mode in MODES if mode.key == key)

    def log(self, message: str, level: str = "normal") -> None:
        if message.startswith("Final subtitles written: "):
            raw_path = message.removeprefix("Final subtitles written: ").strip()
            path = Path(raw_path)
            self._latest_export_base = path.with_suffix("")
        self._update_status_summary_from_message(message)
        self._write_log_line(level, message)
        self._pending_ui_logs.append(message)
        if hasattr(self, "status_summary"):
            self.status_summary.setText(self._status_summary)

    def _update_status_summary_from_message(self, message: str) -> None:
        message = message.strip()
        if not message:
            return
        progress_suffix = ""
        if self._current_progress:
            current, total = self._current_progress
            progress_suffix = f" | 进度 {current}/{total}"
        key_phrases = (
            "Preparing",
            "Transcribing",
            "Downloading",
            "Loaded",
            "Final subtitles written",
            "Done:",
            "Failed",
            "Realtime route",
            "Listing AVFoundation",
            "Audio duration:",
            "All chunks transcribed",
        )
        if any(message.startswith(prefix) for prefix in key_phrases) or "chunk" in message.lower():
            self._status_summary = f"{message}{progress_suffix}"

    def _flush_ui_logs(self) -> None:
        if not self._pending_ui_logs:
            return
        chunk = self._pending_ui_logs[:40]
        del self._pending_ui_logs[:40]
        self.status.append("\n".join(chunk))
        cursor = self.status.textCursor()
        doc = self.status.document()
        while doc.blockCount() > self._max_status_lines:
            cursor.movePosition(cursor.MoveOperation.Start)
            cursor.select(cursor.SelectionType.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()
            doc = self.status.document()

    def _set_status_summary(self, text: str) -> None:
        self._status_summary = text
        if hasattr(self, "status_summary"):
            self.status_summary.setText(text)

    def _flush_file_logs(self) -> None:
        if not self._pending_file_logs or not self._log_file_path:
            return
        self._log_file_path.parent.mkdir(parents=True, exist_ok=True)
        chunk = "".join(self._pending_file_logs)
        self._pending_file_logs.clear()
        with self._log_file_path.open("a", encoding="utf-8") as fh:
            fh.write(chunk)

    def rewind_5s(self) -> None:
        self.seek_chrome(-5)

    def rewind_10s(self) -> None:
        self.seek_chrome(-10)

    def forward_5s(self) -> None:
        self.seek_chrome(5)

    def forward_10s(self) -> None:
        self.seek_chrome(10)

    def toggle_playback(self) -> None:
        if self.controlled_segments:
            if self._controlled_paused:
                self.resume_controlled_playback()
            else:
                self.pause_controlled_playback()
            return
        playing = chrome_toggle_playback()
        if playing is None:
            self.log("无法切换 Chrome 视频播放状态")
            self.overlay.set_caption("没有找到 Chrome 视频")
            return
        self._buffering_paused = False
        state = "Playing" if playing else "Paused"
        self.log(f"Chrome video: {state.lower()}")
        self.overlay.set_caption(state)

    def pause_controlled_playback(self) -> None:
        current = self._read_precise_chrome_time()
        if current is not None:
            self._controlled_resume_time = current
        if self._controlled_url:
            chrome_pause_url(self._controlled_url)
        else:
            chrome_pause()
        if self.controlled_timer.isActive():
            self.controlled_timer.stop()
        self._controlled_paused = True
        self.overlay.set_caption(f"已暂停：{self._controlled_resume_time:.1f}s")
        self.log(
            f"Controlled captions paused at {self._controlled_resume_time:.1f}s; "
            "Chrome control returned to you."
        )

    def resume_controlled_playback(self) -> None:
        target_url = self._controlled_url or chrome_get_url()
        ok = False
        if target_url:
            ok = chrome_play_url_from(target_url, self._controlled_resume_time)
        if not ok:
            chrome_play_from(self._controlled_resume_time)
        self._controlled_paused = False
        self._buffering_paused = False
        self._set_playback_anchor(self._controlled_resume_time)
        self.controlled_timer.start()
        self.log(f"Controlled captions resumed at {self._controlled_resume_time:.1f}s")
        self._tick_controlled_captions()

    def seek_chrome(self, delta_seconds: float) -> None:
        if self.controlled_segments and self._controlled_url:
            current = chrome_seek_url_relative(self._controlled_url, delta_seconds)
            if current is None:
                current = chrome_seek_relative(delta_seconds)
        else:
            current = chrome_seek_relative(delta_seconds)
        if current is None:
            self.log("无法快进或后退 Chrome 视频")
            self.overlay.set_caption("没有找到 Chrome 视频")
            return
        self._buffering_paused = False
        self._set_playback_anchor(current)
        direction = "back" if delta_seconds < 0 else "forward"
        self.log(f"Seeked Chrome {direction} {abs(delta_seconds):.0f}s -> {current:.1f}s")
        self._tick_controlled_captions()

    def adjust_subtitle_offset(self, delta_seconds: float) -> None:
        self.subtitle_offset += delta_seconds
        self._save_subtitle_offset()
        self.log(f"Subtitle offset: {self.subtitle_offset:+.1f}s")
        self.overlay.set_caption(f"字幕偏移 {self.subtitle_offset:+.1f}s")
        self._tick_controlled_captions()

    def sync_current_subtitle_line(self) -> None:
        current = self._read_precise_chrome_time()
        if current is None or not (0 <= self._current_caption_index < len(self.controlled_segments)):
            self.log("当前无法同步：没有活跃字幕行")
            self.overlay.set_caption("当前没有活跃字幕行")
            return
        segment = self.controlled_segments[self._current_caption_index]
        self.subtitle_offset = segment.start - current
        self._save_subtitle_offset()
        self.log(
            f"Synced current line: video {current:.2f}s -> subtitle {segment.start:.2f}s; "
            f"offset {self.subtitle_offset:+.2f}s"
        )
        self.overlay.set_caption(f"已同步偏移 {self.subtitle_offset:+.2f}s")
        self._tick_controlled_captions()

    def _subtitle_offset_path(self) -> Optional[Path]:
        if not self._controlled_cache_dir:
            return None
        return self._controlled_cache_dir / "subtitle-sync.json"

    def _save_subtitle_offset(self) -> None:
        path = self._subtitle_offset_path()
        if not path:
            return
        payload = {
            "offset": self.subtitle_offset,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_subtitle_offset(self) -> None:
        path = self._subtitle_offset_path()
        self.subtitle_offset = DEFAULT_SUBTITLE_OFFSET
        if not path or not path.exists():
            return
        try:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"字幕偏移缓存 JSON 无法解析 ({path}): {exc}") from exc
            self.subtitle_offset = float(data.get("offset", DEFAULT_SUBTITLE_OFFSET))
            self.log(f"Loaded subtitle offset for this video: {self.subtitle_offset:+.2f}s")
        except Exception as exc:
            self.log(f"无法加载字幕偏移量：{exc}")

    def start_realtime(self) -> None:
        mode = self.current_mode()
        if not mode.realtime:
            idx = self.mode_combo.findData("realtime_nuc")
            if idx < 0:
                idx = self.mode_combo.findData("realtime_small")
            if idx >= 0:
                self.mode_combo.setCurrentIndex(idx)
                mode = self.current_mode()
                self.log(f"Switched to {mode.label} for realtime captions.")
            else:
                QMessageBox.information(self, "模式", "请先选择一个适合实时字幕的模式。")
                return
        if self.realtime_thread:
            self.log("Realtime is already running")
            return
        capture_id = self._capture_id()
        self._save_capture_id()
        self.overlay.show()
        self.overlay.set_caption("正在监听 Loopback 输入…")

        if mode.backend == "nuc_asr":
            self.log(
                f"NUC realtime: Loopback :{capture_id} → "
                f"{mode.model} (3s chunks, CUDA large-v3)"
            )
            self.realtime_thread = QThread()
            self.realtime_worker = NUCRealtimeWorker(
                base_url=str(mode.model),
                capture_id=capture_id,
                chunk_seconds=3.0,
            )
        else:
            self.log(
                f"Realtime route: Chrome/player -> SoundSource -> Loopback input; "
                f"whisper-stream capture id {capture_id}"
            )
            self.realtime_thread = QThread()
            self.realtime_worker = RealtimeWorker(mode, capture_id=capture_id)

        self.realtime_worker.moveToThread(self.realtime_thread)
        self.realtime_thread.started.connect(self.realtime_worker.run)
        self.realtime_worker.caption.connect(self.overlay.set_caption)
        self.realtime_worker.caption.connect(self.log)
        self.realtime_worker.status.connect(self.log)
        if hasattr(self.realtime_worker, "session_saved"):
            self.realtime_worker.session_saved.connect(
                lambda p: self.log(f"Realtime session saved to: {p}")
            )
        self.realtime_worker.finished.connect(self.realtime_thread.quit)
        self.realtime_thread.finished.connect(self._clear_realtime)
        self.realtime_button.setEnabled(False)
        self.realtime_button.setText("实时字幕运行中")
        self.realtime_thread.start()

    def list_audio_input_devices(self) -> None:
        self.log("Listing AVFoundation audio input devices...")
        stream_mode = next((mode for mode in MODES if mode.key == "realtime_small" and mode.available), None)
        stream_output = ""
        if stream_mode:
            try:
                stream_result = subprocess.run(
                    [
                        WHISPER_STREAM,
                        "-c", "-1",
                        "-m", str(stream_mode.model),
                        "-t", "1",
                        "-l", "zh",
                        "--step", "1000",
                        "--length", "1000",
                        "--keep", "100",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=5,
                    check=False,
                )
                stream_output = stream_result.stdout.strip()
            except Exception as exc:
                stream_output = f"whisper-stream device probe failed: {exc}"
        try:
            result = subprocess.run(
                [
                    FFMPEG,
                    "-hide_banner",
                    "-f", "avfoundation",
                    "-list_devices", "true",
                    "-i", "",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=8,
                check=False,
            )
        except Exception as exc:
            self.log(f"Could not list audio input devices: {exc}")
            return
        lines = []
        include = False
        for line in result.stdout.splitlines():
            if "AVFoundation audio devices" in line:
                include = True
                lines.append(line)
                continue
            if include:
                if "AVFoundation video devices" in line:
                    include = False
                    continue
                if line.strip():
                    lines.append(line)
        if not lines:
            lines = result.stdout.splitlines()[-20:]
        message = "whisper-stream probe:\n"
        message += stream_output or "(no whisper-stream output)"
        message += "\n\nAVFoundation audio input devices:\n" + "\n".join(lines)
        device_lines = [line for line in lines if re.search(r"\[\d+\]\s*.+$", line)]
        if "found 0 capture devices" in stream_output or not device_lines:
            message += (
                "\n\nNo concrete audio input IDs were visible from this process. "
                "Check macOS Microphone permission for the app/terminal and confirm the Loopback virtual device exists."
            )
        else:
            self._suggest_capture_id(device_lines)
        self.log(message)

    def _suggest_capture_id(self, lines: list[str]) -> None:
        suggested_id: Optional[int] = None
        for line in lines:
            match = re.search(r"\[(\d+)\]\s*(.+)$", line)
            if not match:
                continue
            device_id = int(match.group(1))
            label = match.group(2).strip()
            if any(token in label.lower() for token in ("loopback", "whisper captions", "whispercaptions")):
                suggested_id = device_id
                break
        if suggested_id is not None:
            self.capture_id_input.setText(str(suggested_id))
            self._save_capture_id()

    def stop_all(self) -> None:
        if self.realtime_worker:
            self.realtime_worker.stop()
        if self.queue_worker:
            self.queue_worker.stop()
        if self.controlled_worker:
            self.controlled_worker.stop()
        if self.llm_text_worker:
            self.llm_text_worker.stop()
        if self.controlled_timer.isActive():
            self.controlled_timer.stop()
        self._buffering_paused = False
        self._rolling_all_done = False
        self._flush_ui_logs()
        self._flush_file_logs()

    def active_threads(self) -> list[QThread]:
        return [
            thread
            for thread in (
                self.realtime_thread,
                self.queue_thread,
                self.controlled_thread,
                self.llm_text_thread,
            )
            if thread is not None and thread.isRunning()
        ]

    def wait_for_threads(self, timeout_ms: int = 25000) -> bool:
        deadline = time.monotonic() + timeout_ms / 1000
        for thread in self.active_threads():
            remaining = max(100, int((deadline - time.monotonic()) * 1000))
            thread.quit()
            if not thread.wait(remaining):
                self.log("A worker is still finishing; waiting to avoid Qt thread crash.")
                if not thread.wait(5000):
                    return False
        return not self.active_threads()

    def _clear_realtime(self) -> None:
        if self.realtime_worker:
            self.realtime_worker.deleteLater()
        if self.realtime_thread:
            self.realtime_thread.deleteLater()
        self.realtime_thread = None
        self.realtime_worker = None
        if hasattr(self, "realtime_button"):
            self.realtime_button.setEnabled(True)
            self.realtime_button.setText("实时字幕")
        self._load_realtime_sessions()

    def _load_realtime_sessions(self) -> None:
        if not hasattr(self, "session_list"):
            return
        self.session_list.clear()
        if not REALTIME_DIR.exists():
            return
        
        sessions = [d for d in REALTIME_DIR.iterdir() if d.is_dir() and (d / "manifest.json").exists()]
        sessions.sort(key=lambda d: d.name, reverse=True)
        
        for session_dir in sessions:
            try:
                with open(session_dir / "manifest.json", "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                dur = manifest.get("duration_seconds", 0)
                chunks = manifest.get("num_chunks", 0)
                label = f"{session_dir.name[:8]} {session_dir.name[9:11]}:{session_dir.name[11:13]} ({dur:.1f}s, {chunks}c)"
                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, str(session_dir))
                self.session_list.addItem(item)
            except Exception:
                pass

    def _on_session_selected(self) -> None:
        if not hasattr(self, "session_list") or not self.session_list.currentItem():
            return
            
        session_dir = Path(self.session_list.currentItem().data(Qt.ItemDataRole.UserRole))
        view_idx = self.session_view_combo.currentIndex()
        
        # 0 = Raw, 1 = Polished
        if view_idx == 1:
            json_path = session_dir / "polished-segments.json"
            if not json_path.exists():
                self.session_transcript.setPlainText("未找到校对后的字幕，请先执行【LLM 校对】。")
                return
        else:
            json_path = session_dir / "raw-segments.json"
            if not json_path.exists():
                self.session_transcript.setPlainText("未找到原始字幕。")
                return
                
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            segments = [SubtitleSegment(s["start"], s["end"], s["text"]) for s in data]
            text = "\n".join(f"[{s.start:05.1f}-{s.end:05.1f}] {s.text}" for s in segments)
            self.session_transcript.setPlainText(text)
        except Exception as exc:
            self.session_transcript.setPlainText(f"加载失败: {exc}")

    def _polish_session(self) -> None:
        if not hasattr(self, "session_list") or not self.session_list.currentItem():
            return
            
        session_dir = Path(self.session_list.currentItem().data(Qt.ItemDataRole.UserRole))
        json_path = session_dir / "raw-segments.json"
        if not json_path.exists():
            self.log("找不到 raw-segments.json，无法进行校对。")
            return
            
        provider = self.current_llm_provider()
        api_key = self.current_api_key()
        if not provider or not llm_provider_ready(provider, api_key):
            self.log("请先在【设置】中配置有效的 LLM 提供商和 API Key。")
            return
            
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            segments = [SubtitleSegment(s["start"], s["end"], s["text"]) for s in data]
            
            self.realtime_polish_thread = QThread()
            self.realtime_polish_worker = RealtimePolishWorker(
                session_dir,
                segments,
                provider,
                api_key,
                self.llm_custom_url_input.text().strip(),
                self.llm_custom_model_input.text().strip(),
            )
            self.realtime_polish_worker.moveToThread(self.realtime_polish_thread)
            self.realtime_polish_thread.started.connect(self.realtime_polish_worker.run)
            self.realtime_polish_worker.status.connect(self.log)
            self.realtime_polish_worker.finished.connect(self.realtime_polish_thread.quit)
            self.realtime_polish_thread.finished.connect(self._clear_realtime_polish)
            self.realtime_polish_thread.start()
            
            self.session_polish_button.setEnabled(False)
            self.log("开始后台规整...")
        except Exception as exc:
            self.log(f"启动规整失败: {exc}")

    def _clear_realtime_polish(self) -> None:
        if self.realtime_polish_worker:
            self.realtime_polish_worker.deleteLater()
            self.realtime_polish_worker = None
        if self.realtime_polish_thread:
            self.realtime_polish_thread.deleteLater()
            self.realtime_polish_thread = None
        if hasattr(self, "session_polish_button"):
            self.session_polish_button.setEnabled(True)
        self._on_session_selected()

    def _re_recognize_session(self) -> None:
        if not hasattr(self, "session_list") or not self.session_list.currentItem():
            return
            
        session_dir = Path(self.session_list.currentItem().data(Qt.ItemDataRole.UserRole))
        if not (session_dir / "full-audio.wav").exists():
            self.log("没有找到 full-audio.wav，无法重新识别。")
            return
            
        mode = self.current_mode()
        if mode.backend != "nuc_asr":
            self.log("请在下拉框中选择一个 nuc_asr 模型作为识别后端。")
            return
            
        provider = self.current_llm_provider()
        api_key = self.current_api_key()
        
        self.realtime_re_recognize_thread = QThread()
        self.realtime_re_recognize_worker = RealtimeReRecognizeWorker(
            session_dir,
            str(mode.model),
            provider if llm_provider_ready(provider, api_key) else None,
            api_key,
            self.llm_custom_url_input.text().strip(),
            self.llm_custom_model_input.text().strip(),
        )
        self.realtime_re_recognize_worker.moveToThread(self.realtime_re_recognize_thread)
        self.realtime_re_recognize_thread.started.connect(self.realtime_re_recognize_worker.run)
        self.realtime_re_recognize_worker.status.connect(self.log)
        self.realtime_re_recognize_worker.finished.connect(self.realtime_re_recognize_thread.quit)
        self.realtime_re_recognize_thread.finished.connect(self._clear_realtime_re_recognize)
        self.realtime_re_recognize_thread.start()
        
        self.session_rerecognize_button.setEnabled(False)
        self.log("开始重识别...")

    def _clear_realtime_re_recognize(self) -> None:
        if self.realtime_re_recognize_worker:
            self.realtime_re_recognize_worker.deleteLater()
            self.realtime_re_recognize_worker = None
        if self.realtime_re_recognize_thread:
            self.realtime_re_recognize_thread.deleteLater()
            self.realtime_re_recognize_thread = None
        if hasattr(self, "session_rerecognize_button"):
            self.session_rerecognize_button.setEnabled(True)
        self._on_session_selected()

    def _open_session_dir(self) -> None:
        if not hasattr(self, "session_list") or not self.session_list.currentItem():
            return
        session_dir = Path(self.session_list.currentItem().data(Qt.ItemDataRole.UserRole))
        subprocess.run(["open", str(session_dir)])

    def _delete_session(self) -> None:
        if not hasattr(self, "session_list") or not self.session_list.currentItem():
            return
        session_dir = Path(self.session_list.currentItem().data(Qt.ItemDataRole.UserRole))
        reply = QMessageBox.question(self, "确认删除", f"确定要删除会话目录吗？\n{session_dir}", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            import shutil
            try:
                shutil.rmtree(session_dir)
                self._load_realtime_sessions()
            except Exception as exc:
                self.log(f"删除失败: {exc}")

    def add_queue_item(self) -> None:
        text = self._normalize_source_input(self.url_input.text())
        if not text:
            return
        self.queue.addItem(QListWidgetItem(text))
        self.url_input.clear()

    @staticmethod
    def _normalize_source_input(source: str) -> str:
        source = source.strip().strip("\"'")
        if source.startswith("file://"):
            parsed = urlparse(source)
            return unquote(parsed.path).strip()
        return source

    def move_item(self, delta: int) -> None:
        row = self.queue.currentRow()
        if row < 0:
            return
        new_row = max(0, min(self.queue.count() - 1, row + delta))
        if new_row == row:
            return
        item = self.queue.takeItem(row)
        self.queue.insertItem(new_row, item)
        self.queue.setCurrentRow(new_row)

    def process_queue(self) -> None:
        if self.queue_thread:
            self.log("Queue is already running")
            return
        mode = self.current_mode()
        if mode.realtime:
            QMessageBox.information(self, "模式", "队列处理请使用非实时的高质量模式。")
            return
        items = [
            self._normalize_source_input(self.queue.item(i).text().lstrip("✓✗ ").strip())
            for i in range(self.queue.count())
        ]
        if not items:
            return
        self.queue_thread = QThread()
        self.queue_worker = QueueWorker(items, mode)
        self.queue_worker.moveToThread(self.queue_thread)
        self.queue_thread.started.connect(self.queue_worker.run)
        self.queue_worker.status.connect(self.log)
        self.queue_worker.caption.connect(self.overlay.set_caption)
        self.queue_worker.output_ready.connect(self._load_queue_output)
        self.queue_worker.finished_item.connect(self._mark_item)
        self.queue_worker.finished.connect(self.queue_thread.quit)
        self.queue_thread.finished.connect(self._clear_queue)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(0)
        self._set_status_summary(f"队列处理中 | 模式 {mode.label}")
        self.queue_thread.start()

    def process_local_media(self) -> None:
        if self.queue_thread:
            self.log("Queue is already running")
            return
        source = self._normalize_source_input(self.url_input.text())
        if not source:
            source, _ = QFileDialog.getOpenFileName(
                self,
                "选择本地视频或音频文件",
                str(Path.home()),
                "Media files (*.mp4 *.m4v *.mov *.mkv *.webm *.mp3 *.m4a *.wav *.aac *.flac *.opus);;All files (*)",
            )
            source = self._normalize_source_input(source)
            if not source:
                return
        path = Path(source).expanduser()
        if not path.exists() or not path.is_file():
            QMessageBox.warning(self, "文件不存在", f"没有找到本地文件：\n\n{path}")
            return
        mode = self.current_mode()
        if not mode.available:
            QMessageBox.warning(self, "模型缺失", f"当前所选模式不可用：{mode.label}")
            return
        self.queue_thread = QThread()
        self.queue_worker = QueueWorker([str(path)], mode)
        self.queue_worker.moveToThread(self.queue_thread)
        self.queue_thread.started.connect(self.queue_worker.run)
        self.queue_worker.status.connect(self.log)
        self.queue_worker.caption.connect(self.overlay.set_caption)
        self.queue_worker.output_ready.connect(self._load_queue_output)
        self.queue_worker.finished_item.connect(self._mark_item)
        self.queue_worker.finished.connect(self.queue_thread.quit)
        self.queue_thread.finished.connect(self._clear_queue)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(0)
        self.url_input.setText(str(path))
        self.overlay.show()
        self.overlay.set_caption("正在处理本地文件。")
        self._set_status_summary(f"本地转写中 | 模式 {mode.label} | {path.name}")
        self.queue_thread.start()

    def start_controlled_url(self) -> None:
        if self.controlled_thread:
            self.log("Controlled URL preparation is already running")
            return

        source = self.url_input.text().strip()
        queue_item = self.queue.currentItem()
        
        # If the input URL is exactly what's currently in Chrome, or if it's empty, 
        # we try to fetch the latest URL from Chrome to ensure it's up to date.
        chrome_url = chrome_get_url()
        if chrome_url and chrome_url.startswith(("http://", "https://")):
            # If the user hasn't manually typed a different valid URL, and there is no 
            # active queue item selected (or the input matches the old Chrome URL), override it.
            if not source or source == self._last_auto_url or not source.startswith(("http://", "https://")):
                source = chrome_url
                self.log(f"Auto-detected Chrome URL: {source}")
                self.url_input.setText(source)
                self._last_auto_url = source

        # If still no valid source, check the queue
        if not source.startswith(("http://", "https://")) and queue_item:
            source = queue_item.text().lstrip("✓✗ ").strip()

        if not source.startswith(("http://", "https://")):
            QMessageBox.information(self, "需要网址",
                "没有找到可处理的网址。请粘贴视频网址、从队列里选择一个，或先在 Chrome 中打开视频页面。")
            return
        
        # Validate URL is suitable for yt-dlp
        is_valid, error_msg = validate_url_for_yt_dlp(source)
        if not is_valid:
            QMessageBox.critical(self, "网址无效",
                f"这个网址无法被 yt-dlp 处理。\n\n{error_msg}\n\n支持的平台包括：YouTube、Bilibili、Vimeo、Twitch 等。")
            return
        mode = self.current_mode()
        if not mode.available:
            QMessageBox.warning(self, "模型缺失", f"当前所选模式不可用：{mode.label}")
            return

        # Save LLM settings before starting
        self._save_current_llm_key()
        llm_provider = None
        llm_api_key = ""
        llm_api_url = ""
        llm_model_id = ""
        if self.llm_group.isChecked():
            key = self.llm_provider_combo.currentData()
            llm_provider = next(p for p in LLM_PROVIDERS if p.key == key)
            llm_api_key = self.llm_api_key_input.text().strip()
            if key == "custom":
                llm_api_url = self.llm_custom_url_input.text().strip()
                llm_model_id = self.llm_custom_model_input.text().strip()

        self._rolling_all_done = False
        self._buffering_paused = False
        self._controlled_paused = False
        self._controlled_resume_time = 0.0
        self._controlled_url = canonical_media_url(source)
        job_key = cache_slug(self._controlled_url, mode.backend, mode.model_name, 30)
        self._controlled_cache_dir = CACHE_DIR / job_key
        self._current_caption_index = -1
        self._last_caption_debug_second = -1
        self._load_subtitle_offset()
        self._set_controlled_segments([])
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(0)  # indeterminate until chunks calculated
        self.overlay.show()
        self.overlay.set_caption("正在准备字幕，Chrome 已暂停。")
        self._set_status_summary(f"网址受控字幕准备中 | 模式 {mode.label}")
        self.controlled_thread = QThread()
        self.controlled_worker = RollingPrefetchWorker(
            source, mode,
            llm_provider=llm_provider, llm_api_key=llm_api_key,
            llm_api_url=llm_api_url, llm_model_id=llm_model_id,
        )
        self.controlled_worker.moveToThread(self.controlled_thread)
        self.controlled_thread.started.connect(self.controlled_worker.run)
        self.controlled_worker.status.connect(self.log)
        self.controlled_worker.native_subtitles_detected.connect(self._handle_native_subtitles_detected)
        self.controlled_worker.first_segments.connect(self._start_rolling_playback)
        self.controlled_worker.more_segments.connect(self._add_rolling_segments)
        self.controlled_worker.progress.connect(self._update_progress)
        self.controlled_worker.all_done.connect(self._on_rolling_done)
        self.controlled_worker.finished.connect(self.controlled_thread.quit)
        self.controlled_thread.finished.connect(self._clear_controlled_worker)
        self.controlled_thread.start()

    def _start_rolling_playback(self, segments: list[SubtitleSegment]) -> None:
        self._set_controlled_segments(segments)
        self._refresh_transcript_list()
        first_start = min((segment.start for segment in self.controlled_segments), default=0.0)
        start_at = 0.0
        self._controlled_resume_time = start_at
        self._set_playback_anchor(start_at)
        self.overlay.set_caption("字幕已就绪，开始播放视频。")
        if not chrome_play_url_from(self._controlled_url, start_at):
            chrome_play_from(start_at)
        self.log(
            f"Starting Chrome at {start_at:.2f}s; first subtitle starts at {first_start:.2f}s; "
            f"subtitle offset {self.subtitle_offset:+.2f}s"
        )
        self.controlled_timer.start()

    def _load_queue_output(self, source: str, output_base: str) -> None:
        base = Path(output_base)
        srt_path = base.with_suffix(".srt")
        txt_path = base.with_suffix(".txt")
        self._latest_export_base = base
        if srt_path.exists():
            try:
                self._set_controlled_segments(parse_srt(srt_path))
                self._controlled_url = ""
                self._controlled_cache_dir = None
                self._rolling_all_done = True
                self._current_caption_index = -1
                self._refresh_transcript_list()
                self.analysis_context_output.setPlainText(
                    f"已载入本地字幕：{srt_path}\n来源：{source}"
                )
                self.log(f"Loaded local transcript into analysis panel: {srt_path}")
                self.overlay.set_caption("本地字幕已生成并载入。")
                return
            except Exception as exc:
                self.log(f"Could not load local SRT output: {exc}")
        if txt_path.exists():
            self.analysis_context_output.setPlainText(
                f"已生成本地转写文本：{txt_path}\n来源：{source}"
            )
            self.log(f"Local transcript text ready: {txt_path}")
            self.overlay.set_caption("本地转写文本已生成。")

    def _handle_native_subtitles_detected(self, segments: list[SubtitleSegment], message: str) -> None:
        self.stop_all()
        self.progress_bar.setVisible(False)
        self.controlled_timer.stop()
        # 加载内置字幕
        self._set_controlled_segments(segments)
        self._refresh_transcript_list()
        self._rolling_all_done = True
        self._current_caption_index = -1
        self._controlled_resume_time = 0.0
        self._set_playback_anchor(0.0)
        # 恢复播放
        if not (self._controlled_url and chrome_play_url_from(self._controlled_url, 0.0)):
            chrome_play_from(0.0)
        self.controlled_timer.start()
        self._tick_controlled_captions()
        # 更新状态
        self.overlay.set_caption("视频自带字幕已加载")
        self.log(message)
        self._set_status_summary("已加载视频自带字幕")
        QMessageBox.information(self, "检测到视频自带字幕", message)

    def _add_rolling_segments(self, new_segments: list[SubtitleSegment]) -> None:
        self._extend_controlled_segments(new_segments)
        self._refresh_transcript_list()
        if self._buffering_paused:
            current = self._estimated_playback_time()
            last_end = self.controlled_segments[-1].end if self.controlled_segments else 0
            caption_time = max(0.0, current + self.subtitle_offset) if current is not None else None
            if caption_time is not None and last_end - caption_time >= BUFFER_RESUME_MARGIN:
                if self._controlled_url and not chrome_resume_url(self._controlled_url, activate_tab=False):
                    chrome_resume()
                elif not self._controlled_url:
                    chrome_resume()
                self._buffering_paused = False
                self.overlay.set_caption("正在恢复播放。")
                self.log(f"Buffer sufficient ({last_end - caption_time:.1f}s ahead), resuming Chrome")

    def _on_rolling_done(self) -> None:
        self._rolling_all_done = True
        self.progress_bar.setVisible(False)
        self.log("All caption chunks transcribed")
        self._current_progress = None
        self._set_status_summary("所有字幕块已完成")
        if self._buffering_paused:
            if self._controlled_url and not chrome_resume_url(self._controlled_url, activate_tab=False):
                chrome_resume()
            elif not self._controlled_url:
                chrome_resume()
            self._buffering_paused = False
            self.overlay.set_caption("所有字幕已就绪，恢复播放。")

    def _update_progress(self, current: int, total: int) -> None:
        self._current_progress = (current, total)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self._set_status_summary(f"处理中 | 进度 {current}/{total}")

    def _caption_index_at_time(self, caption_time: float) -> int:
        """Find the subtitle index for a given caption_time.

        Optimization strategy:
        1. Check cached current index (usually the next to display)
        2. Check adjacent indices (1-3 steps, normal playback/rewind)
        3. Fall back to bisect for large seeks or initial load
        """
        segments = self.controlled_segments
        if not segments:
            return -1

        # 1. Check cached current position
        current = self._current_caption_index
        if 0 <= current < len(segments):
            seg = segments[current]
            if seg.start - 0.1 <= caption_time <= seg.end + 0.1:
                return current

        # 2. Check adjacent indices (forward/backward 1-3 steps)
        offsets = (1, 2, 3, -1)
        for offset in offsets:
            idx = current + offset if current >= 0 else offset
            if 0 <= idx < len(segments):
                seg = segments[idx]
                if seg.start - 0.1 <= caption_time <= seg.end + 0.1:
                    return idx

        # 3. Fall back to bisect for large seeks
        i = bisect.bisect_right(self._controlled_segment_starts, caption_time)
        if i > 0:
            seg = segments[i - 1]
            if seg.start <= caption_time <= seg.end:
                return i - 1

        return -1

    def _tick_controlled_captions(self) -> None:
        if self._controlled_paused:
            return
        current = self._estimated_playback_time()
        if current is None:
            self.log("Caption tick: could not read Chrome video time", "debug")
            return
        self._controlled_resume_time = current
        caption_time = max(0.0, current + self.subtitle_offset)
        debug_second = int(current)
        if debug_second != self._last_caption_debug_second and debug_second % 2 == 0:
            self._last_caption_debug_second = debug_second
            first = self.controlled_segments[0].start if self.controlled_segments else -1
            last = self.controlled_segments[-1].end if self.controlled_segments else -1
            self.log(
                f"Caption tick: video={current:.2f}s caption={caption_time:.2f}s "
                f"range={first:.2f}-{last:.2f}s offset={self.subtitle_offset:+.2f}s",
                "trace",
            )
        if self._buffering_paused:
            return
        # Buffer check: pause Chrome if running out of transcribed captions
        if not self._rolling_all_done and self.controlled_segments:
            last_end = self.controlled_segments[-1].end
            if caption_time >= last_end - BUFFER_PAUSE_MARGIN:
                if self._controlled_url and chrome_pause_url(self._controlled_url, activate_tab=False) is None:
                    chrome_pause()
                elif not self._controlled_url:
                    chrome_pause()
                self._buffering_paused = True
                self.overlay.set_caption("字幕缓冲中…")
                self.log(f"Buffer exhausted at {caption_time:.1f}s (buffer ends {last_end:.1f}s), pausing Chrome")
                return
        # Normal subtitle matching (optimized with cached index + bisect)
        index = self._caption_index_at_time(caption_time)
        if index >= 0:
            self._current_caption_index = index
            self.overlay.set_caption_context(self.controlled_segments, index)
        elif self._current_caption_index != -1:
            self.overlay.clear_caption()
            self._current_caption_index = -1
        # End detection
        if self._rolling_all_done and self.controlled_segments and caption_time > self.controlled_segments[-1].end + 2:
            self.controlled_timer.stop()
            self.overlay.set_caption("字幕播放完成")

    def _set_playback_anchor(self, current_time: float) -> None:
        self._playback_anchor_time = current_time
        self._playback_anchor_monotonic = time.monotonic()
        self._last_chrome_probe_time = current_time
        self._last_chrome_probe_monotonic = self._playback_anchor_monotonic

    def _read_precise_chrome_time(self) -> Optional[float]:
        current = chrome_current_time_url(self._controlled_url) if self._controlled_url else chrome_current_time()
        if current is not None:
            self._set_playback_anchor(current)
        return current

    def _estimated_playback_time(self) -> Optional[float]:
        now = time.monotonic()
        needs_probe = (
            self._last_chrome_probe_time is None
            or (now - self._last_chrome_probe_monotonic) >= 1.0
        )
        if needs_probe:
            current = self._read_precise_chrome_time()
            if current is not None:
                return current
        if self._last_chrome_probe_time is None:
            return None
        elapsed = max(0.0, now - self._playback_anchor_monotonic)
        return self._playback_anchor_time + elapsed

    def _clear_controlled_worker(self) -> None:
        if self.controlled_worker:
            self.controlled_worker.deleteLater()
        if self.controlled_thread:
            self.controlled_thread.deleteLater()
        self.controlled_thread = None
        self.controlled_worker = None

    def _clear_llm_text_worker(self) -> None:
        if self.llm_text_worker:
            self.llm_text_worker.deleteLater()
        if self.llm_text_thread:
            self.llm_text_thread.deleteLater()
        self.llm_text_thread = None
        self.llm_text_worker = None
        self.summarize_button.setEnabled(True)
        self.article_button.setEnabled(True)

    def _mark_item(self, source: str, ok: bool) -> None:
        marker = "✓" if ok else "✗"
        for i in range(self.queue.count()):
            item = self.queue.item(i)
            item_source = item.text().lstrip("✓✗ ").strip()
            if item_source == source:
                item.setText(f"{marker} {source}")
                break

    def _clear_queue(self) -> None:
        if self.queue_worker:
            self.queue_worker.deleteLater()
        if self.queue_thread:
            self.queue_thread.deleteLater()
        self.queue_thread = None
        self.queue_worker = None
        self.progress_bar.setVisible(False)
        self._current_progress = None
        self._set_status_summary("队列任务已结束")
        self.log("Queue worker finished")

    def closeEvent(self, event: QCloseEvent) -> None:
        self.hide()
        event.ignore()


class App:
    def __init__(self) -> None:
        self.qt = QApplication(sys.argv)
        self.qt.setApplicationName("Whisper Captioner")
        self.overlay = SubtitleOverlay()
        self.window = MainWindow(self.overlay)
        self.tray = QSystemTrayIcon(QIcon.fromTheme("audio-input-microphone"))
        self.tray.setToolTip("Whisper 字幕助手")
        self._build_menu()
        self.qt.aboutToQuit.connect(self.shutdown)
        self.tray.show()

    def _build_menu(self) -> None:
        menu = QMenu()
        show = QAction("显示控制面板")
        show.triggered.connect(self.window.show)
        qwen_chat = QAction("打开 Qwen3-8B 聊天")
        qwen_chat.triggered.connect(self.window.open_qwen_chat)
        overlay = QAction("显示/隐藏字幕浮窗")
        overlay.triggered.connect(lambda: self.overlay.setVisible(not self.overlay.isVisible()))
        pin = QAction("切换浮窗置顶")
        pin.triggered.connect(self.overlay.toggle_pin)
        rewind_5 = QAction("后退 5 秒")
        rewind_5.triggered.connect(self.window.rewind_5s)
        rewind = QAction("后退 10 秒")
        rewind.triggered.connect(self.window.rewind_10s)
        play_pause = QAction("暂停/继续")
        play_pause.triggered.connect(self.window.toggle_playback)
        forward_5 = QAction("前进 5 秒")
        forward_5.triggered.connect(self.window.forward_5s)
        forward = QAction("前进 10 秒")
        forward.triggered.connect(self.window.forward_10s)
        stop = QAction("停止")
        stop.triggered.connect(self.window.stop_all)
        quit_action = QAction("退出")
        quit_action.triggered.connect(self.quit)

        menu.addAction(show)
        menu.addAction(qwen_chat)
        menu.addAction(overlay)
        menu.addAction(pin)
        menu.addSeparator()
        menu.addAction(rewind_5)
        menu.addAction(rewind)
        menu.addAction(play_pause)
        menu.addAction(forward_5)
        menu.addAction(forward)
        menu.addSeparator()
        menu.addAction(stop)
        menu.addSeparator()
        menu.addAction(quit_action)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(lambda reason: self.window.show() if reason == QSystemTrayIcon.ActivationReason.Trigger else None)

    def shutdown(self) -> None:
        self.window.stop_all()
        self.window.qwen_chat_service.stop()
        self.window._flush_ui_logs()
        self.window._flush_file_logs()
        self.window.wait_for_threads()

    def quit(self) -> None:
        self.shutdown()
        if self.window.active_threads():
            self.window.log("Quit delayed: waiting for worker thread to finish safely.")
            QTimer.singleShot(1000, self.quit)
            return
        self.qt.quit()

    def run(self) -> int:
        self.window.show()
        return self.qt.exec()


if __name__ == "__main__":
    from whisper_captioner.config import OUTPUT_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sys.exit(App().run())
