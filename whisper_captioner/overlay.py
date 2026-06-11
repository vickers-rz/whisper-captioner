"""Subtitle overlay widget for displaying captions on top of other applications."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPoint, QSettings, QSize, Qt, Signal
from PySide6.QtGui import QAction, QCloseEvent, QContextMenuEvent, QFont, QMouseEvent, QResizeEvent
from PySide6.QtWidgets import (
    QApplication,
    QFontDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QPushButton,
    QSizeGrip,
    QVBoxLayout,
    QWidget,
)

from whisper_captioner.models import SubtitleSegment


class SubtitleOverlay(QWidget):
    _instances: list[SubtitleOverlay] = []
    rewind_5_requested = Signal()
    rewind_requested = Signal()
    forward_5_requested = Signal()
    forward_requested = Signal()
    play_pause_requested = Signal()
    chapter_seek_requested = Signal(float)

    def __init__(self) -> None:
        super().__init__()
        SubtitleOverlay._instances.append(self)
        self.setWindowTitle("Whisper Captions")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Window
        )
        self.setWindowFlag(Qt.WindowType.WindowDoesNotAcceptFocus, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_AlwaysShowToolTips)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        # --- Load persisted settings ---
        self._settings = QSettings("WhisperCaptioner", "overlay")
        self._opacity = self._settings.value("opacity", 0.85, type=float)
        self._font_family = self._settings.value("fontFamily", "Avenir Next", type=str)
        self._font_size = self._settings.value("fontSize", 28, type=int)
        self._font_weight = self._settings.value("fontWeight", QFont.Weight.DemiBold.value, type=int)
        self._overlay_width = self._settings.value("overlayWidth", 980, type=int)
        self._overlay_height = self._settings.value("overlayHeight", 260, type=int)
        self._always_on_top = self._settings.value("alwaysOnTop", True, type=bool)

        # --- Drag state ---
        self._drag_pos: Optional[QPoint] = None
        self._resize_edge: Optional[Qt.Edge] = None
        self._resize_start: Optional[QPoint] = None
        self._resize_init_size: Optional[QSize] = None

        self._build_ui()
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, self._always_on_top)
        self._load_position()
        self._apply_style()

    def _build_ui(self) -> None:
        self._chapter_starts: list[float] = []
        self._active_chapter_index = -1
        self.chapter_container = QWidget()
        self.chapter_container.setObjectName("ChapterContainer")
        chapter_layout = QVBoxLayout(self.chapter_container)
        chapter_layout.setContentsMargins(14, 10, 14, 8)
        chapter_nav = QHBoxLayout()
        self.previous_chapter_button = QPushButton("")
        self.current_chapter_button = QPushButton("")
        self.next_chapter_button = QPushButton("")
        for button in (
            self.previous_chapter_button,
            self.current_chapter_button,
            self.next_chapter_button,
        ):
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setMinimumHeight(34)
        self.previous_chapter_button.clicked.connect(
            lambda: self._emit_chapter_seek(self._active_chapter_index - 1)
        )
        self.current_chapter_button.clicked.connect(
            lambda: self._emit_chapter_seek(self._active_chapter_index)
        )
        self.next_chapter_button.clicked.connect(
            lambda: self._emit_chapter_seek(self._active_chapter_index + 1)
        )
        chapter_nav.addWidget(self.previous_chapter_button, 1)
        chapter_nav.addWidget(self.current_chapter_button, 2)
        chapter_nav.addWidget(self.next_chapter_button, 1)
        self.chapter_title_label = QLabel("")
        self.chapter_title_label.setObjectName("ChapterTitle")
        self.chapter_title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.chapter_description_label = QLabel("")
        self.chapter_description_label.setObjectName("ChapterDescription")
        self.chapter_description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.chapter_description_label.setWordWrap(True)
        chapter_layout.addLayout(chapter_nav)
        chapter_layout.addWidget(self.chapter_title_label)
        chapter_layout.addWidget(self.chapter_description_label)
        self.chapter_container.hide()

        self.previous_label = QLabel("")
        self.previous_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.previous_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.previous_label.setWordWrap(True)
        self.previous_label.setObjectName("PreviousCaption")
        self.label = QLabel("就绪")
        self.label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setWordWrap(True)
        self.label.setObjectName("CurrentCaption")
        self._apply_font()

        self.pin_button = QPushButton("📌")
        self.rewind_5_button = QPushButton("-5s")
        self.rewind_button = QPushButton("-10s")
        self.play_pause_button = QPushButton("暂停")
        self.forward_5_button = QPushButton("+5s")
        self.forward_button = QPushButton("+10s")
        for button in (
            self.pin_button,
            self.rewind_5_button,
            self.rewind_button,
            self.play_pause_button,
            self.forward_5_button,
            self.forward_button,
        ):
            button.setFixedHeight(26)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setStyleSheet(
                """
                QPushButton {
                    color: white;
                    background: rgba(0,0,0,120);
                    border: 1px solid rgba(255,255,255,80);
                    border-radius: 10px;
                    padding: 2px 10px;
                }
                QPushButton:hover {
                    background: rgba(255,255,255,55);
                }
                """
            )
        self.pin_button.clicked.connect(self.toggle_pin)
        self.rewind_5_button.clicked.connect(self.rewind_5_requested.emit)
        self.rewind_button.clicked.connect(self.rewind_requested.emit)
        self.play_pause_button.clicked.connect(self.play_pause_requested.emit)
        self.forward_5_button.clicked.connect(self.forward_5_requested.emit)
        self.forward_button.clicked.connect(self.forward_requested.emit)
        self._sync_pin_button()

        # Resize handle (bottom-right corner grip)
        self._resize_grip = QSizeGrip(self)
        self._resize_grip.setFixedSize(20, 20)
        self._resize_grip.setStyleSheet(
            """
            QSizeGrip {
                background: rgba(255,255,255,30);
                border-radius: 4px;
                margin: 4px;
            }
            QSizeGrip:hover {
                background: rgba(255,255,255,60);
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        controls = QHBoxLayout()
        controls.setContentsMargins(8, 6, 8, 0)
        controls.addStretch()
        controls.addWidget(self.rewind_5_button)
        controls.addWidget(self.rewind_button)
        controls.addWidget(self.play_pause_button)
        controls.addWidget(self.pin_button)
        controls.addWidget(self.forward_5_button)
        controls.addWidget(self.forward_button)
        layout.addLayout(controls)
        layout.addWidget(self.chapter_container)
        layout.addWidget(self.previous_label)
        layout.addWidget(self.label)
        # Size grip sits below label via a nested layout
        bottom = QHBoxLayout()
        bottom.addStretch()
        bottom.addWidget(self._resize_grip, 0, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)
        layout.addLayout(bottom)

        self.setMinimumSize(500, 180)
        self.resize(self._overlay_width, self._overlay_height)

    def _apply_font(self) -> None:
        weight_val = max(1, min(99, self._font_weight))
        self.previous_label.setFont(QFont(self._font_family, self._font_size, max(1, weight_val - 15)))
        self.label.setFont(QFont(self._font_family, self._font_size, weight_val))

    def _apply_style(self) -> None:
        alpha = int(self._opacity * 255)
        self.previous_label.setStyleSheet(
            f"""
            QLabel#PreviousCaption {{
              color: rgba(255, 255, 255, 170);
              background: rgba(0, 0, 0, {max(60, alpha - 90)});
              border-top-left-radius: 18px;
              border-top-right-radius: 18px;
              border-bottom-left-radius: 10px;
              border-bottom-right-radius: 10px;
              padding: 12px 28px 8px 28px;
            }}
            """
        )
        self.label.setStyleSheet(
            f"""
            QLabel#CurrentCaption {{
              color: white;
              background: rgba(0, 0, 0, {alpha});
              border-top-left-radius: 10px;
              border-top-right-radius: 10px;
              border-bottom-left-radius: 18px;
              border-bottom-right-radius: 18px;
              padding: 10px 28px 18px 28px;
            }}
            """
        )
        self.chapter_container.setStyleSheet(
            f"""
            QWidget#ChapterContainer {{
              background: rgba(0, 0, 0, {max(100, alpha - 45)});
              border-radius: 16px;
            }}
            QPushButton {{
              color: rgba(255, 255, 255, 210);
              background: rgba(0, 0, 0, 120);
              border: none;
              border-bottom: 2px solid rgba(255, 255, 255, 70);
              border-radius: 0px;
              padding: 5px 10px;
              font-weight: 600;
            }}
            QPushButton:hover {{
              color: #55b8ff;
              border-bottom: 2px solid #55b8ff;
            }}
            QPushButton:disabled {{
              color: rgba(255, 255, 255, 50);
            }}
            QLabel#ChapterTitle {{
              color: #d8ffe8;
              font-size: 20px;
              font-weight: 700;
              padding-top: 6px;
            }}
            QLabel#ChapterDescription {{
              color: rgba(255, 255, 255, 205);
              font-size: 15px;
              padding: 2px 20px 5px 20px;
            }}
            """
        )
        self.setStyleSheet(f"SubtitleOverlay {{ background: transparent; }}")

    def _sync_pin_button(self) -> None:
        self.pin_button.setText("📌" if self._always_on_top else "📍")
        self.pin_button.setToolTip("关闭置顶" if self._always_on_top else "保持字幕置顶")

    def _load_position(self) -> None:
        pos = self._settings.value("pos", None)
        if pos is not None:
            self.move(pos)
        else:
            screen = QApplication.primaryScreen().availableGeometry()
            self.move((screen.width() - self.width()) // 2, screen.height() - 210)

    def _save_position(self) -> None:
        self._settings.setValue("pos", self.pos())

    # --- caption ---
    def set_caption(self, text: str) -> None:
        cleaned = text.strip()
        if cleaned:
            self.label.setTextFormat(Qt.TextFormat.PlainText)
            self.label.setText(cleaned)
            self.keep_on_top(raise_window=False)

    def clear_caption(self) -> None:
        self.previous_label.setText("")
        self.label.setText("")
        self.keep_on_top(raise_window=False)

    def set_chapters(self, chapters: list[object]) -> None:
        self._chapter_starts = [
            max(0.0, float(getattr(chapter, "start_seconds")))
            for chapter in chapters
        ]
        self._chapters = list(chapters)
        self._active_chapter_index = -1
        self.chapter_container.setVisible(bool(self._chapters))

    def clear_chapters(self) -> None:
        self._chapters = []
        self._chapter_starts = []
        self._active_chapter_index = -1
        self.chapter_container.hide()

    def set_chapter_at_time(self, seconds: float) -> None:
        if not getattr(self, "_chapters", None):
            return
        import bisect

        index = max(0, bisect.bisect_right(self._chapter_starts, seconds) - 1)
        if index == self._active_chapter_index:
            return
        self._active_chapter_index = index
        chapter = self._chapters[index]
        previous = self._chapters[index - 1].title if index > 0 else ""
        following = self._chapters[index + 1].title if index + 1 < len(self._chapters) else ""
        self.previous_chapter_button.setText(previous or "已是第一章")
        self.previous_chapter_button.setEnabled(index > 0)
        self.current_chapter_button.setText(chapter.title)
        self.next_chapter_button.setText(following or "已是最后一章")
        self.next_chapter_button.setEnabled(index + 1 < len(self._chapters))
        self.chapter_title_label.setText(chapter.title)
        self.chapter_description_label.setText(chapter.description)
        self.chapter_description_label.setVisible(bool(chapter.description))

    def _emit_chapter_seek(self, index: int) -> None:
        if 0 <= index < len(getattr(self, "_chapters", [])):
            self.chapter_seek_requested.emit(self._chapters[index].start_seconds)

    def set_caption_context(self, segments: list[SubtitleSegment], current_index: int) -> None:
        if not segments or current_index < 0 or current_index >= len(segments):
            return
        previous_text = segments[current_index - 1].text if current_index - 1 >= 0 else ""
        self.previous_label.setText(previous_text.strip())
        self.set_caption(segments[current_index].text)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.keep_on_top()

    def keep_on_top(self, raise_window: bool = True) -> None:
        if self._always_on_top:
            self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        if not self.isVisible():
            self.show()
        if raise_window:
            self.raise_()

    def toggle_pin(self) -> None:
        self.set_always_on_top(not self._always_on_top)

    def set_always_on_top(self, enabled: bool) -> None:
        self._always_on_top = enabled
        self._settings.setValue("alwaysOnTop", enabled)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, enabled)
        self.setWindowFlag(Qt.WindowType.WindowDoesNotAcceptFocus, True)
        self._sync_pin_button()
        self.show()
        if enabled:
            self.raise_()

    # --- drag ---
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint()
            self._resize_edge = self._hit_edge(event.globalPosition().toPoint())
            if self._resize_edge:
                self._resize_start = event.globalPosition().toPoint()
                self._resize_init_size = self.size()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if event.buttons() != Qt.MouseButton.LeftButton:
            return
        if self._resize_edge:
            self._do_resize(event.globalPosition().toPoint())
        elif self._drag_pos:
            delta = event.globalPosition().toPoint() - self._drag_pos
            self.move(self.pos() + delta)
            self._drag_pos = event.globalPosition().toPoint()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if not self._resize_edge:
                self._save_position()
            self._drag_pos = None
            self._resize_edge = None
            self._resize_start = None
            self._resize_init_size = None
            self._save_position()
            event.accept()

    def _hit_edge(self, global_pos: QPoint) -> Optional[Qt.Edge]:
        margin = 8
        rect = self.frameGeometry()
        if global_pos.x() >= rect.right() - margin:
            if global_pos.y() >= rect.bottom() - margin:
                return Qt.Edge.RightEdge | Qt.Edge.BottomEdge
            return Qt.Edge.RightEdge
        if global_pos.y() >= rect.bottom() - margin:
            return Qt.Edge.BottomEdge
        return None

    def _do_resize(self, global_pos: QPoint) -> None:
        if not self._resize_edge or not self._resize_start or not self._resize_init_size:
            return
        delta = global_pos - self._resize_start
        w = self._resize_init_size.width()
        h = self._resize_init_size.height()
        if self._resize_edge & Qt.Edge.RightEdge:
            w = max(400, self._resize_init_size.width() + delta.x())
        if self._resize_edge & Qt.Edge.BottomEdge:
            h = max(80, self._resize_init_size.height() + delta.y())
        self.resize(w, h)

    def resizeEvent(self, event: QResizeEvent) -> None:
        self._settings.setValue("overlayWidth", self.width())
        self._settings.setValue("overlayHeight", self.height())
        super().resizeEvent(event)

    # --- context menu for settings ---
    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
        menu = QMenu(self)
        font_act = QAction("字体…", self)
        font_act.triggered.connect(self._choose_font)
        menu.addAction(font_act)

        bigger_act = QAction("增大字号", self)
        bigger_act.triggered.connect(lambda: self._adjust_font_size(2))
        menu.addAction(bigger_act)

        smaller_act = QAction("减小字号", self)
        smaller_act.triggered.connect(lambda: self._adjust_font_size(-2))
        menu.addAction(smaller_act)

        opacity_act = QAction("透明度…", self)
        opacity_act.triggered.connect(self._choose_opacity)
        menu.addAction(opacity_act)

        pin_act = QAction("取消置顶" if self._always_on_top else "置顶显示", self)
        pin_act.triggered.connect(self.toggle_pin)
        menu.addAction(pin_act)

        reset_act = QAction("重置位置", self)
        reset_act.triggered.connect(self._reset_position)
        menu.addAction(reset_act)

        menu.exec(event.globalPosition().toPoint())

    def _choose_font(self) -> None:
        font, ok = QFontDialog.getFont(
            QFont(self._font_family, self._font_size, self._font_weight),
            self,
            "字幕字体",
        )
        if ok:
            self._font_family = font.family()
            self._font_size = font.pointSize()
            self._font_weight = font.weight()
            self._settings.setValue("fontFamily", self._font_family)
            self._settings.setValue("fontSize", self._font_size)
            self._settings.setValue("fontWeight", self._font_weight)
            self._apply_font()

    def _adjust_font_size(self, delta: int) -> None:
        self._font_size = max(10, min(96, self._font_size + delta))
        self._settings.setValue("fontSize", self._font_size)
        self._apply_font()

    def _choose_opacity(self) -> None:
        val, ok = QInputDialog.getDouble(
            self,
            "字幕透明度",
            "透明度（0.1 – 1.0）：",
            self._opacity,
            0.1, 1.0, 2,
        )
        if ok:
            self._opacity = val
            self._settings.setValue("opacity", self._opacity)
            self._apply_style()

    def adjust_opacity(self, delta: float) -> None:
        self._opacity = max(0.1, min(1.0, self._opacity + delta))
        self._settings.setValue("opacity", self._opacity)
        self._apply_style()

    def _reset_position(self) -> None:
        self._settings.remove("pos")
        screen = QApplication.primaryScreen().availableGeometry()
        self.move((screen.width() - self.width()) // 2, screen.height() - 210)
        self._save_position()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self in SubtitleOverlay._instances:
            SubtitleOverlay._instances.remove(self)
        super().closeEvent(event)

    @classmethod
    def hide_all(cls) -> None:
        for inst in cls._instances:
            inst.hide()
