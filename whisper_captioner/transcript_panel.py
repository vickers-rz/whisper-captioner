from __future__ import annotations

from PySide6.QtWidgets import QLabel, QListWidget, QVBoxLayout, QWidget


def build_transcript_panel(window) -> QWidget:
    window.transcript_list = QListWidget()
    window.transcript_list.setAlternatingRowColors(True)
    transcript_tab = QWidget()
    transcript_layout = QVBoxLayout(transcript_tab)
    transcript_layout.addWidget(QLabel("点击任意字幕行，可让视频跳转到该字幕开始时间。"))
    transcript_layout.addWidget(window.transcript_list)
    return transcript_tab
