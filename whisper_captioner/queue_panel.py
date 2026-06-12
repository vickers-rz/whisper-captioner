"""
队列面板 UI 构建模块

负责构建主界面中的“队列”选项卡。
提供 URL 输入框、加入队列、上下移动和批量处理排队任务等控件布局。
"""
from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget


def build_queue_panel(window) -> QWidget:
    queue_tab = QWidget()
    queue_layout = QVBoxLayout(queue_tab)
    queue_layout.addWidget(window.url_input)
    queue_layout.addWidget(window.add_button)
    queue_layout.addWidget(window.queue)
    queue_row = QHBoxLayout()
    queue_row.addWidget(window.up_button)
    queue_row.addWidget(window.down_button)
    queue_row.addWidget(window.process_button)
    queue_layout.addLayout(queue_row)
    queue_layout.addWidget(window.controlled_button)
    return queue_tab
