"""
仪表盘面板 UI 构建模块

负责构建主界面顶部的运行面板。
提供模式选择、日志级别、核心播放控制、字幕浮窗控制以及缓存目录管理等界面的快速访问按钮。
"""
from __future__ import annotations

from PySide6.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QVBoxLayout


def build_dashboard_panel(window):
    header_card = QGroupBox("运行面板")
    header_layout = QVBoxLayout(header_card)
    mode_row = QHBoxLayout()
    mode_row.addWidget(QLabel("模式"))
    mode_row.addWidget(window.mode_combo)
    mode_row.addWidget(QLabel("日志级别"))
    mode_row.addWidget(window.log_level_combo)
    header_layout.addLayout(mode_row)

    dashboard_row = QHBoxLayout()

    action_card = QGroupBox("核心动作")
    action_layout = QVBoxLayout(action_card)
    action_layout.addWidget(QLabel("处理当前 Chrome 或输入框中的视频网址。"))
    action_layout.addWidget(window.controlled_button_2)
    action_layout.addWidget(window.stop_button)

    playback_card = QGroupBox("播放与同步")
    playback_layout = QVBoxLayout(playback_card)
    playback_row_1 = QHBoxLayout()
    playback_row_1.addWidget(window.rewind_5_button)
    playback_row_1.addWidget(window.rewind_button)
    playback_row_1.addWidget(window.play_pause_button)
    playback_row_1.addWidget(window.forward_5_button)
    playback_row_1.addWidget(window.forward_button)
    playback_layout.addLayout(playback_row_1)
    playback_row_2 = QHBoxLayout()
    playback_row_2.addWidget(window.subtitle_earlier_button)
    playback_row_2.addWidget(window.subtitle_later_button)
    playback_row_2.addWidget(window.subtitle_sync_button)
    playback_layout.addLayout(playback_row_2)

    overlay_card = QGroupBox("字幕浮窗")
    overlay_layout = QVBoxLayout(overlay_card)
    overlay_row_1 = QHBoxLayout()
    overlay_row_1.addWidget(window.overlay_show_button)
    overlay_row_1.addWidget(window.overlay_font_button)
    overlay_row_1.addWidget(window.overlay_smaller_button)
    overlay_row_1.addWidget(window.overlay_bigger_button)
    overlay_layout.addLayout(overlay_row_1)
    overlay_row_2 = QHBoxLayout()
    overlay_row_2.addWidget(window.overlay_less_opacity_button)
    overlay_row_2.addWidget(window.overlay_more_opacity_button)
    overlay_row_2.addWidget(window.overlay_reset_button)
    overlay_layout.addLayout(overlay_row_2)

    file_card = QGroupBox("缓存与文件")
    file_layout = QVBoxLayout(file_card)
    file_layout.addWidget(window.clear_cache_button)
    file_layout.addWidget(window.clear_local_audio_cache_button)
    file_layout.addWidget(window.open_cache_button)
    file_layout.addWidget(window.open_outputs_button)

    dashboard_row.addWidget(playback_card)
    dashboard_row.addWidget(overlay_card)
    dashboard_row.addWidget(file_card)
    dashboard_row.addWidget(action_card)
    return header_card, dashboard_row
