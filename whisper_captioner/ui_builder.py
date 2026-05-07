from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from whisper_captioner.analysis_panel import build_analysis_panel
from whisper_captioner.models import LLM_PROVIDERS, MODES
from whisper_captioner.transcript_panel import build_transcript_panel


def build_main_window_ui(window) -> None:
    window.mode_combo = QComboBox()
    for mode in MODES:
        suffix = "" if mode.available else " (model missing)"
        window.mode_combo.addItem(mode.label + suffix, mode.key)
    window.mode_combo.currentIndexChanged.connect(window._save_mode_selection)

    window.status = QTextEdit()
    window.status.setReadOnly(True)
    window.status.setPlaceholderText("状态日志")
    window.status.setMaximumHeight(150)
    window.status.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
    window.status_summary = QLabel("就绪")
    window.status_summary.setWordWrap(True)
    window.status_summary.setObjectName("StatusSummary")

    window.log_level_combo = QComboBox()
    window.log_level_combo.addItem("安静", "quiet")
    window.log_level_combo.addItem("普通", "normal")
    window.log_level_combo.addItem("调试", "debug")
    window.log_level_combo.addItem("追踪", "trace")
    window.log_level_combo.currentIndexChanged.connect(window._on_log_level_changed)

    window.progress_bar = QProgressBar()
    window.progress_bar.setVisible(False)
    window.progress_bar.setFormat("Chunk %v / %m")

    window.queue = QListWidget()
    window.url_input = QLineEdit()
    window.url_input.setPlaceholderText("粘贴视频网址或本地媒体文件路径")

    window.stop_button = QPushButton("停止")
    window.add_button = QPushButton("加入队列")
    window.up_button = QPushButton("上移")
    window.down_button = QPushButton("下移")
    window.process_button = QPushButton("处理队列")
    window.process_local_button = QPushButton("处理本地视频")
    window.realtime_button = QPushButton("实时字幕")
    realtime_help = (
        "<html><body style='white-space: normal;'>"
        "<b>实时字幕使用方法</b><br><br>"
        "1. 在 SoundSource 里选择 Chrome 或视频播放器。<br>"
        "2. 把该 App 的输出设备改为 Loopback 虚拟设备。<br>"
        "3. 如果想同时自己听到声音，用 SoundSource 输出组或 macOS 多输出设备，"
        "同时输出到扬声器/耳机和 Loopback。<br>"
        "4. 在这里设置 Loopback 输入编号，然后点“实时字幕”。<br><br>"
        "<b>链路</b><br>"
        "Chrome / 视频播放器 -> SoundSource -> Loopback -> whisper-stream -> 字幕浮窗<br><br>"
        "<b>提示</b><br>"
        "不知道编号时，到“设置”点“列出音频输入设备”。Loopback 常见编号是 0。"
        "</body></html>"
    )
    window.realtime_button.setToolTip(realtime_help)
    window.realtime_help_button = QToolButton()
    window.realtime_help_button.setText("?")
    window.realtime_help_button.setToolTip(realtime_help)
    window.realtime_help_button.setAutoRaise(True)
    window.realtime_help_button.setFixedSize(28, 28)
    window.list_audio_devices_button = QPushButton("列出音频输入设备")
    capture_id_help = (
        "<html><body style='white-space: normal;'>"
        "<b>Loopback 输入是什么意思？</b><br><br>"
        "这里填的是 <code>whisper-stream -c</code> 使用的音频捕获设备编号，"
        "也就是 Whisper 要监听哪一个 macOS 输入设备。<br><br>"
        "如果 SoundSource 已经把 Chrome / 视频播放器输出到 Loopback，"
        "这里就要填 Loopback 虚拟输入设备在 AVFoundation 里的编号。<br><br>"
        "<b>怎么确认编号</b><br>"
        "到“设置”点击“列出音频输入设备”，在日志里找 Loopback 或 Whisper Captions，"
        "前面的数字就是这里要填的值。<br><br>"
        "<b>常见情况</b><br>"
        "Loopback 经常是 0；如果没有字幕或日志提示找不到设备，就改成列表里的实际编号。"
        "</body></html>"
    )
    window.capture_id_input = QLineEdit()
    window.capture_id_input.setText("0")
    window.capture_id_input.setPlaceholderText("0")
    window.capture_id_input.setToolTip(capture_id_help)
    window.capture_id_help_button = QToolButton()
    window.capture_id_help_button.setText("?")
    window.capture_id_help_button.setToolTip(capture_id_help)
    window.capture_id_help_button.setAutoRaise(True)
    window.capture_id_help_button.setFixedSize(28, 28)
    window.controlled_button = QPushButton("网址受控字幕")
    window.controlled_button_2 = QPushButton("网址受控字幕")
    window.overlay_show_button = QPushButton("显示字幕浮窗")
    window.overlay_font_button = QPushButton("字体…")
    window.overlay_bigger_button = QPushButton("A+")
    window.overlay_smaller_button = QPushButton("A-")
    window.overlay_more_opacity_button = QPushButton("更不透明")
    window.overlay_less_opacity_button = QPushButton("更透明")
    window.overlay_reset_button = QPushButton("重置浮窗")
    window.clear_cache_button = QPushButton("删除当前视频缓存")
    window.open_cache_button = QPushButton("打开当前缓存")
    window.open_outputs_button = QPushButton("在 Finder 中定位文件")
    window.rewind_5_button = QPushButton("-5s")
    window.rewind_button = QPushButton("-10s")
    window.play_pause_button = QPushButton("暂停/继续")
    window.forward_5_button = QPushButton("+5s")
    window.forward_button = QPushButton("+10s")
    window.subtitle_earlier_button = QPushButton("Sub -0.5s")
    window.subtitle_later_button = QPushButton("Sub +0.5s")
    window.subtitle_sync_button = QPushButton("同步当前行")
    window.llm_group = QGroupBox("LLM 校对")
    window.llm_group.setCheckable(True)
    window.llm_group.setChecked(False)
    window.llm_provider_combo = QComboBox()
    for provider in LLM_PROVIDERS:
        window.llm_provider_combo.addItem(provider.label, provider.key)
    window.llm_api_key_input = QLineEdit()
    window.llm_api_key_input.setPlaceholderText("API Key（本地 Rapid-MLX 不需要）")
    window.llm_api_key_input.setEchoMode(QLineEdit.Password)
    window.llm_test_button = QPushButton("测试连接")
    window.llm_custom_url_input = QLineEdit()
    window.llm_custom_url_input.setPlaceholderText("自定义 API URL（用于自定义模式）")
    window.llm_custom_model_input = QLineEdit()
    window.llm_custom_model_input.setPlaceholderText("自定义模型 ID（用于自定义模式）")

    root = QWidget()
    root_layout = QVBoxLayout(root)
    root_layout.setContentsMargins(12, 12, 12, 12)

    header_card = QGroupBox("运行面板")
    header_layout = QVBoxLayout(header_card)
    mode_row = QHBoxLayout()
    mode_row.addWidget(QLabel("模式"))
    mode_row.addWidget(window.mode_combo)
    mode_row.addWidget(QLabel("日志级别"))
    mode_row.addWidget(window.log_level_combo)
    header_layout.addLayout(mode_row)
    root_layout.addWidget(header_card)

    dashboard_row = QHBoxLayout()

    action_card = QGroupBox("核心动作")
    action_layout = QVBoxLayout(action_card)
    action_layout.addWidget(QLabel("处理当前 Chrome 或输入框中的视频网址。"))
    action_layout.addWidget(window.controlled_button_2)
    realtime_row = QHBoxLayout()
    realtime_row.addWidget(QLabel("Loopback 输入"))
    realtime_row.addWidget(window.capture_id_input)
    realtime_row.addWidget(window.capture_id_help_button)
    action_layout.addLayout(realtime_row)
    realtime_button_row = QHBoxLayout()
    realtime_button_row.addWidget(window.realtime_button)
    realtime_button_row.addWidget(window.realtime_help_button)
    action_layout.addLayout(realtime_button_row)
    action_layout.addWidget(window.process_local_button)
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
    file_layout.addWidget(window.open_cache_button)
    file_layout.addWidget(window.open_outputs_button)

    dashboard_row.addWidget(playback_card)
    dashboard_row.addWidget(overlay_card)
    dashboard_row.addWidget(file_card)
    dashboard_row.addWidget(action_card)
    root_layout.addLayout(dashboard_row)

    tabs = QTabWidget()

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

    analysis_tab = build_analysis_panel(window)
    transcript_tab = build_transcript_panel(window)

    settings_tab = QWidget()
    settings_layout = QVBoxLayout(settings_tab)
    llm_layout = QVBoxLayout(window.llm_group)
    llm_layout.addWidget(QLabel("提供商："))
    llm_layout.addWidget(window.llm_provider_combo)
    llm_layout.addWidget(window.llm_custom_url_input)
    llm_layout.addWidget(window.llm_custom_model_input)
    llm_layout.addWidget(QLabel("API Key："))
    llm_layout.addWidget(window.llm_api_key_input)
    llm_layout.addWidget(window.llm_test_button)
    settings_layout.addWidget(window.llm_group)
    audio_route_card = QGroupBox("SoundSource / Loopback")
    audio_route_layout = QVBoxLayout(audio_route_card)
    audio_route_layout.addWidget(
        QLabel(
            "Chrome 或视频播放器 -> SoundSource 输出到 Loopback；"
            "想同时听到声音时，用 SoundSource 输出组或 macOS 多输出设备，同时送到扬声器和 Loopback。"
        )
    )
    audio_route_layout.addWidget(window.list_audio_devices_button)
    settings_layout.addWidget(audio_route_card)
    settings_layout.addStretch()

    tabs.addTab(queue_tab, "队列")
    tabs.addTab(analysis_tab, "分析")
    tabs.addTab(transcript_tab, "全文字幕")
    tabs.addTab(settings_tab, "设置")
    root_layout.addWidget(tabs)
    root_layout.addWidget(window.progress_bar)
    root_layout.addWidget(window.status_summary)
    root_layout.addWidget(window.status)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.Shape.NoFrame)
    scroll.setWidget(root)
    window.setCentralWidget(scroll)
