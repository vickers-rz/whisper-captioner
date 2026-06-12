"""
UI Shell 构建模块

提供在不同环境下启动 QApplication 的辅助方法。
用于将核心界面的生命周期与系统的事件循环绑定，支持静默启动等选项。
"""
from __future__ import annotations

WINDOW_STYLESHEET = """
QMainWindow {
    background: #262624;
}
QGroupBox {
    border: 1px solid #4c4c49;
    border-radius: 14px;
    margin-top: 12px;
    padding: 12px;
    background: #2f2f2d;
    font-weight: 600;
    color: #f1f1ed;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}
QLabel {
    color: #efefe9;
}
QPushButton {
    min-height: 38px;
    border-radius: 10px;
    border: 1px solid #62625d;
    background: #6a6a68;
    color: #f5f5f2;
    padding: 6px 12px;
    font-weight: 600;
}
QPushButton:hover {
    background: #7b7b78;
}
QPushButton:pressed {
    background: #565653;
}
QToolButton {
    min-width: 28px;
    min-height: 28px;
    border-radius: 14px;
    border: 1px solid #62625d;
    background: #3d3d3a;
    color: #f5f5f2;
    font-weight: 700;
}
QToolButton:hover {
    background: #0b75d9;
}
QLineEdit, QComboBox, QTextEdit, QListWidget {
    border: 1px solid #575754;
    border-radius: 10px;
    background: #1f1f1d;
    color: #f3f3ef;
    padding: 8px;
}
QTabWidget::pane {
    border: 1px solid #4c4c49;
    border-radius: 12px;
    background: #2a2a28;
    top: -1px;
}
QTabBar::tab {
    background: #5a5a57;
    color: #f2f2ee;
    padding: 8px 18px;
    margin-right: 4px;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
}
QTabBar::tab:selected {
    background: #0b75d9;
}
QProgressBar {
    border: 1px solid #4c4c49;
    border-radius: 8px;
    background: #1f1f1d;
    color: #f3f3ef;
    text-align: center;
    min-height: 22px;
}
QProgressBar::chunk {
    background: #0b75d9;
    border-radius: 6px;
}
QLabel {
    color: #efefe9;
}
QLabel#StatusSummary {
    border: 1px solid #4c4c49;
    border-radius: 10px;
    background: #1f1f1d;
    color: #f3f3ef;
    padding: 8px 10px;
}
"""
