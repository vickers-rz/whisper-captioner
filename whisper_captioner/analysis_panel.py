from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QVBoxLayout, QWidget


def build_analysis_panel(window) -> QWidget:
    window.summarize_button = QPushButton("总结并分析视频")
    window.article_button = QPushButton("将字幕改写成文章")
    window.ask_button = QPushButton("基于字幕提问")

    window.analysis_output = QTextEdit()
    window.analysis_output.setReadOnly(True)
    window.analysis_output.setPlaceholderText("视频总结、分析、文章改写与问答结果会显示在这里。")
    window.analysis_question_input = QLineEdit()
    window.analysis_question_input.setPlaceholderText("输入一个基于当前字幕的深入问题")
    window.analysis_context_output = QTextEdit()
    window.analysis_context_output.setReadOnly(True)
    window.analysis_context_output.setPlaceholderText("相关字幕证据会显示在这里。")

    analysis_tab = QWidget()
    analysis_layout = QVBoxLayout(analysis_tab)
    analysis_layout.addWidget(
        QLabel("这里会复用当前视频已生成完成的字幕文本。请先运行“网址受控字幕”，或重新打开同一网址以加载缓存。")
    )
    analysis_buttons = QHBoxLayout()
    analysis_buttons.addWidget(window.summarize_button)
    analysis_buttons.addWidget(window.article_button)
    analysis_buttons.addWidget(window.ask_button)
    analysis_layout.addLayout(analysis_buttons)
    analysis_layout.addWidget(window.analysis_question_input)
    analysis_layout.addWidget(window.analysis_context_output)
    analysis_layout.addWidget(window.analysis_output)
    return analysis_tab
