from __future__ import annotations

import base64
import json
import threading
import time
import uuid
import urllib.parse
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import QSettings

from whisper_captioner.config import OUTPUT_DIR, QWEN_CHAT_DIR
from whisper_captioner.llm_handler import llm_generate_text, llm_provider_ready
from whisper_captioner.models import LLM_PROVIDERS, LLMProvider, SubtitleSegment
from whisper_captioner.subtitle_io import (
    format_srt_timestamp,
    parse_subtitle_file,
    save_segments_as_srt,
    save_segments_as_txt,
)


CHAT_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>字幕后处理工作台</title>
  <style>
    :root {
      --bg: #f3efe7;
      --paper: #fcfbf8;
      --sidebar: #ece3d5;
      --border: #d7c8b3;
      --text: #2c241d;
      --muted: #76695a;
      --accent: #9f4e2f;
      --accent-2: #d69058;
      --user: #fff3e5;
      --assistant: #fffefe;
      --warn: #fff2d9;
      --warn-border: #e3ba6c;
      --shadow: 0 12px 28px rgba(71, 49, 27, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "SF Pro Text", "PingFang SC", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(218, 171, 111, 0.24), transparent 30%),
        linear-gradient(180deg, #f8f2e8 0%, var(--bg) 100%);
      height: 100vh;
      overflow: hidden;
    }
    .app {
      display: grid;
      grid-template-columns: 300px minmax(0, 1fr) 320px;
      height: 100vh;
      overflow: hidden;
    }
    .panel {
      backdrop-filter: blur(10px);
      background: rgba(252, 251, 248, 0.82);
    }
    .sidebar {
      border-right: 1px solid var(--border);
      background: linear-gradient(180deg, rgba(255,255,255,0.52), rgba(236,227,213,0.96));
      padding: 16px 14px;
      overflow: auto;
      min-height: 0;
    }
    .workspace {
      display: grid;
      grid-template-rows: auto auto auto minmax(0, 1fr) auto;
      min-width: 0;
      min-height: 0;
      overflow: hidden;
    }
    .inspector {
      border-left: 1px solid var(--border);
      padding: 16px 14px;
      overflow: auto;
      background: linear-gradient(180deg, rgba(255,255,255,0.7), rgba(247,242,233,0.92));
      min-height: 0;
    }
    .brand {
      font-size: 18px;
      font-weight: 800;
      letter-spacing: 0.02em;
    }
    .sub {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
      margin-top: 6px;
    }
    .section-title {
      font-size: 12px;
      color: var(--muted);
      margin: 16px 0 8px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    button, select, input, textarea {
      font: inherit;
    }
    button {
      border: 0;
      border-radius: 14px;
      padding: 11px 14px;
      cursor: pointer;
      transition: transform 120ms ease, opacity 120ms ease;
    }
    button:hover { transform: translateY(-1px); }
    button:disabled { opacity: 0.5; cursor: default; transform: none; }
    .primary {
      color: white;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      box-shadow: var(--shadow);
    }
    .secondary {
      background: rgba(255,255,255,0.9);
      border: 1px solid var(--border);
      color: var(--text);
    }
    .ghost {
      background: rgba(255,255,255,0.54);
      border: 1px solid rgba(0,0,0,0.04);
      color: var(--text);
    }
    .sidebar-stack, .inspector-stack {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .list {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .item {
      border-radius: 14px;
      padding: 12px;
      background: rgba(255,255,255,0.58);
      border: 1px solid transparent;
      cursor: pointer;
    }
    .item.active {
      background: rgba(255,255,255,0.98);
      border-color: rgba(159, 78, 47, 0.25);
      box-shadow: var(--shadow);
    }
    .title {
      font-size: 14px;
      font-weight: 650;
      line-height: 1.35;
    }
    .meta {
      color: var(--muted);
      font-size: 12px;
      margin-top: 4px;
      line-height: 1.4;
    }
    .topbar {
      border-bottom: 1px solid var(--border);
      padding: 18px 22px 12px;
      background: rgba(252,251,248,0.88);
    }
    .topbar-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }
    .topbar h1 {
      margin: 0;
      font-size: 20px;
    }
    .status {
      color: var(--muted);
      font-size: 13px;
      margin-top: 6px;
      line-height: 1.45;
    }
    .toolbar {
      padding: 12px 22px;
      border-bottom: 1px solid var(--border);
      background: rgba(255,255,255,0.55);
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }
    .toolbar select {
      min-width: 180px;
      border-radius: 12px;
      padding: 10px 12px;
      border: 1px solid var(--border);
      background: white;
    }
    .toolbar-spacer {
      flex: 1 1 auto;
    }
    .danger {
      background: rgba(167, 43, 43, 0.12);
      border: 1px solid rgba(167, 43, 43, 0.2);
      color: #7b1d1d;
    }
    .messages {
      padding: 22px;
      overflow: auto;
      display: flex;
      flex-direction: column;
      gap: 16px;
      min-width: 0;
      min-height: 0;
    }
    .message {
      max-width: min(920px, 94%);
      padding: 16px 18px;
      border-radius: 20px;
      line-height: 1.7;
      white-space: pre-wrap;
      box-shadow: var(--shadow);
      border: 1px solid rgba(116, 94, 63, 0.08);
    }
    .message.user { align-self: flex-end; background: var(--user); }
    .message.assistant { align-self: flex-start; background: var(--assistant); }
    .empty {
      margin: auto;
      max-width: 760px;
      text-align: center;
      color: var(--muted);
      line-height: 1.8;
    }
    .warning {
      margin: 0 22px;
      margin-top: 14px;
      padding: 14px 16px;
      border-radius: 16px;
      background: var(--warn);
      border: 1px solid var(--warn-border);
      line-height: 1.6;
      display: none;
    }
    .composer {
      padding: 14px 22px 24px;
      border-top: 1px solid rgba(215, 200, 179, 0.7);
      background:
        linear-gradient(180deg, rgba(252,251,248,0.4), rgba(252,251,248,0.94) 22%, rgba(252,251,248,0.98) 100%);
      backdrop-filter: blur(12px);
      position: relative;
      z-index: 2;
    }
    .composer-shell {
      display: flex;
      flex-direction: column;
      gap: 10px;
      border: 1px solid rgba(215, 200, 179, 0.9);
      border-radius: 24px;
      background: rgba(255,255,255,0.96);
      box-shadow: 0 18px 32px rgba(58, 40, 23, 0.08);
      padding: 14px;
    }
    textarea {
      width: 100%;
      min-height: 92px;
      max-height: 240px;
      resize: none;
      border-radius: 18px;
      border: 0;
      background: transparent;
      padding: 8px 6px 2px;
      color: var(--text);
      outline: none;
    }
    .actions {
      display: flex;
      flex-direction: row;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }
    .composer-left {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    .composer-right {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-left: auto;
    }
    .chip {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 10px;
      border-radius: 999px;
      background: rgba(255,255,255,0.75);
      border: 1px solid var(--border);
      font-size: 12px;
      color: var(--muted);
    }
    .inspector-box {
      padding: 12px;
      border-radius: 14px;
      background: rgba(255,255,255,0.65);
      border: 1px solid rgba(0,0,0,0.04);
      line-height: 1.6;
    }
    .mini {
      font-size: 12px;
      color: var(--muted);
    }
    .mono {
      font-family: ui-monospace, "SFMono-Regular", Menlo, monospace;
      word-break: break-all;
    }
    .desktop-only {
      display: inline-flex;
    }
    .chip.clickable {
      cursor: pointer;
    }
    .chip.clickable:hover {
      border-color: rgba(159, 78, 47, 0.35);
      background: rgba(255,255,255,0.92);
    }
    .hidden { display: none; }
    @media (max-width: 1360px) {
      .app { grid-template-columns: 280px minmax(0, 1fr); }
      .inspector {
        position: fixed;
        top: 0;
        right: 0;
        width: min(340px, 88vw);
        height: 100vh;
        border-left: 1px solid var(--border);
        box-shadow: -24px 0 40px rgba(53, 36, 19, 0.14);
        z-index: 20;
        transform: translateX(100%);
        transition: transform 180ms ease;
      }
      .inspector.open {
        transform: translateX(0);
      }
    }
    @media (max-width: 880px) {
      .app { grid-template-columns: 1fr; }
      .sidebar { border-right: 0; border-bottom: 1px solid var(--border); }
      .workspace { grid-template-rows: auto auto auto minmax(0, 1fr) auto; }
      .actions { flex-direction: column; align-items: stretch; }
      .composer-right { margin-left: 0; width: 100%; justify-content: space-between; }
      .desktop-only { display: none; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar panel">
      <div class="brand">字幕后处理工作台</div>
      <div class="sub">这里不跑转录第一段。你可以直接上传第三方字幕，或加载之前已经生成的字幕，然后做规整、转写成文稿，或继续围绕字幕内容与 LLM 对话。</div>
      <div class="section-title">会话</div>
      <div class="sidebar-stack">
        <button id="new-chat" class="primary">新建工作会话</button>
        <button id="refresh" class="secondary">刷新</button>
      </div>
      <div id="conversation-list" class="list" style="margin-top: 12px;"></div>
      <div class="section-title">历史字幕资产</div>
      <div class="sidebar-stack">
        <button id="pick-file" class="secondary">上传 SRT / VTT / TXT</button>
        <input id="file-input" type="file" accept=".srt,.vtt,.txt" class="hidden">
      </div>
      <div id="asset-list" class="list" style="margin-top: 12px;"></div>
    </aside>
    <main class="workspace panel">
      <div class="topbar">
        <div class="topbar-row">
          <div>
            <h1 id="conversation-title">新工作会话</h1>
            <div class="status" id="conversation-status">就绪</div>
          </div>
          <div class="chip" id="subtitle-chip">未挂载字幕</div>
        </div>
      </div>
      <div class="toolbar">
        <select id="provider-select"></select>
        <button id="provider-settings" class="ghost">模型设置</button>
        <button id="attach-current-asset" class="secondary">把当前选中字幕挂到会话</button>
        <button id="cleanup" class="secondary">语句规整</button>
        <button id="article" class="secondary">转写成文稿</button>
        <div class="toolbar-spacer"></div>
        <button id="delete-conversation" class="danger">删除对话</button>
        <button id="toggle-inspector" class="ghost desktop-only">当前资产</button>
        <button id="open-instructions" class="ghost">提示词说明</button>
      </div>
      <div id="provider-panel" class="warning" style="display:none;">
        <div style="font-weight:700; margin-bottom:10px;">当前 Web LLM 设置</div>
        <div style="display:grid; gap:10px;">
          <select id="provider-editor-select"></select>
          <input id="provider-api-key" type="password" placeholder="API Key / Token">
          <input id="provider-api-url" type="text" placeholder="自定义 API URL（仅自定义供应商必填）">
          <input id="provider-model-id" type="text" placeholder="自定义 Model ID（仅自定义供应商必填）">
          <div style="display:flex; gap:10px; flex-wrap:wrap;">
            <button id="save-provider-settings" class="primary">保存当前供应商设置</button>
            <button id="close-provider-settings" class="secondary">收起</button>
          </div>
          <div class="mini" id="provider-help"></div>
        </div>
      </div>
      <div id="long-warning" class="warning"></div>
      <div id="messages" class="messages">
        <div class="empty">左侧可以选历史字幕或上传第三方字幕。挂到当前会话后，你既可以让模型直接回答字幕内容相关问题，也可以一键做“语句规整”与“转写成文稿”。</div>
      </div>
      <div class="composer">
        <div class="composer-shell">
          <textarea id="prompt" placeholder="围绕当前字幕提问，或直接输入后处理要求。按 Cmd/Ctrl + Enter 发送。"></textarea>
          <div class="actions">
            <div class="composer-left">
              <button id="composer-upload" class="ghost">添加文件</button>
              <div class="mini">长字幕建议切到 Gemini 2.5 Pro。</div>
            </div>
            <div class="composer-right">
              <button id="send" class="primary">发送</button>
            </div>
          </div>
        </div>
      </div>
    </main>
    <aside class="inspector panel">
      <div class="brand" style="font-size: 16px;">当前资产</div>
      <div class="sub">这里显示当前选中的字幕文件信息。你可以先在左侧点某份历史字幕，再把它挂到当前工作会话里。</div>
      <div class="inspector-stack" style="margin-top: 14px;">
        <div class="inspector-box">
          <div class="mini">文件名</div>
          <div id="asset-name">未选择</div>
        </div>
        <div class="inspector-box">
          <div class="mini">来源路径</div>
          <div id="asset-path" class="mono">-</div>
        </div>
        <div class="inspector-box">
          <div class="mini">内容规模</div>
          <div id="asset-size">-</div>
        </div>
        <div class="inspector-box">
          <div class="mini">预览</div>
          <div id="asset-preview">-</div>
        </div>
      </div>
    </aside>
  </div>
  <script>
    const state = {
      currentId: null,
      currentConversation: null,
      conversations: [],
      assets: [],
      selectedAssetId: null,
      sending: false,
      providers: [],
      config: null,
    };

    const els = {
      list: document.getElementById("conversation-list"),
      assetList: document.getElementById("asset-list"),
      title: document.getElementById("conversation-title"),
      status: document.getElementById("conversation-status"),
      messages: document.getElementById("messages"),
      prompt: document.getElementById("prompt"),
      send: document.getElementById("send"),
      newChat: document.getElementById("new-chat"),
      refresh: document.getElementById("refresh"),
      deleteConversation: document.getElementById("delete-conversation"),
      cleanup: document.getElementById("cleanup"),
      article: document.getElementById("article"),
      provider: document.getElementById("provider-select"),
      providerSettings: document.getElementById("provider-settings"),
      providerPanel: document.getElementById("provider-panel"),
      providerEditorSelect: document.getElementById("provider-editor-select"),
      providerApiKey: document.getElementById("provider-api-key"),
      providerApiUrl: document.getElementById("provider-api-url"),
      providerModelId: document.getElementById("provider-model-id"),
      saveProviderSettings: document.getElementById("save-provider-settings"),
      closeProviderSettings: document.getElementById("close-provider-settings"),
      providerHelp: document.getElementById("provider-help"),
      pickFile: document.getElementById("pick-file"),
      composerUpload: document.getElementById("composer-upload"),
      fileInput: document.getElementById("file-input"),
      attachCurrentAsset: document.getElementById("attach-current-asset"),
      subtitleChip: document.getElementById("subtitle-chip"),
      warning: document.getElementById("long-warning"),
      assetName: document.getElementById("asset-name"),
      assetPath: document.getElementById("asset-path"),
      assetSize: document.getElementById("asset-size"),
      assetPreview: document.getElementById("asset-preview"),
      openInstructions: document.getElementById("open-instructions"),
      toggleInspector: document.getElementById("toggle-inspector"),
      inspector: document.querySelector(".inspector"),
    };

    function fmtTime(value) {
      if (!value) return "";
      const d = new Date(value);
      if (Number.isNaN(d.getTime())) return value;
      return d.toLocaleString();
    }

    function setStatus(text) {
      els.status.textContent = text;
    }

    async function fetchJson(url, options = {}) {
      const resp = await fetch(url, {
        headers: { "Content-Type": "application/json" },
        ...options,
      });
      if (!resp.ok) {
        let detail = "";
        try {
          const data = await resp.json();
          detail = data.error || JSON.stringify(data);
        } catch (err) {
          detail = await resp.text();
        }
        throw new Error(detail || `HTTP ${resp.status}`);
      }
      return resp.json();
    }

    function shortText(text, max = 140) {
      if (!text) return "";
      return text.length > max ? text.slice(0, max) + "..." : text;
    }

    function renderConversationList() {
      els.list.innerHTML = "";
      for (const convo of state.conversations) {
        const item = document.createElement("div");
        item.className = "item" + (convo.id === state.currentId ? " active" : "");
        const subtitle = convo.subtitle ? `字幕: ${convo.subtitle.filename}` : "未挂字幕";
        item.innerHTML = `
          <div class="title">${convo.title || "新工作会话"}</div>
          <div class="meta">${subtitle}</div>
          <div class="meta">${convo.message_count || 0} 条消息 · ${fmtTime(convo.updated_at)}</div>
        `;
        item.addEventListener("click", () => openConversation(convo.id));
        els.list.appendChild(item);
      }
    }

    function renderAssetList() {
      els.assetList.innerHTML = "";
      for (const asset of state.assets) {
        const item = document.createElement("div");
        item.className = "item" + (asset.id === state.selectedAssetId ? " active" : "");
        item.innerHTML = `
          <div class="title">${asset.filename}</div>
          <div class="meta">${asset.segment_count} 段 · ${asset.char_count} 字符</div>
          <div class="meta">${asset.source_label}</div>
        `;
        item.addEventListener("click", () => selectAsset(asset.id));
        els.assetList.appendChild(item);
      }
    }

    function renderMessages(convo) {
      state.currentConversation = convo;
      els.messages.innerHTML = "";
      els.title.textContent = convo.title || "新工作会话";
      const subtitle = convo.subtitle;
      els.subtitleChip.textContent = subtitle ? `已挂字幕: ${subtitle.filename}` : "未挂载字幕";
      els.subtitleChip.classList.toggle("clickable", Boolean(subtitle));
      if (!convo.messages.length) {
        els.messages.innerHTML = `<div class="empty">当前会话还没有消息。${subtitle ? "你现在可以直接针对这份字幕提问，或点击上方动作按钮。" : "先从左侧选择一份字幕并挂到会话，或者直接聊天测试模型能力。"} </div>`;
      } else {
        for (const msg of convo.messages) {
          const div = document.createElement("div");
          div.className = "message " + msg.role;
          div.textContent = stripThinkingBlocks(msg.content);
          els.messages.appendChild(div);
        }
        els.messages.scrollTop = els.messages.scrollHeight;
      }
      updateWarning(convo);
      syncProvider(convo.provider_key || "");
    }

    function syncProvider(currentKey) {
      if (!state.providers.length) return;
      els.provider.innerHTML = "";
      els.providerEditorSelect.innerHTML = "";
      for (const provider of state.providers) {
        const option = document.createElement("option");
        option.value = provider.key;
        option.textContent = provider.label;
        els.provider.appendChild(option);
        const editorOption = document.createElement("option");
        editorOption.value = provider.key;
        editorOption.textContent = provider.label;
        els.providerEditorSelect.appendChild(editorOption);
      }
      els.provider.value = currentKey || state.config.default_provider_key;
      els.providerEditorSelect.value = els.provider.value;
      syncProviderEditor();
    }

    function stripThinkingBlocks(text) {
      if (!text) return "";
      return text.replace(/<think>[\\s\\S]*?<\\/think>/g, "").trim() || text;
    }

    function updateWarning(convo) {
      const warning = convo.long_context_warning;
      if (!warning) {
        els.warning.style.display = "none";
        els.warning.textContent = "";
        return;
      }
      els.warning.style.display = "block";
      const currentProvider = convo.provider_label || "";
      els.warning.textContent = `${warning} 当前模型：${currentProvider}。如果你已配置 Gemini API Key，建议直接切到 Gemini 2.5 Pro。`;
    }

    async function selectAsset(assetId) {
      state.selectedAssetId = assetId;
      renderAssetList();
      const asset = state.assets.find((item) => item.id === assetId);
      if (!asset) return;
      els.assetName.textContent = asset.filename;
      els.assetPath.textContent = asset.path || "-";
      els.assetSize.textContent = `${asset.segment_count} 段 · ${asset.char_count} 字符`;
      els.assetPreview.textContent = shortText(asset.preview, 320) || "-";
    }

    async function refreshConfig() {
      const data = await fetchJson("/api/config");
      state.providers = data.providers || [];
      state.config = data;
      syncProvider(data.default_provider_key);
    }

    async function refreshAssets(preserve = true) {
      const data = await fetchJson("/api/assets");
      state.assets = data.assets || [];
      renderAssetList();
      if (preserve && state.selectedAssetId) {
        const hit = state.assets.find((item) => item.id === state.selectedAssetId);
        if (hit) {
          await selectAsset(hit.id);
          return;
        }
      }
      if (state.assets.length) {
        await selectAsset(state.assets[0].id);
      }
    }

    async function refreshConversations(preferredId = state.currentId) {
      const data = await fetchJson("/api/conversations");
      state.conversations = data.conversations || [];
      renderConversationList();
      if (preferredId) {
        const hit = state.conversations.find((c) => c.id === preferredId);
        if (hit) {
          await openConversation(preferredId, false);
          return;
        }
      }
      if (!state.currentId && state.conversations.length) {
        await openConversation(state.conversations[0].id, false);
      }
    }

    async function createConversation() {
      const data = await fetchJson("/api/conversations", { method: "POST", body: JSON.stringify({}) });
      state.currentId = data.id;
      await refreshConversations(data.id);
      setStatus("已创建新工作会话");
      els.prompt.focus();
    }

    async function openConversation(id, refreshList = true) {
      const convo = await fetchJson(`/api/conversations/${id}`);
      state.currentId = convo.id;
      if (refreshList) {
        await refreshConversations(id);
      } else {
        renderConversationList();
      }
      renderMessages(convo);
      setStatus(`最后更新：${fmtTime(convo.updated_at)}`);
    }

    async function deleteConversation() {
      if (!state.currentId) {
        setStatus("当前没有可删除的对话");
        return;
      }
      if (!window.confirm("确定要删除当前对话吗？此操作不可撤销。")) {
        return;
      }
      try {
        await fetchJson(`/api/conversations/${state.currentId}/delete`, {
          method: "POST",
          body: JSON.stringify({}),
        });
        const removedId = state.currentId;
        state.currentId = null;
        state.currentConversation = null;
        await refreshConversations();
        if (!state.currentId) {
          els.messages.innerHTML = `<div class="empty">当前对话已删除。你可以新建工作会话，或先从左侧选择一份字幕继续处理。</div>`;
          els.title.textContent = "新工作会话";
          els.subtitleChip.textContent = "未挂载字幕";
          els.subtitleChip.classList.remove("clickable");
        }
        setStatus(`已删除对话：${removedId.slice(0, 8)}`);
      } catch (err) {
        setStatus(`删除失败：${err.message}`);
      }
    }

    async function ensureConversation() {
      if (!state.currentId) {
        await createConversation();
      }
    }

    async function sendMessage() {
      const content = els.prompt.value.trim();
      if (!content || state.sending) return;
      await ensureConversation();
      state.sending = true;
      els.send.disabled = true;
      setStatus("模型正在处理...");
      try {
        const convo = await fetchJson(`/api/conversations/${state.currentId}/messages`, {
          method: "POST",
          body: JSON.stringify({ content }),
        });
        els.prompt.value = "";
        renderMessages(convo);
        await refreshConversations(convo.id);
        setStatus(`已回复 · ${fmtTime(convo.updated_at)}`);
      } catch (err) {
        setStatus(`发送失败：${err.message}`);
      } finally {
        state.sending = false;
        els.send.disabled = false;
      }
    }

    async function setProvider() {
      if (!state.currentId) return;
      try {
        const convo = await fetchJson(`/api/conversations/${state.currentId}/provider`, {
          method: "POST",
          body: JSON.stringify({ provider_key: els.provider.value }),
        });
        renderMessages(convo);
        await refreshConversations(convo.id);
        setStatus(`已切换模型：${convo.provider_label}`);
      } catch (err) {
        setStatus(`切换失败：${err.message}`);
      }
    }

    function syncProviderEditor() {
      const currentKey = els.providerEditorSelect.value || els.provider.value;
      const provider = state.providers.find((item) => item.key === currentKey);
      if (!provider) return;
      els.providerApiKey.value = provider.api_key || "";
      els.providerApiUrl.value = provider.api_url || "";
      els.providerModelId.value = provider.model_id || "";
      els.providerApiUrl.style.display = provider.key === "custom" ? "block" : "none";
      els.providerModelId.style.display = provider.key === "custom" ? "block" : "none";
      els.providerHelp.textContent = provider.help_text || "";
    }

    async function saveProviderSettings() {
      const providerKey = els.providerEditorSelect.value;
      try {
        const data = await fetchJson("/api/provider_settings", {
          method: "POST",
          body: JSON.stringify({
            provider_key: providerKey,
            api_key: els.providerApiKey.value,
            api_url: els.providerApiUrl.value,
            model_id: els.providerModelId.value,
          }),
        });
        state.providers = data.providers || [];
        state.config = data;
        syncProvider(els.provider.value || data.default_provider_key);
        setStatus(`已保存 ${providerKey} 配置`);
      } catch (err) {
        setStatus(`保存失败：${err.message}`);
      }
    }

    async function attachSelectedAsset() {
      if (!state.selectedAssetId) {
        setStatus("请先在左侧选一份字幕");
        return;
      }
      await ensureConversation();
      try {
        const convo = await fetchJson(`/api/conversations/${state.currentId}/attach_subtitle`, {
          method: "POST",
          body: JSON.stringify({ asset_id: state.selectedAssetId }),
        });
        renderMessages(convo);
        await refreshConversations(convo.id);
        setStatus(`已挂载字幕：${convo.subtitle.filename}`);
      } catch (err) {
        setStatus(`挂载失败：${err.message}`);
      }
    }

    async function detachSubtitle() {
      if (!state.currentId || !state.currentConversation?.subtitle) {
        return;
      }
      try {
        const convo = await fetchJson(`/api/conversations/${state.currentId}/detach_subtitle`, {
          method: "POST",
          body: JSON.stringify({}),
        });
        renderMessages(convo);
        await refreshConversations(convo.id);
        setStatus("已解除当前对话挂载的字幕");
      } catch (err) {
        setStatus(`解除挂载失败：${err.message}`);
      }
    }

    async function runAction(action) {
      await ensureConversation();
      setStatus(action === "cleanup" ? "正在规整字幕..." : "正在转写成文稿...");
      try {
        const convo = await fetchJson(`/api/conversations/${state.currentId}/actions`, {
          method: "POST",
          body: JSON.stringify({ action }),
        });
        renderMessages(convo);
        await refreshConversations(convo.id);
        setStatus(`动作完成 · ${fmtTime(convo.updated_at)}`);
      } catch (err) {
        setStatus(`动作失败：${err.message}`);
      }
    }

    async function uploadFile(file) {
      const bytes = await file.arrayBuffer();
      const binary = new Uint8Array(bytes);
      let text = "";
      const chunk = 0x8000;
      for (let i = 0; i < binary.length; i += chunk) {
        text += String.fromCharCode(...binary.slice(i, i + chunk));
      }
      const contentBase64 = btoa(text);
      const asset = await fetchJson("/api/assets/upload", {
        method: "POST",
        body: JSON.stringify({
          filename: file.name,
          content_base64: contentBase64,
        }),
      });
      await refreshAssets(false);
      await selectAsset(asset.id);
      setStatus(`已导入字幕：${asset.filename}`);
    }

    els.send.addEventListener("click", sendMessage);
    els.newChat.addEventListener("click", createConversation);
    els.refresh.addEventListener("click", async () => {
      await refreshAssets();
      await refreshConversations();
      setStatus("已刷新");
    });
    els.deleteConversation.addEventListener("click", deleteConversation);
    els.prompt.addEventListener("keydown", (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        event.preventDefault();
        sendMessage();
      }
    });
    els.provider.addEventListener("change", setProvider);
    els.providerSettings.addEventListener("click", () => {
      const isHidden = els.providerPanel.style.display === "none";
      els.providerPanel.style.display = isHidden ? "block" : "none";
      els.providerEditorSelect.value = els.provider.value;
      syncProviderEditor();
    });
    els.providerEditorSelect.addEventListener("change", syncProviderEditor);
    els.saveProviderSettings.addEventListener("click", saveProviderSettings);
    els.closeProviderSettings.addEventListener("click", () => {
      els.providerPanel.style.display = "none";
    });
    els.attachCurrentAsset.addEventListener("click", attachSelectedAsset);
    els.subtitleChip.addEventListener("click", () => {
      if (state.currentConversation?.subtitle) {
        detachSubtitle();
      }
    });
    els.cleanup.addEventListener("click", () => runAction("cleanup"));
    els.article.addEventListener("click", () => runAction("article"));
    els.pickFile.addEventListener("click", () => els.fileInput.click());
    els.composerUpload.addEventListener("click", () => els.fileInput.click());
    els.fileInput.addEventListener("change", async () => {
      const file = els.fileInput.files[0];
      if (!file) return;
      try {
        await uploadFile(file);
      } catch (err) {
        setStatus(`上传失败：${err.message}`);
      } finally {
        els.fileInput.value = "";
      }
    });
    els.openInstructions.addEventListener("click", () => {
      const lines = [
        "1. 先上传或选中一份字幕。",
        "2. 点击“把当前选中字幕挂到会话”。",
        "3. 之后可直接对字幕提问，也可点“语句规整”或“转写成文稿”。",
        "4. 如果字幕很长，建议切换到 Gemini 2.5 Pro。"
      ];
      setStatus(lines.join(" "));
    });
    els.toggleInspector.addEventListener("click", () => {
      els.inspector.classList.toggle("open");
    });

    Promise.all([refreshConfig(), refreshAssets(), refreshConversations()])
      .catch((err) => setStatus(`初始化失败：${err.message}`));
  </script>
</body>
</html>
"""


SUPPORTED_PROVIDER_KEYS = {
    "local_rapidmlx_8b",
    "gpt4o_mini",
    "gpt4o",
    "deepseek",
    "gemini_flash",
    "gemini_pro",
    "minimax_m27",
    "claude_sonnet",
    "custom",
}
DEFAULT_PROVIDER_KEY = "local_rapidmlx_8b"
LONG_CONTEXT_CHAR_THRESHOLD = 45000
LONG_CONTEXT_SEGMENT_THRESHOLD = 1200
UPLOAD_DIRNAME = "uploads"
EXPORT_DIRNAME = "exports"


class QwenChatServiceManager:
    def __init__(
        self,
        storage_dir: Path = QWEN_CHAT_DIR,
        host: str = "127.0.0.1",
        port: int = 8767,
    ) -> None:
        self.storage_dir = storage_dir
        self.host = host
        self.port = port
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def uploads_dir(self) -> Path:
        return self.storage_dir / UPLOAD_DIRNAME

    @property
    def exports_dir(self) -> Path:
        return self.storage_dir / EXPORT_DIRNAME

    def is_running(self) -> bool:
        return self._server is not None and self._thread is not None and self._thread.is_alive()

    def start(self) -> str:
        with self._lock:
            if self.is_running():
                return self.base_url
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self.uploads_dir.mkdir(parents=True, exist_ok=True)
            self.exports_dir.mkdir(parents=True, exist_ok=True)
            server = ThreadingHTTPServer((self.host, self.port), self._build_handler())
            self._server = server
            self._thread = threading.Thread(target=server.serve_forever, daemon=True, name="QwenChatService")
            self._thread.start()
            return self.base_url

    def stop(self) -> None:
        with self._lock:
            if not self._server:
                return
            self._server.shutdown()
            self._server.server_close()
            self._server = None
            self._thread = None

    def _build_handler(self):
        manager = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                parsed = urllib.parse.urlparse(self.path)
                if parsed.path == "/":
                    self._write_html(CHAT_HTML)
                    return
                if parsed.path == "/api/health":
                    self._write_json({"ok": True, "base_url": manager.base_url})
                    return
                if parsed.path == "/api/config":
                    self._write_json(manager._config_payload())
                    return
                if parsed.path == "/api/assets":
                    self._write_json({"assets": manager._list_assets()})
                    return
                if parsed.path == "/api/conversations":
                    self._write_json({"conversations": manager._list_conversations()})
                    return
                if parsed.path.startswith("/api/conversations/"):
                    convo_id = parsed.path.removeprefix("/api/conversations/")
                    convo = manager._load_conversation(convo_id)
                    if not convo:
                        self._write_json({"error": "Conversation not found"}, status=HTTPStatus.NOT_FOUND)
                        return
                    self._write_json(manager._decorate_conversation(convo))
                    return
                self._write_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:
                parsed = urllib.parse.urlparse(self.path)
                payload = self._read_json_body()
                if parsed.path == "/api/conversations":
                    convo = manager._create_conversation(payload.get("title") if isinstance(payload, dict) else None)
                    self._write_json(manager._decorate_conversation(convo), status=HTTPStatus.CREATED)
                    return
                if parsed.path == "/api/assets/upload":
                    try:
                        asset = manager._upload_asset(payload)
                    except Exception as exc:
                        self._write_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                        return
                    self._write_json(asset, status=HTTPStatus.CREATED)
                    return
                if parsed.path == "/api/provider_settings":
                    try:
                        config = manager._save_provider_settings(payload)
                    except Exception as exc:
                        self._write_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                        return
                    self._write_json(config)
                    return
                if parsed.path == "/api/conversations/from-asset":
                    try:
                        convo = manager._create_conversation_from_asset(str(payload.get("asset_id", "")))
                    except Exception as exc:
                        self._write_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                        return
                    self._write_json(manager._decorate_conversation(convo), status=HTTPStatus.CREATED)
                    return
                if parsed.path.startswith("/api/conversations/") and parsed.path.endswith("/messages"):
                    convo_id = parsed.path.removeprefix("/api/conversations/").removesuffix("/messages")
                    content = str(payload.get("content", "")).strip() if isinstance(payload, dict) else ""
                    if not content:
                        self._write_json({"error": "Message content is required"}, status=HTTPStatus.BAD_REQUEST)
                        return
                    try:
                        convo = manager._append_and_reply(convo_id, content)
                    except Exception as exc:
                        self._write_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                        return
                    self._write_json(manager._decorate_conversation(convo))
                    return
                if parsed.path.startswith("/api/conversations/") and parsed.path.endswith("/attach_subtitle"):
                    convo_id = parsed.path.removeprefix("/api/conversations/").removesuffix("/attach_subtitle")
                    try:
                        convo = manager._attach_subtitle(convo_id, payload)
                    except Exception as exc:
                        self._write_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                        return
                    self._write_json(manager._decorate_conversation(convo))
                    return
                if parsed.path.startswith("/api/conversations/") and parsed.path.endswith("/detach_subtitle"):
                    convo_id = parsed.path.removeprefix("/api/conversations/").removesuffix("/detach_subtitle")
                    try:
                        convo = manager._detach_subtitle(convo_id)
                    except Exception as exc:
                        self._write_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                        return
                    self._write_json(manager._decorate_conversation(convo))
                    return
                if parsed.path.startswith("/api/conversations/") and parsed.path.endswith("/delete"):
                    convo_id = parsed.path.removeprefix("/api/conversations/").removesuffix("/delete")
                    try:
                        manager._delete_conversation(convo_id)
                    except Exception as exc:
                        self._write_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                        return
                    self._write_json({"ok": True, "id": convo_id})
                    return
                if parsed.path.startswith("/api/conversations/") and parsed.path.endswith("/actions"):
                    convo_id = parsed.path.removeprefix("/api/conversations/").removesuffix("/actions")
                    try:
                        convo = manager._run_action(convo_id, str(payload.get("action", "")))
                    except Exception as exc:
                        self._write_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                        return
                    self._write_json(manager._decorate_conversation(convo))
                    return
                if parsed.path.startswith("/api/conversations/") and parsed.path.endswith("/provider"):
                    convo_id = parsed.path.removeprefix("/api/conversations/").removesuffix("/provider")
                    try:
                        convo = manager._set_provider(convo_id, str(payload.get("provider_key", "")))
                    except Exception as exc:
                        self._write_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                        return
                    self._write_json(manager._decorate_conversation(convo))
                    return
                self._write_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

            def log_message(self, format: str, *args: object) -> None:
                return

            def _read_json_body(self) -> Any:
                length = int(self.headers.get("Content-Length", "0") or "0")
                if length <= 0:
                    return {}
                raw = self.rfile.read(length)
                if not raw:
                    return {}
                try:
                    return json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    return {}

            def _write_html(self, html_text: str) -> None:
                data = html_text.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _write_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        return Handler

    def _conversation_path(self, convo_id: str) -> Path:
        return self.storage_dir / f"{convo_id}.json"

    def _now(self) -> str:
        return datetime.now().isoformat(timespec="seconds")

    def _provider_map(self) -> dict[str, LLMProvider]:
        return {provider.key: provider for provider in LLM_PROVIDERS if provider.key in SUPPORTED_PROVIDER_KEYS}

    def _default_provider_key(self) -> str:
        settings = QSettings("WhisperCaptioner", "App")
        key = str(settings.value("llm/provider", DEFAULT_PROVIDER_KEY))
        return key if key in SUPPORTED_PROVIDER_KEYS else DEFAULT_PROVIDER_KEY

    def _provider_config(self, provider_key: str) -> tuple[LLMProvider, str, str, str]:
        providers = self._provider_map()
        provider = providers.get(provider_key)
        if not provider:
            raise RuntimeError(f"Unsupported provider: {provider_key}")
        settings = QSettings("WhisperCaptioner", "App")
        api_key = str(settings.value(f"llm/apikey/{provider.key}", ""))
        api_url = str(settings.value("llm/custom_url", "")) if provider.key == "custom" else ""
        model_id = str(settings.value("llm/custom_model", "")) if provider.key == "custom" else ""
        return provider, api_key, api_url, model_id

    def _config_payload(self) -> dict[str, Any]:
        settings = QSettings("WhisperCaptioner", "App")
        provider_map = self._provider_map()
        providers = []
        ordered_keys = (
            "local_rapidmlx_8b",
            "gpt4o_mini",
            "gpt4o",
            "deepseek",
            "gemini_flash",
            "gemini_pro",
            "minimax_m27",
            "claude_sonnet",
            "custom",
        )
        for key in ordered_keys:
            provider = provider_map[key]
            api_key = str(settings.value(f"llm/apikey/{provider.key}", ""))
            api_url = str(settings.value("llm/custom_url", "")) if provider.key == "custom" else provider.api_url
            model_id = str(settings.value("llm/custom_model", "")) if provider.key == "custom" else provider.model_id
            providers.append(
                {
                    "key": provider.key,
                    "label": provider.label,
                    "requires_api_key": provider.requires_api_key,
                    "ready": llm_provider_ready(provider, api_key),
                    "api_key": api_key,
                    "api_url": api_url,
                    "model_id": model_id,
                    "help_text": self._provider_help_text(provider.key),
                }
            )
        return {
            "default_provider_key": self._default_provider_key(),
            "providers": providers,
        }

    def _provider_help_text(self, provider_key: str) -> str:
        if provider_key == "local_rapidmlx_8b":
            return "本地 Rapid-MLX，不需要 API Key。"
        if provider_key == "gemini_flash":
            return "使用 Google Gemini 2.5 Flash。请在此填入 Gemini API Key。"
        if provider_key == "gemini_pro":
            return "长字幕或长文稿建议切到 Gemini 2.5 Pro。"
        if provider_key == "minimax_m27":
            return (
                "MiniMAX Token Plan 走 Anthropic 兼容接口。默认 URL 为 https://api.minimaxi.com/anthropic ，"
                "模型默认是 MiniMax-M2.7。"
            )
        if provider_key == "custom":
            return "自定义 OpenAI-compatible 接口时，请同时填写 API URL 和 Model ID。"
        return "填写对应供应商的 API Key 后即可在 Web 工作台里直接切换使用。"

    def _save_provider_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        provider_key = str(payload.get("provider_key", "")).strip()
        if provider_key not in self._provider_map():
            raise RuntimeError("Unsupported provider")
        settings = QSettings("WhisperCaptioner", "App")
        settings.setValue(f"llm/apikey/{provider_key}", str(payload.get("api_key", "")).strip())
        if provider_key == "custom":
            settings.setValue("llm/custom_url", str(payload.get("api_url", "")).strip())
            settings.setValue("llm/custom_model", str(payload.get("model_id", "")).strip())
        return self._config_payload()

    def _read_conversation_file(self, path: Path) -> Optional[dict[str, Any]]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        if not isinstance(data.get("messages", []), list):
            return None
        return data

    def _load_conversation(self, convo_id: str) -> Optional[dict[str, Any]]:
        return self._read_conversation_file(self._conversation_path(convo_id))

    def _save_conversation(self, convo: dict[str, Any]) -> None:
        path = self._conversation_path(str(convo["id"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(convo, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def _asset_from_path(self, path: Path, source_label: str) -> Optional[dict[str, Any]]:
        if not path.exists() or not path.is_file():
            return None
        try:
            segments = parse_subtitle_file(path)
        except Exception:
            return None
        if not segments:
            return None
        transcript = self._segments_to_plain_text(segments)
        asset_id = str(path.resolve())
        return {
            "id": asset_id,
            "filename": path.name,
            "path": str(path.resolve()),
            "source_label": source_label,
            "segment_count": len(segments),
            "char_count": len(transcript),
            "preview": transcript[:400],
            "segments": [self._segment_payload(segment) for segment in segments],
            "transcript_text": transcript,
            "timestamped_text": self._segments_to_timestamped_text(segments),
        }

    def _list_assets(self) -> list[dict[str, Any]]:
        assets: list[dict[str, Any]] = []
        seen: set[str] = set()
        dedupe: dict[str, dict[str, Any]] = {}
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

        for path in sorted(self.uploads_dir.glob("*")):
            asset = self._asset_from_path(path, "手动上传")
            if asset and asset["id"] not in seen:
                seen.add(asset["id"])
                dedupe[self._asset_dedupe_key(asset)] = asset

        excluded_roots = {
            self.storage_dir.resolve(),
            (OUTPUT_DIR / "cache").resolve(),
            (OUTPUT_DIR / "logs").resolve(),
            (OUTPUT_DIR / "notes").resolve(),
        }
        for path in sorted(OUTPUT_DIR.rglob("*")):
            if path.suffix.lower() not in {".srt", ".vtt", ".txt"}:
                continue
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if any(root == resolved or root in resolved.parents for root in excluded_roots):
                continue
            asset = self._asset_from_path(path, "历史生成")
            if asset and asset["id"] not in seen:
                seen.add(asset["id"])
                key = self._asset_dedupe_key(asset)
                existing = dedupe.get(key)
                if existing is None or self._prefer_asset(asset, existing):
                    dedupe[key] = asset

        assets = list(dedupe.values())
        assets.sort(key=lambda item: (item["source_label"] != "手动上传", item["filename"].lower()))
        return assets

    def _asset_dedupe_key(self, asset: dict[str, Any]) -> str:
        return Path(str(asset["filename"])).stem.lower()

    def _prefer_asset(self, candidate: dict[str, Any], current: dict[str, Any]) -> bool:
        candidate_suffix = Path(str(candidate["filename"])).suffix.lower()
        current_suffix = Path(str(current["filename"])).suffix.lower()
        if candidate_suffix == ".txt" and current_suffix != ".txt":
            return True
        if candidate_suffix != ".txt" and current_suffix == ".txt":
            return False
        return str(candidate["filename"]).lower() < str(current["filename"]).lower()

    def _find_asset(self, asset_id: str) -> Optional[dict[str, Any]]:
        for asset in self._list_assets():
            if asset["id"] == asset_id:
                return asset
        return None

    def _list_conversations(self) -> list[dict[str, Any]]:
        conversations: list[dict[str, Any]] = []
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(self.storage_dir.glob("*.json"), reverse=True):
            convo = self._read_conversation_file(path)
            if not convo:
                continue
            subtitle = convo.get("subtitle") if isinstance(convo.get("subtitle"), dict) else None
            conversations.append(
                {
                    "id": convo["id"],
                    "title": convo.get("title", "新工作会话"),
                    "updated_at": convo.get("updated_at"),
                    "message_count": len(convo.get("messages", [])),
                    "subtitle": {"filename": subtitle.get("filename")} if subtitle else None,
                }
            )
        conversations.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        return conversations

    def _create_conversation(self, title: Optional[str] = None) -> dict[str, Any]:
        now = self._now()
        convo = {
            "id": uuid.uuid4().hex,
            "title": title or "新工作会话",
            "created_at": now,
            "updated_at": now,
            "provider_key": self._default_provider_key(),
            "messages": [],
            "subtitle": None,
        }
        self._save_conversation(convo)
        return convo

    def _create_conversation_from_asset(self, asset_id: str) -> dict[str, Any]:
        convo = self._create_conversation()
        return self._attach_subtitle(convo["id"], {"asset_id": asset_id})

    def _set_provider(self, convo_id: str, provider_key: str) -> dict[str, Any]:
        convo = self._load_required_conversation(convo_id)
        if provider_key not in self._provider_map():
            raise RuntimeError("Unsupported provider")
        convo["provider_key"] = provider_key
        convo["updated_at"] = self._now()
        self._save_conversation(convo)
        return convo

    def _attach_subtitle(self, convo_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        convo = self._load_required_conversation(convo_id)
        asset_id = str(payload.get("asset_id", "")).strip()
        asset = self._find_asset(asset_id) if asset_id else None
        if not asset and payload.get("filename") and payload.get("content_base64"):
            asset = self._upload_asset(payload)
        if not asset:
            raise RuntimeError("Subtitle asset not found")
        convo["subtitle"] = {
            "asset_id": asset["id"],
            "filename": asset["filename"],
            "path": asset["path"],
            "source_label": asset["source_label"],
            "segment_count": asset["segment_count"],
            "char_count": asset["char_count"],
            "preview": asset["preview"],
            "timestamped_text": asset["timestamped_text"],
            "transcript_text": asset["transcript_text"],
            "segments": asset["segments"],
            "attached_at": self._now(),
        }
        if convo.get("title", "新工作会话") == "新工作会话":
            convo["title"] = f"{asset['filename']} 工作台"
        convo["updated_at"] = self._now()
        self._save_conversation(convo)
        return convo

    def _detach_subtitle(self, convo_id: str) -> dict[str, Any]:
        convo = self._load_required_conversation(convo_id)
        convo["subtitle"] = None
        convo["updated_at"] = self._now()
        self._save_conversation(convo)
        return convo

    def _delete_conversation(self, convo_id: str) -> None:
        path = self._conversation_path(convo_id)
        if not path.exists():
            raise RuntimeError("Conversation not found")
        path.unlink()

    def _append_and_reply(self, convo_id: str, content: str) -> dict[str, Any]:
        convo = self._load_required_conversation(convo_id)
        convo.setdefault("messages", []).append(
            {
                "id": uuid.uuid4().hex,
                "role": "user",
                "content": content,
                "created_at": self._now(),
            }
        )
        if convo.get("title", "新工作会话") == "新工作会话":
            convo["title"] = self._derive_title(content)
        assistant_text = self._chat_for_conversation(convo, content)
        convo["messages"].append(
            {
                "id": uuid.uuid4().hex,
                "role": "assistant",
                "content": assistant_text,
                "created_at": self._now(),
            }
        )
        convo["updated_at"] = self._now()
        self._save_conversation(convo)
        return convo

    def _run_action(self, convo_id: str, action: str) -> dict[str, Any]:
        convo = self._load_required_conversation(convo_id)
        subtitle = convo.get("subtitle")
        if not subtitle:
            raise RuntimeError("请先给当前会话挂载一份字幕")
        if action not in {"cleanup", "article"}:
            raise RuntimeError("Unsupported action")

        prompt, system_prompt = self._action_prompt(action, subtitle)
        result = self._generate_reply(convo, prompt, system_prompt)

        label = "语句规整结果" if action == "cleanup" else "转写成文稿结果"
        convo.setdefault("messages", []).append(
            {
                "id": uuid.uuid4().hex,
                "role": "user",
                "content": f"[系统动作] {label}",
                "created_at": self._now(),
            }
        )
        convo["messages"].append(
            {
                "id": uuid.uuid4().hex,
                "role": "assistant",
                "content": result,
                "created_at": self._now(),
            }
        )
        self._export_action_result(convo, action, result)
        convo["updated_at"] = self._now()
        self._save_conversation(convo)
        return convo

    def _upload_asset(self, payload: dict[str, Any]) -> dict[str, Any]:
        filename = Path(str(payload.get("filename", "")).strip()).name
        if not filename:
            raise RuntimeError("filename is required")
        suffix = Path(filename).suffix.lower()
        if suffix not in {".srt", ".vtt", ".txt"}:
            raise RuntimeError("Only .srt, .vtt, and .txt are supported")
        content_base64 = str(payload.get("content_base64", "")).strip()
        if not content_base64:
            raise RuntimeError("content_base64 is required")
        try:
            raw = base64.b64decode(content_base64)
        except Exception as exc:
            raise RuntimeError("Invalid base64 content") from exc
        path = self.uploads_dir / f"{int(time.time())}-{uuid.uuid4().hex[:8]}-{filename}"
        path.write_bytes(raw)
        asset = self._asset_from_path(path, "手动上传")
        if not asset:
            raise RuntimeError("Uploaded file could not be parsed as subtitle content")
        return asset

    def _segment_payload(self, segment: SubtitleSegment) -> dict[str, Any]:
        return {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
        }

    def _segments_to_plain_text(self, segments: list[SubtitleSegment]) -> str:
        return "\n".join(segment.text for segment in segments if segment.text.strip())

    def _segments_to_timestamped_text(self, segments: list[SubtitleSegment]) -> str:
        return "\n".join(
            f"[{format_srt_timestamp(segment.start)} - {format_srt_timestamp(segment.end)}] {segment.text}"
            for segment in segments
            if segment.text.strip()
        )

    def _action_prompt(self, action: str, subtitle: dict[str, Any]) -> tuple[str, str]:
        transcript_text = str(subtitle.get("transcript_text", ""))
        timestamped_text = str(subtitle.get("timestamped_text", ""))
        if action == "cleanup":
            system_prompt = (
                "你是一个严谨的中文字幕后处理助手。请基于用户提供的字幕内容做语句规整，"
                "消除明显口癖、冗余停顿和标点混乱，使其更适合阅读，但不要改变原意，不要臆造新信息。"
            )
            prompt = (
                "请对下面字幕做语句规整，输出整理后的完整文本。"
                "如果原字幕按行分句已经较清晰，可以适度保留分段。\n\n"
                f"{transcript_text}"
            )
            return prompt, system_prompt

        system_prompt = (
            "你是一个严谨的中文文稿整理助手。请仅根据用户提供的字幕内容写成更适合阅读的完整文稿，"
            "可以按自然段组织，但不要杜撰字幕中没有的信息。"
        )
        prompt = (
            "请把下面字幕转写成一篇顺畅、可读的中文文稿。"
            "如果内容明显是教程或访谈，请保留核心逻辑与层次。\n\n"
            f"{timestamped_text}"
        )
        return prompt, system_prompt

    def _chat_for_conversation(self, convo: dict[str, Any], user_content: str) -> str:
        subtitle = convo.get("subtitle")
        if subtitle:
            system_prompt = (
                "你是一个字幕内容助手。优先严格基于当前挂载的字幕回答问题，不要虚构字幕中没有出现的信息。"
                "如果用户要求的是整理、总结、提炼、解释，请明确只依据字幕内容完成。"
            )
            prompt = (
                "下面是当前挂载字幕的全文，请围绕它回答用户。\n\n"
                "【字幕全文】\n"
                f"{subtitle.get('timestamped_text', '')}\n\n"
                "【用户问题】\n"
                f"{user_content}"
            )
            return self._generate_reply(convo, prompt, system_prompt)

        system_prompt = "你是一个严谨的视频字幕与文本后处理助手。"
        return self._generate_reply(convo, user_content, system_prompt)

    def _generate_reply(self, convo: dict[str, Any], user_text: str, system_prompt: str) -> str:
        provider_key = str(convo.get("provider_key") or self._default_provider_key())
        provider, api_key, api_url, model_id = self._provider_config(provider_key)
        if provider.requires_api_key and not api_key:
            raise RuntimeError(f"{provider.label} 需要先在桌面 App 设置里填写 API Key")
        return llm_generate_text(
            user_text,
            provider,
            api_key,
            api_url,
            model_id,
            system_prompt=system_prompt,
            timeout=300,
            max_tokens=24000 if provider.key == "gemini_pro" else 16000,
        )

    def _export_action_result(self, convo: dict[str, Any], action: str, result: str) -> None:
        subtitle = convo.get("subtitle") or {}
        base_name = Path(str(subtitle.get("filename", "subtitle"))).stem
        stamp = time.strftime("%Y%m%d-%H%M%S")
        safe_action = "cleanup" if action == "cleanup" else "article"
        out_dir = self.exports_dir / convo["id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        txt_path = out_dir / f"{base_name}-{safe_action}-{stamp}.txt"
        txt_path.write_text(result, encoding="utf-8")

        if action == "cleanup":
            segments = [
                SubtitleSegment(
                    float(item["start"]),
                    float(item["end"]),
                    str(item["text"]),
                )
                for item in subtitle.get("segments", [])
                if isinstance(item, dict) and "start" in item and "end" in item and "text" in item
            ]
            if segments:
                cleaned_lines = [line.strip() for line in result.splitlines() if line.strip()]
                if cleaned_lines:
                    remapped: list[SubtitleSegment] = []
                    for index, segment in enumerate(segments):
                        text = cleaned_lines[index] if index < len(cleaned_lines) else segment.text
                        remapped.append(SubtitleSegment(segment.start, segment.end, text))
                    save_segments_as_srt(out_dir / f"{base_name}-{safe_action}-{stamp}.srt", remapped)
                    save_segments_as_txt(out_dir / f"{base_name}-{safe_action}-{stamp}-plain.txt", remapped)

    def _load_required_conversation(self, convo_id: str) -> dict[str, Any]:
        convo = self._load_conversation(convo_id)
        if not convo:
            raise RuntimeError("Conversation not found")
        if convo.get("provider_key") not in self._provider_map():
            convo["provider_key"] = self._default_provider_key()
        return convo

    @staticmethod
    def _derive_title(content: str) -> str:
        clean = " ".join(content.strip().split())
        return clean[:36] or "新工作会话"

    def _decorate_conversation(self, convo: dict[str, Any]) -> dict[str, Any]:
        subtitle = convo.get("subtitle")
        char_count = 0
        segment_count = 0
        if isinstance(subtitle, dict):
            char_count = int(subtitle.get("char_count", 0) or 0)
            segment_count = int(subtitle.get("segment_count", 0) or 0)

        provider_key = str(convo.get("provider_key") or self._default_provider_key())
        provider = self._provider_map().get(provider_key)
        decorated = dict(convo)
        decorated["provider_key"] = provider_key
        decorated["provider_label"] = provider.label if provider else provider_key
        decorated["long_context_warning"] = self._long_context_warning(char_count, segment_count, provider_key)
        return decorated

    def _long_context_warning(self, char_count: int, segment_count: int, provider_key: str) -> str:
        if char_count < LONG_CONTEXT_CHAR_THRESHOLD and segment_count < LONG_CONTEXT_SEGMENT_THRESHOLD:
            return ""
        if provider_key == "gemini_pro":
            return ""
        return (
            f"当前字幕约 {segment_count} 段、{char_count} 字符，已经接近或超过本地 Qwen3-8B 更稳妥的单次长文本处理范围。"
        )
