from __future__ import annotations

import json
import threading
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

from whisper_captioner.config import (
    QWEN_CHAT_DIR,
    RAPIDMLX_8B_MODEL,
    RAPIDMLX_8B_PORT,
    RAPIDMLX_8B_SERVED_MODEL,
    RAPIDMLX_HOST,
)
from whisper_captioner.llm_handler import ensure_local_rapidmlx_server


CHAT_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Local Rapid-MLX Qwen3-8B Chat</title>
  <style>
    :root {
      --bg: #f4f1ea;
      --panel: #fbfaf7;
      --sidebar: #efe7da;
      --border: #d8ccb9;
      --text: #2b241d;
      --muted: #7b6d5c;
      --accent: #a34a28;
      --accent-2: #d8894f;
      --user: #fff3e6;
      --assistant: #ffffff;
      --shadow: 0 10px 30px rgba(73, 49, 24, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "SF Pro Text", "PingFang SC", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(216, 171, 112, 0.24), transparent 28%),
        linear-gradient(180deg, #f7f1e8 0%, var(--bg) 100%);
      min-height: 100vh;
    }
    .app {
      display: grid;
      grid-template-columns: 290px 1fr;
      min-height: 100vh;
    }
    .sidebar {
      border-right: 1px solid var(--border);
      background: linear-gradient(180deg, rgba(255,255,255,0.52), rgba(239,231,218,0.94));
      padding: 18px 14px;
      backdrop-filter: blur(12px);
    }
    .sidebar-header {
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin-bottom: 14px;
    }
    .brand {
      font-size: 18px;
      font-weight: 700;
      letter-spacing: 0.02em;
    }
    .sub {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }
    .sidebar button,
    .composer button {
      border: 0;
      border-radius: 14px;
      padding: 12px 14px;
      font-size: 14px;
      cursor: pointer;
      transition: transform 120ms ease, opacity 120ms ease;
    }
    .sidebar button:hover,
    .composer button:hover {
      transform: translateY(-1px);
    }
    .primary {
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      color: white;
      box-shadow: var(--shadow);
    }
    .secondary {
      background: rgba(255,255,255,0.85);
      color: var(--text);
      border: 1px solid var(--border);
    }
    .conversation-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin-top: 16px;
      max-height: calc(100vh - 180px);
      overflow: auto;
      padding-right: 4px;
    }
    .conversation-item {
      border: 1px solid transparent;
      border-radius: 14px;
      padding: 12px;
      background: rgba(255,255,255,0.54);
      cursor: pointer;
    }
    .conversation-item.active {
      border-color: rgba(163, 74, 40, 0.28);
      background: rgba(255,255,255,0.95);
      box-shadow: var(--shadow);
    }
    .conversation-title {
      font-size: 14px;
      font-weight: 600;
      margin-bottom: 4px;
    }
    .conversation-meta {
      font-size: 12px;
      color: var(--muted);
    }
    .main {
      display: grid;
      grid-template-rows: auto 1fr auto;
      min-height: 100vh;
    }
    .topbar {
      border-bottom: 1px solid var(--border);
      padding: 18px 24px;
      background: rgba(251,250,247,0.86);
      backdrop-filter: blur(12px);
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
    }
    .topbar h1 {
      margin: 0;
      font-size: 20px;
    }
    .status {
      color: var(--muted);
      font-size: 13px;
    }
    .messages {
      padding: 24px;
      overflow: auto;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .message {
      max-width: min(840px, 92%);
      padding: 16px 18px;
      border-radius: 20px;
      line-height: 1.65;
      white-space: pre-wrap;
      box-shadow: var(--shadow);
      border: 1px solid rgba(121, 94, 61, 0.08);
    }
    .message.user {
      align-self: flex-end;
      background: var(--user);
    }
    .message.assistant {
      align-self: flex-start;
      background: var(--assistant);
    }
    .empty {
      margin: auto;
      max-width: 720px;
      text-align: center;
      color: var(--muted);
      line-height: 1.7;
    }
    .composer {
      padding: 18px 24px 24px;
      border-top: 1px solid var(--border);
      background: rgba(251,250,247,0.9);
      backdrop-filter: blur(10px);
    }
    .composer-shell {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      align-items: end;
      max-width: 1080px;
      margin: 0 auto;
    }
    textarea {
      width: 100%;
      min-height: 94px;
      max-height: 240px;
      resize: vertical;
      border-radius: 18px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.95);
      padding: 16px 18px;
      font: inherit;
      color: var(--text);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.6);
    }
    .composer-actions {
      display: flex;
      flex-direction: column;
      gap: 10px;
      min-width: 160px;
    }
    .hint {
      font-size: 12px;
      color: var(--muted);
      text-align: right;
    }
    @media (max-width: 900px) {
      .app { grid-template-columns: 1fr; }
      .sidebar { border-right: 0; border-bottom: 1px solid var(--border); }
      .conversation-list { max-height: 220px; }
      .composer-shell { grid-template-columns: 1fr; }
      .composer-actions { min-width: 0; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="sidebar-header">
        <div class="brand">Qwen3-8B Local Chat</div>
        <div class="sub">直接对话本机 Rapid-MLX Qwen3-8B。与 Whisper Captioner 转录链路无关。</div>
        <button id="new-chat" class="primary">新建对话</button>
      </div>
      <div id="conversation-list" class="conversation-list"></div>
    </aside>
    <main class="main">
      <div class="topbar">
        <div>
          <h1 id="conversation-title">新对话</h1>
          <div class="status" id="conversation-status">就绪</div>
        </div>
        <button id="refresh" class="secondary">刷新列表</button>
      </div>
      <div id="messages" class="messages">
        <div class="empty">左侧可以切换历史会话。新建后像 ChatGPT 一样直接输入，服务会自动唤起本机 `Rapid-MLX Qwen3-8B`。</div>
      </div>
      <div class="composer">
        <div class="composer-shell">
          <textarea id="prompt" placeholder="输入你的问题，按 Cmd/Ctrl + Enter 发送。"></textarea>
          <div class="composer-actions">
            <button id="send" class="primary">发送</button>
            <div class="hint">模型：Local Rapid-MLX Qwen3-8B</div>
          </div>
        </div>
      </div>
    </main>
  </div>
  <script>
    const state = {
      currentId: null,
      conversations: [],
      sending: false,
    };

    const els = {
      list: document.getElementById("conversation-list"),
      title: document.getElementById("conversation-title"),
      status: document.getElementById("conversation-status"),
      messages: document.getElementById("messages"),
      prompt: document.getElementById("prompt"),
      send: document.getElementById("send"),
      newChat: document.getElementById("new-chat"),
      refresh: document.getElementById("refresh"),
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

    function renderConversationList() {
      els.list.innerHTML = "";
      for (const convo of state.conversations) {
        const item = document.createElement("div");
        item.className = "conversation-item" + (convo.id === state.currentId ? " active" : "");
        item.innerHTML = `
          <div class="conversation-title">${convo.title || "新对话"}</div>
          <div class="conversation-meta">${convo.message_count || 0} 条消息 · ${fmtTime(convo.updated_at)}</div>
        `;
        item.addEventListener("click", () => openConversation(convo.id));
        els.list.appendChild(item);
      }
    }

    function renderMessages(convo) {
      els.messages.innerHTML = "";
      els.title.textContent = convo.title || "新对话";
      if (!convo.messages.length) {
        els.messages.innerHTML = `<div class="empty">这个会话还没有消息。直接在下方输入开始测试 Qwen3-8B 的能力边界。</div>`;
        return;
      }
      for (const msg of convo.messages) {
        const div = document.createElement("div");
        div.className = "message " + msg.role;
        div.textContent = msg.content;
        els.messages.appendChild(div);
      }
      els.messages.scrollTop = els.messages.scrollHeight;
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
      setStatus("已创建新对话");
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

    async function sendMessage() {
      const content = els.prompt.value.trim();
      if (!content || state.sending) return;
      if (!state.currentId) {
        await createConversation();
      }
      state.sending = true;
      els.send.disabled = true;
      setStatus("Qwen3-8B 正在思考...");
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

    els.send.addEventListener("click", sendMessage);
    els.newChat.addEventListener("click", createConversation);
    els.refresh.addEventListener("click", () => refreshConversations());
    els.prompt.addEventListener("keydown", (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        event.preventDefault();
        sendMessage();
      }
    });

    refreshConversations().catch((err) => setStatus(`初始化失败：${err.message}`));
  </script>
</body>
</html>
"""


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

    def is_running(self) -> bool:
        return self._server is not None and self._thread is not None and self._thread.is_alive()

    def start(self) -> str:
        with self._lock:
            if self.is_running():
                return self.base_url
            self.storage_dir.mkdir(parents=True, exist_ok=True)
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
                if parsed.path == "/api/conversations":
                    self._write_json({"conversations": manager._list_conversations()})
                    return
                if parsed.path.startswith("/api/conversations/"):
                    convo_id = parsed.path.removeprefix("/api/conversations/")
                    convo = manager._load_conversation(convo_id)
                    if not convo:
                        self._write_json({"error": "Conversation not found"}, status=HTTPStatus.NOT_FOUND)
                        return
                    self._write_json(convo)
                    return
                self._write_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:
                parsed = urllib.parse.urlparse(self.path)
                payload = self._read_json_body()
                if parsed.path == "/api/conversations":
                    convo = manager._create_conversation(payload.get("title") if isinstance(payload, dict) else None)
                    self._write_json(convo, status=HTTPStatus.CREATED)
                    return
                if parsed.path.startswith("/api/conversations/") and parsed.path.endswith("/messages"):
                    convo_id = parsed.path.removeprefix("/api/conversations/").removesuffix("/messages")
                    content = ""
                    if isinstance(payload, dict):
                        content = str(payload.get("content", "")).strip()
                    if not content:
                        self._write_json({"error": "Message content is required"}, status=HTTPStatus.BAD_REQUEST)
                        return
                    try:
                        convo = manager._append_and_reply(convo_id, content)
                    except Exception as exc:
                        self._write_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                        return
                    self._write_json(convo)
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

    def _list_conversations(self) -> list[dict[str, Any]]:
        conversations: list[dict[str, Any]] = []
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(self.storage_dir.glob("*.json"), reverse=True):
            convo = self._read_conversation_file(path)
            if not convo:
                continue
            conversations.append(
                {
                    "id": convo["id"],
                    "title": convo.get("title", "新对话"),
                    "updated_at": convo.get("updated_at"),
                    "message_count": len(convo.get("messages", [])),
                }
            )
        conversations.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        return conversations

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

    def _create_conversation(self, title: Optional[str] = None) -> dict[str, Any]:
        now = self._now()
        convo = {
            "id": uuid.uuid4().hex,
            "title": title or "新对话",
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }
        self._save_conversation(convo)
        return convo

    def _append_and_reply(self, convo_id: str, content: str) -> dict[str, Any]:
        convo = self._load_conversation(convo_id)
        if not convo:
            raise RuntimeError("Conversation not found")
        user_message = {
            "id": uuid.uuid4().hex,
            "role": "user",
            "content": content,
            "created_at": self._now(),
        }
        convo.setdefault("messages", []).append(user_message)
        if convo.get("title", "新对话") == "新对话":
            convo["title"] = self._derive_title(content)
        convo["updated_at"] = self._now()
        assistant_text = self._chat_with_qwen(convo["messages"])
        assistant_message = {
            "id": uuid.uuid4().hex,
            "role": "assistant",
            "content": assistant_text,
            "created_at": self._now(),
        }
        convo["messages"].append(assistant_message)
        convo["updated_at"] = self._now()
        self._save_conversation(convo)
        return convo

    @staticmethod
    def _derive_title(content: str) -> str:
        clean = " ".join(content.strip().split())
        return clean[:36] or "新对话"

    def _chat_with_qwen(self, messages: list[dict[str, Any]]) -> str:
        ensure_local_rapidmlx_server(
            model=RAPIDMLX_8B_MODEL,
            served_model=RAPIDMLX_8B_SERVED_MODEL,
            port=RAPIDMLX_8B_PORT,
        )
        payload = {
            "model": RAPIDMLX_8B_SERVED_MODEL,
            "messages": [
                {"role": msg["role"], "content": msg["content"]}
                for msg in messages
                if msg.get("role") in {"user", "assistant", "system"}
            ],
            "temperature": 0.7,
            "max_tokens": 4096,
            "no_thinking": True,
        }
        req = urllib.request.Request(
            f"http://{RAPIDMLX_HOST}:{RAPIDMLX_8B_PORT}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Qwen3-8B request failed: {exc}") from exc
        try:
            data = json.loads(raw)
            return str(data["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Qwen3-8B returned an unexpected response: {raw[:400]}") from exc
