"""
ASR 历史记录管理模块

负责管理本地音频缓存和 ASR（自动语音识别）任务的历史记录。
主要职责包括：
1. 维护 JSON 格式的持久化历史列表。
2. 处理同一源文件或 URL 的缓存命中逻辑，避免重复下载和识别。
3. 提供缓存路径重定位机制，适应外部硬盘等路径可能变化的场景。
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from whisper_captioner.cache import cache_slug, canonical_media_url
from whisper_captioner.config import (
    ASR_HISTORY_PATH,
    LOCAL_AUDIO_CACHE_DIR,
    OUTPUT_DIR,
)


VALID_STATUSES = {"running", "ready", "failed", "audio_cache_pruned"}
OLD_OUTPUT_ROOTS = (
    Path("/Users/vickers/Movies/WhisperCaptioner"),
    Path("/Volumes/T7/MacBackup/Movies/WhisperCaptioner"),
    Path("/Volumes/T7/MacBackup/Movies"),
)


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def canonical_source(source: str) -> tuple[str, str]:
    value = source.strip()
    if value.startswith(("http://", "https://")):
        canonical = canonical_media_url(value)
        return "url", canonical
    return "file", str(Path(value).expanduser().resolve())


def history_id(source: str) -> str:
    kind, canonical = canonical_source(source)
    return f"{kind}:{canonical}"


@dataclass
class ASRHistoryEntry:
    id: str
    source: str
    canonical_source: str
    title: str
    kind: str
    audio_cache_key: str = ""
    audio_cache_wav: str = ""
    audio_cache_exists: bool = False
    last_mode_key: str = ""
    last_mode_label: str = ""
    output_base: str = ""
    subtitle_cache_dir: str = ""
    status: str = "running"
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ASRHistoryEntry":
        allowed = {field.name for field in fields(cls)}
        values = {key: value for key, value in data.items() if key in allowed}
        source = str(values.get("source") or values.get("canonical_source") or "")
        kind, canonical = canonical_source(source)
        values.setdefault("id", f"{kind}:{canonical}")
        values.setdefault("canonical_source", canonical)
        values.setdefault("kind", kind)
        values.setdefault("title", Path(canonical).stem if kind == "file" else canonical)
        values["status"] = (
            str(values.get("status")) if values.get("status") in VALID_STATUSES else "failed"
        )
        values.setdefault("created_at", utc_timestamp())
        values.setdefault("updated_at", values["created_at"])
        return cls(**values)


class ASRHistoryStore:
    def __init__(
        self,
        path: Path = ASR_HISTORY_PATH,
        *,
        output_dir: Path = OUTPUT_DIR,
        local_audio_cache_dir: Path = LOCAL_AUDIO_CACHE_DIR,
    ) -> None:
        self.path = Path(path)
        self.output_dir = Path(output_dir)
        self.local_audio_cache_dir = Path(local_audio_cache_dir)
        self._lock = threading.RLock()

    def list_entries(self, *, refresh_paths: bool = True) -> list[ASRHistoryEntry]:
        with self._lock:
            entries = self._load()
            changed = False
            if refresh_paths:
                for entry in entries:
                    changed |= self._refresh_paths(entry)
            if changed:
                self._write(entries)
            return sorted(entries, key=lambda item: item.updated_at, reverse=True)

    def get(self, entry_id: str) -> ASRHistoryEntry | None:
        return next((entry for entry in self.list_entries() if entry.id == entry_id), None)

    def upsert(self, source: str, **updates: Any) -> ASRHistoryEntry:
        with self._lock:
            entries = self._load()
            entry_id = history_id(source)
            entry = next((item for item in entries if item.id == entry_id), None)
            now = utc_timestamp()
            kind, canonical = canonical_source(source)
            if entry is None:
                entry = ASRHistoryEntry(
                    id=entry_id,
                    source=source,
                    canonical_source=canonical,
                    title=Path(canonical).stem if kind == "file" else canonical,
                    kind=kind,
                    created_at=now,
                    updated_at=now,
                )
                entries.append(entry)
            for key, value in updates.items():
                if hasattr(entry, key):
                    setattr(entry, key, value)
            entry.source = source
            entry.canonical_source = canonical
            entry.kind = kind
            entry.updated_at = now
            if entry.status not in VALID_STATUSES:
                entry.status = "failed"
            self._refresh_paths(entry)
            self._write(entries)
            return entry

    def delete(self, entry_id: str) -> bool:
        with self._lock:
            entries = self._load()
            remaining = [entry for entry in entries if entry.id != entry_id]
            if len(remaining) == len(entries):
                return False
            self._write(remaining)
            return True

    def _load(self) -> list[ASRHistoryEntry]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            raw_entries = data if isinstance(data, list) else data.get("entries", [])
            return [
                ASRHistoryEntry.from_dict(item)
                for item in raw_entries
                if isinstance(item, dict)
            ]
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            stamp = time.strftime("%Y%m%d-%H%M%S")
            backup = self.path.with_name(f"{self.path.name}.corrupt-{stamp}")
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(self.path, backup)
            self._write([])
            return []

    def _write(self, entries: list[ASRHistoryEntry]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        payload = [asdict(entry) for entry in entries]
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def _rebase_path(self, value: str) -> str:
        if not value:
            return ""
        path = Path(value)
        for old_root in OLD_OUTPUT_ROOTS:
            try:
                relative = path.relative_to(old_root)
            except ValueError:
                continue
            return str(self.output_dir / relative)
        return str(path)

    def _refresh_paths(self, entry: ASRHistoryEntry) -> bool:
        changed = False
        for field_name in ("audio_cache_wav", "output_base", "subtitle_cache_dir"):
            old_value = getattr(entry, field_name)
            new_value = self._rebase_path(old_value)
            if new_value != old_value:
                setattr(entry, field_name, new_value)
                changed = True

        wav = Path(entry.audio_cache_wav) if entry.audio_cache_wav else None
        if not wav or not wav.exists():
            relocated = self._find_audio_cache(entry)
            if relocated and str(relocated) != entry.audio_cache_wav:
                entry.audio_cache_wav = str(relocated)
                entry.audio_cache_key = relocated.parent.name
                wav = relocated
                changed = True
        exists = bool(wav and wav.exists())
        if exists != entry.audio_cache_exists:
            entry.audio_cache_exists = exists
            changed = True
        if not exists and entry.status == "ready":
            entry.status = "audio_cache_pruned"
            changed = True
        return changed

    def _find_audio_cache(self, entry: ASRHistoryEntry) -> Path | None:
        if entry.audio_cache_key:
            candidate = self.local_audio_cache_dir / entry.audio_cache_key / "audio-16k-mono.wav"
            if candidate.exists():
                return candidate
        if not self.local_audio_cache_dir.exists():
            return None
        for metadata_path in self.local_audio_cache_dir.glob("*/metadata.json"):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            source = str(metadata.get("source") or "")
            identity = str(metadata.get("identity") or "")
            candidates = {source, identity}
            if entry.source not in candidates and entry.canonical_source not in candidates:
                continue
            raw_wav = str(metadata.get("wav") or "")
            wav = Path(self._rebase_path(raw_wav)) if raw_wav else metadata_path.parent / "audio-16k-mono.wav"
            if wav.exists():
                return wav
        return None


def audio_cache_key_for_url(source: str) -> str:
    return cache_slug(canonical_media_url(source))
