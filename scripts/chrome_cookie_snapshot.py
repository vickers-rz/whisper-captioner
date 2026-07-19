#!/usr/bin/env python3
"""Create an isolated Chrome cookie profile for yt-dlp.

Recent Chrome profiles can contain nested extension databases named
``Cookies``. yt-dlp selects the newest matching file recursively, which may be
an extension database instead of the profile's primary cookie store. This
module snapshots only the primary database into a temporary profile.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


DEFAULT_CHROME_ROOT = Path(
    os.environ.get(
        "CHROME_USER_DATA_DIR",
        "~/Library/Application Support/Google/Chrome",
    )
).expanduser()


@dataclass(frozen=True)
class YtDlpCookieSession:
    browser_spec: str | None = None
    source_database: Path | None = None
    profile: str | None = None
    total_cookies: int = 0
    youtube_cookies: int = 0

    @property
    def arguments(self) -> list[str]:
        if not self.browser_spec:
            return []
        return ["--cookies-from-browser", self.browser_spec]


def chrome_profile_path(profile: str) -> Path:
    expanded = Path(profile).expanduser()
    if expanded.is_absolute() or len(expanded.parts) > 1 or profile.startswith("."):
        return expanded.resolve()
    return (DEFAULT_CHROME_ROOT / profile).resolve()


def primary_cookie_database(profile_dir: Path) -> Path:
    # Chromium has used both locations. Deliberately do not search recursively:
    # nested extension databases are the source of the yt-dlp mis-selection.
    candidates = (profile_dir / "Cookies", profile_dir / "Network" / "Cookies")
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        raise RuntimeError(
            f"Chrome primary cookie database not found under {profile_dir}. "
            "Check the Chrome profile name and sign in to YouTube first."
        )
    return max(existing, key=lambda path: path.stat().st_mtime)


def snapshot_database(source: Path, target: Path) -> tuple[int, int]:
    target.parent.mkdir(parents=True, exist_ok=True)
    source_connection = sqlite3.connect(
        f"{source.resolve().as_uri()}?mode=ro",
        uri=True,
        timeout=10,
    )
    target_connection = sqlite3.connect(target)
    try:
        source_connection.backup(target_connection)
        total = int(target_connection.execute("SELECT COUNT(*) FROM cookies").fetchone()[0])
        youtube = int(
            target_connection.execute(
                "SELECT COUNT(*) FROM cookies WHERE host_key LIKE ?",
                ("%youtube.com",),
            ).fetchone()[0]
        )
    finally:
        target_connection.close()
        source_connection.close()
    target.chmod(0o600)
    return total, youtube


@contextmanager
def yt_dlp_cookie_session(
    *,
    enabled: bool,
    chrome_profile: str = "Default",
    forwarded_browser_spec: str | None = None,
) -> Iterator[YtDlpCookieSession]:
    if forwarded_browser_spec:
        yield YtDlpCookieSession(browser_spec=forwarded_browser_spec)
        return
    if not enabled:
        yield YtDlpCookieSession()
        return

    profile_dir = chrome_profile_path(chrome_profile)
    source = primary_cookie_database(profile_dir)
    with tempfile.TemporaryDirectory(prefix="whisper-captioner-chrome-") as temporary:
        isolated_profile = Path(temporary) / "Profile"
        target = isolated_profile / "Cookies"
        total, youtube = snapshot_database(source, target)
        if youtube == 0:
            raise RuntimeError(
                f"Chrome profile {chrome_profile!r} has no YouTube cookies. "
                "Open YouTube in that profile and sign in, then retry."
            )
        yield YtDlpCookieSession(
            browser_spec=f"chrome:{isolated_profile}",
            source_database=source,
            profile=chrome_profile,
            total_cookies=total,
            youtube_cookies=youtube,
        )
