"""
Chrome 浏览器控制模块

通过 AppleScript (osascript) 和 JavaScript 与 macOS 上的 Google Chrome 浏览器进行交互。
主要职责包括：
1. 获取当前活动标签页的 URL。
2. 在指定的标签页中注入并执行 JavaScript 代码，以控制网页内的 `<video>` 元素。
3. 提供播放、暂停、跳转、获取当前时间等针对视频播放的控制接口，用于“网址受控字幕”模式。
"""
from __future__ import annotations

from dataclasses import dataclass
import subprocess
from typing import Optional
from urllib.parse import urlparse


VIDEO_PICKER_JS = (
    "const videos = Array.from(document.querySelectorAll('video')).filter(v => {"
    "const r = v.getBoundingClientRect();"
    "const s = getComputedStyle(v);"
    "return r.width > 50 && r.height > 50 && s.visibility !== 'hidden' && s.display !== 'none';"
    "});"
    "const scored = videos.map((v, i) => ({v, score:"
    "(!v.muted ? 10000 : 0) + (!v.paused ? 5000 : 0) + "
    "(Number.isFinite(v.duration) ? v.duration : 0) + "
    "((v.clientWidth || 0) * (v.clientHeight || 0) / 100000)}));"
    "const picked = scored.sort((a,b) => b.score - a.score)[0];"
    "const v = picked && picked.v;"
)

MEDIA_HOSTS = (
    "youtube.com",
    "youtu.be",
    "bilibili.com",
    "b23.tv",
    "vimeo.com",
    "twitch.tv",
    "dailymotion.com",
    "instagram.com",
    "tiktok.com",
    "rumble.com",
    "odysee.com",
)


@dataclass(frozen=True)
class ChromeMediaTab:
    title: str
    url: str
    window_index: int = -1
    tab_index: int = -1
    is_front_window: bool = False
    is_active_tab: bool = False


def chrome_is_running() -> bool:
    proc = subprocess.run(
        ["/usr/bin/pgrep", "-x", "Google Chrome"],
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.returncode == 0


def _is_likely_media_url(url: str) -> bool:
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"}:
        return False
    host = parsed.netloc.lower()
    if any(host == domain or host.endswith(f".{domain}") for domain in MEDIA_HOSTS):
        return True
    path = parsed.path.lower()
    return (
        path.endswith((".mp3", ".mp4", ".m4a", ".wav", ".flac", ".webm"))
        or "/watch" in path
        or "/video/" in path
        or "/videos/" in path
        or "/play/" in path
    )


def _chrome_active_tabs() -> list[ChromeMediaTab]:
    if not chrome_is_running():
        return []
    apple_script = (
        'tell application "Google Chrome"\n'
        '  set frontWindowIndex to -1\n'
        '  try\n'
        '    set frontWindowIndex to index of front window\n'
        '  end try\n'
        '  set activeTabs to ""\n'
        '  repeat with w in windows\n'
        '    set windowIndex to index of w\n'
        '    set activeTabIndex to active tab index of w\n'
        '    repeat with tabIndex from 1 to (count of tabs of w)\n'
        '      set t to tab tabIndex of w\n'
        '      if activeTabs is not "" then set activeTabs to activeTabs & linefeed\n'
        '      set activeTabs to activeTabs & (title of t) & (ASCII character 31) & (URL of t) & (ASCII character 31) & (windowIndex as string) & (ASCII character 31) & (tabIndex as string) & (ASCII character 31) & ((windowIndex is frontWindowIndex) as string) & (ASCII character 31) & ((tabIndex is activeTabIndex) as string)\n'
        '    end repeat\n'
        '  end repeat\n'
        '  return activeTabs\n'
        'end tell\n'
    )
    proc = subprocess.run(
        ["/usr/bin/osascript", "-e", apple_script],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    tabs: list[ChromeMediaTab] = []
    for line in proc.stdout.splitlines():
        parts = line.split("\x1f")
        if len(parts) < 2:
            continue
        title = parts[0].strip()
        url = parts[1].strip()
        if not url:
            continue
        window_index = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else -1
        tab_index = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else -1
        is_front_window = len(parts) > 4 and parts[4].strip().lower() == "true"
        is_active_tab = len(parts) > 5 and parts[5].strip().lower() == "true"
        tabs.append(
            ChromeMediaTab(
                title=title,
                url=url,
                window_index=window_index,
                tab_index=tab_index,
                is_front_window=is_front_window,
                is_active_tab=is_active_tab,
            )
        )
    return tabs


def chrome_media_tabs() -> list[ChromeMediaTab]:
    tabs = [tab for tab in _chrome_active_tabs() if _is_likely_media_url(tab.url)]
    tabs.sort(
        key=lambda tab: (
            0 if tab.is_front_window and tab.is_active_tab else 1,
            0 if tab.is_front_window else 1,
            tab.window_index if tab.window_index >= 0 else 10**6,
            tab.tab_index if tab.tab_index >= 0 else 10**6,
        )
    )
    return tabs


def run_chrome_script(script: str) -> str:
    if not chrome_is_running():
        return ""
    escaped = script.replace("\\", "\\\\").replace('"', '\\"')
    apple_script = (
        'tell application "Google Chrome"\n'
        '  if not (exists window 1) then return ""\n'
        f'  execute active tab of front window javascript "{escaped}"\n'
        "end tell\n"
    )
    proc = subprocess.run(
        ["/usr/bin/osascript", "-e", apple_script],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return proc.stdout.strip()


def run_chrome_script_for_url(target_url: str, script: str, activate_tab: bool = True) -> str:
    if not chrome_is_running():
        return ""
    safe_url = target_url.replace("\\", "\\\\").replace('"', '\\"')
    escaped = script.replace("\\", "\\\\").replace('"', '\\"')
    activation = ""
    if activate_tab:
        activation = (
            '        set active tab index of w to tabIndex\n'
            '        set index of w to 1\n'
            '        activate\n'
        )
    apple_script = (
        'tell application "Google Chrome"\n'
        '  repeat with w in windows\n'
        '    repeat with tabIndex from 1 to (count of tabs of w)\n'
        '      set t to tab tabIndex of w\n'
        f'      if URL of t starts with "{safe_url}" then\n'
        f'{activation}'
        f'        return execute t javascript "{escaped}"\n'
        '      end if\n'
        '    end repeat\n'
        '  end repeat\n'
        '  return ""\n'
        'end tell\n'
    )
    proc = subprocess.run(
        ["/usr/bin/osascript", "-e", apple_script],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return proc.stdout.strip()


def chrome_pause() -> None:
    run_chrome_script(
        f"(() => {{ {VIDEO_PICKER_JS} "
        "if (!v) return 'no-video'; v.pause(); return String(v.currentTime || 0); })()"
    )


def chrome_pause_url(target_url: str, activate_tab: bool = True) -> Optional[float]:
    value = run_chrome_script_for_url(
        target_url,
        f"(() => {{ {VIDEO_PICKER_JS} "
        "if (!v) return ''; v.pause(); return String(v.currentTime || 0); })()",
        activate_tab=activate_tab,
    )
    try:
        return float(value)
    except ValueError:
        return None


def chrome_play_from(seconds: float = 0.0) -> None:
    run_chrome_script(
        f"(() => {{ {VIDEO_PICKER_JS} "
        "if (!v) return 'no-video'; "
        f"v.currentTime = Math.max(0, {seconds}); "
        "v.play(); return 'playing'; })()"
    )


def chrome_play_url_from(target_url: str, seconds: float) -> bool:
    value = run_chrome_script_for_url(
        target_url,
        f"(() => {{ {VIDEO_PICKER_JS} "
        "if (!v) return ''; "
        f"v.currentTime = Math.max(0, {seconds}); "
        "v.play(); return 'playing'; })()",
    )
    return value == "playing"


def chrome_resume() -> None:
    run_chrome_script(
        f"(() => {{ {VIDEO_PICKER_JS} "
        "if (!v) return 'no-video'; v.play(); return 'resumed'; })()"
    )


def chrome_resume_url(target_url: str, activate_tab: bool = True) -> bool:
    value = run_chrome_script_for_url(
        target_url,
        f"(() => {{ {VIDEO_PICKER_JS} "
        "if (!v) return ''; v.play(); return 'resumed'; })()",
        activate_tab=activate_tab,
    )
    return value == "resumed"


def chrome_toggle_playback() -> Optional[bool]:
    value = run_chrome_script(
        f"(() => {{ {VIDEO_PICKER_JS} "
        "if (!v) return ''; "
        "if (v.paused) { v.play(); return 'playing'; } "
        "v.pause(); return 'paused'; })()"
    )
    if value == "playing":
        return True
    if value == "paused":
        return False
    return None


def chrome_current_time() -> Optional[float]:
    value = run_chrome_script(
        f"(() => {{ {VIDEO_PICKER_JS} "
        "if (!v) return ''; return String(v.currentTime || 0); })()"
    )
    try:
        return float(value)
    except ValueError:
        return None


def chrome_current_time_url(target_url: str) -> Optional[float]:
    value = run_chrome_script_for_url(
        target_url,
        f"(() => {{ {VIDEO_PICKER_JS} "
        "if (!v) return ''; return String(v.currentTime || 0); })()",
        activate_tab=False,
    )
    try:
        return float(value)
    except ValueError:
        return None


def chrome_seek_relative(delta_seconds: float) -> Optional[float]:
    value = run_chrome_script(
        f"(() => {{ {VIDEO_PICKER_JS} "
        "if (!v) return ''; "
        f"const next = Math.max(0, Math.min(Number.isFinite(v.duration) ? v.duration : Infinity, (v.currentTime || 0) + ({delta_seconds}))); "
        "v.currentTime = next; "
        "return String(v.currentTime || 0); })()"
    )
    try:
        return float(value)
    except ValueError:
        return None


def chrome_seek_url_relative(target_url: str, delta_seconds: float) -> Optional[float]:
    value = run_chrome_script_for_url(
        target_url,
        f"(() => {{ {VIDEO_PICKER_JS} "
        "if (!v) return ''; "
        f"const next = Math.max(0, Math.min(Number.isFinite(v.duration) ? v.duration : Infinity, (v.currentTime || 0) + ({delta_seconds}))); "
        "v.currentTime = next; "
        "return String(v.currentTime || 0); })()",
    )
    try:
        return float(value)
    except ValueError:
        return None


def chrome_get_url() -> str:
    tabs = chrome_media_tabs()
    return tabs[0].url if tabs else ""
