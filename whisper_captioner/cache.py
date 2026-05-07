from __future__ import annotations

import hashlib
import json
import re
from urllib.parse import parse_qs, urlparse


def _canonicalize_bilibili_url(parsed) -> str | None:
    host = parsed.netloc.lower()
    if "bilibili.com" not in host:
        return None
    path = parsed.path.rstrip("/")
    match = re.search(r"/video/([^/?#]+)", path)
    if not match:
        return None
    canonical = f"https://www.bilibili.com/video/{match.group(1)}"
    page = parse_qs(parsed.query).get("p", [""])[0]
    if page:
        canonical += f"?p={page}"
    return canonical


def _canonicalize_youtube_url(parsed) -> str | None:
    host = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    if host in {"youtu.be", "www.youtu.be"}:
        video_id = path.strip("/").split("/")[0]
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"
        return None
    if "youtube.com" not in host:
        return None
    query = parse_qs(parsed.query)
    video_id = query.get("v", [""])[0]
    if not video_id:
        shorts_match = re.search(r"^/shorts/([^/?#]+)", path)
        live_match = re.search(r"^/live/([^/?#]+)", path)
        if shorts_match:
            video_id = shorts_match.group(1)
        elif live_match:
            video_id = live_match.group(1)
    if video_id:
        return f"https://www.youtube.com/watch?v={video_id}"
    return None


def canonical_media_url(url: str) -> str:
    parsed = urlparse(url.strip())
    bilibili = _canonicalize_bilibili_url(parsed)
    if bilibili:
        return bilibili
    youtube = _canonicalize_youtube_url(parsed)
    if youtube:
        return youtube
    return url.strip()


def cache_slug(*parts: object) -> str:
    raw = json.dumps(parts, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def validate_url_for_yt_dlp(url: str) -> tuple[bool, str]:
    """
    Check if a URL is likely downloadable by yt-dlp (video/audio content).
    Returns (is_valid, error_message).
    """
    url = url.strip()
    
    # Check basic format
    if not url.startswith(("http://", "https://")):
        return False, "URL must start with http:// or https://"
    
    # Unsupported domains that clearly aren't video content
    unsupported_domains = {
        "docs.github.com",
        "github.com",
        "wikipedia.org",
        "google.com",
        "stackoverflow.com",
        "reddit.com",
        "twitter.com",
        "x.com",
    }
    
    for domain in unsupported_domains:
        if domain in url.lower():
            return False, f"'{domain}' is not supported. Please provide a video/audio URL (e.g., YouTube, Bilibili, Vimeo)."
    
    # Basic heuristic: video URLs typically contain video-related keywords or come from known platforms
    known_video_domains = {
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
    }
    
    url_lower = url.lower()
    has_known_domain = any(domain in url_lower for domain in known_video_domains)
    
    if not has_known_domain:
        # Still allow it, but warn the user if it looks suspicious
        if any(keyword in url_lower for keyword in [".mp3", ".mp4", ".wav", ".flac", "/watch", "/video", "/play"]):
            return True, ""
        return True, ""  # Optimistic: yt-dlp might support it
    
    return True, ""
