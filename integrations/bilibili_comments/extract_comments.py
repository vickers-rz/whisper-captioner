#!/usr/bin/env python3
"""Export Bilibili comments as Reddit-style Markdown and flat CSV."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote
from zoneinfo import ZoneInfo

import requests


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.bilibili.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Origin": "https://www.bilibili.com",
}


@dataclass
class CommentRow:
    index: int
    depth: int
    bvid: str
    aid: int
    title: str
    source_comment_count: int
    exported_root_count: int
    exported_reply_count: int
    root_rpid: str
    parent_rpid: str
    rpid: str
    author_mid: str
    author_name: str
    like: int
    reply_count: int
    ctime: int
    send_date: str
    send_datetime: str
    message: str


def extract_bvid(value: str) -> str:
    match = re.search(r"BV[0-9A-Za-z]+", value)
    if not match:
        raise ValueError(f"Could not find a BV id in {value!r}")
    return match.group(0)


def get_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    last_data: dict[str, Any] | None = None
    for attempt in range(4):
        response = requests.get(url, params=params, headers=HEADERS, timeout=20)
        response.raise_for_status()
        data = response.json()
        if data.get("code") == 0:
            return data
        last_data = data
        if data.get("code") in {-352, -412} and attempt < 3:
            time.sleep(1.0 + attempt * 1.5)
            continue
        break
    raise RuntimeError(f"Bilibili API error: {json.dumps(last_data, ensure_ascii=False)[:500]}")


def shanghai_time(timestamp: int) -> tuple[str, str]:
    dt = datetime.fromtimestamp(timestamp, ZoneInfo("Asia/Shanghai"))
    return dt.strftime("%Y-%m-%d"), dt.strftime("%Y-%m-%d %H:%M:%S")


def clean_message(message: str) -> str:
    return html.unescape(message or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def row_from_reply(
    reply: dict[str, Any],
    *,
    index: int,
    depth: int,
    bvid: str,
    aid: int,
    title: str,
    source_comment_count: int,
    root_rpid: str,
) -> CommentRow:
    ctime = int(reply.get("ctime") or 0)
    send_date, send_datetime = shanghai_time(ctime) if ctime else ("", "")
    member = reply.get("member") or {}
    content = reply.get("content") or {}
    return CommentRow(
        index=index,
        depth=depth,
        bvid=bvid,
        aid=aid,
        title=title,
        source_comment_count=source_comment_count,
        exported_root_count=0,
        exported_reply_count=0,
        root_rpid=str(root_rpid or reply.get("rpid_str") or reply.get("rpid") or ""),
        parent_rpid=str(reply.get("parent_str") or reply.get("parent") or ""),
        rpid=str(reply.get("rpid_str") or reply.get("rpid") or ""),
        author_mid=str(member.get("mid") or reply.get("mid") or ""),
        author_name=str(member.get("uname") or ""),
        like=int(reply.get("like") or 0),
        reply_count=int(reply.get("rcount") or reply.get("count") or 0),
        ctime=ctime,
        send_date=send_date,
        send_datetime=send_datetime,
        message=clean_message(str(content.get("message") or "")),
    )


def fetch_nested_replies(aid: int, root_rpid: str, page_size: int, sleep_seconds: float) -> list[dict[str, Any]]:
    replies: list[dict[str, Any]] = []
    page = 1
    while True:
        data = get_json(
            "https://api.bilibili.com/x/v2/reply/reply",
            {
                "type": 1,
                "oid": aid,
                "root": root_rpid,
                "pn": page,
                "ps": page_size,
            },
        ).get("data") or {}
        page_replies = data.get("replies") or []
        replies.extend(page_replies)
        page_info = data.get("page") or {}
        count = int(page_info.get("count") or len(replies))
        if not page_replies or len(replies) >= count:
            break
        page += 1
        if sleep_seconds:
            time.sleep(sleep_seconds)
    return replies


def fetch_comments(
    bvid: str,
    *,
    max_pages: int,
    page_size: int,
    include_nested: bool,
    sort: int,
    sleep_seconds: float,
) -> tuple[dict[str, Any], list[CommentRow]]:
    view = get_json("https://api.bilibili.com/x/web-interface/view", {"bvid": bvid}).get("data") or {}
    aid = int(view["aid"])
    title = str(view.get("title") or "")
    source_comment_count = int((view.get("stat") or {}).get("reply") or 0)
    rows: list[CommentRow] = []
    index = 1
    next_cursor = 0

    for page in range(1, max_pages + 1):
        data = get_json(
            "https://api.bilibili.com/x/v2/reply/main",
            {"type": 1, "oid": aid, "next": next_cursor, "ps": page_size, "mode": sort},
        ).get("data") or {}
        replies = data.get("replies") or []
        if not replies:
            break

        for reply in replies:
            root_rpid = str(reply.get("rpid_str") or reply.get("rpid") or "")
            rows.append(
                row_from_reply(
                    reply,
                    index=index,
                    depth=0,
                    bvid=bvid,
                    aid=aid,
                    title=title,
                    source_comment_count=source_comment_count,
                    root_rpid=root_rpid,
                )
            )
            index += 1

            inline_replies = reply.get("replies") or []
            nested_replies = inline_replies
            if include_nested and int(reply.get("rcount") or 0) > len(inline_replies):
                nested_replies = fetch_nested_replies(aid, root_rpid, page_size, sleep_seconds)

            for child in nested_replies:
                rows.append(
                    row_from_reply(
                        child,
                        index=index,
                        depth=1,
                        bvid=bvid,
                        aid=aid,
                        title=title,
                        source_comment_count=source_comment_count,
                        root_rpid=root_rpid,
                    )
                )
                index += 1

        cursor = data.get("cursor") or {}
        if cursor.get("all_count"):
            source_comment_count = int(cursor.get("all_count") or source_comment_count)
        if cursor.get("is_end"):
            break
        next_cursor = int(cursor.get("next") or 0)
        if not next_cursor:
            break
        if sleep_seconds:
            time.sleep(sleep_seconds)

    exported_root_count = sum(1 for row in rows if row.depth == 0)
    exported_reply_count = len(rows) - exported_root_count
    for row in rows:
        row.exported_root_count = exported_root_count
        row.exported_reply_count = exported_reply_count

    return view, rows


def write_csv(path: Path, rows: list[CommentRow]) -> None:
    fields = list(CommentRow.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def write_markdown(path: Path, view: dict[str, Any], rows: list[CommentRow]) -> None:
    title = str(view.get("title") or "")
    bvid = str(view.get("bvid") or "")
    lines = [
        f"# {title}",
        "",
        f"- BVID: `{bvid}`",
        f"- AID: `{view.get('aid')}`",
        f"- Source comments: `{(view.get('stat') or {}).get('reply', '')}`",
        f"- Exported comments: `{len(rows)}`",
        "",
    ]
    for row in rows:
        prefix = "" if row.depth == 0 else "  "
        bullet = "-" if row.depth == 0 else "  -"
        author = row.author_name or row.author_mid or "unknown"
        message = row.message.replace("\n", f"\n{prefix}  ")
        lines.append(f"{prefix}{bullet} **{author}** · {row.send_datetime} · {row.like} likes · rpid `{row.rpid}`")
        lines.append(f"{prefix}  {message}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bvid_or_url", help="Bilibili BV id or video URL")
    parser.add_argument("--out-dir", default=".", help="Output directory")
    parser.add_argument("--max-pages", type=int, default=5, help="Maximum top-level comment pages to fetch")
    parser.add_argument("--page-size", type=int, default=20, help="Comments per page")
    parser.add_argument("--sort", type=int, default=2, help="Bilibili comment mode, usually 2 for latest and 3 for hot")
    parser.add_argument("--include-nested", action="store_true", help="Fetch full nested replies for each root comment")
    parser.add_argument("--all", action="store_true", help="Fetch all available top-level comment pages")
    parser.add_argument("--sleep", type=float, default=0.25, help="Delay between paged requests")
    args = parser.parse_args()

    bvid = extract_bvid(args.bvid_or_url)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    view, rows = fetch_comments(
        bvid,
        max_pages=1_000_000 if args.all else args.max_pages,
        page_size=args.page_size,
        include_nested=args.include_nested,
        sort=args.sort,
        sleep_seconds=args.sleep,
    )

    safe_bvid = quote(bvid, safe="")
    csv_path = out_dir / f"bilibili_comments_{safe_bvid}.csv"
    md_path = out_dir / f"bilibili_comments_{safe_bvid}.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, view, rows)

    print(
        json.dumps(
            {
                "bvid": bvid,
                "aid": view.get("aid"),
                "title": view.get("title"),
                "source_comment_count": (view.get("stat") or {}).get("reply"),
                "count": len(rows),
                "csv": str(csv_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
