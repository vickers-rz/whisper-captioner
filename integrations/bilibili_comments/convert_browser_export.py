#!/usr/bin/env python3
"""Convert browser-context Bilibili comment export JSON to CSV and Markdown."""

from __future__ import annotations

import argparse
import csv
import html
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


FIELDS = [
    "index",
    "depth",
    "bvid",
    "aid",
    "title",
    "source_comment_count",
    "exported_root_count",
    "exported_reply_count",
    "root_rpid",
    "parent_rpid",
    "rpid",
    "author_mid",
    "author_name",
    "like",
    "reply_count",
    "ctime",
    "send_date",
    "send_datetime",
    "message",
]


def clean_message(value: object) -> str:
    return html.unescape(str(value or "")).replace("\r\n", "\n").replace("\r", "\n").strip()


def shanghai_time(timestamp: int) -> tuple[str, str]:
    if not timestamp:
        return "", ""
    dt = datetime.fromtimestamp(timestamp, ZoneInfo("Asia/Shanghai"))
    return dt.strftime("%Y-%m-%d"), dt.strftime("%Y-%m-%d %H:%M:%S")


def convert(input_path: Path, out_dir: Path) -> dict[str, object]:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    view = data["view"]
    bvid = data["bvid"]
    aid = int(data["aid"])
    title = str(view.get("title") or "")
    source_comment_count = int(data.get("source_comment_count") or view.get("stat", {}).get("reply") or 0)
    rows: list[dict[str, object]] = []

    def add_row(reply: dict[str, object], depth: int, root_rpid: str) -> None:
        member = reply.get("member") or {}
        content = reply.get("content") or {}
        ctime = int(reply.get("ctime") or 0)
        send_date, send_datetime = shanghai_time(ctime)
        rows.append(
            {
                "index": len(rows) + 1,
                "depth": depth,
                "bvid": bvid,
                "aid": aid,
                "title": title,
                "source_comment_count": source_comment_count,
                "exported_root_count": 0,
                "exported_reply_count": 0,
                "root_rpid": str(root_rpid or reply.get("rpid_str") or reply.get("rpid") or ""),
                "parent_rpid": str(reply.get("parent_str") or reply.get("parent") or ""),
                "rpid": str(reply.get("rpid_str") or reply.get("rpid") or ""),
                "author_mid": str(member.get("mid") or reply.get("mid") or ""),
                "author_name": str(member.get("uname") or ""),
                "like": int(reply.get("like") or 0),
                "reply_count": int(reply.get("rcount") or reply.get("count") or 0),
                "ctime": ctime,
                "send_date": send_date,
                "send_datetime": send_datetime,
                "message": clean_message(content.get("message") if isinstance(content, dict) else ""),
            }
        )

    nested_by_root = data.get("nestedByRoot") or {}
    for root_reply in data.get("roots") or []:
        root_rpid = str(root_reply.get("rpid_str") or root_reply.get("rpid") or "")
        add_row(root_reply, 0, root_rpid)
        for child in nested_by_root.get(root_rpid, []):
            add_row(child, 1, root_rpid)

    root_count = sum(1 for row in rows if row["depth"] == 0)
    reply_count = len(rows) - root_count
    for row in rows:
        row["exported_root_count"] = root_count
        row["exported_reply_count"] = reply_count

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"bilibili_comments_{bvid}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    md_path = out_dir / f"bilibili_comments_{bvid}.md"
    md_path.write_text(
        "\n".join(
            [
                f"# {title}",
                "",
                f"- BVID: `{bvid}`",
                f"- AID: `{aid}`",
                f"- Source comments: `{source_comment_count}`",
                f"- Exported comments: `{len(rows)}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "csv": str(csv_path),
        "markdown": str(md_path),
        "rows": len(rows),
        "root_count": root_count,
        "reply_count": reply_count,
        "source_comment_count": source_comment_count,
        "complete": len(rows) >= source_comment_count if source_comment_count else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json", help="JSON exported by chrome_page_export_comments.js")
    parser.add_argument("--out-dir", default=".", help="Output directory")
    args = parser.parse_args()

    print(json.dumps(convert(Path(args.input_json), Path(args.out_dir)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
