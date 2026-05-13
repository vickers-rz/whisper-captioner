# Bilibili Comments Integration

This folder is a staging area for Bilibili comment extraction.

Goal:

- Accept a Bilibili `BV` id or video URL.
- Resolve the video's `aid`.
- Fetch top-level comments and nested replies.
- Export a Reddit-style threaded Markdown file and a flat CSV file.

API notes:

- Resolve video metadata: `https://api.bilibili.com/x/web-interface/view?bvid=<bvid>`
- Fetch top-level comments: `https://api.bilibili.com/x/v2/reply?type=1&oid=<aid>&pn=<page>&ps=<page_size>&sort=<sort>`
- Fetch nested replies: `https://api.bilibili.com/x/v2/reply/reply?type=1&oid=<aid>&root=<rpid>&pn=<page>&ps=<page_size>`

Output shape:

- Markdown preserves comment threads with indentation and score/date metadata.
- CSV preserves one row per comment/reply with `depth`, `root_rpid`, `parent_rpid`, `rpid`, `ctime`, `like`, author fields, and text.

Limitations:

- Public endpoints may return only comments visible to the current anonymous/session context.
- Very large comment sections should be fetched with rate limits and page caps.
- Some deleted, folded, or moderated replies may not be returned.

Safe fallback modes:

1. Direct API mode: run `extract_comments.py` first. This is easiest, but Bilibili may return anti-bot errors such as `-352`.
2. Chrome page mode: run requests from an already-open Bilibili video tab using `chrome_page_export_comments.js`, then convert the resulting JSON with `convert_browser_export.py`.

Chrome page mode intentionally does not read or copy cookies. The browser keeps the login/session state, and only the returned comment JSON is written locally.
