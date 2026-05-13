#!/usr/bin/env python3
"""Render exported Bilibili comments CSV as a Reddit-style static HTML tree."""

from __future__ import annotations

import argparse
import csv
import html
from collections import defaultdict
from pathlib import Path


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def render_comment(
    row: dict[str, str],
    children_by_parent: dict[str, list[dict[str, str]]],
    rows_by_rpid: dict[str, dict[str, str]],
    depth: int = 0,
    root_index: int = 0,
) -> str:
    author = html.escape(row.get("author_name") or row.get("author_mid") or "unknown")
    message = html.escape(row.get("message") or "").replace("\n", "<br>")
    send_datetime = html.escape(row.get("send_datetime") or "")
    like = html.escape(row.get("like") or "0")
    rpid = html.escape(row.get("rpid") or "")
    parent_rpid = row.get("parent_rpid") or ""
    root_rpid = row.get("root_rpid") or ""
    parent = rows_by_rpid.get(parent_rpid)
    parent_author = ""
    if parent and parent_rpid != root_rpid:
        parent_author = parent.get("author_name") or parent.get("author_mid") or ""

    replying_to = (
        f'<a class="reply-target" href="#comment-{html.escape(parent_rpid)}">回复 @{html.escape(parent_author)}</a>'
        if parent_author
        else ""
    )
    avatar_text = html.escape((row.get("author_name") or row.get("author_mid") or "?")[:1].upper())

    child_rows = children_by_parent.get(row.get("rpid") or "", [])
    child_count = len(child_rows)
    child_html = "\n".join(
        render_comment(child, children_by_parent, rows_by_rpid, depth + 1, root_index)
        for child in child_rows
    )
    child_block = f'<div class="children">{child_html}</div>' if child_html else ""
    toggle_disabled = "" if child_rows else " disabled"
    toggle_label = "Collapse comment thread" if child_rows else "No replies"
    toggle_text = "-" if child_rows else "·"

    return f"""
<article class="comment" id="comment-{rpid}" data-depth="{depth}" data-root-index="{root_index}" data-rpid="{rpid}" style="--depth: {depth};">
  <div class="thread-gutter">
    <button class="collapse-dot" aria-label="{toggle_label}" data-toggle="thread"{toggle_disabled}>{toggle_text}</button>
    <div class="thread-line"></div>
  </div>
  <div class="comment-body">
    <div class="comment-head">
      <div class="avatar">{avatar_text}</div>
      <div class="meta">
        <span class="author">{author}</span>
        <span class="dot">•</span>
        <span>{send_datetime}</span>
        {replying_to}
      </div>
    </div>
    <div class="message">{message}</div>
    <div class="actions">
      <span class="score">{like}</span>
      <span class="vote heart">♥</span>
      <span><span class="reply-count">{child_count}</span> 条下级回复</span>
      <span>展开/收起</span>
      <a href="#comment-{rpid}">#{rpid}</a>
    </div>
    {child_block}
  </div>
</article>""".strip()


def render_html(rows: list[dict[str, str]]) -> str:
    title = rows[0].get("title") if rows else "Bilibili Comments"
    bvid = rows[0].get("bvid") if rows else ""
    aid = rows[0].get("aid") if rows else ""
    source_comment_count = rows[0].get("source_comment_count") if rows else ""

    rows_by_rpid = {row.get("rpid") or "": row for row in rows}
    roots: list[dict[str, str]] = []
    children_by_parent: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        rpid = row.get("rpid") or ""
        parent_rpid = row.get("parent_rpid") or ""
        if row.get("depth") == "0" or not parent_rpid or parent_rpid == "0" or parent_rpid == rpid:
            roots.append(row)
        else:
            children_by_parent[parent_rpid].append(row)

    comments_html = "\n".join(
        render_comment(root, children_by_parent, rows_by_rpid, root_index=index)
        for index, root in enumerate(roots, start=1)
    )
    exported_reply_count = max(len(rows) - len(roots), 0)
    exported_total_count = len(rows)
    page_size = 20
    page_count = max((len(roots) + page_size - 1) // page_size, 1)
    try:
        source_total = int(source_comment_count or 0)
    except ValueError:
        source_total = 0
    completeness_label = "完整导出" if source_total and exported_total_count >= source_total else "未完整导出"
    completeness_class = "ok" if completeness_label == "完整导出" else "warn"
    source_pill = (
        f'<span class="pill source-count">源评论总数: {html.escape(source_comment_count)}</span>'
        if source_comment_count
        else ""
    )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title or "Bilibili Comments")}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #ffffff;
      --surface: #ffffff;
      --surface-hover: #f7f8f9;
      --text: #1a1a1b;
      --muted: #576f76;
      --faint: #878a8c;
      --line: #edeff1;
      --line-strong: #d7dadd;
      --accent: #ff4500;
      --link: #0079d3;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Microsoft YaHei", sans-serif;
      font-size: 15px;
      line-height: 1.62;
      letter-spacing: 0.01em;
    }}

    header {{
      position: sticky;
      top: 0;
      z-index: 20;
      background: rgba(255, 255, 255, 0.96);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(12px);
      box-shadow: 0 2px 10px rgba(26, 26, 27, 0.06);
    }}

    .header-inner {{
      width: min(1440px, calc(100% - 56px));
      margin: 0 auto;
      padding: 14px 0;
    }}

    h1 {{
      margin: 0;
      font-size: 22px;
      font-weight: 700;
      letter-spacing: 0;
    }}

    .summary {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
    }}

    .pager {{
      position: sticky;
      top: 0;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px;
      margin-top: 12px;
    }}

    .pager button {{
      min-height: 30px;
      padding: 4px 12px;
      border: 1px solid var(--line-strong);
      border-radius: 999px;
      background: var(--surface);
      color: var(--text);
      font: inherit;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
    }}

    .pager button:disabled {{
      cursor: not-allowed;
      opacity: 0.45;
    }}

    .pager-status {{
      color: var(--muted);
      font-size: 13px;
      font-weight: 600;
    }}

    .pill {{
      display: inline-flex;
      align-items: center;
      min-height: 26px;
      padding: 3px 9px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--surface);
    }}

    .pill.ok {{
      border-color: #9bd4b4;
      background: #f0fff6;
      color: #1f7a3f;
    }}

    .pill.warn {{
      border-color: #ffd08a;
      background: #fff8e8;
      color: #9a5a00;
    }}

    main {{
      width: min(1440px, calc(100% - 56px));
      margin: 16px auto 48px;
    }}

    .thread {{
      display: grid;
      gap: 18px;
      padding: 4px 0 24px;
    }}

    .comment {{
      display: grid;
      grid-template-columns: 38px minmax(0, 1fr);
      column-gap: 0;
      position: relative;
      background: var(--surface);
      border: 0;
      border-radius: 0;
      padding: 0;
    }}

    .comment:hover > .comment-body {{
      background: var(--surface-hover);
    }}

    .thread > .comment.is-selected > .comment-body {{
      background: #fff7fb;
      outline: 2px solid #fb7299;
      outline-offset: 2px;
    }}

    .thread > .comment.is-focused-root > .comment-body {{
      background: #fff4f8;
      outline: 3px solid #fb7299;
      outline-offset: 3px;
    }}

    .thread-gutter {{
      display: flex;
      flex-direction: column;
      align-items: center;
      padding-top: 6px;
    }}

    .collapse-dot {{
      width: 18px;
      height: 18px;
      border: 1px solid var(--line-strong);
      border-radius: 50%;
      background: var(--surface);
      color: var(--muted);
      font-size: 13px;
      line-height: 14px;
      padding: 0;
      cursor: default;
      font-weight: 700;
    }}

    .collapse-dot:not(:disabled) {{
      cursor: pointer;
    }}

    .collapse-dot:disabled {{
      color: #a8b0b7;
      background: #f8fafc;
      opacity: 1;
    }}

    .thread-line {{
      flex: 1;
      width: 18px;
      min-height: 18px;
      margin-top: 4px;
      position: relative;
      background: transparent;
    }}

    .thread-line::before {{
      content: "";
      position: absolute;
      top: 0;
      bottom: 14px;
      left: 8px;
      width: 2px;
      background: var(--line-strong);
      border-radius: 999px;
    }}

    .thread-line::after {{
      content: "";
      position: absolute;
      bottom: 0;
      left: 8px;
      width: 18px;
      height: 16px;
      border-left: 2px solid var(--line-strong);
      border-bottom: 2px solid var(--line-strong);
      border-bottom-left-radius: 12px;
    }}

    .comment-body {{
      min-width: 0;
      padding: 6px 8px 9px 0;
      border-radius: 4px;
      max-width: 100%;
    }}

    .comment-head {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
      align-items: center;
      color: var(--muted);
      font-size: 13px;
    }}

    .meta a {{
      color: var(--link);
      text-decoration: none;
    }}

    .meta a:hover,
    .reply-target:hover {{
      color: var(--accent);
      text-decoration: underline;
    }}

    .avatar {{
      display: inline-grid;
      width: 28px;
      height: 28px;
      flex: 0 0 28px;
      place-items: center;
      border-radius: 50%;
      background: #dbe4ea;
      color: #33444d;
      font-size: 12px;
      font-weight: 700;
    }}

    .author {{
      color: var(--text);
      font-weight: 700;
    }}

    .dot {{
      color: var(--faint);
    }}

    .message {{
      margin: 8px 0 0 36px;
      padding: 10px 12px;
      width: fit-content;
      max-width: calc(100% - 36px);
      overflow-wrap: anywhere;
      white-space: normal;
      background: #f8fafc;
      border: 1px solid #e3e8ee;
      border-radius: 10px;
      color: var(--text);
      font-size: 15px;
      line-height: 1.72;
      letter-spacing: 0.018em;
      box-shadow: 0 1px 0 rgba(26, 26, 27, 0.03);
    }}

    .children .message {{
      background: #ffffff;
      border-color: #e7ebf0;
    }}

    .actions {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
      margin: 9px 0 0 36px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 600;
    }}

    .actions a {{
      color: var(--muted);
      text-decoration: none;
    }}

    .actions a:hover {{
      color: var(--link);
      text-decoration: underline;
    }}

    .vote {{
      color: var(--faint);
      font-size: 17px;
      line-height: 1;
    }}

    .heart {{
      color: #fb7299;
    }}

    .score {{
      min-width: 12px;
      color: var(--muted);
      text-align: center;
    }}

    .reply-count {{
      color: #f59e0b;
      font-weight: 800;
    }}

    .children {{
      display: grid;
      gap: 0;
      margin-top: 4px;
      margin-left: 0;
      padding-left: 28px;
      border-left: 0;
    }}

    .comment.is-collapsed > .comment-body > .children {{
      display: none;
    }}

    .comment.is-page-hidden {{
      display: none;
    }}

    .children .comment {{
      background: transparent;
    }}

    .children .comment::before {{
      content: "";
      position: absolute;
      top: 15px;
      left: -28px;
      width: 28px;
      height: 18px;
      border-left: 2px solid var(--line-strong);
      border-bottom: 2px solid var(--line-strong);
      border-bottom-left-radius: 12px;
    }}

    .children .children {{
      margin-left: 0;
      padding-left: 28px;
    }}

    .reply-target {{
      color: var(--link);
      font-weight: 600;
      text-decoration: none;
    }}

    .empty {{
      padding: 28px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      color: var(--muted);
      text-align: center;
    }}

    @media (max-width: 640px) {{
      .header-inner,
      main {{
        width: min(100% - 20px, 1440px);
      }}

      h1 {{
        font-size: 20px;
      }}

      .comment {{
        grid-template-columns: 30px minmax(0, 1fr);
      }}

      .children {{
        padding-left: 18px;
      }}

      .children .comment::before {{
        left: -18px;
        width: 18px;
      }}

      .message,
      .actions {{
        margin-left: 0;
      }}

      .message {{
        max-width: 100%;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="header-inner">
      <h1>{html.escape(title or "Bilibili Comments")}</h1>
      <div class="summary">
        <span class="pill">BVID: {html.escape(bvid or "")}</span>
        <span class="pill">AID: {html.escape(aid or "")}</span>
        {source_pill}
        <span class="pill {completeness_class}">{completeness_label}: {exported_total_count} / {html.escape(source_comment_count or str(exported_total_count))}</span>
        <span class="pill">已导出一级评论: {len(roots)}</span>
        <span class="pill">已导出回复: {exported_reply_count}</span>
      </div>
      <div class="pager" data-page-size="{page_size}" data-page-count="{page_count}" data-total-roots="{len(roots)}">
        <button type="button" data-page-action="prev">上一页</button>
        <span class="pager-status" data-page-status>第 1 / {page_count} 页</span>
        <button type="button" data-page-action="next">下一页</button>
        <button type="button" data-collapse-all>只看一级评论</button>
        <button type="button" data-focus-current>聚焦当前一级评论</button>
      </div>
    </div>
  </header>
  <main>
    <section class="thread">
      {comments_html if comments_html else '<div class="empty">No comments found.</div>'}
    </section>
  </main>
  <script>
    (() => {{
      const pager = document.querySelector("[data-page-size]");
      const pageSize = Number(pager?.dataset.pageSize || 20);
      const pageCount = Number(pager?.dataset.pageCount || 1);
      const totalRoots = Number(pager?.dataset.totalRoots || 0);
      const status = document.querySelector("[data-page-status]");
      const prev = document.querySelector('[data-page-action="prev"]');
      const next = document.querySelector('[data-page-action="next"]');
      const collapseAll = document.querySelector('[data-collapse-all]');
      const focusCurrent = document.querySelector('[data-focus-current]');
      let page = 1;
      let rootsOnly = false;
      let focusedRootIndex = null;
      let selectedRootIndex = null;

      function rootComments() {{
        return Array.from(document.querySelectorAll('.thread > .comment[data-root-index]'));
      }}

      function setThreadCollapsed(comment, collapsed) {{
        const button = comment.querySelector(':scope > .thread-gutter > [data-toggle="thread"]');
        if (!button || button.disabled) return;
        comment.classList.toggle('is-collapsed', collapsed);
        button.textContent = collapsed ? '+' : '-';
        button.setAttribute('aria-label', collapsed ? 'Expand comment thread' : 'Collapse comment thread');
      }}

      function expandSubtree(root) {{
        root.querySelectorAll('.comment').forEach((comment) => setThreadCollapsed(comment, false));
      }}

      function currentRootIndex() {{
        const headerBottom = document.querySelector('header')?.getBoundingClientRect().bottom || 0;
        const visibleRoots = rootComments().filter((node) => !node.classList.contains('is-page-hidden'));
        let best = visibleRoots[0];
        let bestDistance = Number.POSITIVE_INFINITY;
        visibleRoots.forEach((node) => {{
          const rect = node.getBoundingClientRect();
          if (rect.bottom <= headerBottom) return;
          const distance = Math.abs(rect.top - headerBottom - 12);
          if (distance < bestDistance) {{
            best = node;
            bestDistance = distance;
          }}
        }});
        return best ? Number(best.dataset.rootIndex || 0) : null;
      }}

      function updatePage() {{
        const start = (page - 1) * pageSize + 1;
        const end = Math.min(page * pageSize, totalRoots);
        rootComments().forEach((node) => {{
          const rootIndex = Number(node.dataset.rootIndex || 0);
          const outsidePage = rootIndex < start || rootIndex > end;
          const outsideFocus = focusedRootIndex !== null && rootIndex !== focusedRootIndex;
          node.classList.toggle('is-page-hidden', outsidePage || outsideFocus);
          node.classList.toggle('is-focused-root', focusedRootIndex !== null && rootIndex === focusedRootIndex);
          node.classList.toggle('is-selected', focusedRootIndex === null && selectedRootIndex !== null && rootIndex === selectedRootIndex);
        }});
        if (status) {{
          const focusText = focusedRootIndex !== null ? ` · 聚焦第 ${{focusedRootIndex}} 条` : "";
          status.textContent = `第 ${{page}} / ${{pageCount}} 页 · 显示 ${{totalRoots ? start : 0}}-${{end}} / ${{totalRoots}} 条一级评论${{focusText}}`;
        }}
        if (prev) prev.disabled = page <= 1;
        if (next) next.disabled = page >= pageCount;
      }}

      prev?.addEventListener('click', () => {{
        if (page > 1) {{
          page -= 1;
          focusedRootIndex = null;
          selectedRootIndex = null;
          if (focusCurrent) focusCurrent.textContent = '聚焦当前一级评论';
          updatePage();
          window.scrollTo({{ top: 0, behavior: 'smooth' }});
        }}
      }});

      next?.addEventListener('click', () => {{
        if (page < pageCount) {{
          page += 1;
          focusedRootIndex = null;
          selectedRootIndex = null;
          if (focusCurrent) focusCurrent.textContent = '聚焦当前一级评论';
          updatePage();
          window.scrollTo({{ top: 0, behavior: 'smooth' }});
        }}
      }});

      document.querySelectorAll('[data-toggle="thread"]').forEach((button) => {{
        button.addEventListener('click', () => {{
          const comment = button.closest('.comment');
          if (!comment || button.disabled) return;
          const collapsed = comment.classList.toggle('is-collapsed');
          button.textContent = collapsed ? '+' : '-';
          button.setAttribute('aria-label', collapsed ? 'Expand comment thread' : 'Collapse comment thread');
        }});
      }});

      rootComments().forEach((root) => {{
        root.addEventListener('click', (event) => {{
          if (event.target.closest('button, a')) return;
          if (focusedRootIndex !== null) return;
          selectedRootIndex = Number(root.dataset.rootIndex || 0);
          updatePage();
        }});
      }});

      collapseAll?.addEventListener('click', () => {{
        rootsOnly = !rootsOnly;
        document.querySelectorAll('.comment').forEach((comment) => {{
          const shouldCollapse = rootsOnly ? Number(comment.dataset.depth || 0) === 0 : false;
          setThreadCollapsed(comment, shouldCollapse);
        }});
        collapseAll.textContent = rootsOnly ? '全部展开' : '只看一级评论';
      }});

      focusCurrent?.addEventListener('click', () => {{
        if (focusedRootIndex !== null) {{
          focusedRootIndex = null;
          focusCurrent.textContent = '聚焦当前一级评论';
          updatePage();
          return;
        }}
        focusedRootIndex = selectedRootIndex || currentRootIndex();
        const focusedRoot = document.querySelector(`.thread > .comment[data-root-index="${{focusedRootIndex}}"]`);
        if (focusedRoot) {{
          expandSubtree(focusedRoot);
          rootsOnly = false;
          selectedRootIndex = null;
          if (collapseAll) collapseAll.textContent = '只看一级评论';
          focusCurrent.textContent = '退出聚焦';
          updatePage();
          focusedRoot.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
        }}
      }});

      updatePage();
    }})();
  </script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="CSV exported by extract_comments.py")
    parser.add_argument("--out", help="Output HTML path")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    out_path = Path(args.out) if args.out else csv_path.with_suffix(".html")
    rows = read_rows(csv_path)
    out_path.write_text(render_html(rows), encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
