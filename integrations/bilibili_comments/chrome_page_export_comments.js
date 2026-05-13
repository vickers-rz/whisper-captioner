// Run this in a Bilibili video page console or Chrome automation context.
// It fetches comments using the page's own Bilibili session and posts only the
// exported comment JSON to a local receiver. It does not read or copy cookies.

async function exportBilibiliCommentsFromPage(options = {}) {
  const bvid =
    options.bvid ||
    location.href.match(/BV[0-9A-Za-z]+/)?.[0] ||
    document.querySelector('meta[itemprop="url"]')?.content?.match(/BV[0-9A-Za-z]+/)?.[0];
  if (!bvid) throw new Error("Could not find BVID on this page.");

  const receiverUrl = options.receiverUrl || "http://127.0.0.1:17329/";
  const pageSize = options.pageSize || 20;
  const mode = options.mode ?? 2; // 2 = latest, 3 = hot
  const sleepMs = options.sleepMs ?? 180;
  const nestedSleepMs = options.nestedSleepMs ?? 120;
  const maxPages = options.maxPages || 1000;
  const maxNestedPages = options.maxNestedPages || 200;

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const getJson = async (url) => {
    const response = await fetch(url, { credentials: "include" });
    const data = await response.json();
    if (data.code !== 0) throw new Error(`${url} -> ${data.code} ${data.message}`);
    return data.data;
  };

  const view = await getJson(`https://api.bilibili.com/x/web-interface/view?bvid=${encodeURIComponent(bvid)}`);
  const aid = view.aid;
  const roots = [];
  const nestedByRoot = {};
  let sourceCommentCount = view.stat?.reply || 0;
  let next = 0;

  for (let page = 0; page < maxPages; page += 1) {
    const payload = await getJson(
      `https://api.bilibili.com/x/v2/reply/main?type=1&oid=${aid}&mode=${mode}&next=${next}&ps=${pageSize}`
    );
    const replies = payload.replies || [];
    roots.push(...replies);
    if (payload.cursor?.all_count) sourceCommentCount = payload.cursor.all_count;

    for (const reply of replies) {
      const root = String(reply.rpid_str || reply.rpid || "");
      const inlineReplies = reply.replies || [];
      nestedByRoot[root] = inlineReplies.slice();
      const total = Number(reply.rcount || 0);

      if (total > inlineReplies.length) {
        const nested = [];
        for (let pn = 1; pn < maxNestedPages; pn += 1) {
          const childPayload = await getJson(
            `https://api.bilibili.com/x/v2/reply/reply?type=1&oid=${aid}&root=${root}&pn=${pn}&ps=20`
          );
          const children = childPayload.replies || [];
          nested.push(...children);
          const count = Number(childPayload.page?.count || nested.length);
          if (!children.length || nested.length >= count) break;
          await sleep(nestedSleepMs);
        }
        nestedByRoot[root] = nested;
      }
      await sleep(60);
    }

    if (payload.cursor?.is_end || !payload.cursor?.next || !replies.length) break;
    next = payload.cursor.next;
    await sleep(sleepMs);
  }

  const result = {
    bvid,
    aid,
    view,
    source_comment_count: sourceCommentCount,
    root_count: roots.length,
    roots,
    nestedByRoot,
  };

  const post = await fetch(receiverUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(result),
  }).then((response) => response.json());

  return {
    post,
    source_comment_count: sourceCommentCount,
    root_count: roots.length,
    nested_total: Object.values(nestedByRoot).reduce((sum, replies) => sum + replies.length, 0),
  };
}

exportBilibiliCommentsFromPage();
