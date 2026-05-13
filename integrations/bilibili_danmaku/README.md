# Bilibili Danmaku Integration

This folder is a staging area for adding Bilibili danmaku extraction into the app.

Current artifact:

- `bilibili_danmaku_BV1dvfsBrEFJ.csv`: exported danmaku for Bilibili video `BV1dvfsBrEFJ`

CSV columns:

- `index`: row number after sorting
- `bvid`: Bilibili video id
- `cid`: Bilibili content id used by the danmaku endpoint
- `title`: video title
- `video_seconds`: danmaku position on the video timeline
- `video_time`: formatted video timeline position
- `send_date`: send date in `Asia/Shanghai`
- `send_datetime`: send datetime in `Asia/Shanghai`
- `send_timestamp`: original Unix timestamp from Bilibili
- `mode`, `size`, `color`, `pool`: Bilibili danmaku display metadata
- `user_hash`: hashed sender id from the danmaku payload
- `danmaku_id`: Bilibili danmaku id
- `text`: danmaku text

Future integration notes:

- Resolve `cid` from `https://api.bilibili.com/x/web-interface/view?bvid=<bvid>`.
- Fetch danmaku XML from `https://api.bilibili.com/x/v1/dm/list.so?oid=<cid>`.
- Parse `<d p="...">text</d>` entries, sort by video timestamp, and expose them as a timeline track.
- Consider adding import/export support near the existing subtitle or transcript flows.
