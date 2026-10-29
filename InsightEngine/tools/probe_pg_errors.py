# -*- coding: utf-8 -*-
"""Show the real exception for each InsightEngine DB tool (the tools swallow
errors via _execute_query, so we bypass it and call fetch_all directly)."""

import asyncio
import sys

sys.path.insert(0, "/home/wclaw/projects/BettaFish")

from InsightEngine.utils.db import fetch_all  # noqa: E402


async def main():
    cases = {
        "search_hot_content (MySQL dialect)": (
            "SELECT 'weibo' as p, 'note' as t, content as title, nickname as author, "
            "note_url as url, create_date_time as ts, "
            "(COALESCE(CAST(liked_count AS UNSIGNED), 0) * 1.0) as hotness_score, "
            "source_keyword, 'weibo_note' as tbl FROM `weibo_note` "
            "WHERE `create_date_time` >= %s ORDER BY hotness_score DESC LIMIT %s",
            ("2026-10-01 00:00:00", 5),
        ),
        "_get_table_columns (SHOW COLUMNS)": (
            "SHOW COLUMNS FROM `weibo_note`", None),
        "get_comments_for_topic (MySQL UNION)": (
            "SELECT 'weibo' as platform, `content`, `nickname` as author, "
            "`create_date_time` as ts, '0' as likes, 'weibo_note_comment' as source_table "
            "FROM `weibo_note_comment` WHERE `content` LIKE %s",
            ("%人工智能%",),
        ),
        "search_topic_globally (PG dialect, works)": (
            'SELECT * FROM "weibo_note" WHERE "content" LIKE :t ORDER BY id DESC LIMIT :limit',
            {"t": "%人工智能%", "limit": 5},
        ),
    }
    for name, (q, p) in cases.items():
        try:
            r = await fetch_all(q, p)
            print(f"[OK]   {name}: {len(r)} rows")
        except Exception as e:
            print(f"[ERR]  {name}: {type(e).__name__}: {str(e).splitlines()[0][:160]}")


asyncio.run(main())
