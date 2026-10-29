# -*- coding: utf-8 -*-
"""Verify every InsightEngine DB tool against PostgreSQL.

MediaCrawlerDB._execute_query swallows exceptions and returns [], which makes
broken tools look like "0 results".  So here we monkeypatch _execute_query to
re-raise, then run each tool and report the real outcome.
"""

import os
import sys
import traceback

sys.path.insert(0, "/home/wclaw/projects/BettaFish")
os.chdir("/home/wclaw/projects/BettaFish")

from InsightEngine.tools.search import MediaCrawlerDB  # noqa: E402
from InsightEngine.utils.config import settings  # noqa: E402

# make failures loud instead of silently returning []
_orig = MediaCrawlerDB._execute_query


def loud_execute(self, query, params=None):
    import asyncio
    from InsightEngine.utils.db import fetch_all

    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(fetch_all(query, params))


MediaCrawlerDB._execute_query = loud_execute


def probe(name, fn):
    try:
        res = fn()
        if isinstance(res, list):
            print(f"[PASS] {name}: {len(res)} columns -> {res[:6]}")
            return True
        print(f"[PASS] {name}: results_count={res.results_count}")
        for r in res.results[:2]:
            title = (r.title_or_content or "")[:40].replace("\n", " ")
            print(f"         - [{r.platform}/{r.content_type}] {title} | hot={r.hotness_score}")
        return True
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
        return False


def main():
    print(f"DB_DIALECT={settings.DB_DIALECT!r} DB={settings.DB_HOST}/{settings.DB_NAME}")
    db = MediaCrawlerDB()
    results = []
    results.append(probe("search_hot_content(week)",
                         lambda: db.search_hot_content(time_period="week", limit=10)))
    results.append(probe("search_hot_content(24h)",
                         lambda: db.search_hot_content(time_period="24h", limit=10)))
    results.append(probe("search_hot_content(year)",
                         lambda: db.search_hot_content(time_period="year", limit=10)))
    results.append(probe("search_topic_globally(人工智能)",
                         lambda: db.search_topic_globally("人工智能", limit_per_table=5)))
    results.append(probe("search_topic_by_date",
                         lambda: db.search_topic_by_date("人工智能", "2026-10-01", "2026-12-31", 5)))
    results.append(probe("search_topic_on_platform(weibo)",
                         lambda: db.search_topic_on_platform("weibo", "人工智能", limit=5)))
    results.append(probe("search_topic_on_platform(weibo, dated)",
                         lambda: db.search_topic_on_platform(
                             "weibo", "人工智能", "2026-10-01", "2026-12-31", 5)))
    results.append(probe("get_comments_for_topic",
                         lambda: db.get_comments_for_topic("人工智能", limit=5)))
    results.append(probe("_get_table_columns(weibo_note)",
                         lambda: db._get_table_columns("weibo_note")))
    print(f"\n=== {sum(results)}/{len(results)} tools working ===")
    return 0 if all(results) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(2)
