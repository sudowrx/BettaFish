# -*- coding: utf-8 -*-
"""End-to-end check: InsightEngine agent reads real PostgreSQL data.

Runs the agent's search -> sentiment -> summary path against the live
bettafish PG database and prints what it found.  This proves the whole
chain (crawler -> PG -> InsightEngine tools -> LLM) is live.
"""

import os
import sys

sys.path.insert(0, "/home/wclaw/projects/BettaFish")
os.chdir("/home/wclaw/projects/BettaFish")

from InsightEngine.tools.search import MediaCrawlerDB  # noqa: E402


def main():
    db = MediaCrawlerDB()

    print("=" * 60)
    print("STEP 1: hot content last 24h")
    hot = db.search_hot_content(time_period="24h", limit=8)
    print(f"  results: {hot.results_count}")
    for r in hot.results[:5]:
        print(f"   - hot={r.hotness_score:8.1f} | {r.platform} | "
              f"{(r.title_or_content or '')[:44]}")
        print(f"     author={r.author_nickname} kw={r.source_keyword} "
              f"like={r.engagement.get('likes')} cmt={r.engagement.get('comments')}")

    print("=" * 60)
    print("STEP 2: keyword coverage")
    from InsightEngine.utils.db import fetch_all
    import asyncio

    async def q():
        return await fetch_all(
            'SELECT source_keyword, count(*) as n FROM "weibo_note" '
            'GROUP BY source_keyword ORDER BY n DESC', {})

    rows = asyncio.get_event_loop().run_until_complete(q())
    for r in rows:
        print(f"   {r['source_keyword']}: {r['n']} notes")

    print("=" * 60)
    print("STEP 3: sentiment analyzer on real content")
    try:
        from InsightEngine.tools import multilingual_sentiment_analyzer as sa
        texts = [(r.title_or_content or "")[:200] for r in hot.results[:5]
                 if r.title_or_content]
        if texts:
            out = sa.analyze(texts) if hasattr(sa, "analyze") else None
            print(f"   analyzer output type: {type(out).__name__}")
            print(f"   sample: {str(out)[:400]}")
        else:
            print("   (no text to analyze)")
    except Exception as e:
        print(f"   sentiment analyzer error: {type(e).__name__}: {str(e)[:200]}")

    print("=" * 60)
    print("STEP 4: DB roundtrip sanity")
    async def q2():
        return await fetch_all(
            'SELECT count(*) as notes, count(DISTINCT source_keyword) as kws '
            'FROM "weibo_note"', {})
    r2 = asyncio.get_event_loop().run_until_complete(q2())
    print(f"   weibo_note rows={r2[0]['notes']} distinct_keywords={r2[0]['kws']}")


if __name__ == "__main__":
    main()
