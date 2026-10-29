"""给 weibo_note 增加情感标注列并回填。

背景：LLM 报告里的"正面%/负面%/中性%"全是编的 —— weibo_note 没有情感列。
本脚本用已有的 WeiboMultilingualSentimentAnalyzer 离线补标，标签和置信度写回库。
"""
import os, sys, asyncio
sys.path.insert(0, "/home/wclaw/projects/BettaFish")
os.chdir("/home/wclaw/projects/BettaFish/InsightEngine")
from dotenv import load_dotenv
load_dotenv("/home/wclaw/projects/BettaFish/.env")

import asyncpg
from InsightEngine.tools.sentiment_analyzer import multilingual_sentiment_analyzer as SA


async def ensure_columns(conn):
    await conn.execute('''
        ALTER TABLE weibo_note
            ADD COLUMN IF NOT EXISTS sentiment_label VARCHAR(32),
            ADD COLUMN IF NOT EXISTS sentiment_confidence DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS sentiment_probs JSONB
    ''')
    print("列已就绪:", await conn.fetchval("""
        SELECT string_agg(column_name, ', ' ORDER BY column_name)
        FROM information_schema.columns
        WHERE table_name='weibo_note' AND column_name LIKE 'sentiment%'
    """))


async def main():
    conn = await asyncpg.connect(
        host=os.environ["DB_HOST"], port=int(os.environ["DB_PORT"]),
        user=os.environ["DB_USER"], password=os.environ["DB_PASSWORD"],
        database=os.environ["DB_NAME"])
    await ensure_columns(conn)

    print("初始化情感模型...")
    SA.initialize()
    print("  is_initialized =", SA.is_initialized)

    rows = await conn.fetch("""
        SELECT id, content FROM weibo_note
        WHERE sentiment_label IS NULL AND content IS NOT NULL AND content <> ''
        ORDER BY id
    """)
    print(f"待标注 {len(rows)} 条")

    ok = fail = 0
    dist = {}
    for i, r in enumerate(rows, 1):
        res = SA.analyze_single_text(r["content"])
        if res.success and res.analysis_performed:
            await conn.execute(
                "UPDATE weibo_note SET sentiment_label=$1, sentiment_confidence=$2, sentiment_probs=$3 WHERE id=$4",
                res.sentiment_label, res.confidence,
                __import__("json").dumps(res.probability_distribution), r["id"])
            dist[res.sentiment_label] = dist.get(res.sentiment_label, 0) + 1
            ok += 1
        else:
            fail += 1
        if i % 10 == 0:
            print(f"  {i}/{len(rows)}")

    print(f"\n标注完成: 成功 {ok}, 失败 {fail}")
    print("标签分布:", dist)
    if ok:
        print("库内占比:", {k: f"{v/ok*100:.1f}%" for k, v in dist.items()})

    # 验证
    print("\n验证读取:")
    for row in await conn.fetch("""
        SELECT sentiment_label, COUNT(*) n FROM weibo_note
        WHERE sentiment_label IS NOT NULL GROUP BY sentiment_label ORDER BY n DESC
    """):
        print(f"  {row['sentiment_label']}: {row['n']}")

    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
