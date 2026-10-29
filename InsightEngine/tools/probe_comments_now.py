#!/usr/bin/env python3
"""临时探针: 让 _execute_query 的异常穿透, 看到 get_comments_for_topic 的真实错误。
只读, 不改任何业务代码。用完可删。"""
import sys
sys.path.insert(0, '/home/wclaw/projects/BettaFish')

from InsightEngine.tools.search import MediaCrawlerDB

db = MediaCrawlerDB()

# 1. 先看 weibo_note_comment 的内容里有没有"人工智能"
raw = db._execute_query(
    'SELECT id, content, create_date_time FROM weibo_note_comment WHERE content LIKE :t LIMIT 5',
    {'t': '%人工智能%'},
)
print(f'[查询] content LIKE %人工智能% → {len(raw)} 行')
for r in raw[:3]:
    print('   ', str(r.get('content'))[:60])

# 2. 逐个评论表, 看哪张表没有"人工智能"
for t in ['bilibili_video_comment','douyin_aweme_comment','kuaishou_video_comment',
          'weibo_note_comment','xhs_note_comment','zhihu_comment','tieba_comment']:
    try:
        cols = db._get_table_columns(t)
        n = db._execute_query(
            f'SELECT count(*) AS c FROM "{t}" WHERE "content" LIKE :t', {'t': '%人工智能%'})
        total = db._execute_query(f'SELECT count(*) AS c FROM "{t}"')
        print(f'  {t:30s} cols={len(cols):2d} 命中={n[0]["c"] if n else "?"} 总数={total[0]["c"] if total else "?"}')
    except Exception as e:
        print(f'  {t:30s} ERR {type(e).__name__}: {str(e)[:120]}')
