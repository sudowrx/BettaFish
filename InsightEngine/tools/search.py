"""
专为 AI Agent 设计的本地舆情数据库查询工具集 (MediaCrawlerDB)

版本: 3.0
最后更新: 2025-08-23

此脚本将复杂的本地MySQL数据库查询功能封装成一系列目标明确、参数清晰的独立工具，
专为AI Agent调用而设计。Agent只需根据任务意图（如搜索热点、全局搜索话题、
按时间范围分析、获取评论）选择合适的工具，无需编写复杂的SQL语句。

V3.0 核心更新:
- 智能热度计算: `search_hot_content`不再需要`sort_by`参数，改为内部使用统一的加权热度算法，
  综合点赞、评论、分享、观看等数据计算热度分值，使结果更智能、更符合综合热度。
- 新增平台精搜工具: 新增 `search_topic_on_platform` 工具，作为特例，
  允许Agent在特定平台（B站、微博等七大平台）上对某一话题进行精确搜索，并支持时间筛选。
- 结构优化: 调整了数据结构与函数文档，以适应新功能。

主要工具:
- search_hot_content: 查找指定时间范围内的综合热度最高的内容。
- search_topic_globally: 在整个数据库中全局搜索与特定话题相关的所有内容和评论。
- search_topic_by_date: 在指定的历史日期范围内搜索与特定话题相关的内容。
- get_comments_for_topic: 专门提取公众对于某一特定话题的评论数据。
- search_topic_on_platform: 在指定的单个社交媒体平台上搜索特定话题。
"""

import os
import json
from loguru import logger
import asyncio
import threading
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field
from ..utils.db import fetch_all
from datetime import datetime, timedelta, date
from InsightEngine.utils.config import settings

# --- 1. 数据结构定义 ---

@dataclass
class QueryResult:
    """统一的数据库查询结果数据类"""
    platform: str
    content_type: str
    title_or_content: str
    author_nickname: Optional[str] = None
    url: Optional[str] = None
    publish_time: Optional[datetime] = None
    engagement: Dict[str, int] = field(default_factory=dict)
    source_keyword: Optional[str] = None
    hotness_score: float = 0.0
    source_table: str = ""
    sentiment_label: Optional[str] = None
    sentiment_confidence: Optional[float] = None

@dataclass
class DBResponse:
    """封装工具的完整返回结果"""
    tool_name: str
    parameters: Dict[str, Any]
    results: List[QueryResult] = field(default_factory=list)
    results_count: int = 0
    error_message: Optional[str] = None

# --- 2. 核心客户端与专用工具集 ---

class MediaCrawlerDB:
    """包含多种专用舆情数据库查询工具的客户端"""
    # 权重定义
    W_LIKE = 1.0
    W_COMMENT = 5.0
    W_SHARE = 10.0  # 分享/转发/收藏/投币等高价值互动
    W_VIEW = 0.1
    W_DANMAKU = 0.5

    def __init__(self):
        """
        初始化客户端。
        """
        pass
        
    def _execute_query(self, query: str, params: Any = None) -> List[Dict[str, Any]]:
        """
        执行数据库查询。

        这里必须处理"event loop 已在运行"的场景：
        - agent 流程是 async 的，调用本方法时当前线程已经有一个正在运行的 loop，
          此时 loop.run_until_complete() 会抛 "This event loop is already running"
        - 原实现捕获异常后 return []，导致所有查询静默返回空结果，
          LLM 拿不到数据就只能编造内容（曾出现"17个平台/3.2亿用户"的幻觉数据）

        修复：若当前 loop 正在运行，则在独立线程里新建 loop 执行。
        """
        try:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            if loop.is_running():
                # 当前线程已有运行中的 loop -> 用独立线程跑一个新的 loop
                return self._run_in_new_thread(query, params)

            # 直接运行协程
            return loop.run_until_complete(fetch_all(query, params))

        except Exception as e:
            logger.exception(f"数据库查询时发生错误: {e}")
            # 不再静默吞掉：至少要让上层知道这次查询失败了
            logger.warning(f"查询失败并返回空结果: {str(query)[:120]}")
            return []

    @staticmethod
    def _run_in_new_thread(query: str, params: Any = None) -> List[Dict[str, Any]]:
        """在独立线程的新 event loop 中执行查询（用于当前 loop 已在运行的场景）"""
        result_box = {}
        error_box = {}

        def _runner():
            new_loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(new_loop)
                result_box["result"] = new_loop.run_until_complete(fetch_all(query, params))
            except Exception as e:  # noqa: BLE001
                error_box["error"] = e
            finally:
                # 清理这个临时 loop 的 engine（连接池），否则会随查询次数无限增长
                # 注意：必须在 close 之前 dispose
                try:
                    from ..utils.db import dispose_engine_for_loop

                    dispose_engine_for_loop(new_loop)
                except Exception:
                    pass
                try:
                    new_loop.close()
                except Exception:
                    pass

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()

        if "error" in error_box:
            raise error_box["error"]
        return result_box.get("result", [])

    @staticmethod
    def _to_datetime(ts: Any) -> Optional[datetime]:
        if not ts: return None
        try:
            if isinstance(ts, datetime): return ts
            if isinstance(ts, date): return datetime.combine(ts, datetime.min.time())
            if isinstance(ts, (int, float)) or str(ts).isdigit():
                val = float(ts)
                return datetime.fromtimestamp(val / 1000 if val > 1_000_000_000_000 else val)
            if isinstance(ts, str):
                return datetime.fromisoformat(ts.split('+')[0].strip())
        except (ValueError, TypeError): return None

    _table_columns_cache = {}

    def _extract_engagement(self, row: Dict[str, Any]) -> Dict[str, int]:
        """从数据行中提取并统一互动指标"""
        engagement = {}
        mapping = { 'likes': ['liked_count', 'like_count', 'voteup_count', 'comment_like_count'], 'comments': ['video_comment', 'comments_count', 'comment_count', 'total_replay_num', 'sub_comment_count'], 'shares': ['video_share_count', 'shared_count', 'share_count', 'total_forwards'], 'views': ['video_play_count', 'viewd_count'], 'favorites': ['video_favorite_count', 'collected_count'], 'coins': ['video_coin_count'], 'danmaku': ['video_danmaku'], }
        for key, potential_cols in mapping.items():
            for col in potential_cols:
                if col in row and row[col] is not None:
                    try: engagement[key] = int(row[col])
                    except (ValueError, TypeError): engagement[key] = 0
                    break
        return engagement

    def _extract_sentiment(self, row: Dict[str, Any]) -> tuple:
        """提取情感标注（若该表已标注）。

        返回 (label, confidence)；未标注时返回 (None, None)。
        """
        label = row.get("sentiment_label")
        if not label:
            return None, None
        try:
            conf = float(row.get("sentiment_confidence")) if row.get("sentiment_confidence") is not None else None
        except (TypeError, ValueError):
            conf = None
        return label, conf

    def search_hot_content(
        self,
        time_period: Literal['24h', 'week', 'year'] = 'week',
        limit: int = 50
    ) -> DBResponse:
        """
        【工具】查找热点内容: 获取最近一段时间内综合热度最高的内容。

        Args:
            time_period (Literal['24h', 'week', 'year']): 时间范围，默认为 'week'。
            limit (int): 返回结果的最大数量，默认为 50。

        Returns:
            DBResponse: 包含按综合热度排序后的内容列表。
        """
        params_for_log = {'time_period': time_period, 'limit': limit}
        logger.info(f"--- TOOL: 查找热点内容 (params: {params_for_log}) ---")
        
        now = datetime.now()
        start_time = now - timedelta(days={'24h': 1, 'week': 7}.get(time_period, 365))

        # 定义各平台的热度计算SQL片段
        # 计数类字段在 MediaCrawler 里是 Text（可能有非数字脏值），统一 CAST 成整数
        hotness_formulas = {
            'bilibili_video': f"(COALESCE({self._cast_int('liked_count')}, 0) * {self.W_LIKE} + COALESCE({self._cast_int('video_comment')}, 0) * {self.W_COMMENT} + COALESCE({self._cast_int('video_share_count')}, 0) * {self.W_SHARE} + COALESCE({self._cast_int('video_favorite_count')}, 0) * {self.W_SHARE} + COALESCE({self._cast_int('video_coin_count')}, 0) * {self.W_SHARE} + COALESCE({self._cast_int('video_danmaku')}, 0) * {self.W_DANMAKU} + COALESCE({self._cast_float('video_play_count')}, 0) * {self.W_VIEW})",
            'douyin_aweme':   f"(COALESCE({self._cast_int('liked_count')}, 0) * {self.W_LIKE} + COALESCE({self._cast_int('comment_count')}, 0) * {self.W_COMMENT} + COALESCE({self._cast_int('share_count')}, 0) * {self.W_SHARE} + COALESCE({self._cast_int('collected_count')}, 0) * {self.W_SHARE})",
            'weibo_note':     f"(COALESCE({self._cast_int('liked_count')}, 0) * {self.W_LIKE} + COALESCE({self._cast_int('comments_count')}, 0) * {self.W_COMMENT} + COALESCE({self._cast_int('shared_count')}, 0) * {self.W_SHARE})",
            'xhs_note':       f"(COALESCE({self._cast_int('liked_count')}, 0) * {self.W_LIKE} + COALESCE({self._cast_int('comment_count')}, 0) * {self.W_COMMENT} + COALESCE({self._cast_int('share_count')}, 0) * {self.W_SHARE} + COALESCE({self._cast_int('collected_count')}, 0) * {self.W_SHARE})",
            'kuaishou_video': f"(COALESCE({self._cast_float('viewd_count')}, 0) * {self.W_VIEW})",
            'zhihu_content':  f"(COALESCE({self._cast_int('voteup_count')}, 0) * {self.W_LIKE} + COALESCE({self._cast_int('comment_count')}, 0) * {self.W_COMMENT})",
        }

        all_queries, params = [], {}
        for idx, (table, formula) in enumerate(hotness_formulas.items()):
            pname = f"tf_{idx}"
            if table == 'weibo_note':
                time_filter_sql = f"{self._q('create_date_time')} >= :{pname}"
                time_filter_param = start_time.strftime('%Y-%m-%d %H:%M:%S')
            elif table in ['kuaishou_video', 'xhs_note', 'douyin_aweme']:
                time_col = 'time' if table == 'xhs_note' else 'create_time'
                time_filter_sql = f"{self._q(time_col)} >= :{pname}"
                time_filter_param = int(start_time.timestamp() * 1000)
            elif table == 'zhihu_content':
                time_filter_sql = f"{self._cast_int('created_time')} >= :{pname}"
                time_filter_param = int(start_time.timestamp())
            else:
                time_filter_sql = f"{self._q('create_time')} >= :{pname}"
                time_filter_param = int(start_time.timestamp())

            content_type = 'note' if table in ['weibo_note', 'xhs_note'] else 'content' if table == 'zhihu_content' else 'video'
            # UNION 要求各分支列类型一致：ts 在 weibo/zhihu 是 varchar，
            # 在其它平台是 bigint；hotness_score 里既有整数也有浮点。
            # 统一 ::text / ::double precision 后再 UNION。
            # 额外把各表的互动计数字段一起 SELECT 出来，否则
            # _extract_engagement 取不到 likes/comments，UI 上互动数全是空的。
            query_template = ("SELECT '{platform}' as p, '{type}' as t, {title} as title, "
                              "{author} as author, {url} as url, ({ts})::text as ts, "
                              "({formula})::double precision as hotness_score, "
                              "source_keyword, '{tbl}' as tbl, {extra_cols} "
                              "FROM {tbl} WHERE {time_filter}")

            field_subs = {'platform': table.split('_')[0], 'type': content_type, 'title': self._q('title'), 'author': self._q('nickname'), 'url': self._q('video_url'), 'ts': self._q('create_time'), 'formula': formula, 'tbl': self._q(table), 'time_filter': time_filter_sql, 'extra_cols': 'NULL::text as liked_count, NULL::text as comments_count, NULL::text as shared_count, NULL::text as sentiment_label, NULL::text as sentiment_confidence'}
            if table == 'weibo_note': field_subs.update({'title': self._q('content'), 'url': self._q('note_url'), 'ts': self._q('create_date_time'), 'extra_cols': f"({self._q('liked_count')})::text as liked_count, ({self._q('comments_count')})::text as comments_count, ({self._q('shared_count')})::text as shared_count, {self._q('sentiment_label')}::text as sentiment_label, {self._q('sentiment_confidence')}::text as sentiment_confidence"})
            elif table == 'xhs_note': field_subs.update({'ts': self._q('time'), 'url': self._q('note_url'), 'extra_cols': f"({self._q('liked_count')})::text as liked_count, ({self._q('comment_count')})::text as comments_count, ({self._q('share_count')})::text as shared_count, NULL::text as sentiment_label, NULL::text as sentiment_confidence"})
            elif table == 'zhihu_content': field_subs.update({'author': self._q('user_nickname'), 'url': self._q('content_url'), 'ts': self._q('created_time'), 'extra_cols': f"({self._q('voteup_count')})::text as liked_count, ({self._q('comment_count')})::text as comments_count, NULL::text as shared_count, NULL::text as sentiment_label, NULL::text as sentiment_confidence"})
            elif table == 'douyin_aweme': field_subs.update({'url': self._q('aweme_url'), 'extra_cols': f"({self._q('liked_count')})::text as liked_count, ({self._q('comment_count')})::text as comments_count, ({self._q('share_count')})::text as shared_count, NULL::text as sentiment_label, NULL::text as sentiment_confidence"})
            elif table == 'bilibili_video': field_subs.update({'extra_cols': f"({self._q('liked_count')})::text as liked_count, ({self._q('video_comment')})::text as comments_count, ({self._q('video_share_count')})::text as shared_count, NULL::text as sentiment_label, NULL::text as sentiment_confidence"})
            elif table == 'kuaishou_video': field_subs.update({'extra_cols': f"({self._q('liked_count')})::text as liked_count, NULL::text as comments_count, NULL::text as shared_count, NULL::text as sentiment_label, NULL::text as sentiment_confidence"})

            all_queries.append(query_template.format(**field_subs))
            params[pname] = time_filter_param

        params['hot_limit'] = int(limit)
        # PG 不允许 UNION 后面直接跟 ORDER BY，要包一层子查询
        if self._is_postgres:
            final_query = f"SELECT * FROM ({' UNION ALL '.join(all_queries)}) AS hot_content ORDER BY hotness_score DESC LIMIT :hot_limit"
        else:
            final_query = f"({' ) UNION ALL ( '.join(all_queries)}) ORDER BY hotness_score DESC LIMIT :hot_limit"
        raw_results = self._execute_query(final_query, params)

        formatted_results = [QueryResult(platform=r['p'], content_type=r['t'], title_or_content=r['title'], author_nickname=r.get('author'), url=r['url'], publish_time=self._to_datetime(r['ts']), engagement=self._extract_engagement(r), hotness_score=r.get('hotness_score', 0.0), source_keyword=r.get('source_keyword'), source_table=r['tbl'], sentiment_label=r.get('sentiment_label'), sentiment_confidence=float(r['sentiment_confidence']) if r.get('sentiment_confidence') else None) for r in raw_results]
        return DBResponse("search_hot_content", params_for_log, results=formatted_results, results_count=len(formatted_results))    

    def _wrap_query_field_with_dialect(self, field: str) -> str:
        """根据数据库方言包装SQL查询"""
        if settings.DB_DIALECT == 'postgresql':
            return f'"{field}"'
        return f'`{field}`'

    # ---------- PostgreSQL 方言适配 ----------
    # 下面这些工具最初是按 MySQL 方言写的（反引号标识符、%s 占位符、
    # CAST(x AS UNSIGNED)、SHOW COLUMNS、UNION ALL ... ORDER BY）。
    # BettaFish 已经切到 PostgreSQL，这些写法全部会报语法错误。
    # 这里的做法：统一改写成两种方言都认的写法 —— 标识符用标准双引号，
    # 数值转换用 CAST(x AS BIGINT)，占位符统一用命名参数 (:name)，
    # 列信息用 information_schema 查询。
    @property
    def _is_postgres(self) -> bool:
        return (settings.DB_DIALECT or "mysql").lower() in ("postgresql", "postgres")

    def _q(self, field: str) -> str:
        """标识符引用（两种方言通用）"""
        return f'"{field}"'

    def _cast_int(self, col: str) -> str:
        """把计数字段转成整数（两种方言通用）

        MediaCrawler 把点赞/评论数存成 Text，实际值可能是 ''、'1.2万'、
        '赞' 这类脏数据；但部分表（bilibili_video.liked_count、
        zhihu_content.voteup_count）的列类型本身是 integer。
        SUBSTRING 只接受 text，所以先 ::text 统一，再正则抽出开头的数字
        部分，最后转成整数。
        """
        if self._is_postgres:
            return (
                f"CAST(NULLIF(SUBSTRING(({self._q(col)})::text FROM '^[-0-9]*'), '') AS BIGINT)"
            )
        return f"CAST({self._q(col)} AS UNSIGNED)"

    def _cast_float(self, col: str) -> str:
        if self._is_postgres:
            return (
                f"CAST(NULLIF(SUBSTRING(({self._q(col)})::text FROM '^[-0-9]*[.]?[0-9]*'), '') AS DOUBLE PRECISION)"
            )
        return f"CAST({self._q(col)} AS DECIMAL(20,2))"

    def _limit_clause(self, limit: int, offset: int = 0) -> str:
        """LIMIT 写法（两种方言通用）"""
        if self._is_postgres:
            return f"LIMIT {int(limit)} OFFSET {int(offset)}"
        return f"LIMIT {int(limit)} OFFSET {int(offset)}"

    def _get_table_columns(self, table_name: str) -> List[str]:
        """读取表列名（MySQL 用 SHOW COLUMNS，PG 用 information_schema）"""
        if table_name in self._table_columns_cache: return self._table_columns_cache[table_name]
        if self._is_postgres:
            query = (
                "SELECT column_name FROM information_schema.columns "
                f"WHERE table_schema = 'public' AND table_name = :tname "
                "ORDER BY ordinal_position"
            )
            results = self._execute_query(query, {"tname": table_name})
            columns = [r["column_name"] for r in results] if results else []
        else:
            results = self._execute_query(f"SHOW COLUMNS FROM `{table_name}`")
            columns = [row['Field'] for row in results] if results else []
        self._table_columns_cache[table_name] = columns
        return columns

    def search_topic_globally(self, topic: str, limit_per_table: int = 100) -> DBResponse:
        """
        【工具】全局话题搜索: 在数据库中（内容、评论、标签、来源关键字）全面搜索指定话题。

        Args:
            topic (str): 要搜索的话题关键词。
            limit_per_table (int): 从每个相关表中返回的最大记录数，默认为 100。

        Returns:
            DBResponse: 包含所有匹配结果的聚合列表。
        """
        params_for_log = {'topic': topic, 'limit_per_table': limit_per_table}
        logger.info(f"--- TOOL: 全局话题搜索 (params: {params_for_log}) ---")
        
        search_term, all_results = f"%{topic}%", []
        search_configs = { 'bilibili_video': {'fields': ['title', 'desc', 'source_keyword'], 'type': 'video'}, 'bilibili_video_comment': {'fields': ['content'], 'type': 'comment'}, 'douyin_aweme': {'fields': ['title', 'desc', 'source_keyword'], 'type': 'video'}, 'douyin_aweme_comment': {'fields': ['content'], 'type': 'comment'}, 'kuaishou_video': {'fields': ['title', 'desc', 'source_keyword'], 'type': 'video'}, 'kuaishou_video_comment': {'fields': ['content'], 'type': 'comment'}, 'weibo_note': {'fields': ['content', 'source_keyword'], 'type': 'note'}, 'weibo_note_comment': {'fields': ['content'], 'type': 'comment'}, 'xhs_note': {'fields': ['title', 'desc', 'tag_list', 'source_keyword'], 'type': 'note'}, 'xhs_note_comment': {'fields': ['content'], 'type': 'comment'}, 'zhihu_content': {'fields': ['title', 'desc', 'content_text', 'source_keyword'], 'type': 'content'}, 'zhihu_comment': {'fields': ['content'], 'type': 'comment'}, 'tieba_note': {'fields': ['title', 'desc', 'source_keyword'], 'type': 'note'}, 'tieba_comment': {'fields': ['content'], 'type': 'comment'}, 'daily_news': {'fields': ['title'], 'type': 'news'}, }
        
        for table, config in search_configs.items():
            param_dict = {}
            where_clauses = []
            for idx, field in enumerate(config['fields']):
                pname = f"term_{idx}"
                where_clauses.append(f'{self._wrap_query_field_with_dialect(field)} LIKE :{pname}')
                param_dict[pname] = search_term
            param_dict['limit'] = limit_per_table
            where_clause = " OR ".join(where_clauses)
            query = f'SELECT * FROM {self._wrap_query_field_with_dialect(table)} WHERE {where_clause} ORDER BY id DESC LIMIT :limit'
            raw_results = self._execute_query(query, param_dict)
            for row in raw_results:
                _slabel, _sconf = self._extract_sentiment(row)
                content = (row.get('title') or row.get('content') or row.get('desc') or row.get('content_text', ''))
                time_key = row.get('create_time') or row.get('time') or row.get('created_time') or row.get('publish_time') or row.get('crawl_date')
                all_results.append(QueryResult(
                    platform=table.split('_')[0], content_type=config['type'],
                    title_or_content=content if content else '',
                    author_nickname=row.get('nickname') or row.get('user_nickname') or row.get('user_name'),
                    url=row.get('video_url') or row.get('note_url') or row.get('content_url') or row.get('url') or row.get('aweme_url'),
                    publish_time=self._to_datetime(time_key),
                    engagement=self._extract_engagement(row),
                    source_keyword=row.get('source_keyword'),
                    source_table=table,
                    sentiment_label=_slabel,
                    sentiment_confidence=_sconf
                ))
        return DBResponse("search_topic_globally", params_for_log, results=all_results, results_count=len(all_results))

    def search_topic_by_date(self, topic: str, start_date: str, end_date: str, limit_per_table: int = 100) -> DBResponse:
        """
        【工具】按日期搜索话题: 在明确的历史时间段内，搜索与特定话题相关的内容。

        Args:
            topic (str): 要搜索的话题关键词。
            start_date (str): 开始日期，格式 'YYYY-MM-DD'。
            end_date (str): 结束日期，格式 'YYYY-MM-DD'。
            limit_per_table (int): 从每个相关表中返回的最大记录数，默认为 100。

        Returns:
            DBResponse: 包含在指定日期范围内找到的结果的聚合列表。
        """
        params_for_log = {'topic': topic, 'start_date': start_date, 'end_date': end_date, 'limit_per_table': limit_per_table}
        logger.info(f"--- TOOL: 按日期搜索话题 (params: {params_for_log}) ---")
        
        try:
            start_dt, end_dt = datetime.strptime(start_date, '%Y-%m-%d'), datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
        except ValueError:
            return DBResponse("search_topic_by_date", params_for_log, error_message="日期格式错误，请使用 'YYYY-MM-DD' 格式。")
        
        search_term, all_results = f"%{topic}%", []
        search_configs = {
            'bilibili_video': {'fields': ['title', 'desc', 'source_keyword'], 'type': 'video', 'time_col': 'create_time', 'time_type': 'sec'}, 'douyin_aweme': {'fields': ['title', 'desc', 'source_keyword'], 'type': 'video', 'time_col': 'create_time', 'time_type': 'ms'},
            'kuaishou_video': {'fields': ['title', 'desc', 'source_keyword'], 'type': 'video', 'time_col': 'create_time', 'time_type': 'ms'}, 'weibo_note': {'fields': ['content', 'source_keyword'], 'type': 'note', 'time_col': 'create_date_time', 'time_type': 'str'},
            'xhs_note': {'fields': ['title', 'desc', 'tag_list', 'source_keyword'], 'type': 'note', 'time_col': 'time', 'time_type': 'ms'}, 'zhihu_content': {'fields': ['title', 'desc', 'content_text', 'source_keyword'], 'type': 'content', 'time_col': 'created_time', 'time_type': 'sec_str'},
            'tieba_note': {'fields': ['title', 'desc', 'source_keyword'], 'type': 'note', 'time_col': 'publish_time', 'time_type': 'str'}, 'daily_news': {'fields': ['title'], 'type': 'news', 'time_col': 'crawl_date', 'time_type': 'date_str'},
        }

        for table, config in search_configs.items():
            param_dict = {}
            where_clauses = []
            for idx, field in enumerate(config['fields']):
                pname = f"term_{idx}"
                where_clauses.append(f'{self._wrap_query_field_with_dialect(field)} LIKE :{pname}')
                param_dict[pname] = search_term
            param_dict['limit'] = limit_per_table
            where_clause = ' OR '.join(where_clauses)
            query = f'SELECT * FROM {self._wrap_query_field_with_dialect(table)} WHERE {where_clause} ORDER BY id DESC LIMIT :limit'
            raw_results = self._execute_query(query, param_dict)
            for row in raw_results:
                _slabel, _sconf = self._extract_sentiment(row)
                content = (row.get('title') or row.get('content') or row.get('desc') or row.get('content_text', ''))
                time_key = row.get('create_time') or row.get('time') or row.get('created_time') or row.get('publish_time') or row.get('crawl_date')
                all_results.append(QueryResult(
                    platform=table.split('_')[0], content_type=config['type'],
                    title_or_content=content if content else '',
                    author_nickname=row.get('nickname') or row.get('user_nickname') or row.get('user_name'),
                    url=row.get('video_url') or row.get('note_url') or row.get('content_url') or row.get('url') or row.get('aweme_url'),
                    publish_time=self._to_datetime(time_key),
                    engagement=self._extract_engagement(row),
                    source_keyword=row.get('source_keyword'),
                    source_table=table,
                    sentiment_label=_slabel,
                    sentiment_confidence=_sconf
                ))
        return DBResponse("search_topic_by_date", params_for_log, results=all_results, results_count=len(all_results))
        
    def get_comments_for_topic(self, topic: str, limit: int = 500) -> DBResponse:
        """
        【工具】获取话题评论: 专门搜索并返回所有平台中与特定话题相关的公众评论数据。

        Args:
            topic (str): 要搜索的话题关键词。
            limit (int): 返回评论的总数量上限，默认为 500。

        Returns:
            DBResponse: 包含匹配的评论列表。
        """
        params_for_log = {'topic': topic, 'limit': limit}
        logger.info(f"--- TOOL: 获取话题评论 (params: {params_for_log}) ---")
        
        search_term = f"%{topic}%"
        comment_tables = ['bilibili_video_comment', 'douyin_aweme_comment', 'kuaishou_video_comment', 'weibo_note_comment', 'xhs_note_comment', 'zhihu_comment', 'tieba_comment']

        all_queries, params = [], {}
        for idx, table in enumerate(comment_tables):
            cols = self._get_table_columns(table)
            author_col = 'user_nickname' if 'user_nickname' in cols else 'nickname'
            like_col = 'comment_like_count' if 'comment_like_count' in cols else 'like_count' if 'like_count' in cols else None
            time_col = 'publish_time' if 'publish_time' in cols else 'create_date_time' if 'create_date_time' in cols else 'create_time'
            # UNION 要求所有分支的列类型一致，否则 PG 报
            # "UNION types text and integer cannot be matched"。
            # likes 各表有 text / integer 两种，统一 CAST 成 bigint；
            # ts 各表有 bigint / varchar 两种，统一 CAST 成 text。
            if like_col:
                like_select = f"{self._cast_int(like_col)} as likes"
            else:
                like_select = "0 as likes"
            ts_select = f"({self._q(time_col)})::text as ts"

            pname = f"cm_{idx}"
            query = (f"SELECT '{table.split('_')[0]}' as platform, {self._q('content')}, "
                     f"{self._q(author_col)} as author, {ts_select}, {like_select}, "
                     f"'{table}' as source_table "
                     f"FROM {self._q(table)} WHERE {self._q('content')} LIKE :{pname}")
            all_queries.append(query)
            params[pname] = search_term

        params['cm_limit'] = int(limit)
        if self._is_postgres:
            final_query = f"SELECT * FROM ({' UNION ALL '.join(all_queries)}) AS comments ORDER BY ts DESC LIMIT :cm_limit"
        else:
            final_query = f"({' ) UNION ALL ( '.join(all_queries)}) ORDER BY ts DESC LIMIT :cm_limit"
        raw_results = self._execute_query(final_query, params)
        
        formatted = [QueryResult(platform=r['platform'], content_type='comment', title_or_content=r['content'], author_nickname=r['author'], publish_time=self._to_datetime(r['ts']), engagement={'likes': int(r['likes']) if str(r['likes']).isdigit() else 0}, source_table=r['source_table']) for r in raw_results]
        return DBResponse("get_comments_for_topic", params_for_log, results=formatted, results_count=len(formatted))

    def search_topic_on_platform(
        self,
        platform: Literal['bilibili', 'weibo', 'douyin', 'kuaishou', 'xhs', 'zhihu', 'tieba'],
        topic: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 20
    ) -> DBResponse:
        """
        【工具】平台定向搜索: (新增) 在指定的单个社交媒体平台上搜索特定话题。

        Args:
            platform (Literal['bilibili', ...]): 要搜索的平台，必须是七个支持的平台之一。
            topic (str): 要搜索的话题关键词。
            start_date (Optional[str]): 开始日期，格式 'YYYY-MM-DD'。默认为None。
            end_date (Optional[str]): 结束日期，格式 'YYYY-MM-DD'。默认为None。
            limit (int): 返回结果的最大数量，默认为 20。

        Returns:
            DBResponse: 包含在该平台找到的结果列表。
        """
        params_for_log = {'platform': platform, 'topic': topic, 'start_date': start_date, 'end_date': end_date, 'limit': limit}
        logger.info(f"--- TOOL: 平台定向搜索 (params: {params_for_log}) ---")

        all_configs = { 'bilibili': [{'table': 'bilibili_video', 'fields': ['title', 'desc', 'source_keyword'], 'type': 'video', 'time_col': 'create_time', 'time_type': 'sec'}, {'table': 'bilibili_video_comment', 'fields': ['content'], 'type': 'comment'}], 'douyin': [{'table': 'douyin_aweme', 'fields': ['title', 'desc', 'source_keyword'], 'type': 'video', 'time_col': 'create_time', 'time_type': 'ms'}, {'table': 'douyin_aweme_comment', 'fields': ['content'], 'type': 'comment'}], 'kuaishou': [{'table': 'kuaishou_video', 'fields': ['title', 'desc', 'source_keyword'], 'type': 'video', 'time_col': 'create_time', 'time_type': 'ms'}, {'table': 'kuaishou_video_comment', 'fields': ['content'], 'type': 'comment'}], 'weibo': [{'table': 'weibo_note', 'fields': ['content', 'source_keyword'], 'type': 'note', 'time_col': 'create_date_time', 'time_type': 'str'}, {'table': 'weibo_note_comment', 'fields': ['content'], 'type': 'comment'}], 'xhs': [{'table': 'xhs_note', 'fields': ['title', 'desc', 'tag_list', 'source_keyword'], 'type': 'note', 'time_col': 'time', 'time_type': 'ms'}, {'table': 'xhs_note_comment', 'fields': ['content'], 'type': 'comment'}], 'zhihu': [{'table': 'zhihu_content', 'fields': ['title', 'desc', 'content_text', 'source_keyword'], 'type': 'content', 'time_col': 'created_time', 'time_type': 'sec_str'}, {'table': 'zhihu_comment', 'fields': ['content'], 'type': 'comment'}], 'tieba': [{'table': 'tieba_note', 'fields': ['title', 'desc', 'source_keyword'], 'type': 'note', 'time_col': 'publish_time', 'time_type': 'str'}, {'table': 'tieba_comment', 'fields': ['content'], 'type': 'comment'}] }
        
        if platform not in all_configs:
            return DBResponse("search_topic_on_platform", params_for_log, error_message=f"不支持的平台: {platform}")

        search_term, all_results = f"%{topic}%", []
        platform_configs = all_configs[platform]

        time_clause, time_params_tuple = "", ()
        if start_date and end_date:
            try:
                start_dt, end_dt = datetime.strptime(start_date, '%Y-%m-%d'), datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
            except ValueError:
                return DBResponse("search_topic_on_platform", params_for_log, error_message="日期格式错误，请使用 'YYYY-MM-DD' 格式。")
        else:
            start_dt, end_dt = None, None

        for idx, config in enumerate(platform_configs):
            table = config['table']
            pname = f"tp_{idx}"
            topic_clause = " OR ".join([f"{self._q(field)} LIKE :{pname}" for field in config['fields']])
            query = f"SELECT * FROM {self._q(table)} WHERE {topic_clause}"
            params: Dict[str, Any] = {pname: search_term}

            if start_dt and end_dt and 'time_col' in config:
                time_col, time_type = config['time_col'], config['time_type']
                tstart, tend = f"t_{idx}_s", f"t_{idx}_e"
                if time_type == 'sec':
                    params[tstart], params[tend] = int(start_dt.timestamp()), int(end_dt.timestamp())
                elif time_type == 'ms':
                    params[tstart], params[tend] = int(start_dt.timestamp() * 1000), int(end_dt.timestamp() * 1000)
                elif time_type in ['str', 'date_str']:
                    params[tstart], params[tend] = start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d')
                else:
                    params[tstart], params[tend] = str(int(start_dt.timestamp())), str(int(end_dt.timestamp()))

                if table == 'zhihu_content':
                    t_clause = f"{self._cast_int(time_col)} >= :{tstart} AND {self._cast_int(time_col)} < :{tend}"
                else:
                    t_clause = f"{self._q(time_col)} >= :{tstart} AND {self._q(time_col)} < :{tend}"

                query += f" AND ({t_clause})"

            query += f" ORDER BY id DESC LIMIT :tp_limit_{idx}"
            params[f"tp_limit_{idx}"] = int(limit)

            raw_results = self._execute_query(query, params)
            for row in raw_results:
                _slabel, _sconf = self._extract_sentiment(row)
                content = (row.get('title') or row.get('content') or row.get('desc') or row.get('content_text', ''))
                time_key = config.get('time_col') and row.get(config.get('time_col'))
                all_results.append(QueryResult(platform=platform, content_type=config['type'], title_or_content=content if content else '', author_nickname=row.get('nickname') or row.get('user_nickname'), url=row.get('video_url') or row.get('note_url') or row.get('content_url') or row.get('url') or row.get('aweme_url'), publish_time=self._to_datetime(time_key), engagement=self._extract_engagement(row), source_keyword=row.get('source_keyword'), source_table=table, sentiment_label=_slabel, sentiment_confidence=_sconf))
        
        return DBResponse("search_topic_on_platform", params_for_log, results=all_results, results_count=len(all_results))

# --- 3. 测试与使用示例 ---
def print_response_summary(response: DBResponse):
    """简化的打印函数，用于展示测试结果"""
    if response.error_message:
        logger.info(f"工具 '{response.tool_name}' 执行出错: {response.error_message}")
        return

    params_str = ", ".join(f"{k}='{v}'" for k, v in response.parameters.items())
    logger.info(f"查询: 工具='{response.tool_name}', 参数=[{params_str}]")
    logger.info(f"找到 {response.results_count} 条相关记录。")
    
    # 统一为一个消息输出
    output_lines = []
    output_lines.append("==== 查询结果预览（最多前5条） ====")
    if response.results and len(response.results) > 0:
        for idx, res in enumerate(response.results[:5], 1):
            content_preview = (res.title_or_content.replace('\n', ' ')[:70] + '...') if res.title_or_content and len(res.title_or_content) > 70 else (res.title_or_content or '')
            author_str = res.author_nickname or "N/A"
            publish_time_str = res.publish_time.strftime('%Y-%m-%d %H:%M') if res.publish_time else "N/A"
            hotness_str = f", hotness: {res.hotness_score:.2f}" if getattr(res, "hotness_score", 0) > 0 else ""
            engagement_dict = getattr(res, "engagement", {}) or {}
            engagement_str = ", ".join(f"{k}: {v}" for k, v in engagement_dict.items() if v)
            output_lines.append(
                f"{idx}. [{res.platform.upper()}/{res.content_type}] {content_preview}\n"
                f"   作者: {author_str} | 时间: {publish_time_str}"
                f"{hotness_str} | 源关键词: '{res.source_keyword or 'N/A'}'\n"
                f"   链接: {res.url or 'N/A'}\n"
                f"   互动数据: {{{engagement_str}}}"
            )
    else:
        output_lines.append("暂无相关内容。")
    output_lines.append("=" * 60)
    logger.info('\n'.join(output_lines))

if __name__ == "__main__":
    
    try:
        db_agent_tools = MediaCrawlerDB()
        logger.info("数据库工具初始化成功，开始执行测试场景...\n")
        
        # 场景1: (新) 查找过去一周综合热度最高的内容 (不再需要sort_by)
        response1 = db_agent_tools.search_hot_content(time_period='week', limit=5)
        print_response_summary(response1)

        # 场景2: 查找过去24小时内综合热度最高的内容
        response2 = db_agent_tools.search_hot_content(time_period='24h', limit=5)
        print_response_summary(response2)

        # 场景3: 全局搜索"罗永浩"
        response3 = db_agent_tools.search_topic_globally(topic="罗永浩", limit_per_table=2)
        print_response_summary(response3)

        # 场景4: (新增) 在B站上精确搜索"论文"
        response4 = db_agent_tools.search_topic_on_platform(platform='bilibili', topic="论文", limit=5)
        print_response_summary(response4)

        # 场景5: (新增) 在微博上精确搜索 "许凯" 在特定一天内的内容
        response5 = db_agent_tools.search_topic_on_platform(platform='weibo', topic="许凯", start_date='2025-08-22', end_date='2025-08-22', limit=5)
        print_response_summary(response5)

    except ValueError as e:
        logger.exception(f"初始化失败: {e}")
        logger.exception("请确保相关的数据库环境变量已正确设置, 或在代码中直接提供连接信息。")
    except Exception as e:
        logger.exception(f"测试过程中发生未知错误: {e}")