"""
通用数据库工具（异步）

此模块提供基于 SQLAlchemy 2.x 异步引擎的数据库访问封装，支持 MySQL 与 PostgreSQL。
数据模型定义位置：
- 无（本模块仅提供连接与查询工具，不定义数据模型）
"""

from __future__ import annotations
import asyncio
import os
from typing import Any, Dict, Iterable, List, Optional, Union

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.engine import URL, make_url
from sqlalchemy import text
from InsightEngine.utils.config import settings

__all__ = [
    "get_async_engine",
    "fetch_all",
]


_engine: Optional[AsyncEngine] = None
# engine 必须按 event loop 分开缓存。
# SQLAlchemy 的 async engine 会把连接池绑定到「创建它的那个 loop」，
# 跨 loop 复用同一个 engine 时 asyncpg 会抛:
#   RuntimeError: ... got Future ... attached to a different loop
# 原实现是全局单例，而调用方（InsightEngine/tools/search.py 的 _execute_query）
# 每次都 asyncio.get_event_loop()，可能拿到不同的 loop，于是踩坑。
# 切到 PostgreSQL(asyncpg) 后这个问题才暴露 —— MySQL 的 aiomysql 不这么严格。
_engines_by_loop: Dict[Any, AsyncEngine] = {}


def _parse_database_url(database_url: str) -> URL:
    """Parse an explicit URL and reject clearly unescaped userinfo separators."""
    try:
        scheme_separator = database_url.find("://")
        if scheme_separator < 0:
            raise ValueError("missing URL scheme separator")

        authority = database_url[scheme_separator + 3 :].split("/", 1)[0]
        if authority.count("@") > 1:
            raise ValueError("userinfo contains an unescaped '@' character")

        return make_url(database_url)
    except Exception as exc:
        raise ValueError(
            "环境变量 DATABASE_URL 不是合法的数据库连接 URL。"
            "如果用户名或密码包含特殊字符，请对 userinfo 部分进行 percent-encode（例如 '@'→'%40'，':'→'%3A'）。"
            "或者不要使用 DATABASE_URL，改用 DB_USER/DB_PASSWORD/DB_HOST/DB_PORT/DB_NAME 等分字段配置。"
        ) from exc


def _build_database_url() -> str:
    dialect: str = (settings.DB_DIALECT or "mysql").lower()
    host: str = settings.DB_HOST or ""
    port_int: int = settings.DB_PORT
    user: str = settings.DB_USER or ""
    password: str = settings.DB_PASSWORD or ""
    db_name: str = settings.DB_NAME or ""

    # 如果外部提供了完整的 DATABASE_URL，优先交给 SQLAlchemy 解析后再渲染。
    # 明显包含多个未转义 @ 的 userinfo 会被拒绝，避免静默解析为错误主机。
    database_url_env = os.getenv("DATABASE_URL")
    if database_url_env:
        url_obj = _parse_database_url(database_url_env)
        return url_obj.render_as_string(hide_password=False)

    # 使用 SQLAlchemy 的 URL.create 来安全构建连接 URL，避免手动拼接导致的转义/编码问题
    if dialect in ("postgresql", "postgres"):
        # PostgreSQL 使用 asyncpg 驱动
        url_obj = URL.create(
            drivername="postgresql+asyncpg",
            username=user or None,
            password=password or None,
            host=host or None,
            port=port_int,
            database=db_name or None,
        )
        try:
            return url_obj.render_as_string(hide_password=False)
        except Exception:
            return str(url_obj)

    # 默认 MySQL 使用 aiomysql 驱动
    url_obj = URL.create(
        drivername="mysql+aiomysql",
        username=user or None,
        password=password or None,
        host=host or None,
        port=port_int,
        database=db_name or None,
    )
    try:
        return url_obj.render_as_string(hide_password=False)
    except Exception:
        return str(url_obj)


def get_async_engine() -> AsyncEngine:
    """取得当前 event loop 专属的 engine（每个 loop 一个独立连接池）。"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # 不在 async 上下文中（理论上不会发生），退回全局单例
        global _engine
        if _engine is None:
            _engine = _make_engine()
        return _engine

    engine = _engines_by_loop.get(loop)
    if engine is None:
        engine = _make_engine()
        _engines_by_loop[loop] = engine
    return engine


def dispose_engine_for_loop(loop) -> None:
    """
    关闭并移除指定 event loop 的 engine（连接池清理）。

    场景：agent 在"当前 loop 已在运行"时会另开线程 + 新 loop 跑查询，
    每次都会创建一个新 engine。若不清理，连接池会随查询次数无限增长。
    """
    engine = _engines_by_loop.pop(loop, None)
    if engine is None:
        return
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        pass
    try:
        # 在该 loop 内同步执行 dispose
        loop.run_until_complete(engine.dispose())
    except Exception:
        # 尽力而为：清理失败不影响主流程
        try:
            engine.sync_engine.dispose()
        except Exception:
            pass


def _make_engine() -> AsyncEngine:
    database_url: str = _build_database_url()
    return create_async_engine(
        database_url,
        pool_pre_ping=True,
        pool_recycle=1800,
    )


async def fetch_all(query: str, params: Optional[Union[Iterable[Any], Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """
    执行只读查询并返回字典列表。
    """
    engine: AsyncEngine = get_async_engine()
    async with engine.connect() as conn:
        result = await conn.execute(text(query), params or {})
        rows = result.mappings().all()
        # 将 RowMapping 转换为普通字典
        return [dict(row) for row in rows]

