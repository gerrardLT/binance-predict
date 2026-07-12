"""
BTC 5min LLM 预测系统 V2 - 异步数据库引擎

使用 SQLAlchemy 2.0 异步引擎 + asyncpg 驱动。
支持 PostgreSQL + TimescaleDB 扩展。
"""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ..config.settings import settings

# 创建异步引擎（echo=True 仅开发环境，生产应关闭）
# Fix #17: 扩大连接池。双 Worker(PREDICT+HEAVY) + tracker/archiver + API 请求
# 并发访问，原 3+5=8 上限易耗尽导致获取连接超时。
engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_size=10,
    max_overflow=10,
    pool_timeout=30,       # 获取连接最长等待 30s，避免无限阻塞
    pool_pre_ping=True,  # 连接池预检，避免使用已断开的连接
    pool_recycle=1800,   # 30 分钟回收连接，防止服务端断开空闲连接
)

# 异步会话工厂
async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncSession:
    """FastAPI 依赖注入：获取只读数据库会话。

    Fix #8: 本依赖仅用于只读查询端点，结束时回滚而非 commit，
    避免只读端点产生无意义的事务提交。写入操作由业务层
    （SentimentAgent 等）通过 async_session_factory 自行管理事务。
    """
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.rollback()
            await session.close()
