"""
Alembic 异步迁移环境配置

使用 asyncpg 驱动，从项目 settings 读取数据库 URL。
运行方式：
    alembic revision --autogenerate -m "initial"
    alembic upgrade head
"""

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# 导入项目 ORM 模型，确保 Base.metadata 包含所有表
from binance_predict.db.models import Base  # noqa: F401
from binance_predict.config.settings import settings

# Alembic Config 对象
config = context.config

# 从项目 settings 覆盖数据库 URL（async_engine_from_config 接受 postgresql+asyncpg:// 格式）
config.set_main_option("sqlalchemy.url", settings.database_url)

# Python logging 配置
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 目标 metadata（autogenerate 用）
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """离线模式：不需要数据库连接，生成 SQL 脚本"""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """在同步连接上运行迁移"""
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """异步在线模式：使用 asyncpg 引擎"""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """在线模式入口"""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
