# ============================================================
# 后端多阶段构建：Python 3.11 + uv
# ============================================================

# --- Stage 1: Builder ---
FROM python:3.11-slim AS builder

# 安装 uv
RUN pip install --no-cache-dir uv

WORKDIR /app

# 先复制依赖声明，利用 Docker 层缓存
COPY pyproject.toml uv.lock .python-version ./

# 安装第三方依赖（不安装项目本身）
RUN uv sync --frozen --no-dev --no-install-project

# 复制源码和迁移脚本
COPY src/ src/
COPY main.py alembic.ini README.md ./
COPY alembic/ alembic/

# 安装项目本身（hatchling 构建 wheel）
RUN uv sync --frozen --no-dev

# --- Stage 2: Runtime ---
FROM python:3.11-slim AS runtime

# 系统依赖：curl 用于健康检查
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# 非 root 用户
RUN groupadd -r app && useradd -r -g app app

WORKDIR /app

# 从 builder 复制虚拟环境和源码
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/main.py /app/
COPY --from=builder /app/alembic.ini /app/
COPY --from=builder /app/alembic /app/alembic

# 将虚拟环境 bin 加入 PATH
ENV PATH="/app/.venv/bin:$PATH"

# 切换到非 root 用户
USER app

EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# 启动：先执行数据库迁移，再启动应用
CMD ["sh", "-c", "alembic upgrade head && python main.py"]
