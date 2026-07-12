# ============================================================
# 前端多阶段构建：Node 22 → Nginx
# 构建上下文：项目根目录（-f docker/frontend.Dockerfile）
# ============================================================

# --- Stage 1: Builder ---
FROM node:22-alpine AS builder

WORKDIR /app

# 先复制依赖声明，利用 Docker 层缓存
COPY frontend/package.json frontend/package-lock.json ./

# 安装依赖（npm ci 保证可重复构建）
RUN npm ci

# 复制前端源码并构建
COPY frontend/ .
RUN npm run build

# --- Stage 2: Runtime (Nginx) ---
FROM nginx:1.27-alpine AS runtime

# 复制构建产物
COPY --from=builder /app/dist /usr/share/nginx/html

# 复制 Nginx 配置
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
