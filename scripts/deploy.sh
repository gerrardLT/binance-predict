#!/usr/bin/env bash
# ============================================================
# 宝塔面板 VPS 首次部署初始化脚本
# ============================================================
# 在宝塔服务器上执行：
#   1. 确认 Docker + Docker Compose 已安装（宝塔软件商店安装）
#   2. 创建项目目录
#   3. 登录 GHCR（拉取私有镜像）
#   4. 拉取并启动 DB + 后端容器
#   5. 输出宝塔 Nginx 反向代理配置指引
#
# 使用方式：
#   chmod +x scripts/deploy.sh
#   ./scripts/deploy.sh
# ============================================================

set -euo pipefail

PROJECT_DIR="/www/wwwroot/binance-predict"
COMPOSE_FILE="docker-compose.bt.yml"
GHCR_USER="${GHCR_USER:-gerrardLT}"
GHCR_TOKEN="${GHCR_TOKEN:-}"

echo "============================================"
echo " Binance Predict - 宝塔面板部署初始化"
echo "============================================"

# 1. 检查 Docker（宝塔用户通常在软件商店安装）
if ! command -v docker &> /dev/null; then
    echo "[1/5] Docker 未安装！"
    echo "  请在宝塔面板 → 软件商店 → 搜索 Docker 并安装"
    echo "  或执行: curl -fsSL https://get.docker.com | sh"
    read -rp "  安装完成后按回车继续..."
fi
echo "[1/5] Docker: $(docker --version)"

# 2. 确认 Docker Compose 插件
if ! docker compose version &> /dev/null; then
    echo "[2/5] 安装 Docker Compose 插件..."
    apt-get update && apt-get install -y docker-compose-plugin 2>/dev/null \
        || yum install -y docker-compose-plugin 2>/dev/null \
        || { echo "  请手动安装 Docker Compose 插件"; exit 1; }
else
    echo "[2/5] Docker Compose: $(docker compose version --short)"
fi

# 3. 创建项目目录
echo "[3/5] 创建项目目录: $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# 下载宝塔版 docker-compose（如果不存在）
if [ ! -f "$COMPOSE_FILE" ]; then
    echo "  下载 $COMPOSE_FILE..."
    curl -fsSL "https://raw.githubusercontent.com/gerrardLT/binance-predict/main/docker/$COMPOSE_FILE" \
        -o "$COMPOSE_FILE"
fi

# 4. 登录 GHCR
echo "[4/5] GHCR 登录"
if [ -z "$GHCR_TOKEN" ]; then
    echo "  请提供 GHCR Personal Access Token（用于拉取私有镜像）"
    echo "  在 GitHub Settings > Developer settings > Personal access tokens 创建"
    echo "  需要权限: read:packages"
    read -rp "  GHCR Token: " GHCR_TOKEN
fi
echo "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USER" --password-stdin
echo "  GHCR 登录成功"

# 5. 检查 .env 文件
echo "[5/5] 环境变量检查"
if [ ! -f .env ]; then
    echo ""
    echo "  ⚠  .env 文件不存在！"
    echo "  请创建 $PROJECT_DIR/.env 并填入所有生产环境变量。"
    echo "  可参考 .env.example："
    echo "    curl -fsSL https://raw.githubusercontent.com/gerrardLT/binance-predict/main/.env.example -o .env.example"
    echo ""
    echo "  必须配置的变量："
    echo "    DATABASE_URL, DEEPSEEK_API_KEY, DASHSCOPE_API_KEY"
    echo "    BINANCE_API_KEY, BINANCE_API_SECRET, API_AUTH_TOKEN"
    echo "    DB_USER, DB_PASSWORD, DB_NAME"
    echo ""
    read -rp "  创建 .env 后按回车继续..."
fi

# 拉取镜像并启动
echo ""
echo "============================================"
echo " 拉取镜像并启动服务（DB + 后端）"
echo "============================================"
docker compose -f "$COMPOSE_FILE" pull
docker compose -f "$COMPOSE_FILE" up -d

echo ""
echo "============================================"
echo " 部署完成！"
echo "============================================"
echo ""
docker compose -f "$COMPOSE_FILE" ps

# 健康检查
sleep 8
echo ""
echo "后端健康检查："
curl -sf http://localhost:8002/api/health && echo "" || echo "  WARN: 后端尚未就绪，请稍后重试 curl http://localhost:8002/api/health"

echo ""
echo "============================================"
echo " 宝塔 Nginx 反向代理配置指引"
echo "============================================"
echo ""
echo "  1. 宝塔面板 → 网站 → 添加站点"
echo "     域名: 你的域名（如 predict.example.com）"
echo "     根目录: /www/wwwroot/predict"
echo ""
echo "  2. 站点设置 → 配置文件，在 server {} 块内添加："
echo ""
echo '     location /api/ {'
echo '         proxy_pass http://127.0.0.1:8002;'
echo '         proxy_set_header Host $host;'
echo '         proxy_set_header X-Real-IP $remote_addr;'
echo '         proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;'
echo '         proxy_set_header X-Forwarded-Proto $scheme;'
echo '         proxy_buffering off;'
echo '         proxy_cache off;'
echo '         proxy_read_timeout 300s;'
echo '     }'
echo ""
echo "  3. 前端静态文件："
echo "     将 frontend/dist/ 目录内容上传到站点根目录"
echo ""
echo "常用命令："
echo "  查看日志:  docker compose -f $COMPOSE_FILE logs -f"
echo "  重启服务:  docker compose -f $COMPOSE_FILE restart"
echo "  停止服务:  docker compose -f $COMPOSE_FILE down"
