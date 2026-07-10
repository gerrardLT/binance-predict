# Binance Prediction 情绪 Agent

BTC 预测市场的情绪曲线自进化系统。核心自动链路是 Agent 的 Learn → Predict → Validate → Evolve 闭环；概率动量仅保留为手动分析接口。

## 运行架构

```text
Binance Prediction Markets（报价）
    -> 15 秒采样 -> 5 分钟情绪窗口
    -> AgentScheduler（单 worker 优先级队列）
    -> Learn / Predict / Validate / Evolve
    -> PatternMemory / AgentPrediction

Binance Spot bookTicker
    -> mid_price
    -> 情绪窗口的 entry_price / exit_price
```

- 路径 B：情绪曲线 Agent Loop，是唯一自动决策引擎。
- 路径 C：`POST /api/sentiment/momentum-predict`，纯算法、手动触发、不落库。
- `POST /api/sentiment/backtest` 与 `POST /api/sentiment/predict` 已退役，仅返回提示以保持兼容。

## 交易安全

`AGENT_AUTO_TRADE` 默认是 `false`。即使 Agent 返回 `UP` 或 `DOWN` 且置信度超过阈值，也不会下单；只有显式设置为 `true` 才允许继续经过方向与置信度门控。

实盘前应确认 API 权限、钱包配置、单笔金额和风险承受范围。该开关不是仓位或亏损保护机制。

## 快速开始

```powershell
uv sync
Copy-Item .env.example .env
# 编辑 .env，填入数据库和 API 配置；保持 AGENT_AUTO_TRADE=false
uv run alembic upgrade head
uv run python main.py
```

前端开发：

```powershell
Set-Location frontend
npm install
npm run dev
```

## 数据库迁移

应用启动不会自动运行 Alembic 迁移。部署前先备份数据库，再人工执行：

```powershell
uv run alembic current
uv run alembic upgrade head
```

`b2c3d4e5f6a7_drop_path_a_tables.py` 会永久删除路径 A 的 7 张遗留表：`predictions`、`prediction_results`、`feature_snapshots`、`custom_rules`、`rule_versions`、`prompt_versions`、`review_memories`。没有确认备份前不要执行第二条命令。

## API

| 端点 | 说明 |
| --- | --- |
| `GET /api/health` | 服务与中间价状态 |
| `GET /api/trades/latest` | 最近订单 |
| `GET /api/prediction-markets` | 活跃预测市场 |
| `GET /api/prediction-wallet` | 预测钱包信息 |
| `GET /api/chart/prediction-market` | 当前市场报价曲线 |
| `GET /api/sentiment/windows` | 已归档情绪窗口 |
| `POST /api/sentiment/momentum-predict` | 手动概率动量分析 |
| `GET /api/sentiment/agent/predictions` | Agent 预测历史 |
| `GET /api/sentiment/agent/patterns` | 模式库 |
| `GET /api/sentiment/agent/patterns/{id}/history` | 模式变更历史 |
| `GET /api/sentiment/agent/status` | Agent 状态 |

## 项目结构

```text
src/binance_predict/
├── config/       # Pydantic Settings
├── db/           # SQLAlchemy async ORM
├── models/       # Agent Loop Pydantic 契约
├── prompts/      # Learn / Predict / Evolve 提示词
├── services/     # Agent、交易、市场报价和中间价采集
└── main.py       # FastAPI、生命周期和 API
```

## 验证

```powershell
uv run pytest -q
Set-Location frontend
npm run build
```
