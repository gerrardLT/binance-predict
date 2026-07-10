# 运行安全与死代码清理 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 默认禁止 Agent 自动下单，并移除路径 A 遗留代码、依赖和配置，使测试与文档反映当前 Agent Loop 架构。

**Architecture:** 交易权限收敛到 `agent_logic.should_trade`，由 `Settings.agent_auto_trade` 统一控制。市场采集器只保留现货 `bookTicker` 产生的中间价；其余路径 A 数据结构及依赖被删除。数据库 DROP 迁移继续由部署人员手工执行。

**Tech Stack:** Python 3.11、FastAPI、Pydantic Settings、SQLAlchemy async、pytest、Hypothesis、React/Vite。

---

## 文件结构

- `src/binance_predict/services/agent_logic.py`：真实交易的纯函数门控。
- `src/binance_predict/config/settings.py`：总开关及遗留配置清理。
- `src/binance_predict/services/data_collector.py`：仅保留 `bookTicker` 的中间价采集器。
- `src/binance_predict/models/schemas.py`：仅保留 Agent Loop 数据契约及其依赖枚举。
- `src/binance_predict/main.py`：健康检查移除失真的 REST 状态字段。
- `pyproject.toml`、`.gitignore`、`.env.example`、`README.md`：依赖、密钥保护和人工迁移说明。
- `tests/test_agent_trade_gate.py`、`tests/test_data_collector_cleanup.py`：本次新增行为与清理回归测试。
- `tests/test_bug_condition_exploration.py`：删除路径 A 已退役回测的 3 个断言。

### Task 1: 交易总开关

**Files:**

- Create: `tests/test_agent_trade_gate.py`
- Modify: `src/binance_predict/services/agent_logic.py:135-164`
- Modify: `src/binance_predict/services/sentiment_agent.py:741-750`
- Modify: `src/binance_predict/db/models.py:39-45`
- Modify: `src/binance_predict/config/settings.py:102-105`
- Modify: `.env`
- Create: `.env.example`

- [ ] **Step 1: 写入失败测试**

```python
from binance_predict.services.agent_logic import should_trade


def test_disabled_auto_trade_blocks_high_confidence_direction() -> None:
    allowed, reason = should_trade(
        "UP", confidence=0.99, threshold=0.6, auto_trade_enabled=False
    )
    assert allowed is False
    assert "总开关" in reason


def test_enabled_auto_trade_keeps_existing_confidence_rule() -> None:
    allowed, _ = should_trade(
        "DOWN", confidence=0.61, threshold=0.6, auto_trade_enabled=True
    )
    assert allowed is True
```

- [ ] **Step 2: 确认测试因缺少新参数失败**

Run: `uv run pytest tests/test_agent_trade_gate.py -q`

Expected: `TypeError`，提示 `auto_trade_enabled` 不是 `should_trade` 的已知参数。

- [ ] **Step 3: 最小化实现总开关**

```python
def should_trade(
    direction: FinalPrediction,
    confidence: float,
    threshold: float = 0.6,
    auto_trade_enabled: bool = False,
) -> tuple[bool, str]:
    if not auto_trade_enabled:
        return False, "自动交易总开关未开启"
    if direction not in ("UP", "DOWN"):
        return False, f"方向为 {direction}，非可交易方向（仅 UP/DOWN 触发交易）"
    if confidence <= threshold:
        return False, f"置信度 {confidence:.4f} 未超过阈值 {threshold}（需严格大于）"
    return True, f"方向 {direction} 且置信度 {confidence:.4f} > 阈值 {threshold}，执行交易"
```

`SentimentAgent._write_prediction_and_trade` 调用 `should_trade` 时传入 `auto_trade_enabled=settings.agent_auto_trade`。Settings 新增 `agent_auto_trade: bool = False`，`.env` 追加 `AGENT_AUTO_TRADE=false`。
同步更新订单模型的说明，避免继续描述已删除的 `_auto_trade` 路径。

- [ ] **Step 4: 创建不含密钥的环境模板**

`.env.example` 只保留环境变量名、默认值和占位符，例如：

```dotenv
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/binance_predict
DEEPSEEK_API_KEY=
BINANCE_API_KEY=
BINANCE_API_SECRET=
PREDICTION_TRADE_AMOUNT_USDT=2.0
AGENT_AUTO_TRADE=false
```

`.gitignore` 新增 `.env`。

- [ ] **Step 5: 验证门控**

Run: `uv run pytest tests/test_agent_trade_gate.py -q`

Expected: `2 passed`。

### Task 2: 裁剪 Data Collector

**Files:**

- Create: `tests/test_data_collector_cleanup.py`
- Modify: `src/binance_predict/services/data_collector.py:1-490`
- Modify: `src/binance_predict/config/settings.py:40-48`
- Modify: `src/binance_predict/main.py:474-483`

- [ ] **Step 1: 写入失败测试**

```python
import inspect

from binance_predict.services.data_collector import BinanceDataCollector


def test_spot_collector_subscribes_only_to_book_ticker() -> None:
    source = inspect.getsource(BinanceDataCollector.connect_spot_ws)
    assert "@bookTicker" in source
    assert "@kline" not in source
    assert "@aggTrade" not in source
    assert "@depth" not in source
```

- [ ] **Step 2: 确认测试失败**

Run: `uv run pytest tests/test_data_collector_cleanup.py -q`

Expected: 断言失败，因为当前订阅列表仍包含 K 线、聚合成交和深度。

- [ ] **Step 3: 最小化实现**

将 `MarketDataStore` 限定为 `best_bid`、`best_ask`、`last_ws_spot_update`、`ws_spot_connected` 和 `mid_price`；`connect_spot_ws` 的 streams 固定为一个 `f"{symbol_lower}@bookTicker"`；`_handle_spot_message` 仅解析 `bookTicker`。

删除 `fetch_historical_klines`、`_fetch_and_store_klines`、`_fetch_mark_price`、`fetch_futures_rest_data`、`_safe_fetch`、合约回调、K 线事件和未使用的 HTTPX 导入。`start()` 只设置运行状态；Settings 删除不再使用的 REST/合约 WebSocket URL。健康检查只保留 `status`、`symbol`、`mid_price`、`ws_spot_connected`。

- [ ] **Step 4: 验证采集器**

Run: `uv run pytest tests/test_data_collector_cleanup.py -q`

Expected: `1 passed`。

### Task 3: 删除路径 A Schema 与死依赖

**Files:**

- Create: `tests/test_path_a_cleanup.py`
- Modify: `src/binance_predict/models/schemas.py:19-403`
- Modify: `pyproject.toml:7-31`
- Modify: `uv.lock`
- Modify: `src/binance_predict/config/settings.py:34-38`
- Delete: `src/binance_predict/scheduler/__init__.py`
- Delete: `src/binance_predict/api/__init__.py`

- [ ] **Step 1: 写入失败测试**

```python
from pathlib import Path


def test_path_a_models_and_dependencies_are_removed() -> None:
    schemas = Path("src/binance_predict/models/schemas.py").read_text(encoding="utf-8")
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    for symbol in ("DecisionOutput", "ReviewOutput", "MarketSnapshot", "PredictionRecord", "ReviewMemory"):
        assert f"class {symbol}" not in schemas
    assert "redis[hiredis]" not in pyproject
    assert "apscheduler" not in pyproject
```

- [ ] **Step 2: 确认测试失败**

Run: `uv run pytest tests/test_path_a_cleanup.py -q`

Expected: 断言失败，因为路径 A 模型和两项依赖仍存在。

- [ ] **Step 3: 最小化实现**

从 `schemas.py` 移除无消费者的路径 A 模型及只服务它们的枚举，保留 `FinalPrediction`、`ActualLabel`、`PatternStatus`、`ChangeType` 和 Learn/Predict/Evolve/查询 API 相关模型。删除 `TradeOrderRecord`，因为 API 直接生成字典且全仓库没有引用此模型。

从 `pyproject.toml` 删除 Redis、APScheduler；从 Settings 删除 `redis_url`；运行 `uv lock` 同步锁文件；删除两个空包文件和空目录。

- [ ] **Step 4: 验证清理**

Run: `uv run pytest tests/test_path_a_cleanup.py -q`

Expected: `1 passed`。

### Task 4: 文档、遗留测试与迁移操作边界

**Files:**

- Modify: `README.md`
- Modify: `tests/test_bug_condition_exploration.py:223-338`
- Preserve: `alembic/versions/b2c3d4e5f6a7_drop_path_a_tables.py`

- [ ] **Step 1: 删除已退役回测路径的断言**

删除 `test_bug_1_6_archive_should_invalidate_backtest_cache`、`test_bug_1_7_predict_should_auto_trigger_backtest` 和 `test_bug_1_8_predict_should_use_tracker_window_end`；它们要求已经删除的 `_last_backtest_result`、`_auto_run_backtest`、旧 `/api/sentiment/predict` 实现继续存在。

- [ ] **Step 2: 更新当前架构与人工迁移说明**

将 README 改为当前 Agent Loop 架构：`bookTicker` 仅提供中间价；路径 B 是唯一自动引擎；路径 C 是手动概率动量；旧回测与预测端点仅返回退役提示。删除已不存在的路径 A API、服务、Redis、APScheduler 和合约采集说明。

在迁移章节增加：先备份数据库，再运行以下命令；明确第二条会永久删除 7 张路径 A 表，应用启动不会执行它。

```powershell
uv run alembic current
uv run alembic upgrade head
```

- [ ] **Step 3: 验证全量后端测试**

Run: `uv run pytest -q`

Expected: 所有测试通过，不含路径 A 断言失败。

### Task 5: 最终静态检查与前端构建

**Files:**

- Verify only: 已修改文件。

- [ ] **Step 1: 验证无死引用**

Run: `rg -n "apscheduler|redis\[hiredis\]|redis_url|DecisionOutput|ReviewOutput|MarketSnapshot|@kline|@aggTrade|@depth" src pyproject.toml tests`

Expected: 无生产代码匹配；允许 README 中的迁移说明和已移除测试名称以外不出现任何结果。

- [ ] **Step 2: 验证前端构建**

Run: `npm run build`

Working directory: `frontend`

Expected: `tsc -b && vite build` 退出码为 0；若保留既有 bundle 大小警告，在交付中单独说明。

- [ ] **Step 3: 审查改动并提交**

Run: `git diff --check; git status --short`

只暂存本计划涉及的文件，确认 `.env` 仍未暂存后提交：

```powershell
git add .gitignore .env.example README.md pyproject.toml `
  src/binance_predict/config/settings.py `
  src/binance_predict/db/models.py `
  src/binance_predict/main.py `
  src/binance_predict/models/schemas.py `
  src/binance_predict/services/agent_logic.py `
  src/binance_predict/services/data_collector.py `
  src/binance_predict/services/sentiment_agent.py `
  tests/test_agent_trade_gate.py tests/test_data_collector_cleanup.py `
  tests/test_path_a_cleanup.py tests/test_bug_condition_exploration.py `
  docs/superpowers
git commit -m "refactor: 添加交易总开关并清理路径A遗留"
```
