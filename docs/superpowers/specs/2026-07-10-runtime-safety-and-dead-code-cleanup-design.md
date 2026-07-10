# 运行安全与死代码清理设计

## 目标

在不重写 Agent 主链路的前提下，增加真实交易总开关，移除路径 A 遗留模型、采集、依赖和空包，并保留数据库 DROP 迁移为人工操作。

## 范围

### 交易安全开关

- 在 `Settings` 中新增 `agent_auto_trade: bool = False`。
- `should_trade` 必须先检查总开关；关闭时无论方向和置信度如何均返回“不交易”及非空原因。
- 只有 `agent_auto_trade=true`、方向为 `UP/DOWN` 且置信度严格大于配置阈值时，才允许调用 `execute_trade`。
- `.env` 默认写入 `AGENT_AUTO_TRADE=false`；新增无密钥的 `.env.example`。
- `.gitignore` 必须忽略 `.env`，避免真实密钥被误提交。

### Schema 清理

- 删除 Features、`MarketSnapshot`、`DecisionOutput`、`ReviewOutput`、`PredictionRecord`、`PredictionResult`、`CustomRule`、`ReviewMemory`、`TradeOrderRecord` 及其仅内部引用的路径 A 枚举。
- 保留 Agent Loop 仍在导入的 `FinalPrediction`、`ActualLabel`、`PatternStatus`、`ChangeType` 和 Agent 系列 Schema。

### Data Collector 清理

- 现货 WebSocket 仅订阅 `bookTicker`。
- `MarketDataStore` 仅保留 `mid_price` 计算和当前运行链路实际读取的连接状态字段。
- 删除历史 K 线预拉取、K 线事件、聚合成交、深度盘口、合约 REST 采集及其无消费者存储字段。
- 健康检查不再报告已经失去真实数据来源的 `rest_api_ok`。
- 保持断线重连、指数退避和优雅关闭行为。

### 依赖、配置与空包

- 从 `pyproject.toml` 删除 `apscheduler`、`redis[hiredis]`。
- 从 Settings 和环境示例删除仅服务于上述死路径的 Redis 配置。
- 删除空的 `src/binance_predict/scheduler` 与 `src/binance_predict/api` 包。

### 数据库迁移

- 不在应用启动时自动执行 `alembic upgrade head`。
- 保留 `b2c3d4e5f6a7_drop_path_a_tables.py`。
- 在 README 中说明：执行前备份数据库，人工运行 `uv run alembic current` 与 `uv run alembic upgrade head`，该迁移会永久删除 7 张路径 A 表。
- 当前数据库不可连接时，只报告“迁移状态未验证”，不得声称已经执行。

## 数据流

```text
Binance bookTicker
    -> best_bid / best_ask
    -> mid_price
    -> 窗口 entry_price / exit_price

Agent Predict
    -> agent_auto_trade=false：记录跳过原因，不调用交易接口
    -> agent_auto_trade=true：继续检查方向与置信度
    -> 满足全部条件：调用 execute_trade
```

## 错误处理

- 开关关闭属于正常门控，不记录为异常，但必须写入 `skip_trade_reason`。
- WebSocket 解析错误、断连和网络错误继续保留明确日志。
- 人工迁移连接失败时不修改数据库，也不回退为自动建删表。

## 测试与验收

- 先增加失败测试，证明默认配置禁止真实交易。
- 覆盖开关关闭、开关开启但置信度不足、开关开启且满足条件三种门控结果。
- 删除只验证已退役回测路径存在的 3 个过期测试。
- 静态检查路径 A Schema、Redis、APScheduler、K 线/成交/深度采集不再被生产代码引用。
- 运行 `uv run pytest -q`，要求全部通过。
- 运行 `npm run build`，要求前端构建成功；现有大包体积警告不属于本次范围。

## 非目标

- 不自动执行破坏性数据库迁移。
- 不修改 Agent 的 Learn、Predict、Validate、Evolve 算法。
- 不调整交易金额、置信度阈值或 LLM 提示词。
- 不处理预测与订单跨事务落库问题；该问题应单独设计和修复，避免扩大本次清理范围。
