# 新研究隔离规范

## 目标

每轮研究独立生成假设，避免旧结论、旧参数、旧特征和旧代码造成锚定；同时保留数据血缘、统计纪律、交易安全和可复现性。

**假设从零开始；方法学和安全规则不重置。**

## 阶段门禁

| 阶段 | 可读取内容 | 禁止行为 | 退出条件 |
|---|---|---|---|
| `PREREGISTRATION` | 本文件、研究问题、字段字典、数据 manifest | 查看结果分布、旧研究、旧策略实现 | 冻结研究问题、指标、切分、假设预算、停止条件 |
| `DISCOVERY` | 训练段原始数据和本轮代码 | 读取旧结论；读取 validation/holdout；搜索全仓库 | 产出候选及完整试验账本 |
| `VALIDATION` | 冻结候选、验证段 | 调参后反复重跑同一验证段；读取 holdout | 候选、参数和执行模型冻结 |
| `FROZEN_HOLDOUT` | 冻结实现、最终盲测段 | 修改候选；查询旧研究或实盘策略结果 | 生成并锁定盲测报告与产物哈希 |
| `COMPARISON` | 冷归档和生产结果 | 回写盲测结论 | 仅比较重复发现、冲突与新发现 |

除非用户明确指定当前阶段，新的研究请求默认从 `PREREGISTRATION` 开始。

## 发现阶段白名单

可读取：

- `docs/RESEARCH_ISOLATION.md`
- 本轮目录 `output/research_runs/<run_id>/`
- 经 manifest 冻结的原始 BTC OHLCV 文件
- 原始预测市场采样；数据库中仅限：
  - `prediction_market_samples`
  - `sentiment_windows`
- 中性的字段定义、数据采集代码和通用统计工具
- 交易所公开 API 文档

当前本地 BTC 数据基线：

- `output/klines_5m_720d.csv`：原始 5m OHLCV
- `output/klines_15m_720d.csv`：由完整 5m 桶严格聚合的派生数据
- `output/klines_5m_cache_720d.json`：Binance 原始响应缓存
- `output/klines_history_2160d_manifest.json`：2160 天数据血缘；CSV 不在当前工作区，使用前必须恢复或重拉并校验 SHA-256

开始研究前必须为实际输入生成独立 manifest，记录来源、时间范围、行数、重复/断点、文件 SHA-256 和切分边界。不能仅凭文件名认定数据版本。

## 发现阶段黑名单

在 `PREREGISTRATION`、`DISCOVERY`、`VALIDATION`、`FROZEN_HOLDOUT` 阶段禁止读取、搜索、索引或让子代理概括：

- 外部冷归档：`D:/project-archives/binance-predict/legacy-research-2026-09/`
- 旧研究报告：
  - `docs/research/edge-structure/`
  - `docs/research/REVERSAL_CONTINUATION_STUDY_20260917.md`
  - `docs/research/INTRAWINDOW_REVERSAL_TREND_START_V2_20260918.md`
  - `docs/G_SERIES_REVIEW_2026-09-11.md`
  - `docs/FIRSTHIT_SIGNAL_CONDITIONS.md`
  - `docs/FIRSTHIT_VISUAL_GUIDE.md`
  - `docs/ITERATIONS.md`
- 旧研究结果：除本轮目录及白名单原始数据外的 `output/`
- 所有 `.pytest_tmp/` 历史内容；本轮只允许新建 `output/research_runs/<run_id>/scratch/`
- 旧研究参数：`config/discovery_rounds/`、`config/qm_v2_hypotheses.json`
- 旧研究框架与策略实现：
  - `src/binance_predict/discovery/`
  - 旧 `scripts/local_*`、`scripts/research_*`、`scripts/scan_*`、`scripts/*backtest*`
  - 生产 detector、`live_channels.py` 和既有信号条件
- Git 历史：`git log`、`git blame`、历史分支和历史提交中的研究内容
- 旧结论数据库表：
  - `pattern_memory`
  - `fake_breakout_signals`
  - `misalignment_signals`
  - `kline_shadow_signals`
  - `pattern_shadow_signals`
  - `firsthit_shadow_signals`
  - 其他策略、假设、候选或影子信号表
- 在候选冻结前禁止读取 `trade_orders` 与 `shadow_execution_assessments`；冻结后只可作为执行可行性盲测输入，不得据此改写候选逻辑

禁止全仓库搜索。需要查找数据加载方法时，只在白名单路径内搜索。

## 永久保留的纪律

以下不是“旧策略经验”，不得删除或绕过：

- 防未来数据泄漏；所有特征必须仅使用决策时刻可得数据
- 时序切分，禁止随机打散时间序列
- 训练、验证、最终盲测物理隔离；holdout 只触碰一次
- 多重检验控制、样本量、置信区间、费用和真实报价
- 原始数据不可就地修改；派生数据必须记录代码版本和输入哈希
- `amount_in` 的 wei 语义、贴线执行价弃单、每通道每窗口一单等生产安全约束
- 新信号先 record-only / shadow；不得因单次历史回测直接实盘

## 本轮目录

每轮研究使用唯一目录：

```text
output/research_runs/<YYYYMMDDTHHMMSSZ>-<neutral-topic>/
├── PREREGISTRATION.md
├── DATA_MANIFEST.json
├── EXPERIMENT_LEDGER.jsonl
├── src/
├── scratch/
├── discovery/
├── validation/
├── holdout/
└── FINAL_REPORT.md
```

主题名必须中性，不得包含旧策略名或预期方向。

`PREREGISTRATION.md` 至少冻结：

1. 研究问题与可证伪的原假设；
2. 目标变量、经济指标和费用模型；
3. 时间切分边界；
4. 特征来源及决策时可得性；
5. 最大假设/参数试验预算；
6. 多重检验方法；
7. 晋级、失败和停止条件；
8. 哪些证据会推翻结论。

每次试验无论成功或失败都追加到 `EXPERIMENT_LEDGER.jsonl`，不得覆盖或只保留赢家。

## 解封与比较

只有同时满足以下条件才能读取冷归档：

1. 候选实现和参数已经冻结；
2. 最终盲测已完成；
3. `FINAL_REPORT.md`、输入、代码和结果均已有 SHA-256；
4. 用户明确同意进入 `COMPARISON`。

比较阶段只回答：是否独立重复发现、哪里冲突、什么是真正新增、是否重复失败。不得修改已经冻结的盲测结果。

## 冷归档

旧研究冷归档位于项目外：

```text
D:/project-archives/binance-predict/legacy-research-2026-09/
```

其 `README.md` 和 `metadata/MANIFEST.json` 记录范围、排除项和逐文件 SHA-256。生产源码和数据库未删除；归档中的生产源码只是审计快照。

## 生产边界

研究隔离不授权修改生产交易链路、删表、清库、删除订单或关闭现有通道。发现阶段不得用生产策略实现启发候选。任何研究结果进入生产，仍需单独评审、交易链路测试、影子验证和用户明确授权。
