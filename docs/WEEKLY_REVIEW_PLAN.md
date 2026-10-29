# LLM 每周复盘 + 动态仓位（%）实施计划

> 状态：进行中｜创建 2026-10-30｜口径与硬帽在本文件冻结，改动必须先改这里再改代码。
> 性质：**现有信号研究/运维增强**，不触发 `docs/RESEARCH_ISOLATION.md` 的新研究隔离。

---

## 0. 要做什么（一句话）

每周一自动产出一份**可追溯、可应用**的组合复盘：总体报告 / 单信号报告 / 改进报告 / 注意点 /
探索性研究方案 / 仓位（% 基准本金）建议；LLM 只做假设与建议，金额与参数由确定性代码算、
人工一键应用。

## 1. 已拍板口径（2026-10-30，用户确认，不再讨论）

| # | 决策 | 值 | 影响 |
|---|---|---|---|
| D1 | 「总金额」= 仓位分母 | **手动固定基准本金** `base_equity_usdt`（默认 200），不随实时权益浮动 | 下单路径无需查余额 API，金额可复现、可审计 |
| D2 | 决定权 | 规则算金额；LLM 只建议 `pct`；**人工**一键应用 | LLM 永远不进交易决策路径 |
| D3 | 载体 | 落库（新表）+ 前端**新 tab「每周复盘」**；不推微信/邮件 | 无推送噪声；微信/邮件留作后续 |
| D4 | 探索性研究 | 只产出「下周可跑的方案清单」，**不自动跑回测** | 研究纪律与算力成本不进自动化 |

## 2. 硬帽与安全边界（代码强制，不靠自律）

| 约束 | 值 | 强制点 |
|---|---|---|
| `pct` 合法区间 | `[0.002, 0.05]`（0.2%~5%） | `parse_channel_config` / `set_channel` 拒改 |
| 单笔金额下限 | `0.1` USDT（币安可下单最小粒度） | `position_sizing.resolve_amount` |
| 单笔金额硬上限 | `min(amount_cap_usdt, MAX_ORDER_AMOUNT_USDT=50)` | 同上 |
| `amount_cap_usdt` 默认 | `20.0`（当前在用最大单笔），调高必须显式填 | `ChannelConfig` 默认值 |
| 基准本金合法区间 | `[10, 5000]` USDT | `set_base_equity` 拒改 |
| LLM 对 `pct` 的方向 | **只许下调**；上调只能人工手填 | `verify_weekly_review` |
| 应用冷冻期 | 复用 `strategy_advice_apply_freeze_days/max`（7 天 ≤2 次） | apply 端点 |
| 金额粒度 | 向下取整到 0.1（`floor`，绝不向上越线） | `resolve_amount` |
| 失败降级 | 配置缺失/非法 → **回落固定 `amount_usdt`**，绝不放大也绝不停火 | `_sizing_amount` |

**不变式**：LLM 输出永远只是建议；任何改钱的动作都必须 `POST .../apply`（服务端二次校验）。
`pct` 模式下，调用点已施加的乘数（如 KDC 高波半仓 `highvol_amount_factor`）**保留**——
sizing 只替换「基准金额」这一层，见 §3.3。

## 3. 数据契约

### 3.1 事实卡（LLM 的唯一数字来源）

`services/weekly_review.py::collect_facts()` 产出，`schema = "weekly_review_facts_v1"`：

```
week{key,start_ms,end_ms,prev_start_ms,prev_end_ms}
portfolio{n,cost,pnl,roi,win_rate,mean_breakeven,edge,
          ev_bootstrap{p_le_zero,significant_positive},
          prev{...},delta{...},
          concentration{top1_channel,top1_share,top3_share,by_channel[]},
          daily[{day,n,cost,pnl}],
          rejected{total,guard,other},
          equity{base_equity_usdt,available_usdt|null,open_cost,drift_ratio|null}}
channels[{channel,display_name,enabled,params{...},
          n,cost,pnl,roi,win_rate,wilson_low,benchmark,benchmark_gap,
          mean_breakeven,edge,trend_level,p_edge_negative,
          facts{streak,edge_decomp,trend,entry_band_ev,hour_bucket_ev},
          avg_stake,rule_stake,rejected{total,guard},first_order_ts}]
watchouts[{code,severity,channel,statement,evidence_ids,numbers[]}]
evidence_ids[]
```

- 每个数值都有 `evidence_id`（与 `attribution_engine.compute_channel_facts` 同风格前缀）。
- 单通道派生指标**复用** `compute_channel_facts`，不新写一套口径。
- 弃单分类复用 `live_performance.is_price_guard_rejection`。

### 3.2 注意点：10 条确定性规则（不经过 LLM）

| code | 触发 | 级别 |
|---|---|---|
| `CONCENTRATION` | 单通道资金占比 ≥30%，或 top3 ≥65% | P1 |
| `SMALL_SAMPLE_RANK` | `n < 30` 且本周 PnL > 0（排名不可信） | P2 |
| `BENCHMARK_DIVERGENCE` | `n ≥ 30` 且 `\|benchmark_gap\| ≥ 0.10` | P1 |
| `GUARD_AT_BREAKEVEN` | `max_exec_price ≥ 0.98 × benchmark`（护栏卡在保本线，拦正 EV） | P1 |
| `GUARD_REJECT_RATE` | 弃单率 ≥50%（护栏过紧或不可成交） | P1 |
| `UNSETTLED_BACKLOG` | 窗口结束 >30min 仍 `settled_at IS NULL` | P0 |
| `EQUITY_DRIFT` | `\|available − base_equity\| / base_equity ≥ 20%`（余额不可得则跳过） | P2 |
| `TREND_UNRESOLVED` | 趋势 `L2/L3` 且该通道最新建议仍 `PENDING` | P1 |
| `EDGE_NEGATIVE` | `edge < 0` 且 `p_edge_negative ≥ 0.90` 且 `n ≥ 30` | P0 |
| `NEW_CHANNEL_NO_GATE` | `enabled` 且首单距今 <60 天（影子门槛未满已开实盘） | P1 |

LLM 可以补充注意点，但必须过 claim 闸（证据 ID 存在 + 数值精确匹配），且**不能降级**规则产出的级别。

### 3.3 仓位规则（`services/position_sizing.py`，纯函数）

```
ratio  = incoming_amount / cfg.amount_usdt        # 保留调用点乘数（高波半仓=0.5）
raw    = base_equity_usdt × pct × ratio
amount = floor(clamp(raw, 0.1, min(cap_usdt, 50)) × 10) / 10
if raw < 0.1:  flag below_floor（取 0.1，记录审计）
```

- `pct is None` → **完全不动**（当前行为 100% 不变，默认全通道关闭）。
- 输出 `{amount, pct, base_equity, cap, ratio, below_floor, rule}` 写进
  `assessment_context.patch_values["sizing"]` 与日志，供事后核对「实际注码 vs 规则注码」。

## 4. 任务表（唯一追踪源：改状态必须同时填证据）

状态只允许 `TODO / DOING / DONE / BLOCKED`。`DONE` 必须附可复现证据（命令 + 关键输出）。

| ID | 任务 | 验收标准 | 涉及文件 | 测试 | 状态 | 证据 |
|---|---|---|---|---|---|---|
| WR-01 | 本任务文档 | 口径/硬帽/任务表/变更日志齐备 | `docs/WEEKLY_REVIEW_PLAN.md` | — | DONE | 本文件 |
| WR-02 | 配置项 | 新增配置全默认关闭；导入通过 | `config/settings.py` | 全量回归绿 | DONE | `1671 passed`；`import binance_predict.main` OK |
| WR-03 | 组合+单信号事实卡 | `collect_facts()` 可用；空窗口返回零样本卡 | `services/weekly_review.py` | `tests/test_weekly_review.py` | DONE | `test_collect_facts_empty_week` |
| WR-04 | 10 条注意点规则 | 每条规则正反例各一 | 同上 | 同上 | DONE | `test_concentration_watchout` … `test_watchouts_sorted_by_severity` |
| WR-05 | 仓位纯函数 | pct 关闭不动；截断/下限/below_floor 可复现 | `services/position_sizing.py` | `tests/test_position_sizing.py` | DONE | `15 passed` |
| WR-06 | 下单接缝 | `_exec_with_exclusive` 统一改写金额 + 审计入 assessment | `services/multi_live_trader.py` | `tests/test_multi_live_trader.py` | DONE | 组 9 九条用例全绿 |
| WR-07 | 配置持久化 | 两列 + 单行表 + 迁移；重启恢复；越界拒改 | `db/models.py`、`live_channels.py`、`multi_live_trader.py`、`alembic/versions/w8r1v2w3e4k5_*` | 同 WR-06 | DONE | `test_set_channel_rejects_out_of_range_pct` / `test_set_base_equity_bounds` |
| WR-08 | LLM 周报 | prompt + claim 闸；pct 只许降；探索项必须带停止条件 | `services/llm_panel.py` | `tests/test_weekly_review.py` | DONE | `test_gate_*` 十一条 |
| WR-09 | 落库与端点 | 列表/详情/触发/应用四端点 | `main.py`、`db/models.py` | `tests/test_weekly_review.py` | DONE | 路由注册已验；`import main` OK |
| WR-10 | 前端 tab | `TAB_IDS` 加 `weekly`；六段式 | `frontend/src/App.tsx` | `npm run build` / `npm run lint` | DONE | `✓ built in 4.62s`；lint 仅 2 条既有 warning |
| WR-11 | 定时任务 | 周一 00:00–00:20 UTC 先组合周报再逐通道归因 | `main.py` | 代码审阅 + 手动触发复用同一函数 | DONE | `_attribution_weekly_loop` |
| WR-12 | 上线观察与回滚 | 文档含观察清单 + 逐步回滚；**真机手动生成一次** | 本文件 | — | DOING | 观察/回滚已写 §5；真机验证待部署后（需用户指令 push） |

### 后续候选（不在本次范围）

- `WR-13` 统一影子 loader：把 8 张影子信号表归一，替代 `BENCHMARK_DIVERGENCE` 的代理口径。
- `WR-14` 探索性研究「一键立项」：生成 `output/research_runs/<run_id>/PREREGISTRATION.md` 骨架（仍不跑数据）。
- `WR-15` 周报微信/邮件推送（`send_plain_email` 已可用）。

## 5. 上线观察与回滚

**观察清单（上线后前 4 周）**
1. `weekly_reviews` 每周恰生成 1 行 `WEEKLY` + 若干 `MANUAL`，`status=DONE`；`FAILED` 必须带 `llm_error`。
2. 事实卡 `portfolio.cost` 与 `/api/live/performance-summary` 同窗口 `total_cost` 的差 **应恰好等于该周未结算 FILLED 单的本金之和**（周报含未结算、面板只取已结算）；时间口径已统一为 `window_start`，分桶不应再有差异。
3. 未开 `pct` 的通道：`trade_orders.amount_in/1e18` 与配置金额完全一致（零行为变化）。
4. 开 `pct` 的通道：日志「仓位裁决 | 请求 xU → 实下 yU」与实付一致；`assessment_context.patch_values.sizing` 可回查。
5. 注意点 P0 数量、误报率；连续两周同一 `code` 复现则视为规则阈值需校准。
6. 每周报 `rejected` 数量：持续偏高（>10 条）说明 prompt 与事实卡字段名对不上，需修字段命名而非放宽闸门。
7. 同一周只应有一行 `WEEKLY`（幂等已加）；出现两行说明定位到 `find_weekly_review` 失效。

**回滚（每步独立）**
1. 关仓位：通道 `amount_pct` 置 NULL（`POST /api/live/toggle` + `reset_amount_pct=true`）或 `live_amount_pct_enabled=false` → 立即回落固定金额。
2. 关周报 LLM：`weekly_review_llm_enabled=false` → 只产出确定性事实卡与注意点。
3. 关定时：`weekly_review_enabled=false` → 不再自动生成，历史行只读保留。
4. 代码回滚：`git revert` 单条迁移 + 服务重启；两张新表为**只读分析层**，删除不影响下单链路。

## 6. 变更日志（append-only，只在末尾追加）

- 2026-10-30｜WR-01｜建文档：冻结 D1–D4 与 8 条硬帽；确认 10 条注意点规则（影子对比降级为 `BENCHMARK_DIVERGENCE` 代理口径，真实影子 loader 顺延 WR-13）。
- 2026-10-30｜WR-02..WR-11｜代码落地：`position_sizing`（纯函数）+ `weekly_review`（事实卡/规则/编排）+ `llm_panel` 周报段（prompt + 硬闸）+ 迁移 `w8r1v2w3e4k5` + 四端点 + 前端 `weekly` tab。
- 2026-10-30｜口径修正｜弃单统计改由 `is_price_guard_rejection` 分类（护栏 vs 其它）；`UNSETTLED_BACKLOG` 改为「已成交 + 未结算 + 窗口结束超 30min」计数（避免把 FAILED 单误计成结算卡住）。
- 2026-10-30｜闸门修正｜周报断言必须至少带 1 个 `evidence_id`（空证据原先会静默过闸，已修）。
- 2026-10-30｜代码审核｜P0：周报分母改用**运行时** `base_equity`（原先读 env 默认，与 `rule_stake` 不同源）；P1：分桶改 `window_start`、`weekly_review_timeout_s` 接通、删死配置 `weekly_review_lookback_days`、触发失败不再返回 `null`；P2：乘数不再被 floor 抬升、sizing 基数缺失直接拒、应用时重算当前金额、同周 `WEEKLY` 幂等、重复应用 noop。
- 2026-10-30｜验证｜`389 passed` / 全量 `1676 passed, 15 skipped`；`npm run build` OK。

## 7. 实施记录（代码地图 / 验证方式 / 已知边界）

### 7.1 代码地图

| 层 | 文件 | 关键入口 |
|---|---|---|
| 仓位规则 | `services/position_sizing.py` | `resolve_amount` / `validate_pct` / `validate_base_equity` |
| 下单接缝 | `services/multi_live_trader.py` | `_sizing_amount` / `_exec_with_exclusive` / `set_base_equity` |
| 事实卡+规则 | `services/weekly_review.py` | `collect_facts` / `derive_watchouts` / `run_weekly_review` |
| LLM 周报 | `services/llm_panel.py` | `WEEKLY_SYSTEM_PROMPT` / `build_weekly_prompt` / `verify_weekly_review` |
| 端点 | `main.py` | `/api/weekly/reviews*`、`/api/live/sizing-config` |
| 前端 | `frontend/src/App.tsx` | `WeeklyReviewTab`（tab id `weekly`） |
| 存储 | `db/models.py` + `alembic/versions/w8r1v2w3e4k5_*` | `live_sizing_config`、`weekly_reviews`、`live_channel_overrides.amount_pct/amount_cap_usdt` |

迁移由容器启动时 `alembic upgrade head` 自动执行（`docker/backend.Dockerfile` CMD），无需手工跑。

### 7.2 验证命令

```powershell
.venv\Scripts\python.exe -m pytest tests/test_weekly_review.py tests/test_position_sizing.py -q   # 56 passed
.venv\Scripts\python.exe -m pytest tests/test_multi_live_trader.py -q                            # 322 passed
.venv\Scripts\python.exe -m pytest tests/ -q                                                     # 1676 passed, 15 skipped
cd frontend; npm run build                                                                        # tsc + vite OK
```

### 7.3 已知边界（诚实清单）

1. **影子对比是代理口径**：`BENCHMARK_DIVERGENCE` 比的是「前向胜率 vs 冻结研究基准」，不是「实盘 vs 影子同事件」；真实影子 join 顺延 WR-13。
2. **注意点只看本周窗口**：`UNSETTLED_BACKLOG` 只扫本周内的订单；更早的卡单不会出现在周报（需另跑 settle-scan 巡检）。
3. **权益口径**：`available_usdt` 取预测钱包可用 USDT（不含现货、不含未实现盈亏），且只在 `EQUITY_DRIFT` 提示里用，不参与金额算术。
4. **仓位默认关闭**：`live_amount_pct_enabled=false` + 通道 `amount_pct=NULL` 双闸，当前行为与上线前 100% 一致。
5. **无推送**：周报只在库与前端（D3 拍板）；微信/邮件顺延 WR-15。
6. **探索性研究不执行**：只出方案（D4）；是否立项由人工另行发起，仍受现有信号研究纪律约束。
7. **开 % 模式需改 env 并重启**：`live_amount_pct_enabled` 是 env-only（不是 DB 热调），开启要改 `.env` + 部署；关闸则可纯运行时（`reset_amount_pct=true`）。
8. **只对下周生效**：周报事实卡与仓位建议基于上一个完整周，不会盘中重算。

### 7.4 代码审核结论（2026-10-30）

审核范围：本次全部改动（迁移 + 服务层 + 端点 + 前端）。已修复项均为 P0–P2，P3 保留为已知边界。

| 级别 | 发现 | 处置 |
|---|---|---|
| P0 | 周报分母用 env 默认值，与 `rule_stake`（运行时 base）不同源 | `run_weekly_review(base_equity=...)` 透传，`_run_llm` 用事实卡同一值 |
| P1 | 分桶用 `created_at`，与性能面板 `window_start` 不一致 | `_load_orders` 改 `window_start`，NULL 才回退 `created_at` |
| P1 | `weekly_review_timeout_s` / `lookback_days` 是死配置 | 超时传入 `LLMPanel(timeout_s=)`；删除 lookback 配置 |
| P1 | 事实卡失败时端点返回 `null`，前端 `res.id` 抛错 | 端点返回 `{error}`；前端先判 `res?.id` |
| P2 | `RATIO_FLOOR` 会把极小乘数抬回并放大金额 | 去掉 floor，比值夹到 `(0, 1]` |
| P2 | `rule_stake` 缺失时「当前金额=0」绕过放大闸 | 回落 `amount_usdt`；两者都缺 → `sizing_basis_missing` 拒掉 |
| P2 | 应用时用报告快照的 `current_amount`（可能过期） | 改用 `status()` 的 `rule_amount/amount_usdt` 重算 |
| P2 | 00:00–00:20 UTC 重启会为同周再插一行 `WEEKLY` | `find_weekly_review(week_key,'WEEKLY')` 幂等返回 |
| P3 | 重复应用同一 pct 会消耗冷冻期额度 | `amount_pct` 已相等 → 返回 `noop` 不落日志 |
| P3 | `amount_pct=None` 的路径进 `evidence_ids` 但不在数值表 | 保留（仅造成拒绝噪音，不影响安全）；后续统一路径生成 |

