# 实盘信号与盈利趋势增加盈利率（ROI）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为实盘通道与订单分析全面引入资金加权盈利率（ROI），并在实盘主页面通道列表及右侧抽屉「盈利趋势」中详细展现与排序。

**Architecture:** 后端 `/api/live/pnl-curve` 在 SQL 中补充 `amount_in` 计算每笔投入 `cost`、累加 `cum_cost`、逐笔及汇总派生 `roi`；前端扩展类型并在抽屉卡片中增加总 ROI 卡片、ROI 折线趋势图、明细表投入与 ROI 列及排序，同时在实盘主界面通道卡片同步展示通道战绩摘要。

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy (PostgreSQL/SQLite), React 19, TypeScript, Tailwind CSS, Recharts.

---

### Task 1: 后端 `/api/live/pnl-curve` 支持投入本金与盈利率计算

**Files:**
- Modify: `src/binance_predict/main.py:2485-2568`
- Test: `tests/test_multi_live_trader.py:2045-2102`

- [ ] **Step 1: 在 `tests/test_multi_live_trader.py` 中更新并编写带 `amount_in` 与 ROI 断言的测试**

在 `tests/test_multi_live_trader.py` 的 `test_live_pnl_curve_endpoint` 中为测试数据增加第 6 列 `amount_in`（wei 字符串），并增加对 `cost`、`cum_roi`、`total_cost`、`roi` 的断言：

```python
    rows = [
        # (signal_version, window_start, settled_at, win, pnl, amount_in) 按 window_start 升序
        ("quote_contrarian_v1", 1000,
         datetime(2026, 8, 30, 10, 0, tzinfo=timezone.utc), True, 4.0, "2000000000000000000"),  # 2 U
        ("scene_bull_exhaust", 1500, None, True, 1.5, "1000000000000000000"),  # 1 U
        ("quote_contrarian_v1", 2000,
         datetime(2026, 8, 30, 10, 5, tzinfo=timezone.utc), False, -2.0, "2000000000000000000"),  # 2 U
    ]
```
断言：
- `out["total"]["total_cost"] == 5.0`
- `out["total"]["roi"] == 0.7` (3.5 / 5.0)
- `q["total_cost"] == 4.0`
- `q["roi"] == 0.5` (2.0 / 4.0)
- `q["points"][0]["cost"] == 2.0`
- `q["points"][0]["cum_roi"] == 2.0` (4.0 / 2.0)
- `q["points"][1]["cum_roi"] == 0.5` (2.0 / 4.0)

- [ ] **Step 2: 运行测试以验证红灯失败**

运行：`.venv/Scripts/python.exe -m pytest tests/test_multi_live_trader.py -k test_live_pnl_curve -q`
预期：FAIL（因解包缺少字段或断言字段缺失）。

- [ ] **Step 3: 修改 `src/binance_predict/main.py` 中的 `live_pnl_curve` 端点**

在 `live_pnl_curve` 中：
1. `sa_select` 增加 `TradeOrderModel.amount_in`；
2. 循环解包 `ver, ws, settled_at, win, pnl, amount_in`；
3. 计算 `cost = round(float(amount_in or 0) / 1e18, 4)`；
4. 维护逐通道累加 `cum_cost += cost`，计算 `cum_roi = round(cum / cum_cost, 4) if cum_cost > 0 else 0.0`；
5. `points` 输出包含 `cost`、`cum_cost`、`cum_roi`；
6. 通道对象包含 `total_cost`、`roi`；
7. `total` 汇总对象包含 `total_cost`、`roi`。

- [ ] **Step 4: 运行测试验证绿灯通过**

运行：`.venv/Scripts/python.exe -m pytest tests/test_multi_live_trader.py -k test_live_pnl_curve -q`
预期：PASS（全部通过）。

- [ ] **Step 5: 运行相关测试全集**

运行：`.venv/Scripts/python.exe -m pytest tests/test_multi_live_trader.py -q`
预期：全部用例通过。

---

### Task 2: 前端右侧抽屉「盈利趋势」卡片（`PnlCurveCard`）全维度展示盈利率与排序

**Files:**
- Modify: `frontend/src/App.tsx:2220-2415`

- [ ] **Step 1: 扩展前端接口类型定义**

在 `frontend/src/App.tsx` 中：
- `PnlPoint`: 补充 `cost: number; cum_cost?: number; cum_roi?: number`
- `PnlChannel`: 补充 `total_cost: number; roi: number | null`
- `PnlCurveData`: `total` 补充 `total_cost: number; roi: number | null`
- `PnlMetric`: 扩展为 `'cum' | 'rate' | 'pnl' | 'roi'`，在 `PNL_METRICS` 中增加 `{ key: 'roi', label: '盈利率(ROI)' }`

- [ ] **Step 2: 改造抽屉顶部数据卡片栏**

在 `PnlCurveCard` 顶部汇总区域中：
- 增加「总盈利率（ROI）」卡片：
  - 显示大号百分比数字（如 `+28.4%` 或 `-12.1%`，带颜色高亮）。
  - 副标题展示「累计投入 XX.XX USDT」。

- [ ] **Step 3: 折线图支持「盈利率趋势 (ROI)」模式**

在 `rows` 数据行生成中：
- 当 `metric === 'roi'` 时，提取当前点的 `cum_roi`（或由前端根据 `cum / cum_cost` 兜底计算百分比）。
- `YAxis` 的 `tickFormatter`：当 `metric === 'rate' || metric === 'roi'` 时格式化为 `${v}%`。
- `Tooltip` 的 `formatter`：当 `metric === 'roi'` 时显示为 `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`。
- `ReferenceLine` 基准线：当 `metric === 'roi'` 时停留在 `0%`（盈亏平衡线）。

- [ ] **Step 4: 通道明细表格增加「总投入」和「盈利率(ROI)」列，并支持多维度排序**

在表格中：
- 表头增加 `总投入 (USDT)` 和 `盈利率 (ROI)`。
- 增加排序状态 `sortField`（默认按 `roi` 或 `total_pnl` 排序）和 `sortAsc`，支持点击表头切换升降序。
- 盈利率单元格带正负颜色高亮（正绿色，负红色，0灰色）。

- [ ] **Step 5: 验证前端构建与 Lint**

运行：
`cd frontend && npm run lint && npm run build`
预期：oxlint 0 errors，tsc 构建成功无类型报错。

---

### Task 3: 前端实盘主界面通道卡片（`LiveChannelsCard`）集成战绩与盈利率摘要

**Files:**
- Modify: `frontend/src/App.tsx:2460-2945`

- [ ] **Step 1: 在 `LiveTradingTab` 中拉取并传递 `PnlCurveData` 数据**

在 `LiveTradingTab` 中：
- `refresh` 函数中并行获取 `api.getLivePnlCurve()` 并存入 `pnlData` 状态。
- 计算通道快捷索引映射 `pnlMap: Record<string, PnlChannel>`。
- 将 `pnlMap` 传递给 `LiveChannelsCard`。

- [ ] **Step 2: 在 `LiveChannelsCard` 通道列表行展示盈利率与战绩**

在每行通道（在“今日成交/开火”与“金额设置”之间，或通道标题右侧）：
- 若该通道有已结算单：
  展示战绩徽标：如 `+3.50 U (+17.5%)`（绿色）或 `-1.20 U (-8.3%)`（红色），鼠标悬浮提示 `已结算 N 单 · 总投入 X.XX USDT`。
- 若该通道无已结算单：
  展示灰色微标 `无单` 或 `--`。

- [ ] **Step 3: 验证前端编译与类型检查**

运行：
`cd frontend && npm run lint && npm run build`
预期：通过且无错误。

---

### Task 4: 完整审计、回归测试与最终推送

**Files:**
- All modified files
- Run Git status & Git diff check

- [ ] **Step 1: 运行全量后端测试**

运行：`.venv/Scripts/python.exe -m pytest tests/ -q`
预期：全绿（0 失败）。

- [ ] **Step 2: 运行前端构建与 Lint 全检**

运行：`cd frontend && npm run lint && npm run build`
预期：全部通过，产物无警告。

- [ ] **Step 3: 审查 Git 变更范围**

运行：`git diff`
预期：改动仅限于 ROI 相关的计算、呈现和单测，未夹带无关变更。

- [ ] **Step 4: 提交并推送到 GitHub (CI 触发部署)**

执行：
- `git add src/binance_predict/main.py tests/test_multi_live_trader.py frontend/src/App.tsx docs/superpowers/`
- `git commit -m "feat(ui): 实盘信号与盈利趋势全面增加盈利率ROI指标及通道战绩展示"`
- `git push origin main`
- 使用 `gh run watch` 监控 Actions CI 流水线直到全部通过。
