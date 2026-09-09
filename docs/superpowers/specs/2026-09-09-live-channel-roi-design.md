# 需求与设计规格：实盘信号与盈利趋势增加盈利率（ROI）

## 1. 背景与目标
目前实盘监控只展示「累计盈亏（USDT）」绝对金额。当各通道的单笔下注金额被频繁调大或调小时，绝对金额无法反映信号本身的真实预测强度与胜率赔率质量（例如：小金额高质量信号被大金额但平庸的信号掩盖）。
本次目标：全链路为实盘通道与订单分析引入**资金加权收益率（ROI）**指标，在实盘主页面及右侧抽屉「盈利趋势」中详细展现。

## 2. 口径定义（无歧义计算标准）
- 单笔投入金额：`cost = amount_in / 1e18` (USDT)
- 单笔盈亏：`pnl` (USDT)
- 通道级累计已结算投入：`total_cost = sum(cost)` (USDT)
- 通道级累计盈亏：`total_pnl = sum(pnl)` (USDT)
- 通道级总盈利率：`roi = total_pnl / total_cost`（若 total_cost <= 0 则为 null）
- 第 $n$ 笔滚动盈利率：`cum_roi = cum_pnl / cum_cost`（用于时序走势图）

## 3. 具体修改设计

### 3.1 后端修改 (`src/binance_predict/main.py`)
- 端点：`GET /api/live/pnl-curve`
  - SQL 查询补充字段：`TradeOrderModel.amount_in`
  - 逐通道每个点 `points` 增加字段：
    - `cost`: round(amount_in / 1e18, 4)
    - `cum_cost`: 累计投入 (USDT)
    - `cum_roi`: round(cum_pnl / cum_cost, 4) if cum_cost > 0 else 0.0
  - 逐通道 `channels` 对象增加字段：
    - `total_cost`: round(sum(cost), 4)
    - `roi`: round(total_pnl / total_cost, 4) if total_cost > 0 else None
  - 顶层 `total` 汇总对象增加字段：
    - `total_cost`: round(sum(c.total_cost), 4)
    - `roi`: round(total_pnl / total_cost, 4) if total_cost > 0 else None

### 3.2 前端修改 (`frontend/src/App.tsx`)
1. **接口定义扩展**：
   - `PnlPoint` 补充 `cost: number; cum_cost?: number; cum_roi?: number`
   - `PnlChannel` 补充 `total_cost: number; roi: number | null`
   - `PnlCurveData.total` 补充 `total_cost: number; roi: number | null`
2. **右侧抽屉「盈利趋势」Tab (`PnlCurveCard`) 强化**：
   - **顶部卡片**：新增「总收益率 (ROI)」卡片，正值绿色、负值红色，附带小字展示总本金投入。
   - **折线图指标切换**：在 `[累计盈亏, 胜率趋势, 单笔盈亏]` 基础上增加 **`收益率趋势 (ROI)`**（百分比，Y轴带 `%` 格式化，0%基准线）。
   - **下方明细表格**：新增「总投入」和「盈利率(ROI)」两列。支持按 ROI 或盈亏点击表头排序，直观对比哪条通道信号强度最高。
3. **实盘主页面「信号实盘通道」列表 (`LiveChannelsCard`) 强化**：
   - 将 `pnlCurveData` 数据同步挂载或共享到 `LiveChannelsCard`。
   - 在每个通道行的金额/护栏旁展示战绩标签：如 `累计 +12.3U (+21.5%)` 或 `无单`。

### 3.3 审计与测试保障
- 单元测试覆盖：更新并补充 `tests/test_multi_live_trader.py` 中 `test_live_pnl_curve_endpoint`，断言 `total_cost`、`roi` 及 `points[].cum_roi` 计算逻辑。
- 前端编译检查：`cd frontend && npm run build`（含 strict tsc 检查）通过。
- Oxlint 检查：`cd frontend && npm run lint` 通过。
- 全量后端测试：`.venv/Scripts/python.exe -m pytest tests/ -q` 必须保持全绿。
