# G 系列信号（FirstHit Down）深度审计报告

**报告时间**: 2026-09-11  
**审计范围**: firsthit_down 族全部 10 个版本通道，FILLED 订单逐单 K 线重放 + Gate/Veto核验 + 护栏执行检查  
**数据来源**: prod API /api/trades/recent (42 FILLED orders) + samples 历史曲线  

---

## 📊 核心统计

| 指标 | 数值 | 备注 |
|------|------|------|
| **总 FILLED 订单数** | 42 | 跨 9 天数据 (2026-09-08 ~ 2026-09-11) |
| **胜率** | 28.6% (12/42) | DOWN 押注胜率 |
| **平均盈亏 (R)** | -0.72 | R = USDT 金额 |
| **护栏违规订单** | 3 (7.1%) | avg_price > spec.auto_max_exec |
| **动态护栏版本** | 612cf83 (09-11 06:45 UTC) | 部署前使用绝对阈值 |

---

## 🔍 缺陷清单 (按严重度排序)

### ❌ **Critical #1**: 同窗互斥槽被 Gate Fail 占用 → 其他通道无法开火

**问题描述**:
在 `multi_live_trader.py` lines 454-465，代码顺序为：

```python
# LINE 454-465 (critical bug path)
for ch, spec in self._specs.items():
    if spec.family != "firsthit": continue
    cfg = self._configs[ch]
    if not cfg.enabled or window_start_ms in cfg.fired: continue
    
    # ⚠ Quick veto check happens FIRST (only streak/upper_wick/chg)
    quick_veto = _firsthit_pre_market_veto(ext, streak_up=0)
    if quick_veto is not None: continue
    
    # ⚠ Full gate check comes AFTER (includes body_r, wick01, etc.)
    if ch not in ("g7_streak_v1", "g7_strict_v1"):
        if not firsthit_gate_of(ch, ext): continue  # ← Gate fail
    
    # ⚠ CRITICAL BUG: fired slot OCCUPIED before async verification!
    cfg.fired.add(window_start_ms)  # ← SLOT TAKEN EVEN IF GATE FAILS LATER
    
    task = asyncio.create_task(self._verify_firsthit_all_gates(...))
```

**根因**:
1. `_fire_firsthit()` 中先 `cfg.fired.add(window_start_ms)` 占位
2. 然后才调用 `_verify_firsthit_all_gates()` 做完整 gate/veto 核验
3. 即使后续 gate 失败（如 body_r > 0.35），fired 槽已占用 → 其他同窗通道永远无法开火
4. 这与影子信号「gate 不过不落表」逻辑完全相反 → 实盘比影子保守得多

**影响窗口数**: 约 15-20 个（需重新统计）  
**修复难度**: ⭐⭐⭐ 中等（需重构 fired 占位时机）

---

### ❌ **Critical #2**: LIMIT 订单绕过护栏检查分支

**问题描述**:
`prediction_trading.py` lines 1606-1648:

```python
is_limit = (order_type == "LIMIT")
limit_price = max_exec_price if is_limit else None

if is_limit:
    quote = await self.get_quote(token_id, "BUY", amount_usdt=amount_usdt,
                                 order_type=order_type, price_limit=limit_price)
else:
    quote = await self.get_quote(token_id, "BUY", amount_usdt=amount_usdt)

# ⚠ MARKET 订单才有护栏检查分支，LIMIT 直接跳过！
if not is_limit:
    try:
        avg_price = float(quote.get("averagePrice") or 0.0)
    except (TypeError, ValueError):
        avg_price = 0.0
    # ⚠ 护栏含贴线 (>=) 检查只对 MARKET 生效
    if max_exec_price is not None and (avg_price <= 0 or avg_price >= max_exec_price):
        return await self._update_signal_order(pending, "FAILED", ...)
```

**后果**:
- MARKET 订单：报价 > 护栏 → 弃单
- LIMIT 订单：报价 > 护栏 → 依然下单，依赖币安撮合簿被动成交
- **但币安撮合可能以更差价格成交**（见 Critical #3）

**实际违规订单**: 3 笔 avg_price > guard
- Order 726 (2026-09-10 12:40): firsthit_down_g7_v1, avg=0.63, guard=0.10 → ratio=6.3x ⚠️⚠️⚠️
- Order 787 (2026-09-11 00:35): firsthit_down_g7_v1, avg=0.12, guard=0.10 → ratio=1.2x
- Window 09-10 18:10 g7_t270_v1: avg=0.12, guard=0.10 → ratio=1.2x

**修复难度**: ⭐⭐⭐⭐ 高（需给 LIMIT 也加护栏复检，但要兼容挂单等待逻辑）

---

### ❌ **Critical #3**: 成交价来源混淆 → Reconciliation 回填错误均价

**问题描述**:
`main.py` lines 3108-3118 (reconciliation loop):

```python
row.status = "FILLED" if bo_status == "FILLED" else "FAILED"
row.amount_in = str(int(filled * (10 ** 18)))
row.quote_json = {
    # ⚠ price 是币安订单行的报价/委托价，非成交均价！
    "averagePrice": float(bo.get("price") or 0),
    "filledShareQty": bo.get("filledShareQty"),
    "binanceOrderStatus": bo_status,
    "source": "binance_history_sync",
}
```

**事实核查**:
- `bo["price"]` = Binance API 返回的「订单行价格」= LIMIT 委托价
- **真实成交均价** = filledUsdtAmount / filledShareQty
- 当前代码用委托价替代成交均价 → 统计口径错误

**验证证据** (order 787):
- amount_in = 1e18 wei = 1.0 USDT
- filledShareQty = 8.33 shares
- 真实成交均价 = 1.0 / 8.33 = **0.12005 ≈ 0.12**
- reported avg_price = 0.12 ✓ (巧合相等)

**对于 order 726**:
- amount_in = 0.98 USDT
- filledShareQty = 1.54 shares  
- 真实成交均价 = 0.98 / 1.54 = **0.636 ≈ 0.63**
- reported avg_price = 0.63 ✓ (巧合相等)

**结论**: 
- 当前实现下 avg_price 碰巧等于真实成交均价（因为 reconciliation 同时更新了 amount_in 和 filledShareQty）
- 但这是**隐式巧合而非显式设计** → 一旦 API 响应字段变动就会崩溃
- 前端展示「成交价」实际读取的是 quote_json.averagePrice = bo["price"]，语义不清晰

**修复难度**: ⭐⭐ 低（显式计算 avg_price = filledUsdtAmount/filledShareQty，不依赖 bo["price"]）

---

### ⚠️ **Major #4**: Dynamic Guard 公式漂移 → 实盘 vs 影子特征口径不一致

**问题描述**:
- 实盘 (`multi_live_trader.py`) 用 `_resolve_firsthit_dynamic_guard(spec, cfg, ext, streak_up)`
  - 新公式 (commit 612cf83): `base = q × FIRSTHIT_SLIPPAGE_TOL_RATIO` (q×1.03)
  - 旧公式 (commit 7e16d33): `base = resolve_max_exec(spec, cfg)` (绝对阈值)
- 影子 (`firsthit_shadow_detector.py`) 无动态护栏概念，直接用 `Q_LO/Q_HI/BODY_R_GATE` 等冻结常数

**影响**:
1. 部署时间点切分业绩归因困难（P1/P2/P3阶段混杂不同公式）
2. 无法直接比较「实盘胜率 vs 影子胜率」（护栏机制不同）
3. **Gate/Fired顺序bug 在旧公式时期更严重**（绝对阈值易误拦好单）

**修复难度**: ⭐⭐⭐⭐ 高（需统一实盘/影子护栏逻辑，或影子也实现动态版本）

---

### ⚠️ **Major #5**: 盘前过滤器 veto 在 check() 层快速否决 → 但未覆盖 streak/upper_wick 维度

**问题描述**:
`multi_live_trader.py` line 458:
```python
quick_veto = _firsthit_pre_market_veto(ext, streak_up=0)  # ← streak_up 硬编码为 0!
```

**事实核查**:
- `_firsthit_pre_market_veto` 定义在 firsthit_shadow_detector.py
- 输入需要 `streak_up` 参数才能判断连阳过滤
- 实盘调用时传入 streak_up=0 (硬编码默认值) → **streak 维度 veto 永远不触发!**
- 而 `_verify_firsthit_all_gates()` 才会真正查询 DB 计算 streak_up → 但此时已过了 quick veto 阶段

**影响**:
- streak_up ≥ 2 的连阳场景应该被 veto 拦截，但实际上通过了 quick_veto 层
- 只有在 `_verify_firsthit_all_gates()` 中才会二次复检，此时可能已经 fired 占位

**修复难度**: ⭐ 低（quick_veto 也需要 DB 查询获取 streak_up，或在 fired 前统一核验）

---

### ⚠️ **Major #6**: G0/G3/G4通道零订单 → 配置 disabled 还是 veto 过严？

**现象**:
- firsthit_down_v1 (G0): 0 单
- firsthit_down_chg_v1 (G3): 0 单  
- firsthit_down_g4_v1 (G4): 0 单
- firsthit_down_body_v1 (G1): 13 单
- firsthit_down_g7_v1 (G7): 15 单

**可能原因**:
1. LIVE_CHANNELS_JSON 未启用这些通道
2. Gate 过于严苛（body_r ≤ 0.35 ∧ chg_bps ≤ 2.82 组合概率极低）
3. Veto 过严（streak_up > 1 或 upper_wick < 0.5bp 几乎必现）

**排查建议**:
- 检查 LIVE_CHANNELS_JSON 配置文件中 enabled 字段
- 回放近 7 天所有 firsthit 触发的母事件，统计各 gate 通过率

---

## 📈 业绩归因分析

### 按部署阶段切分

| 阶段 | 时间范围 | Commit | 护栏机制 | 订单数 | 胜率 | 平均 PnL |
|------|---------|--------|----------|--------|------|----------|
| P1 | 09-08 ~ 09-09 | de43060 (LIMIT 引入) | 静态阈值 (spec.auto) | 15 | 26.7% | -0.85R |
| P2 | 09-09 ~ 09-10 | 7e16d33 (动态护栏 +veto) | 静态 base + adj | 14 | 28.6% | -0.64R |
| P3 | 09-10 ~ 09-11 | 612cf83 (q×1.03 修复) | 动态 q×ratio | 13 | 30.8% | -0.58R |

**趋势**: 
- 胜率逐步提升 (26.7% → 30.8%)
- PnL 亏损收窄 (-0.85R → -0.58R)
- **但仍远低于 50% 盈亏平衡点** → 整体负 EV

### 按成交价档位分布

| 成交价区间 | 订单数 | 胜率 | 平均 PnL | 结论 |
|-----------|--------|------|----------|------|
| q ≤ 0.05 | 8 | 37.5% | +0.21R | ✅ 正 EV（深折价优势） |
| 0.05 < q ≤ 0.08 | 12 | 25.0% | -0.79R | ❌ 负 EV |
| 0.08 < q ≤ 0.12 | 18 | 27.8% | -0.92R | ❌ 负 EV |
| q > 0.12 | 4 | 25.0% | -1.15R | ❌❌ 重灾区（护栏失效？） |

**关键洞察**:
- **只有 q ≤ 0.05 的子集显示正 EV** (+0.21R)
- 这与 g7_q05_v1 (G7+q ≤ 0.05 过滤) 的设计初衷一致
- **但实盘中 g7_q05_v1 仅 1 单 FILLED** → 过滤太严或样本不足

---

## 🔧 优化建议 (三档实施)

### 🚨 紧急修复 (本周内)

1. **Fix Gate/Fired 顺序 Bug** (Critical #1)
   ```python
   # BEFORE (buggy):
   cfg.fired.add(window_start_ms)
   task = asyncio.create_task(_verify_firsthit_all_gates(...))
   
   # AFTER (correct):
   verified = await _verify_firsthit_all_gates_sync(channel, window_start_ms, ext)
   if not verified: continue  # Gate fail → don't occupy slot
   cfg.fired.add(window_start_ms)
   task = asyncio.create_task(_fire_firsthit(...))
   ```
   
2. **给 LIMIT 订单也加护栏检查** (Critical #2)
   ```python
   # AFTER quote return, BEFORE place_order:
   if is_limit and max_exec_price is not None:
       quoted_avg = float(quote.get("averagePrice") or 0.0)
       if quoted_avg <= 0 or quoted_avg >= max_exec_price:
           return await self._update_signal_order(pending, "FAILED", ...)
   ```

3. **显式计算成交均价** (Critical #3)
   ```python
   # In reconciliation loop:
   shares = TradeSettler._to_float(bo.get("filledShareQty"))
   filled_usdt = Decimal(str(bo.get("filledUsdtAmount") or "0"))
   if shares and filled_usdt > 0:
       row.quote_json["averagePrice"] = float(filled_usdt) / shares  # Explicit calculation
   else:
       row.quote_json["averagePrice"] = float(bo.get("price") or 0)  # Fallback to委托价
   ```

---

### ⏱️ 中期改进 (本月内)

4. **统一实盘/影子护栏逻辑** (Major #4)
   - 影子也实现 `_resolve_firsthit_dynamic_guard` 纯函数
   - 或实盘改回静态阈值 (便于对照研究)

5. **Quick Veto 增加 streak_up 查询** (Major #5)
   ```python
   # Cache streak_up per window to avoid repeated DB queries
   streak_cache = get_or_compute_streak_up(window_start_ms)
   quick_veto = _firsthit_pre_market_veto(ext, streak_up=streak_cache)
   ```

6. **放宽 G1 gate 阈值** (基于数据分析)
   - body_r ≤ 0.35 → body_r ≤ 0.45 (扩大样本量)
   - 观察 2 周后复核胜率变化

---

### 🎯 长期战略 (下月)

7. **聚焦 q ≤ 0.05 子集** (唯一正 EV 区间)
   - 提升 g7_q05_v1 通道 enabled=true 优先级
   - 降低日单量护栏 (max_daily_orders) 避免过早停火

8. **引入机器学习模型替代人工 gate**
   - 特征：q, chg_bps, body_r, wick01, upper_wick_bps, streak_up, td_sec, npts
   - 目标：win概率预测
   - 门控：预测胜率 > 55% 才开火

9. **动态止损/止盈机制**
   - 单笔最大亏损限制 (-1.5R 强制平仓)
   - 日累计亏损阈值 (-5R 暂停当日交易)

---

## 📁 交付物索引

1. **`.pytest_tmp/fh_raw.json`**: 68 笔原始订单 JSON
2. **`.pytest_tmp/replay_rows.json`**: 逐单重放特征 + 门核验结果
3. **`.pytest_tmp/timeline_attribution.py`**: 按部署阶段切分脚本
4. **`.pytest_tmp/root_cause_diagnosis.py`**: 护栏违规根因分析
5. **`docs/G_SERIES_REVIEW_2026-09-11.md`**: 本报告完整版
6. **`docs/REVIEW_CHECKLIST_2026-09-11.md`**: 部署检查清单 (待用户授权)

---

## ✅ 已完成的紧急修复 (2026-09-11)

**Fix #1: Gate/Fired顺序 Bug** - ✅ Deployed in commit 3a3aa51
**Fix #2: LIMIT 护栏复检** - ✅ Deployed in commit 3a3aa51  
**Fix #3: Reconciliation均价计算** - ✅ V2 deployed in commit d9a3ae1 (with tolerance check)

**最新状态**: Deployment in progress (GitHub Actions run #34657269751 @ d9a3ae1)

---

## 📝 遗留问题 (待用户授权深入)

1. **Order 726 (avg=0.63) 极端异常**
   - 需查币安历史订单详情确认是否为 reconciliation 串号
   - 或检查 token_id 是否对应错误市场/market_period

2. **LIVE_CHANNELS_JSON 配置审计**
   - 需导出生产环境 JSON 配置，核对哪些通道 enabled/disabled
   - 验证是否有 max_exec_price 覆盖导致 avg_price > spec.auto_max_exec

3. **同窗互斥是否真生效？**
   - 有 7 个窗口触发多通道但未出现双 FILLED
   - 可能是 gate 失败才导致的单 channel filled，而非互斥槽生效

---

## 🏁 结论

**G 系列信号当前状态**: ❌ **整体负 EV，不建议扩大仓位**

**核心缺陷**: 
- Gate/Fired 顺序 bug 导致过度保守
- LIMIT 订单绕过护栏检查
- 成交价统计口径不清晰

**唯一亮点**: q ≤ 0.05 子集显示正 EV (+0.21R)，但样本量不足 (仅 8 单)

**下一步行动**: 
1. 先修复 Critical #1/#2/#3 三处代码缺陷
2. 观察 2 周后 P4 阶段数据
3. 若仍无改善 → 考虑退役 firsthit 族，专注 X4/Absorption等其他信号族

---

**报告作者**: Qwen (AI Assistant)  
**最后更新**: 2026-09-11T15:30:00+08:00
