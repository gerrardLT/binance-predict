# 迭代记录

新功能 / bug 修复 / 功能改造，改完在**最上方**追加一条。格式：问题 / 改动 / 验证 / 遗留，每条 5-8 行。
豁免：纯 UI 微调、纯文档、一次性脚本。触碰交易语义 / 资金 / DB / 部署 / 新功能则必记。

## 2026-09-13 Rev2 双周期孕线精选通道 `pending commit`

**问题** 15m Rev2 的反方向短影 25% 口径过宽，且 5m 精选 HM 尚未接入影子与实盘链路。
**改动** 新增 `hm_inside_5m_v2`：短上影≤10%、主下影[75%,90%)；15m HM/IH 短影统一收紧至≤10%。
**改动** 5m/15m 检测器独立轮询、报价缓存、结算与实盘开火；三个 LIMIT 通道默认护栏保持 0.30，支持现有热覆盖下调。
**验证** 已增加 10%/75%/90% 几何边界、5m 报价快照、实盘透传与结算路由回归测试；专项测试通过。
**遗留** 5m 候选真实报价样本仍较少，先按影子与小额前向观察；本次不修改既有用户热配置和 LIMIT GTC 语义。

---

## 2026-09-11 X4_v2/v3 入场价白名单修复（死区止损） `pending commit`

**问题** 实盘数据审计发现 `[0.10, 0.30)` 价格带为"死区"：x4_v2 (n=28, pnl=-42.22 USDT), x4_v3 (n=4, pnl=-3.77 USDT) 持续亏损；`[0.30, 0.40)` 利润带表现优异 (n=26, pnl=+26.60 USDT, WR=61.5%)。根因：v2 的 `entry_band_whitelist=None`（裸奔），v3 虽有白名单但覆盖 `[0.0, 0.2)` 导致仍买入死区左半部。

**改动** 
1. `live_channels.py`: 
   - x4_v2: `None` → `((0.0, 0.1), (0.3, 0.4))`（挖掉 `[0.1, 0.3)` 死区）
   - x4_v3: `((0.0, 0.2), (0.3, 0.4))` → `((0.0, 0.1), (0.3, 0.4))`（统一口径，停止 v3 在 `[0.1, 0.2)` 流血）
2. `misalignment_detector.py`: 仅注释修正（±10min 容差导致实际回看 50min），零行为改变，通过 AST 验证等价性
3. `test_multi_live_trader.py`: 
   - `test_x4_poll_passes_v3_whitelist` 断言值同步更新
   - `_V3_BANDS` 常量同步更新
   - `test_signal_trade_whitelist_band_matrix` parametrize 用例矩阵重构（旧 [0,0.2)→新 [0,0.1)，增加死区边界点）
   - `test_signal_trade_fok_retry_whitelist_rechecks_retry_quote` 首报价格从 0.19→0.08（确保在死区外）

**验证** 
- AST 等价性验证：`misalignment_detector.py` 修改前后语法树完全一致（纯注释）
- 全量测试 **181 PASS**（`test_multi_live_trader.py::test_x4_poll_passes_v3_whitelist` 从 FAIL 转为 PASS）
- 白名单边界测试覆盖：avg=0.05/0.09 (带内真), 0.10/0.19/0.25/0.40/0.49 (带外假), 0.30/0.39 (利润带真)
- FOK 重试复检逻辑：首报 0.08 放行成交，重试报价 0.22 死区拦截，不产生第二单

**遗留**
- DB 覆盖层配置：生产环境 `live_channel_overrides.max_exec_price` 仍为 x4_v2=0.35、x4_v3=0.32（需手动调 >0.40 如 0.45），否则 `[0.30, 0.40)` 利润带仍被护栏先弃（白名单与护栏两道独立防线，max_exec_price 必须 >0.40）
- 影子观测周期：建议 7 天观察期内监控死区拦截率提升比例、`[0.30, 0.40)` 盈利单捕获率是否显著改善（预估 n≈20 新信号可统计）
- 代码注释同步：`_past_1h_chg_pct` docstring 已澄清净位移 ≠ 波动率且保留 RV 会拦最优子集（低 NET+ 高 RV 象限 EV+0.84），无需阈值敏感性调整（0.3~0.8% 区间平坦）

---

## 2026-09-11 firsthit G7 护栏逆选择 Bug 修复 `pending commit`

**问题** 静态绝对阈值护拦（base=0.10~0.12）造成严重逆向选择：反转成功窗（DOWN 报价从 0.08 弹升 0.36-0.92）被拦截 → 17 笔 FAILED 弃单；单边暴拉必死单（ DOWN 报价暴跌至 0.01-0.07）成交 → 14 笔 FILLED 单胜率 0%。生产数据铁证：id=647 FAILED(averagePrice=0.65 ≥ 0.12)，id=572 FILLED(averagePrice=0.01 → 全损)

**改动** `_resolve_firsthit_dynamic_guard()` 重构：护拦从「绝对阈值」改为「相对滑点比率」（q × FIRSTHIT_SLIPPAGE_TOL_RATIO=1.03，3% 容忍度）；新增常量 `FIRSTHIT_SLIPPAGE_TOL_RATIO = 1.03`；保留微调收紧逻辑（streak==1/upper_wick<1bp → adj-=0.01）但基准已从绝对改为相对；clamp 下限 0.05 防止低报价区无意义护拦

**验证** 新增 `tests/test_firsthit_g7_fix.py::test_resolve_dynamic_guard_allows_reversal_winners_vs_static_guard_blocks_them`（TDD RED→GREEN 完整流程，验证 q=0.08 → guard=0.0824；doom order q=0.08+upper_wick=0.3 → tightened guard=0.0724）；修复 `test_multi_live_trader.py` 三个测试（期望 max_exec_price 从静态 0.08/0.10 改为动态 0.0721/0.05）；全量测试 179 PASS 无回归

**坑** 旧逻辑护拦=max_exec_price(用户配置或 auto_max_exec)，新逻辑护拦=q×1.03（触发价相对滑点）→ **max_exec_price 字段不再用于 firsthit 族下单护拦**（仅作为历史参考），实盘开火时 dyn_guard 由函数计算而非配置读取

---

## 2026-09-11 firsthit G7 早期首触死锁 Bug 修复 `pending commit`

**问题** 离线回测显示 G7/P≈20%/EV+2.34，但线上 31 笔订单全损（14 笔成交单 P=0%），根因：**提取特征时 npts 只计算 t≤trigger_ts 的 BTC 采样**→ 若首触发生在窗口前 105s（npts=2），后续所有循环都因路径太稀疏被永久抑制，导致实盘单全部被逼到窗口最后 14~30 秒（t=270~286s），根本没时间反转

**改动** `firsthit_shadow_detector.py:extract_firsthit_features` 关键修复：`max_trigger_ts` 存在时使用 **截至当前时刻的最新历史** 而非仅 ≤trigger_ts 的旧点 → 允许早期首触窗口在 t≥105s 后当路径成熟时正常开火；同时保留 `trigger_ts` 钉死在首个入区点（正确语义）

**验证** 新增 `tests/test_firsthit_g7_fix.py::test_extract_firsthit_features_early_touch_should_wait_for_npts_to_mature`（TDD 流程完成：RED 失败确认 bug，GREEN 代码最小修改通过，所有 `test_firsthit_shadow_detector.py` 18 用例 PASS）；模拟 t=30s 首触 + 150s 成熟窗口成功提取 (npts=11)

**坑** `MAX_TRIGGER_TS` 用于实时重放防未来函数：只看≤max 的点数找首触，再基于该最大时间戳计算 npts（允许窗口内后期成熟），而非 trigger_ts 的旧采样——这才是影子/实盘真正的口径一致点

---

## 2026-09-10 15m 影子族结算分流修复  `4a2b3a1`

**问题** 15m 无 `scene_signal_id` 的 FILLED 单被误判 EXPIRED/pnl=0；`id=466` 真实 −1.00 记为 +0.00（波及 s2_cond t4/t5d、hm/ih_inside_15m_v2 四通道）
**改动** `trade_settler` 新增 `_settle_kline_shadow_row`，结算源两路改三路（按**有无 scene_signal_id** 分流，不硬编码通道名 → 新增 15m 影子族零改动接入）；回读 `KlineShadowSignal`(version + target_bar_start)；pnl 口径与其余两路同源（赢 shares−amount，输 −amount，NOISE win=None/pnl=0）
**验证** `tests/test_trade_settler.py` 新增 8 用例（win/lose/noise/缺失重试/缺失超期/PENDING 重试/幂等守卫/nextbar 方向）+ `test_15m_with_scene_signal_id_uses_fake_breakout` 防回归；历史误判行走 workflow `fix-s2-settlement-data-repair.yml`（默认 dry-run，只命中 `settled_at IS NOT NULL AND outcome='EXPIRED' AND pnl=0`，重置为 NULL 由 60s 轮询自动重算）
**坑** 只认 `settle_outcome` 不认 `status`——检测器把 NOISE 行 status 记为 EXPIRED，按 status 判会把平盘当「无结算源」
**遗留** 历史损失量化未做（需查 prod 全量 s2_cond/hm/ih FILLED 单重算）；hm/ih 两通道前向表现须修复后才能评估

---

## 2026-09-10 firsthit 族盘前过滤器 + 动态护栏  `7e16d33`

**问题** 固定护栏造成逆向选择：「反弹窗被 0.08 一刀切切除、暴拉窗低价全成交」→ 成交集合系统性劣化
**改动** G1(`firsthit_down_body_v1`) 护栏 **0.12→0.15**（唯一前向正 EV 版本：n=30 wr13.3% cum_ev+28.8；影子端 15/22 赢单落在 q∈(0.08,0.10]）；新增盘前 veto（连阳>1 / \|chg\|>50bp / 上影<0.5bp 跳过，不占 fired）；动态护栏 `base+adj`（adj≤0 只收紧不放松，clamp[0.05,base]）；firsthit 全通道统一走 `_verify_firsthit_all_gates` 异步核验（原仅 g7_streak/g7_strict）
**验证** ⚠ **两个新纯函数零直接单测**（`grep _firsthit_pre_market_veto\|_resolve_firsthit_dynamic_guard tests/` 无匹配），只适配了 6 个既有用例的曲线构造（单调上升 → 冲高回落留上影，否则 upper_wick=0 被 veto 拦）；缺贴线正反例与 `adj≤0` 不变量守护
**坑** 浮点贴线：`(100.5/100-1)*1e4 = 50.00000000000004` > 50 会**误拦**，测试改用 100.45(+45bp) 避开
**遗留** 三个 veto 阈值与两个 −0.01 调整量均为「保守先验，未经离线回测」，需前向观测裁决；`FIRSTHIT_*` 常量硬编码不可热调（护栏 base 却可经 `LIVE_CHANNELS_JSON` 热调，不对称）
