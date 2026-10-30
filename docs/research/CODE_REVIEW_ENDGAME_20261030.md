# 代码审核：尾盘买热门（endgame 族）影子 + 实盘通道

审核对象：`endgame_fav_15m_v1` 落地改动（2026-10-30）
审核方式：读 diff + 跑全量测试 + 逐条对照上游契约（excur 族、`_hook_gate`、`kline_shadow_signals` 语义）
结论：**可上线（影子默认开、实盘默认关）；发现 1 个真 bug（已修）、5 项需知悉的限制**

---

## 一、改动清单

| 文件 | 内容 | 行数 |
|---|---|---|
| `services/endgame_shadow_detector.py`（新增） | 检测器：触发判定 / 落表 / 结算 / 实盘钩子 | 214 |
| `services/live_channels.py` | 注册 `endgame_fav_15m_v1`（family=endgame、护栏 0.86、10U、默认 OFF）+ 加入保留名单 + family 注释 | +10 |
| `services/multi_live_trader.py` | `on_endgame_signal` 钩子 + `_fire_endgame` 下单任务（15s 晚到硬门） | +71 |
| `main.py` | 全局实例、启动注入钩子、`_handle_15m_quote` 采样调用、stop | +24 |
| `frontend/src/App.tsx` | `SIGNAL_INFO` 中文名与说明（与后端同源） | +4 |
| `tests/test_endgame_shadow_detector.py`（新增） | 带边界 / 三道门禁 / 去重 / DOWN 侧 | 4 例 |
| `tests/test_multi_live_trader.py` | 注册默认关 / 钩子开关 / 晚到弃单；修 3 处计数断言 | 3 例 |

验证：`pytest tests/` **1687 passed / 15 skipped**；`npm run build` 通过；`npm run lint` 仅剩 2 条历史 warning（App.tsx:3752/3754，非本次改动）。

---

## 二、发现与处置

### P1（真 bug，已修）

**1. 去重标记写在 `_ensure_row` 内部，调用侧没有标记**
单测暴露：同一窗口连续两次采样会各落一行（`_ensure_row` 被替换/未来重构时，`observe_sample` 的去重依赖失效）。
→ 改为在 `observe_sample` 拿到 `row_id` 后立即 `_fired.add(window_start)`；`_ensure_row` 内保留（DB 已存在路径也要打标）。

**2. `entry_quote_ts` 用 `window_start + t_rel*1000` 反算**
`t_rel` 由 `ts_ms` 算出又反算回去，绕一圈且对 epoch 量级无必要。
→ 改为直接传 `ts_ms`（`_ensure_row` 签名一并调整）。

**3. 三处计数断言需同步**（新通道导致预期变化，属正常维护）
`test_parse_defaults_all_off`（10U 名单）、`test_channels_registry_shape`（35→36）、`test_status_shape`（78→79）。

### P2（已修）

**4. `ChannelSpec.family` 注释**未列 `endgame` → 已补，避免后续按注释枚举时漏族。
**5. `pick_favorite` 口径未说明** → 已注明「入参是展示价而非中间价」，见下 B1。

### 未修（需知悉，非阻断）

| # | 事项 | 影响 | 处置 |
|---|---|---|---|
| B1 | 触发用**币安展示价**（≈ask），回测触发用**中间价** ∈[0.65,0.85) | 影子触发分布比回测略窄（上边界更严）→ 前向胜率估计偏保守 | 不修正（无法从展示价还原 mid），已写进 docstring |
| B2 | 没有 `settings.endgame_shadow_enabled` 开关（excur 有） | 影子**无条件启动**；要临时停采只能改代码/停进程 | 用户明确要影子常开，保持；如需开关再加一行 |
| B3 | `_hook_gate` 先占位 `cfg.fired`，`_fire_endgame` 再判「剩余 ≤15s」弃单 | 晚到单会把这个窗口**烧掉且不重试** | 有意行为（晚到无边际，2s 延迟只剩 48%），日志有 `endgame_late_skip` 可审计 |
| B4 | 与其它通道**无互斥** | 同窗最多可能叠加 2–3 单敞口（`max_channels_per_window=3` 仍生效） | 先观察影子期同窗分布，再决定是否并入 `SAME_WINDOW_EXCLUSIVE` |
| B5 | `target_bar_start` = **本窗**（模型注释是「次根」） | 结算按本根收盘，语义正确；下游 `/api/...shadow` 把它当 `window_start` 展示，也对 | 已在代码注释显式说明；无下游按「次根」假设读取本版本 |

另已修一处防御性缺陷：`start()` 时若未注入 `collector`，原来会静默不结算（行永远 PENDING），
现在启动即 `logger.error`。

---

## 三、逐项契约核对

| 契约 | 核对结果 |
|---|---|
| 实盘默认关闭 | `default_enabled=False`；`parse_channel_config()` 实测 `enabled=False`（有断言） |
| 通道 ID 与影子版本一致 | 两侧同为 `endgame_fav_15m_v1`，订单 `signal_version` 直接对账 |
| `_hook_gate` 四道门 | stop / 注册表族匹配 / 启用 / 同窗去重，全部走公共门禁 |
| 结算口径与 cic/nextbar 一致 | `win = (outcome == direction) if outcome != "NOISE" else None` |
| 钱包/订单路径 | 复用 `_exec_with_exclusive` + `_prepare_runtime_execution`（日限、余额预检、评估链天然继承） |
| 护栏 | `auto_max_exec=0.86` = 带内上界 0.85 + 滑点余量；贴线弃单由 `prediction_trading` 保证 |
| 前端同步 | `SIGNAL_INFO` 已加，名称/说明与 `live_channels.py` 对齐 |
| 表与模型 | 复用 `kline_shadow_signals`（唯一键 `version+signal_bar_start` 天然防重） |

---

## 四、上线前检查清单

1. `alembic upgrade head` —— 无新表/新列（复用 `kline_shadow_signals`），**不需要迁移**。
2. 部署后确认日志：`endgame 尾盘买热门检测器启动` + `endgame 影子：未注入 collector` **不出现**。
3. 首日抽检 1–2 行：`SELECT version, direction, entry_up_price, entry_down_price, feature_snapshot->>'t_rel_s', status FROM kline_shadow_signals WHERE version='endgame_fav_15m_v1' ORDER BY id DESC LIMIT 5;`
   - 期望：`t_rel_s ∈ [855, 900)`、`fav_side` 与 `direction` 一致、`status` 15 分钟内转 `SETTLED`。
4. 实盘开关状态：`GET /api/live/channels` 里 `endgame_fav_15m_v1.enabled == false`。

---

## 五、影子期验收判据（预注册，先写死再跑）

**样本**：≥ 300 笔且 ≥ 30 天（按研究频率 13 笔/天，约 23 天满 300）。

| 指标 | 通过线 | 依据 |
|---|---|---|
| 胜率 | ≥ 78% | 回测 81.0%，留 3pp 给展示价口径与延迟 |
| 净 EV（按真实费率） | ≥ +1.5% | 回测 +5.2%（Δ=0）/ +3.3%（Δ=2s），留一半余量 |
| 按天聚类 t | ≥ 2.0 | 回测 2.97 |
| 月度一致性 | ≥ 3/4 个月为正 | 回测 4/4 |
| 成交价 − 触发价 | 中位 ≤ +0.01 | 一档价可成交（历史实盘中位 0.000） |
| 延迟（下单时延日志） | p50 ≤ 1.2s | 回测 1.8s 时剩 52%，压到 1.2s 可留 70% |

**否决条件（任一命中即回滚/不上实盘）**：胜率 < 74%、或净 EV < 0、或月度 < 2/4 为正。

---

## 六、剩余风险

1. **本质是速度套利**：edge 对延迟极敏感（1s 剥 32%、2s 剥 52%）。若生产延迟压不到 1.2s，前向 EV 会明显低于回测。
2. **样本内**：111 天、1,450 笔的回测结论，7 月是平的（−0.4%），收益集中在 8–9 月。
3. **容量小**：单笔 10U 时日均约 3.2U；放大到 50U 日收益也只有 7.1U（走单吃掉边际）。
4. **未验证成交**：影子只记报价，不校验真实可成交量；实盘开闸前建议先跑 20 笔 2U 小额。
