"""K 线族影子检测器共用的「目标窗口入场报价快照」helper（口径单一事实源）。

背景：KREV / 反转(P1/P2) / nextbar 三族影子过去只按次根 K 线涨跌结算、不记报价，
故面板 EV/累计 EV 恒空。本 helper 让三族在**信号落库时刻**（信号根收盘 = 目标窗
开盘后的首次轮询，~0~60s 内）从实时市场报价缓存快照目标窗口的 UP/DOWN 真实报价，
使聚合层能按既有 `_shadow_realized_ev` 口径（赢 0.98/q−1 / 输 −1）现算真实 EV。

入场时点口径（与 HM 影子 _finalize_entry 的报价守卫同源，禁止另立第二套）：
    1) 窗口对齐：cache["start_date"] == target_bar_start（缓存跟踪的正是目标窗口市场，
       该市场即以目标根涨跌结算——押注的正确标的）；
    2) 近开盘：-ENTRY_CLOCK_SKEW_TOLERANCE_MS <= updated_ts - target_bar_start <=
       ENTRY_MAX_OFFSET_MS（报价取自目标窗开盘后近端，非盘中深处；下沿含小时钟偏差
       容忍，见常量注释；冷启动回补的历史信号因缓存已切到当前窗 → 对齐失败 → None）；
    3) 报价合法：up/down 价均在 (0,1)。
任一守卫不满足 → 返回全 None（保守：无对齐报价 → 该笔 EV 不计，与既有纯 K 线口径一致，
前端显示 '—'）。offset 可由 entry_quote_ts - target_bar_start 审计。

只读共享缓存（main._pm_15m_latest / main._pm_market_info）：单写者事件循环，读者容忍
毫秒级旧值，无需加锁（与 HM 读 _pm_15m_latest 同构）。
"""
from __future__ import annotations

# 入场报价距目标窗开盘的最大偏移（近开盘守卫上沿）：轮询 60s + 缓存刷新 ≤15s，正常 ≤~75s，
# 取 120s 宽松上限，仅拒绝真正错位/陈旧（缓存停在旧窗或深处）的报价。
ENTRY_MAX_OFFSET_MS = 120_000

# 近开盘守卫下沿的时钟偏差容忍：updated_ts 取本地系统时钟（main.aligned_ts = time.time()），
# target_bar_start 取币安服务器时（K 线 open_time），二者跨时钟域；本地钟略落后服务器时，
# 目标窗近开盘报价的 offset 可能被算成轻微负值。守卫 1（start_date == target_bar_start）已
# 确保缓存跟踪的正是目标窗（真正的防错窗/防未来函数闸门），故对 offset 下沿放宽 2s
# 容忍，仅吸收亚秒~秒级 NTP 偏差，避免误拒合法的近开盘报价（超出容忍仍拒，防真正陈旧）。
ENTRY_CLOCK_SKEW_TOLERANCE_MS = 2_000


def snapshot_entry_quote(
    cache: dict | None, target_bar_start: int
) -> tuple[float | None, float | None, int | None]:
    """从实时报价缓存快照目标窗入场价。

    返回 (up_price, down_price, quote_ts)；任一守卫不满足 → (None, None, None)。
    up_price = 押 UP 的入场价、down_price = 押 DOWN 的入场价（与各检测器落库、
    聚合层 `_shadow_realized_ev` 的 q 口径一致：按 direction 取对应侧）。
    """
    if not cache:
        return None, None, None
    start = cache.get("start_date")
    up = cache.get("up_price")
    down = cache.get("down_price")
    ts = cache.get("updated_ts")
    if start is None or up is None or down is None or ts is None:
        return None, None, None
    try:
        start_i, ts_i = int(start), int(ts)
        up_f, down_f = float(up), float(down)
    except (TypeError, ValueError):
        return None, None, None
    # 守卫 1：窗口对齐（缓存跟踪的必须是目标窗口市场）
    if start_i != int(target_bar_start):
        return None, None, None
    # 守卫 3：报价合法（二元市场 token 价开区间 (0,1)）
    if not (0.0 < up_f < 1.0) or not (0.0 < down_f < 1.0):
        return None, None, None
    # 守卫 2：近开盘（报价取自目标窗开盘后近端；下沿含时钟偏差容忍，见常量注释）
    offset = ts_i - int(target_bar_start)
    if offset < -ENTRY_CLOCK_SKEW_TOLERANCE_MS or offset > ENTRY_MAX_OFFSET_MS:
        return None, None, None
    return up_f, down_f, ts_i
