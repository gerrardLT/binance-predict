"""多通道实盘注册表（MultiLiveTrader 的静态描述层 + 配置解析，纯函数无 DB）。

通道 ID 与影子信号版本名对齐（订单 signal_version 直接用通道名，对账/统计天然一致）。
三族触发机制（由 MultiLiveTrader 分别驱动，见 multi_live_trader.py）：
- quote_edge：5m 采样循环喂价 → 窗内报价区间命中（v2 附加 BTC 门禁，实时喂价解锁；
  v3 环境门禁版再叠加前窗 DOWN/距日高回落，异步 DB 核验后下单）；
- x4：轮询 misalignment_signals PENDING → 次窗 +150s 决策点下单；
- scene：fake_breakout_detector fire 钩子 → 次周期开盘下单（15m 市场）。

护栏数值依据：盈亏平衡入场价 entry* = wr×(1−FEE)（干净口径历史胜率）：
S1 wr64.4%→0.63 / S5 78.5%→0.77 / S2 53.6%→0.525 / S4 55.4%→0.54 /
x4_v1 41.2%→0.40 / x4_v2 45.3%→0.44，护栏设在平衡价附近或略下方。
S2 真实 UP 报价常在 0.79+（跌态无折扣），护栏 0.55 会保护性弃单——正确行为（EV 保护）。
所有护栏可被 LIVE_CHANNELS_JSON 按通道覆盖。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from loguru import logger

from binance_predict.config.settings import settings

from .quote_edge_detector import QUOTE_EDGE_RULES

MAX_ORDER_AMOUNT_USDT = 50.0    # 单笔金额硬上限（配置误写拒绝启动，不靠自律）
MAX_DAILY_ORDERS_CAP = 500      # 日限硬上限（防配置误写 ×100 敞口失控）
_QE_GUARD_MARGIN = 0.03         # quote_edge 族自动护栏 = 入场区间上界 + 此余量（滑点容忍）


@dataclass(frozen=True)
class ChannelSpec:
    """通道静态描述（冻结，运行时不可变）。"""

    channel: str
    family: str                   # quote_edge | x4 | scene
    market_period: str            # 5m | 15m
    direction: str                # 典型下单方向（scene 族实际由信号 side 决定）
    auto_max_exec: float          # 默认执行价护栏（可被配置覆盖）
    display_name: str
    v2_guard: str | None = None   # quote_edge v2 门禁模式：min_drop | max_rise | None
    v3_env: bool = False          # quote_edge v3 环境门禁（前窗DOWN [+距日高回落]，异步核验）
    regime_gate: bool = False     # quote_edge v4 regime 门禁（ret24≤阈值，K 线异步核验）
    streak_gate: bool = False     # quote_edge v3 非连涨门禁（末收15m，K 线异步核验）


def _qe_guard(version: str) -> float:
    """quote_edge 族自动护栏：区间上界 + 0.03（与旧 QuoteEdgeLiveTrader 同推导）。"""
    return round(QUOTE_EDGE_RULES[version][3] + _QE_GUARD_MARGIN, 4)


LIVE_CHANNELS: dict[str, ChannelSpec] = {
    # --- quote_edge 族（5m 窗内报价触发，区间引用 QUOTE_EDGE_RULES 冻结口径）---
    # v2 门禁版：contrarian v1 冻结区间 + BTC 门禁（chg<+0.10%，阈值引用 V2_PRICE_GUARDS）
    # ⚠ 区间/护栏仍派生自已退役的 quote_contrarian_v1 冻结条目（QUOTE_EDGE_RULES
    # 是回测口径事实源，整体保留）；退役只停「v1 自身作为通道/影子版本」。
    "quote_contrarian_v2": ChannelSpec(
        "quote_contrarian_v2", "quote_edge", "5m", "DOWN", _qe_guard("quote_contrarian_v1"),
        "报价反向·门禁版", v2_guard="max_rise",
    ),
    # --- x4 族（影子 PENDING → 次窗 +150s 决策点，入场价历史偏低）---
    "x4_v2": ChannelSpec("x4_v2", "x4", "5m", "DOWN", 0.50, "情绪错位·平静市门禁版"),
    # --- 场景族（15m 市场次周期开盘入场；S5 为 +5min 确认入场）---
    # S1 护栏 0.70：2026-08-30 盘口数据校准（原 0.60 拍脑袋值）——
    # 41 个已结算信号中入场价 >=0.60 的 8 个胜率 75%（> 放行区间 61%），
    # 开盘 DOWN 贵=回落已在发生（确认而非警告），0.60 误拦正收益单。
    "scene_bull_exhaust": ChannelSpec(
        "scene_bull_exhaust", "scene", "15m", "DOWN", 0.70, "场景S1 多头耗尽（押DOWN）",
    ),
    "scene_bull_exhaust_confirm": ChannelSpec(
        "scene_bull_exhaust_confirm", "scene", "15m", "DOWN", 0.75, "场景S5 确认入场（押DOWN）",
    ),
    # S2 护栏 0.65：2026-08-30 盘口数据校准（与 S1 同病）——47% 信号被旧 0.55 拦截，
    # 被拦段 8 单胜率 88%、均值EV +0.39（与放行段同样赚钱）；开盘 UP 贵=反弹已启动。
    "scene_bear_exhaust": ChannelSpec(
        "scene_bear_exhaust", "scene", "15m", "UP", 0.65, "场景S2 空头耗尽（押UP）",
    ),
    "scene_momentum_fade": ChannelSpec(
        "scene_momentum_fade", "scene", "15m", "DOWN", 0.55, "场景S4 动量衰竭（押DOWN）",
    ),
    # S5 深档（2026-09-03 影子 → 实盘小金额前向验证）：S5 确认钩子 z5≤−20bp 子集，
    # 通道名与 fake_breakout_detector.S5_DEEP_VERSION 同名（订单 signal_version 与
    # 影子版本对账对齐）；+5min 确认即下单（15m 市场）。护栏 0.88：盈亏平衡入场价
    # = 回测 91.3%×0.98≈0.895 略下方；深档 EV 偏乐观含机械成分，小金额前向验证。
    # 与 S5 确认通道同窗互斥（SAME_WINDOW_EXCLUSIVE，至多一单成交）。
    "s5_deep_z20_v1": ChannelSpec(
        "s5_deep_z20_v1", "scene", "15m", "DOWN", 0.88, "S5深档·深回落门禁版",
    ),
}


# ---------------------------------------------------------------------------
# 已退役通道（2026-09-04 用户拍板「信号直接不要了」）：不在 LIVE_CHANNELS
# → 执行器不装配、不可 toggle 上线、status 不列出。spec 本体保留在此处，
# 两个用途：① parse_channel_config 对 LIVE_CHANNELS_JSON / live_channel_overrides
# 里残留的退役通道名做「告警跳过」而非 raise（防生产配置引用退役名导致
# 整个实盘执行器装配失败静默全停）；② 机制类单测的 fixture 源（v3_env /
# regime_gate / streak_gate / 同窗互斥 等门禁机制代码仍在，用退役 spec 驱动）。
# 退役依据同 shadow_version_gate.RETIRED_VERSIONS（两处名单必须一致）。
# ---------------------------------------------------------------------------
RETIRED_CHANNEL_SPECS: dict[str, ChannelSpec] = {
    # A1 momentum 族：「深折价顺势」假设被前向数据证伪
    "quote_momentum_v1": ChannelSpec(
        "quote_momentum_v1", "quote_edge", "5m", "DOWN", _qe_guard("quote_momentum_v1"),
        "报价动量（A格顺势）",
    ),
    "quote_momentum_v2": ChannelSpec(
        "quote_momentum_v2", "quote_edge", "5m", "DOWN", _qe_guard("quote_momentum_v1"),
        "报价动量·门禁版", v2_guard="min_drop",
    ),
    "quote_momentum_v3": ChannelSpec(
        "quote_momentum_v3", "quote_edge", "5m", "DOWN", _qe_guard("quote_momentum_v1"),
        "报价动量·非连涨门禁版", streak_gate=True,
    ),
    # A2 被同名 v2 严格支配（v2 = v1 触发集纯子集 + 门禁）
    "quote_contrarian_v1": ChannelSpec(
        "quote_contrarian_v1", "quote_edge", "5m", "DOWN", _qe_guard("quote_contrarian_v1"),
        "报价反向（B格逆势）",
    ),
    "x4_v1": ChannelSpec("x4_v1", "x4", "5m", "DOWN", 0.45, "情绪错位（收阳押次窗DOWN）"),
    # B 档：门禁未兑现，被 contrarian_v2 支配
    "quote_contrarian_v3a": ChannelSpec(
        "quote_contrarian_v3a", "quote_edge", "5m", "DOWN", _qe_guard("quote_contrarian_v1"),
        "报价反向·交替环境版", v2_guard="max_rise", v3_env=True,
    ),
    "quote_contrarian_v3b": ChannelSpec(
        "quote_contrarian_v3b", "quote_edge", "5m", "DOWN", _qe_guard("quote_contrarian_v1"),
        "报价反向·日高回落版", v2_guard="max_rise", v3_env=True,
    ),
    "quote_contrarian_v4": ChannelSpec(
        "quote_contrarian_v4", "quote_edge", "5m", "DOWN", _qe_guard("quote_contrarian_v1"),
        "报价反向·下跌周期版", regime_gate=True,
    ),
}

RETIRED_CHANNELS: frozenset[str] = frozenset(RETIRED_CHANNEL_SPECS)


# 同窗互斥组：组内每市场窗口至多一个通道成交（防同源通道同窗同向叠加敞口；
# 未成交（护栏弃单/失败/占位）不占坑，组内互补通道仍可下单）。
# - S5 深档与 S5 确认通道同源（+5min 确认钩子）、价格带互补（确认护栏 0.75
#   自拦深档高报价窗），互斥防未来护栏调高后叠加。
# momentum 族 v1/v2/v3 互斥组已随三成员全退役而移除（见
# RETIRED_SAME_WINDOW_EXCLUSIVE，仅留作机制测试 fixture）。
SAME_WINDOW_EXCLUSIVE: tuple[frozenset[str], ...] = (
    frozenset({"scene_bull_exhaust_confirm", "s5_deep_z20_v1"}),
)

# 退役的同窗互斥组（不参与生产判定：exclusive_group 只读 SAME_WINDOW_EXCLUSIVE）
RETIRED_SAME_WINDOW_EXCLUSIVE: tuple[frozenset[str], ...] = (
    frozenset({"quote_momentum_v1", "quote_momentum_v2", "quote_momentum_v3"}),
)


def exclusive_group(channel: str) -> frozenset[str] | None:
    """通道所属同窗互斥组（None = 不与他通道同窗成交限制）。"""
    for grp in SAME_WINDOW_EXCLUSIVE:
        if channel in grp:
            return grp
    return None


@dataclass
class ChannelConfig:
    """通道运行时配置（toggle/热调可变；重启回落 live_channels_json 解析结果）。"""

    enabled: bool = False
    amount_usdt: float = 2.0
    max_daily_orders: int = 100
    max_exec_price: float | None = None   # None → ChannelSpec.auto_max_exec
    fired: set[int] = field(default_factory=set, repr=False)  # 本进程已开火 window_start（防同窗重单）
    fire_total: int = 0


def resolve_max_exec(spec: ChannelSpec, cfg: ChannelConfig) -> float:
    """生效护栏：显式配置优先，缺省回落 spec.auto_max_exec。"""
    return cfg.max_exec_price if cfg.max_exec_price is not None else spec.auto_max_exec


def parse_channel_config() -> dict[str, ChannelConfig]:
    """默认配置 + LIVE_CHANNELS_JSON 覆盖 + 校验（非法值 ValueError 拒启）。

    默认全部 OFF（enabled=False），金额/日限取全局默认；JSON 里逐通道覆盖。
    """
    configs: dict[str, ChannelConfig] = {
        ch: ChannelConfig(
            enabled=False,
            amount_usdt=settings.live_default_amount_usdt,
            max_daily_orders=settings.live_default_max_daily_orders,
        )
        for ch in LIVE_CHANNELS
    }

    raw = (settings.live_channels_json or "").strip()
    if not raw:
        return configs

    try:
        overrides: dict = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"多通道实盘：LIVE_CHANNELS_JSON 解析失败：{exc}") from exc
    if not isinstance(overrides, dict):
        raise ValueError("多通道实盘：LIVE_CHANNELS_JSON 顶层必须是对象 {channel: {...}}")

    for ch, ov in overrides.items():
        if ch not in LIVE_CHANNELS:
            if ch in RETIRED_CHANNELS:
                # 退役通道在配置里残留（env JSON / DB 覆盖行）：告警跳过而非 raise。
                # 若这里 raise，main 装配只捕 ValueError 后会把 multi_live_trader
                # 置 None → 整个实盘（含保留通道）静默全停。拼写错误仍走下方 raise。
                logger.warning("多通道实盘：配置中的退役通道已忽略 | {}", ch)
                continue
            raise ValueError(
                f"多通道实盘：未知通道 {ch!r}（白名单 {list(LIVE_CHANNELS)}）")
        if ov is None:
            continue
        if not isinstance(ov, dict):
            raise ValueError(f"多通道实盘：通道 {ch} 的配置必须是对象，got {type(ov).__name__}")
        cfg = configs[ch]
        if "enabled" in ov:
            cfg.enabled = bool(ov["enabled"])
        if "amount_usdt" in ov:
            try:
                amount = float(ov["amount_usdt"])
            except (TypeError, ValueError) as exc:
                # null/非数值必须归一为 ValueError（main 装配只捕 ValueError）；
                # 若放 TypeError 穿透 lifespan 会拖垮整个服务启动
                raise ValueError(
                    f"多通道实盘：通道 {ch} amount_usdt 非法：{ov['amount_usdt']!r}"
                    "（需为数值）") from exc
            if not (0.1 <= amount <= MAX_ORDER_AMOUNT_USDT):
                raise ValueError(
                    f"多通道实盘：通道 {ch} 单笔金额 {amount} 超界"
                    f" [0.1, {MAX_ORDER_AMOUNT_USDT}]（配置误写拒绝启动）")
            cfg.amount_usdt = amount
        if "max_daily_orders" in ov:
            raw_daily = ov["max_daily_orders"]
            # 拒绝小数/布尔/null：日限必须是干净整数，防 int(2.7)→2 静默截断
            if isinstance(raw_daily, bool) or not isinstance(raw_daily, int):
                raise ValueError(
                    f"多通道实盘：通道 {ch} max_daily_orders 非法：{raw_daily!r}"
                    "（需为整数）")
            daily = raw_daily
            if not (1 <= daily <= MAX_DAILY_ORDERS_CAP):
                raise ValueError(
                    f"多通道实盘：通道 {ch} 日限 {daily} 超界 [1, {MAX_DAILY_ORDERS_CAP}]")
            cfg.max_daily_orders = daily
        if "max_exec_price" in ov and ov["max_exec_price"] is not None:
            try:
                exec_price = float(ov["max_exec_price"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"多通道实盘：通道 {ch} max_exec_price 非法：{ov['max_exec_price']!r}"
                    "（需为数值）") from exc
            if not (0.01 <= exec_price <= 0.99):
                raise ValueError(f"多通道实盘：通道 {ch} 护栏 {exec_price} 超界 [0.01, 0.99]")
            cfg.max_exec_price = exec_price
    return configs


def scene_pattern_to_channel(pattern_type: str) -> str | None:
    """场景 pattern_type → 通道名（fake_breakout 钩子 payload 映射）。"""
    ch = f"scene_{pattern_type}"
    return ch if ch in LIVE_CHANNELS else None
