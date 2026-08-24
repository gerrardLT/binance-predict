"""多通道实盘注册表（MultiLiveTrader 的静态描述层 + 配置解析，纯函数无 DB）。

通道 ID 与影子信号版本名对齐（订单 signal_version 直接用通道名，对账/统计天然一致）。
三族触发机制（由 MultiLiveTrader 分别驱动，见 multi_live_trader.py）：
- quote_edge：5m 采样循环喂价 → 窗内报价区间命中（v2 附加 BTC 门禁，实时喂价解锁）；
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


def _qe_guard(version: str) -> float:
    """quote_edge 族自动护栏：区间上界 + 0.03（与旧 QuoteEdgeLiveTrader 同推导）。"""
    return round(QUOTE_EDGE_RULES[version][3] + _QE_GUARD_MARGIN, 4)


LIVE_CHANNELS: dict[str, ChannelSpec] = {
    # --- quote_edge 族（5m 窗内报价触发，区间引用 QUOTE_EDGE_RULES 冻结口径）---
    "quote_momentum_v1": ChannelSpec(
        "quote_momentum_v1", "quote_edge", "5m", "DOWN", _qe_guard("quote_momentum_v1"),
        "报价动量（A格顺势）",
    ),
    "quote_contrarian_v1": ChannelSpec(
        "quote_contrarian_v1", "quote_edge", "5m", "DOWN", _qe_guard("quote_contrarian_v1"),
        "报价反向（B格逆势）",
    ),
    # v2 门禁版：v1 区间 + BTC 门禁（chg≤−0.10% / chg<+0.10%，阈值引用 V2_PRICE_GUARDS）
    "quote_momentum_v2": ChannelSpec(
        "quote_momentum_v2", "quote_edge", "5m", "DOWN", _qe_guard("quote_momentum_v1"),
        "报价动量·门禁版", v2_guard="min_drop",
    ),
    "quote_contrarian_v2": ChannelSpec(
        "quote_contrarian_v2", "quote_edge", "5m", "DOWN", _qe_guard("quote_contrarian_v1"),
        "报价反向·门禁版", v2_guard="max_rise",
    ),
    # --- x4 族（影子 PENDING → 次窗 +150s 决策点，入场价历史偏低）---
    "x4_v1": ChannelSpec("x4_v1", "x4", "5m", "DOWN", 0.45, "情绪错位（收阳押次窗DOWN）"),
    "x4_v2": ChannelSpec("x4_v2", "x4", "5m", "DOWN", 0.50, "情绪错位·平静市门禁版"),
    # --- 场景族（15m 市场次周期开盘入场；S5 为 +5min 确认入场）---
    "scene_bull_exhaust": ChannelSpec(
        "scene_bull_exhaust", "scene", "15m", "DOWN", 0.60, "场景S1 多头耗尽（押DOWN）",
    ),
    "scene_bull_exhaust_confirm": ChannelSpec(
        "scene_bull_exhaust_confirm", "scene", "15m", "DOWN", 0.75, "场景S5 确认入场（押DOWN）",
    ),
    "scene_bear_exhaust": ChannelSpec(
        "scene_bear_exhaust", "scene", "15m", "UP", 0.55, "场景S2 空头耗尽（押UP）",
    ),
    "scene_momentum_fade": ChannelSpec(
        "scene_momentum_fade", "scene", "15m", "DOWN", 0.55, "场景S4 动量衰竭（押DOWN）",
    ),
}


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
            amount = float(ov["amount_usdt"])
            if not (0.1 <= amount <= MAX_ORDER_AMOUNT_USDT):
                raise ValueError(
                    f"多通道实盘：通道 {ch} 单笔金额 {amount} 超界"
                    f" [0.1, {MAX_ORDER_AMOUNT_USDT}]（配置误写拒绝启动）")
            cfg.amount_usdt = amount
        if "max_daily_orders" in ov:
            daily = int(ov["max_daily_orders"])
            if not (1 <= daily <= MAX_DAILY_ORDERS_CAP):
                raise ValueError(
                    f"多通道实盘：通道 {ch} 日限 {daily} 超界 [1, {MAX_DAILY_ORDERS_CAP}]")
            cfg.max_daily_orders = daily
        if "max_exec_price" in ov and ov["max_exec_price"] is not None:
            exec_price = float(ov["max_exec_price"])
            if not (0.01 <= exec_price <= 0.99):
                raise ValueError(f"多通道实盘：通道 {ch} 护栏 {exec_price} 超界 [0.01, 0.99]")
            cfg.max_exec_price = exec_price
    return configs


def scene_pattern_to_channel(pattern_type: str) -> str | None:
    """场景 pattern_type → 通道名（fake_breakout 钩子 payload 映射）。"""
    ch = f"scene_{pattern_type}"
    return ch if ch in LIVE_CHANNELS else None
