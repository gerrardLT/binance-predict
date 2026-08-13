"""科学发现系统 V1.1 经济闸测试（宪法 Q6 第 5 步 + 第八条规则 8）。

覆盖：
- services/ev_gate.py
  truncate_to_decision_point 决策点截断（三通道+价格曲线同步，不偷看未来）
  entry_price_at 入场价提取（真实价优先 / chance 代理回退 / 缺失）
  bet_pnl 逐注盈亏（费 2%+溢价 0.01，防除零截断）
  hypothesis_ev 经济账审判（EV 点估计 + bootstrap CI + 闸判定）
- services/sentiment_agent.py  _screen_and_serialize 经济闸降级
  （双轨 ACTIVE 但 EV 不过 → OBSERVE；闸开关关闭/无价格窗口时保持原裁决）

数值案例均经手工核算（见各用例注释）。
"""

from __future__ import annotations

from unittest.mock import patch

from binance_predict.config.settings import settings
from binance_predict.models.schemas import PredicateHypothesis
from binance_predict.services.discovery import VERDICT_ACTIVE, VERDICT_OBSERVE
from binance_predict.services.ev_gate import (
    MIN_EV_FIRES,
    bet_pnl,
    entry_price_at,
    hypothesis_ev,
    truncate_to_decision_point,
)
from binance_predict.services.sentiment_agent import SentimentAgent
from binance_predict.services.symbolizer import ChannelView, WindowView

# ============================================================
# 共用构造
# ============================================================

DAY_MS = 86_400_000

# 命中谓词：sentiment 通道至少含 1 个「急升」
HIT_PREDICATE = {
    "pred": "count_symbol",
    "channel": "sentiment",
    "symbol": "急升",
    "cmp": ">=",
    "value": 1,
}

T0 = 1_800_000_000_000  # 任意基准时刻（毫秒）


def _mk_view(i: int, outcome: str, hit: bool) -> WindowView:
    symbols = ["急升", "平", "平"] if hit else ["平", "平", "平"]
    return WindowView(
        start_time=T0 + i * DAY_MS,
        outcome=outcome,
        channels={
            "sentiment": ChannelView(
                symbols=symbols, geometry={}, point_count=len(symbols) + 1
            )
        },
    )


def _mk_window(i: int, outcome: str, up_price: float | None) -> dict:
    """与 _mk_view 同序对应的截断窗口 dict（入场价 + 结算来源）。

    采样点位于开窗后 60s/120s（均 <=150s 决策点），entry_price_at 取后者。
    """
    start = T0 + i * DAY_MS
    ret = 0.01 if outcome == "UP" else -0.01
    return {
        "id": i,
        "start_time": start,
        "outcome": outcome,
        "actual_return": ret,
        "curve_up_price": (
            [
                {"t": start + 60_000, "v": up_price - 0.05},
                {"t": start + 120_000, "v": up_price},
            ]
            if up_price is not None
            else []
        ),
        "curve_down_price": [],
        "curve_up_pct": [
            {"t": start + 60_000, "v": 45.0},
            {"t": start + 120_000, "v": 55.0},
        ],
        "curve_down_pct": [],
    }


def _active_case(n_hits: int = 12, up_price: float = 0.5):
    """强信号 ACTIVE 场景（40 窗口）+ 对应窗口 dict。

    命中集 = 前 n_hits 个间隔 3 天的窗口（全 UP）；对照池 UP 占比 ~1/3。
    hit=12/12=1.0 vs base≈9/28≈0.32 ⇒ lift≈3.1（CI 下界 exp(1.14-0.52)>1）。
    EV（全赢，price=0.5）：0.98/0.51-1 ≈ +0.92/注 ⇒ CI 下界恒=点估计>0。
    """
    hits = {2 + 3 * k for k in range(n_hits)}
    views, windows = [], []
    for i in range(40):
        outcome = "UP" if (i in hits or i % 3 == 0) else "DOWN"
        hit = i in hits
        views.append(_mk_view(i, outcome, hit))
        windows.append(_mk_window(i, outcome, up_price if hit else None))
    return views, windows


# ============================================================
# truncate_to_decision_point
# ============================================================


class TestTruncateToDecisionPoint:
    def test_truncates_all_curve_keys(self):
        # 截断以曲线首采样点为相对基准（与 feature_bench.truncate_windows 同口径）
        w = {
            "start_time": T0,
            "outcome": "UP",
            "actual_return": 0.01,
            "curve_up_pct": [
                {"t": T0, "v": 50.0},
                {"t": T0 + 100_000, "v": 51.0},
                {"t": T0 + 200_000, "v": 52.0},  # 首点后 200s >150s 应被截掉
            ],
            "curve_up_price": [
                {"t": T0 + 60_000, "v": 0.5},
                {"t": T0 + 300_000, "v": 0.9},  # 末端高价不得泄漏进决策视图
            ],
        }
        out = truncate_to_decision_point([w], 150.0)[0]
        assert [p["t"] for p in out["curve_up_pct"]] == [T0, T0 + 100_000]
        assert [p["v"] for p in out["curve_up_price"]] == [0.5]
        # 结算字段不随截断改变（在第 150s 决策、等整窗结算）
        assert out["outcome"] == "UP"
        assert out["actual_return"] == 0.01

    def test_short_curve_kept_as_is(self):
        # 截断后不足 2 点的曲线保持原样（由符号化层防御跳过），不制造伪点
        w = {"start_time": T0, "curve_up_pct": [{"t": T0 + 300_000, "v": 1.0}]}
        out = truncate_to_decision_point([w], 150.0)[0]
        assert len(out["curve_up_pct"]) == 1

    def test_empty_curves_untouched(self):
        w = {"start_time": T0, "curve_up_pct": None, "curve_btc_price": []}
        out = truncate_to_decision_point([w], 150.0)[0]
        assert out["curve_up_pct"] is None
        assert out["curve_btc_price"] == []


# ============================================================
# entry_price_at
# ============================================================


class TestEntryPriceAt:
    def test_real_price_preferred_and_no_future_peek(self):
        w = {
            "start_time": T0,
            "curve_up_price": [
                {"t": T0 + 60_000, "v": 0.55},
                {"t": T0 + 120_000, "v": 0.62},  # <=150s 最后一个 → 0.62
                {"t": T0 + 240_000, "v": 0.95},  # 未来价不得使用
            ],
        }
        price, kind = entry_price_at(w, "UP", 150.0)
        assert price == 0.62
        assert kind == "real"

    def test_proxy_fallback_when_no_price_curve(self):
        w = {
            "start_time": T0,
            "curve_up_price": [],
            "curve_up_pct": [{"t": T0 + 120_000, "v": 55.0}],
        }
        price, kind = entry_price_at(w, "UP", 150.0)
        assert price == 0.55
        assert kind == "proxy"

    def test_down_direction_uses_down_curve(self):
        w = {
            "start_time": T0,
            "curve_down_price": [{"t": T0 + 90_000, "v": 0.31}],
        }
        price, kind = entry_price_at(w, "DOWN", 150.0)
        assert price == 0.31
        assert kind == "real"

    def test_missing_when_nothing_available(self):
        price, kind = entry_price_at({"start_time": T0}, "UP", 150.0)
        assert price is None
        assert kind == "missing"


# ============================================================
# bet_pnl
# ============================================================


class TestBetPnl:
    def test_win_after_fee_and_premium(self):
        # (1-0.02)/min(max(0.5+0.01,0.01),0.99) - 1 = 0.98/0.51 - 1 ≈ 0.9216
        assert abs(bet_pnl(True, 0.5) - (0.98 / 0.51 - 1)) < 1e-9

    def test_lose_is_minus_one(self):
        assert bet_pnl(False, 0.5) == -1.0

    def test_extreme_price_clamped(self):
        # price+premium=1.05 → 截断 0.99：0.98/0.99-1 ≈ -0.0101（高价必输口径）
        assert abs(bet_pnl(True, 1.04) - (0.98 / 0.99 - 1)) < 1e-9


# ============================================================
# hypothesis_ev
# ============================================================


class TestHypothesisEv:
    def test_all_win_cheap_entry_passes(self):
        """12 注全赢 + 入场价 0.5：EV=0.98/0.51-1≈+0.92，CI 恒等于点估计 → 过闸。"""
        views = [_mk_view(i, "UP", True) for i in range(MIN_EV_FIRES + 2)]
        windows = [_mk_window(i, "UP", 0.5) for i in range(MIN_EV_FIRES + 2)]
        r = hypothesis_ev(HIT_PREDICATE, "UP", views, windows)
        assert r.n_fires == MIN_EV_FIRES + 2
        assert r.n_real_price == MIN_EV_FIRES + 2
        assert r.win_rate == 1.0
        assert abs(r.avg_entry_price - 0.5) < 1e-9
        assert r.ev is not None and r.ev > 0.9
        assert r.ev_ci_lower is not None and r.ev_ci_lower > 0
        assert r.passed is True

    def test_all_win_expensive_entry_fails(self):
        """全赢但入场价 0.98：pnl=0.98/0.99-1≈-0.0101/注 → EV<0，不过闸。

        经济直觉：二元市场高价票全赢也亏钱（费+溢价后赔付不足以覆盖成本）。
        """
        views = [_mk_view(i, "UP", True) for i in range(MIN_EV_FIRES + 2)]
        windows = [_mk_window(i, "UP", 0.98) for i in range(MIN_EV_FIRES + 2)]
        r = hypothesis_ev(HIT_PREDICATE, "UP", views, windows)
        assert r.ev is not None and r.ev < 0
        assert r.passed is False

    def test_insufficient_fires_not_passed(self):
        """EV 为正但注数 < MIN_EV_FIRES：经济功效不足，不过闸（不判模式无效）。"""
        views = [_mk_view(i, "UP", True) for i in range(MIN_EV_FIRES - 1)]
        windows = [_mk_window(i, "UP", 0.5) for i in range(MIN_EV_FIRES - 1)]
        r = hypothesis_ev(HIT_PREDICATE, "UP", views, windows)
        assert r.n_fires == MIN_EV_FIRES - 1
        assert r.ev is not None and r.ev > 0  # 点估计仍给出（证据展示）
        assert r.passed is False

    def test_no_fire_returns_zero_fires(self):
        views = [_mk_view(i, "UP", False) for i in range(5)]
        windows = [_mk_window(i, "UP", 0.5) for i in range(5)]
        r = hypothesis_ev(HIT_PREDICATE, "UP", views, windows)
        assert r.n_fires == 0
        assert r.ev is None
        assert r.passed is False

    def test_zero_return_excluded(self):
        """actual_return=0（平盘）剔除：命中但不计注。"""
        views = [_mk_view(i, "UP", True) for i in range(3)]
        windows = []
        for i in range(3):
            w = _mk_window(i, "UP", 0.5)
            w["actual_return"] = 0.0
            windows.append(w)
        r = hypothesis_ev(HIT_PREDICATE, "UP", views, windows)
        assert r.n_fires == 0

    def test_all_wrong_direction_negative_ev(self):
        """全押错方向：pnl=-1/注 → EV=-1，不过闸。"""
        views = [_mk_view(i, "DOWN", True) for i in range(MIN_EV_FIRES + 2)]
        windows = [_mk_window(i, "DOWN", 0.5) for i in range(MIN_EV_FIRES + 2)]
        r = hypothesis_ev(HIT_PREDICATE, "UP", views, windows)
        assert r.win_rate == 0.0
        assert r.ev == -1.0
        assert r.passed is False


# ============================================================
# _screen_and_serialize 经济闸（V1.1）
# ============================================================


def _mk_hypothesis() -> PredicateHypothesis:
    return PredicateHypothesis(
        pattern_name="早段急升",
        description="情绪前段急升后 UP 偏向",
        predicate=HIT_PREDICATE,
        target_outcome="UP",
        confidence_score=0.7,
        rationale="多窗口共性结构",
    )


class TestScreenEvGate:
    def test_cheap_entry_keeps_active(self):
        """双轨 ACTIVE + 入场价 0.5（全赢 EV≈+0.92）→ 保持 ACTIVE，闸通过。"""
        agent = SentimentAgent.__new__(SentimentAgent)
        views, windows = _active_case(n_hits=12, up_price=0.5)
        out = agent._screen_and_serialize(
            [_mk_hypothesis()], views, "v-test", holdout_windows=windows
        )
        d = out[0]
        assert d["screen_verdict"] == VERDICT_ACTIVE
        assert d["screen_ev_passed"] is True
        assert d["screen_ev"] is not None and d["screen_ev"] > 0.9
        assert d["screen_ev_fires"] == 12
        assert d["screen_ev_ci_lower"] is not None and d["screen_ev_ci_lower"] > 0

    def test_expensive_entry_downgrades_to_observe(self):
        """双轨 ACTIVE 但入场价 0.98（全赢 EV≈-1%）→ 降 OBSERVE，lift 证据保留。"""
        agent = SentimentAgent.__new__(SentimentAgent)
        views, windows = _active_case(n_hits=12, up_price=0.98)
        out = agent._screen_and_serialize(
            [_mk_hypothesis()], views, "v-test", holdout_windows=windows
        )
        d = out[0]
        assert d["screen_verdict"] == VERDICT_OBSERVE  # 降级
        assert d["screen_lift"] >= 1.4  # lift 审判证据不丢
        assert d["screen_ev_passed"] is False
        assert d["screen_ev"] is not None and d["screen_ev"] < 0

    def test_gate_disabled_keeps_verdict(self):
        """闸开关关闭：即使 EV 为负也保持 ACTIVE，ev 证据不计算。"""
        agent = SentimentAgent.__new__(SentimentAgent)
        views, windows = _active_case(n_hits=12, up_price=0.98)
        with patch.object(settings, "agent_ev_gate_enabled", False):
            out = agent._screen_and_serialize(
                [_mk_hypothesis()], views, "v-test", holdout_windows=windows
            )
        d = out[0]
        assert d["screen_verdict"] == VERDICT_ACTIVE
        assert d["screen_ev"] is None
        assert d["screen_ev_passed"] is None

    def test_no_windows_backward_compatible(self):
        """不传 holdout_windows（旧调用形态）：经济闸跳过，verdict 保持。"""
        agent = SentimentAgent.__new__(SentimentAgent)
        views, _ = _active_case(n_hits=12, up_price=0.5)
        out = agent._screen_and_serialize([_mk_hypothesis()], views, "v-test")
        d = out[0]
        assert d["screen_verdict"] == VERDICT_ACTIVE
        assert d["screen_ev"] is None
