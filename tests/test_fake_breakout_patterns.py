"""场景收盘质量判定（classify_close_pattern / is_momentum_fade）与破位 pending 状态机回归测试。

对应 2026-08-17 真 OOS 修正版升级（360 天发现→验证集盲验→真 OOS 终验）：
- S1 bull_exhaust：破 4h 阻力 + 收阳 + 光头（close_pos ≥ 0.85）+ 4h 区间上沿（pos4h ≥ 0.9）→ 次周期 DOWN
- S2 bear_exhaust：破 4h 支撑 + 收阴 + 放量（vol_ratio ≥ 2.0）→ 次周期 UP
- S4 momentum_fade：连阳 ≥ 3（含当前根）× 光头阳，无破位要求（每周期独立判定）

只测纯函数与内存状态机（不触 DB/邮件/klines）：
- classify_close_pattern：S1/S2 通过/不通过/边界值/防御分支（未命中不回传指标）
- is_momentum_fade：S4 连阳口径（streak 含当前根，前 2 根阳即可）
- compute_pattern_stats：统计纯函数（stats API 与结算回填共用）
- _record_breakout：每方向每周期只记首次破位
- _on_cycle_boundary：停机跳变（>1 周期）不确认且过期 pending 已清理
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from binance_predict.db.models import FakeBreakoutSignal, PatternShadowSignal
from binance_predict.services import fake_breakout_detector as fbd
from binance_predict.services.fake_breakout_detector import (
    CLOSE_POS_MIN,
    FEE,
    PATTERN_GROUP,
    POS4H_MIN,
    RESEARCH_WIN_RATES,
    S5_DEEP_RULE_TEXT,
    S5_DEEP_VERSION,
    S5_DEEP_Z5,
    VOL_RATIO_MIN,
    FakeBreakoutDetector,
    classify_close_pattern,
    compute_pattern_stats,
    confirm_bull_exhaust_5m,
    is_momentum_fade,
)


# ============================================================
# classify_close_pattern：S1（high：收阳 + 光头 + 4h 区间上沿）
# ============================================================

def test_pattern1_bull_exhaust_pass() -> None:
    """光头阳线（close_pos 0.909 ≥ 0.85，pos4h 0.95 ≥ 0.9）：命中，指标回传供审计。"""
    ok, pattern_type, close_pos, vol_ratio = classify_close_pattern(
        "high", o=100.0, h=110.0, l=99.0, c=109.0, volume=50.0, vol_ma=40.0,
        pos4h=0.95,
    )
    assert ok is True
    assert pattern_type == "bull_exhaust"
    assert close_pos == pytest.approx((109.0 - 99.0) / (110.0 - 99.0))
    assert vol_ratio == pytest.approx(50.0 / 40.0)


def test_pattern1_upper_shadow_fail() -> None:
    """长上影（close_pos 0.5 < 0.85）：多头未满仓，不命中。"""
    ok, pattern_type, close_pos, _ = classify_close_pattern(
        "high", o=100.0, h=110.0, l=99.0, c=104.5, volume=50.0, vol_ma=40.0,
        pos4h=0.95,
    )
    assert ok is False
    assert pattern_type is None and close_pos is None


def test_pattern1_red_close_fail() -> None:
    """收阴（c < o）：即使收盘位置高也不命中（S1 要求收阳）。"""
    ok, _, close_pos, _ = classify_close_pattern(
        "high", o=109.0, h=110.0, l=99.0, c=108.5, volume=50.0, vol_ma=40.0,
        pos4h=0.95,
    )
    assert ok is False
    assert close_pos is None  # 未命中不回传指标


def test_pattern1_close_pos_boundary() -> None:
    """收盘位置恰好 0.85（≥ 含等号）：命中。"""
    ok, _, close_pos, _ = classify_close_pattern(
        "high", o=100.0, h=110.0, l=100.0, c=100.0 + 10.0 * CLOSE_POS_MIN,
        volume=50.0, vol_ma=40.0, pos4h=0.95,
    )
    assert ok is True
    assert close_pos == pytest.approx(CLOSE_POS_MIN)


def test_pattern1_pos4h_below_min_fail() -> None:
    """4h 区间位置不足（pos4h 0.85 < 0.9）：F25 条件拒绝（真 OOS 修正版新增）。"""
    ok, _, _, _ = classify_close_pattern(
        "high", o=100.0, h=110.0, l=100.0, c=109.5, volume=50.0, vol_ma=40.0,
        pos4h=POS4H_MIN - 0.05,
    )
    assert ok is False


def test_pattern1_pos4h_boundary() -> None:
    """pos4h 恰好 0.9（≥ 含等号）：命中。"""
    ok, _, _, _ = classify_close_pattern(
        "high", o=100.0, h=110.0, l=100.0, c=109.5, volume=50.0, vol_ma=40.0,
        pos4h=POS4H_MIN,
    )
    assert ok is True


def test_pattern1_pos4h_none_fail_conservative() -> None:
    """4h 历史不足（pos4h=None）：保守不命中（无审计依据）。"""
    ok, _, _, _ = classify_close_pattern(
        "high", o=100.0, h=110.0, l=100.0, c=109.5, volume=50.0, vol_ma=40.0,
        pos4h=None,
    )
    assert ok is False


# ============================================================
# classify_close_pattern：场景②（low：收阴 + 放量）
# ============================================================

def test_pattern2_bear_exhaust_pass() -> None:
    """收阴 + 放量（量比 2.67 ≥ 2.0）：命中。"""
    ok, pattern_type, close_pos, vol_ratio = classify_close_pattern(
        "low", o=110.0, h=111.0, l=100.0, c=101.0, volume=80.0, vol_ma=30.0,
    )
    assert ok is True
    assert pattern_type == "bear_exhaust"
    assert close_pos == pytest.approx(1.0 / 11.0)
    assert vol_ratio == pytest.approx(80.0 / 30.0)


def test_pattern2_low_volume_fail() -> None:
    """收阴但缩量（量比 1.67 < 2.0）：不命中。"""
    ok, _, _, vol_ratio = classify_close_pattern(
        "low", o=110.0, h=111.0, l=100.0, c=101.0, volume=50.0, vol_ma=30.0,
    )
    assert ok is False
    assert vol_ratio is None  # 未命中不回传指标


def test_pattern2_vol_ratio_boundary() -> None:
    """量比恰好 2.0（≥ 含等号）：命中。"""
    ok, _, _, vol_ratio = classify_close_pattern(
        "low", o=110.0, h=111.0, l=100.0, c=101.0,
        volume=30.0 * VOL_RATIO_MIN, vol_ma=30.0,
    )
    assert ok is True
    assert vol_ratio == pytest.approx(VOL_RATIO_MIN)


def test_pattern2_no_vol_ma_fail_conservative() -> None:
    """均量数据不足（vol_ma=None）：S2 保守不命中（无审计依据）。"""
    ok, _, _, vol_ratio = classify_close_pattern(
        "low", o=110.0, h=111.0, l=100.0, c=101.0, volume=500.0, vol_ma=None,
    )
    assert ok is False
    assert vol_ratio is None


def test_pattern2_green_close_fail() -> None:
    """收阳：即使巨量也不命中（S2 要求收阴）。"""
    ok, _, _, _ = classify_close_pattern(
        "low", o=100.0, h=111.0, l=100.0, c=110.0, volume=80.0, vol_ma=30.0,
    )
    assert ok is False


# ============================================================
# classify_close_pattern：防御分支
# ============================================================

def test_zero_range_defensive() -> None:
    """一字板 H==L（rng=0）：除零防御，不命中且指标为 None。"""
    ok, pattern_type, close_pos, vol_ratio = classify_close_pattern(
        "high", o=100.0, h=100.0, l=100.0, c=100.0, volume=50.0, vol_ma=40.0,
        pos4h=0.95,
    )
    assert ok is False
    assert pattern_type is None
    assert close_pos is None
    assert vol_ratio is None


def test_zero_open_defensive() -> None:
    """开盘价异常（o ≤ 0）：防御不命中。"""
    ok, _, _, _ = classify_close_pattern(
        "low", o=0.0, h=111.0, l=100.0, c=101.0, volume=80.0, vol_ma=30.0,
    )
    assert ok is False


# ============================================================
# S4 momentum_fade（连阳 ≥ 3 含当前根 × 光头阳，无破位要求）
# ============================================================

def test_momentum_fade_streak3_pass() -> None:
    """前 2 根阳 + 信号 K 阳（streak=3 含当前根）+ 光头：命中。"""
    hit, close_pos = is_momentum_fade(100.0, 110.0, 100.0, 109.5, [1, 1, -1, 1, 1])
    assert hit is True
    assert close_pos == pytest.approx(0.95)


def test_momentum_fade_streak4_pass() -> None:
    """前 3 根全阳（streak=4，更长连阳仍满足 ≥3）：命中。"""
    hit, _ = is_momentum_fade(100.0, 110.0, 100.0, 109.5, [1, 1, 1])
    assert hit is True


def test_momentum_fade_prev_red_fail() -> None:
    """前 2 根有阴（streak 断）：不命中。"""
    hit, _ = is_momentum_fade(100.0, 110.0, 100.0, 109.5, [1, -1, 1, 1, -1])
    assert hit is False


def test_momentum_fade_insufficient_history_fail() -> None:
    """历史不足 2 根：保守不命中。"""
    hit, _ = is_momentum_fade(100.0, 110.0, 100.0, 109.5, [1])
    assert hit is False


def test_momentum_fade_none_prev_dir_fail() -> None:
    """prev_dir 缺失（None）：保守不命中。"""
    hit, _ = is_momentum_fade(100.0, 110.0, 100.0, 109.5, None)
    assert hit is False


def test_momentum_fade_red_close_fail() -> None:
    """信号 K 收阴：不命中。"""
    hit, _ = is_momentum_fade(105.0, 110.0, 100.0, 100.5, [1, 1, 1])
    assert hit is False


def test_momentum_fade_close_pos_fail() -> None:
    """光头不足（close_pos 0.7 < 0.85）：不命中。"""
    hit, _ = is_momentum_fade(100.0, 110.0, 100.0, 107.0, [1, 1, 1])
    assert hit is False


# ============================================================
# compute_pattern_stats（统计纯函数：stats API 与结算回填共用）
# ============================================================

def _settled_row(side: str, outcome: str, entry: float | None,
                 pattern_type: str = "bull_exhaust") -> SimpleNamespace:
    return SimpleNamespace(
        side=side,
        settle_outcome=outcome,
        entry_down_price_15m=entry if side == "high" else None,
        entry_up_price_15m=entry if side == "low" else None,
        pattern_type=pattern_type,
    )


def test_pattern_stats_2win_1loss() -> None:
    """2 胜 1 负（entry=0.51）：累计 EV / 收益曲线 / 回撤 / 入场 EV 口径。"""
    rows = [
        _settled_row("high", "DOWN", 0.51),  # 赢：(1-FEE)/0.51−1 ≈ 0.9216
        _settled_row("high", "UP", 0.51),    # 输：−1
        _settled_row("low", "UP", None),     # 赢：entry 缺失回退 0.51（含溢价理论价）
    ]
    s = compute_pattern_stats(rows)
    assert s["n"] == 3 and s["wins"] == 2
    assert s["winrate"] == pytest.approx(2 / 3)
    ret_win = (1.0 - FEE) / 0.51 - 1.0
    assert s["cumulative_ev"] == pytest.approx((ret_win - 1.0 + ret_win) / 3, abs=1e-4)
    assert s["equity_curve"][1] == pytest.approx(ret_win - 1.0, abs=1e-3)
    # peak 0.9216 → 谷 −0.0784：回撤恰为 1.0
    assert s["max_drawdown"] == pytest.approx(1.0, abs=1e-3)
    # 入场时刻 EV=p×(1-FEE)/entry−1 只看 p 与 entry，与实际输赢无关
    p = RESEARCH_WIN_RATES["bull_exhaust"]
    assert s["avg_ev_at_entry"] == pytest.approx(p * (1.0 + ret_win) - 1.0, abs=1e-4)


def test_pattern_stats_empty_safe() -> None:
    """空列表：n=0 指标全 None，曲线空（stats API 零信号安全）。"""
    s = compute_pattern_stats([])
    assert s["n"] == 0
    assert s["winrate"] is None and s["cumulative_ev"] is None
    assert s["equity_curve"] == [] and s["max_drawdown"] == 0.0


# ============================================================
# 破位 pending 状态机（内存去重 / 边界跳变保护）
# ============================================================

def _make_detector() -> FakeBreakoutDetector:
    """detector 实例：不启动循环，只注入位势测内存状态机。"""
    d = FakeBreakoutDetector(collector=None, pm_15m_latest={})  # type: ignore[arg-type]
    d._levels = {"4h": {"resistance": 100.0, "support": 90.0}}
    return d


def test_record_breakout_first_per_side_per_cycle() -> None:
    """每方向每周期只记首次破位：同周期二次冲高不覆盖首破记录。"""
    d = _make_detector()
    t0 = 900_000 * 100 + 60_000  # 周期 100 内 1min
    d._record_breakout(t0, mid=100.2)
    d._record_breakout(t0 + 60_000, mid=101.5)  # 更高，但同周期同方向
    assert d._pending_breaks["high"]["break_price"] == 100.2
    assert d._pending_breaks["high"]["cycle_id"] == 100
    # 支撑方向未破：无记录
    assert "low" not in d._pending_breaks


def test_record_breakout_after_boundary_cleanup() -> None:
    """周期边界清理后，新周期破位可重新记录（每周期最多一条的另一半）。"""
    d = _make_detector()
    d._pending_breaks = {
        "high": {"cycle_id": 100, "level": "4h", "broken_level": 100.0,
                 "break_price": 100.2, "break_time": 0},
    }
    # 周期边界：清理只保留 cycle_id >= 101 的记录
    d._pending_breaks = {
        side: rec for side, rec in d._pending_breaks.items()
        if rec["cycle_id"] >= 101
    }
    assert d._pending_breaks == {}
    # 周期 101 内再次冲高：记新 pending
    d._record_breakout(900_000 * 101 + 30_000, mid=100.4)
    assert d._pending_breaks["high"]["cycle_id"] == 101
    assert d._pending_breaks["high"]["break_price"] == 100.4


@pytest.mark.asyncio
async def test_cycle_boundary_gap_skips_confirm(monkeypatch: pytest.MonkeyPatch) -> None:
    """停机跳变（跨 >1 周期）：不确认（次周期已走远），过期 pending 已清理。"""
    d = _make_detector()
    d._pending_breaks = {
        "high": {"cycle_id": 100, "level": "4h", "broken_level": 100.0,
                 "break_price": 100.2, "break_time": 0},
    }
    fired: list[dict] = []

    async def fake_confirm(*args: object, **kwargs: object) -> None:
        fired.append({"args": args})

    monkeypatch.setattr(d, "_confirm_and_fire", fake_confirm)
    await d._on_cycle_boundary(prev_cycle=100, cur_cycle=103, now_ms=900_000 * 103)
    assert fired == []            # 跳变不确认
    assert d._pending_breaks == {}  # 过期 pending 已清理


@pytest.mark.asyncio
async def test_cycle_boundary_normal_dispatches_confirm(monkeypatch: pytest.MonkeyPatch) -> None:
    """正常相邻边界（+1）：上一周期的 pending 派发给收盘确认。"""
    d = _make_detector()
    d._pending_breaks = {
        "high": {"cycle_id": 100, "level": "4h", "broken_level": 100.0,
                 "break_price": 100.2, "break_time": 0},
    }
    fired: list[tuple[object, ...]] = []

    async def fake_confirm(*args: object, **kwargs: object) -> None:
        fired.append(args)

    monkeypatch.setattr(d, "_confirm_and_fire", fake_confirm)
    await d._on_cycle_boundary(prev_cycle=100, cur_cycle=101, now_ms=900_000 * 101)
    assert len(fired) == 1
    due = fired[0][0]  # 第一个位置参数 due: dict[side, rec]
    assert "high" in due
    assert due["high"]["cycle_id"] == 100


@pytest.mark.asyncio
async def test_status_snapshot_exposes_scene_state() -> None:
    """status_snapshot 暴露 pending/重试/周期号，供线上排查（t15 验证用）。"""
    d = _make_detector()
    d._pending_breaks = {
        "low": {"cycle_id": 2071, "level": "4h", "broken_level": 90.0,
                "break_price": 89.9, "break_time": 0},
    }
    d._confirm_retries = [{"due": {}, "prev_cycle": 2070, "cur_cycle": 2071, "retry": 1, "at": 0}]
    d._last_cycle_id = 2071
    snap = d.status_snapshot
    assert snap["pending_breaks"]["low"]["cycle_id"] == 2071
    assert snap["pending_breaks"]["low"]["broken_level"] == 90.0
    assert snap["confirm_retries"] == 1
    assert snap["last_cycle_id"] == 2071

# ============================================================
# S5 bull_exhaust_confirm（S1 +5min 回落确认，2026-08-18）
# ============================================================

def test_s5_confirm_falling_pass() -> None:
    """次周期第 1 根 5m 收盘 < 周期开盘（z5<0）：确认买 DOWN。"""
    confirmed, reason = confirm_bull_exhaust_5m(c5_close=99.8, anchor=100.0)
    assert confirmed is True
    assert reason == "CONFIRM"


def test_s5_skip_rising() -> None:
    """5m 收盘 > 周期开盘（反向上涨，对照组胜率仅 34%）：放弃。"""
    confirmed, reason = confirm_bull_exhaust_5m(c5_close=100.3, anchor=100.0)
    assert confirmed is False
    assert reason == "RISING"


def test_s5_skip_flat_and_invalid() -> None:
    """平价（NOISE）与异常输入：均放弃不落 S5 行。"""
    assert confirm_bull_exhaust_5m(c5_close=100.0, anchor=100.0) == (False, "FLAT")
    assert confirm_bull_exhaust_5m(c5_close=0.0, anchor=100.0) == (False, "INVALID")
    assert confirm_bull_exhaust_5m(c5_close=100.0, anchor=0.0) == (False, "INVALID")


def test_s5_registered_in_constants() -> None:
    """S5 胜率点估计 0.785 入表；pattern 旧列映射 bull_exhaust（String(16) 上限）。"""
    assert RESEARCH_WIN_RATES["bull_exhaust_confirm"] == 0.785
    assert PATTERN_GROUP["bull_exhaust_confirm"] == "bull_exhaust"
    assert len(PATTERN_GROUP["bull_exhaust_confirm"]) <= 16


def test_s5_pattern_stats_ev_uses_785_entry() -> None:
    """S5 统计：入场 EV 用 p=0.785 与 +5min 真实入场价（entry=0.65 例）。"""
    rows = [_settled_row("high", "DOWN", 0.65, pattern_type="bull_exhaust_confirm")]
    s = compute_pattern_stats(rows)
    p = RESEARCH_WIN_RATES["bull_exhaust_confirm"]
    ret = (1.0 - FEE) / 0.65 - 1.0
    assert s["winrate"] == 1.0
    assert s["cumulative_ev"] == pytest.approx(ret, abs=1e-6)
    assert s["avg_ev_at_entry"] == pytest.approx(p * (1.0 + ret) - 1.0, abs=1e-6)
    # 盈亏平衡入场价 ≈ 0.769：0.65 入场为正 EV，与回测敏感性表一致
    assert s["avg_ev_at_entry"] > 0


# ============================================================
# S5 深档影子（z5 ≤ −20bp → pattern_shadow_signals，纯影子不下注，2026-09-03）
# 门禁在 _fire_s5_signal：z5<=S5_DEEP_Z5 且 matched（+5min 报价属本目标周期）才落；
# 落表复用 _fire_s5_deep_shadow（幂等：唯一约束 version+signal_bar_start，先查后插）。
# 不触真实 DB：async_session_factory / clock_sync 全替身，按类型过滤捕获的 add 对象。
# ============================================================

NEXT_START = 1_000_000_000_000
NEXT_END = NEXT_START + 900_000


class _S5Result:
    def __init__(self, scalar) -> None:
        self._scalar = scalar

    def scalar_one_or_none(self):
        return self._scalar


class _S5Session:
    """假 session：get 返回父信号；execute().scalar_one_or_none() 返回幂等探针值；add 累积。"""

    def __init__(self, parent=None, exists=None) -> None:
        self._parent = parent
        self._exists = exists
        self.added: list = []
        self.commits = 0

    async def get(self, model, pk):
        return self._parent

    async def execute(self, stmt):
        return _S5Result(self._exists)

    def add(self, obj) -> None:
        self.added.append(obj)

    async def commit(self) -> None:
        self.commits += 1

    async def refresh(self, obj) -> None:
        return None


class _S5Ctx:
    def __init__(self, session: _S5Session) -> None:
        self._s = session

    async def __aenter__(self) -> _S5Session:
        return self._s

    async def __aexit__(self, *exc) -> bool:
        return False


class _FakeStore:
    def __init__(self, mid: float) -> None:
        self.mid_price = mid


class _FakeCollector:
    def __init__(self, mid: float = 64_000.0) -> None:
        self.store = _FakeStore(mid)


def _s5_parent() -> SimpleNamespace:
    return SimpleNamespace(
        level="4h", side="high", resistance=65_000.0, eps=0.001,
        market_start_15m=NEXT_START, market_end_15m=NEXT_END,
        cycle_open_price_15m=100.0, close_pos=0.9, vol_ratio=1.2,
        version="v1", settle_deadline=NEXT_END,
    )


def _mk_s5_detector(monkeypatch, pm: dict, session: _S5Session,
                    now_ms: int = NEXT_START + 5_000) -> FakeBreakoutDetector:
    det = FakeBreakoutDetector(collector=_FakeCollector(), pm_15m_latest=pm)
    det._running = True
    monkeypatch.setattr(fbd, "async_session_factory", lambda: _S5Ctx(session))
    clock = MagicMock()
    clock.now_ms = MagicMock(return_value=now_ms)
    monkeypatch.setattr(fbd, "clock_sync", clock)
    return det


def _shadows(session: _S5Session) -> list:
    return [o for o in session.added if isinstance(o, PatternShadowSignal)]


@pytest.mark.asyncio
async def test_s5_deep_shadow_lands_on_z20(monkeypatch) -> None:
    """z5=−30bp ≤ −20bp + 报价匹配本周期 → 落 pattern_shadow，逐字段校验口径。"""
    pm = {"start_date": NEXT_START, "end_date": NEXT_END,
          "down_price": 0.63, "up_price": 0.36, "updated_ts": NEXT_START + 4_000}
    session = _S5Session(parent=_s5_parent(), exists=None)
    det = _mk_s5_detector(monkeypatch, pm, session)

    await det._fire_s5_signal(7, NEXT_START, NEXT_END, c5_close=99.7, anchor=100.0)

    shadows = _shadows(session)
    assert len(shadows) == 1
    sh = shadows[0]
    assert sh.version == S5_DEEP_VERSION == "s5_deep_z20_v1"
    assert sh.rule_text == S5_DEEP_RULE_TEXT
    assert sh.signal_bar_start == NEXT_START - 900_000  # 父 S1 周期起点（保唯一约束）
    assert sh.signal_bar_end == NEXT_START
    assert sh.target_bar_start == NEXT_START            # 结算目标根（次周期）
    assert sh.entry_state == "TOUCHED"                  # +5min 确认即入场，无触价等待
    assert sh.touch_ts == NEXT_START + 5_000
    assert sh.entry_down_quote == 0.63                  # 快照 +5min 真实 DOWN 报价
    assert sh.status == "PENDING"


@pytest.mark.asyncio
async def test_s5_deep_shadow_boundary_z_at_threshold(monkeypatch) -> None:
    """贴线：z5 取阈值 S5_DEEP_Z5（−20bp）→ 门禁含等号（<=），落深档影子。"""
    anchor = 100.0
    c5_close = anchor * (1.0 + S5_DEEP_Z5)          # z5 恰在 −20bp 阈值
    assert c5_close / anchor - 1.0 <= S5_DEEP_Z5    # 与门禁同源算术，含等号成立
    pm = {"start_date": NEXT_START, "down_price": 0.62, "up_price": 0.37,
          "updated_ts": NEXT_START + 4_000}
    session = _S5Session(parent=_s5_parent(), exists=None)
    det = _mk_s5_detector(monkeypatch, pm, session)

    await det._fire_s5_signal(7, NEXT_START, NEXT_END, c5_close=c5_close, anchor=anchor)

    assert len(_shadows(session)) == 1


@pytest.mark.asyncio
async def test_s5_deep_shadow_skips_shallow_z(monkeypatch) -> None:
    """z5=−10bp > −20bp：S5 主行照落，但深档影子不落（深回落子集才前向验证）。"""
    pm = {"start_date": NEXT_START, "down_price": 0.60, "up_price": 0.39,
          "updated_ts": NEXT_START + 4_000}
    session = _S5Session(parent=_s5_parent(), exists=None)
    det = _mk_s5_detector(monkeypatch, pm, session)

    await det._fire_s5_signal(7, NEXT_START, NEXT_END, c5_close=99.9, anchor=100.0)

    assert _shadows(session) == []
    # S5 主行仍落（浅回落也是确认信号，只是不进深档影子）
    assert any(isinstance(o, FakeBreakoutSignal) for o in session.added)


@pytest.mark.asyncio
async def test_s5_deep_shadow_skips_unmatched_quote(monkeypatch) -> None:
    """深回落(z5=−50bp)但报价属旧市场(start_date 不匹配)：无法定标 EV → 保守不落。"""
    pm = {"start_date": NEXT_START - 900_000, "down_price": 0.99, "up_price": 0.01,
          "updated_ts": NEXT_START - 60_000}
    session = _S5Session(parent=_s5_parent(), exists=None)
    det = _mk_s5_detector(monkeypatch, pm, session)

    await det._fire_s5_signal(7, NEXT_START, NEXT_END, c5_close=99.5, anchor=100.0)

    assert _shadows(session) == []  # z5 深，但 matched=False → 跳过


@pytest.mark.asyncio
async def test_s5_deep_shadow_idempotent(monkeypatch) -> None:
    """幂等：exists 探针命中（同 version+signal_bar_start 已有行）→ 不重复落、不 commit。"""
    session = _S5Session(exists=123)  # 已存在
    det = _mk_s5_detector(monkeypatch, {}, session)

    await det._fire_s5_deep_shadow(
        NEXT_START, NEXT_START + 5_000, z5=-0.003, down_quote=0.63,
    )

    assert _shadows(session) == []
    assert session.commits == 0


@pytest.mark.asyncio
async def test_s5_deep_shadow_none_quote_records_null(monkeypatch) -> None:
    """防御：down_quote=None → entry_down_quote 记 NULL（仍落行，matched 由调用方保证）。"""
    session = _S5Session(exists=None)
    det = _mk_s5_detector(monkeypatch, {}, session)

    await det._fire_s5_deep_shadow(
        NEXT_START, NEXT_START + 5_000, z5=-0.0025, down_quote=None,
    )

    shadows = _shadows(session)
    assert len(shadows) == 1
    assert shadows[0].entry_down_quote is None
    assert shadows[0].status == "PENDING"
