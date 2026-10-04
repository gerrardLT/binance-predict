"""S4 跳过续发（实盘+影子同口径，检测器层）与 S4 派生影子（延迟入场 / 高信心档，record-only）。

续发 = 上一根 15m 本身也是正式 S4 触发（上一窗押 DOWN 已输、连阳延长再触发）。
不触 DB/网络：collector 替身返回固定 K 线，_fire_confirmed_signal 捕获替身。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import binance_predict.services.s4_variant_shadow_detector as s4v
from binance_predict.services.fake_breakout_detector import (
    FakeBreakoutDetector,
    is_s4_hiconf,
    s4_is_refire,
)
from binance_predict.services.s4_variant_shadow_detector import (
    S4_DELAY_VERSION,
    S4_HICONF_VERSION,
    S4VariantShadowDetector,
    first_minute_dropped,
)

M15 = 900_000


def _bar(cycle: int, o: float, h: float, l: float, c: float, v: float = 40.0) -> dict:
    return {"open_time": cycle * M15, "open": o, "high": h, "low": l, "close": c, "volume": v}


def _bull(cycle: int, base: float, cp: float = 0.95) -> dict:
    """收阳光头 K：振幅 10，收盘位置 cp。"""
    return _bar(cycle, base, base + 10, base, base + 10 * cp)


def _bear(cycle: int, base: float) -> dict:
    return _bar(cycle, base + 5, base + 6, base, base + 1)


class _Collector:
    """按调用次序返回预设 15m K 线序列（每次 _confirm_and_fire 一次 fetch）。"""

    store = SimpleNamespace(mid_price=100.0)

    def __init__(self, seqs: list[list[dict]]) -> None:
        self._seqs = list(seqs)

    async def fetch_recent_klines(self, interval: str, limit: int) -> list[dict]:
        return self._seqs.pop(0)


def _history(end_cycle: int) -> list[dict]:
    """end_cycle 之前 20 根阴线（不构成连阳），满足均量窗口。"""
    return [_bear(end_cycle - 20 + i, 90.0) for i in range(20)]


def _det(seqs: list[list[dict]]) -> tuple[FakeBreakoutDetector, list[dict], list[dict], list[dict]]:
    d = FakeBreakoutDetector(collector=_Collector(seqs), pm_15m_latest={})
    fired, delay, hiconf = [], [], []

    async def fake_fire(side, rec, sig_k, close_pos, vol_ratio, cur_cycle, now_ms,
                        version="v1", shadow=False, pattern_type=None):
        fired.append({"cycle": rec["cycle_id"], "pattern_type": pattern_type, "shadow": shadow})

    d._fire_confirmed_signal = fake_fire  # type: ignore[method-assign]
    d._on_s4_delay = delay.append
    d._on_s4_hiconf = hiconf.append
    return d, fired, delay, hiconf


def _seq_upto(cycle: int, n_bull: int) -> list[dict]:
    """以 cycle 为信号根、末尾连续 n_bull 根阳线（含信号根）的 21 根 K 线。"""
    hist = _history(cycle - n_bull + 1)
    bulls = [_bull(cycle - n_bull + 1 + i, 100.0 + 10 * i) for i in range(n_bull)]
    return (hist + bulls)[-21:]


# ---------------- 纯函数 ----------------

def test_s4_is_refire_uses_memory_when_previous_cycle_evaluated() -> None:
    assert s4_is_refire(101, 100, 100, prev_bar_shape=False) is True   # 上一周期命中
    assert s4_is_refire(101, 100, 99, prev_bar_shape=True) is False    # 上一周期评估过但未命中（如破位）
    assert s4_is_refire(101, 100, None, prev_bar_shape=True) is False


def test_s4_is_refire_cold_start_falls_back_to_kline_shape() -> None:
    assert s4_is_refire(101, None, None, prev_bar_shape=True) is True
    assert s4_is_refire(101, 98, 98, prev_bar_shape=False) is False    # 跳变：不连续 → K 线口径


def test_is_s4_hiconf_boundaries() -> None:
    o, h, l = 100.0, 110.0, 100.0
    assert is_s4_hiconf(o, h, l, 109.0, [1, 1, 1, 1]) is True            # 连阳5，cp=0.9 贴线含
    assert is_s4_hiconf(o, h, l, 108.99, [1, 1, 1, 1]) is False          # cp<0.9
    assert is_s4_hiconf(o, h, l, 109.5, [1, 1, 1]) is False              # 连阳仅4
    assert is_s4_hiconf(o, h, l, 109.5, [-1, 1, 1, 1, 1]) is True        # 只看最近4根
    assert is_s4_hiconf(110.0, 110.0, 100.0, 105.0, [1, 1, 1, 1]) is False  # 收阴


def test_first_minute_dropped_requires_strict_drop() -> None:
    ws = 101 * M15
    assert first_minute_dropped({"open_time": ws, "open": 100.0, "close": 99.9}, ws) is True
    assert first_minute_dropped({"open_time": ws, "open": 100.0, "close": 100.0}, ws) is False  # 平盘不算
    assert first_minute_dropped({"open_time": ws, "open": 100.0, "close": 100.1}, ws) is False
    assert first_minute_dropped({"open_time": ws - 60_000, "open": 100.0, "close": 99.0}, ws) is False
    assert first_minute_dropped(None, ws) is False


# ---------------- 检测器：跳过续发 ----------------

@pytest.mark.asyncio
async def test_refire_chain_skips_second_and_third_trigger() -> None:
    """连阳 3→4→5 连续三个周期：只首发下单，续发两根跳过；续发不派生延迟入场影子。"""
    d, fired, delay, _ = _det([_seq_upto(100, 3), _seq_upto(101, 4), _seq_upto(102, 5)])
    for c in (100, 101, 102):
        await d._confirm_and_fire({}, prev_cycle=c, cur_cycle=c + 1, now_ms=(c + 1) * M15, retry=0)
    assert [f["cycle"] for f in fired if f["pattern_type"] == "momentum_fade"] == [100]
    assert d.status_snapshot["s4_refire_skip_count"] == 2
    assert [p["market_start_15m"] for p in delay] == [101 * M15]


@pytest.mark.asyncio
async def test_first_fire_after_break_in_chain_is_not_refire() -> None:
    """上一周期连阳被阴线打断 → 本周期重新连阳 3 根是首发，正常下单。"""
    seq_a = _seq_upto(100, 3)
    seq_b = (_history(102) + [_bear(102, 90.0)] + [_bull(103 + i, 100.0 + 10 * i) for i in range(3)])[-21:]
    d, fired, _, _ = _det([seq_a, seq_b])
    await d._confirm_and_fire({}, prev_cycle=100, cur_cycle=101, now_ms=101 * M15, retry=0)
    await d._confirm_and_fire({}, prev_cycle=105, cur_cycle=106, now_ms=106 * M15, retry=0)
    assert [f["cycle"] for f in fired if f["pattern_type"] == "momentum_fade"] == [100, 105]
    assert d.status_snapshot["s4_refire_skip_count"] == 0


@pytest.mark.asyncio
async def test_previous_cycle_break_hit_is_not_refire_when_evaluated() -> None:
    """上一周期 S4 形态成立但处于 high 侧破位（走 S1 不走 S4）→ 本周期不算续发（研究口径 A）。"""
    d, fired, _, _ = _det([_seq_upto(100, 3), _seq_upto(101, 4)])
    due_break = {"high": {"cycle_id": 100, "level": "4h", "broken_level": 99.0,
                          "break_price": 101.0, "break_time": 0}}
    await d._confirm_and_fire(due_break, prev_cycle=100, cur_cycle=101, now_ms=101 * M15, retry=0)
    await d._confirm_and_fire({}, prev_cycle=101, cur_cycle=102, now_ms=102 * M15, retry=0)
    assert [f["cycle"] for f in fired if f["pattern_type"] == "momentum_fade"] == [101]


@pytest.mark.asyncio
async def test_cold_start_uses_kline_shape_for_refire() -> None:
    """进程冷启动后首个周期：上一根 K 本身是 S4 形态 → 判续发跳过（保守，定义 B）。"""
    d, fired, _, _ = _det([_seq_upto(101, 4)])
    await d._confirm_and_fire({}, prev_cycle=101, cur_cycle=102, now_ms=102 * M15, retry=0)
    assert [f for f in fired if f["pattern_type"] == "momentum_fade"] == []
    assert d.status_snapshot["s4_refire_skip_count"] == 1


@pytest.mark.asyncio
async def test_cold_start_first_fire_when_previous_bar_not_s4() -> None:
    """冷启动首周期且上一根只有 1 根前置阳线（不构成 S4）→ 正常首发。"""
    d, fired, _, _ = _det([_seq_upto(101, 3)])
    await d._confirm_and_fire({}, prev_cycle=101, cur_cycle=102, now_ms=102 * M15, retry=0)
    assert [f["cycle"] for f in fired if f["pattern_type"] == "momentum_fade"] == [101]


# ---------------- 检测器：高信心档派生 ----------------

@pytest.mark.asyncio
async def test_hiconf_fires_independently_including_break_cycle_and_refire() -> None:
    """高信心档：连阳≥5∧cp≥0.9 每周期独立派生，破位周期与续发周期都记录；连阳4不派生。"""
    d, _, _, hiconf = _det([_seq_upto(100, 4), _seq_upto(101, 5)])
    await d._confirm_and_fire({}, prev_cycle=100, cur_cycle=101, now_ms=101 * M15, retry=0)
    due_break = {"high": {"cycle_id": 101, "level": "4h", "broken_level": 99.0,
                          "break_price": 101.0, "break_time": 0}}
    await d._confirm_and_fire(due_break, prev_cycle=101, cur_cycle=102, now_ms=102 * M15, retry=0)
    assert len(hiconf) == 1
    assert hiconf[0]["market_start_15m"] == 102 * M15
    assert hiconf[0]["signal_bar_start"] == 101 * M15
    assert hiconf[0]["hi_break"] is True


@pytest.mark.asyncio
async def test_variant_hook_exception_does_not_break_detection() -> None:
    """首发 S4（连阳3，冷启动上一根非 S4 形态）派生延迟入场钩子抛错 → 正式信号照常。"""
    d, fired, _, _ = _det([_seq_upto(100, 3)])

    def _boom(_p):
        raise RuntimeError("hook down")

    d._on_s4_delay = _boom
    d._on_s4_hiconf = _boom
    await d._confirm_and_fire({}, prev_cycle=100, cur_cycle=101, now_ms=101 * M15, retry=0)
    assert [f["cycle"] for f in fired if f["pattern_type"] == "momentum_fade"] == [100]


# ---------------- 派生影子检测器：落库 / 门禁 / 结算 ----------------

class _Sess:
    def __init__(self, exists=None, rows=None) -> None:
        self.exists, self.rows, self.added, self.commits = exists, rows or [], [], 0

    async def execute(self, _stmt):
        s = self

        class _R:
            def scalar_one_or_none(self_inner):
                return s.exists

            def scalars(self_inner):
                return self_inner

            def all(self_inner):
                return s.rows
        return _R()

    def add(self, obj) -> None:
        self.added.append(obj)

    async def commit(self) -> None:
        self.commits += 1


class _Ctx:
    def __init__(self, s: _Sess) -> None:
        self.s = s

    async def __aenter__(self):
        return self.s

    async def __aexit__(self, *exc):
        return False


class _C1m:
    def __init__(self, bar: dict | None) -> None:
        self.bar = bar

    async def fetch_recent_klines(self, interval: str, limit: int) -> list[dict]:
        return [self.bar] if self.bar else []


def _patch(monkeypatch, sess: _Sess, now: int) -> None:
    monkeypatch.setattr(s4v, "async_session_factory", lambda: _Ctx(sess))
    monkeypatch.setattr(s4v.shadow_gate, "is_enabled", lambda _v: True)
    monkeypatch.setattr(s4v.clock_sync, "now_ms", lambda: now)


WS = 101 * M15
PM = {"start_date": WS, "up_price": 0.38, "down_price": 0.62, "updated_ts": WS + 61_000}
SIG = {"signal_bar_start": 100 * M15, "market_start_15m": WS, "market_end_15m": WS + M15, "close_pos": 0.93}


@pytest.mark.asyncio
async def test_delay_records_when_first_minute_dropped(monkeypatch) -> None:
    sess = _Sess()
    _patch(monkeypatch, sess, WS + 70_000)
    det = S4VariantShadowDetector(_C1m({"open_time": WS, "open": 100.0, "close": 99.8}), PM)
    det._running = True
    await det._confirm(S4_DELAY_VERSION, SIG)
    assert len(sess.added) == 1
    row = sess.added[0]
    assert row.version == S4_DELAY_VERSION and row.direction == "DOWN"
    assert row.target_bar_start == WS and row.signal_bar_start == 100 * M15
    assert row.entry_down_price == 0.62
    assert row.feature_snapshot["m1_close"] == 99.8


@pytest.mark.asyncio
async def test_delay_skips_when_first_minute_not_dropped(monkeypatch) -> None:
    sess = _Sess()
    _patch(monkeypatch, sess, WS + 70_000)
    det = S4VariantShadowDetector(_C1m({"open_time": WS, "open": 100.0, "close": 100.0}), PM)
    det._running = True
    await det._confirm(S4_DELAY_VERSION, SIG)
    assert sess.added == []


@pytest.mark.asyncio
async def test_hiconf_records_and_is_idempotent(monkeypatch) -> None:
    sess = _Sess()
    _patch(monkeypatch, sess, WS + 10_000)
    det = S4VariantShadowDetector(_C1m(None), PM)
    det._running = True
    await det._confirm(S4_HICONF_VERSION, {**SIG, "hi_break": True})
    assert len(sess.added) == 1 and sess.added[0].version == S4_HICONF_VERSION
    assert sess.added[0].feature_snapshot["hi_break"] is True
    dup = _Sess(exists=7)
    monkeypatch.setattr(s4v, "async_session_factory", lambda: _Ctx(dup))
    await det._confirm(S4_HICONF_VERSION, SIG)
    assert dup.added == []


@pytest.mark.asyncio
async def test_settle_down_win_and_flat_noise(monkeypatch) -> None:
    r_win = SimpleNamespace(version=S4_DELAY_VERSION, target_bar_start=WS, direction="DOWN", status="PENDING",
                            settle_outcome=None, win=None, settle_open=None, settle_close=None, settled_at=None)
    r_flat = SimpleNamespace(**{**vars(r_win), "target_bar_start": WS + M15})
    sess = _Sess(rows=[r_win, r_flat])
    monkeypatch.setattr(s4v, "async_session_factory", lambda: _Ctx(sess))
    det = S4VariantShadowDetector(_C1m(None), {})
    await det.settle([{"open_time": WS, "open": 100.0, "close": 99.0},
                      {"open_time": WS + M15, "open": 100.0, "close": 100.0}])
    assert (r_win.status, r_win.settle_outcome, r_win.win) == ("SETTLED", "DOWN", True)
    assert (r_flat.status, r_flat.settle_outcome, r_flat.win) == ("EXPIRED", "NOISE", None)


def test_spawn_dedup_and_gate(monkeypatch) -> None:
    det = S4VariantShadowDetector(_C1m(None), {})
    monkeypatch.setattr(s4v.shadow_gate, "is_enabled", lambda _v: False)
    det._running = True
    det.on_hiconf(SIG)
    assert det._watched == set()  # 影子下线 → 不派生
    det._running = False
    monkeypatch.setattr(s4v.shadow_gate, "is_enabled", lambda _v: True)
    det.on_hiconf(SIG)
    assert det._watched == set()  # 未启动 → 不派生
