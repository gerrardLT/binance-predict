"""gap_crowd 影子检测器测试（2026-10-10 注册）。

覆盖：首触发落 PENDING / 同窗不重复 / 门未过不落表 / shadow_gate 下线停采 /
启动前窗口守卫 / K 线口径结算（SETTLED+win+ev）/ 平局 VOID / 超期 VOID /
结算幂等（非 PENDING 不重写）。
注意：`_settle_pending` 的 SQL 时间窗过滤（window_end <= now-grace）由生产 SQL
承担，本文件用替身会话按调用点验证状态机；SQL 过滤本身不在单测范围。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import binance_predict.services.gap_crowd_shadow_detector as gcd


class _FakeResult:
    def __init__(self, rows=None, scalar=None, first=None) -> None:
        self._rows = rows or []
        self._scalar = scalar
        self._first = first

    def scalars(self) -> "_FakeResult":
        return self

    def all(self) -> list:
        return self._rows

    def first(self):
        return self._first

    def scalar_one_or_none(self):
        return self._scalar


class _FakeSession:
    """脚本化会话：results 依次弹出（execute 每次取一个）。"""

    def __init__(self, results: list | None = None) -> None:
        self.results = list(results or [])
        self.added: list = []
        self.committed = 0

    async def execute(self, _stmt) -> _FakeResult:
        if self.results:
            return self.results.pop(0)
        return _FakeResult()

    def add(self, obj) -> None:
        self.added.append(obj)

    async def commit(self) -> None:
        self.committed += 1


class _FakeSessionCtx:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session

    async def __aenter__(self) -> _FakeSession:
        return self._session

    async def __aexit__(self, *exc) -> bool:
        return False


class _FakeCollector:
    """K 线结算替身：按 window_start 返回预置 bar（open/close）。"""

    def __init__(self, bars: dict[int, tuple[float, float]]) -> None:
        self.bars = bars
        self.calls: list[tuple[str, int]] = []

    async def fetch_klines_ending_at(self, interval: str, limit: int, end_ms: int) -> list[dict]:
        self.calls.append((interval, end_ms))
        out = []
        for start, (o, c) in self.bars.items():
            out.append({"open_time": start, "open": o, "high": max(o, c),
                        "low": min(o, c), "close": c, "volume": 1.0})
        return out


WS = 1_700_000_000_000          # 窗口起点（ms，2023-11 真实量级，晚于测试基准守卫由各例自设）
WE = WS + 300_000
WE15 = WS + 900_000


def _detector(monkeypatch, session: _FakeSession, collector=None) -> gcd.GapCrowdShadowDetector:
    d = gcd.GapCrowdShadowDetector(collector=collector)
    d._running = True
    d._started_ms = 0                      # 豁免「启动后才开始」守卫，逐例单独测守卫
    monkeypatch.setattr(gcd, "async_session_factory", lambda: _FakeSessionCtx(session))
    return d


def _ok_gate(monkeypatch, enabled: bool = True) -> None:
    monkeypatch.setattr(gcd.shadow_gate, "is_enabled", lambda _v: enabled)


@pytest.mark.asyncio
async def test_observe_sample_persists_pending(monkeypatch) -> None:
    """命中 → 落 PENDING 行（字段与触发快照一致，押 UP、真实报价入场）。"""
    session = _FakeSession([_FakeResult(first=None)])
    d = _detector(monkeypatch, session)
    _ok_gate(monkeypatch)
    d.observe_sample("5m", WS, WE, WS + 160_000, 0.40, 0.60, 53.0, 210.0, 12345.0)
    for t in list(d._persist_tasks):
        await t
    assert len(session.added) == 1
    row = session.added[0]
    assert row.version == "gap_crowd_5m_v1" and row.market_period == "5m"
    assert row.status == "PENDING" and row.direction == "UP"
    assert row.window_start == WS and row.window_end == WE
    assert row.trigger_ts == WS + 160_000
    assert row.gap == pytest.approx(-0.07, abs=1e-9)
    assert row.down_pct == pytest.approx(53.0)
    assert row.entry_up_price == pytest.approx(0.40)
    assert row.participants == 210.0 and row.trade_volume == 12345.0
    assert d._trigger_count == 1


@pytest.mark.asyncio
async def test_observe_sample_same_window_no_duplicate(monkeypatch) -> None:
    """同窗后续采样不再落表（首触发语义）。"""
    session = _FakeSession([_FakeResult(first=None)])
    d = _detector(monkeypatch, session)
    _ok_gate(monkeypatch)
    d.observe_sample("5m", WS, WE, WS + 160_000, 0.40, 0.60, 53.0)
    for t in list(d._persist_tasks):
        await t
    d.observe_sample("5m", WS, WE, WS + 175_000, 0.38, 0.62, 54.0)
    for t in list(d._persist_tasks):
        await t
    assert len(session.added) == 1


@pytest.mark.asyncio
async def test_observe_sample_gate_not_met_no_row(monkeypatch) -> None:
    """门未过（前半窗 / gap 不足 / 越界）→ 不落表。"""
    session = _FakeSession()
    d = _detector(monkeypatch, session)
    _ok_gate(monkeypatch)
    for ts, up, dn, dpct in (
        (WS + 100_000, 0.40, 0.60, 53.0),   # 前半窗
        (WS + 160_000, 0.45, 0.55, 53.0),   # gap=-0.02
        (WS + 160_000, 0.01, 0.99, 53.0),   # 报价越界
        (WS + 160_000, 0.40, 0.60, None),   # 缺 down_pct
    ):
        d.observe_sample("5m", WS, WE, ts, up, dn, dpct)
    for t in list(d._persist_tasks):
        await t
    assert session.added == [] and d._trigger_count == 0


@pytest.mark.asyncio
async def test_observe_sample_shadow_gate_offline(monkeypatch) -> None:
    """shadow_gate 下线 → 停止采集该版本（命中也不落表）。"""
    session = _FakeSession([_FakeResult(first=None)])
    d = _detector(monkeypatch, session)
    _ok_gate(monkeypatch, enabled=False)
    d.observe_sample("5m", WS, WE, WS + 160_000, 0.40, 0.60, 53.0)
    for t in list(d._persist_tasks):
        await t
    assert session.added == []


@pytest.mark.asyncio
async def test_observe_sample_startup_window_guard(monkeypatch) -> None:
    """启动前已开始的窗口不做半途入场（研究只验证首触发）。"""
    session = _FakeSession()
    d = _detector(monkeypatch, session)
    d._started_ms = WS + 1               # 窗口起点早于启动时刻
    _ok_gate(monkeypatch)
    d.observe_sample("5m", WS, WE, WS + 160_000, 0.40, 0.60, 53.0)
    for t in list(d._persist_tasks):
        await t
    assert session.added == []


@pytest.mark.asyncio
async def test_settle_one_up_win(monkeypatch) -> None:
    """结算：K 线 close>open → outcome=UP，押 UP 命中，ev=0.98/q−1。"""
    row = SimpleNamespace(id=1, version="gap_crowd_5m_v1", market_period="5m",
                          window_start=WS, window_end=WE, direction="UP",
                          entry_up_price=0.40, status="PENDING",
                          settle_outcome=None, win=None, ev_at_entry=None)
    collector = _FakeCollector({WS: (100.0, 100.5)})
    session = _FakeSession([_FakeResult(scalar=row)])   # _finalize 的 scalar_one_or_none
    d = _detector(monkeypatch, session, collector=collector)
    await d._settle_one(row, now_ms=WE + 30_000)
    assert row.status == "SETTLED" and row.settle_outcome == "UP" and row.win is True
    assert row.ev_at_entry == pytest.approx(0.98 / 0.40 - 1.0, rel=1e-9)
    assert collector.calls == [("5m", WE)]


@pytest.mark.asyncio
async def test_settle_one_down_loss(monkeypatch) -> None:
    """结算：close<open → outcome=DOWN，押 UP 未命中，ev=−1。"""
    row = SimpleNamespace(id=2, version="gap_crowd_5m_v1", market_period="5m",
                          window_start=WS, window_end=WE, direction="UP",
                          entry_up_price=0.42, status="PENDING",
                          settle_outcome=None, win=None, ev_at_entry=None)
    collector = _FakeCollector({WS: (100.5, 100.0)})
    session = _FakeSession([_FakeResult(scalar=row)])
    d = _detector(monkeypatch, session, collector=collector)
    await d._settle_one(row, now_ms=WE + 30_000)
    assert row.status == "SETTLED" and row.settle_outcome == "DOWN" and row.win is False
    assert row.ev_at_entry == pytest.approx(-1.0)


@pytest.mark.asyncio
async def test_settle_one_tie_void(monkeypatch) -> None:
    """精确平局（close==open）→ VOID（与研究剔除平局同口径，不污染统计）。"""
    row = SimpleNamespace(id=3, version="gap_crowd_5m_v1", market_period="5m",
                          window_start=WS, window_end=WE, direction="UP",
                          entry_up_price=0.40, status="PENDING",
                          settle_outcome=None, win=None, ev_at_entry=None)
    collector = _FakeCollector({WS: (100.0, 100.0)})
    session = _FakeSession([_FakeResult(scalar=row)])
    d = _detector(monkeypatch, session, collector=collector)
    await d._settle_one(row, now_ms=WE + 30_000)
    assert row.status == "VOID" and row.win is None and row.ev_at_entry is None


@pytest.mark.asyncio
async def test_settle_one_expired_void_without_kline(monkeypatch) -> None:
    """超 24h 仍 PENDING（K 线缺失/宕机）→ VOID，不无限挂账。"""
    row = SimpleNamespace(id=4, version="gap_crowd_5m_v1", market_period="5m",
                          window_start=WS, window_end=WE, direction="UP",
                          entry_up_price=0.40, status="PENDING",
                          settle_outcome=None, win=None, ev_at_entry=None)
    collector = _FakeCollector({})                      # 无 bar
    session = _FakeSession([_FakeResult(scalar=row)])
    d = _detector(monkeypatch, session, collector=collector)
    await d._settle_one(row, now_ms=WE + int(86_400_000) + 1)
    assert row.status == "VOID"
    assert collector.calls == []                        # 超期直接 VOID，不再拉 K 线


@pytest.mark.asyncio
async def test_settle_one_kline_not_ready_stays_pending(monkeypatch) -> None:
    """K 线未就绪（窗口 bar 不在返回集）→ 保持 PENDING 下轮重试。"""
    row = SimpleNamespace(id=5, version="gap_crowd_5m_v1", market_period="5m",
                          window_start=WS, window_end=WE, direction="UP",
                          entry_up_price=0.40, status="PENDING",
                          settle_outcome=None, win=None, ev_at_entry=None)
    collector = _FakeCollector({WS - 300_000: (100.0, 101.0)})   # 只有上一根
    session = _FakeSession()
    d = _detector(monkeypatch, session, collector=collector)
    await d._settle_one(row, now_ms=WE + 30_000)
    assert row.status == "PENDING"


@pytest.mark.asyncio
async def test_finalize_idempotent_on_non_pending(monkeypatch) -> None:
    """结算幂等：非 PENDING 行不被重写（重复结算/并发保护）。"""
    row = SimpleNamespace(id=6, version="gap_crowd_5m_v1", market_period="5m",
                          window_start=WS, window_end=WE, direction="UP",
                          entry_up_price=0.40, status="SETTLED",
                          settle_outcome="UP", win=True, ev_at_entry=1.45)
    session = _FakeSession([_FakeResult(scalar=row)])
    d = _detector(monkeypatch, session)
    await d._finalize(6, status="VOID", outcome=None, win=None, ev=None)
    assert row.status == "SETTLED" and row.settle_outcome == "UP" and row.win is True
    assert session.committed == 0


@pytest.mark.asyncio
async def test_settle_pending_empty_scan(monkeypatch) -> None:
    """无 PENDING 行 → 空转不报错。"""
    session = _FakeSession([_FakeResult(rows=[])])
    d = _detector(monkeypatch, session)
    await d._settle_pending()
    assert d._settled_count == 0 and d._void_count == 0


def test_status_shape() -> None:
    d = gcd.GapCrowdShadowDetector()
    s = d.status()
    assert set(s) == {"running", "trigger_count", "settled_count", "void_count", "fired_windows"}
    assert s["running"] is False and s["trigger_count"] == 0


def test_specs_and_window_units() -> None:
    """周期/窗口映射与冻结常量（单位契约：down_pct 为百分比 0-100）。"""
    assert gcd.GAP_CROWD_SPECS == {"gap_crowd_5m_v1": "5m", "gap_crowd_15m_v1": "15m"}
    assert gcd.WINDOW_MS == {"5m": 300_000, "15m": 900_000}
    assert gcd.GAP_THRESHOLD == -0.05 and gcd.LATE_FRACTION == 0.5
    assert gcd.MIN_REMAIN_S == 60.0 and (gcd.Q_LO, gcd.Q_HI) == (0.02, 0.98)
    assert gcd.QUOTE_SUM_TOL == 0.05   # AMENDMENT-2 一致性门容差


@pytest.mark.asyncio
async def test_observe_sample_incoherent_quote_no_row(monkeypatch) -> None:
    """报价对不一致快照不落表（AMENDMENT-2 硬前置；生产实况 up=0.40/dn=0.27）。"""
    session = _FakeSession()
    d = _detector(monkeypatch, session)
    _ok_gate(monkeypatch)
    d.observe_sample("5m", WS, WE, WS + 160_000, 0.40, 0.27, 27.5)
    for t in list(d._persist_tasks):
        await t
    assert session.added == [] and d._trigger_count == 0
