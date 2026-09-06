"""S2 条件单影子检测器测试：纯函数判定 + 入场确认编排 + 报价快照 + 结算方向 + 幂等/闸 + 物理隔离。

不触网络/真实 DB：collector/session/clock 全用替身。口径与审计锚点
scripts/s2_cond_freeze_counts_720d.py 逐字一致——t=4 价<开（全深度）/ t=5 剔深
（0<ln(open/px)<15bp），均由实盘 S2(bear_exhaust) 派生、押次周期 15m UP（收阳赢）。
"""
from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace

import pytest

import binance_predict.services.s2_cond_shadow_detector as s2c
from binance_predict.services.s2_cond_shadow_detector import (
    DIP_MAX,
    S2_COND_VERSIONS,
    VAR_T4,
    VAR_T5D,
    S2CondShadowDetector,
    dip,
    judge_t4,
    judge_t5d,
)

BAR_MS_15M = 900_000
NEXT_START = 1_700_000_000_000 // BAR_MS_15M * BAR_MS_15M  # 对齐 15m 网格
NEXT_END = NEXT_START + BAR_MS_15M


# ============================================================
# 纯函数判定（冻结口径 + 边界）
# ============================================================

def test_judge_t4_below_open_hits() -> None:
    assert judge_t4(100.0, 99.0) is True       # px4<open → 命中（全深度回落）


def test_judge_t4_equal_open_misses() -> None:
    assert judge_t4(100.0, 100.0) is False     # px==open 边界 → 严格< 不命中


def test_judge_t4_above_open_misses() -> None:
    assert judge_t4(100.0, 101.0) is False


def test_dip_sign_and_zero() -> None:
    assert dip(100.0, 100.0) == 0.0            # 价==开 → 0
    assert dip(100.0, 99.0) > 0                # 价<开 → 正回落
    assert dip(100.0, 101.0) < 0               # 价>开 → 负


def test_judge_t5d_mid_dip_hits() -> None:
    px5 = 100.0 * math.exp(-0.001)             # dip=10bp ∈ (0,15bp)
    assert judge_t5d(100.0, px5) is True


def test_judge_t5d_zero_dip_misses() -> None:
    assert judge_t5d(100.0, 100.0) is False    # dip==0 边界（价==开，非回落）→ 严格> 不命中


def test_judge_t5d_just_under_15bp_hits() -> None:
    px5 = 100.0 * math.exp(-(DIP_MAX - 1e-6))  # dip≈14.99bp < 15bp → 命中
    assert judge_t5d(100.0, px5) is True


def test_judge_t5d_just_over_15bp_misses() -> None:
    px5 = 100.0 * math.exp(-(DIP_MAX + 1e-6))  # dip≈15.01bp > 15bp → 剔深排除
    assert judge_t5d(100.0, px5) is False


def test_judge_t5d_above_open_misses() -> None:
    assert judge_t5d(100.0, 101.0) is False    # dip<0（价>开）→ 不命中


# ============================================================
# DB 编排替身（不触真实库；镜像 test_nextbar_shadow_detector）
# ============================================================

class _FakeResult:
    def __init__(self, rows: list, scalar=None) -> None:
        self._rows, self._scalar = rows, scalar

    def scalars(self) -> "_FakeResult":
        return self

    def all(self) -> list:
        return self._rows

    def scalar_one_or_none(self):
        return self._scalar


class _FakeSession:
    def __init__(self, rows: list | None = None, scalar=None) -> None:
        self.rows, self.scalar = rows or [], scalar
        self.added: list = []
        self.committed = False

    async def execute(self, _stmt) -> _FakeResult:
        return _FakeResult(self.rows, self.scalar)

    def add(self, obj) -> None:
        self.added.append(obj)

    async def commit(self) -> None:
        self.committed = True


class _CapSession(_FakeSession):
    """捕获传入 execute 的语句（供隔离测试编译校验 WHERE version 过滤）。"""

    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)
        self.stmt = None

    async def execute(self, stmt) -> _FakeResult:
        self.stmt = stmt
        return await super().execute(stmt)


class _FakeSessionCtx:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session

    async def __aenter__(self) -> _FakeSession:
        return self._session

    async def __aexit__(self, *exc) -> bool:
        return False


class _FakeCollector:
    def __init__(self, bars_1m: list[dict]) -> None:
        self._bars = bars_1m

    async def fetch_recent_klines(self, tf: str, n: int) -> list[dict]:
        return list(self._bars) if tf == "1m" else []


def _make_1m_bars(next_start: int, open_px: float, px4: float, px5: float) -> list[dict]:
    """合成次周期前 5 根 1m K：offset0.open=周期开盘；offset180s.close=px4；offset240s.close=px5。"""
    def bar(off: int, o: float, c: float) -> dict:
        return {"open_time": next_start + off, "open": o, "high": max(o, c),
                "low": min(o, c), "close": c, "volume": 1.0}
    return [
        bar(0, open_px, open_px),
        bar(60_000, open_px, open_px),
        bar(120_000, open_px, open_px),
        bar(180_000, open_px, px4),   # 该根 close = t=4 价
        bar(240_000, px4, px5),       # 该根 close = t=5 价
    ]


def _pending(version: str, target_start: int) -> SimpleNamespace:
    return SimpleNamespace(
        version=version, signal_bar_start=target_start - BAR_MS_15M,
        target_bar_start=target_start, status="PENDING", win=None, direction="UP",
        settle_outcome=None, settle_open=None, settle_close=None, settled_at=None,
    )


def _patch_clock(monkeypatch, now_ms: int) -> None:
    """固定时钟 + 打桩 _sleep_until 为 no-op（免真实睡眠；fetch 死线由 now_ms 控制）。"""
    monkeypatch.setattr(s2c.clock_sync, "now_ms", lambda: now_ms)

    async def _noop_sleep(self, target_ms: int) -> None:
        return None

    monkeypatch.setattr(s2c.S2CondShadowDetector, "_sleep_until", _noop_sleep)


# ============================================================
# 入场确认编排（_confirm_entry：合成 1m K + 报价 → 落对行）
# ============================================================

@pytest.mark.asyncio
async def test_confirm_entry_both_variants_hit(monkeypatch) -> None:
    """open=100/px4=99（t4 命中）/px5=99.9（dip=10bp，t5d 命中）→ 落 2 行，坐标/方向/报价齐。"""
    session = _FakeSession(scalar=None)  # 不存在 → 落行
    monkeypatch.setattr(s2c, "async_session_factory", lambda: _FakeSessionCtx(session))
    now = NEXT_START + 300_000           # 落在两 fetch 死线（+330k/+390k）内
    _patch_clock(monkeypatch, now)
    collector = _FakeCollector(_make_1m_bars(NEXT_START, 100.0, 99.0, 99.9))
    cache = {"start_date": NEXT_START, "up_price": 0.42, "down_price": 0.60,
             "updated_ts": now - 5_000}  # 窗口对齐 + 龄 5s ≤20s
    d = S2CondShadowDetector(collector=collector, pm_15m_latest=cache)
    d._running = True
    await d._confirm_entry(parent_id=7, next_start=NEXT_START, next_end=NEXT_END)

    assert sorted(r.version for r in session.added) == [VAR_T4, VAR_T5D]
    for r in session.added:
        assert r.direction == "UP"
        assert r.target_bar_start == NEXT_START
        assert r.signal_bar_start == NEXT_START - BAR_MS_15M  # 父 S2 根（保唯一约束）
        assert r.signal_bar_end == NEXT_START
        assert r.timeframe == "15m" and r.status == "PENDING"
        assert r.entry_up_price == 0.42 and r.entry_quote_ts == now - 5_000
        assert r.feature_snapshot["parent_id"] == 7
        assert r.feature_snapshot["open"] == 100.0


@pytest.mark.asyncio
async def test_confirm_entry_t4_only_when_t5d_too_deep(monkeypatch) -> None:
    """px5=98（dip≈202bp>15bp，t5d 剔深排除）而 px4=99（t4 命中）→ 只落 VAR_T4 一行。"""
    session = _FakeSession(scalar=None)
    monkeypatch.setattr(s2c, "async_session_factory", lambda: _FakeSessionCtx(session))
    _patch_clock(monkeypatch, NEXT_START + 300_000)
    collector = _FakeCollector(_make_1m_bars(NEXT_START, 100.0, 99.0, 98.0))
    cache = {"start_date": NEXT_START, "up_price": 0.40, "down_price": 0.62,
             "updated_ts": NEXT_START + 295_000}
    d = S2CondShadowDetector(collector=collector, pm_15m_latest=cache)
    d._running = True
    await d._confirm_entry(1, NEXT_START, NEXT_END)
    assert [r.version for r in session.added] == [VAR_T4]


@pytest.mark.asyncio
async def test_confirm_entry_no_hit_no_record(monkeypatch) -> None:
    """px4=101>open（t4 不命中）、px5=101（dip<0，t5d 不命中）→ 零落库。"""
    session = _FakeSession(scalar=None)
    monkeypatch.setattr(s2c, "async_session_factory", lambda: _FakeSessionCtx(session))
    _patch_clock(monkeypatch, NEXT_START + 300_000)
    collector = _FakeCollector(_make_1m_bars(NEXT_START, 100.0, 101.0, 101.0))
    d = S2CondShadowDetector(collector=collector, pm_15m_latest={})
    d._running = True
    await d._confirm_entry(1, NEXT_START, NEXT_END)
    assert session.added == []


@pytest.mark.asyncio
async def test_confirm_entry_no_bars_past_deadline_no_record(monkeypatch) -> None:
    """时钟已过两 fetch 死线且无 1m 数据 → 直接放弃、零落库（不回补、不阻塞）。"""
    session = _FakeSession(scalar=None)
    monkeypatch.setattr(s2c, "async_session_factory", lambda: _FakeSessionCtx(session))
    _patch_clock(monkeypatch, NEXT_START + 500_000)  # > +330k/+390k 死线
    d = S2CondShadowDetector(collector=_FakeCollector([]), pm_15m_latest={})
    d._running = True
    await d._confirm_entry(1, NEXT_START, NEXT_END)
    assert session.added == []


# ============================================================
# 报价快照守卫（窗口对齐 / 错位 / 超龄）
# ============================================================

def test_snapshot_up_quote_aligned() -> None:
    now = NEXT_START + 300_000
    d = S2CondShadowDetector(collector=None, pm_15m_latest={
        "start_date": NEXT_START, "up_price": 0.44, "down_price": 0.58,
        "updated_ts": now - 3_000,
    })
    d_now = s2c.clock_sync.now_ms
    try:
        s2c.clock_sync.now_ms = lambda: now  # type: ignore[assignment]
        up, down, ts = d._snapshot_up_quote(NEXT_START)
    finally:
        s2c.clock_sync.now_ms = d_now  # type: ignore[assignment]
    assert (up, down, ts) == (0.44, 0.58, now - 3_000)


def test_snapshot_up_quote_misaligned_window() -> None:
    """缓存停在下一窗（start_date≠next_start）→ 报价留空（行仍落、EV 不计）。"""
    d = S2CondShadowDetector(collector=None, pm_15m_latest={
        "start_date": NEXT_START + BAR_MS_15M, "up_price": 0.44,
        "down_price": 0.58, "updated_ts": s2c.clock_sync.now_ms(),
    })
    assert d._snapshot_up_quote(NEXT_START) == (None, None, None)


def test_snapshot_up_quote_stale() -> None:
    """报价超龄（now−updated_ts>20s）→ 留空（保守，防旧市场残值）。"""
    now = s2c.clock_sync.now_ms()
    d = S2CondShadowDetector(collector=None, pm_15m_latest={
        "start_date": NEXT_START, "up_price": 0.44, "down_price": 0.58,
        "updated_ts": now - 30_000,
    })
    # start_date 需与 next_start 对齐才走到龄守卫；此处用当前墙钟构造 next_start
    assert d._snapshot_up_quote(NEXT_START) == (None, None, None)


def test_snapshot_up_quote_out_of_range() -> None:
    """up 报价越界（∉(0,1)）→ 留空（防脏值污染 EV）。"""
    now = NEXT_START + 300_000
    d = S2CondShadowDetector(collector=None, pm_15m_latest={
        "start_date": NEXT_START, "up_price": 1.5, "down_price": 0.5,
        "updated_ts": now - 3_000,
    })
    d_now = s2c.clock_sync.now_ms
    try:
        s2c.clock_sync.now_ms = lambda: now  # type: ignore[assignment]
        assert d._snapshot_up_quote(NEXT_START) == (None, None, None)
    finally:
        s2c.clock_sync.now_ms = d_now  # type: ignore[assignment]


# ============================================================
# 结算（UP 方向感知：收阳赢 / 收阴输 / 平盘 NOISE）
# ============================================================

@pytest.mark.asyncio
async def test_settle_win_on_green(monkeypatch) -> None:
    session = _FakeSession(rows=[_pending(VAR_T4, NEXT_START)])
    monkeypatch.setattr(s2c, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = S2CondShadowDetector(collector=None, pm_15m_latest={})
    await d._settle_pending([{"open_time": NEXT_START, "open": 100.0, "close": 101.5}])
    sig = session.rows[0]
    assert sig.status == "SETTLED" and sig.win is True and sig.settle_outcome == "UP"
    assert sig.settle_open == 100.0 and sig.settle_close == 101.5 and session.committed


@pytest.mark.asyncio
async def test_settle_lose_on_red(monkeypatch) -> None:
    session = _FakeSession(rows=[_pending(VAR_T5D, NEXT_START)])
    monkeypatch.setattr(s2c, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = S2CondShadowDetector(collector=None, pm_15m_latest={})
    await d._settle_pending([{"open_time": NEXT_START, "open": 100.0, "close": 99.0}])
    sig = session.rows[0]
    assert sig.status == "SETTLED" and sig.win is False and sig.settle_outcome == "DOWN"


@pytest.mark.asyncio
async def test_settle_noise_expired(monkeypatch) -> None:
    session = _FakeSession(rows=[_pending(VAR_T4, NEXT_START)])
    monkeypatch.setattr(s2c, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = S2CondShadowDetector(collector=None, pm_15m_latest={})
    await d._settle_pending([{"open_time": NEXT_START, "open": 100.0, "close": 100.0}])
    sig = session.rows[0]
    assert sig.status == "EXPIRED" and sig.win is None and sig.settle_outcome == "NOISE"


@pytest.mark.asyncio
async def test_expire_stale_pending(monkeypatch) -> None:
    import time as _time
    old = int(_time.time() * 1000) - 10 * 3_600_000  # 目标根起点后 10h（>4h 超时窗）
    session = _FakeSession(rows=[_pending(VAR_T4, old)])
    monkeypatch.setattr(s2c, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = S2CondShadowDetector(collector=None, pm_15m_latest={})
    await d._expire_stale_pending()
    assert session.rows[0].status == "EXPIRED" and session.committed


# ============================================================
# 落库幂等 / 版本闸 / 结算隔离
# ============================================================

@pytest.mark.asyncio
async def test_record_writes_row(monkeypatch) -> None:
    session = _FakeSession(scalar=None)
    monkeypatch.setattr(s2c, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = S2CondShadowDetector(collector=None, pm_15m_latest={})
    await d._record(VAR_T5D, s2c.DISCOVERY_ID_T5D, s2c.RULE_TEXT_T5D,
                    NEXT_START, 100.0, 99.9, 0.44, 0.58, 555, 9)
    assert len(session.added) == 1
    r = session.added[0]
    assert r.version == VAR_T5D and r.discovery_id == "s2c_t5d" and r.direction == "UP"
    assert r.signal_bar_start == NEXT_START - BAR_MS_15M
    assert r.target_bar_start == NEXT_START and r.signal_bar_end == NEXT_START
    assert r.entry_up_price == 0.44 and r.entry_down_price == 0.58 and r.entry_quote_ts == 555
    assert r.timeframe == "15m" and r.status == "PENDING" and session.committed


@pytest.mark.asyncio
async def test_record_idempotent(monkeypatch) -> None:
    """已存在 (version, signal_bar_start) → 不重复落行。"""
    session = _FakeSession(scalar=123)  # 存在性查询命中
    monkeypatch.setattr(s2c, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = S2CondShadowDetector(collector=None, pm_15m_latest={})
    await d._record(VAR_T4, s2c.DISCOVERY_ID_T4, s2c.RULE_TEXT_T4,
                    NEXT_START, 100.0, 99.0, 0.42, 0.60, 111, 7)
    assert session.added == []


@pytest.mark.asyncio
async def test_record_blocked_by_version_gate(monkeypatch) -> None:
    """版本闸 is_enabled=False（前端下线）→ 停止采集新信号，零落库。"""
    session = _FakeSession(scalar=None)
    monkeypatch.setattr(s2c, "async_session_factory", lambda: _FakeSessionCtx(session))
    monkeypatch.setattr(s2c.shadow_gate, "is_enabled", lambda v: False)
    d = S2CondShadowDetector(collector=None, pm_15m_latest={})
    await d._record(VAR_T4, s2c.DISCOVERY_ID_T4, s2c.RULE_TEXT_T4,
                    NEXT_START, 100.0, 99.0, 0.42, 0.60, 111, 7)
    assert session.added == []


@pytest.mark.asyncio
async def test_settle_query_isolated_to_s2_versions(monkeypatch) -> None:
    """结算 WHERE 只过滤 S2_COND_VERSIONS——绝不触碰共表的 nb_/combo_/krev_/rev_/hm 行。"""
    from sqlalchemy.dialects import postgresql
    session = _CapSession(rows=[])
    monkeypatch.setattr(s2c, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = S2CondShadowDetector(collector=None, pm_15m_latest={})
    await d._settle_pending([{"open_time": NEXT_START, "open": 100.0, "close": 101.0}])
    sql = str(session.stmt.compile(dialect=postgresql.dialect(),
                                   compile_kwargs={"literal_binds": True}))
    assert VAR_T4 in sql and VAR_T5D in sql
    for other in ("nb_zschamp_15m_v1", "combo_p1_v1", "krev_a_v1", "rev_p1_v1",
                  "hm_touch_down_v1"):
        assert other not in sql, f"结算查询泄漏到其他家族版本 {other}"


# ============================================================
# 实盘 S2 钩子（去重 spawn / 停机不派生 / 异常只告警）
# ============================================================

@pytest.mark.asyncio
async def test_on_s2_signal_dedup_spawns_once(monkeypatch) -> None:
    spawned: list = []
    d = S2CondShadowDetector(collector=None, pm_15m_latest={})
    d._running = True

    async def _fake_confirm(parent_id, ns, ne) -> None:
        spawned.append((parent_id, ns, ne))

    monkeypatch.setattr(d, "_confirm_entry", _fake_confirm)
    payload = {"id": 3, "pattern_type": "bear_exhaust",
               "market_start_15m": NEXT_START, "market_end_15m": NEXT_END}
    d.on_s2_signal(payload)
    d.on_s2_signal(payload)          # 同 next_start 重复 → 去重
    await asyncio.sleep(0)           # 让 create_task 调度
    for t in list(d._tasks):
        await t
    assert spawned == [(3, NEXT_START, NEXT_END)]


@pytest.mark.asyncio
async def test_on_s2_signal_ignores_when_stopped() -> None:
    d = S2CondShadowDetector(collector=None, pm_15m_latest={})
    d._running = False
    d.on_s2_signal({"id": 1, "market_start_15m": NEXT_START, "market_end_15m": NEXT_END})
    assert len(d._tasks) == 0 and len(d._watched) == 0


def test_notify_s2_cond_calls_hook_with_payload() -> None:
    from binance_predict.services.fake_breakout_detector import FakeBreakoutDetector
    received: list = []
    fbd = FakeBreakoutDetector.__new__(FakeBreakoutDetector)  # 免构造（只测钩子转发）
    fbd._on_s2_cond = lambda payload: received.append(payload)
    fbd._notify_s2_cond(11, NEXT_START, NEXT_END)
    assert received == [{"id": 11, "pattern_type": "bear_exhaust",
                         "market_start_15m": NEXT_START, "market_end_15m": NEXT_END}]


def test_notify_s2_cond_swallows_hook_exception() -> None:
    from binance_predict.services.fake_breakout_detector import FakeBreakoutDetector
    fbd = FakeBreakoutDetector.__new__(FakeBreakoutDetector)

    def _boom(payload):
        raise RuntimeError("hook down")

    fbd._on_s2_cond = _boom
    fbd._notify_s2_cond(11, NEXT_START, NEXT_END)  # 不抛（绝不阻塞实盘检测循环）


def test_notify_s2_cond_noop_without_hook() -> None:
    from binance_predict.services.fake_breakout_detector import FakeBreakoutDetector
    fbd = FakeBreakoutDetector.__new__(FakeBreakoutDetector)
    fbd._on_s2_cond = None
    fbd._notify_s2_cond(11, NEXT_START, NEXT_END)  # 未装配 → 静默返回


# ============================================================
# 物理隔离 / 列宽 / 默认开关
# ============================================================

def test_versions_isolated_from_trading_path() -> None:
    from binance_predict.services.live_channels import LIVE_CHANNELS
    from binance_predict.services.multi_live_trader import X4_VERSIONS
    for v in S2_COND_VERSIONS:
        assert v not in X4_VERSIONS, f"{v} 不得进入 X4 下单白名单"
        assert v not in LIVE_CHANNELS, f"{v} 不得注册实盘通道"


def test_versions_disjoint_from_other_shadow_families() -> None:
    from binance_predict.services.hm_shadow_detector import ALL_VERSIONS as HM_VERSIONS
    from binance_predict.services.nextbar_shadow_detector import NEXTBAR_VERSIONS
    others = set(NEXTBAR_VERSIONS) | set(HM_VERSIONS)
    assert not (set(S2_COND_VERSIONS) & others), "S2 版本不得与其他共表家族重叠（结算隔离）"


def test_version_and_discovery_id_within_column_width() -> None:
    for v in S2_COND_VERSIONS:
        assert len(v) <= 24, f"{v} 超出 version 列宽 String(24)"
    assert len(s2c.DISCOVERY_ID_T4) <= 16 and len(s2c.DISCOVERY_ID_T5D) <= 16


def test_settings_default_on() -> None:
    """默认开启：与其他影子信号一致（record-only 零资金风险，部署即采集）。"""
    from binance_predict.config.settings import settings
    assert settings.s2_cond_shadow_enabled is True
