"""X4 情绪错位影子检测器单元测试（M4 影子并行，2026-08-19）。

不触网络/真实 DB：async_session_factory 注入伪 session。
核心分支：
- 判定口径：outcome=UP & end_pct≤40 触发（乱序曲线取末点；DOWN/NOISE/end>40 不触发）
- 结算：PENDING 目标匹配次窗 → SETTLED（win/报价/EV 回填）；NOISE → EXPIRED
- 报价：≤150s 最晚点；real 优先 / chance proxy 回退 / 缺失 NULL
- EV：赢 0.98/(p+0.01)−1（截断 [0.01,0.99]）/ 输 −1
- 幂等：同窗口重复触发被唯一约束查询拦下
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from binance_predict.services import misalignment_detector as md
from binance_predict.services.misalignment_detector import (
    MisalignmentDetector,
    _curve_end_pct,
    _entry_quote,
    _ev_at_entry,
    _price_at,
)


# ============================================================
# 纯函数：判定/报价/EV 口径（与回测逐字段对齐）
# ============================================================

MIN = 60_000  # ms


def test_curve_end_pct_orders_by_t_and_takes_last() -> None:
    """乱序曲线按 t 排序取末点（回测 _line_feats['end'] 同口径）。"""
    curve = [{"t": 3 * MIN, "v": 35.0}, {"t": 0, "v": 55.0}, {"t": MIN, "v": 50.0}]
    assert _curve_end_pct(curve) == 35.0


def test_curve_end_pct_skips_none_values() -> None:
    curve = [{"t": 0, "v": 55.0}, {"t": MIN, "v": None}, {"t": 2 * MIN, "v": 38.0}]
    assert _curve_end_pct(curve) == 38.0


def test_curve_end_pct_empty() -> None:
    assert _curve_end_pct(None) is None
    assert _curve_end_pct([]) is None


def test_price_at_takes_latest_within_150s() -> None:
    """≤150s 内最晚采样点（回测 price_at 同口径），超时刻点排除。"""
    start = 1_000_000
    curve = [
        {"t": start + 100_000, "v": 0.50},   # +100s
        {"t": start + 150_000, "v": 0.52},   # +150s（边界，含）
        {"t": start + 151_000, "v": 0.90},   # +151s（排除）
        {"t": start + 200_000, "v": 0.95},   # +200s（排除）
    ]
    v, t = _price_at(curve, start)
    assert v == 0.52 and t == start + 150_000


def test_price_at_before_window_returns_none() -> None:
    """全部采样点在决策点之后 → None。"""
    start = 1_000_000
    curve = [{"t": start + 160_000, "v": 0.60}, {"t": start + 200_000, "v": 0.40}]
    assert _price_at(curve, start) == (None, None)


def test_entry_quote_real_priority() -> None:
    start = 1_000_000
    down = [{"t": start + 100_000, "v": 0.48}]
    up = [{"t": start + 100_000, "v": 0.50}]
    pct = [{"t": start + 100_000, "v": 47.0}]
    d, u, ts, kind = _entry_quote(down, up, pct, start)
    assert d == 0.48 and u == 0.50 and kind == "real"


def test_entry_quote_proxy_fallback() -> None:
    """真实 DOWN token 价缺失 → chance/100 代理（回测 entry_quote 同回退序）。"""
    start = 1_000_000
    pct = [{"t": start + 100_000, "v": 46.0}]
    d, _, _, kind = _entry_quote(None, None, pct, start)
    assert d == pytest.approx(0.46) and kind == "proxy"


def test_entry_quote_missing() -> None:
    d, _, ts, kind = _entry_quote(None, None, None, 1_000_000)
    assert d is None and ts is None and kind is None


def test_ev_at_entry_win_lose_and_clamp() -> None:
    # 赢：0.98/(0.50+0.01)−1
    assert _ev_at_entry(True, 0.50) == pytest.approx(0.98 / 0.51 - 1.0)
    # 输：−1
    assert _ev_at_entry(False, 0.50) == -1.0
    # entry 截断上限 0.99：0.98/0.99−1
    assert _ev_at_entry(True, 0.99) == pytest.approx(0.98 / 0.99 - 1.0)
    # 无胜负/无报价 → NULL
    assert _ev_at_entry(None, 0.50) is None
    assert _ev_at_entry(True, None) is None


# ============================================================
# 判定与结算编排（伪 session）
# ============================================================

class _FakeSessionCtx:
    def __init__(self, session: object) -> None:
        self._session = session

    async def __aenter__(self) -> object:
        return self._session

    async def __aexit__(self, *exc: object) -> bool:
        return False


def _win(start: int, outcome: str, curve=None, ret=None, prices=None, pcts=None) -> SimpleNamespace:
    """伪 SentimentWindow。"""
    return SimpleNamespace(
        start_time=start,
        end_time=start + 5 * MIN,
        outcome=outcome,
        actual_return=ret if ret is not None else (0.001 if outcome == "UP" else -0.001),
        curve_up_pct=curve if curve is not None else [{"t": start, "v": 55.0}, {"t": start + 4 * MIN, "v": 38.0}],
        curve_down_price=prices,
        curve_up_price=None,
        curve_down_pct=pcts,
    )


def _sig(window_start: int, status: str = "PENDING") -> SimpleNamespace:
    return SimpleNamespace(
        version="x4_v1", window_start=window_start, window_end=window_start + 5 * MIN,
        end_pct=38.0, outcome_base="UP", direction="DOWN",
        target_window_start=window_start + 5 * MIN,
        entry_down_price=None, entry_up_price=None, entry_quote_ts=None,
        entry_quote_kind=None, settle_outcome=None, win=None, ev_at_entry=None,
        status=status,
    )


async def _run_process(det_win, pending_sigs=(), dup_rows=None) -> tuple[MagicMock, list]:
    """跑 _process_window：session.execute 按调用序返回（结算查询→幂等查询）。"""
    det = MisalignmentDetector()
    added: list = []
    session = MagicMock()
    settle_result = MagicMock()
    settle_result.scalars.return_value.all.return_value = list(pending_sigs)
    dup_result = MagicMock()
    dup_result.first.return_value = (dup_rows or [None])[0]
    session.execute = AsyncMock(side_effect=[settle_result, dup_result])
    session.add = MagicMock(side_effect=added.append)
    session.commit = AsyncMock()
    factory = MagicMock(return_value=_FakeSessionCtx(session))
    with patch.object(md, "async_session_factory", factory):
        await det._process_window(det_win)
    return session, added


@pytest.mark.asyncio
async def test_trigger_on_up_window_with_low_end_pct() -> None:
    """收阳 & end≤40 → 落 PENDING 信号。"""
    session, added = await _run_process(_win(100 * MIN, "UP"))
    assert len(added) == 1
    sig = added[0]
    assert sig.version == "x4_v1" and sig.direction == "DOWN"
    assert sig.end_pct == 38.0 and sig.status == "PENDING"
    assert sig.target_window_start == 105 * MIN  # 次窗 = 本窗 end


@pytest.mark.asyncio
async def test_no_trigger_on_down_or_high_end() -> None:
    session, added = await _run_process(_win(100 * MIN, "DOWN"))
    assert added == []
    session, added = await _run_process(_win(100 * MIN, "UP", curve=[{"t": 0, "v": 55.0}, {"t": 1, "v": 45.0}]))
    assert added == []  # end=45 > 40
    session, added = await _run_process(_win(100 * MIN, "NOISE"))
    assert added == []


@pytest.mark.asyncio
async def test_no_duplicate_on_same_window() -> None:
    """幂等：唯一约束预查询命中 → 不再 add。"""
    session, added = await _run_process(_win(100 * MIN, "UP"), dup_rows=[123])
    assert added == []


@pytest.mark.asyncio
async def test_settle_pending_win_with_real_quote() -> None:
    """次窗结算：PENDING 目标匹配 → SETTLED，报价/胜负/EV 回填。"""
    start = 100 * MIN
    sig = _sig(start)
    next_win = _win(
        start + 5 * MIN, "DOWN",
        prices=[{"t": start + 5 * MIN + 150_000, "v": 0.50}],  # 决策点 DOWN 价 0.50
    )
    det = MisalignmentDetector()
    session = MagicMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = [sig]
    session.execute = AsyncMock(side_effect=[result])
    session.commit = AsyncMock()
    factory = MagicMock(return_value=_FakeSessionCtx(session))
    with patch.object(md, "async_session_factory", factory):
        await det._process_window(next_win)
    assert sig.status == "SETTLED" and sig.win is True
    assert sig.settle_outcome == "DOWN"
    assert sig.entry_down_price == 0.50 and sig.entry_quote_kind == "real"
    assert sig.ev_at_entry == pytest.approx(0.98 / 0.51 - 1.0)


@pytest.mark.asyncio
async def test_settle_pending_lose_with_proxy_quote() -> None:
    start = 100 * MIN
    sig = _sig(start)
    next_win = _win(
        start + 5 * MIN, "UP", ret=0.002,
        curve=[{"t": 0, "v": 50.0}, {"t": 1, "v": 55.0}],  # 末点 55：不触发新信号
        pcts=[{"t": start + 5 * MIN + 100_000, "v": 52.0}],  # 无真实价 → proxy 0.52
    )
    det = MisalignmentDetector()
    session = MagicMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = [sig]
    session.execute = AsyncMock(side_effect=[result])
    session.commit = AsyncMock()
    factory = MagicMock(return_value=_FakeSessionCtx(session))
    with patch.object(md, "async_session_factory", factory):
        await det._process_window(next_win)
    assert sig.status == "SETTLED" and sig.win is False
    assert sig.entry_down_price == pytest.approx(0.52) and sig.entry_quote_kind == "proxy"
    assert sig.ev_at_entry == -1.0


@pytest.mark.asyncio
async def test_settle_pending_noise_expires() -> None:
    """次窗 NOISE（actual_return=0）→ 无法判向 → EXPIRED，win 保持 NULL。"""
    start = 100 * MIN
    sig = _sig(start)
    next_win = _win(start + 5 * MIN, "NOISE", ret=0.0)
    det = MisalignmentDetector()
    session = MagicMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = [sig]
    session.execute = AsyncMock(side_effect=[result])
    session.commit = AsyncMock()
    factory = MagicMock(return_value=_FakeSessionCtx(session))
    with patch.object(md, "async_session_factory", factory):
        await det._process_window(next_win)
    assert sig.status == "EXPIRED" and sig.win is None and sig.ev_at_entry is None


# ============================================================
# API 端点回归：GET /api/misalignment/signals
# ============================================================

@pytest.mark.asyncio
async def test_api_signals_endpoint_builds_sql_and_stats() -> None:
    """回归：统计聚合语句必须可构建（曾因 func.case 误用致生产 500）。

    直接调用端点函数：agg 语句在 db.execute 前构建，构建即验证，
    mock 只拦截执行结果。附验 stats 四件套计算。
    """
    from binance_predict.db.models import MisalignmentSignal
    import binance_predict.main as m

    def _row(start: int, *, status: str, win, ev, kind) -> MisalignmentSignal:
        return MisalignmentSignal(
            version="x4_v1", window_start=start, window_end=start + 5 * MIN,
            end_pct=37.0, outcome_base="UP", direction="DOWN",
            target_window_start=start + 5 * MIN,
            entry_down_price=0.53 if kind else None,
            entry_up_price=None, entry_quote_ts=None, entry_quote_kind=kind,
            settle_outcome=("DOWN" if status == "SETTLED" else None),
            win=win, ev_at_entry=ev, status=status,
        )

    rows = [
        _row(100 * MIN, status="SETTLED", win=True, ev=0.6, kind="real"),
        _row(101 * MIN, status="SETTLED", win=False, ev=-1.0, kind="proxy"),
        _row(102 * MIN, status="PENDING", win=None, ev=None, kind=None),
    ]
    db = MagicMock()
    r1 = MagicMock()
    r1.scalars.return_value.all.return_value = rows
    r2 = MagicMock()
    # (count, win_valid, wins, n_ev, avg_ev, n_real)：SETTLED 2 注，1 赢，实价 1
    r2.one.return_value = (2, 2, 1, 2, -0.2, 1)
    db.execute = AsyncMock(side_effect=[r1, r2])

    out = await m.list_misalignment_signals(limit=50, db=db)

    assert out["total"] == 3 and out["signals"][0]["end_pct"] == 37.0
    assert out["stats"] == {
        "settled": 2, "win_rate": 0.5, "avg_ev": -0.2, "real_quote_coverage": 0.5,
    }


@pytest.mark.asyncio
async def test_api_signals_endpoint_empty_table() -> None:
    """空表：sum 全 None，stats 不炸（除零/None 保护）。"""
    import binance_predict.main as m

    db = MagicMock()
    r1 = MagicMock()
    r1.scalars.return_value.all.return_value = []
    r2 = MagicMock()
    r2.one.return_value = (0, None, None, None, None, None)
    db.execute = AsyncMock(side_effect=[r1, r2])

    out = await m.list_misalignment_signals(limit=50, db=db)

    assert out["signals"] == [] and out["total"] == 0
    assert out["stats"] == {
        "settled": 0, "win_rate": None, "avg_ev": None, "real_quote_coverage": None,
    }
