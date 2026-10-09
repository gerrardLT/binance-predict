"""冻结 q10+BTC recovery 前向 record-only 采集测试。"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
from sqlalchemy.dialects import postgresql

from binance_predict.config.settings import settings
from binance_predict.services.firsthit_forward_recorder import (
    DOWN_CANDIDATE,
    DOWN_CONTROL,
    Q_THRESHOLD,
    RECOVERY_THRESHOLD,
    SCHEMA_VERSION,
    UP_MIRROR,
    FirstHitForwardRecorder,
    _post_features,
    _upsert_settled_event,
    extract_forward_trigger,
)

START = 1_800_000_000_000 // 300_000 * 300_000


def test_forward_sample_overflow_and_capture_failure_isolated(monkeypatch):
    from binance_predict.services import shadow_forward_evidence as evidence
    recorder = FirstHitForwardRecorder(trader=object())
    recorder._running = True
    monkeypatch.setattr(evidence, "OVERFLOW_COUNTS", {})
    sample = dict(window_start=START, window_end=START+300000, ts_ms=START+120000,
                  btc_open=100, up_curve=[], down_curve=[], btc_curve=[])
    recorder._tasks = set(range(32))
    recorder.observe_sample(**sample)
    assert evidence.OVERFLOW_COUNTS["firsthit_recorder"] == 1
    recorder._tasks.clear()
    monkeypatch.setattr(recorder, "_observe_forward_sample",
                        lambda **kwargs: (_ for _ in ()).throw(ValueError("injected")))
    recorder.observe_sample(**sample)
    assert evidence.OVERFLOW_COUNTS["firsthit_recorder"] == 2
    assert not recorder._tasks


@pytest.mark.asyncio
async def test_g3_old_trigger_is_restored_not_live(monkeypatch):
    import asyncio
    from binance_predict.services import firsthit_shadow_detector as detector
    from binance_predict.services import firsthit_forward_recorder as module
    from unittest.mock import AsyncMock
    recorder = FirstHitForwardRecorder(trader=object())
    recorder._running = True
    monkeypatch.setattr(recorder, "_observe_absorption", lambda *args: None)
    monkeypatch.setattr(detector, "extract_firsthit_features", lambda *args, **kwargs: {"trigger_ts": START+110000, "q": .08})
    monkeypatch.setattr(detector, "_gate_of", lambda *args: True)
    monkeypatch.setattr(module, "extract_forward_trigger", lambda **kwargs: None)
    record = AsyncMock()
    monkeypatch.setattr(recorder, "_record_absorption", record)
    recorder.observe_sample(window_start=START, window_end=START+300000, ts_ms=START+120000,
        btc_open=100, up_curve=[], down_curve=[], btc_curve=[])
    await asyncio.gather(*recorder._tasks)
    event = record.await_args.args[0]
    assert event["trigger_ts"] == START+110000
    assert event["capture_mode"] == "RESTORED"


@pytest.mark.asyncio
async def test_absorption_failed_td_does_not_consume_retry(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from binance_predict.services import live_execution_policy as policy
    recorder = FirstHitForwardRecorder(trader=object())
    recorder.absorption_calibrator = SimpleNamespace(live_calibration=lambda version: (1, 0, 1, 1))
    record = AsyncMock()
    monkeypatch.setattr(recorder, "_record_absorption", record)
    selected = [False]
    monkeypatch.setattr(policy, "evaluate_absorption_policy", lambda **kwargs: SimpleNamespace(eligible=selected[0], prediction="DOWN"))
    curve = [{"t": START, "v": 100}]
    recorder._observe_absorption(START, START+300000, START+120000, 100, curve, curve, curve)
    assert ("absorption_follow_td120_v1", START) not in recorder._abs_seen
    selected[0] = True
    recorder._observe_absorption(START, START+300000, START+140000, 100, curve, curve, curve)
    await asyncio.gather(*recorder._tasks)
    assert ("absorption_follow_td120_v1", START) in recorder._abs_seen
    assert record.await_count == 1



@pytest.mark.asyncio
async def test_settlement_failure_does_not_advance_watermark(monkeypatch):
    from types import SimpleNamespace
    from binance_predict.services import firsthit_forward_recorder as module

    windows = [SimpleNamespace(start_time=START, end_time=START + 300_000),
               SimpleNamespace(start_time=START + 300_000, end_time=START + 600_000)]

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def execute(self, stmt):
            return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: windows))

    monkeypatch.setattr(module, "async_session_factory", Session)
    recorder = FirstHitForwardRecorder(trader=object())
    recorder._last_window_end = START
    attempts = []

    async def settle(window):
        attempts.append(window.start_time)
        if len(attempts) == 1:
            raise RuntimeError("temporary database failure")

    monkeypatch.setattr(recorder, "_settle_window", settle)
    await recorder._poll_once()
    assert recorder._last_window_end == START
    assert attempts == [START]
    await recorder._poll_once()
    assert attempts == [START, START, START + 300_000]
    assert recorder._last_window_end == START + 600_000


def _curve(values, *, step=15_000):
    return [{"t": START + i * step, "v": value} for i, value in enumerate(values)]


def test_down_firsthit_recovery_boundary_and_first_touch_are_frozen():
    # DOWN side_sign=-1。BTC 先涨 4bp（最不利 -4bp），触发时回落到涨 3bp，
    # recovery=(−3−−4)/4=25%，贴线应通过；首次 q=0.10 不得被后续 0.05 覆盖。
    down = _curve([0.60, 0.20, 0.10, 0.05])
    up = _curve([0.40, 0.80, 0.90, 0.95])
    btc = _curve([100_000.0, 100_040.0, 100_030.0, 100_020.0])
    event = extract_forward_trigger(
        side="DOWN", window_start=START, window_end=START + 300_000,
        btc_open=100_000.0, up_curve=up, down_curve=down, btc_curve=btc,
    )
    assert event is not None
    assert event["trigger_q"] == pytest.approx(Q_THRESHOLD)
    assert event["trigger_ts"] == START + 30_000
    assert event["btc_recovery_frac"] == pytest.approx(RECOVERY_THRESHOLD)
    assert event["recovery_pass"] is True
    assert event["labels"] == {DOWN_CONTROL: True, DOWN_CANDIDATE: True}


def test_down_nonrecovery_control_is_kept_not_filtered():
    down = _curve([0.40, 0.10])
    up = _curve([0.60, 0.90])
    btc = _curve([100_000.0, 100_050.0])  # 最不利点就是触发点，recovery=0
    event = extract_forward_trigger(
        side="DOWN", window_start=START, window_end=START + 300_000,
        btc_open=100_000.0, up_curve=up, down_curve=down, btc_curve=btc,
    )
    assert event is not None
    assert event["recovery_pass"] is False
    assert event["labels"] == {DOWN_CONTROL: True, DOWN_CANDIDATE: False}


def test_up_mirror_and_missing_btc_are_kept():
    event = extract_forward_trigger(
        side="UP", window_start=START, window_end=START + 300_000,
        btc_open=None, up_curve=_curve([0.30, 0.08]),
        down_curve=_curve([0.70, 0.92]), btc_curve=[],
    )
    assert event is not None
    assert event["side"] == "UP"
    assert event["labels"] == {UP_MIRROR: False}
    assert {"btc_open", "btc_trigger", "btc_recovery"} <= set(event["data_missing"])


def test_zero_quote_is_audited_without_division_by_zero():
    event = extract_forward_trigger(
        side="DOWN", window_start=START, window_end=START + 300_000,
        btc_open=100.0, up_curve=_curve([0.4, 1.0]),
        down_curve=_curve([0.6, 0.0]), btc_curve=_curve([100.0, 101.0]),
    )
    assert event is not None and event["trigger_q"] == 0.0
    assert "trigger_q_nonpositive" in event["data_missing"]


def test_max_trigger_ts_prevents_future_touch():
    kwargs = dict(
        side="DOWN", window_start=START, window_end=START + 300_000,
        btc_open=100.0, up_curve=_curve([0.4, 0.6, 0.92]),
        down_curve=_curve([0.6, 0.4, 0.08]), btc_curve=_curve([100, 101, 100.5]),
    )
    assert extract_forward_trigger(**kwargs, max_trigger_ts=START + 15_000) is None
    assert extract_forward_trigger(**kwargs, max_trigger_ts=START + 30_000) is not None


def test_post_features_use_first_sample_at_or_after_horizon():
    snapshot = extract_forward_trigger(
        side="DOWN", window_start=START, window_end=START + 300_000,
        btc_open=100.0,
        up_curve=_curve([0.4, 0.92, 0.88, 0.84, 0.80, 0.76]),
        down_curve=_curve([0.6, 0.08, 0.12, 0.16, 0.20, 0.24]),
        btc_curve=_curve([100.0, 101.0, 100.5, 100.2, 100.0, 99.8]),
    )
    assert snapshot is not None
    future, post, retrospective = _post_features(
        snapshot,
        _curve([0.4, 0.92, 0.88, 0.84, 0.80, 0.76]),
        _curve([0.6, 0.08, 0.12, 0.16, 0.20, 0.24]),
        _curve([100.0, 101.0, 100.5, 100.2, 100.0, 99.8]),
        99.8,
    )
    assert future["15"]["q"] == pytest.approx(0.12)
    assert future["60"]["q"] == pytest.approx(0.24)
    assert future["120"] is None
    assert post["rebound"] is True
    assert retrospective["source"] == "btc_prefix_le_trigger"


def test_retrospective_is_ex_ante_and_ignores_settlement_price():
    """2026-10-05 look-ahead 修复：body_ratio 只用 ≤trigger_ts 的 BTC 前缀与触发时刻价，
    结算价（旧实现误作 close）与触发后路径不得影响 retrospective。"""
    btc = _curve([100.0, 101.0, 100.5, 90.0, 80.0])   # 后三点=触发后（100.5 在 t=30s）
    snapshot = extract_forward_trigger(
        side="DOWN", window_start=START, window_end=START + 300_000,
        btc_open=100.0, up_curve=_curve([0.4, 0.9]), down_curve=_curve([0.6, 0.08]),
        btc_curve=btc,
    )
    assert snapshot is not None and snapshot["trigger_ts"] == START + 15_000
    _, _, retro_trigger = _post_features(snapshot, None, None, btc, 100.5)
    assert retro_trigger["source"] == "btc_prefix_le_trigger"
    # 前缀 = [100.0, 101.0]（触发后 100.5/90/80 全部排除）+ bo：
    # hi=101.0, lo=100.0, span=1.0, close=100.5 → body=0.5, major_wick=0.5。
    # 旧口径 hi/lo 用全窗曲线会把 80/90 拉进区间（body=|100.5-100|/21）。
    assert retro_trigger["body_ratio"] == pytest.approx(0.5)
    assert retro_trigger["major_wick_ratio"] == pytest.approx(0.5)
    _, _, retro_none = _post_features(snapshot, None, None, btc, None)
    assert retro_none["long_wick"] is None, "close 缺失 → 无特征（fail-safe）"


class _ExecuteSession:
    def __init__(self):
        self.statements = []

    async def execute(self, statement):
        self.statements.append(statement)


@pytest.mark.asyncio
async def test_settlement_upsert_uses_real_q_and_is_idempotent_by_side_window():
    snapshot = extract_forward_trigger(
        side="DOWN", window_start=START, window_end=START + 300_000,
        btc_open=100.0, up_curve=_curve([0.4, 0.93]),
        down_curve=_curve([0.6, 0.07]), btc_curve=_curve([100.0, 101.0]),
    )
    session = _ExecuteSession()
    await _upsert_settled_event(
        session, snapshot, outcome="DOWN", future={}, post={}, retrospective={}
    )
    compiled = session.statements[0].compile(dialect=postgresql.dialect())
    params = compiled.params
    assert params["settle_outcome"] == "DOWN"
    assert params["ev_at_entry"] == pytest.approx(0.98 / 0.07 - 1.0)
    assert "ON CONFLICT (side, window_start) DO UPDATE" in str(compiled)


class _QuoteOnlyTrader:
    def __init__(self):
        import asyncio
        self._trade_lock = asyncio.Lock()
        self._5m_start_date = START
        self._down_token_id = "down-token"
        self._up_token_id = "up-token"
        self._wallet_address = "0xwallet-ready"
        self.calls = []

    async def list_markets(self):
        return []

    async def get_quote(self, token_id, side, amount_usdt=None):
        self.calls.append((token_id, side, amount_usdt))
        return {
            "quoteId": f"q-{amount_usdt}",
            "averagePrice": 0.08 + amount_usdt / 1000,
            "amountOut": str(amount_usdt / 0.08),
        }

    async def place_order(self, *_args, **_kwargs):
        raise AssertionError("record-only 采集器不得调用 place_order")


class _EmptyWalletTrader(_QuoteOnlyTrader):
    """wallet 为空的独立 trader：fetch_wallet_info 成功后地址被填充。"""

    def __init__(self, wallet_ok=True):
        super().__init__()
        self._wallet_address = ""
        self.wallet_ok = wallet_ok
        self.fetch_calls = 0

    async def fetch_wallet_info(self):
        self.fetch_calls += 1
        if not self.wallet_ok:
            return None
        self._wallet_address = "0xwallet-fetched"
        return {"walletAddress": self._wallet_address}


@pytest.mark.asyncio
async def test_execution_ladder_is_quote_only_never_places_order():
    trader = _QuoteOnlyTrader()
    recorder = FirstHitForwardRecorder(trader=trader)
    result = await recorder._probe_execution_quotes("DOWN", START, 0.08)
    from binance_predict.services.live_channels import parse_channel_config
    current = parse_channel_config().get("firsthit_down_chg_v2")
    assert [call[2] for call in trader.calls] == sorted({1.0, 5.0, 10.0, current.amount_usdt if current else 1.0})
    assert all(result[key]["available"] for key in ("1", "5", "10"))
    assert all(result[key]["latency_ms"] >= 0 for key in ("1", "5", "10"))


@pytest.mark.asyncio
async def test_execution_ladder_auto_fetches_missing_wallet_address():
    """独立 trader 的 _wallet_address 为空（生产 -1102 根因）：触发时自动
    fetch_wallet_info 补齐后才走报价阶梯，且不再重复获取。"""
    trader = _EmptyWalletTrader()
    recorder = FirstHitForwardRecorder(trader=trader)
    result = await recorder._probe_execution_quotes("DOWN", START, 0.08)

    assert trader.fetch_calls == 1
    from binance_predict.services.live_channels import parse_channel_config
    current = parse_channel_config().get("firsthit_down_chg_v2")
    assert [call[2] for call in trader.calls] == sorted({1.0, 5.0, 10.0, current.amount_usdt if current else 1.0})
    assert all(result[key]["available"] for key in ("1", "5", "10"))
    # 补齐后再次触发：地址已在，不重取
    await recorder._probe_execution_quotes("DOWN", START, 0.08)
    assert trader.fetch_calls == 1


@pytest.mark.asyncio
async def test_execution_ladder_wallet_unavailable_short_circuits():
    """fetch_wallet_info 失败 → 短路返回不可用原因，不打 3 次注定失败的 get_quote。"""
    trader = _EmptyWalletTrader(wallet_ok=False)
    recorder = FirstHitForwardRecorder(trader=trader)
    result = await recorder._probe_execution_quotes("DOWN", START, 0.08)

    assert trader.fetch_calls == 1
    assert trader.calls == []  # 不触报价阶梯
    assert result == {"error": {"available": False,
                                "reason": "wallet_address_unavailable"}}


def test_record_only_versions_are_not_live_channels_and_default_on():
    from binance_predict.services.live_channels import LIVE_CHANNELS
    assert settings.firsthit_forward_enabled is True
    assert SCHEMA_VERSION == "firsthit_reversal_forward_v1"
    for version in (DOWN_CANDIDATE, DOWN_CONTROL, UP_MIRROR):
        assert version not in LIVE_CHANNELS


def test_recorder_has_no_order_placement_call():
    """静态硬闸：未来重构也不得把 record-only 采集器接到真钱调用。"""
    tree = ast.parse(Path("src/binance_predict/services/firsthit_forward_recorder.py")
                     .read_text(encoding="utf8"))
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "place_order" not in called
    assert "execute_trade" not in called
    assert "execute_signal_trade" not in called
