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
    assert retrospective["source"] == "sampled_btc_path"


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


@pytest.mark.asyncio
async def test_execution_ladder_is_quote_only_never_places_order():
    trader = _QuoteOnlyTrader()
    recorder = FirstHitForwardRecorder(trader=trader)
    result = await recorder._probe_execution_quotes("DOWN", START, 0.08)
    assert [call[2] for call in trader.calls] == [1.0, 5.0, 10.0]
    assert all(result[key]["available"] for key in ("1", "5", "10"))
    assert all(result[key]["latency_ms"] >= 0 for key in ("1", "5", "10"))


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
