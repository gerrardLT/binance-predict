"""研究账本纯函数单测（Phase 1）。

口径断言来自规范 §4.1（四互斥层完备）与 §4.2（intention EV）。
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from binance_predict.research.firsthit_ledger import (
    AUDIT_VERSION,
    EXPECTED_SAMPLES_PER_WINDOW,
    FEATURE_VERSION,
    build_window_audit,
    build_mother_event,
    intention_ev,
    rows_from_window,
    stratum_of,
)

L = 300_000


def _win(*, down=None, btc=None, up=None, participants=None,
         entry_price=100_000.0, outcome="DOWN", actual_return=-0.001,
         sample_count=None, start=1_788_000_000_000):
    """构造 SentimentWindow 鸭子类型替身（字段名与 ORM 一致）。"""
    n = max(len(down or []), len(btc or []))
    ts = [start + i * 15_000 for i in range(n)]
    return SimpleNamespace(
        start_time=start, end_time=start + L, entry_price=entry_price,
        curve_down_price=[{"t": t, "v": v} for t, v in zip(ts, down or [])] if down is not None else None,
        curve_btc_price=[{"t": t, "v": v} for t, v in zip(ts, btc or [])] if btc is not None else None,
        curve_up_price=[{"t": t, "v": v} for t, v in zip(ts, up or [])] if up is not None else None,
        curve_trade_volume=[{"t": t, "v": float(i)} for i, t in enumerate(ts)],
        curve_participants=participants,
        outcome=outcome, actual_return=actual_return, sample_count=sample_count,
    )


# ============================================================
# Task 1: stratum_of / intention_ev
# ============================================================

def test_stratum_of_four_mutually_exclusive_layers():
    assert stratum_of(g1=True, g3=True) == "both"
    assert stratum_of(g1=True, g3=False) == "g1_only"
    assert stratum_of(g1=False, g3=True) == "g3_only"
    assert stratum_of(g1=False, g3=False) == "neither"


def test_intention_ev_win_lose_and_unknown():
    # 赢：0.98/q − 1（费 2% 无溢价，逐事件真实触发价）
    assert intention_ev(True, 0.05) == 0.98 / 0.05 - 1.0
    assert intention_ev(False, 0.07) == -1.0
    # 结算未知 / 报价缺失 → None（技术性缺失不填 0）
    assert intention_ev(None, 0.07) is None
    assert intention_ev(True, None) is None


# ============================================================
# Task 2: build_window_audit
# ============================================================

def test_window_audit_no_firsthit():
    w = _win(down=[0.3] * 10, btc=[100_000.0] * 10)
    row = build_window_audit(w)
    assert row["raw_firsthit_detected"] is False
    assert row["firsthit_detected"] is False
    assert row["exclusion_reason"] == "no_firsthit"
    assert row["audit_version"] == AUDIT_VERSION
    assert row["expected_samples"] == EXPECTED_SAMPLES_PER_WINDOW
    assert row["actual_samples"] == 10
    assert row["down_pts"] == 10 and row["btc_pts"] == 10


def test_window_audit_sparse_path_excluded():
    # 有首触但触发前 btc 点数 < 8 → 整窗被排除（npts 门）
    w = _win(down=[0.3] * 8 + [0.07, 0.07], btc=[100_000.0] * 3)
    row = build_window_audit(w)
    assert row["raw_firsthit_detected"] is True
    assert row["firsthit_detected"] is False
    assert row["exclusion_reason"] == "npts_lt_8"


def test_window_audit_valid_and_settled():
    down = [0.3] * 8 + [0.07, 0.07]
    btc = [100_000.0, 100_020, 100_050, 100_040, 100_030, 100_020, 100_010,
           100_005, 100_005, 100_005]
    w = _win(down=down, btc=btc, outcome="DOWN", actual_return=-0.002)
    row = build_window_audit(w)
    assert row["raw_firsthit_detected"] is True
    assert row["firsthit_detected"] is True
    assert row["exclusion_reason"] is None
    assert row["settled"] is True
    assert row["outcome"] == "DOWN"
    assert row["max_gap_ms"] == 15_000


def test_window_audit_unsettled_keeps_row():
    down = [0.3] * 8 + [0.07, 0.07]
    btc = [100_000.0] * 9 + [100_005]
    w = _win(down=down, btc=btc, outcome=None, actual_return=None)
    row = build_window_audit(w)
    assert row["settled"] is False
    assert row["outcome"] is None
    assert row["firsthit_detected"] is True  # 触发与结算独立记录（Q2 缺失审计）


# ============================================================
# Task 3: build_mother_event
# ============================================================

def _valid_g1_g3_window():
    """G1∩G3∩G7 母事件：小实体 + 上影 + 微涨，q=0.07，触发前 9 个 btc 点。"""
    down = [0.3] * 8 + [0.07, 0.07]
    btc = [100_000.0, 100_020, 100_050, 100_040, 100_030, 100_020, 100_010,
           100_005, 100_005, 100_005]
    up = [0.7] * 8 + [0.93, 0.93]
    return _win(down=down, btc=btc, up=up, outcome="DOWN", actual_return=-0.002)


def test_mother_event_labels_and_stratum():
    ev = build_mother_event(_valid_g1_g3_window())
    assert ev is not None
    assert ev["feature_version"] == FEATURE_VERSION
    assert ev["q"] == 0.07 and ev["trigger_ts"] == 1_788_000_000_000 + 8 * 15_000
    assert ev["q_prev"] == 0.3 and ev["dt_prev_ms"] == 15_000
    assert ev["up_price_at_trigger"] == 0.93 and ev["sum_gap"] == 0.0
    # 标签：与 _gate_of 同源（g1/g3/g4/g7 命中；streak 族未知 → None）
    assert ev["labels"]["g0"] is True
    assert ev["labels"]["firsthit_down_body_v1"] is True
    assert ev["labels"]["firsthit_down_chg_v1"] is True
    assert ev["labels"]["firsthit_down_g4_v1"] is True
    assert ev["labels"]["firsthit_down_g7_v1"] is True
    assert ev["labels"]["g7_streak_v1"] is None
    assert ev["stratum"] == "both"
    # 结算与 intention EV
    assert ev["win"] is True
    assert ev["intention_ev"] == 0.98 / 0.07 - 1.0


def test_mother_event_missing_discipline():
    # participants 曲线缺失 → dpar=None + dpar_missing=True，绝不填 0
    ev = build_mother_event(_valid_g1_g3_window())
    assert ev["dvol"] is not None and ev["dvol_missing"] is False
    assert ev["dpar"] is None and ev["dpar_missing"] is True


def test_mother_event_neither_stratum():
    # 大实体路径（body_r>0.35）+ 涨幅>2.82bp → neither 层
    down = [0.3] * 8 + [0.09, 0.09]
    btc = [100_000.0] + [100_050] * 9  # 单边上行：body=1.0，chg=+5bp
    w = _win(down=down, btc=btc, outcome="UP", actual_return=0.005)
    ev = build_mother_event(w)
    assert ev["stratum"] == "neither"
    assert ev["labels"]["firsthit_down_body_v1"] is False
    assert ev["win"] is False and ev["intention_ev"] == -1.0


def test_mother_event_returns_none_when_excluded():
    w = _win(down=[0.3] * 10, btc=[100_000.0] * 10)  # 无首触
    assert build_mother_event(w) is None


# ============================================================
# Task 5: rows_from_window
# ============================================================

def test_ledger_models_match_window_and_mother_contracts():
    from binance_predict.db.models import Base, FirstHitMotherEvent, FirstHitWindowAudit

    assert {"firsthit_window_audit", "firsthit_mother_event"} <= set(Base.metadata.tables)
    audit_cols = {col.name for col in FirstHitWindowAudit.__table__.columns}
    assert {"window_start", "raw_firsthit_detected", "firsthit_detected", "settled"} <= audit_cols
    mother_cols = {col.name for col in FirstHitMotherEvent.__table__.columns}
    assert {"window_start", "labels", "stratum", "intention_ev", "dpar_missing"} <= mother_cols


def test_rows_from_window_pair():
    w = _valid_g1_g3_window()
    audit, mother = rows_from_window(w)
    assert audit["window_start"] == mother["window_start"]
    assert audit["firsthit_detected"] is True
    assert mother["stratum"] == "both"
    assert mother["labels"]["firsthit_down_g7_v1"] is True
    # 无首触窗：audit 有行、mother 为 None（全窗口留痕，规范 §12.1）
    w2 = _win(down=[0.3] * 10, btc=[100_000.0] * 10)
    audit2, mother2 = rows_from_window(w2)
    assert audit2["firsthit_detected"] is False and mother2 is None


@pytest.mark.asyncio
async def test_ledger_upsert_awaits_audit_and_mother_writes():
    spec = importlib.util.spec_from_file_location(
        "build_firsthit_ledger", Path("scripts/build_firsthit_ledger.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    session = SimpleNamespace(execute=AsyncMock())

    assert await module.upsert_window(session, _valid_g1_g3_window()) == (True, True)
    assert session.execute.await_count == 2
