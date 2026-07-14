"""Agent 运行健康监控纯函数的单元测试。

覆盖 services/health.py 中不含 I/O 的聚合/派生逻辑：连续性、匹配率、校准分桶、
阶段心跳、告警派生、总体状态与自然语言诊断。DB 层 build_report 依赖真实会话，
不在此覆盖（由集成/手工验证）。
"""

from __future__ import annotations

from binance_predict.services.health import (
    _WINDOW_INTERVAL_MS,
    _WINDOW_INTERVAL_S,
    build_summary,
    compute_calibration,
    compute_phase_ages,
    compute_predict_stats,
    compute_window_continuity,
    derive_alerts,
    derive_overall_status,
)


# ============================================================
# compute_window_continuity
# ============================================================

def test_window_continuity_empty() -> None:
    result = compute_window_continuity([], now_ms=1_000_000)
    assert result["last_window_age_s"] is None
    assert result["gap_count"] == 0
    assert result["recent_count"] == 0


def test_window_continuity_no_gap() -> None:
    # 三个连续窗口，间隔恰为一个周期
    base = 1_000_000_000
    starts = [base, base + _WINDOW_INTERVAL_MS, base + 2 * _WINDOW_INTERVAL_MS]
    now_ms = base + 2 * _WINDOW_INTERVAL_MS + 60_000  # 最近窗口 60s 前
    result = compute_window_continuity(starts, now_ms)
    assert result["gap_count"] == 0
    assert result["recent_count"] == 3
    assert result["last_window_age_s"] == 60.0


def test_window_continuity_detects_gap() -> None:
    # 缺 2 个窗口：相邻间隔为 3 个周期 → 缺口应为 2
    base = 1_000_000_000
    starts = [base, base + 3 * _WINDOW_INTERVAL_MS]
    now_ms = base + 3 * _WINDOW_INTERVAL_MS
    result = compute_window_continuity(starts, now_ms)
    assert result["gap_count"] == 2


def test_window_continuity_unordered_input() -> None:
    base = 1_000_000_000
    starts = [base + 2 * _WINDOW_INTERVAL_MS, base, base + _WINDOW_INTERVAL_MS]
    now_ms = base + 2 * _WINDOW_INTERVAL_MS
    result = compute_window_continuity(starts, now_ms)
    assert result["gap_count"] == 0
    assert result["last_window_age_s"] == 0.0


# ============================================================
# compute_predict_stats
# ============================================================

def test_predict_stats_basic() -> None:
    directions = ["UP", "DOWN", "NO_TRADE", "UP"]
    matched = [True, False, False, True]
    result = compute_predict_stats(directions, matched, active_pattern_count=4)
    assert result["total"] == 4
    assert result["matched"] == 2
    assert result["match_rate"] == 0.5
    assert result["direction_distribution"] == {"UP": 2, "DOWN": 1, "NO_TRADE": 1}
    assert result["active_pattern_count"] == 4


def test_predict_stats_empty() -> None:
    result = compute_predict_stats([], [], active_pattern_count=0)
    assert result["total"] == 0
    assert result["match_rate"] == 0.0


# ============================================================
# compute_calibration
# ============================================================

def test_calibration_buckets_and_gap() -> None:
    # 0.7~0.8 桶：两条，命中 1 条 → hit_rate=0.5, avg_conf≈0.75, gap≈+0.25（过度自信）
    pairs = [(0.72, True), (0.78, False)]
    buckets = compute_calibration(pairs)
    assert len(buckets) == 1
    b = buckets[0]
    assert b.range == "0.70~0.80"
    assert b.count == 2
    assert b.hit_rate == 0.5
    assert b.gap is not None and b.gap > 0


def test_calibration_skips_empty_buckets() -> None:
    pairs = [(0.95, True), (0.92, True)]
    buckets = compute_calibration(pairs)
    # 仅 0.90~1.00 桶非空
    assert len(buckets) == 1
    assert buckets[0].range == "0.90~1.00"
    assert buckets[0].hit_rate == 1.0


def test_calibration_empty_input() -> None:
    assert compute_calibration([]) == []


# ============================================================
# compute_phase_ages
# ============================================================

def test_phase_ages_from_snapshot() -> None:
    now = 1000.0
    snapshot = {"phases": {"PREDICT": {"last_success_at": 940.0}}}
    ages = compute_phase_ages(snapshot, now_epoch=now)
    assert ages["PREDICT"] == 60.0
    # 未出现的阶段应为 None
    assert ages["EVOLVE"] is None


def test_phase_ages_none_snapshot() -> None:
    ages = compute_phase_ages(None, now_epoch=1000.0)
    assert all(v is None for v in ages.values())


# ============================================================
# derive_alerts / derive_overall_status
# ============================================================

def _clean_continuity() -> dict:
    return {"last_window_age_s": 60.0, "gap_count": 0, "recent_count": 10}


def test_derive_alerts_window_stale_critical() -> None:
    wc = {"last_window_age_s": 9999.0, "gap_count": 0, "recent_count": 5}
    alerts = derive_alerts(
        window_continuity=wc,
        predict_stats={"active_pattern_count": 0, "total": 0, "matched": 0},
        phase_ages={},
        queue_depth=None,
        llm=None,
        consecutive_failures=None,
        has_memory=False,
    )
    codes = {a.code for a in alerts}
    assert "WINDOW_STALE" in codes
    assert derive_overall_status(alerts) == "CRITICAL"


def test_derive_alerts_no_match_after_bootstrap() -> None:
    alerts = derive_alerts(
        window_continuity=_clean_continuity(),
        predict_stats={"active_pattern_count": 4, "total": 20, "matched": 0},
        phase_ages={},
        queue_depth=None,
        llm=None,
        consecutive_failures=None,
        has_memory=False,
    )
    codes = {a.code for a in alerts}
    assert "NO_MATCH" in codes
    assert derive_overall_status(alerts) == "WARN"


def test_derive_alerts_db_only_skips_memory_alerts() -> None:
    # has_memory=False 时即便传入 queue_depth/consecutive_failures 也不产内存态告警
    alerts = derive_alerts(
        window_continuity=_clean_continuity(),
        predict_stats={"active_pattern_count": 0, "total": 0, "matched": 0},
        phase_ages={"PREDICT": 99999.0},
        queue_depth=9999,
        llm={"phase_success_rates": {"PREDICT": {"success_rate": 0.1, "total": 50}}},
        consecutive_failures={"PREDICT": 99},
        has_memory=False,
    )
    assert alerts == []
    assert derive_overall_status(alerts) == "OK"


def test_derive_alerts_predict_stale_and_failures() -> None:
    stale = settings_predict_stale_limit() + 1
    alerts = derive_alerts(
        window_continuity=_clean_continuity(),
        predict_stats={"active_pattern_count": 0, "total": 0, "matched": 0},
        phase_ages={"PREDICT": stale},
        queue_depth=0,
        llm={"phase_success_rates": {}},
        consecutive_failures={"LEARN": 999},
        has_memory=True,
    )
    codes = {a.code for a in alerts}
    assert "PREDICT_STALE" in codes
    assert "LLM_FAILURES" in codes
    assert derive_overall_status(alerts) == "CRITICAL"


def settings_predict_stale_limit() -> float:
    from binance_predict.config.settings import settings

    return settings.agent_health_predict_stale_multiplier * _WINDOW_INTERVAL_S


# ============================================================
# build_summary
# ============================================================

def test_summary_cold_start() -> None:
    summary = build_summary(
        window_continuity=_clean_continuity(),
        predict_stats={"active_pattern_count": 0, "total": 0, "match_rate": 0.0},
        calibration=[],
        alerts=[],
        has_memory=True,
    )
    assert "冷启动" in summary
    assert "无告警" in summary


def test_summary_db_only_note() -> None:
    summary = build_summary(
        window_continuity=_clean_continuity(),
        predict_stats={"active_pattern_count": 4, "total": 20, "match_rate": 0.3},
        calibration=[],
        alerts=[],
        has_memory=False,
    )
    assert "仅 DB 层指标" in summary
