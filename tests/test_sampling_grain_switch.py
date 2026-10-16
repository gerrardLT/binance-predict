"""采样粒度 15s→5s 全量切换（2026-10-16）口径守卫。

防止后续单独调整采样间隔或 PREDICT 触发阈值导致触发时点漂移：
- 30 点 × 5s ≈ 150s（半窗），与旧口径 10 点 × 15s 的触发时点一致；
- _pm_history 覆盖时长不因粒度切换缩水（≥8 小时）；
- 三个图表端点响应的 poll_interval_sec 与实际采样间隔一致。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from binance_predict.config import settings as settings_mod

MAIN_SRC = Path(__file__).resolve().parents[1] / "src" / "binance_predict" / "main.py"


@pytest.fixture(scope="module")
def main_module():
    from binance_predict import main
    return main


def test_poll_interval_is_5s(main_module):
    assert main_module.PM_POLL_INTERVAL_S == 5


def test_predict_trigger_keeps_half_window_timing(main_module):
    """触发时点不变量：N × 间隔 ≈ 150s（半窗），改动任一侧须同步另一侧。"""
    n = settings_mod.settings.agent_predict_trigger_samples
    assert n * main_module.PM_POLL_INTERVAL_S == 150


def test_pm_history_covers_at_least_8h(main_module):
    assert main_module._pm_history.maxlen is not None
    assert main_module._pm_history.maxlen * main_module.PM_POLL_INTERVAL_S >= 8 * 3600


def test_chart_endpoints_report_actual_interval():
    src = MAIN_SRC.read_text(encoding="utf-8")
    assert len(re.findall(r'"poll_interval_sec": 5', src)) == 3
    assert '"poll_interval_sec": 15' not in src


def test_chart_endpoints_keep_about_100min_window():
    """默认 limit=1200 / 内存切片 1200：5s 粒度下约 100 分钟，与切换前 400×15s 覆盖一致。"""
    src = MAIN_SRC.read_text(encoding="utf-8")
    assert src.count("limit: int = 1200,") == 2
    assert "list(_pm_history)[-1200:]" in src
