"""场景研究员与调度器单元测试（M2）：schema 校验 + 触发冷却 + 基线口径守护。

不触网络与真实 DB（冷却短路路径在 DB 查询之前）。
"""

from __future__ import annotations

import time

import pytest

from binance_predict.config.settings import settings
from binance_predict.services.research_scheduler import (
    BASELINE_FULL,
    SCENE_BASELINES,
    ResearchScheduler,
)
from binance_predict.services.scene_researcher import (
    ResearchAssessment,
    ResearchHypothesis,
    SceneResearcher,
)


def test_baseline_constants_guard_m1_calibration() -> None:
    """基线常量必须与 M1 引擎实测（真实样本外口径）一致——口径守护。"""
    assert SCENE_BASELINES["bull_exhaust"] == {"p": 0.620, "ci_lower": 0.540, "n": 150}
    assert SCENE_BASELINES["bear_exhaust"] == {"p": 0.564, "ci_lower": 0.484, "n": 149}
    assert BASELINE_FULL["bull_exhaust"]["n"] == 457
    assert BASELINE_FULL["bear_exhaust"]["n"] == 508


def test_research_assessment_schema() -> None:
    """结构化输出 schema：默认维持现状 + 假设条目字段完整 + 影响幅度边界。"""
    empty = ResearchAssessment(assessment="ok")
    assert empty.maintain_status_quo is True
    assert empty.hypotheses == []

    h = ResearchHypothesis(
        change_suggestion="close_pos 0.85→0.88",
        mechanism_reason="更严的光头要求过滤上影残余买力",
        expected_impact_pp=3.0,
        param_overrides={"close_pos_min": 0.88},
    )
    a = ResearchAssessment(assessment="...", maintain_status_quo=False, hypotheses=[h])
    assert a.hypotheses[0].param_overrides == {"close_pos_min": 0.88}

    with pytest.raises(ValueError):
        ResearchHypothesis(
            change_suggestion="负改善", mechanism_reason="x",
            expected_impact_pp=-1.0, param_overrides={},
        )


@pytest.mark.asyncio
async def test_scheduler_cooldown_short_circuits_before_db(monkeypatch) -> None:
    """冷却期内 _check_triggers 直接 None（在 DB 查询之前短路，不触库）。"""
    monkeypatch.setattr(settings, "scene_research_cooldown_hours", 24)
    scheduler = ResearchScheduler(researcher=SceneResearcher(decision_client=None))  # type: ignore[arg-type]
    scheduler._last_run = time.time() - 3600  # 1 小时前评估过 → 冷却期内
    # 若未短路，后续 DB 查询会因无数据库而抛异常（测试环境）——短路则返回 None
    assert await scheduler._check_triggers() is None
