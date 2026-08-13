"""科学发现系统 Phase 4 集成测试（宪法 Q7：双轨死因 + 反馈循环）。

覆盖：
- services/verification.py  live_lift_summary（recent lift + 前缀峰值扫描 + 防御）
- live_lift_summary × diagnose_death 联合真值表（SPURIOUS / EXPIRED / ALIVE）
- services/sentiment_agent.py  _diagnose_pattern_deaths 编排：
  判死 RETIRE + death_cause/lifespan_days 落库、样本不足跳过、ALIVE 不动
- services/sentiment_agent.py  _load_discovery_feedback 反馈包组装
- services/llm_service.py  _build_discovery_user_msg 反馈区块渲染（Q7-2）

数值案例（对照池基准率 0.5，手算验证）：
- SPURIOUS：20 次命中 10 真 → recent lift=1.0，CI=[0.54,1.86] 覆盖 1，
  前缀峰值 1.2<1.3（从未显著）→ 假规律
- EXPIRED：前 5 次全真（峰值 lift=2.0≥1.3），最近 20 次 4 真 → lift≈0.42<1.1
  → 曾显著后衰减
- ALIVE：20 次命中 16 真 → lift=1.6（近期仍强）→ 存活
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from binance_predict.services.llm_service import LLMService
from binance_predict.services.sentiment_agent import SentimentAgent
from binance_predict.services.verification import (
    DEATH_ALIVE,
    DEATH_EXPIRED,
    DEATH_SPURIOUS,
    diagnose_death,
    live_lift_summary,
)

DAY = 86_400_000
W0 = 1_700_000_000_000
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)

# ============================================================
# 内核：live_lift_summary 纯函数
# ============================================================


class TestLiveLiftSummary:
    def test_recent_window_truncation_and_peak(self):
        """25 次命中：recent 只取最近 20 次；峰值在最早全真前缀 n=5。

        hits = [T]*5 + [T,F,F,F,F]*4，基准 10/20=0.5：
        - recent（后 20）= 4 真 → lift = 0.2/0.5 = 0.4
        - peak：n=5 前缀全真 → 1.0/0.5 = 2.0；之后被稀释 → peak=2.0
        """
        hits = [True] * 5 + [True, False, False, False, False] * 4
        s = live_lift_summary(hits, base_events=10, base_total=20)
        assert s.hit_count == 25
        assert s.recent.hit_total == 20
        assert s.recent.hit_events == 4
        assert s.recent.lift == pytest.approx(0.4)
        assert s.peak_lift == pytest.approx(2.0)

    def test_short_history_below_min_prefix_gives_zero_peak(self):
        """命中数 < min_prefix(5)：峰值不参与（防小样本假峰），recent 仍计算。

        3 次全真，基准 0.5 → recent lift = 1.0/0.5 = 2.0，peak=0.0。
        """
        s = live_lift_summary([True, True, True], base_events=10, base_total=20)
        assert s.hit_count == 3
        assert s.recent.lift == pytest.approx(2.0)
        assert s.peak_lift == 0.0

    def test_empty_baseline_is_conservative(self):
        """基准池为空 / 零事件：recent 与 peak 均退化为 0（绝不假显著）。"""
        s = live_lift_summary([True] * 25, base_events=0, base_total=0)
        assert s.recent.lift == 0.0
        assert s.recent.p_value == 1.0
        assert s.peak_lift == 0.0

    def test_zero_baseline_events_is_conservative(self):
        """基准池零事件（全 DOWN 对照）：分母事件为 0 → 保守退化。"""
        s = live_lift_summary([True] * 10, base_events=0, base_total=30)
        assert s.recent.lift == 0.0
        assert s.peak_lift == 0.0


# ============================================================
# 内核联合真值表：live_lift_summary → diagnose_death
# ============================================================


def _verdict(hits: list[bool], base_events: int = 10, base_total: int = 20) -> str:
    s = live_lift_summary(hits, base_events, base_total)
    return diagnose_death(
        s.recent.lift,
        s.recent.ci_lower,
        s.recent.ci_upper,
        s.peak_lift,
        s.hit_count,
    )


class TestDeathDiagnosisTruthTable:
    def test_spurious_never_significant(self):
        """假规律：20 次命中 10 真（lift=1.0，CI 覆盖 1），峰值 1.2<1.3。"""
        hits = [True, False] * 10
        assert _verdict(hits) == DEATH_SPURIOUS

    def test_expired_decayed_after_significance(self):
        """过期规律：前 5 次全真（峰值 2.0≥1.3），最近 20 次 4 真（lift=0.4<1.1）。"""
        hits = [True] * 5 + [True, False, False, False, False] * 4
        assert _verdict(hits) == DEATH_EXPIRED

    def test_alive_when_recent_still_strong(self):
        """存活：20 次命中 16 真 → recent lift=1.6（不衰减）；峰值≥1.3 排除 SPURIOUS。"""
        hits = [True] * 16 + [False] * 4
        assert _verdict(hits) == DEATH_ALIVE

    def test_alive_when_insufficient_hits(self):
        """存活：19 次命中 < MIN_DEATH_HITS(20) → 样本不足不判死（全真也不判）。"""
        assert _verdict([True] * 19) == DEATH_ALIVE


# ============================================================
# 编排：_diagnose_pattern_deaths（mock session）
# ============================================================


class _FakeCtx:
    """伪 async 上下文（begin_nested / session 通用）。"""

    def __init__(self, session: object = None) -> None:
        self._session = session

    async def __aenter__(self) -> object:
        return self._session

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _Result:
    """按调用形态返回预设数据的伪查询结果。"""

    def __init__(
        self,
        *,
        scalars: list | None = None,
        rows: list | None = None,
        one: object = None,
    ) -> None:
        self._scalars = scalars
        self._rows = rows
        self._one = one

    def scalars(self) -> SimpleNamespace:
        return SimpleNamespace(all=lambda: self._scalars or [])

    def all(self) -> list:
        return self._rows or []

    def scalar_one_or_none(self) -> object:
        return self._one


def _mk_pattern(pid: int = 1, status: str = "ACTIVE") -> SimpleNamespace:
    """PatternMemory 形态（含 _pattern_to_snapshot 所需全部字段）。"""
    return SimpleNamespace(
        id=pid,
        pattern_name=f"P{pid}",
        description=f"模式 {pid}",
        curve_features={},
        conditions={},
        predicted_direction="UP",
        win_rate=0.5,
        sample_count=20,
        correct_count=10,
        confidence_score=0.7,
        status=status,
        predicate={"pred": "count_symbol", "channel": "sentiment",
                   "symbol": "急升", "cmp": ">=", "value": 1},
        binning_version="v1",
        death_cause=None,
        lifespan_days=None,
        created_at=T0,
    )


def _mk_hit_rows(pid: int, flags: list[bool]) -> list[tuple]:
    """已结算命中行：(pattern_id, is_correct, prediction_time, window_start)。

    命中窗口间隔 1 天：W0 + (i+1)*DAY。
    """
    return [
        (pid, ok, T0 + timedelta(hours=i), W0 + (i + 1) * DAY)
        for i, ok in enumerate(flags)
    ]


def _mk_window_rows(n_hits: int, up_count: int) -> list[tuple]:
    """窗口行 (start_time, outcome)：命中窗口（会被对照池排除）+ 对照窗口。

    对照窗口位于命中 +12h（必落在某命中的 ±3 天内），前 up_count 个 outcome=UP。
    """
    rows = [(W0 + (i + 1) * DAY, "UP") for i in range(n_hits)]  # 命中窗口自身
    rows += [
        (W0 + (i + 1) * DAY + DAY // 2, "UP" if i < up_count else "DOWN")
        for i in range(n_hits)
    ]
    return rows


def _diagnose_session(
    pattern: SimpleNamespace, hit_rows: list[tuple], window_rows: list[tuple]
) -> MagicMock:
    """按 _diagnose_pattern_deaths 的 4 次 execute 顺序排好应答队列。"""
    session = MagicMock()
    session.execute = AsyncMock(side_effect=[
        _Result(scalars=[pattern]),          # Step 1: 候选模式
        _Result(rows=hit_rows),              # Step 2: 已结算命中
        _Result(rows=window_rows),           # Step 3: 局部基准原料
        _Result(one=pattern),                # apply_pattern_change 内部回查
    ])
    session.begin_nested.side_effect = lambda: _FakeCtx(session)
    return session


def _agent() -> SentimentAgent:
    return SentimentAgent.__new__(SentimentAgent)


@pytest.mark.asyncio
async def test_diagnose_retires_spurious_with_cause_and_lifespan():
    """SPURIOUS：RETIRE + death_cause/lifespan_days 落库 + 进负样本统计。

    20 次命中 [T,F]*10（10 真），对照 10/20=0.5 → recent lift=1.0 CI 覆盖 1，
    峰值 1.2<1.3 → 假规律。created_at 距今 2026-08-11-T0 ≈ 222 天级，仅断言 >0。
    """
    pattern = _mk_pattern(pid=1)
    flags = [True, False] * 10
    session = _diagnose_session(
        pattern, _mk_hit_rows(1, flags), _mk_window_rows(20, up_count=10)
    )

    stats = await _agent()._diagnose_pattern_deaths(session, "evo-test")

    assert stats["checked"] == 1
    assert stats["spurious"] == [{"id": 1, "name": "P1"}]
    assert stats["expired"] == []
    assert pattern.status == "RETIRED"
    assert pattern.death_cause == DEATH_SPURIOUS
    assert pattern.lifespan_days is not None and pattern.lifespan_days > 0
    # RETIRE 变更日志落库
    assert session.add.called


@pytest.mark.asyncio
async def test_diagnose_retires_expired():
    """EXPIRED：曾显著（峰值≈2.08≥1.3）后衰减（recent≈0.42<1.1）→ 过期归档。

    25 次命中 [T]*5+[T,F,F,F,F]*4，对照 12/25=0.48。
    """
    pattern = _mk_pattern(pid=2)
    flags = [True] * 5 + [True, False, False, False, False] * 4
    session = _diagnose_session(
        pattern, _mk_hit_rows(2, flags), _mk_window_rows(25, up_count=12)
    )

    stats = await _agent()._diagnose_pattern_deaths(session, "evo-test")

    assert stats["checked"] == 1
    assert stats["expired"] == [{"id": 2, "name": "P2"}]
    assert stats["spurious"] == []
    assert pattern.status == "RETIRED"
    assert pattern.death_cause == DEATH_EXPIRED
    assert pattern.lifespan_days is not None


@pytest.mark.asyncio
async def test_diagnose_alive_pattern_untouched():
    """ALIVE：20 次命中 16 真（recent lift=1.6 仍强）→ 不 RETIRE、不写死因。"""
    pattern = _mk_pattern(pid=3)
    flags = [True] * 16 + [False] * 4
    session = _diagnose_session(
        pattern, _mk_hit_rows(3, flags), _mk_window_rows(20, up_count=10)
    )

    stats = await _agent()._diagnose_pattern_deaths(session, "evo-test")

    assert stats["checked"] == 1
    assert stats["spurious"] == [] and stats["expired"] == []
    assert pattern.status == "ACTIVE"
    assert pattern.death_cause is None
    assert not session.add.called


@pytest.mark.asyncio
async def test_diagnose_skips_insufficient_hits():
    """样本不足：19 次命中 < 20 → skipped，不诊断不 RETIRE（全真也不判死）。"""
    pattern = _mk_pattern(pid=4)
    session = _diagnose_session(
        pattern, _mk_hit_rows(4, [True] * 19), _mk_window_rows(19, up_count=9)
    )

    stats = await _agent()._diagnose_pattern_deaths(session, "evo-test")

    assert stats["checked"] == 0
    assert stats["skipped"] == 1
    assert pattern.status == "ACTIVE"
    assert not session.add.called


# ============================================================
# 编排：_load_discovery_feedback（patch async_session_factory）
# ============================================================


def _feedback_session(
    negatives: list, pos_rows: list[tuple], lifespans: list[float]
) -> MagicMock:
    session = MagicMock()
    session.execute = AsyncMock(side_effect=[
        _Result(scalars=negatives),   # 负样本
        _Result(rows=pos_rows),       # 正样本统计
        _Result(scalars=lifespans),   # 存活期
    ])
    return session


@pytest.mark.asyncio
async def test_load_discovery_feedback_full():
    """反馈包全量：负样本留谓词结构、正样本只留统计、存活期 mean/median/max。

    正样本 3 个（UP×2 胜率 0.8/0.6，DOWN×1 胜率 0.7）→ avg=0.7；
    存活期 [10,20,30,40] → mean=25.0，median=(20+30)/2=25.0，max=40.0。
    """
    neg = SimpleNamespace(
        pattern_name="假规律X",
        predicate={"pred": "sync", "channel_a": "sentiment",
                   "channel_b": "price", "cmp": ">=", "value": 0.8},
        predicted_direction="UP",
        description="情绪价格同步假说",
        win_rate=0.45,
        sample_count=20,
    )
    session = _feedback_session(
        [neg], [("UP", 0.8), ("UP", 0.6), ("DOWN", 0.7)], [10.0, 20.0, 30.0, 40.0]
    )
    factory = MagicMock(return_value=_FakeCtx(session))

    with patch(
        "binance_predict.services.sentiment_agent.async_session_factory", factory
    ):
        fb = await _agent()._load_discovery_feedback()

    assert fb["negatives"][0]["pattern_name"] == "假规律X"
    assert fb["negatives"][0]["predicate"]["pred"] == "sync"  # 负样本全量细节
    assert fb["positive_summary"] == {
        "count": 3,
        "avg_win_rate": 0.7,
        "up_count": 2,
        "down_count": 1,
    }
    assert fb["lifespan_stats"] == {
        "count": 4,
        "mean": 25.0,
        "median": 25.0,
        "max": 40.0,
    }


@pytest.mark.asyncio
async def test_load_discovery_feedback_empty_library():
    """冷启动：无死亡/无存活 → 空负样本、零统计、None 存活期字段。"""
    session = _feedback_session([], [], [])
    factory = MagicMock(return_value=_FakeCtx(session))

    with patch(
        "binance_predict.services.sentiment_agent.async_session_factory", factory
    ):
        fb = await _agent()._load_discovery_feedback()

    assert fb["negatives"] == []
    assert fb["positive_summary"]["count"] == 0
    assert fb["lifespan_stats"] == {"count": 0, "mean": None, "median": None, "max": None}


# ============================================================
# prompt：_build_discovery_user_msg 反馈区块渲染
# ============================================================


def _svc() -> LLMService:
    return LLMService.__new__(LLMService)


def _one_window() -> list[dict]:
    return [{
        "start_time": W0,
        "outcome": "UP",
        "channels": {
            "sentiment": {"symbols": ["平", "急升"], "geometry": {}},
        },
    }]


class TestDiscoveryFeedbackMsg:
    def test_cold_start_without_feedback(self):
        """feedback=None（首轮发现）→ 冷启动文案 + 窗口区块仍在。"""
        msg = _svc()._build_discovery_user_msg(_one_window(), None)
        assert "暂无历史审判记录（首轮发现）" in msg
        assert "共 1 个" in msg  # 窗口区块不因反馈缺失而丢

    def test_full_feedback_blocks(self):
        """全量反馈：负样本谓词全量 + 正样本统计 + 存活期，且在窗口数据之前。"""
        feedback = {
            "negatives": [{
                "pattern_name": "假规律X",
                "predicate": {"pred": "sync", "channel_a": "sentiment",
                              "channel_b": "price", "cmp": ">=", "value": 0.8},
                "predicted_direction": "UP",
                "description": "情绪价格同步假说",
                "win_rate": 0.45,
                "sample_count": 20,
            }],
            "positive_summary": {"count": 3, "avg_win_rate": 0.7,
                                 "up_count": 2, "down_count": 1},
            "lifespan_stats": {"count": 4, "mean": 25.0, "median": 25.0, "max": 40.0},
        }
        msg = _svc()._build_discovery_user_msg(_one_window(), feedback)

        assert "已被证伪的假设（1 条" in msg
        assert "禁止重提" in msg
        assert '"pred":"sync"' in msg  # 负样本谓词结构全量呈现
        assert "胜率 0.45" in msg
        assert "存活 ACTIVE 模式 3 个" in msg
        assert "结构对你不可见" in msg  # 正样本不给谓词（防近亲繁殖）
        assert "平均 25.0 天" in msg
        # 反馈区块先于窗口数据呈现
        assert msg.index("已被证伪的假设") < msg.index("发现集窗口")

    def test_empty_negative_and_lifespan_blocks(self):
        """负样本空 → 「尚无假规律被处决」；存活期 count=0 → 不出现存活期区块。"""
        feedback = {
            "negatives": [],
            "positive_summary": {"count": 0, "avg_win_rate": 0.0,
                                 "up_count": 0, "down_count": 0},
            "lifespan_stats": {"count": 0, "mean": None, "median": None, "max": None},
        }
        msg = _svc()._build_discovery_user_msg(_one_window(), feedback)

        assert "尚无假规律被处决" in msg
        assert "模式库为空" in msg
        assert "规律存活期分布" not in msg
