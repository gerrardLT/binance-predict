"""科学发现系统 Phase 3 集成测试（宪法第八条：Predict 谓词化）。

覆盖：
- services/agent_logic.py  resolve_predicate_hits 命中解析四分支（顺序不可换）
  与 pattern_confidence 程序置信合成（live win_rate 优先 / 先验回退）
- services/llm_validator.py  validate_arbitrate_output 仲裁输出校验
  （HARD：选定 id 限冲突候选集合；SOFT：reasoning / confidence / 放弃时机）
- services/sentiment_agent.py  _current_window_dict（PREDICT 事件三通道拆分）
- services/sentiment_agent.py  predict() 谓词化流水线：
  冷启动 / 零命中不调 LLM / 单命中与多命中同向程序直采 /
  多命中异向 LLM 仲裁（选定 / 放弃 / 失败 / 越界）/
  仪器精度对齐（缺 binning_version、快照查不到的组跳过）

数值案例：sentiment 快照 edges=(0.1, 0.2, 0.3, 0.4)，up_pct 差值 1.0 ≥ q80
→ 「急升」（命中）；差值 0.25 ∈ [q40, q60) → 「平」（不命中）。
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from binance_predict.models.schemas import ArbitrateOutput
from binance_predict.services.agent_logic import (
    HIT_CONCORDANT,
    HIT_CONFLICT,
    HIT_NONE,
    HIT_SINGLE,
    PredicateHit,
    pattern_confidence,
    resolve_predicate_hits,
)
from binance_predict.services.llm_validator import validate_arbitrate_output
from binance_predict.services.sentiment_agent import SentimentAgent

# ============================================================
# 共用构造
# ============================================================

# 命中谓词：sentiment 通道至少含 1 个「急升」
HIT_PREDICATE = {
    "pred": "count_symbol",
    "channel": "sentiment",
    "symbol": "急升",
    "cmp": ">=",
    "value": 1,
}

# sentiment 通道快照分位边界（q20, q40, q60, q80）
SNAP_EDGES = [0.1, 0.2, 0.3, 0.4]

# 窗口结束时间：未来 5 分钟，保证仲裁路径 remaining 远超 stale 阈值（30s）
WINDOW_END = int(time.time() * 1000) + 300_000


def _mk_curve(last_up_pct: float) -> list[dict]:
    """两点实时曲线（PREDICT 事件形态）。delta = last - 50.0：
    - 1.0 ≥ q80(0.4) → 「急升」（命中 HIT_PREDICATE）
    - 0.25 ∈ [q40, q60) → 「平」（不命中）
    """
    return [
        {"t": WINDOW_END - 300_000, "up_pct": 50.0, "down_pct": 50.0,
         "btc_price": 100_000.0, "trade_volume": 5.0},
        {"t": WINDOW_END - 240_000, "up_pct": last_up_pct, "down_pct": 100.0 - last_up_pct,
         "btc_price": 100_010.0, "trade_volume": 6.0},
    ]


HIT_CURVE = _mk_curve(51.0)   # delta=1.0 → 急升
MISS_CURVE = _mk_curve(50.25)  # delta=0.25 → 平


def _mk_pattern(
    pid: int,
    direction: str,
    *,
    predicate: dict | None = HIT_PREDICATE,
    version: str | None = "v1",
    status: str = "ACTIVE",
    win_rate: float = 0.8,
    samples: int = 10,
    prior: float = 0.7,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=pid,
        pattern_name=f"P{pid}",
        description=f"模式 {pid}",
        predicted_direction=direction,
        predicate=predicate,
        binning_version=version,
        status=status,
        win_rate=win_rate,
        sample_count=samples,
        confidence_score=prior,
    )


def _snapshot_rows(version: str = "v1") -> list[SimpleNamespace]:
    """binning_snapshots 行（仅 sentiment 通道；price/volume 无快照则通道跳过）。"""
    return [
        SimpleNamespace(
            version=version,
            channel="sentiment",
            edges=SNAP_EDGES,
            sample_count=100,
            created_at_epoch=1_000.0,
        )
    ]


class _FakeCtx:
    """伪 async 上下文（session 通用）。"""

    def __init__(self, session: object = None) -> None:
        self._session = session

    async def __aenter__(self) -> object:
        return self._session

    async def __aexit__(self, *exc: object) -> bool:
        return False


def _predict_session(
    patterns: list, snapshot_rows: list | None = None
) -> MagicMock:
    """伪 session：predict() 三次 execute 按序消费——
    #1 select(PatternMemory) / #2 select(SentimentWindow.id) / #3 select(BinningSnapshotModel)。
    提前返回的路径（如冷启动）少消费不报错。
    """
    r1 = MagicMock()
    r1.scalars.return_value.all.return_value = patterns
    r2 = MagicMock()
    r2.scalar_one_or_none.return_value = None  # 窗口未归档
    r3 = MagicMock()
    r3.scalars.return_value.all.return_value = snapshot_rows or []
    session = MagicMock()
    session.execute = AsyncMock(side_effect=[r1, r2, r3])
    return session


def _mk_agent() -> SentimentAgent:
    """绕过 __init__：LLM 通道与落库辅助替换为 mock（聚焦决策逻辑断言）。"""
    agent = SentimentAgent.__new__(SentimentAgent)
    agent._llm = MagicMock()
    agent._llm.agent_arbitrate = AsyncMock()
    agent._write_prediction_and_trade = AsyncMock(return_value=None)
    return agent


async def _run_predict(agent: SentimentAgent, session: MagicMock, curve: list) -> dict:
    """执行 predict() 并返回 _write_prediction_and_trade 的调用 kwargs。"""
    factory = MagicMock(return_value=_FakeCtx(session))
    with patch(
        "binance_predict.services.sentiment_agent.async_session_factory", factory
    ):
        await agent.predict(window_end_ms=WINDOW_END, current_curve=curve)
    agent._write_prediction_and_trade.assert_awaited_once()
    return agent._write_prediction_and_trade.await_args.kwargs


# ============================================================
# agent_logic.resolve_predicate_hits（命中解析四分支）
# ============================================================


class TestResolvePredicateHits:
    def test_none_when_empty(self):
        branch, chosen = resolve_predicate_hits([])
        assert branch == HIT_NONE
        assert chosen is None

    def test_single(self):
        hit = PredicateHit(pattern_id=1, pattern_name="A", direction="UP")
        branch, chosen = resolve_predicate_hits([hit])
        assert branch == HIT_SINGLE
        assert chosen is hit

    def test_concordant_picks_highest_win_rate(self):
        weak = PredicateHit(1, "A", "UP", win_rate=0.6, sample_count=30)
        strong = PredicateHit(2, "B", "UP", win_rate=0.8, sample_count=10)
        branch, chosen = resolve_predicate_hits([weak, strong])
        assert branch == HIT_CONCORDANT
        assert chosen is strong

    def test_concordant_tiebreak_samples_then_id(self):
        # win_rate 并列 → sample_count 降序；再并列 → id 升序（确定性）
        fewer = PredicateHit(5, "A", "DOWN", win_rate=0.7, sample_count=10)
        more = PredicateHit(9, "B", "DOWN", win_rate=0.7, sample_count=20)
        branch, chosen = resolve_predicate_hits([fewer, more])
        assert branch == HIT_CONCORDANT
        assert chosen is more
        a = PredicateHit(3, "C", "DOWN", win_rate=0.7, sample_count=20)
        b = PredicateHit(7, "D", "DOWN", win_rate=0.7, sample_count=20)
        assert resolve_predicate_hits([b, a])[1] is a

    def test_conflict_on_mixed_directions(self):
        up = PredicateHit(1, "A", "UP", win_rate=0.9, sample_count=50)
        down = PredicateHit(2, "B", "DOWN", win_rate=0.1, sample_count=1)
        branch, chosen = resolve_predicate_hits([up, down])
        assert branch == HIT_CONFLICT
        assert chosen is None


class TestPatternConfidence:
    def test_live_win_rate_when_samples_sufficient(self):
        # sample_count >= min_samples → live win_rate（最诚实的证据）
        assert pattern_confidence(0.8123, 10, 0.7, min_samples=5) == 0.8123

    def test_prior_fallback_when_insufficient(self):
        # 样本不足 → 回退 LLM 先验（交易门控规则 4 二次保护）
        assert pattern_confidence(0.9, 2, 0.654321, min_samples=5) == 0.6543


# ============================================================
# llm_validator.validate_arbitrate_output（仲裁输出校验）
# ============================================================


class TestValidateArbitrateOutput:
    def test_valid_selection_no_hard(self):
        out = ArbitrateOutput(
            reasoning="两模式冲突，DOWN 证据更强",
            selected_pattern_id=2,
            confidence=0.66,
            entry_timing="NOW",
        )
        hard, soft = validate_arbitrate_output(out, {1, 2})
        assert hard == []

    def test_hard_when_selected_not_in_candidates(self):
        out = ArbitrateOutput(
            reasoning="选了候选外的模式",
            selected_pattern_id=99,
            confidence=0.8,
            entry_timing="NOW",
        )
        hard, _ = validate_arbitrate_output(out, {1, 2})
        assert len(hard) == 1
        assert "99" in hard[0]

    def test_abandon_no_hard(self):
        out = ArbitrateOutput(
            reasoning="冲突不可调和",
            selected_pattern_id=None,
            confidence=0.0,
            entry_timing="SKIP",
        )
        hard, _ = validate_arbitrate_output(out, {1, 2})
        assert hard == []

    def test_soft_empty_reasoning(self):
        out = ArbitrateOutput(
            reasoning="  ",
            selected_pattern_id=1,
            confidence=0.8,
            entry_timing="NOW",
        )
        _, soft = validate_arbitrate_output(out, {1})
        assert any("reasoning" in s for s in soft)

    def test_soft_abandon_but_non_skip_timing(self):
        out = ArbitrateOutput(
            reasoning="放弃但仍想入场（矛盾输出）",
            selected_pattern_id=None,
            confidence=0.5,
            entry_timing="NOW",
        )
        _, soft = validate_arbitrate_output(out, {1})
        assert any("SKIP" in s for s in soft)


# ============================================================
# sentiment_agent._current_window_dict（PREDICT 事件三通道拆分）
# ============================================================


class TestCurrentWindowDict:
    def test_three_channels_split(self):
        d = SentimentAgent._current_window_dict(WINDOW_END, HIT_CURVE)
        assert d["start_time"] == WINDOW_END - 300_000  # 首个采样时刻
        assert d["curve_up_pct"] == [
            {"t": WINDOW_END - 300_000, "v": 50.0},
            {"t": WINDOW_END - 240_000, "v": 51.0},
        ]
        assert d["curve_btc_price"][1]["v"] == 100_010.0
        assert d["curve_trade_volume"][0]["v"] == 5.0

    def test_missing_points_keep_none_for_defensive_skip(self):
        curve = [
            {"t": 1, "up_pct": 50.0, "down_pct": 50.0,
             "btc_price": None, "trade_volume": None},
            {"t": 2, "up_pct": 51.0, "down_pct": 49.0,
             "btc_price": 100.0, "trade_volume": 6.0},
        ]
        d = SentimentAgent._current_window_dict(WINDOW_END, curve)
        # None 原样保留，由 symbolizer._series_values 跳过（不伪造数据）
        assert d["curve_btc_price"][0]["v"] is None

    def test_empty_curve_fallback_start_time(self):
        d = SentimentAgent._current_window_dict(WINDOW_END, [])
        assert d["start_time"] == WINDOW_END - 300_000
        assert d["curve_up_pct"] == []


# ============================================================
# sentiment_agent.predict() 谓词化流水线（第八条操作化规则 1~7）
# ============================================================


@pytest.mark.asyncio
class TestPredictPredicatePipeline:
    async def test_cold_start_no_predicate_candidates(self):
        """规则 1：库内仅旧模式（predicate=None）→ NO_TRADE 冷启动，不调 LLM。"""
        agent = _mk_agent()
        legacy = _mk_pattern(1, "UP", predicate=None)
        session = _predict_session([legacy])
        kw = await _run_predict(agent, session, HIT_CURVE)
        assert kw["predicted_direction"] == "NO_TRADE"
        assert kw["matched_pattern_id"] is None
        assert "谓词模式库为空" in kw["reasoning"]
        agent._llm.agent_arbitrate.assert_not_awaited()

    async def test_zero_hit_no_llm(self):
        """规则 4：谓词零命中 → NO_TRADE，不调 LLM。"""
        agent = _mk_agent()
        session = _predict_session([_mk_pattern(1, "UP")], _snapshot_rows())
        kw = await _run_predict(agent, session, MISS_CURVE)
        assert kw["predicted_direction"] == "NO_TRADE"
        assert "谓词零命中" in kw["reasoning"]
        assert kw["entry_timing"] == "SKIP"
        agent._llm.agent_arbitrate.assert_not_awaited()

    async def test_single_hit_adopted_without_llm(self):
        """规则 4/5/7：单命中 → 程序直采，confidence=live win_rate（样本足），
        entry_timing 恒 NOW，不调 LLM。"""
        agent = _mk_agent()
        p = _mk_pattern(1, "UP", win_rate=0.8, samples=10, prior=0.7)
        session = _predict_session([p], _snapshot_rows())
        kw = await _run_predict(agent, session, HIT_CURVE)
        assert kw["predicted_direction"] == "UP"
        assert kw["matched_pattern_id"] == 1
        assert kw["matched_pattern_name"] == "P1"
        assert kw["confidence"] == 0.8  # live win_rate（10 >= min 5）
        assert kw["entry_timing"] == "NOW"
        assert "单命中" in kw["reasoning"]
        agent._llm.agent_arbitrate.assert_not_awaited()

    async def test_single_hit_low_samples_falls_back_to_prior(self):
        """规则 5：live 样本 < min_pattern_samples → confidence 回退 LLM 先验。"""
        agent = _mk_agent()
        p = _mk_pattern(2, "DOWN", status="EVOLVING", win_rate=1.0, samples=2, prior=0.65)
        session = _predict_session([p], _snapshot_rows())
        kw = await _run_predict(agent, session, HIT_CURVE)
        assert kw["predicted_direction"] == "DOWN"
        assert kw["confidence"] == 0.65  # 先验回退而非 1.0
        assert kw["entry_timing"] == "NOW"

    async def test_non_predicate_patterns_excluded(self):
        """规则 1：predicate 为空的旧模式不参与匹配（同库谓词模式照常）。"""
        agent = _mk_agent()
        legacy = _mk_pattern(1, "DOWN", predicate=None)
        modern = _mk_pattern(2, "UP")
        session = _predict_session([legacy, modern], _snapshot_rows())
        kw = await _run_predict(agent, session, HIT_CURVE)
        # 旧模式排除后仅剩 modern 单命中
        assert kw["matched_pattern_id"] == 2
        assert kw["predicted_direction"] == "UP"
        agent._llm.agent_arbitrate.assert_not_awaited()

    async def test_concordant_picks_strongest_without_llm(self):
        """规则 4：多命中同向 → 取证据最强者（win_rate 降序），不调 LLM。"""
        agent = _mk_agent()
        weak = _mk_pattern(1, "UP", win_rate=0.6, samples=30)
        strong = _mk_pattern(2, "UP", win_rate=0.85, samples=8)
        session = _predict_session([weak, strong], _snapshot_rows())
        kw = await _run_predict(agent, session, HIT_CURVE)
        assert kw["matched_pattern_id"] == 2
        assert kw["confidence"] == 0.85
        assert "多命中同向" in kw["reasoning"]
        agent._llm.agent_arbitrate.assert_not_awaited()

    async def test_conflict_arbitrate_selected(self):
        """规则 6：多命中异向 → LLM 仲裁选定；direction 由程序从模式推导。"""
        agent = _mk_agent()
        up = _mk_pattern(1, "UP", win_rate=0.9, samples=50)
        down = _mk_pattern(2, "DOWN", win_rate=0.7, samples=20)
        agent._llm.agent_arbitrate.return_value = ArbitrateOutput(
            reasoning="DOWN 模式谓词更贴合当前量价结构",
            selected_pattern_id=2,
            confidence=0.66,
            entry_timing="WAIT",
        )
        session = _predict_session([up, down], _snapshot_rows())
        kw = await _run_predict(agent, session, HIT_CURVE)
        # 仲裁通道被调用且仅一次；候选含两个冲突模式
        agent._llm.agent_arbitrate.assert_awaited_once()
        call_kw = agent._llm.agent_arbitrate.await_args.kwargs
        assert {c["id"] for c in call_kw["candidates"]} == {1, 2}
        assert "window_payload" in call_kw
        # direction 程序推导（非 LLM 输出字段）；confidence/timing 取仲裁输出
        assert kw["predicted_direction"] == "DOWN"
        assert kw["matched_pattern_id"] == 2
        assert kw["confidence"] == 0.66
        assert kw["entry_timing"] == "WAIT"
        assert "仲裁选定" in kw["reasoning"]

    async def test_conflict_arbitrate_abandon(self):
        """规则 6：仲裁放弃（selected=None）→ NO_TRADE。"""
        agent = _mk_agent()
        agent._llm.agent_arbitrate.return_value = ArbitrateOutput(
            reasoning="两模式方向相反且证据均弱",
            selected_pattern_id=None,
            confidence=0.0,
            entry_timing="SKIP",
        )
        session = _predict_session(
            [_mk_pattern(1, "UP"), _mk_pattern(2, "DOWN")], _snapshot_rows()
        )
        kw = await _run_predict(agent, session, HIT_CURVE)
        assert kw["predicted_direction"] == "NO_TRADE"
        assert kw["matched_pattern_id"] is None
        assert "仲裁放弃" in kw["reasoning"]
        assert kw["skip_trade_reason"] == "仲裁放弃交易"

    async def test_conflict_arbitrate_llm_failure(self):
        """规则 6：LLM 失败/超时 → NO_TRADE + 明确原因（无静默降级）。"""
        agent = _mk_agent()
        agent._llm.agent_arbitrate.side_effect = TimeoutError("boom")
        session = _predict_session(
            [_mk_pattern(1, "UP"), _mk_pattern(2, "DOWN")], _snapshot_rows()
        )
        kw = await _run_predict(agent, session, HIT_CURVE)
        assert kw["predicted_direction"] == "NO_TRADE"
        assert "LLM 仲裁调用失败" in kw["reasoning"]
        assert kw["skip_trade_reason"] == "LLM 调用失败"

    async def test_conflict_arbitrate_out_of_candidates(self):
        """规则 6：选定 id 越界（非冲突候选）→ 程序 HARD 拦截降级 NO_TRADE。"""
        agent = _mk_agent()
        agent._llm.agent_arbitrate.return_value = ArbitrateOutput(
            reasoning="幻觉输出",
            selected_pattern_id=999,
            confidence=0.9,
            entry_timing="NOW",
        )
        session = _predict_session(
            [_mk_pattern(1, "UP"), _mk_pattern(2, "DOWN")], _snapshot_rows()
        )
        kw = await _run_predict(agent, session, HIT_CURVE)
        assert kw["predicted_direction"] == "NO_TRADE"
        assert "越界" in kw["reasoning"]
        assert kw["skip_trade_reason"] == "LLM 输出语义验证失败"

    async def test_missing_binning_version_skipped(self):
        """规则 2：模式缺 binning_version → 跳过记 warning（不静默），
        无其他候选时零命中 NO_TRADE。"""
        agent = _mk_agent()
        p = _mk_pattern(1, "UP", version=None)
        session = _predict_session([p])  # 快照查询不会被触发（版本集为空）
        kw = await _run_predict(agent, session, HIT_CURVE)
        assert kw["predicted_direction"] == "NO_TRADE"
        assert "谓词零命中" in kw["reasoning"]

    async def test_snapshot_version_not_found_skipped(self):
        """规则 2：binning_snapshots 查不到模式的出生版本 → 跳过记 warning。"""
        agent = _mk_agent()
        p = _mk_pattern(1, "UP", version="v-ghost")
        session = _predict_session([p], _snapshot_rows("v1"))  # 表中仅 v1
        kw = await _run_predict(agent, session, HIT_CURVE)
        assert kw["predicted_direction"] == "NO_TRADE"
        assert "谓词零命中" in kw["reasoning"]

    async def test_multi_version_groups_symbolized_independently(self):
        """规则 2 仪器精度对齐：不同出生版本的模式各用其版本边界符号化——
        v2 边界（q80=2.0）下 delta=1.0 落在 [q40, q60) 档为「平」，该组模式不命中。"""
        agent = _mk_agent()
        p1 = _mk_pattern(1, "UP", version="v1")   # q80=0.4：命中
        p2 = _mk_pattern(2, "UP", version="v2")   # q80=2.0：不命中
        rows_v2 = [
            SimpleNamespace(
                version="v2", channel="sentiment",
                edges=[0.5, 1.0, 1.5, 2.0], sample_count=100,
                created_at_epoch=2_000.0,
            )
        ]
        session = _predict_session([p1, p2], _snapshot_rows("v1") + rows_v2)
        kw = await _run_predict(agent, session, HIT_CURVE)
        # 仅 v1 组命中 → 单命中程序直采
        assert kw["matched_pattern_id"] == 1
        assert "单命中" in kw["reasoning"]
