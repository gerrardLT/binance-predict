"""科学发现系统 Phase 2 集成测试（宪法第〇条角色分离 + Q4/Q5/Q6）。

覆盖：
- services/discovery.py  screen_hypotheses 初筛流水线
  （DSL 校验 → 谓词执行 → lift 检验 → BH-FDR → 合成裁决，顺序不可换）
- services/llm_service.py  _build_discovery_user_msg（符号化窗口 user msg 组装）
- services/sentiment_agent.py  _view_to_payload / _screen_and_serialize
  （WindowView → LLM payload；假设 → Q6 审判 → 预览序列化）
- services/sentiment_agent.py  commit_deep_learn 双轨准入闸门
  （谓词轨 Q6 裁决：ACTIVE→ACTIVE / OBSERVE→EVOLVING / 其余拒绝；
  旧轨 P0-3 闸门保持原行为）

数值案例均经手工核算（见各用例注释）。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from binance_predict.db.models import PatternChangeLog, PatternMemory
from binance_predict.models.schemas import PredicateHypothesis
from binance_predict.services.discovery import (
    VERDICT_ACTIVE,
    VERDICT_OBSERVE,
    VERDICT_REJECT,
    _compose_verdict,
    screen_hypotheses,
)
from binance_predict.services.llm_service import LLMService
from binance_predict.services.sentiment_agent import SentimentAgent
from binance_predict.services.symbolizer import ChannelView, WindowView

# ============================================================
# 共用构造
# ============================================================

DAY_MS = 86_400_000

# 命中谓词：sentiment 通道至少含 1 个「急升」
HIT_PREDICATE = {
    "pred": "count_symbol",
    "channel": "sentiment",
    "symbol": "急升",
    "cmp": ">=",
    "value": 1,
}

# 永不命中谓词（「急升」计数 >=10，测试窗口仅 3 个符号）
NEVER_PREDICATE = {
    "pred": "count_symbol",
    "channel": "sentiment",
    "symbol": "急升",
    "cmp": ">=",
    "value": 10,
}


def _mk_view(i: int, outcome: str, hit: bool) -> WindowView:
    symbols = ["急升", "平", "平"] if hit else ["平", "平", "平"]
    return WindowView(
        start_time=i * DAY_MS,
        outcome=outcome,
        channels={
            "sentiment": ChannelView(
                symbols=symbols, geometry={}, point_count=len(symbols) + 1
            )
        },
    )


def _active_case_views() -> list[WindowView]:
    """强信号验证集（40 窗口，间隔 1 天）。

    命中集 H={5,6,10,15,20,25,30,35}（8 个，其中 UP 7 个：35 为 DOWN）。
    局部对照池（各命中窗口 ±3 天非命中窗口，去重）=29 个，其中 UP 9 个。
    ⇒ hit=7/8=0.875 vs base=9/29≈0.310 ⇒ lift≈2.82（≥1.4 且 CI 下界>1）。
    """
    hits = {5, 6, 10, 15, 20, 25, 30, 35}
    hit_up = {5, 6, 10, 15, 20, 25, 30}
    pool_up = {2, 7, 11, 16, 21, 26, 31, 36, 38}
    views: list[WindowView] = []
    for i in range(40):
        if i in hits:
            views.append(_mk_view(i, "UP" if i in hit_up else "DOWN", True))
        else:
            views.append(_mk_view(i, "UP" if i in pool_up else "DOWN", False))
    return views


def _observe_case_views() -> list[WindowView]:
    """弱信号验证集（40 窗口，间隔 1 天）。

    命中集 H={4,9,14,19,24,29,34}（7 个，其中 UP 3 个：4/14/24）。
    局部对照池=30 个，其中 UP 10 个。
    ⇒ hit=3/7≈0.4286 vs base=10/30≈0.3333 ⇒ lift≈1.286 ∈ [1.15, 1.4)。
    """
    hits = {4, 9, 14, 19, 24, 29, 34}
    hit_up = {4, 14, 24}
    pool_up = {1, 3, 6, 8, 11, 16, 21, 26, 31, 36}
    views: list[WindowView] = []
    for i in range(40):
        if i in hits:
            views.append(_mk_view(i, "UP" if i in hit_up else "DOWN", True))
        else:
            views.append(_mk_view(i, "UP" if i in pool_up else "DOWN", False))
    return views


# ============================================================
# discovery.screen_hypotheses（Q6 初筛流水线）
# ============================================================


class TestScreenHypotheses:
    def test_active_track_with_fdr(self):
        """强信号：lift≥1.4 且 CI 下界>1 且 FDR 通过 → ACTIVE。"""
        out = screen_hypotheses(
            [{"predicate": HIT_PREDICATE, "target_outcome": "UP"}],
            _active_case_views(),
        )
        assert len(out) == 1
        s = out[0]
        assert s.verdict == VERDICT_ACTIVE
        assert s.fdr_passed is True
        assert s.reject_reason is None
        assert len(s.hit_start_times) == 8
        lr = s.lift_result
        assert lr is not None
        assert lr.hit_events == 7 and lr.hit_total == 8
        assert lr.base_events == 9 and lr.base_total == 29
        assert lr.lift == pytest.approx(0.875 / (9 / 29), rel=1e-6)
        assert lr.lift >= 1.4 and lr.ci_lower > 1.0

    def test_observe_track(self):
        """弱信号：1.15≤lift<1.4 → OBSERVE（观察仓，不要求 FDR）。"""
        out = screen_hypotheses(
            [{"predicate": HIT_PREDICATE, "target_outcome": "UP"}],
            _observe_case_views(),
        )
        s = out[0]
        assert s.verdict == VERDICT_OBSERVE
        assert s.lift_result is not None
        assert 1.15 <= s.lift_result.lift < 1.4
        assert len(s.hit_start_times) == 7

    def test_reject_insufficient_hits(self):
        """命中 <5 → 直接 REJECT，不进统计检验（lift_result 为 None）。"""
        out = screen_hypotheses(
            [{"predicate": NEVER_PREDICATE, "target_outcome": "UP"}],
            _active_case_views(),
        )
        s = out[0]
        assert s.verdict == VERDICT_REJECT
        assert s.lift_result is None
        assert s.reject_reason.startswith("insufficient_hits")

    def test_reject_invalid_predicate(self):
        """DSL 白名单外谓词 → REJECT（predicate_invalid）。"""
        out = screen_hypotheses(
            [{"predicate": {"pred": "fft_energy"}, "target_outcome": "UP"}],
            _active_case_views(),
        )
        s = out[0]
        assert s.verdict == VERDICT_REJECT
        assert s.predicate is None
        assert s.reject_reason.startswith("predicate_invalid")

    def test_reject_invalid_target(self):
        """target_outcome 非 UP/DOWN → REJECT（invalid_target）。"""
        out = screen_hypotheses(
            [{"predicate": HIT_PREDICATE, "target_outcome": "NOISE"}],
            _active_case_views(),
        )
        s = out[0]
        assert s.verdict == VERDICT_REJECT
        assert s.reject_reason.startswith("invalid_target")

    def test_results_aligned_with_input_order(self):
        """混合输入：返回与输入等长同序，各裁决互不串位。"""
        hyps = [
            {"predicate": HIT_PREDICATE, "target_outcome": "UP"},
            {"predicate": {"pred": "fft_energy"}, "target_outcome": "UP"},
            {"predicate": NEVER_PREDICATE, "target_outcome": "UP"},
            {"predicate": HIT_PREDICATE, "target_outcome": "NOISE"},
        ]
        out = screen_hypotheses(hyps, _active_case_views())
        assert [s.index for s in out] == [0, 1, 2, 3]
        assert out[0].verdict == VERDICT_ACTIVE
        assert out[1].reject_reason.startswith("predicate_invalid")
        assert out[2].reject_reason.startswith("insufficient_hits")
        assert out[3].reject_reason.startswith("invalid_target")


class TestComposeVerdict:
    """Q6 合成裁决真值表（双轨判定 × FDR 标记）。"""

    def test_active_requires_both_track_and_fdr(self):
        assert _compose_verdict("ACTIVE", True) == VERDICT_ACTIVE

    def test_active_track_without_fdr_downgrades_to_observe(self):
        # 双轨 ACTIVE 但 FDR 未过 = 统计功效不足而非模式无效 → 观察仓攒样本
        assert _compose_verdict("ACTIVE", False) == VERDICT_OBSERVE

    def test_observe_track_stays_observe_regardless_of_fdr(self):
        assert _compose_verdict("OBSERVE", True) == VERDICT_OBSERVE
        assert _compose_verdict("OBSERVE", False) == VERDICT_OBSERVE

    def test_reject_track_always_rejects(self):
        assert _compose_verdict("REJECT", True) == VERDICT_REJECT
        assert _compose_verdict("REJECT", False) == VERDICT_REJECT


# ============================================================
# llm_service._build_discovery_user_msg（Q1/Q2 符号化输入组装）
# ============================================================


class TestDiscoveryUserMsg:
    def test_symbols_geometry_outcome_and_missing_channel(self):
        svc = LLMService.__new__(LLMService)  # 绕过客户端初始化（纯组装逻辑）
        msg = svc._build_discovery_user_msg([
            {
                "start_time": 1754000000000,
                "outcome": "UP",
                "channels": {
                    "sentiment": {
                        "symbols": ["平", "急升"],
                        "geometry": {
                            "peak_count": 1,
                            "area_ratio": 0.62,
                            "curliness": 1.8,
                            "extremum_spacing": "mixed",
                        },
                    },
                    "price": {
                        "symbols": ["平", "缓升"],
                        "geometry": {
                            "peak_count": 0,
                            "area_ratio": 0.5,
                            "curliness": 1.0,
                            "extremum_spacing": "insufficient",
                        },
                    },
                    # volume 通道缺失
                },
            },
        ])
        assert "发现集窗口（共 1 个" in msg
        assert "→ UP" in msg
        assert "sentiment: 平,急升" in msg
        assert "price: 平,缓升" in msg
        assert "volume: 缺" in msg
        assert "几何:" in msg
        assert "峰1" in msg
        assert "面积比0.62" in msg
        assert "间距mixed" in msg

    def test_empty_windows_still_has_header(self):
        svc = LLMService.__new__(LLMService)
        msg = svc._build_discovery_user_msg([])
        assert "共 0 个" in msg
        assert "时间范围未知" in msg


# ============================================================
# sentiment_agent._view_to_payload / _screen_and_serialize
# ============================================================


class TestViewToPayload:
    def test_extrema_stripped_and_symbols_kept(self):
        view = WindowView(
            start_time=123,
            outcome="DOWN",
            channels={
                "sentiment": ChannelView(
                    symbols=["急升"],
                    geometry={
                        "peak_count": 1,
                        "extrema": [{"pos": 0.5, "kind": "peak"}],
                        "area_ratio": 0.9,
                        "curliness": 2.0,
                        "extremum_spacing": "mixed",
                    },
                    point_count=2,
                )
            },
        )
        payload = SentimentAgent._view_to_payload(view)
        assert payload["start_time"] == 123
        assert payload["outcome"] == "DOWN"
        ch = payload["channels"]["sentiment"]
        assert ch["symbols"] == ["急升"]
        # extrema 明细不进 LLM payload（token 开销），统计结论保留
        assert "extrema" not in ch["geometry"]
        assert ch["geometry"]["peak_count"] == 1


class TestScreenAndSerialize:
    def test_field_mapping_and_incomplete_filtered(self):
        agent = SentimentAgent.__new__(SentimentAgent)
        h_ok = PredicateHypothesis(
            pattern_name="尾盘急升",
            description="情绪尾盘急升后 UP 偏向",
            predicate=HIT_PREDICATE,
            target_outcome="UP",
            confidence_score=0.7,
            rationale="多窗口共性结构",
        )
        h_partial = SimpleNamespace(  # 流式半成品：predicate 未填满
            pattern_name="半成品", predicate=None, target_outcome="UP"
        )
        out = agent._screen_and_serialize(
            [h_ok, h_partial], _active_case_views(), "v-test"
        )
        assert len(out) == 1  # 半成品被过滤
        d = out[0]
        assert d["operation"] == "CREATE"
        assert d["pattern_name"] == "尾盘急升"
        assert d["predicted_direction"] == "UP"  # target_outcome 映射
        assert d["change_reason"] == "多窗口共性结构"  # rationale 映射
        assert d["predicate"] == HIT_PREDICATE
        assert d["binning_version"] == "v-test"
        assert d["discovery_method"] == "LLM_DEEP"
        # Q6 审判证据
        assert d["screen_verdict"] == VERDICT_ACTIVE
        assert d["screen_lift"] >= 1.4
        assert d["screen_hit_count"] == 8
        assert d["screen_reject_reason"] is None
        # 旧轨 holdout_* 三列不由新轨填充
        assert d.get("holdout_ci_lower") is None


# ============================================================
# sentiment_agent.commit_deep_learn 双轨准入闸门
# ============================================================


class _FakeCtx:
    """伪 async 上下文（session / begin_nested 通用）。"""

    def __init__(self, session: object = None) -> None:
        self._session = session

    async def __aenter__(self) -> object:
        return self._session

    async def __aexit__(self, *exc: object) -> bool:
        return False


def _commit_session(added: list) -> MagicMock:
    """伪 session：无既有 ACTIVE 模式；add 的对象收集进 added。"""
    session = MagicMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = []
    session.execute = AsyncMock(return_value=result)
    session.add = MagicMock(side_effect=lambda obj: added.append(obj))
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.begin_nested = MagicMock(return_value=_FakeCtx(session))
    return session


async def _run_commit(discoveries: list[dict]) -> tuple[dict, list]:
    agent = SentimentAgent.__new__(SentimentAgent)
    agent._deep_learn_lock = asyncio.Lock()
    agent._pattern_write_lock = asyncio.Lock()
    added: list = []
    session = _commit_session(added)
    factory = MagicMock(return_value=_FakeCtx(session))
    with patch(
        "binance_predict.services.sentiment_agent.async_session_factory", factory
    ):
        result = await agent.commit_deep_learn(discoveries)
    return result, added


def _added_patterns(added: list) -> list[PatternMemory]:
    return [o for o in added if isinstance(o, PatternMemory)]


def _predicate_discovery(verdict: str | None, operation: str = "CREATE") -> dict:
    d = {
        "operation": operation,
        "pattern_name": "尾盘急升",
        "description": "情绪尾盘急升后 UP 偏向",
        "predicted_direction": "UP",
        "confidence_score": 0.7,
        "change_reason": "多窗口共性结构",
        "discovery_method": "LLM_DEEP",
        "predicate": HIT_PREDICATE,
        "binning_version": "v-test",
        "screen_lift": 2.8,
        "screen_hit_count": 8,
    }
    if verdict is not None:
        d["screen_verdict"] = verdict
    return d


def _legacy_discovery(ci_lower: float, samples: int) -> dict:
    return {
        "operation": "CREATE",
        "pattern_name": "PY-UP-C0-n30",
        "description": "聚类簇",
        "curve_features": {"_feature_vector": [0.1, 0.2]},
        "conditions": {},
        "predicted_direction": "UP",
        "confidence_score": 0.9,
        "change_reason": "PY_CLUSTER 自动发现",
        "discovery_method": "PY_CLUSTER",
        "holdout_win_rate": 0.65,
        "holdout_sample_count": samples,
        "holdout_ci_lower": ci_lower,
    }


@pytest.mark.asyncio
class TestCommitDualTrackGate:
    async def test_predicate_active_writes_active(self):
        result, added = await _run_commit([_predicate_discovery("ACTIVE")])
        assert result["written"] == 1
        assert result["rejected"] == [] and result["failed"] == []
        patterns = _added_patterns(added)
        assert len(patterns) == 1
        p = patterns[0]
        assert p.status == "ACTIVE"  # Q6 直上线
        assert p.predicate == HIT_PREDICATE
        assert p.binning_version == "v-test"
        assert p.discovery_method == "LLM_DEEP"
        # confidence 保留 LLM 主观先验（审判证据在 predicate/screen_*）
        assert p.confidence_score == 0.7
        # holdout_* 三列不由谓词轨填充
        assert p.holdout_ci_lower is None
        # 变更日志同步落库
        assert any(isinstance(o, PatternChangeLog) for o in added)

    async def test_predicate_observe_writes_evolving(self):
        result, added = await _run_commit([_predicate_discovery("OBSERVE")])
        assert result["written"] == 1
        p = _added_patterns(added)[0]
        assert p.status == "EVOLVING"  # 观察仓攒样本
        assert p.predicate == HIT_PREDICATE

    async def test_predicate_reject_verdict_rejected(self):
        result, added = await _run_commit([
            _predicate_discovery("REJECT"),
        ])
        assert result["written"] == 0
        assert len(result["rejected"]) == 1
        assert "REJECT" in result["rejected"][0]["reason"]
        assert _added_patterns(added) == []

    async def test_predicate_missing_verdict_rejected(self):
        result, _ = await _run_commit([_predicate_discovery(None)])
        assert result["written"] == 0
        assert "缺失" in result["rejected"][0]["reason"]

    async def test_predicate_update_rejected(self):
        # 谓词假设无 UPDATE 语义（Phase 2 仅 CREATE）
        result, _ = await _run_commit([_predicate_discovery("ACTIVE", "UPDATE")])
        assert result["written"] == 0
        assert "仅支持 CREATE" in result["rejected"][0]["reason"]

    async def test_legacy_gate_pass_keeps_p0_3_behavior(self):
        min_s = 50  # settings.agent_deep_learn_min_holdout_samples 默认
        result, added = await _run_commit([_legacy_discovery(0.6, min_s + 10)])
        assert result["written"] == 1
        p = _added_patterns(added)[0]
        assert p.status == "ACTIVE"
        assert p.predicate is None
        # P0-3：confidence 用 holdout_ci_lower 覆盖 LLM 主观值
        assert p.confidence_score == 0.6
        assert p.holdout_ci_lower == 0.6
        assert p.holdout_sample_count == min_s + 10

    async def test_legacy_gate_reject_low_ci(self):
        result, _ = await _run_commit([_legacy_discovery(0.4, 60)])
        assert result["written"] == 0
        assert "未过准入闸门" in result["rejected"][0]["reason"]

    async def test_legacy_missing_curve_features_rejected(self):
        d = _legacy_discovery(0.6, 60)
        d["curve_features"] = {}
        result, _ = await _run_commit([d])
        assert result["written"] == 0
        assert "字段残缺" in result["rejected"][0]["reason"]
