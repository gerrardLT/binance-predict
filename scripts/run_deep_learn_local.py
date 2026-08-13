#!/usr/bin/env python3
"""本地真实跑一次 deep_learn 全流程（真 LLM、真 DB、真管道）。

用途：验证 V1.1（决策点截断对齐 + Q6 经济闸）在真实历史数据上的端到端行为，
并直观展示每条假设的 lift 审判证据与经济闸证据。

数据前提：先跑 scripts/load_windows_to_db.py 把 sentiment_windows.json 灌入 DB。

注意：
- 真调 LLM（settings.decision_model），产生一次 API 费用（符号串输入，量级几分钱）
- 数据是 2026-07 月历史，故把 days_back 放宽到 40 天以覆盖
- deep_learn 只产出预览（不写 pattern_memory），写库需另行 commit_deep_learn

用法：
    python scripts/run_deep_learn_local.py [--max-windows 100]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from sqlalchemy import select  # noqa: E402

from binance_predict.config.settings import settings  # noqa: E402
from binance_predict.db.engine import async_session_factory  # noqa: E402
from binance_predict.db.models import LLMTrace  # noqa: E402
from binance_predict.models.schemas import DiscoveryOutput, PredicateHypothesis  # noqa: E402
from binance_predict.services.llm_service import LLMService  # noqa: E402
from binance_predict.services.prediction_trading import BinancePredictionTrader  # noqa: E402
from binance_predict.services.sentiment_agent import SentimentAgent  # noqa: E402

# mock-llm 模式的代表性假设（取自 2026-08-11 台架全空间扫描，覆盖各裁决档位）
_MOCK_HYPOTHESES = [
    # 台架全空间 EV 最高的 ACTIVE（symbol_at early 缓降 DOWN, EV+0.107）
    ("早段缓降看跌", {"pred": "symbol_at", "channel": "sentiment",
     "segment": "early", "symbol": "缓降"}, "DOWN"),
    # ACTIVE 但台架 EV 为负（count_symbol 缓降>=3 DOWN, lift 高 EV-0.058）→ 经济闸应降级
    ("多次缓降看跌", {"pred": "count_symbol", "channel": "sentiment",
     "symbol": "缓降", "cmp": ">=", "value": 3}, "DOWN"),
    # ACTIVE，台架 EV+0.098（symbol_at late 急降 DOWN）
    ("尾盘急降看跌", {"pred": "symbol_at", "channel": "sentiment",
     "segment": "late", "symbol": "急降"}, "DOWN"),
    # ACTIVE，台架 EV 负（has_subseq 急升→缓升 UP）→ 经济闸应降级
    ("急升后续缓升", {"pred": "has_subseq", "channel": "sentiment",
     "symbols": ["急升", "缓升"]}, "UP"),
    # L2 跨通道（sync sentiment→price）
    ("情价同步", {"pred": "sync", "channel_a": "sentiment", "channel_b": "price",
     "cmp": ">=", "value": 0.7}, "DOWN"),
    # 永不命中（平>=10）→ REJECT 演示
    ("极度平稳", {"pred": "count_symbol", "channel": "sentiment",
     "symbol": "平", "cmp": ">=", "value": 10}, "UP"),
]


def _install_mock_llm(agent: SentimentAgent) -> None:
    """用固定假设集替换真实 LLM 调用（免费、确定性，演示审判管道）。"""
    async def _mock_deep_learn(symbolized_windows, timeout, feedback=None):
        return DiscoveryOutput(
            reasoning=(
                "[mock] 本地 key 失效，用台架代表性假设演示管道。"
                f"输入符号化窗口={len(symbolized_windows)} 个，"
                f"反馈包负样本={len((feedback or {}).get('negatives', []))} 条。"
            ),
            hypotheses=[
                PredicateHypothesis(
                    pattern_name=name,
                    description=f"mock 假设：{name}",
                    predicate=pred,
                    target_outcome=target,
                    confidence_score=0.7,
                    rationale="台架代表性样本",
                )
                for name, pred, target in _MOCK_HYPOTHESES
            ],
        )

    agent._llm.agent_deep_learn = _mock_deep_learn  # type: ignore[method-assign]


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-windows", type=int, default=100)
    ap.add_argument("--days-back", type=int, default=40,
                    help="历史数据为 2026-07 月，默认放宽到 40 天以覆盖")
    ap.add_argument("--out", default="output/deep_learn_local_result.json")
    ap.add_argument("--mock-llm", action="store_true",
                    help="不真调 LLM，用台架代表性假设演示审判管道（免费、确定性）")
    ap.add_argument("--model", default=None, help="覆盖 DECISION_MODEL（试其他模型）")
    ap.add_argument("--base-url", default=None, help="覆盖 DEEPSEEK_BASE_URL（中转端点）")
    ap.add_argument("--api-key", default=None, help="覆盖 DEEPSEEK_API_KEY")
    args = ap.parse_args()

    # CLI 覆盖 LLM 配置（试其他模型/中转，不改 .env）
    if args.base_url:
        settings.deepseek_base_url = args.base_url
    if args.api_key:
        settings.deepseek_api_key = args.api_key
    if args.model:
        settings.decision_model = args.model

    # 放宽取数时间窗以覆盖历史导出数据（本地回放专用，不改 .env）
    settings.agent_deep_learn_days_back = args.days_back

    print(f"[配置] model={settings.decision_model} | max_windows={args.max_windows} "
          f"| days_back={args.days_back} | 决策点={settings.agent_decision_point_sec}s "
          f"| 经济闸={'开' if settings.agent_ev_gate_enabled else '关'}")

    agent = SentimentAgent(llm=LLMService(), trader=BinancePredictionTrader())
    if args.mock_llm:
        _install_mock_llm(agent)
        print("[模式] mock-llm：不真调 LLM，用固定假设演示审判管道")
    result = await agent.deep_learn(max_windows=args.max_windows)

    # 读取本次 DEEP_LEARN 的完整 LLM 轨迹（fire-and-forget 落库，稍等写库）
    trace = None
    if not args.mock_llm:
        await asyncio.sleep(2.0)
        trace = await _read_latest_trace()

    discoveries = result.get("discoveries", [])
    print("\n" + "=" * 100)
    print(f"发现完成 | train={result.get('train_count')} holdout={result.get('holdout_count')} "
          f"| 分箱版本={result.get('binning_version')} | 假设数={len(discoveries)}")
    print("=" * 100)
    print("\n[LLM reasoning 摘要]")
    print((result.get("reasoning") or "")[:800] or "(无)")

    print("\n[逐假设审判结果]")
    hdr = (f"{'名称':<16}{'向':>4}{'裁决':>9}{'闸':>5}{'lift':>7}"
           f"{'筛中':>5}{'EV费后':>9}{'EV_CI下':>9}{'实中':>5}")
    print(hdr)
    print("-" * 100)
    for d in discoveries:
        pred = json.dumps(d.get("predicate"), ensure_ascii=False)
        print(f"{str(d.get('pattern_name'))[:15]:<16}"
              f"{str(d.get('predicted_direction')):>4}"
              f"{str(d.get('screen_verdict')):>9}"
              f"{('过' if d.get('screen_ev_passed') else '否') if d.get('screen_ev_passed') is not None else '-':>5}"
              f"{_fmt(d.get('screen_lift')):>7}"
              f"{_fmt(d.get('screen_hit_count')):>5}"
              f"{_fmt(d.get('screen_ev')):>9}"
              f"{_fmt(d.get('screen_ev_ci_lower')):>9}"
              f"{_fmt(d.get('screen_ev_fires')):>5}")
        print(f"    谓词 {pred[:90]}")

    verdicts = {}
    for d in discoveries:
        verdicts[d.get("screen_verdict")] = verdicts.get(d.get("screen_verdict"), 0) + 1
    gated = [d for d in discoveries
             if d.get("screen_ev_passed") is False
             and d.get("screen_verdict") == "OBSERVE"]
    print("\n[汇总] 裁决分布:", verdicts)
    print(f"[经济闸] 双轨 ACTIVE 但因 EV 不足降级 OBSERVE 的假设数: "
          f"{sum(1 for d in discoveries if d.get('screen_ev_passed') is False)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[已写入] {args.out}")

    # 推理全过程记录（system prompt + user message + LLM 原始输出 + 审判证据）
    md_path = os.path.splitext(args.out)[0] + "_trace.md"
    _export_trace_markdown(trace, result, args, md_path)
    print(f"[推理记录] {md_path}")
    return 0


async def _read_latest_trace() -> LLMTrace | None:
    """读取最近一次 DEEP_LEARN 的完整 LLM 轨迹。"""
    async with async_session_factory() as session:
        row = (await session.execute(
            select(LLMTrace)
            .where(LLMTrace.phase == "DEEP_LEARN")
            .order_by(LLMTrace.id.desc())
            .limit(1)
        )).scalar_one_or_none()
    return row


def _export_trace_markdown(trace, result: dict, args, path: str) -> None:
    """导出推理全过程为可读 markdown（LLM 看了什么/想了什么/程序怎么判）。"""
    L: list[str] = []
    L.append("# Deep Learn 推理全过程记录\n")
    L.append(f"- 模型: {settings.decision_model} @ {settings.deepseek_base_url}")
    L.append(f"- mock_llm: {args.mock_llm} | max_windows={args.max_windows} "
             f"| 决策点={settings.agent_decision_point_sec}s "
             f"| 经济闸={'开' if settings.agent_ev_gate_enabled else '关'}")
    L.append(f"- train={result.get('train_count')} holdout={result.get('holdout_count')} "
             f"| 分箱版本={result.get('binning_version')}")
    if trace is not None:
        L.append(f"- token: prompt={trace.prompt_tokens} completion={trace.completion_tokens} "
                 f"| 成本≈¥{trace.estimated_cost_yuan} | 延迟={trace.latency_s}s")
    L.append("\n---\n\n## 1. System Prompt（发现器角色契约）\n")
    L.append("```\n" + (trace.system_prompt if trace else "(mock 模式无)") + "\n```")
    L.append("\n## 2. User Message（符号化窗口 + 反馈包）\n")
    L.append("```\n" + (trace.user_message if trace else "(mock 模式无)") + "\n```")
    L.append("\n## 3. LLM 原始输出（reasoning + hypotheses）\n")
    if trace and trace.assistant_output:
        L.append("```json\n" + json.dumps(trace.assistant_output, ensure_ascii=False, indent=2) + "\n```")
    else:
        L.append("```json\n" + json.dumps({
            "reasoning": result.get("reasoning"),
            "hypotheses": [d.get("predicate") for d in result.get("discoveries", [])],
        }, ensure_ascii=False, indent=2) + "\n```")
    L.append("\n## 4. 程序审判结果（Q6 初筛 + 经济闸）\n")
    L.append("| 名称 | 方向 | lift | 筛中 | 裁决 | 闸 | EV费后 | EV_CI下 | 实中 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for d in result.get("discoveries", []):
        L.append(
            f"| {d.get('pattern_name')} | {d.get('predicted_direction')} "
            f"| {d.get('screen_lift')} | {d.get('screen_hit_count')} "
            f"| {d.get('screen_verdict')} | {d.get('screen_ev_passed')} "
            f"| {d.get('screen_ev')} | {d.get('screen_ev_ci_lower')} "
            f"| {d.get('screen_ev_fires')} |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
