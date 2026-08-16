"""科学裁决器（M3）：假设 → 回测引擎 → 硬门禁 → 判决。纯代码，LLM 与人都碰不到口径。

裁决流程（对每个 PENDING_REVIEW 的假设）：
  1. 参数重建：SceneParams.from_params_json(version_row.params)
  2. 同窗 A/B：同一份 180 天官方数据，ACTIVE 基线参数与假设参数各跑一遍
     build_events（消除窗口漂移——对照必须同数据）
  3. 硬门禁（代码判定，任一不过即 REJECTED）：
     G1 盲验 n ≥ 60（假设参数下每个场景）
     G2 胜率 ≥ 基线 + 门槛（基础 2pp，随累计假设数按 Bonferroni 近似上调）
     G3 L4 最差月不劣于基线最差月（-5pp 容差）
     G4 功效预检：验证集可检测下限 ≤ 声称改善（review_note 中的 expected_impact_pp）
  4. Verdict（完整引擎输出快照）写回 backtest_report；
     PASS → status=SHADOW（影子并行候选，仍需人工 promote 才 ACTIVE）+ 审核邮件
     FAIL → status=REJECTED + 邮件通知
"""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field

from loguru import logger
from sqlalchemy import select

from ..backtest import build_events, fetch_klines, multiple_testing_threshold, power_preflight, wilson
from ..config.settings import settings
from ..db.engine import async_session_factory
from ..db.models import SceneParamVersion
from ..services.scene_params import SceneParams
from .alerting import send_plain_email

# 硬门禁参数（进化体系的三闸）
MIN_VALIDATION_N = 60          # G1：盲验样本下限
BASE_MARGIN_PP = 2.0           # G2：基础改善门槛（pp），随假设数上调
WORST_MONTH_TOLERANCE_PP = 5.0 # G3：最差月容差
DAYS = 180


@dataclass
class Verdict:
    passed: bool
    gates: list[dict] = field(default_factory=list)
    report: dict = field(default_factory=dict)

    @property
    def reasons(self) -> list[str]:
        return [g["detail"] for g in self.gates if not g["passed"]]


def _scene_stats(events: list[dict]) -> dict:
    """从事件列表统计两场景的验证集成绩与按月稳健性（切分由调用方标记 is_val）。"""
    out = {}
    for tag, win_of in (("scene1", lambda e: bool(e["next_down"])),
                        ("scene2", lambda e: not bool(e["next_down"]))):
        val = [e for e in events if e.get(tag) and e.get("has_next") and e.get("is_val")]
        n = len(val)
        if n == 0:
            out[tag] = {"n": 0, "p": None, "ci": None, "worst_month_p": None}
            continue
        wins = [win_of(e) for e in val]
        p = sum(wins) / n
        lo, hi = wilson(p, n)
        by_month: dict[str, list[bool]] = {}
        for e, w in zip(val, wins):
            by_month.setdefault(e["month"], []).append(w)
        monthly = {m: sum(ws) / len(ws) for m, ws in by_month.items() if len(ws) >= 5}
        out[tag] = {
            "n": n, "k": sum(wins), "p": round(p, 4),
            "ci": [round(lo, 4), round(hi, 4)],
            "worst_month_p": round(min(monthly.values()), 4) if monthly else None,
        }
    return out


def _run_engine(params: SceneParams, c5: list[tuple], now_ms: int, split_ratio: float = 2 / 3) -> list[dict]:
    """CPU/IO 内核（同步）：构建事件并标记验证集。在线程池中执行。"""
    res = build_events(c5, params, now_ms)
    cyc_arr = res["cyc_arr"]
    n = len(cyc_arr)
    if n == 0:
        return []
    split_cyc = int(cyc_arr[int(n * split_ratio)])
    for e in res["events"]:
        e["is_val"] = e["cyc"] >= split_cyc
    return res["events"]


class HypothesisArbiter:
    """科学裁决器（同进程调用 backtest 包，不走子进程）。"""

    async def adjudicate(self, version_id: int) -> Verdict | None:
        """对一条 PENDING_REVIEW 假设执行四层门禁裁决并回写状态。"""
        async with async_session_factory() as session:
            row = await session.get(SceneParamVersion, version_id)
            if row is None or row.status != "PENDING_REVIEW":
                logger.warning("裁决跳过：版本 #{} 不存在或非 PENDING_REVIEW", version_id)
                return None
            hypo_params_raw = dict(row.params)

        active_version, active_params_raw = await self._active_params()
        hypo = SceneParams.from_params_json(hypo_params_raw)
        base = SceneParams.from_params_json(active_params_raw)

        logger.info("裁决开始 #{} | 假设参数={} | 基线={}", version_id, hypo.to_params_json(), active_version)
        t0 = time.monotonic()
        now_ms = int(time.time() * 1000)
        kl = await asyncio.to_thread(fetch_klines, "5m", now_ms - DAYS * 86_400_000, now_ms)
        c5 = [(int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])) for k in kl]
        if c5 and c5[-1][0] + 300_000 > now_ms:
            c5.pop()
        events_hypo = await asyncio.to_thread(_run_engine, hypo, c5, now_ms)
        events_base = await asyncio.to_thread(_run_engine, base, c5, now_ms)
        logger.info("裁决数据就绪 | 事件 假设={} 基线={} | 耗时 {:.0f}s",
                    len(events_hypo), len(events_base), time.monotonic() - t0)

        stats_h = _scene_stats(events_hypo)
        stats_b = _scene_stats(events_base)
        n_hypotheses = await self._hypotheses_total()
        mt = multiple_testing_threshold(BASE_MARGIN_PP, n_hypotheses)
        claimed_pp = await self._claimed_pp(version_id)

        # 受影响场景判定：假设参数相对基线的差异 → 哪些场景被改动
        affected = self._affected_scenes(hypo.to_params_json(), base.to_params_json())
        verdict = self._apply_gates(stats_h, stats_b, mt, claimed_pp, affected)
        verdict.report = {
            "version_id": version_id,
            "hypo_params": hypo.to_params_json(),
            "baseline_version": active_version,
            "baseline_params": base.to_params_json(),
            "affected_scenes": sorted(affected),
            "hypo_validation": stats_h,
            "baseline_validation": stats_b,
            "multiple_testing": mt,
            "claimed_pp": claimed_pp,
            "engine_elapsed_s": round(time.monotonic() - t0, 1),
        }

        new_status = "SHADOW" if verdict.passed else "REJECTED"
        async with async_session_factory() as session:
            row = await session.get(SceneParamVersion, version_id)
            row.status = new_status
            row.backtest_report = verdict.report
            await session.commit()
        await self._notify(row, verdict)
        logger.info("裁决完成 #{} | {} | 门禁: {}", version_id, new_status, verdict.reasons or "全部通过")
        return verdict

    # ------------------------------------------------------------------

    def _apply_gates(self, stats_h: dict, stats_b: dict, mt: dict, claimed_pp: float,
                     affected: set[str] | None = None) -> Verdict:
        """四层门禁。

        G1 样本量对两场景都查（数据完整性）；G2/G3/G4 仅对假设实际改动的
        场景生效（affected）——只改 close_pos_min 的假设不应被未受影响的
        场景②的 +0.0pp 改善卡闸。affected=None 表示两场景都查（保守）。
        """
        if affected is None:
            affected = {"scene1", "scene2"}
        gates = []
        for tag, label in (("scene1", "场景①"), ("scene2", "场景②")):
            h, b = stats_h[tag], stats_b[tag]
            # G1 样本量（两场景都查）
            ok = h["n"] >= MIN_VALIDATION_N
            gates.append({"gate": f"G1-{tag}", "passed": ok,
                          "detail": f"{label} 盲验 n={h['n']}（门槛 {MIN_VALIDATION_N}）" if not ok else ""})
            if not ok or h["p"] is None or b["p"] is None:
                continue
            if tag not in affected:
                continue  # 未受影响的场景不设 G2/G3/G4 障
            # G2 改善幅度（vs 同窗基线，Bonferroni 校正门槛）
            delta_pp = (h["p"] - b["p"]) * 100
            ok = delta_pp >= mt["required_pp"]
            gates.append({"gate": f"G2-{tag}", "passed": ok,
                          "detail": (f"{label} 改善 {delta_pp:+.1f}pp < 门槛 {mt['required_pp']}pp"
                                     if not ok else "")})
            # G3 最差月稳健性
            if h["worst_month_p"] is not None and b["worst_month_p"] is not None:
                drop_pp = (b["worst_month_p"] - h["worst_month_p"]) * 100
                ok = drop_pp <= WORST_MONTH_TOLERANCE_PP
                gates.append({"gate": f"G3-{tag}", "passed": ok,
                              "detail": (f"{label} 最差月 {h['worst_month_p']:.1%} 劣于基线 "
                                         f"{b['worst_month_p']:.1%} 超 {WORST_MONTH_TOLERANCE_PP}pp"
                                         if not ok else "")})
            # G4 功效预检（对假设声称的改善）
            pw = power_preflight(h["n"], claimed_effect_pp=max(claimed_pp, 0.1))
            ok = pw["verdict"] == "OK"
            gates.append({"gate": f"G4-{tag}", "passed": ok,
                          "detail": pw["note"] if not ok else ""})
        return Verdict(passed=all(g["passed"] for g in gates), gates=gates)

    async def _claimed_pp(self, version_id: int) -> float:
        """从 review_note 解析声称改善（格式：'... | 声称改善: X.Xpp'）。"""
        try:
            async with async_session_factory() as session:
                row = await session.get(SceneParamVersion, version_id)
                m = re.search(r"声称改善[:：]\s*([\d.]+)pp", row.review_note or "")
                return float(m.group(1)) if m else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _affected_scenes(hypo: dict, base: dict) -> set[str]:
        """参数差异 → 受影响场景映射。

        close_pos_min → 仅场景①；vol_ratio_min/vol_ma_window → 仅场景②；
        eps/level_lookbacks → 两场景都受影响。无差异（纯重复假设）→ 两场景
        都查（保守，且 G2 的 +0pp 会让它自然被拒）。
        """
        diff = {k for k in ("close_pos_min", "vol_ratio_min", "vol_ma_window", "eps",
                            "level_lookbacks") if hypo.get(k) != base.get(k)}
        if not diff:
            return {"scene1", "scene2"}
        affected: set[str] = set()
        if diff & {"close_pos_min"}:
            affected.add("scene1")
        if diff & {"vol_ratio_min", "vol_ma_window"}:
            affected.add("scene2")
        if diff & {"eps", "level_lookbacks"}:
            affected |= {"scene1", "scene2"}
        return affected or {"scene1", "scene2"}

    async def _active_params(self) -> tuple[str, dict]:
        async with async_session_factory() as session:
            stmt = (select(SceneParamVersion).where(SceneParamVersion.status == "ACTIVE")
                    .order_by(SceneParamVersion.activated_at.desc()).limit(1))
            row = (await session.execute(stmt)).scalars().first()
            if row is None:
                return "v1-default", SceneParams().to_params_json()
            return row.version, dict(row.params)

    async def _hypotheses_total(self) -> int:
        async with async_session_factory() as session:
            stmt = select(SceneParamVersion.id)
            rows = (await session.execute(stmt)).all()
            return len(rows)

    async def _notify(self, row: SceneParamVersion, verdict: Verdict) -> None:
        """裁决结果邮件（PASS 待人工审核；FAIL 通知归档）。fire-and-forget 语义（本方法内已 try）。"""
        try:
            status_label = "通过·待人工审核" if verdict.passed else "否决"
            subject = f"[场景假设裁决·{status_label}] #{row.id} {row.version}"
            body = (
                f"版本：{row.version}\n假设参数：{verdict.report.get('hypo_params')}\n"
                f"对照基线：{verdict.report.get('baseline_version')}\n\n"
                f"验证集对照（同窗 A/B）：\n  假设：{verdict.report.get('hypo_validation')}\n"
                f"  基线：{verdict.report.get('baseline_validation')}\n\n"
                f"多重检验：{verdict.report.get('multiple_testing', {}).get('note')}\n"
                + ("".join(f"未过门禁：{r}\n" for r in verdict.reasons) if verdict.reasons else "全部门禁通过\n")
                + (f"\n通过后为 SHADOW 影子身份（只落表不发邮件）；"
                   f"人工放行请调用 POST /api/scene/versions/{row.id}/promote\n" if verdict.passed else "")
            )
            await send_plain_email(subject, body)
        except Exception as exc:
            logger.warning("裁决邮件发送失败（不影响裁决结果）| {}", exc)
