#!/usr/bin/env python3
"""quote_momentum_v2 进化探索（2026-08-28）：预注册假设 × 统一统计流水线。

输入：
  - output/qm_v2_eval/snapshot.json（local_qm_v2_evaluation.py --stage data 产物）
  - config/qm_v2_hypotheses.json（预注册假设清单，运行前锁定，本脚本只读）

流水线（宪法三段式，顺序不可换）：
  1. 事件域：v1 重放段按时间 70/30 切发现集/留出集；
  2. 全部假设在发现集求值 → 命中数<10 直接 INSUFFICIENT；
  3. 逐条二项比例 z 检验（基准=发现集 v1 全体胜率）→ BH-FDR(q=0.1)；
  4. 经济闸（费 2%+溢价 0.01 逐注 bootstrap 95% CI 下界>0 且 ≥10 注）；
  5. 存活者只在留出集终验一次：方向一致 + EV CI 下界>0 才进 ACTIVE；
     留出集样本不足 → OBSERVE（攒样本）；终验失败 → REJECT（赢家诅咒）。
  6. K 线代理族（1m 粒度 ~62 天）：门禁阈值单调性的历史一致性，仅方向证据。
  7. vol30/ret1h 特征补齐：情绪曲线仅覆盖本窗 5 分钟，无法回看 30min/1h，
     改从 Binance 公共 1m K 线（缓存 output/qm_v2_eval/k1m_event_range.json，
     缺失时自动拉取）取 ≤触发时点最后一根已完结 1m 的收盘价（严格 ex-ante）。

产出：output/qm_v2_eval/candidates.json（候选表）
      output/qm_v2_eval/report.json + report.md（合并评估结果的最终报告）

用法：python -X utf8 scripts/local_qm_v2_evolution_scan.py [--stage scan|report|all]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import local_qm_v2_evaluation as ev  # noqa: E402  （复用纯函数与事件构建，口径同源）

HYP_FILE = "config/qm_v2_hypotheses.json"
OUT_DIR = "output/qm_v2_eval"
SNAPSHOT = os.path.join(OUT_DIR, "snapshot.json")
EVAL_JSON = os.path.join(OUT_DIR, "evaluation.json")
CAND_OUT = os.path.join(OUT_DIR, "candidates.json")
REPORT_JSON = os.path.join(OUT_DIR, "report.json")
REPORT_MD = os.path.join(OUT_DIR, "report.md")
K1M = "output/klines_1m_2224_720d.json"
K1M_EVENT = os.path.join(OUT_DIR, "k1m_event_range.json")  # 事件区间 1m 缓存（范围随事件动态扩展）

MIN_STAT_N = 10
FDR_Q = 0.1


def _norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))


def prop_z_pvalue(k: int, n: int, p0: float) -> float:
    """二项比例双侧 z 检验（基准胜率 p0）。"""
    if n == 0 or p0 <= 0 or p0 >= 1:
        return 1.0
    p = k / n
    z = (p - p0) / math.sqrt(p0 * (1 - p0) / n)
    return min(1.0, 2.0 * _norm_sf(abs(z)))


def bh_fdr(p_values: list[float], q: float = FDR_Q) -> list[bool]:
    """Benjamini-Hochberg（与 services/verification.py 同逻辑的脚本侧副本）。"""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    passed = [False] * m
    k_max = -1
    for rank, idx in enumerate(order, start=1):
        if p_values[idx] <= rank / m * q:
            k_max = rank
    if k_max > 0:
        for rank, idx in enumerate(order, start=1):
            if rank <= k_max:
                passed[idx] = True
    return passed


def _apply_filters(e: dict, filters: list[dict]) -> bool | None:
    """过滤器求值；任一特征缺失 → None（不计入命中/未命中，记缺失）。"""
    for f in filters:
        v = e.get(f["feature"])
        if v is None:
            return None
        op, val = f["op"], f["value"]
        if op == "lte" and not v <= val:
            return False
        elif op == "gte" and not v >= val:
            return False
        elif op == "lt" and not v < val:
            return False
        elif op == "gt" and not v > val:
            return False
        elif op == "eq" and not v == val:
            return False
        elif op == "in" and v not in val:
            return False
        elif op == "between" and not (val[0] <= v < val[1]):
            return False
    return True


def _kline_close_at_or_before(close_by_open: dict, opens: list[int], ts: int) -> float | None:
    """≤ts 的最后一根已完结 1m 的收盘价（K 线 open_time 语义，严格 ex-ante）。"""
    import bisect
    i = bisect.bisect_right(opens, ts) - 1
    return close_by_open[opens[i]] if i >= 0 else None


def _fetch_k1m(start: int, end: int) -> list:
    """从 Binance 公共 API 拉 [start, end) 区间 1m K 线。"""
    import time
    import urllib.request
    out: list = []
    cur = start
    while cur < end:
        url = (f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m"
               f"&startTime={cur}&limit=1000")
        with urllib.request.urlopen(url, timeout=60) as resp:
            batch = json.load(resp)
        if not batch:
            break
        out.extend(batch)
        cur = int(batch[-1][0]) + 60_000
        print(f"[k1m] 已拉 {len(out)} 根（至 {datetime.fromtimestamp(cur / 1000, timezone.utc):%m-%d %H:%M} UTC）")
        time.sleep(0.3)
    return out


def _load_event_range_1m(events: list[dict]) -> list:
    """事件区间 1m K 线：范围按事件时间戳动态计算（min ts−90min ~ max ts+5min），
    先读缓存，缺口段补拉并合并落盘（幂等）。"""
    lo = min(e["ts"] for e in events) - 90 * 60_000
    hi = max(e["ts"] for e in events) + 5 * 60_000
    bars: list = []
    if os.path.exists(K1M_EVENT):
        with open(K1M_EVENT, encoding="utf-8") as f:
            bars = json.load(f)
    by_open = {int(b[0]) for b in bars}
    cache_lo, cache_hi = (min(by_open), max(by_open)) if by_open else (None, None)
    added = 0
    if cache_lo is None:
        bars = _fetch_k1m(lo, hi)
        added = len(bars)
    else:
        if lo < cache_lo:
            seg = [b for b in _fetch_k1m(lo, cache_lo) if int(b[0]) not in by_open]
            bars = seg + bars
            added += len(seg)
        if hi > cache_hi + 60_000:
            seg = [b for b in _fetch_k1m(cache_hi + 60_000, hi) if int(b[0]) not in by_open]
            bars.extend(seg)
            added += len(seg)
    bars.sort(key=lambda b: int(b[0]))
    if added or cache_lo is None:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(K1M_EVENT, "w", encoding="utf-8") as f:
            json.dump(bars, f)
        print(f"[k1m] 缓存已更新 {K1M_EVENT}（+{added} 根，共 {len(bars)} 根）")
    return bars


def enrich_kline_features(events: list[dict]) -> None:
    """用事件区间 1m K 线补 vol30/ret1h（情绪曲线仅覆盖本窗，无法回看）。"""
    bars = _load_event_range_1m(events)
    close_by_open = {int(b[0]): float(b[4]) for b in bars}
    opens = sorted(close_by_open)
    n_ok = 0
    for e in events:
        ts = e["ts"]
        cur = _kline_close_at_or_before(close_by_open, opens, ts)
        p30 = _kline_close_at_or_before(close_by_open, opens, ts - 30 * 60_000)
        p60 = _kline_close_at_or_before(close_by_open, opens, ts - 60 * 60_000)
        if cur is not None and p30 is not None:
            e["vol30"] = abs(cur - p30) / p30 * 100.0
        if cur is not None and p60 is not None:
            e["ret1h"] = (cur - p60) / p60 * 100.0
        if e.get("vol30") is not None:
            n_ok += 1
    print(f"[k1m] vol30/ret1h 特征补齐 {n_ok}/{len(events)} 笔")


def stage_scan() -> dict:
    with open(SNAPSHOT, encoding="utf-8") as f:
        snap = json.load(f)
    with open(HYP_FILE, encoding="utf-8") as f:
        hyp_cfg = json.load(f)
    hyp_hash = hashlib.sha256(open(HYP_FILE, "rb").read()).hexdigest()[:12]

    events, v2_official = ev.build_events(snap)
    if not events:
        raise SystemExit("事件集为空，先跑 local_qm_v2_evaluation.py --stage data")
    enrich_kline_features(events)
    cut = events[0]["window_start"] + int(0.7 * (events[-1]["window_start"] - events[0]["window_start"]))
    disc = [e for e in events if e["window_start"] < cut]
    hold = [e for e in events if e["window_start"] >= cut]
    p0_disc = sum(1 for e in disc if e["win"]) / len(disc) if disc else None
    p0_hold = sum(1 for e in hold if e["win"]) / len(hold) if hold else None
    print(f"[scan] 事件 {len(events)}（发现 {len(disc)} / 留出 {len(hold)}），"
          f"基准胜率 发现={p0_disc:.3f} 留出={p0_hold and round(p0_hold, 3)}")

    # ---- 事件域假设 ----
    results = []
    for h in hyp_cfg["event_hypotheses"]:
        d_hits, d_miss = [], 0
        for e in disc:
            r = _apply_filters(e, h["filters"])
            if r is None:
                d_miss += 1
            elif r:
                d_hits.append(e)
        n, k = len(d_hits), sum(1 for e in d_hits if e["win"])
        row = {
            "id": h["id"], "family": h["family"], "desc": h["desc"],
            "discovery": {"n": n, "wins": k, "wr": k / n if n else None,
                          "wilson": list(ev.wilson(k, n)) if n else None,
                          "p_value": prop_z_pvalue(k, n, p0_disc) if (n and p0_disc) else None,
                          "features_missing": d_miss,
                          "avg_ev_econ": (sum(e["ev_econ"] for e in d_hits) / n) if n else None,
                          "ev_econ_ci95": None},
        }
        if n >= MIN_STAT_N:
            row["discovery"]["ev_econ_ci95"] = list(ev.bootstrap_ci([e["ev_econ"] for e in d_hits]))
        results.append(row)

    # ---- BH-FDR（只对发现集有统计量的假设）----
    screened_idx = [i for i, r in enumerate(results) if r["discovery"]["n"] >= MIN_STAT_N]
    pvals = [results[i]["discovery"]["p_value"] for i in screened_idx]
    fdr_flags = bh_fdr(pvals) if pvals else []
    for i, ok in zip(screened_idx, fdr_flags):
        results[i]["fdr_pass"] = ok
    for i, r in enumerate(results):
        if i not in screened_idx:
            r["fdr_pass"] = None  # INSUFFICIENT

    # ---- 经济闸（发现集）----
    for r in results:
        ci = r["discovery"]["ev_econ_ci95"]
        r["econ_gate"] = (
            "PASS" if (ci and ci[0] > 0)
            else ("FAIL" if ci else "INSUFFICIENT"))

    # ---- 裁决 + 留出集终验（只终验一次，仅对存活者）----
    for r in results:
        d = r["discovery"]
        if d["n"] < MIN_STAT_N:
            r["verdict"] = "INSUFFICIENT"
            continue
        if not r["fdr_pass"]:
            r["verdict"] = "OBSERVE(FDR未过降级)" if d["p_value"] is not None and d["p_value"] < 0.05 else "REJECT"
            continue
        if r["econ_gate"] != "PASS":
            r["verdict"] = "OBSERVE(经济功效不足)"
            continue
        # 存活者 → 留出集终验（一次性）
        h_hits = []
        for e in hold:
            if _apply_filters(e, r_to_filters(r, hyp_cfg)) is True:
                h_hits.append(e)
        hn, hk = len(h_hits), sum(1 for e in h_hits if e["win"])
        r["holdout"] = {"n": hn, "wins": hk, "wr": hk / hn if hn else None,
                        "wilson": list(ev.wilson(hk, hn)) if hn else None,
                        "avg_ev_econ": (sum(e["ev_econ"] for e in h_hits) / hn) if hn else None,
                        "ev_econ_ci95": None}
        if hn >= MIN_STAT_N:
            hci = ev.bootstrap_ci([e["ev_econ"] for e in h_hits])
            r["holdout"]["ev_econ_ci95"] = list(hci) if hci else None
            r["verdict"] = "ACTIVE" if (hci and hci[0] > 0) else "REJECT(留出终验失败)"
        else:
            r["verdict"] = "OBSERVE(留出样本不足)"

    # ---- K 线代理族（单调性历史一致性，仅方向证据）----
    kproxy = kline_proxy(hyp_cfg["kline_proxy_hypotheses"]["hypotheses"])

    out = {
        "split": {"discovery_n": len(disc), "holdout_n": len(hold),
                  "base_wr_discovery": p0_disc, "base_wr_holdout": p0_hold},
        "hypothesis_file_sha256": hyp_hash,
        "event_results": results,
        "kline_proxy": kproxy,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(CAND_OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[scan] 候选表已存 {CAND_OUT}")
    for r in results:
        d = r["discovery"]
        wr = f"{d['wr']:.1%}" if d["wr"] is not None else "-"
        print(f"  {r['id']:<16} n={d['n']:<3} wr={wr:<7} p={d['p_value'] and round(d['p_value'],3)}"
              f" FDR={r['fdr_pass']} econ={r['econ_gate']:<12} → {r['verdict']}")
    print("\nK线代理（方向证据）:")
    for kp in kproxy:
        print(f"  {kp['id']}: n={kp['n']} wr={kp['wr'] and round(kp['wr'], 3)}")
    return out


def r_to_filters(r: dict, hyp_cfg: dict) -> list[dict]:
    for h in hyp_cfg["event_hypotheses"]:
        if h["id"] == r["id"]:
            return h["filters"]
    raise KeyError(r["id"])


def kline_proxy(hypotheses: list[dict]) -> list[dict]:
    """1m K 线 → 5m 窗代理：第 2 根 1m 收盘为触发点价，结局=该 5m 窗收盘<开盘。"""
    if not os.path.exists(K1M):
        print(f"[scan] K 线代理跳过：{K1M} 不存在")
        return []
    with open(K1M, encoding="utf-8") as f:
        bars = json.load(f)
    # 按 5m 窗分组（需完整 5 根 1m）
    groups: dict[int, list] = {}
    for b in bars:
        groups.setdefault(int(b[0]) // 300_000 * 300_000, []).append(b)
    rows = []
    for ws in sorted(groups):
        g = sorted(groups[ws], key=lambda x: int(x[0]))
        if len(g) < 5:
            continue
        open_p = float(g[0][1])
        trigger_p = float(g[1][4])   # 第 2 根 1m 收盘（≈100~120s 决策点）
        close_p = float(g[4][4])
        rows.append((ws, open_p, trigger_p, close_p))
    out = []
    for h in hypotheses:
        thr = h["threshold_pct"]
        hits = [r for r in rows if r[1] > 0 and (r[2] - r[1]) / r[1] * 100.0 <= thr]
        n = len(hits)
        k = sum(1 for r in hits if r[3] < r[1])  # 收盘<开盘 → DOWN 赢
        lo, hi = ev.wilson(k, n) if n else (None, None)
        out.append({"id": h["id"], "threshold_pct": thr, "n": n, "wins": k,
                    "wr": k / n if n else None, "wilson": [lo, hi]})
    return out


# ----------------------------------------------------------------------
# stage report：合并评估 + 候选 → report.json / report.md
# ----------------------------------------------------------------------

def stage_report() -> None:
    with open(EVAL_JSON, encoding="utf-8") as f:
        evaluation = json.load(f)
    with open(CAND_OUT, encoding="utf-8") as f:
        scan = json.load(f)
    with open(SNAPSHOT, encoding="utf-8") as f:
        snap = json.load(f)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reproducibility": {
            "snapshot_fingerprint": snap.get("fingerprint"),
            "hypothesis_file_sha256": scan["hypothesis_file_sha256"],
            "constitution": {"min_hits": MIN_STAT_N, "fdr_q": FDR_Q,
                             "econ_gate": "fee 2% + premium 0.01, bootstrap95 CI lower>0, n>=10",
                             "split": scan["split"]},
        },
        "evaluation": evaluation,
        "candidates": scan["event_results"],
        "kline_proxy": scan["kline_proxy"],
        "verdict_v2": evaluation["verdict"],
    }

    # SHADOW_BENCH 回填建议（官方集全历史口径，经济闸注记随裁决动态）
    off = evaluation["M1_time_split_and_baseline"]["official_all"]
    if off["n"]:
        ev_shadow = off["avg_ev_shadow"]
        gate_v = evaluation["M5_economic_gate"]["v2_official(全历史)"]["gate"]
        gate_note = ("经济闸通过，可评估 promote" if gate_v.startswith("PASS")
                     else "费2%+溢价0.01经济闸未过，维持影子")
        report["shadow_bench_suggestion"] = (
            f'"quote_momentum_v2": ({off["wr"]:.3f}, {ev_shadow:.3f}, '
            f'"顺势v2: v1+触发时已跌≥0.10%（影子行表全历史 n={off["n"]} 胜率{off["wr"]:.1%}'
            f' Wilson[{off["wilson"][0]:.1%},{off["wilson"][1]:.1%}]；{gate_note}）")'
        )
    actives = [r for r in scan["event_results"] if r["verdict"] == "ACTIVE"]
    report["landing_suggestions"] = _landing_drafts(actives) if actives else [
        "无 ACTIVE 候选：不注册新通道。v2 维持影子攒样本，待官方集费后经济闸转正确认。",
        "工程建议：为 /api/misalignment/signals 增加 since 分页参数（当前硬上限 200；本轮已经 Actions DB dump 取回全历史，加分页后日常可免临时通道）。",
    ]

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    md = _render_md(report)
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[report] 最终报告已存 {REPORT_JSON} 与 {REPORT_MD}")


def _landing_drafts(actives: list[dict]) -> list[str]:
    drafts = []
    for r in actives:
        drafts.append(
            f"候选 {r['id']}（{r['desc']}）通过全部闸门。落地草案（追加式，只加不改）：\n"
            f"  1) quote_edge_detector.py：新增 guard 常量条目（仿 V2_PRICE_GUARDS/LN_DD_GUARDS 注册）；\n"
            f"  2) main.py SHADOW_BENCH +1 行（n/胜率/CI/经济闸注记）；\n"
            f"  3) live_channels.py +1 ChannelSpec（默认 OFF，影子先行攒样本）；\n"
            f"  4) tests/test_multi_live_trader.py 仿 L346-353 补门禁边界断言。\n"
            f"  合入时机由用户决策；影子攒满功效样本（≥30 笔且经济闸持续通过）再谈实盘。")
    return drafts


def _render_md(rep: dict) -> str:
    e = rep["evaluation"]
    m1 = e["M1_time_split_and_baseline"]
    off = m1["official_all"]
    m2, m3, m4, m5 = (e[k] for k in (
        "M2_paired_counterfactual", "M3_threshold_monotonicity",
        "M4_time_stability", "M5_economic_gate"))
    L = []
    L.append("# quote_momentum_v2 科学评估与进化探索报告")
    L.append(f"\n生成时间：{rep['generated_at']}（UTC）")
    L.append("\n## 综合裁决")
    L.append(f"\n> **{rep['verdict_v2']}**")
    L.append("\n## M1 基线对账与样本")
    L.append(f"- v2 官方历史集（影子行表全量）：**n={off['n']}，胜率 {off['wr']:.1%}**，"
             f"Wilson [{off['wilson'][0]:.1%}, {off['wilson'][1]:.1%}]")
    L.append(f"- 与 2026-08-26 基线（n=24，75.0%）对账：{m1.get('baseline_check')}")
    es = e["event_summary"]
    L.append(f"- v1 重放段（{es['v1_replay_range_note']}；区间 {es['range'][0]} ~ {es['range'][1]}）："
             f"n={es['v1_replay_n']}，带 ex-ante 特征，仅用于配对评估与假设发现")
    L.append("\n## M2 配对反事实（门禁是否创造价值）")
    L.append(f"- 过门禁段：n={m2['kept']['n']}，胜率 {m2['kept']['wr']:.1%}，费后 EV {m2['kept']['avg_ev_econ']:+.3f}")
    L.append(f"- 被剔除段：n={m2['dropped']['n']}，胜率 {m2['dropped']['wr']:.1%}，费后 EV "
             f"{m2['dropped']['avg_ev_econ']:+.3f}，CI {[round(x,3) for x in m2['dropped']['ev_econ_ci95']] if m2['dropped']['ev_econ_ci95'] else '-'}")
    L.append(f"- 判定：**{m2['verdict']}**")
    L.append("\n## M3 阈值单调性扫描（v1 重放段）")
    L.append(f"- 胜率曲线（阈值 { [s['threshold'] for s in m3['sweep']] }）：{m3['wr_curve']}")
    mono = m3["monotonic_non_decreasing(±3pp)"]
    mono_note = ("单调不减，门禁阈值方向证据成立" if mono
                 else "非单调，阈值选点样本内不可靠（仅方向证据）" if mono is False
                 else "样本不足")
    L.append(f"- 单调不减(±3pp)：{mono}；最优点 {m3['best_point']} —— **{mono_note}**")
    L.append("\n## M4 时间稳定性（官方集）")
    for b in m4["weekly"]:
        wr = f"{b['v2_wr']:.1%}" if b["v2_wr"] is not None else "-"
        L.append(f"- {b['week']}：n={b['v2_n']}，胜率 {wr}")
    pre, post = m4["regime_pre_pump(before 08-19)"], m4["regime_post_pump(after 08-19)"]
    L.append(f"- 大涨前(08-19 前)：n={pre['n']}，胜率 {pre['wr'] and format(pre['wr'], '.1%')}"
             f" / 大涨后：n={post['n']}，胜率 {post['wr'] and format(post['wr'], '.1%')}")
    L.append("\n## M5 经济闸（费 2% + 溢价 0.01）")
    for k in ("v1_replay", "v2_replay(∩门禁)", "v2_official(全历史)"):
        blk = m5[k]
        ci = [round(x, 3) for x in blk["ev_econ_ci95"]] if blk["ev_econ_ci95"] else None
        L.append(f"- {k}：n={blk['n']}，费后 EV "
                 f"{blk['avg_ev_econ']:+.3f}，CI {ci} → **{blk['gate']}**")
    L.append(f"- 盈亏平衡胜率区间（报价带 [0.69,0.75)）：{m5['breakeven_wr_range']}")
    L.append("\n## 进化候选（预注册假设扫描）")
    L.append(f"- 发现集基准胜率：{rep['reproducibility']['constitution']['split']['base_wr_discovery']:.3f}"
             f"；留出集基准：{rep['reproducibility']['constitution']['split']['base_wr_holdout']:.3f}")
    L.append("\n| 假设 | 族 | 发现 n | 胜率 | p | FDR | 经济闸 | 裁决 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in rep["candidates"]:
        d = r["discovery"]
        wr = f"{d['wr']:.1%}" if d["wr"] is not None else "-"
        p = "-" if d["p_value"] is None else f"{d['p_value']:.3f}"
        L.append(f"| {r['id']} | {r['family']} | {d['n']} | {wr} | {p} | "
                 f"{r['fdr_pass']} | {r['econ_gate']} | **{r['verdict']}** |")
    if rep["kline_proxy"]:
        L.append("\n### K 线代理族（仅方向证据，~62 天 1m 粒度）")
        L.append("| 阈值 | 触发 n | 胜率 | Wilson |")
        L.append("|---|---|---|---|")
        for kp in rep["kline_proxy"]:
            wci = f"[{kp['wilson'][0]:.1%},{kp['wilson'][1]:.1%}]" if kp["n"] else "-"
            L.append(f"| {kp['threshold_pct']}% | {kp['n']} | "
                     f"{kp['wr'] and format(kp['wr'], '.1%')} | {wci} |")
    L.append("\n## SHADOW_BENCH 回填建议")
    L.append(f"\n```\n{rep.get('shadow_bench_suggestion', '（无）')}\n```")
    L.append("\n## 落地建议")
    for s in rep["landing_suggestions"]:
        L.append(f"- {s}")
    L.append("\n## 方法与局限")
    L.append("- A 段（影子行∩门禁）与门禁归因同源，结论只认 M2 配对反事实 / M4 regime 注记 / M5 全历史经济闸。")
    L.append("- v1 影子行 API 硬上限 200 已经 GitHub Actions SSH 通道 DB 全量 dump 绕过（行级全历史）；"
             "早期窗口若缺 curve_btc_price，对应门禁特征记缺失（不计入门禁段）。")
    L.append("- 官方集样本仍有限，且绝大多数处于 08-19 大涨后 regime，跨 regime 稳定性以 M4 双段对比为准。")
    L.append("- vol30/ret1h 由事件区间 1m K 线派生（≤触发时点最后一根已完结 1m 收盘，严格 ex-ante）；"
             "情绪曲线仅覆盖本窗 5 分钟，无法支撑 30min/1h 回看。")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["scan", "report", "all"], default="all")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if args.stage in ("scan", "all"):
        stage_scan()
    if args.stage in ("report", "all"):
        stage_report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
