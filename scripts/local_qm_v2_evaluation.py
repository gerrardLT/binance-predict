#!/usr/bin/env python3
"""quote_momentum_v2 科学评估（2026-08-28）：五方法交叉裁决"门禁优化是否合理"。

评估对象：quote_momentum_v2 = quote_momentum_v1（t∈[90,120)s DOWN 报价首进
[0.69,0.75) 押本窗 DOWN）∩ 触发时点 BTC 较窗开盘已跌≥0.10%（chg≤−0.10%）。
v2 是 v1 触发集的纯子集 → 逐笔配对比较天然成立。

五方法：
  M1 IS/OOS 时间切分重算 + 与既有基线（shadow_v2v3_real_backtest_result.json，
     2026-08-26 口径 n=24/wr=0.75）对账；
  M2 配对反事实：过门禁段 vs 被剔除段——门禁是否在销毁/创造价值；
  M3 阈值单调性扫描：−0.05~−0.30 胜率曲线，孤立尖峰=过拟合红旗；
  M4 时间稳定性：按周分桶胜率 + PUMP_TS（08-19）前后 regime；
  M5 经济闸：费 2%+溢价 0.01 逐注 bootstrap 95% CI 下界（宪法判据）。

纪律：命中<10 不做统计判定；一切特征严格 ex-ante；A 段与门禁归因同源
（IS 自循环），合理性结论只认 M2/M3/M4/M5 与留出段，A 段数字仅呈现。

数据事实（2026-08-28 核实）：
  - /api/misalignment/signals 硬上限 200 条（无分页）→ qm_v1 行仅最近 ~6 天，
    只能作"带完整 ex-ante 特征的配对评估段"；
  - qm_v2 影子行表总量 <200（未触底）→ 是 v2 的完整历史真值集，作为官方样本与基线对账；
  - 本地情绪窗 dump 曲线价格字段覆盖率过低（07-13~07-30 段 curve_btc_price 仅
    20/4876 窗），无法支撑全曲线重放 → 事件源 = 线上影子行，环境源 = 生产窗口；
  - output/sentiment_windows_online_fixed.json 无价格曲线，同样不可重放。

用法：python -X utf8 scripts/local_qm_v2_evaluation.py [--stage data|evaluate|report|all]
      [--refresh]（--refresh 忽略快照缓存重新拉生产数据）
产出：output/qm_v2_eval/snapshot.json（数据快照+指纹）
      output/qm_v2_eval/evaluation.json（五方法结构化结果，供报告脚本汇总）
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import urllib.request
from datetime import datetime, timedelta, timezone

BASE = "http://165.154.147.155:8082"
OUT_DIR = "output/qm_v2_eval"
SNAPSHOT = os.path.join(OUT_DIR, "snapshot.json")
EVAL_OUT = os.path.join(OUT_DIR, "evaluation.json")
BASELINE = "output/shadow_v2v3_real_backtest_result.json"  # 2026-08-26 对账靶
DB_DUMP = os.path.join(OUT_DIR, "qm_signals_dump.csv")  # Actions SSH 通道 DB 行级 dump（绕 API 200 上限）
PUMP_TS_MS = int(datetime(2026, 8, 19, tzinfo=timezone.utc).timestamp() * 1000)

FEE = 0.02        # 影子口径：费 2% 无溢价（EV=0.98/q−1）
PREMIUM = 0.01    # 经济闸口径：费 2% + 溢价 0.01（宪法）
MIN_FIRES = 10    # 经济闸最小注数（宪法）
MIN_STAT_N = 10   # 命中 <10 不做统计判定
BOOT_N = 10_000
BOOT_SEED = 20260828
V2_GATE = -0.10   # 冻结门禁：chg ≤ −0.10%（min_drop，含边界）


# ----------------------------------------------------------------------
# 纯函数（拷贝自 scripts/local_shadow_v2v3_real_backtest.py L69-174，
# 冻结口径同源；原脚本不改。新增 bet_pnl/_bootstrap_ci 口径对齐
# src/binance_predict/services/ev_gate.py L123-133。）
# ----------------------------------------------------------------------

def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z = 1.96
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def btc_at_or_before(curve, ts: int):
    """curve_btc_price 中 ≤ts 的最晚采样点（严格 ex-ante）。"""
    if not curve:
        return None
    best = None
    for m in sorted(curve, key=lambda x: x["t"]):
        if m["t"] <= ts:
            best = m
        else:
            break
    return best["v"] if best else None


def window_open_btc(w) -> float | None:
    """窗开盘 BTC = curve_btc_price 首个采样点（检测器同口径）。"""
    curve = w.get("curve_btc_price")
    if not curve:
        return None
    return sorted(curve, key=lambda x: x["t"])[0]["v"]


def build_day_high(ordered_windows) -> list[tuple[int, float]]:
    """按 end_time 升序的 (ts, running_high) 前缀表（当日 UTC 日高，无未来函数）。"""
    prefix: list[tuple[int, float]] = []
    day = None
    hi = None
    for w in ordered_windows:
        curve = w.get("curve_btc_price") or []
        for m in sorted(curve, key=lambda x: x["t"]):
            d = datetime.fromtimestamp(m["t"] / 1000, timezone.utc).date()
            if d != day:
                day, hi = d, m["v"]
            else:
                hi = max(hi, m["v"])
            prefix.append((m["t"], hi))
    return prefix


def day_high_before(prefix: list[tuple[int, float]], ts: int) -> float | None:
    import bisect
    if not prefix:
        return None
    i = bisect.bisect_right(prefix, (ts, float("inf"))) - 1
    return prefix[i][1] if i >= 0 else None


def bet_pnl(win: bool, price: float, fee: float = FEE, premium: float = PREMIUM) -> float:
    """逐注盈亏（宪法经济闸口径）：赢 → (1-fee)/(price+premium) − 1；输 → −1。"""
    if not win:
        return -1.0
    p = min(max(price + premium, 0.01), 0.99)
    return (1.0 - fee) / p - 1.0


def bootstrap_ci(pnls: list[float], n_boot: int = BOOT_N, seed: int = BOOT_SEED) -> tuple[float, float] | None:
    """EV 的 bootstrap 95% CI（对注单重抽，固定 seed 确定性）。注数<MIN_FIRES → None。"""
    import random
    if len(pnls) < MIN_FIRES:
        return None
    rng = random.Random(seed)
    means = []
    n = len(pnls)
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += pnls[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def agg_block(rows: list[dict]) -> dict:
    """一组事件的聚合统计（胜率/CI/逐注 EV/双口径）。"""
    n = len(rows)
    k = sum(1 for r in rows if r["win"])
    pnls_shadow = [r["ev_shadow"] for r in rows]
    pnls_econ = [r["ev_econ"] for r in rows]
    lo, hi = wilson(k, n) if n else (None, None)
    ci = bootstrap_ci(pnls_econ) if n >= MIN_FIRES else None
    return {
        "n": n, "wins": k,
        "wr": k / n if n else None,
        "wilson": [lo, hi],
        "avg_ev_shadow": sum(pnls_shadow) / n if n else None,
        "avg_ev_econ": sum(pnls_econ) / n if n else None,
        "ev_econ_ci95": list(ci) if ci else None,
    }


# ----------------------------------------------------------------------
# stage data：生产 API 拉取 → 快照 + 指纹
# ----------------------------------------------------------------------

def _load_token() -> str:
    """从本地 .env 读 LOGIN_PASSWORD（生产 Bearer token）。"""
    for path in (".env",):
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("LOGIN_PASSWORD="):
                    v = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if v:
                        return v
    raise SystemExit("未找到 .env 中的 LOGIN_PASSWORD（生产 API Bearer token）")


def _get(path: str, token: str, timeout: int = 300):
    req = urllib.request.Request(BASE + path, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _sha256_file(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def stage_data(refresh: bool) -> dict:
    if os.path.exists(SNAPSHOT) and not refresh:
        with open(SNAPSHOT, encoding="utf-8") as f:
            snap = json.load(f)
        print(f"[data] 复用快照 {SNAPSHOT}（--refresh 可强制重拉）")
        return snap

    token = _load_token()

    # 1) 影子行（硬上限 200/版本，先探明触底情况）
    sig_probe: dict[str, dict] = {}
    qm_v1, qm_v2 = [], []
    for version in ("quote_momentum_v1", "quote_momentum_v2"):
        resp = _get(f"/api/misalignment/signals?limit=200&version={version}", token)
        rows = resp["signals"]
        if version == "quote_momentum_v1":
            qm_v1 = rows
        else:
            qm_v2 = rows
        oldest = min((int(r["window_start"]) for r in rows), default=None)
        sig_probe[version] = {
            "fetched": len(rows),
            "api_hard_limit": 200,
            "truncated_suspect": len(rows) >= 200,
            "oldest_window_start": oldest,
            "oldest_date": (datetime.fromtimestamp(oldest / 1000, timezone.utc).strftime("%Y-%m-%d")
                            if oldest else None),
            "stats_all_time": resp.get("stats"),  # 累计统计不受 limit 截断
        }
        print(f"[data] 影子行 {version}: {len(rows)} 条"
              f"{'（疑似触底 200 上限，样本非全历史！）' if len(rows) >= 200 else ''}")

    # 2) 生产全量窗口（/api/sentiment/windows 无上限截断）
    wins = _get("/api/sentiment/windows?limit=50000", token)
    wins = sorted(wins, key=lambda w: int(w["start_time"]))
    n_btc = sum(1 for w in wins if w.get("curve_btc_price"))
    print(f"[data] 生产窗口 {len(wins)} 个（带 BTC 曲线 {n_btc}）")

    # 3) 本地文件指纹（仅记录，不作事件源——曲线价格覆盖率不足，见 docstring）
    local_fp = {
        "sentiment_windows.json": _sha256_file("sentiment_windows.json"),
        "output/sentiment_windows_online_fixed.json": _sha256_file("output/sentiment_windows_online_fixed.json"),
        "baseline_shadow_v2v3_result": _sha256_file(BASELINE),
    }

    snap = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fingerprint": {
            "n_windows": len(wins),
            "windows_range": [int(wins[0]["start_time"]), int(wins[-1]["start_time"])] if wins else None,
            "n_windows_with_btc_curve": n_btc,
            "shadow_probe": sig_probe,
            "local_files": local_fp,
        },
        "qm_v1_signals": qm_v1,
        "qm_v2_signals": qm_v2,
        "windows": wins,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(SNAPSHOT, "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False)
    print(f"[data] 快照已存 {SNAPSHOT}")
    return snap


# ----------------------------------------------------------------------
# DB dump 合并：/api/misalignment/signals 硬上限 200，v1 全历史 505 笔经
# GitHub Actions SSH 通道从生产 DB 导出（qm_signals_dump.csv），
# 字段与 API 行同构，类型转换后替换快照中的影子行（幂等）。
# ----------------------------------------------------------------------

def _csv_val_float(v: str) -> float | None:
    return float(v) if v not in ("", "\\N") else None


def _csv_val_int(v: str) -> int | None:
    return int(v) if v not in ("", "\\N") else None


def merge_db_dump(snap: dict) -> dict:
    """用 DB 行级 dump 替换快照中 qm_v1/qm_v2 影子行（全历史，绕 200 上限）。"""
    if not os.path.exists(DB_DUMP):
        raise SystemExit(f"缺 {DB_DUMP}（需先经 Actions 通道导出）")
    by_version: dict[str, list] = {}
    with open(DB_DUMP, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ver = r["version"]
            if not ver.startswith("quote_momentum"):
                continue
            by_version.setdefault(ver, []).append({
                "id": _csv_val_int(r["id"]),
                "version": ver,
                "window_start": _csv_val_int(r["window_start"]),
                "window_end": _csv_val_int(r["window_end"]),
                "end_pct": _csv_val_float(r["end_pct"]),
                "outcome_base": r["outcome_base"] or None,
                "direction": r["direction"] or None,
                "target_window_start": _csv_val_int(r["target_window_start"]),
                "entry_down_price": _csv_val_float(r["entry_down_price"]),
                "entry_up_price": _csv_val_float(r["entry_up_price"]),
                "entry_quote_ts": _csv_val_int(r["entry_quote_ts"]),
                "entry_quote_kind": r["entry_quote_kind"] or None,
                "settle_outcome": r["settle_outcome"] or None,
                "win": {"t": True, "f": False}.get(r["win"]),
                "ev_at_entry": _csv_val_float(r["ev_at_entry"]),
                "status": r["status"] or None,
            })
    n1 = len(by_version.get("quote_momentum_v1", []))
    n2 = len(by_version.get("quote_momentum_v2", []))
    if n1 <= len(snap.get("qm_v1_signals", [])):
        print(f"[merge] dump 行数（{n1}）未超过快照（{len(snap.get('qm_v1_signals', []))}），跳过合并")
        return snap
    api_n2 = len(snap.get("qm_v2_signals", []))
    snap["qm_v1_signals"] = sorted(by_version["quote_momentum_v1"],
                                   key=lambda r: -r["window_start"])
    snap["qm_v2_signals"] = sorted(by_version.get("quote_momentum_v2", snap.get("qm_v2_signals", [])),
                                   key=lambda r: -(r["window_start"] or 0))
    snap.setdefault("fingerprint", {})["db_dump_merged"] = {
        "source": "GitHub Actions SSH → 生产 PostgreSQL misalignment_signals 全表",
        "csv_sha256": _sha256_file(DB_DUMP),
        "qm_v1_rows": n1,
        "qm_v2_rows": n2,
        "v2_api_vs_db": "一致" if n2 == api_n2 else f"差异（API {api_n2} / DB {n2}）",
    }
    with open(SNAPSHOT, "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False)
    print(f"[merge] 已并入 DB dump：qm_v1 {n1} 行 / qm_v2 {n2} 行（快照已更新）")
    return snap


# ----------------------------------------------------------------------
# 事件构建：v1 影子行（已结算）∩ 窗口 → 逐笔特征（全 ex-ante）
# ----------------------------------------------------------------------

def _event_from_row(r: dict, wins_by_start: dict, prefix: list, with_chg: bool) -> dict | None:
    """影子行 → 事件。with_chg：v1 行重放门禁（算 chg 等特征）；
    False：v2 行表（官方历史集，无 chg 特征）。"""
    ws = int(r["window_start"])
    ts = r.get("entry_quote_ts")
    w = wins_by_start.get(ws)
    if ts is None or w is None:
        return None
    q = r.get("entry_down_price")
    if q is None or float(q) <= 0:
        return None
    q = float(q)
    win = bool(r["win"])
    chg = dd_day = None
    if with_chg:
        base = window_open_btc(w)
        cur = btc_at_or_before(w.get("curve_btc_price"), int(ts))
        chg = (cur - base) / base * 100.0 if (base and cur) else None
        dh = day_high_before(prefix, int(ts))
        dd_day = (cur - dh) / dh * 100.0 if (dh and cur) else None
    # 前窗 / 前窗涨跌 / 触发前波动（均 ≤触发时点，ex-ante）
    prev = wins_by_start.get(ws - 300_000)
    prev_outcome = prev.get("outcome") if prev else None
    prev_chg = None
    if prev:
        pb, pc = window_open_btc(prev), btc_at_or_before(
            prev.get("curve_btc_price"), int(prev.get("end_time") or 0))
        if pb and pc:
            prev_chg = (pc - pb) / pb * 100.0
    vol30, ret1h = None, None
    curve = w.get("curve_btc_price") or []
    cur2 = btc_at_or_before(curve, int(ts))
    if cur2 and curve:
        p30 = btc_at_or_before(curve, int(ts) - 30 * 60_000)
        p60 = btc_at_or_before(curve, int(ts) - 60 * 60_000)
        # 跨窗回看：曲线只覆盖本窗，回看点缺失则用前窗曲线兜底
        if p30 is None and prev:
            p30 = btc_at_or_before(prev.get("curve_btc_price"), int(ts) - 30 * 60_000)
        if p30:
            vol30 = abs(cur2 - p30) / p30 * 100.0
        if p60 is None and prev:
            prev2 = wins_by_start.get(ws - 600_000)
            p60 = btc_at_or_before((prev2 or prev).get("curve_btc_price"), int(ts) - 60 * 60_000)
        if p60:
            ret1h = (cur2 - p60) / p60 * 100.0
    bj_h = (datetime.fromtimestamp(ws / 1000, timezone.utc) + timedelta(hours=8)).hour
    return {
        "window_start": ws,
        "ts": int(ts),
        "q": q,
        "win": win,
        "chg": chg,
        "dd_day": dd_day,
        "prev_outcome": prev_outcome,
        "prev_chg": prev_chg,
        "vol30": vol30,
        "ret1h": ret1h,
        "bj_h": bj_h,
        "t_in": (int(ts) - ws) / 1000.0,
        "ev_shadow": (1 - FEE) / q - 1.0 if win else -1.0,
        "ev_econ": bet_pnl(win, q),
        "gate_pass": (chg is not None and chg <= V2_GATE) if with_chg else True,
        "gate_missing": (chg is None) if with_chg else False,
    }


def build_events(snap: dict) -> tuple[list[dict], list[dict]]:
    """返回 (v1 重放事件段[带特征], v2 官方历史集[v2 影子行表全量])。"""
    wins_by_start = {int(w["start_time"]): w for w in snap["windows"]}
    ordered = [wins_by_start[k] for k in sorted(wins_by_start)]
    prefix = build_day_high(ordered)

    events: list[dict] = []
    for r in snap["qm_v1_signals"]:
        if r.get("status") != "SETTLED" or r.get("win") is None:
            continue
        e = _event_from_row(r, wins_by_start, prefix, with_chg=True)
        if e:
            events.append(e)

    v2_official: list[dict] = []
    for r in snap["qm_v2_signals"]:
        if r.get("status") != "SETTLED" or r.get("win") is None:
            continue
        e = _event_from_row(r, wins_by_start, prefix, with_chg=False)
        if e:
            v2_official.append(e)

    events.sort(key=lambda e: e["window_start"])
    v2_official.sort(key=lambda e: e["window_start"])
    return events, v2_official


def _fmt_ts(ms: int | None) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d") if ms else "-"


# ----------------------------------------------------------------------
# stage evaluate：M1~M5
# ----------------------------------------------------------------------

def stage_evaluate(snap: dict) -> dict:
    events, v2_official = build_events(snap)
    n_missing = sum(1 for e in events if e["gate_missing"])
    print(f"\n[eval] v1 重放段 {len(events)} 笔（门禁特征缺失 {n_missing}，缺失率 "
          f"{n_missing / len(events):.1%}；仅作配对评估）" if events else "\n[eval] 无事件")
    print(f"[eval] v2 官方历史集 {len(v2_official)} 笔（影子行表全量，未触底截断）")
    if not events and not v2_official:
        raise SystemExit("事件集为空，无法评估（检查影子行/窗口数据）")
    if not events:
        raise SystemExit("v1 重放段为空，无法做配对评估")
    # 对账：重放事件应为官方集的子集（覆盖范围内）
    off_set = {e["window_start"] for e in v2_official}
    rep_in_off = sum(1 for e in events if e["gate_pass"] and e["window_start"] in off_set)
    print(f"[eval] 重放∩官方集: {rep_in_off} 笔（重放段覆盖范围内应基本重合）")

    v2_events = [e for e in events if e["gate_pass"]]
    dropped = [e for e in events if not e["gate_pass"] and not e["gate_missing"]]

    out: dict = {
        "event_summary": {
            "v1_replay_n": len(events),
            "v1_replay_range_note": ("v1 影子信号行级全历史（DB dump，绕过 API 上限 200）"
                                       if snap.get("fingerprint", {}).get("db_dump_merged")
                                       else "v1 影子信号（API 最近 200 行）"),
            "gate_missing_n": n_missing,
            "v2_official_n": len(v2_official),
            "v2_official_note": "v2 影子行表全量（总量<200 未触底），官方历史样本",
            "v1_replay_v2_n": len(v2_events),
            "dropped_n": len(dropped),
            "range": [_fmt_ts(events[0]["window_start"]), _fmt_ts(events[-1]["window_start"])],
            "official_range": ([_fmt_ts(v2_official[0]["window_start"]),
                                 _fmt_ts(v2_official[-1]["window_start"])] if v2_official else None),
        },
    }

    # ---- M1：官方集全量 + 重放段时间切分 + 基线对账 ----
    cut = events[0]["window_start"] + int(0.7 * (events[-1]["window_start"] - events[0]["window_start"]))
    disc = [e for e in events if e["window_start"] < cut]
    hold = [e for e in events if e["window_start"] >= cut]
    m1 = {
        "official_all": agg_block(v2_official),
        "replay_split_ts": _fmt_ts(cut),
        "replay_discovery": agg_block([e for e in disc if e["gate_pass"]]),
        "replay_holdout": agg_block([e for e in hold if e["gate_pass"]]),
        "replay_all_v2": agg_block(v2_events),
        "replay_all_v1": agg_block(events),
    }
    # 对账：2026-08-26 基线 n=24/wr=0.75 —— 官方集是完整历史，直接核胜率稳定性；
    # 注意基线样本窗为 07-13~08-26，官方集后续仍在增长，样本数差异属预期。
    if os.path.exists(BASELINE):
        with open(BASELINE, encoding="utf-8") as f:
            base = json.load(f).get("quote_momentum_v2", {})
        bl = base.get("online", {})
        m1["baseline_2026_08_26"] = {"n": bl.get("n"), "wr": bl.get("wr"), "wilson": bl.get("wilson")}
        off = m1["official_all"]
        m1["baseline_check"] = (
            "OK（官方集样本≥基线且胜率差≤15pp，胜率稳定）"
            if (off["n"] >= (bl.get("n") or 0)
                and off["wr"] is not None
                and abs(off["wr"] - (bl.get("wr") or 0)) <= 0.15)
            else "DRIFT（口径或数据漂移，需人工核对）"
        )
    out["M1_time_split_and_baseline"] = m1

    # ---- M2：配对反事实（过门禁段 vs 被剔除段）----
    m2 = {"kept": agg_block(v2_events), "dropped": agg_block(dropped)}
    d = m2["dropped"]
    if d["n"] >= MIN_STAT_N:
        # 剔除段若自身费后期望显著为负 → 门禁有价值；否则门禁只是噪声切分/销毁价值
        ci = d["ev_econ_ci95"]
        m2["verdict"] = (
            "gate_creates_value" if (ci and ci[1] < 0)
            else "gate_destroys_or_neutral" if (ci and ci[0] > -0.15 and (d["avg_ev_econ"] or 0) >= -0.05)
            else "inconclusive"
        )
    else:
        m2["verdict"] = "insufficient_n(<10)"
    out["M2_paired_counterfactual"] = m2

    # ---- M3：阈值单调性扫描 ----
    sweep = []
    for thr in (-0.05, -0.08, -0.10, -0.12, -0.15, -0.20, -0.30):
        seg = [e for e in events if e["chg"] is not None and e["chg"] <= thr]
        blk = agg_block(seg)
        blk["threshold"] = thr
        sweep.append(blk)
    wrs = [s["wr"] for s in sweep if s["wr"] is not None]
    monotonic = all(wrs[i] <= wrs[i + 1] + 0.03 for i in range(len(wrs) - 1)) if len(wrs) >= 3 else None
    peak = max(sweep, key=lambda s: (s["wr"] if s["wr"] is not None else -1, s["n"]))
    out["M3_threshold_monotonicity"] = {
        "sweep": sweep,
        "wr_curve": [round(s["wr"], 4) if s["wr"] is not None else None for s in sweep],
        "monotonic_non_decreasing(±3pp)": monotonic,
        "best_point": {"threshold": peak["threshold"], "wr": peak["wr"], "n": peak["n"]},
        "is_isolated_spike": (not monotonic) if monotonic is not None else None,
    }

    # ---- M4：时间稳定性（官方集按周分桶 + PUMP regime）----
    weekly: dict[str, list] = {}
    for e in v2_official:
        wk = datetime.fromtimestamp(e["window_start"] / 1000, timezone.utc).strftime("%G-W%V")
        weekly.setdefault(wk, []).append(e)
    buckets = []
    for wk in sorted(weekly):
        rows = weekly[wk]
        buckets.append({
            "week": wk, "v2_n": len(rows),
            "v2_wins": sum(1 for r in rows if r["win"]),
            "v2_wr": sum(1 for r in rows if r["win"]) / len(rows) if rows else None,
        })
    pre = [e for e in v2_official if e["window_start"] < PUMP_TS_MS]
    post = [e for e in v2_official if e["window_start"] >= PUMP_TS_MS]
    out["M4_time_stability"] = {
        "weekly": buckets,
        "regime_pre_pump(before 08-19)": agg_block(pre),
        "regime_post_pump(after 08-19)": agg_block(post),
    }

    # ---- M5：经济闸（宪法判据：费2%+溢价0.01 逐注 bootstrap CI 下界>0 且 ≥10 注）----
    def econ_verdict(blk: dict) -> str:
        if blk["n"] < MIN_FIRES:
            return "insufficient_n(<10)"
        ci = blk["ev_econ_ci95"]
        if ci is None:
            return "insufficient"
        return "PASS(EV CI lower>0)" if ci[0] > 0 else "FAIL(EV CI lower<=0)"

    m5 = {
        "cost_model": "fee 2% + premium 0.01（宪法经济闸）；并列影子口径费 2% 无溢价",
        "v1_replay": {**agg_block(events), "gate": econ_verdict(agg_block(events))},
        "v2_replay(∩门禁)": {**agg_block(v2_events), "gate": econ_verdict(agg_block(v2_events))},
        "v2_official(全历史)": {**agg_block(v2_official), "gate": econ_verdict(agg_block(v2_official))},
        "breakeven_wr_range": [round(0.69 / (1 - FEE), 4), round(0.75 / (1 - FEE), 4)],
    }
    out["M5_economic_gate"] = m5

    # ---- 综合裁决（三值）：以官方全历史集经济闸为准，配对反事实佐证 ----
    m2v = m2["verdict"]
    m5v = m5["v2_official(全历史)"]["gate"]
    if m5v.startswith("PASS") and m2v == "gate_creates_value":
        verdict = "合理（官方全历史集经济闸通过且配对反事实确认门禁创造价值）"
    elif m5v.startswith("PASS"):
        verdict = "条件性合理（官方全历史集经济闸通过，但配对反事实未确认门禁本身创造增量价值——增益可能来自近期 regime，需继续影子攒样本验证稳定性）"
    elif m5v.startswith("FAIL"):
        verdict = ("不合理（官方全历史集经济闸未过：费后期望不显著为正"
                   + ("，且被剔除段并非显著更差 → 门禁无增益）" if m2v != "gate_creates_value" else "）"))
    else:
        verdict = "证据不足（样本功效不够给出方向性结论，维持影子攒样本）"
    out["verdict"] = verdict

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(EVAL_OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"[eval] 五方法结果已存 {EVAL_OUT}")
    _print_summary(out)
    return out


def _print_summary(out: dict) -> None:
    m1, m2, m3, m4, m5 = (out[k] for k in (
        "M1_time_split_and_baseline", "M2_paired_counterfactual",
        "M3_threshold_monotonicity", "M4_time_stability", "M5_economic_gate"))
    print("\n" + "=" * 78)
    print("M1 官方集/重放切分/基线对账")
    off = m1["official_all"]
    print(f"  官方全历史: n={off['n']} wr={off['wr'] and format(off['wr'], '.1%')}"
          f" Wilson={[round(x, 3) for x in off['wilson']] if off['wilson'][0] is not None else '-'}")
    for tag, blk in (("重放发现段", m1["replay_discovery"]), ("重放留出段", m1["replay_holdout"])):
        print(f"  {tag}: n={blk['n']} wr={blk['wr'] if blk['wr'] is None else format(blk['wr'], '.1%')}"
              f" Wilson={[round(x, 3) for x in blk['wilson']] if blk['wilson'][0] is not None else '-'}")
    print(f"  基线对账: {m1.get('baseline_check', 'n/a')}（基线 {m1.get('baseline_2026_08_26')}）")
    print("M2 配对反事实")
    print(f"  过门禁段: n={m2['kept']['n']} wr={m2['kept']['wr'] and format(m2['kept']['wr'], '.1%')}"
          f" 费后EV={m2['kept']['avg_ev_econ']:+.3f}")
    print(f"  被剔除段: n={m2['dropped']['n']} wr={m2['dropped']['wr'] and format(m2['dropped']['wr'], '.1%')}"
          f" 费后EV={m2['dropped']['avg_ev_econ'] and format(m2['dropped']['avg_ev_econ'], '+.3f')}"
          f" CI={m2['dropped']['ev_econ_ci95']}")
    print(f"  判定: {m2['verdict']}")
    print("M3 阈值单调性")
    print(f"  wr曲线({[s['threshold'] for s in m3['sweep']]}): {m3['wr_curve']}")
    print(f"  单调不减(±3pp): {m3['monotonic_non_decreasing(±3pp)']}  最优点: {m3['best_point']}")
    print("M4 时间稳定性")
    print(f"  周桶: {[{'w': b['week'], 'n': b['v2_n'], 'wr': b['v2_wr'] and round(b['v2_wr'], 2)} for b in m4['weekly']]}")
    print(f"  大涨前: n={m4['regime_pre_pump(before 08-19)']['n']} wr={m4['regime_pre_pump(before 08-19)']['wr']}"
          f" / 大涨后: n={m4['regime_post_pump(after 08-19)']['n']} wr={m4['regime_post_pump(after 08-19)']['wr']}")
    print("M5 经济闸（费2%+溢价0.01）")
    for k in ("v1_replay", "v2_replay(∩门禁)", "v2_official(全历史)"):
        blk = m5[k]
        ci = [round(x, 3) for x in blk['ev_econ_ci95']] if blk['ev_econ_ci95'] else None
        ev = f"{blk['avg_ev_econ']:+.3f}" if blk['avg_ev_econ'] is not None else "-"
        print(f"  {k}: n={blk['n']} 费后EV={ev} CI={ci} → {blk['gate']}")
    print("=" * 78)
    print(f"综合裁决: {out['verdict']}")


# ----------------------------------------------------------------------
# stage report：评估部分的人读摘要（最终汇总报告由扫描脚本合并生成）
# ----------------------------------------------------------------------

def stage_report() -> None:
    if not os.path.exists(EVAL_OUT):
        raise SystemExit("先跑 --stage evaluate")
    with open(EVAL_OUT, encoding="utf-8") as f:
        out = json.load(f)
    _print_summary(out)
    print(f"\n[report] 评估摘要输出完毕（结构化数据在 {EVAL_OUT}）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["data", "evaluate", "report", "all"], default="all")
    ap.add_argument("--refresh", action="store_true", help="忽略快照缓存重新拉生产数据")
    ap.add_argument("--merge-dump", action="store_true",
                    help="将 DB 行级 dump（qm_signals_dump.csv）并入快照后再评估")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if args.stage in ("data", "all"):
        snap = stage_data(args.refresh)
    if args.stage in ("evaluate", "all"):
        if args.stage == "evaluate" or args.merge_dump:
            with open(SNAPSHOT, encoding="utf-8") as f:
                snap = json.load(f)
        if args.merge_dump:
            snap = merge_db_dump(snap)
        stage_evaluate(snap)
    if args.stage in ("report", "all"):
        stage_report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
