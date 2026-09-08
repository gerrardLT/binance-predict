#!/usr/bin/env python3
"""Phase 0 历史审计重构（生产同源重放 + 配对政策替换 estimand）。

替代旧 comprehensive/audit 脚本的探索出口（规范 §16）：
- 特征/门 100% 复用 firsthit_shadow_detector 冻结纯函数（Q4 零漂移）；
- G1/G3/G4/G7 族比较 = 全部 G0 分母上的 ΔV = mean((A_c−A_p)·R)（§8.2），
  绝不做「嵌套子集胜率 vs G0 基准」的独立二项检验；
- 护栏敏感性只用触发价 q 代理，显式标记 trigger_q_proxy（§4.2，Phase 2 起换正式 quote）；
- 缺失保持 None；输出仅探索级（exploratory=true），不构成升级结论；
- 探索 CI 用日均值 percentile bootstrap（B=4000, seed=20260908）——
  冻结的 studentized MBB+HAC 属 Phase 4 裁决器，此处禁止提前实现变体。

用法：
    .venv\\Scripts\\python.exe scripts/audit_firsthit_parity.py
输入：output/pull_samples_5m_20260907.json + output/klines_5m_cache_720d.json
      + output/klines_5m_tail_20260907.json
输出：output/firsthit_parity_audit_20260908.json（+ 控制台漏斗）
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

from binance_predict.services.firsthit_shadow_detector import (  # noqa: E402
    FEE_RET, FIRSTHIT_SPECS, _gate_of, extract_firsthit_features,
)

SAMPLES = "output/pull_samples_5m_20260907.json"
KLINES = ["output/klines_5m_cache_720d.json", "output/klines_5m_tail_20260907.json"]
L = 300_000
SPLIT = "2026-08-24"          # 冻结切分（历史探索内部再分前后段看稳定性）
BOOT_B, SEED = 4000, 20260908
GUARDS = (0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12)
FEE = FEE_RET


# ---------- 纯函数（单测覆盖） ----------

def samples_to_curves(rows: list[dict]) -> dict[str, list[dict]]:
    """原始采样行 → {down, btc} 曲线（t/v 升序；None 采样剔除）。"""
    down, btc = [], []
    for r in sorted(rows, key=lambda x: int(x["timestamp"])):
        if r.get("down_price") is not None:
            down.append({"t": int(r["timestamp"]), "v": float(r["down_price"])})
        if r.get("btc_price") not in (None, 0):
            btc.append({"t": int(r["timestamp"]), "v": float(r["btc_price"])})
    return {"down": down, "btc": btc}


def paired_policy_value(events: list[dict], *, child: str, parent: str | None) -> dict:
    """政策替换主 estimand（规范 §8.2）：全部事件为分母，A_p 缺省恒 1（G0）。

    ΔV = mean((A_child − A_parent)·R)；contributing = 标签不同的窗口数。
    """
    if not events:
        return {"n_g0": 0, "contributing": 0, "delta_v": None,
                "child_v": None, "parent_v": None}
    rs = np.array([e["r"] for e in events], float)
    a_c = np.array([1.0 if e["labels"].get(child) else 0.0 for e in events])
    a_p = (np.ones(len(events)) if parent is None
           else np.array([1.0 if e["labels"].get(parent) else 0.0 for e in events]))
    diff = a_c - a_p
    return {
        "n_g0": len(events),
        "contributing": int((diff != 0).sum()),
        "delta_v": float((diff * rs).mean()),
        "child_v": float((a_c * rs).mean()),
        "parent_v": float((a_p * rs).mean()),
    }


def trigger_q_proxy_guard_curve(events: list[dict], *, guards) -> dict:
    """触发价 q 代理护栏敏感性（§4.2：仅 intention 层代理，非可成交口径）。"""
    out: dict[str, dict] = {}
    n = len(events)
    for h in guards:
        elig = [e for e in events if e["q"] < h]
        ev = (float(np.mean([FEE / e["q"] - 1.0 if e["win"] else -1.0 for e in elig]))
              if elig else None)
        out[f"{h:.2f}"] = {
            "eligible_n": len(elig), "coverage": len(elig) / n if n else None,
            "ev": ev,
        }
    return {"proxy": "trigger_q_proxy", "guards": out}


# ---------- 重放与主流程 ----------

def load_klines() -> dict[int, tuple[float, float, float, float]]:
    rows: list = []
    for p in KLINES:
        rows.extend(json.load(open(p, encoding="utf-8")))
    return {int(k[0]): (float(k[1]), float(k[2]), float(k[3]), float(k[4])) for k in rows}


def streak_up_of(k5: dict, window_start: int) -> int | None:
    """前驱 5m 连阳根数（cap 4；K 线缺失 → None）。与综合回测同式。"""
    streak, cur = 0, window_start - L
    while streak < 4:
        bar = k5.get(cur)
        if bar and bar[3] > bar[0]:
            streak += 1
            cur -= L
        else:
            return streak if bar else None
    return streak


def replay_events(k5) -> tuple[list[dict], dict]:
    """生产同源重放：extract_firsthit_features + _gate_of（含 streak）。"""
    raw = json.load(open(SAMPLES, encoding="utf-8"))
    by: dict[int, list[dict]] = defaultdict(list)
    for s in raw:
        by[int(s["timestamp"]) // L * L].append(s)
    events, funnel = [], {"windows": 0, "no_kline": 0, "no_firsthit": 0,
                          "npts_lt8": 0, "valid": 0, "unsettled": 0}
    for w, rows in sorted(by.items()):
        funnel["windows"] += 1
        bar = k5.get(w)
        if not bar or bar[0] <= 0:
            funnel["no_kline"] += 1
            continue
        curves = samples_to_curves(rows)
        ext = extract_firsthit_features(w, bar[0], curves["down"], curves["btc"])
        if ext is None:
            trig = any(0.005 < float(r["down_price"]) <= 0.1
                       for r in rows if r.get("down_price") is not None)
            funnel["no_firsthit" if not trig else "npts_lt8"] += 1
            continue
        if bar[3] == bar[0]:  # 开收相等无法判定
            funnel["unsettled"] += 1
            continue
        funnel["valid"] += 1
        streak = streak_up_of(k5, w)
        labels = {v: _gate_of(v, ext, streak) for v, _ in FIRSTHIT_SPECS}
        labels["g0"] = True
        win = bar[3] < bar[0]
        events.append({
            "window_start": w,
            "day": time.strftime("%Y-%m-%d", time.gmtime(w / 1000)),
            "q": float(ext["q"]), "win": win,
            "r": FEE / float(ext["q"]) - 1.0 if win else -1.0,
            "labels": labels,
        })
    return events, funnel


def day_cluster_ci(pairs: list[tuple[str, float]], b: int = BOOT_B) -> list[float] | None:
    """探索级日均值 percentile bootstrap（非裁决口径，仅稳定性参考）。"""
    if not pairs:
        return None
    by_day: dict[str, list[float]] = defaultdict(list)
    for d, v in pairs:
        by_day[d].append(v)
    days = sorted(by_day)
    if len(days) < 5:
        return None
    means = np.array([float(np.mean(by_day[d])) for d in days])
    rng = np.random.default_rng(SEED)
    pick = rng.integers(0, len(days), size=(b, len(days)))
    boots = means[pick].mean(axis=1)
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def main() -> int:
    k5 = load_klines()
    events, funnel = replay_events(k5)
    calib = [e for e in events if e["day"] < SPLIT]
    post = [e for e in events if e["day"] >= SPLIT]

    nodes = [
        ("A1_G1_vs_0", "firsthit_down_body_v1", None),
        ("A2_G3_vs_0", "firsthit_down_chg_v1", None),
        ("B1_G4_vs_G1", "firsthit_down_g4_v1", "firsthit_down_body_v1"),
        ("C1_G7_vs_G1", "firsthit_down_g7_v1", "firsthit_down_body_v1"),
        ("C2_g7streak_vs_G7", "g7_streak_v1", "firsthit_down_g7_v1"),
        ("X1_g7strict_vs_g7streak", "g7_strict_v1", "g7_streak_v1"),
    ]
    registry: dict = {
        "exploratory": True,
        "note": "Phase 0 历史审计：全部 G0 分母配对政策替换 estimand；"
                "post-split 段已烧毁（规范 §3.2），仅作前后段稳定性对照，"
                "不得作为确认证据；护栏曲线为 trigger_q 代理。",
        "split": SPLIT, "funnel": funnel, "nodes": {},
    }

    def _diff_r(e: dict, child: str, parent: str | None) -> float:
        """ΔV 的逐事件贡献：(A_child − A_parent)·R；parent=None 时 A_p 恒 1（G0）。"""
        a_c = 1.0 if e["labels"].get(child) else 0.0
        a_p = 1.0 if parent is None else (1.0 if e["labels"].get(parent) else 0.0)
        return (a_c - a_p) * e["r"]

    for name, child, parent in nodes:
        seg = {}
        for tag, evs in (("calib", calib), ("burned_post_split", post)):
            st = paired_policy_value(evs, child=child, parent=parent)
            st["delta_ci_day_cluster"] = day_cluster_ci(
                [(e["day"], _diff_r(e, child, parent)) for e in evs])
            seg[tag] = st
        registry["nodes"][name] = seg
    registry["guard_sensitivity_trigger_q_proxy"] = {
        tag: trigger_q_proxy_guard_curve(evs, guards=GUARDS)
        for tag, evs in (("calib", calib), ("burned_post_split", post))
    }
    out = "output/firsthit_parity_audit_20260908.json"
    json.dump(registry, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=float)
    print(json.dumps({"funnel": funnel, "n_calib": len(calib), "n_post": len(post),
                      "out": out}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
