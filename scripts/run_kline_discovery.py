#!/usr/bin/env python
"""720d K 线科学发现流水线 CLI（分阶段可重入，崩溃不重跑）。

用法：
    uv run python scripts/run_kline_discovery.py --tf 5m --stage all
    uv run python scripts/run_kline_discovery.py --tf 15m --stage combo,oos,report
    uv run python scripts/run_kline_discovery.py --tf 5m --stage all --replay-legacy

阶段：build（数据+特征缓存）→ l1 → combo → oos → report；
每阶段读前序落盘产物（out/_stage_*.pkl），holdout 只在 oos 阶段触碰一次。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from binance_predict.backtest.stats import multiple_testing_threshold, variance_ratio  # noqa: E402
from binance_predict.discovery import (  # noqa: E402
    build_feature_matrix, build_targets, condition_mask, data_summary,
    load_klines_csv, make_atoms, merge_r3, parse_condition, run_combos,
    run_l1, run_oos, write_outputs,
)
from binance_predict.discovery.features import atr_series  # noqa: E402
from binance_predict.discovery.hypotheses import DEFAULTS, load_rounds  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAR_MS = {"5m": 300_000, "15m": 900_000}
HORIZONS = {"5m": [1, 2, 3], "15m": [1, 2, 3, 6, 12]}
R3_JSON = os.path.join(ROOT, "config", "discovery_rounds", "r3_regime_conditioned.json")
LEGACY_DIR = os.path.join(ROOT, "output")


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _pkl(path: str, obj) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _unpkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _feat_fp(kl, names: list[str]) -> str:
    """内容指纹：构建器版本 + 数据首末时间戳 + 行数 + 列集合哈希（空列集 = 仅数据侧指纹）。

    版本升级（如特征实现修复）时递增 _FEAT_VERSION 使全部缓存自动失效。
    """
    return hashlib.sha256(
        f"v{_FEAT_VERSION}|{kl.t[0]}|{kl.t[-1]}|{len(kl.t)}|{'|'.join(names)}".encode("utf-8")
    ).hexdigest()[:16]


_FEAT_VERSION = 2  # v2: 修复 _roll_sum NaN 毒化（efficiency 等序列特征曾全 NaN）


def _save_featmat(fm, path: str, fp: str) -> None:
    meta = json.dumps({"names": fm.names, "families": fm.families,
                       "dtypes": fm.dtypes, "fp": fp})
    np.savez_compressed(path, _meta=np.array([meta]),
                        **{f"col_{i}": fm.cols[nm] for i, nm in enumerate(fm.names)})


def _load_featmat(path: str, fp: str):
    from binance_predict.discovery.features import FeatureMatrix
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["_meta"][0]))
    if meta["fp"] != fp:
        raise ValueError("特征指纹不匹配，需重建")
    fm = FeatureMatrix()
    for i, nm in enumerate(meta["names"]):
        arr = z[f"col_{i}"]
        fm.names.append(nm)
        fm.families.append(meta["families"][i])
        fm.dtypes.append(meta["dtypes"][i])
        fm.cols[nm] = arr
    return fm


def stage_build(tf: str, out: str, data_csv: str, rounds: tuple[str, ...]) -> dict:
    bar_ms = BAR_MS[tf]
    kl = load_klines_csv(data_csv, bar_ms)
    ds = data_summary(kl, bar_ms)
    log(f"数据：{ds['rows']} 根 {ds['start']} ~ {ds['end']}（gap={ds['gap_count_gt_1_5x_median']}）")
    fp_path = os.path.join(out, f"featmat_{tf}_720d.npz")
    fp = _feat_fp(kl, [])  # 数据侧指纹（构建前可复现，供缓存校验）
    if os.path.exists(fp_path):
        try:
            fm = _load_featmat(fp_path, fp)
            log(f"特征缓存命中：{len(fm)} 列（{fp_path}）")
            return {"kl": kl, "fm": fm, "ds": ds, "cache": True, "fp": fp}
        except Exception as ex:
            log(f"特征缓存失效（{ex}），重建")
    t0 = time.time()
    k5 = kl if tf == "5m" else load_klines_csv(
        os.path.join(LEGACY_DIR, "klines_5m_720d.csv"), BAR_MS["5m"])
    fm = build_feature_matrix(kl, bar_ms, k5 if tf == "15m" else None)
    log(f"特征构建 {len(fm)} 列 × {len(kl)} 根，耗时 {time.time()-t0:.1f}s")
    _save_featmat(fm, fp_path, fp)
    return {"kl": kl, "fm": fm, "ds": ds, "cache": False, "fp": fp}


def main() -> int:
    ap = argparse.ArgumentParser(description="720d K 线科学发现流水线")
    ap.add_argument("--tf", choices=["5m", "15m"], required=True)
    ap.add_argument("--stage", default="all",
                    help="build|l1|combo|oos|report 逗号分隔，或 all")
    ap.add_argument("--rounds", default="r1,r2,r3", help="头脑风暴轮次")
    ap.add_argument("--data", default=None, help="klines CSV 路径（默认 output/）")
    ap.add_argument("--out", default=None, help="产物目录（默认 output/kline_discovery_{tf}_720d_v2）")
    ap.add_argument("--replay-legacy", action="store_true",
                    help="重放旧产物 registry 条件，对照 holdout 数值（±1pp 验收）")
    args = ap.parse_args()

    tf, bar_ms = args.tf, BAR_MS[args.tf]
    out = args.out or os.path.join(ROOT, "output", f"kline_discovery_{tf}_720d_v2")
    os.makedirs(out, exist_ok=True)
    data_csv = args.data or os.path.join(ROOT, "output", f"klines_{tf}_720d.csv")
    rounds = tuple(r.upper() for r in args.rounds.split(","))
    stages = ["build", "l1", "combo", "oos", "report"] if args.stage == "all" \
        else [s.strip() for s in args.stage.split(",")]
    log(f"tf={tf} stages={stages} rounds={rounds} out={out}")

    cfg = dict(DEFAULTS)
    run_config: dict = {"tf": tf, "rounds": list(rounds), "budget": cfg,
                        "generated_at": datetime.now(timezone.utc).isoformat()}

    # ---------- build（所有阶段都需要数据与特征；命中缓存时零成本） ----------
    st = stage_build(tf, out, data_csv, rounds)
    kl, fm = st["kl"], st["fm"]
    run_config["data_summary"] = st["ds"]
    run_config["feature_fingerprint"] = st.get("fp", "")

    n = len(kl)
    atr_abs = atr_series(kl)
    tg = build_targets(kl.t, kl.o, kl.h, kl.l, kl.c, kl.cont, HORIZONS[tf], atr_abs)

    # ---------- l1 ----------
    if "l1" in stages:
        disc = np.zeros(n, dtype=bool)
        disc[: int(n * cfg["discovery_frac"])] = True
        atoms = make_atoms(fm, disc, tuple(r for r in rounds if r in ("R1", "R2")))
        atoms_by_round = {r: sum(1 for a in atoms if a.round == r) for r in ("R1", "R2")}
        log(f"原子化：R1={atoms_by_round['R1']} R2={atoms_by_round['R2']}")
        # 随机漫步体检（发现段对数收益）
        i1 = int(n * cfg["discovery_frac"])
        lr = np.diff(np.log(kl.c[:i1]))
        vr_health = {str(q): variance_ratio(list(lr), q) for q in (5, 10, 50)}
        t0 = time.time()
        l1 = run_l1(fm, atoms, tg, n, cfg)
        log(f"L1 完成 {sum(v['n_tests'] for v in l1.values())} 检验，"
            f"存活 {sum(len(v['kept']) for v in l1.values())}，耗时 {time.time()-t0:.1f}s")
        _pkl(os.path.join(out, "_stage_l1.pkl"),
             {"l1": l1, "atoms": atoms, "atoms_by_round": atoms_by_round,
              "vr_health": vr_health})

    # ---------- combo ----------
    if "combo" in stages:
        l1p = _unpkl(os.path.join(out, "_stage_l1.pkl"))
        l1, atoms = l1p["l1"], l1p["atoms"]
        t0 = time.time()
        cb = run_combos(fm, atoms, tg, l1, n, cfg)
        for tname, res in cb.items():
            log(f"  {tname}: L2 {res['n_l2_tests']}→{len(res['l2_kept'])}，"
                f"L3 {res['n_l3_tests']}→{len(res['l3_kept'])}，shortlist {len(res['shortlist'])}")
        r3_json: list[dict] = []
        if "R3" in rounds and os.path.exists(R3_JSON):
            r3_json = load_rounds([R3_JSON])
            info = merge_r3(fm, tg, r3_json, cb, n, cfg)
            log(f"R3 预注册 {len(r3_json)} 条 → 并入 {info['merged']}，跳过 {len(info['skipped'])}")
        log(f"组合搜索耗时 {time.time()-t0:.1f}s")
        _pkl(os.path.join(out, "_stage_combo.pkl"), {"cb": cb, "r3_json": r3_json,
                                                      "atoms": atoms, "l1": l1})

    # ---------- oos ----------
    if "oos" in stages:
        cp = _unpkl(os.path.join(out, "_stage_combo.pkl"))
        cb = cp["cb"]
        vol_pct = fm.cols.get("atr_pctile_4320")
        t0 = time.time()
        oos = run_oos(tg, kl.t.astype("datetime64[ms]"), cb, n, vol_pct, cfg)
        n_ver = {}
        for lst in oos.values():
            for r in lst:
                n_ver[r["verdict"]] = n_ver.get(r["verdict"], 0) + 1
        log(f"OOS 完成（holdout 只触碰一次），裁决分布：{n_ver}，耗时 {time.time()-t0:.1f}s")
        _pkl(os.path.join(out, "_stage_oos.pkl"), {"oos": oos})

    # ---------- report ----------
    if "report" in stages:
        l1p = _unpkl(os.path.join(out, "_stage_l1.pkl"))
        cp = _unpkl(os.path.join(out, "_stage_combo.pkl"))
        op = _unpkl(os.path.join(out, "_stage_oos.pkl"))
        l1, cb, oos = l1p["l1"], cp["cb"], op["oos"]
        run_config["vr_health"] = l1p.get("vr_health", {})
        run_config["atoms_by_round"] = l1p.get("atoms_by_round", {})
        total = (sum(v["n_tests"] for v in l1.values())
                 + sum(v["n_l2_tests"] + v["n_l3_tests"] for v in cb.values())
                 + sum(len(v["shortlist"]) for v in cb.values()))
        run_config["total_tests"] = total
        run_config["final_holdout_rule"] = cfg["final_holdout_rule"]
        # 多重检验账本：全扫描的有效门槛是「阶段内 BH-FDR + 全局漏斗预算 + 冻结 holdout」，
        # multiple_testing_threshold 仅作参考记录（其 √n 抬升不适用于已通过 FDR 的存量检验）
        mtt = multiple_testing_threshold(cfg["min_lift_pp"], total)
        run_config["multiple_testing"] = {
            "total_tests": total, "fdr_alpha": cfg["fdr_alpha"],
            "nominal_lift_gate_pp": cfg["min_lift_pp"],
            "bonferroni_reference": mtt,
            "note": "最终裁决只认冻结 holdout；FDR 与漏斗预算为有效门槛",
        }
        log(f"总检验数 {total}（FDR q={cfg['fdr_alpha']}，holdout 只触碰一次）")
        paths = write_outputs(out, run_config=run_config, fm=fm, l1_results=l1,
                              combo_results=cb, oos_results=oos,
                              rounds_json=cp.get("r3_json", []),
                              atoms_by_round=l1p.get("atoms_by_round", {}))
        for k, p in paths.items():
            log(f"  产物 {k}: {p}")

    # ---------- replay-legacy（验收：复现旧最强发现 ±1pp） ----------
    if args.replay_legacy:
        _replay_legacy(tf, fm, tg, kl, n, cfg, out)
    return 0


def _replay_legacy(tf: str, fm, tg, kl, n: int, cfg: dict, out: str) -> None:
    """重放旧产物 discovery_registry.csv 的字面量条件，对照 holdout 胜率。"""
    legacy = os.path.join(LEGACY_DIR, f"kline_discovery_{tf}_720d", "discovery_registry.csv")
    if not os.path.exists(legacy):
        legacy = os.path.join(LEGACY_DIR, f"kline_discovery_gen2_{tf}_720d", "discovery_registry.csv")
    if not os.path.exists(legacy):
        log(f"旧产物不存在，跳过重放：{legacy}")
        return
    i2 = int(n * (cfg["discovery_frac"] + cfg["validation_frac"]))
    rows_out, ok_cnt, bad_cnt = [], 0, 0
    with open(legacy, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cond, tname = row.get("condition", ""), row.get("target", "")
            if not cond or tname not in tg.items:
                continue
            try:
                mask = condition_mask(fm, parse_condition(cond))
            except (KeyError, ValueError):
                bad_cnt += 1
                continue
            ts = tg.items[tname]
            m = mask[i2:] & ts.valid[i2:]
            n_h = int(m.sum())
            wr = float(ts.win[i2:][m].mean()) if n_h else float("nan")
            old_wr = row.get("holdout_win_rate", "")
            diff = ""
            if n_h and old_wr:
                d = abs(wr - float(old_wr)) * 100
                diff = f"{d:.2f}pp"
                ok_cnt += int(d <= 1.0)
            rows_out.append({"discovery_id": row.get("discovery_id", ""),
                             "target": tname, "condition": cond,
                             "replay_n": n_h, "replay_win_rate": wr,
                             "legacy_win_rate": old_wr, "diff": diff})
    rp = os.path.join(out, "replay_legacy.csv")
    with open(rp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["discovery_id", "target", "condition",
                                          "replay_n", "replay_win_rate",
                                          "legacy_win_rate", "diff"])
        w.writeheader()
        w.writerows(rows_out)
    log(f"旧产物重放：{len(rows_out)} 条（±1pp 内 {ok_cnt}，特征缺失 {bad_cnt}）→ {rp}")


if __name__ == "__main__":
    sys.exit(main())
