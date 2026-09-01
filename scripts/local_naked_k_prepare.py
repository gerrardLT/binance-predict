#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""裸K验证 Phase B：基线锚定（Step 1）+ 1h 序列构造与官方交叉校验（Step 2）。

铁律（见 docs/research/naked-k/REPORT.md「数据基线」一节）：
- 本研究**只读**本地冻结 CSV，绝不执行 scripts/refresh_and_export_klines.py
  （该脚本原地重写 output/klines_*_720d.csv，会污染与 output/kline_discovery_*_720d_v2/
  冻结产物的可比性），也绝不写 output/kline_discovery_* 目录。
- 1h 由 5m 聚合得到，走 discovery/data.py 的 aggregate_to（要求桶内根数严格 == n_sub
  且桶内无断点）；**不沿用** refresh 脚本的「根数<3」松口径。
- 产物目录名带 run_fp（输入指纹）：任何输入变更 → 新目录，杜绝静默复用。

用法：
    uv run python scripts/local_naked_k_prepare.py            # 全量（含官方交叉校验 + pytest）
    uv run python scripts/local_naked_k_prepare.py --no-xcheck --skip-pytest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from binance_predict.discovery.data import (aggregate_to, data_summary,  # noqa: E402
                                            load_klines_csv)

BAR_MS = {"5m": 300_000, "15m": 900_000, "1h": 3_600_000}
HEADER_PREFIX = "timestamp,open,high,low,close,volume"

CSV_5M = os.path.join(ROOT, "output", "klines_5m_720d.csv")
CSV_15M = os.path.join(ROOT, "output", "klines_15m_720d.csv")
# 报价样本落在仓库根目录（与 scripts/local_s5_real_quote_ev.py 的 SAMPLES 同路径）
SAMPLES = os.path.join(ROOT, "prediction_market_samples_online_20260819.json")
REGISTRY = os.path.join(ROOT, "config", "naked_k_patterns.json")
ENGINE = os.path.join(ROOT, "scripts", "local_naked_k_engine.py")
PREPARE = os.path.join(ROOT, "scripts", "local_naked_k_prepare.py")
REPORT = os.path.join(ROOT, "scripts", "local_naked_k_report.py")
OUT_ROOT = os.path.join(ROOT, "output", "naked_k_validation")

# 统计层单向 import（见 REPORT「架构决策」）：这些源码一旦变化会改变结果，
# 故必须进入 run_fp；只 import 统计层，不 import 特征层。
_SRC = os.path.join(ROOT, "src", "binance_predict")
DEP_MODULES = {
    "discovery_targets": os.path.join(_SRC, "discovery", "targets.py"),
    "discovery_l1_tester": os.path.join(_SRC, "discovery", "l1_tester.py"),
    "discovery_oos_validator": os.path.join(_SRC, "discovery", "oos_validator.py"),
    "discovery_hypotheses": os.path.join(_SRC, "discovery", "hypotheses.py"),
    "discovery_data": os.path.join(_SRC, "discovery", "data.py"),
    "backtest_stats": os.path.join(_SRC, "backtest", "stats.py"),
}

FP_SOURCES = ([("registry", REGISTRY), ("klines_5m_csv", CSV_5M),
               ("klines_15m_csv", CSV_15M), ("market_samples_json", SAMPLES),
               ("engine_py", ENGINE), ("prepare_py", PREPARE),
               ("report_py", REPORT)]
              + [("dep_" + k, v) for k, v in DEP_MODULES.items()])


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def check_header(path: str) -> dict:
    """断言表头严格为 timestamp,open,high,low,close,volume（data.py:31-33 的前置要求）。"""
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n")
    if not header.startswith(HEADER_PREFIX):
        raise ValueError(f"CSV 表头不符：{path} → {header!r}")
    return {"path": os.path.basename(path), "header": header, "ok": True}


def compute_run_fp() -> tuple[str, dict[str, str]]:
    parts: dict[str, str] = {}
    for name, path in FP_SOURCES:
        if not os.path.exists(path):
            continue
        parts[name] = sha256_file(path)
    missing = [n for n, p in FP_SOURCES if n not in parts]
    blob = json.dumps(parts, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12], {"digests": parts, "absent": missing}


# ============================ Step 2：1h 聚合 ============================

def aggregate_1h(kl5, out_csv: str) -> dict:
    """5m → 1h（n_sub=12），并出聚合审计。

    注：aggregate_to 不抛错，它会**静默丢弃**根数不齐或桶内有断点的小时桶；
    故这里自行重算 keep 并断言与库口径逐位一致，把「丢了多少」显式落到审计里。
    """
    sub_ms = int(kl5.t[1] - kl5.t[0])
    n_sub = BAR_MS["1h"] // sub_ms
    if n_sub <= 1 or BAR_MS["1h"] % sub_ms != 0:
        raise ValueError(f"基周期 {sub_ms}ms 无法聚合到 1h")
    bkt = kl5.t // BAR_MS["1h"]
    uniq, counts = np.unique(bkt, return_counts=True)
    span_buckets = int(bkt[-1] - bkt[0] + 1)
    # 复刻 aggregate_to 的 keep：根数齐 **且** 桶内无断点
    gap_idx = np.nonzero(~kl5.cont)[0]
    bad_bkt = np.unique(bkt[gap_idx])
    inner_ok = np.ones(len(uniq), dtype=bool)
    inner_ok[np.searchsorted(uniq, bad_bkt)] = False
    keep = (counts == n_sub) & inner_ok
    audit = {
        "base_bar_ms": sub_ms, "n_sub_expected": n_sub,
        "rows_5m": int(len(kl5)),
        "buckets_present_in_data": int(len(uniq)),
        "buckets_in_span": span_buckets,
        "buckets_missing_entirely": int(span_buckets - len(uniq)),
        "buckets_dropped_incomplete_count": int((counts != n_sub).sum()),
        "buckets_dropped_inner_gap_only": int(((counts == n_sub) & ~inner_ok).sum()),
        "rows_dropped_with_bucket": int(len(kl5) - int(counts[keep].sum())),
        "cont_break_basebars": int((~kl5.cont[1:]).sum()),
    }
    kl1h = aggregate_to(kl5, BAR_MS["1h"])
    audit["rows_1h"] = int(len(kl1h))
    audit["cont_break_1h"] = int((~kl1h.cont[1:]).sum())
    audit["coverage_ratio"] = round(len(kl1h) * n_sub / len(kl5), 6)
    audit["data_summary_1h"] = data_summary(kl1h, BAR_MS["1h"])
    # 逐桶硬校验：keep 口径必须与库一致，且 OHLCV 逐项可复算
    idx_first = np.searchsorted(bkt, uniq, side="left")
    idx_last = np.searchsorted(bkt, uniq, side="right") - 1
    assert len(kl1h) == int(keep.sum()), (
        f"聚合行数 {len(kl1h)} != 自行重算 keep {int(keep.sum())} → keep 口径与库不一致")
    assert np.array_equal(kl1h.t, uniq[keep] * BAR_MS["1h"]), "聚合桶时间戳非桶首对齐"
    assert np.array_equal(kl1h.cont, _adjacent(uniq[keep] * BAR_MS["1h"], BAR_MS["1h"])), \
        "聚合后 cont 与相邻桶判定不一致"
    assert np.allclose(kl1h.o, kl5.o[idx_first[keep]]), "1h open != 桶首根 open"
    assert np.allclose(kl1h.c, kl5.c[idx_last[keep]]), "1h close != 桶末根 close"
    assert np.allclose(kl1h.h, np.maximum.reduceat(kl5.h, idx_first)[keep]), "1h high != 桶内 max(h)"
    assert np.allclose(kl1h.l, np.minimum.reduceat(kl5.l, idx_first)[keep]), "1h low != 桶内 min(l)"
    assert np.allclose(kl1h.v, np.add.reduceat(kl5.v, idx_first)[keep]), "1h volume != 桶内 sum(v)"
    # 包络自检：任何桶的 high 必 ≥ max(o,c)、low 必 ≤ min(o,c)
    assert np.all((kl1h.h >= np.maximum(kl1h.o, kl1h.c))
                  & (kl1h.l <= np.minimum(kl1h.o, kl1h.c))), "1h 包络关系不成立"
    audit["ohlc_envelope_ok"] = True
    write_klines_csv(kl1h, out_csv)
    audit["csv_written"] = os.path.relpath(out_csv, ROOT).replace("\\", "/")
    return audit


def _adjacent(t: np.ndarray, bar_ms: int) -> np.ndarray:
    """与 data.py:43-45 同构的 cont（首根恒 False）。"""
    cont = np.zeros(len(t), dtype=bool)
    if len(t) > 1:
        cont[1:] = (t[1:] - t[:-1]) == bar_ms
    return cont


def write_klines_csv(kl, path: str) -> None:
    """与 klines_*_720d.csv 同格式：带 +00:00 偏移的 ISO 时间戳 + 8 位小数。

    必须保留偏移量：data.py:38 用 naive datetime.fromisoformat().timestamp() 会把无时区
    串按本地时区解析，造成整体时间轴偏移。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(HEADER_PREFIX + "\n")
        for i in range(len(kl)):
            ts = datetime.fromtimestamp(int(kl.t[i]) / 1000, tz=timezone.utc)
            f.write(f"{ts.isoformat()},{kl.o[i]:.8f},{kl.h[i]:.8f},"
                    f"{kl.l[i]:.8f},{kl.c[i]:.8f},{kl.v[i]:.8f}\n")


def crosscheck_official(kl1h, per_request: int = 1000, sleep: float = 0.2) -> dict:
    """向 data-api.binance.vision 拉官方 1h，逐桶比对 OHLC（只校验不替换）。

    口径：只比对时间戳能对齐的桶；官方多出的桶（本地被丢弃的不完整小时）单独计数。
    """
    start, end = int(kl1h.t[0]), int(kl1h.t[-1]) + BAR_MS["1h"]
    t0 = time.time()
    raw = fetch_klines_page_limited(start, end, BAR_MS["1h"], per_request, sleep)
    nreq = getattr(fetch_klines_page_limited, "requests", 0)
    off_t = np.fromiter((int(r[0]) for r in raw), dtype=np.int64)
    off = {f: np.fromiter((float(r[i]) for r in raw), dtype=np.float64)
           for f, i in (("o", 1), ("h", 2), ("l", 3), ("c", 4))}
    j = np.searchsorted(off_t, kl1h.t)
    aligned = (j < len(off_t)) & (off_t[np.clip(j, 0, max(0, len(off_t) - 1))] == kl1h.t)
    jj = np.clip(j, 0, max(0, len(off_t) - 1))
    mism, worst = {}, {}
    for fld in ("o", "h", "l", "c"):
        d = np.abs(off[fld][jj][aligned] - getattr(kl1h, fld)[aligned])
        mism[fld] = int((d > 1e-8).sum())
        worst[fld] = float(d.max()) if d.size else None
    return {
        "endpoint": "https://data-api.binance.vision/api/v3/klines",
        "interval": "1h", "requests": int(nreq),
        "official_rows": int(len(off_t)), "local_rows": int(len(kl1h)),
        "buckets_matched_by_timestamp": int(aligned.sum()),
        "buckets_not_in_official": int((~aligned).sum()),
        "ohlc_mismatch_buckets": mism,
        "worst_abs_diff": worst,
        "elapsed_sec": round(time.time() - t0, 1),
        "note": "官方端点仅作交叉校验；本研究实际使用 5m 聚合出的 1h，未用官方数据替换。",
    }


def fetch_klines_page_limited(start_ms: int, end_ms: int, interval_ms: int,
                              per_request: int, sleep: float) -> list:
    """自带限速的分页拉取（不复用 fetch_klines 的整段 while，便于统计请求数）。"""
    import urllib.request

    interval = {300_000: "5m", 900_000: "15m", 3_600_000: "1h"}[interval_ms]
    url0 = ("https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT"
            f"&interval={interval}&startTime={{cur}}&endTime={end_ms}&limit={per_request}")
    out, cur, nreq = [], start_ms, 0
    while cur < end_ms:
        batch: list = []
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url0.format(cur=cur), timeout=30) as resp:
                    batch = json.loads(resp.read().decode())
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
        nreq += 1
        if not batch:
            break
        out.extend(batch)
        cur = int(batch[-1][0]) + interval_ms
        time.sleep(sleep)
    fetch_klines_page_limited.requests = nreq  # type: ignore[attr-defined]
    return out


# ============================ 生产回归实况（本地） ============================

def run_pytest_baseline() -> dict:
    """记录冻结口径硬闸门在本地 vs CI 的实际执行状态（.gitignore:39 → CI 整组 skip）。

    不只记计数：闸门一旦变 RED，“1 failed”本身不足以定位口径漂移在哪一段，
    故同时留下失败用例名与断言行原文（实测唯一失败项的 holdout 计数差就在这些行里）。
    """
    cmd = [os.path.join(ROOT, ".venv", "Scripts", "python.exe"), "-m", "pytest",
           "tests/test_kline_shadow_detector.py", "-q", "--tb=short"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    tail = (proc.stdout or "").strip().splitlines()
    summary = tail[-1] if tail else ""
    mp = re.search(r"(\d+) passed", summary)
    ms = re.search(r"(\d+) skipped", summary)
    mf = re.search(r"(\d+) failed", summary)
    return {
        "cmd": " ".join(os.path.basename(c) for c in cmd),
        "returncode": proc.returncode,
        "summary_line": summary,
        "passed": int(mp.group(1)) if mp else 0,
        "skipped": int(ms.group(1)) if ms else 0,
        "failed": int(mf.group(1)) if mf else 0,
        "failed_tests": [ln.split(" ")[1] for ln in tail if ln.startswith("FAILED ")],
        # pytest 的 --tb=short 把断言原文以“E   ”前缀输出，其中包括实际计数 vs 注册表计数
        "assertion_lines": [ln.strip()[2:].strip() for ln in tail
                            if ln.strip().startswith("E ")][-12:],
        "stdout_tail": tail[-14:],
        "stderr_tail": (proc.stderr or "").strip().splitlines()[-3:],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="裸K验证：基线锚定 + 1h 构造")
    ap.add_argument("--no-xcheck", action="store_true", help="跳过官方 1h 交叉校验（离线时）")
    ap.add_argument("--skip-pytest", action="store_true", help="跳过生产回归实况记录")
    args = ap.parse_args()

    run_fp, fp_parts = compute_run_fp()
    assert not fp_parts["absent"], (
        f"run_fp 输入源缺失，指纹将不可复现：{fp_parts['absent']}")
    out_dir = os.path.join(OUT_ROOT, run_fp)
    os.makedirs(out_dir, exist_ok=True)
    print(f"run_fp = {run_fp}")
    print("digests:", json.dumps(fp_parts["digests"], indent=2))
    if fp_parts["absent"]:
        print("absent sources:", fp_parts["absent"])

    for p in (CSV_5M, CSV_15M, SAMPLES):
        if not os.path.exists(p):
            raise FileNotFoundError(f"缺少冻结输入：{p}")
    headers = [check_header(p) for p in (CSV_5M, CSV_15M)]

    kl5 = load_klines_csv(CSV_5M, BAR_MS["5m"])
    kl15 = load_klines_csv(CSV_15M, BAR_MS["15m"])

    agg_audit = aggregate_1h(kl5, os.path.join(out_dir, "klines_1h_720d.csv"))
    kl1h = load_klines_csv(os.path.join(out_dir, "klines_1h_720d.csv"), BAR_MS["1h"])
    kl1h_mem = aggregate_to(kl5, BAR_MS["1h"])
    assert len(kl1h) == len(kl1h_mem) == agg_audit["rows_1h"], "1h CSV 回读行数不符（写入损坏？）"
    for _f in ("t", "o", "h", "l", "c", "v"):
        assert np.allclose(getattr(kl1h, _f), getattr(kl1h_mem, _f)), f"1h CSV 回读 {_f} 漂移"
    agg_audit["csv_roundtrip_exact"] = True

    xcheck = {"skipped": True, "reason": "--no-xcheck"}
    if not args.no_xcheck:
        try:
            xcheck = crosscheck_official(kl1h)
        except Exception as ex:
            xcheck = {"error": f"{type(ex).__name__}: {ex}",
                      "note": "官方端点不可达；聚合审计与逐桶硬校验仍构成 1h 正确性证据。"}

    baseline = {
        "run_fp": run_fp,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        # 注册表全文外置哈希（自引用不可内嵌，见 naked_k_patterns.json semantics.pre_reg_hash_ref）
        "registry_sha256": fp_parts["digests"].get("registry"),
        "digests": fp_parts["digests"],
        "absent_fp_sources": fp_parts["absent"],
        "headers": headers,
        "inputs": {
            "klines_5m": data_summary(kl5, BAR_MS["5m"]),
            "klines_15m": data_summary(kl15, BAR_MS["15m"]),
        },
        "aggregate_1h": agg_audit,
        "official_crosscheck_1h": xcheck,
        "pytest_gate": run_pytest_baseline() if not args.skip_pytest else {"skipped": True},
        "isolation": {
            "reads_frozen_csv_only": True,
            "never_runs_refresh_script": True,
            "never_writes": ["output/klines_*_720d.csv", "output/kline_discovery_*"],
            "gitignore_caveat": ("output/ 被 .gitignore:39 忽略 → CI 无 CSV，"
                                 "tests/test_kline_shadow_detector.py 的 full_env 整组 skip；"
                                 "本研究的口径一致性仅由本地验证 + 本文件哈希基线保证。"),
        },
    }
    with open(os.path.join(out_dir, "baseline.json"), "w", encoding="utf-8") as f:
        json.dump(baseline, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in baseline.items() if k != "digests"},
                     ensure_ascii=False, indent=2))
    print(f"\n[OK] baseline → {os.path.relpath(out_dir, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
