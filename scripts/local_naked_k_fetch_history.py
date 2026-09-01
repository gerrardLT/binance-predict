#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""裸K组合研究 · 延长历史拉取（计划 §3 / §5）。

与 scripts/refresh_and_export_klines.py 的**根本区别**（故另起一脚本，不改旧脚本）：
1. DAYS 参数化（默认 2160），**永不**写 720d 冻结路径 —— 见 `_assert_not_frozen()`。
2. 增量缓存用 npz（紧凑、可校验），不是整表 JSON（2160d ≈ 62 万行 JSON 会占 ~450MB 内存）。
3. 15m 用 discovery.data.aggregate_to（严格：桶内根数齐全且无断点），
   与 refresh 脚本的「根数<3 即保留」松口径不同 —— 本仓库 1h/4h/1d 全走严格聚合，
   混用两种口径会让「同一份 5m 得到的 15m」在两条链路上不可比。
4. 落盘前必须做**重叠区逐根比对**：新序列与 output/klines_5m_720d.csv 的交集
   逐根比 OHLCV，一致率 < 闸门即拒绝写盘（防止在无感知的情况下引入数据源断层）。

数据源固定 https://data-api.binance.vision/api/v3/klines（与 720d 冻结输入同源）。
**不得**切 fapi：不同源会引入 SPOT vs 永续口径断层（计划 §8）。

用法：
    .venv\\Scripts\\python.exe scripts\\local_naked_k_fetch_history.py --days 2160
    .venv\\Scripts\\python.exe scripts\\local_naked_k_fetch_history.py --days 2160 --verify-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from binance_predict.discovery.data import (Klines, aggregate_to,  # noqa: E402
                                            load_klines_csv)

import local_naked_k_prepare as prep  # noqa: E402  只 import 常量与写盘函数

API = "https://data-api.binance.vision/api/v3/klines"
SYMBOL = "BTCUSDT"
BAR_MS_5M = 300_000
DAY_MS = 86_400_000
OUT_DIR = os.path.join(ROOT, "output")

# ⛔ 冻结路径清单：任何写盘目标命中即中止（计划 P10 / §8 致命风险）
FROZEN_PATHS = {
    os.path.normcase(os.path.abspath(prep.CSV_5M)),
    os.path.normcase(os.path.abspath(prep.CSV_15M)),
    os.path.normcase(os.path.abspath(os.path.join(OUT_DIR, "klines_5m_cache_720d.json"))),
}
FROZEN_CSVS = (os.path.join(OUT_DIR, "klines_5m_720d.csv"),)
CONSISTENCY_GATE = 0.9999


def _assert_not_frozen(days: int, paths: list[str]) -> None:
    """路径护栏（计划 §9 测试 7 的同源逻辑，运行时也硬挡一道）。"""
    if days == 720:
        raise SystemExit("[FAIL] DAYS=720 是上一轮冻结输入的运行参数，禁止复用。")
    for p in paths:
        ap = os.path.normcase(os.path.abspath(p))
        if ap in FROZEN_PATHS:
            raise SystemExit(f"[FAIL] 目标路径等于冻结输入，拒绝写入：{p}")
        if "720d" in os.path.basename(ap):
            raise SystemExit(f"[FAIL] 目标文件名含 720d，疑似覆盖冻结输入：{p}")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def now_ms() -> int:
    """用服务器时间近似（本地时钟偏差 <1s 量级；只影响丢弃最后一根未收盘柱）。"""
    with urllib.request.urlopen(API + f"?symbol={SYMBOL}&interval=5m&limit=1", timeout=30) as r:
        return int(json.loads(r.read().decode())[0][0]) + BAR_MS_5M


def _get(url: str, retries: int = 4) -> list:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except Exception as ex:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            print(f"  重试 {attempt + 1}/{retries}: {type(ex).__name__}: {ex}")
            time.sleep(2 * (attempt + 1))
    return []


def fetch_pages(start: int, end: int, per_request: int, sleep: float) -> tuple[np.ndarray, int]:
    """分页拉取，返回 (rows[n,7] = ts,o,h,l,c,v,close_time, n_requests)。

    只保留 API 行里本研究用到的 6 列 + close_time（用于丢弃未收盘柱），
    其余字段（quoteVolume/tradeCount/takerBuy…）不进内存 —— 62 万行全字段 ≈ 450MB。
    """
    out: list[np.ndarray] = []
    cur, nreq = start, 0
    t0 = time.time()
    while cur < end:
        batch = _get(f"{API}?symbol={SYMBOL}&interval=5m"
                     f"&startTime={cur}&endTime={end}&limit={per_request}")
        nreq += 1
        if not batch:
            break
        rows = np.empty((len(batch), 7), dtype=np.float64)
        for i, r in enumerate(batch):
            rows[i, 0] = float(r[0])
            for j, k in enumerate((1, 2, 3, 4, 5)):
                rows[i, 1 + j] = float(r[k])
            rows[i, 6] = float(r[6])
        out.append(rows)
        cur = int(batch[-1][0]) + BAR_MS_5M
        if nreq % 25 == 0:
            got = sum(len(x) for x in out)
            print(f"  {nreq} 请求 / {got:,} 根 / {time.time() - t0:.0f}s", flush=True)
        time.sleep(sleep)
    got = sum(len(x) for x in out)
    print(f"  拉取完成：{nreq} 请求 / {got:,} 根 / {time.time() - t0:.1f}s", flush=True)
    arr = np.concatenate(out) if out else np.empty((0, 7))
    del out
    # 去重 + 按时间戳排序（增量补拉会有重叠）
    ts = arr[:, 0].astype(np.int64)
    order = np.argsort(ts, kind="stable")
    arr = arr[order]
    ts = ts[order]
    keep = np.r_[True, ts[1:] != ts[:-1]]
    return arr[keep], nreq


def cache_path(days: int) -> str:
    return os.path.join(OUT_DIR, f"klines_5m_cache_{days}d.npz")


def load_cache(days: int) -> np.ndarray | None:
    p = cache_path(days)
    if not os.path.exists(p):
        return None
    try:
        with np.load(p) as z:
            return z["rows"].copy()
    except Exception as ex:  # noqa: BLE001
        print(f"  [WARN] 缓存不可读（{type(ex).__name__}: {ex}），按无缓存处理")
        return None


def save_cache(days: int, rows: np.ndarray) -> None:
    np.savez_compressed(cache_path(days), rows=rows.astype(np.float64))


def to_klines(rows: np.ndarray) -> Klines:
    t = rows[:, 0].astype(np.int64)
    cont = np.zeros(len(t), dtype=bool)
    if len(t) > 1:
        cont[1:] = (t[1:] - t[:-1]) == BAR_MS_5M
    return Klines(t=t, o=rows[:, 1].copy(), h=rows[:, 2].copy(), l=rows[:, 3].copy(),
                  c=rows[:, 4].copy(), v=rows[:, 5].copy(), cont=cont)


def overlap_check(rows: np.ndarray) -> dict:
    """与冻结 720d CSV 取交集，逐根比对 OHLCV（计划 G0 的写盘前复算）。"""
    if not os.path.exists(FROZEN_CSVS[0]):
        return {"skipped": True, "reason": "冻结 720d CSV 不存在"}
    ref = load_klines_csv(FROZEN_CSVS[0], BAR_MS_5M)
    ts = rows[:, 0].astype(np.int64)
    j = np.searchsorted(ref.t, ts)
    jj = np.clip(j, 0, max(0, len(ref.t) - 1))
    m = (j < len(ref.t)) & (ref.t[jj] == ts)
    n_ov = int(m.sum())
    if n_ov == 0:
        return {"verdict": "FAIL", "overlap_rows": 0,
                "reason": "新序列与 720d CSV 无时间交集（时间轴异常）"}
    a = rows[m][:, 1:6]
    b = np.column_stack([getattr(ref, f)[jj[m]] for f in ("o", "h", "l", "c", "v")])
    bad = (np.abs(a - b) > 1e-8).any(axis=1)
    ratio = 1.0 - float(bad.sum()) / n_ov
    per_field = {nm: int((np.abs(a[:, k] - b[:, k]) > 1e-8).sum())
                 for k, nm in enumerate(("open", "high", "low", "close", "volume"))}
    verdict = "PASS" if ratio >= CONSISTENCY_GATE else "FAIL"
    return {"verdict": verdict, "overlap_rows": n_ov,
            "frozen_rows": int(len(ref)), "endpoint_rows": int(len(ts)),
            "frozen_only_rows": int(len(ref) - n_ov),
            "per_field_mismatch": per_field, "n_bars_with_diff": int(bad.sum()),
            "consistency_ratio": round(ratio, 8), "gate": CONSISTENCY_GATE,
            "first_diff_ts": [int(x) for x in ts[m][np.nonzero(bad)[0][:5]]]}


def write_15m(kl5: Klines, path: str) -> dict:
    kl15 = aggregate_to(kl5, 900_000)
    prep.write_klines_csv(kl15, path)
    sub_ms = int(kl5.t[1] - kl5.t[0])
    n_sub = 900_000 // sub_ms
    bkt = kl5.t // 900_000
    uniq, counts = np.unique(bkt, return_counts=True)
    return {"rows_15m": int(len(kl15)), "n_sub_expected": int(n_sub),
            "buckets_present": int(len(uniq)),
            "buckets_dropped_incomplete": int((counts != n_sub).sum()),
            "coverage_ratio": round(float(len(kl15) * n_sub / len(kl5)), 6),
            "aggregator": "discovery.data.aggregate_to（严格：根数齐全 + 桶内无断点）"}


def main() -> int:
    ap = argparse.ArgumentParser(description="裸K组合研究：延长 5m 历史（不碰 720d 冻结输入）")
    ap.add_argument("--days", type=int, default=2160)
    ap.add_argument("--per-request", type=int, default=1000)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--verify-only", action="store_true",
                    help="只用现有缓存做重叠区校验，不请求、不写盘")
    args = ap.parse_args()
    days = int(args.days)
    n_requests = 0

    csv_5m = os.path.join(OUT_DIR, f"klines_5m_{days}d.csv")
    csv_15m = os.path.join(OUT_DIR, f"klines_15m_{days}d.csv")
    _assert_not_frozen(days, [csv_5m, csv_15m, cache_path(days)])

    cached = load_cache(days)
    if cached is None:
        srv_now = now_ms()
        start = srv_now - days * DAY_MS
        print(f"[fetch] {days}d：{datetime.fromtimestamp(start / 1000, timezone.utc).isoformat()}"
              f" ~ {datetime.fromtimestamp(srv_now / 1000, timezone.utc).isoformat()}")
        rows, n_requests = fetch_pages(start, srv_now, args.per_request, args.sleep)
        if len(rows) == 0:
            raise SystemExit("[FAIL] 端点无数据返回")
        closed = rows[rows[:, 6] <= srv_now]        # 只保留已收盘柱
        print(f"[fetch] 丢弃未收盘 {len(rows) - len(closed)} 根")
        save_cache(days, closed)
        rows = closed
    else:
        print(f"[cache] 命中缓存 {len(cached):,} 根（--verify-only 或跳过请求）")
        rows = cached
        if args.verify_only:
            rows = rows[rows[:, 0] >= int(time.time() * 1000) - days * DAY_MS]

    kl5 = to_klines(rows)
    if len(kl5) < 2:
        raise SystemExit("[FAIL] 序列过短")
    span_days = float(kl5.t[-1] - kl5.t[0]) / DAY_MS
    gaps = int((~kl5.cont[1:]).sum())
    print(f"[data] {len(kl5):,} 根 5m  {iso(kl5.t[0])} ~ {iso(kl5.t[-1])}  "
          f"跨度 {span_days:.1f}d  断点 {gaps}  理论满铺 {int(span_days * 288) + 1:,}")

    ov = overlap_check(rows)
    print(f"[overlap] verdict={ov.get('verdict')} rows={ov.get('overlap_rows')} "
          f"ratio={ov.get('consistency_ratio')} diff_bars={ov.get('n_bars_with_diff')}")
    if ov.get("verdict") == "FAIL":
        print("[FAIL] 重叠区一致率低于闸门，拒绝写盘（避免静默引入数据源断层）")
        json.dump({"overlap": ov, "written": False}, sys.stdout, ensure_ascii=False, indent=2)
        return 1

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": API, "symbol": SYMBOL, "interval": "5m",
        "days_param": days, "requests_this_run": int(n_requests),
        "rows_5m": int(len(kl5)), "first_ts_ms": int(kl5.t[0]), "last_ts_ms": int(kl5.t[-1]),
        "span_days": round(span_days, 3), "inner_gaps_5m": gaps,
        "expected_dense_rows": int(round(span_days * 288)) + 1,
        "density_ratio": round(len(kl5) / (round(span_days * 288) + 1), 6),
        "closed_bars_only": True,
        "overlap_vs_frozen_720d": ov,
        "written": False,
    }

    if args.verify_only:
        manifest["mode"] = "verify_only"
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    prep.write_klines_csv(kl5, csv_5m)
    agg = write_15m(kl5, csv_15m)
    manifest["written"] = True
    manifest["csv_5m"] = {"path": os.path.relpath(csv_5m, ROOT).replace("\\", "/"),
                          "bytes": os.path.getsize(csv_5m), "sha256": sha256_file(csv_5m)}
    manifest["csv_15m"] = {"path": os.path.relpath(csv_15m, ROOT).replace("\\", "/"),
                           "bytes": os.path.getsize(csv_15m), "sha256": sha256_file(csv_15m),
                           "aggregate_audit": agg}
    # 回读自检：写进去必须能原样读回来（防半写/编码损坏）
    back = load_klines_csv(csv_5m, BAR_MS_5M)
    assert len(back) == len(kl5), f"5m CSV 回读行数 {len(back)} != {len(kl5)}"
    for f in ("t", "o", "h", "l", "c", "v"):
        assert np.allclose(getattr(back, f), getattr(kl5, f), rtol=0, atol=5e-9), f"回读 {f} 漂移"
    manifest["csv_roundtrip_exact_5m"] = True
    with open(os.path.join(OUT_DIR, f"klines_history_{days}d_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[OK] {csv_5m}")
    print(f"[OK] {csv_15m}")
    return 0


def iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


if __name__ == "__main__":
    sys.exit(main())
