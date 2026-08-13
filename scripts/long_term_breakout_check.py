#!/usr/bin/env python3
"""长周期验证：用币安 5m K 线回放整段 12.6w→5.8w 空头周期，分段检验假突破信号。

背景：本地窗口数据（2026-07-13 ~ 08-13）处于大空头（2025-10 高点 12.6w 起）
末段的反弹横盘期。本脚本把验证区间拉长到整个空头周期，回答：
"连续3窗+假突破"信号是空头环境专属，还是全周期有效？

分段（按真实走势）：
- 主跌段     2025-10-06 ~ 2026-02-10（12.6w → 6w，跌 52%）
- 反弹段     2026-02-10 ~ 2026-04-20（6w → 7.9w，涨 32%）
- 二次下跌段 2026-04-20 ~ 2026-06-25（7.9w → 5.8w，跌 27%）
- 修复段     2026-06-25 ~ 至今（5.8w → 6.3w 横盘修复）

信号定义（与本地验证同构，粒度适配 5m K 线）：
- 支撑/阻力 = 前 24 根 5m K 线（2 小时）的低点/高点
- 假突破向下：本根 K 线下影线击穿支撑 0.05% 但收盘收回 → 下注 UP
- 假突破向上：上影线冲破阻力 0.05% 但收盘回落 → 下注 DOWN
- 连续 3 窗 = 信号 K 线之前 3 根 K 线连续同向（close vs prev close）
- 结算 = 下一根 K 线涨跌符号（平盘剔除）
- 费后 EV 用 50/50 定价假设（真实市场开盘价中位已验证 ≈0.51，近似成立）
  赢 → (1-0.02)/(0.5+0.01)-1 = +0.9216；输 → -1（盈亏平衡胜率 52.0%）
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from datetime import datetime, timezone

import numpy as np

_KLINES_URL = "https://api.binance.com/api/v3/klines"
_FEE = 0.02
_PREMIUM = 0.01
_WIN_PNL = (1.0 - _FEE) / (0.5 + _PREMIUM) - 1.0  # +0.9216
_LOOKBACK = 24  # 2h 支撑 = 前 24 根 5m K 线
_EPS = 0.0005
_STREAK_K = 3

_SEGMENTS = (
    ("主跌段(12.6w→6w)", "2025-10-06", "2026-02-10"),
    ("反弹段(6w→7.9w)", "2026-02-10", "2026-04-20"),
    ("二次下跌(7.9w→5.8w)", "2026-04-20", "2026-06-25"),
    ("修复段(5.8w→6.3w)", "2026-06-25", "2026-08-13"),
)


def _fetch(start_ms: int, end_ms: int) -> list[list]:
    out: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        url = (
            f"{_KLINES_URL}?symbol=BTCUSDT&interval=5m"
            f"&startTime={cursor}&endTime={end_ms}&limit=1000"
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    data = json.loads(resp.read().decode())
                break
            except Exception:
                time.sleep(2)
        else:
            print(f"拉取失败: {url[:100]} ...")
            break
        if not data:
            break
        out.extend(data)
        cursor = data[-1][0] + 300_000
        if len(data) < 1000:
            break
    return out


def _seg_key(t_ms: int) -> str | None:
    t = datetime.fromtimestamp(t_ms / 1000, tz=timezone.utc)
    for name, s, e in _SEGMENTS:
        s_dt = datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
        e_dt = datetime.fromisoformat(e).replace(tzinfo=timezone.utc)
        if s_dt <= t < e_dt:
            return name
    return None


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    start = datetime(2025, 10, 1, tzinfo=timezone.utc)
    end = datetime(2026, 8, 13, tzinfo=timezone.utc)
    print(f"拉取 5m K 线: {start} → {end} ...")
    rows = _fetch(int(start.timestamp() * 1000), int(end.timestamp() * 1000))
    print(f"共 {len(rows)} 根")

    # 构造窗口序列：open/high/low/close/open_time
    wins = [
        {
            "t": r[0], "o": float(r[1]), "h": float(r[2]),
            "l": float(r[3]), "c": float(r[4]), "prev_c": None,
        }
        for r in rows
    ]
    # 相邻 close 差值（用于连续窗判定与结算）
    for i in range(1, len(wins)):
        wins[i]["prev_c"] = wins[i - 1]["c"]

    # 分段统计
    segs: dict[str, list[dict]] = {}
    for w in wins:
        k = _seg_key(w["t"])
        if k:
            segs.setdefault(k, []).append(w)

    print("\n===== 连续3窗+假突破，各阶段表现（结算=下一根 K 线，定价 0.5）=====")
    all_rows: list[dict] = []
    for seg_name, seg_wins in segs.items():
        for side, direction in (("down", "UP"), ("up", "DOWN")):
            bets: list[bool] = []
            for idx, w in enumerate(seg_wins):
                if idx < _LOOKBACK or w["prev_c"] is None:
                    continue
                # 支撑/阻力 = 前 24 根 K 线极值（不含当前）
                hist = seg_wins[idx - _LOOKBACK: idx]
                support = min(x["l"] for x in hist)
                resistance = max(x["h"] for x in hist)

                if side == "down":
                    broke = w["l"] < support * (1.0 - _EPS)
                    reclaimed = w["c"] >= support
                else:
                    broke = w["h"] > resistance * (1.0 + _EPS)
                    reclaimed = w["c"] <= resistance
                if not (broke and reclaimed):
                    continue

                # 连续 3 窗同向（close vs prev close）
                streak_ok = True
                for j in range(1, _STREAK_K + 1):
                    if idx - j < 0 or seg_wins[idx - j]["prev_c"] is None:
                        streak_ok = False
                        break
                    d = seg_wins[idx - j]["c"] - seg_wins[idx - j]["prev_c"]
                    if side == "down" and d >= 0:
                        streak_ok = False
                        break
                    if side == "up" and d <= 0:
                        streak_ok = False
                        break
                if not streak_ok:
                    continue

                # 结算 = 下一根 K 线
                if idx + 1 >= len(seg_wins):
                    continue
                nxt = seg_wins[idx + 1]
                d = nxt["c"] - nxt["o"]
                if d == 0:
                    continue
                win = (d > 0) if direction == "UP" else (d < 0)
                bets.append(win)

            if not bets:
                print(f"  {seg_name:<18} {side:>4}→{direction}: 0 注")
                continue
            n = len(bets)
            win_rate = sum(bets) / n
            pnls = [_WIN_PNL if w else -1.0 for w in bets]
            ev = float(np.mean(pnls))
            if n >= 8:
                rng = np.random.default_rng(7)
                ix = rng.integers(0, n, size=(2000, n))
                ci = np.percentile(np.asarray(pnls)[ix].mean(axis=1), [2.5, 97.5])
                lo, hi = float(ci[0]), float(ci[1])
            else:
                lo, hi = float("nan"), float("nan")
            all_rows.append({
                "segment": seg_name, "side": side, "direction": direction,
                "n": n, "win_rate": round(win_rate, 4),
                "ev": round(ev, 4), "ci_lower": round(lo, 4), "ci_upper": round(hi, 4),
            })
            print(
                f"  {seg_name:<18} {side:>4}→{direction}: 注数 {n:>4} 胜率 {win_rate:.1%} "
                f"费后EV {ev:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            )

    # 全期合并
    print("\n===== 全周期合并（2025-10-06 ~ 至今）=====")
    for side, direction in (("down", "UP"), ("up", "DOWN")):
        rows_d = [r for r in all_rows if r["side"] == side]
        n = sum(r["n"] for r in rows_d)
        wr = sum(r["win_rate"] * r["n"] for r in rows_d) / n if n else 0
        ev = sum(r["ev"] * r["n"] for r in rows_d) / n if n else 0
        print(f"  {side:>4}→{direction}: 注数 {n:>4} 胜率 {wr:.1%} 费后EV {ev:+.3f}")

    with open("output/long_term_breakout_report.json", "w", encoding="utf-8") as f:
        json.dump({"segments": _SEGMENTS, "rows": all_rows}, f, ensure_ascii=False, indent=2)
    print("\n报告已写入 output/long_term_breakout_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
