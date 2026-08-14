#!/usr/bin/env python3
"""过滤器实验室：A(剩余时间) × B(破位幅度) × C(5m确认入场) 单跑/组合回测。

基于 local_combo_level_matrix_check.py 的周期锚点口径扩展：
- 级别：日线(288) + 4h(48)；方向：破阻力→DOWN / 破支撑→UP
- 结算：周期锚点（周期末价 vs 周期开盘价，与市场真实规则一致）
- 过滤器定义：
  A 剩余时间：信号触发时在 15m 周期内的偏移 offset（0~900s），只保留指定桶
  B 破位幅度：信号时刻现价偏离周期开盘价的幅度（破位方向），只保留指定桶
  C 5m 确认：破位后等当前 5m 窗口收盘，收回到位势内侧才入场
            （入场时点延迟 0~5min，15m token 历史无价——用情景价 + 打平价分析）
- 组合：A+B / A+C / B+C / A+B+C，阈值取单跑扫描中 EV 最优桶（n>=8）

注意：瞬间入场价用 5m 市场 token 价近似（15m token 历史采样缺失），
结论方向可信，绝对数值有偏差；C 的打平价分析不受此影响。

用法：
    python scripts/local_combo_filter_lab.py
"""
from __future__ import annotations

import asyncio
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
from sqlalchemy import select  # noqa: E402

from binance_predict.db.engine import async_session_factory  # noqa: E402
from binance_predict.db.models import SentimentWindow  # noqa: E402

FEE = 0.02
PREMIUM = 0.01
EPS = 0.0005
MIN_N = 8  # 桶内最小注数（不足不参选最优，仍打印）
LEVELS = (("4h", 48), ("日线", 288))

# A 桶：15m 周期内偏移（分钟）
A_BINS = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15)]
# B 桶：破位幅度（%）
B_BINS = [(0.0, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.50), (0.50, 99.0)]
# C 入场价情景（确认后 15m token 价未知，给区间）
C_PRICE_SCENARIOS = (0.25, 0.35, 0.45)


def _sorted_pairs(curve: list | None) -> list[tuple[int, float]]:
    pts = [
        (int(p.get("t", 0)), float(p["v"]))
        for p in (curve or [])
        if p.get("v") is not None
    ]
    pts.sort()
    return pts


def _price_at(pairs: list[tuple[int, float]], t_ms: int) -> float | None:
    if not pairs:
        return None
    if t_ms <= pairs[0][0]:
        return pairs[0][1]
    if t_ms >= pairs[-1][0]:
        return pairs[-1][1]
    for (t0, v0), (t1, v1) in zip(pairs, pairs[1:]):
        if t0 <= t_ms <= t1:
            if t1 == t0:
                return v0
            w = (t_ms - t0) / (t1 - t0)
            return v0 + (v1 - v0) * w
    return pairs[-1][1]


def _bet_pnl(win: bool, price: float) -> float:
    if not win:
        return -1.0
    return (1.0 - FEE) / min(max(price + PREMIUM, 0.01), 0.99) - 1.0


def _ev(pnls: list[float]) -> float:
    return float(np.mean(pnls)) if pnls else 0.0


def _fmt(pnls: list[float], wins: list[bool]) -> str:
    n = len(pnls)
    if not n:
        return "0 注"
    wr = sum(wins) / n
    ev = _ev(pnls)
    if n >= MIN_N:
        rng = np.random.default_rng(7)
        ix = rng.integers(0, n, size=(2000, n))
        ci = np.percentile(np.asarray(pnls)[ix].mean(axis=1), [2.5, 97.5])
        return f"注数 {n:>4} 胜率 {wr:.1%} 费后EV {ev:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"
    return f"注数 {n:>4} 胜率 {wr:.1%} 费后EV {ev:+.3f} (n<{MIN_N}，仅供参考)"


async def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    async with async_session_factory() as session:
        rows = (await session.execute(
            select(
                SentimentWindow.start_time,
                SentimentWindow.curve_btc_price,
                SentimentWindow.curve_down_price,
                SentimentWindow.curve_up_price,
            ).order_by(SentimentWindow.start_time)
        )).all()

    wins: list[dict] = [
        {"btc": r.curve_btc_price, "down": r.curve_down_price, "up": r.curve_up_price}
        for r in rows
    ]
    print(f"5m 窗口总数 {len(wins)}")

    btc_all: list[tuple[int, float]] = []
    for w in wins:
        btc_all.extend(_sorted_pairs(w["btc"]))
    btc_all.sort()
    t_max = btc_all[-1][0]

    # 每窗口 closes 末点（位势计算 + 5m 收盘确认）
    closes: list[float | None] = []
    for w in wins:
        pairs = _sorted_pairs(w["btc"])
        closes.append(pairs[-1][1] if pairs else None)

    def collect(lookback: int, side: str) -> list[dict]:
        """破位事件 + 过滤器所需的全部特征字段。"""
        events: list[dict] = []
        for idx, w in enumerate(wins):
            if idx <= lookback:
                continue
            btc = _sorted_pairs(w["btc"])
            if len(btc) < 2:
                continue
            hist_closes = [c for c in closes[idx - lookback: idx] if c is not None]
            if len(hist_closes) < lookback // 2:
                continue

            if side == "high":
                ext_t, ext_v = max(btc, key=lambda p: p[1])
                level = max(hist_closes)
                broke = ext_v > level * (1.0 + EPS)
                token_curve = _sorted_pairs(w["down"])
            else:
                ext_t, ext_v = min(btc, key=lambda p: p[1])
                level = min(hist_closes)
                broke = ext_v < level * (1.0 - EPS)
                token_curve = _sorted_pairs(w["up"])
            if not broke:
                continue

            cyc15_start = ext_t - (ext_t % 900_000)
            cyc15_end = cyc15_start + 900_000
            cyc5_start = ext_t - (ext_t % 300_000)
            cyc5_end = cyc5_start + 300_000
            if cyc15_end > t_max:
                continue
            entry = _price_at(token_curve, ext_t)
            p_s15, p_e15 = _price_at(btc_all, cyc15_start), _price_at(btc_all, cyc15_end)
            p_s5, p_e5 = _price_at(btc_all, cyc5_start), _price_at(btc_all, cyc5_end)
            if (
                entry is None or entry <= 0
                or p_s15 is None or p_e15 is None
                or p_s5 is None or p_e5 is None
            ):
                continue

            win_close = closes[idx]
            if win_close is None:
                continue
            if side == "high":
                ret = ext_v / p_s15 - 1.0          # 周期内涨幅（破位方向幅度）
                reclaimed = win_close <= level      # 5m 收盘收回阻力位之下
            else:
                ret = 1.0 - ext_v / p_s15           # 周期内跌幅
                reclaimed = win_close >= level      # 5m 收盘收回支撑位之上

            events.append({
                "entry": entry,
                "side": side,
                "offset_min": (ext_t - cyc15_start) / 60_000.0,  # A：周期内偏移（分钟）
                "ret_pct": ret * 100.0,                          # B：破位幅度（%）
                "reclaimed": reclaimed,                          # C：5m 收盘是否收回
                "p_s15": p_s15, "p_e15": p_e15,
                "p_s5": p_s5, "p_e5": p_e5,
            })
        return events

    def win15(e: dict) -> bool:
        return (e["p_e15"] < e["p_s15"]) if e["side"] == "high" else (e["p_e15"] > e["p_s15"])

    def win5(e: dict) -> bool:
        return (e["p_e5"] < e["p_s5"]) if e["side"] == "high" else (e["p_e5"] > e["p_s5"])

    def eval_subset(evts: list[dict], entry_override: float | None = None) -> tuple[str, float, int]:
        """返回 (15m 口径统计串, 15m EV, 注数)。entry_override 用于 C 的情景价。"""
        w15 = [win15(e) for e in evts]
        prices = [entry_override if entry_override is not None else e["entry"] for e in evts]
        p15 = [_bet_pnl(w_, p) for w_, p in zip(w15, prices)]
        return _fmt(p15, w15), _ev(p15), len(evts)

    def eval_5m(evts: list[dict]) -> str:
        w5 = [win5(e) for e in evts]
        p5 = [_bet_pnl(w_, e["entry"]) for w_, e in zip(w5, evts)]
        return _fmt(p5, w5)

    for label, lookback in LEVELS:
        for side, side_cn in (("high", "破阻力→买DOWN"), ("low", "破支撑→买UP")):
            evts = collect(lookback, side)
            print(f"\n{'=' * 72}\n■ {label} {side_cn}（基线 {len(evts)} 注）\n{'=' * 72}")
            if not evts:
                continue

            # ---- 基线 ----
            s15, ev_base, _ = eval_subset(evts)
            print(f"[基线·无过滤] 15m: {s15}")
            print(f"[基线·无过滤]  5m: {eval_5m(evts)}")

            # ---- A 单跑：剩余时间分桶 ----
            print("[A·周期内偏移分桶]（offset 越小=周期越早）")
            a_best: tuple[float, tuple[float, float]] | None = None  # (ev, bin)
            for lo, hi in A_BINS:
                sub = [e for e in evts if lo <= e["offset_min"] < hi]
                s, ev, n = eval_subset(sub)
                print(f"  {lo:>2}~{hi:<2}min: 15m {s}  | 5m {eval_5m(sub)}")
                if n >= MIN_N and (a_best is None or ev > a_best[0]):
                    a_best = (ev, (lo, hi))
            a_bin = a_best[1] if a_best else None
            if a_bin:
                print(f"  → A 最优桶 {a_bin[0]:.0f}~{a_bin[1]:.0f}min（EV {a_best[0]:+.3f}）")
            else:
                print("  → 无满足最小注数的桶，A 不参与组合")

            # ---- B 单跑：破位幅度分桶 ----
            print("[B·破位幅度分桶]")
            b_best: tuple[float, tuple[float, float]] | None = None
            for lo, hi in B_BINS:
                sub = [e for e in evts if lo <= e["ret_pct"] < hi]
                s, ev, n = eval_subset(sub)
                hi_s = f"{hi:.1f}" if hi < 90 else "∞"
                print(f"  {lo:.2f}~{hi_s}%: 15m {s}  | 5m {eval_5m(sub)}")
                if n >= MIN_N and (b_best is None or ev > b_best[0]):
                    b_best = (ev, (lo, hi))
            b_bin = b_best[1] if b_best else None
            if b_bin:
                print(f"  → B 最优桶 {b_bin[0]:.2f}~{b_bin[1]:.2f}%（EV {b_best[0]:+.3f}）")
            else:
                print("  → 无满足最小注数的桶，B 不参与组合")

            # ---- C 单跑：5m 收盘确认（仅 15m 口径）----
            sub_c = [e for e in evts if e["reclaimed"]]
            print(f"[C·5m收盘确认后入场] 确认率 {len(sub_c)}/{len(evts)}")
            if sub_c:
                p_c = sum(win15(e) for e in sub_c) / len(sub_c)
                # 打平价：EV=0 → x* = (1-FEE)×p - PREMIUM；打平基线 → x = (1-FEE)×p/(1+EV_base) - PREMIUM
                be_zero = (1 - FEE) * p_c - PREMIUM
                be_base = (1 - FEE) * p_c / (1 + ev_base) - PREMIUM
                print(f"  确认后 15m 胜率 {p_c:.1%}（基线 {sum(win15(e) for e in evts)/len(evts):.1%}）")
                print(f"  EV=0 打平入场价 {be_zero:.3f} ｜ 打平基线EV({ev_base:+.2f}) 入场价 {be_base:.3f}")
                for px in C_PRICE_SCENARIOS:
                    s, _, _ = eval_subset(sub_c, entry_override=px)
                    print(f"  情景价 {px:.2f}: {s}")
            else:
                print("  0 注确认事件")

            # ---- 组合 ----
            def mask_a(e: dict) -> bool:
                return a_bin is not None and a_bin[0] <= e["offset_min"] < a_bin[1]

            def mask_b(e: dict) -> bool:
                return b_bin is not None and b_bin[0] <= e["ret_pct"] < b_bin[1]

            combos: list[tuple[str, list[dict], bool]] = [
                ("A+B", [e for e in evts if mask_a(e) and mask_b(e)], False),
                ("A+C", [e for e in evts if mask_a(e) and e["reclaimed"]], True),
                ("B+C", [e for e in evts if mask_b(e) and e["reclaimed"]], True),
                ("A+B+C", [e for e in evts if mask_a(e) and mask_b(e) and e["reclaimed"]], True),
            ]
            print("[组合]（C 组合按情景价 0.35 计，另附打平价）")
            for name, sub, is_c in combos:
                if not sub:
                    print(f"  {name}: 0 注")
                    continue
                if not is_c:
                    s, _, _ = eval_subset(sub)
                    print(f"  {name}: 15m {s}  | 5m {eval_5m(sub)}")
                else:
                    p_w = sum(win15(e) for e in sub) / len(sub)
                    be_zero = (1 - FEE) * p_w - PREMIUM
                    be_base = (1 - FEE) * p_w / (1 + ev_base) - PREMIUM
                    s, _, _ = eval_subset(sub, entry_override=0.35)
                    print(
                        f"  {name}: 胜率 {p_w:.1%}（{len(sub)} 注）｜ 情景价0.35 EV {s.split('费后EV')[1].strip()} "
                        f"｜ EV=0打平价 {be_zero:.3f} 打平基线价 {be_base:.3f}"
                    )

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
