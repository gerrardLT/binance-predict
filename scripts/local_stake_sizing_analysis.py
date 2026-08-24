#!/usr/bin/env python3
"""线上信号全量拉取 + 结合余额的单笔下注金额分析（2026-08-24）。

一、拉取（只读）：预测钱包余额 / 最近实盘订单 / 场景信号（FakeBreakout，
   含 S5=bull_exhaust_confirm）/ 影子信号（misalignment，x4_v1、
   quote_momentum_v1/v2、quote_contrarian_v1/v2）/ 实盘通道状态。
二、逐信号族统计：n、胜率、入场价、逐笔 EV（费 2%：赢 0.98/q−1 / 输 −1），
   对照回测冻结基准 + 二项尾概率（线上偏低方向）。
三、下注金额：凯利 f* = p − (1−p)/b（b=0.98/q−1），给出全凯利 / 半凯利 /
   1/4 凯利三档，叠加连败回撤模拟与系统硬上限（50 USDT/单），
   结合预测钱包余额给出每通道建议单笔金额与总敞口约束。

用法：python scripts/local_stake_sizing_analysis.py
输出：stdout + output/stake_sizing_analysis.log
"""
from __future__ import annotations

import json
import math
import sys
import urllib.request

BASE = "http://165.154.147.155:8082"
LOG = "output/stake_sizing_analysis.log"
FEE = 0.02

# 回测冻结基准（research_win_rates / live_channels 同源）
BENCH_SCENE = {
    "bull_exhaust": (0.644, "S1 多头耗尽→买DOWN"),
    "bull_exhaust_confirm": (0.785, "S5 确认入场→买DOWN"),
    "bear_exhaust": (0.536, "S2 空头耗尽→买UP"),
    "momentum_fade": (0.554, "S4 动量衰竭→买DOWN"),
}
BENCH_SHADOW = {
    # x4 族用「干净口径」（实价覆盖子集，护栏推导同源）——线上实测 43.7%/47.3% 与之吻合；
    # 全样本合并 63.5% 仅 19% 实价覆盖（假设价口径），不适用于下注决策。
    "x4_v1": (0.412, "X4 错位：收阳&end≤40→次窗DOWN（干净口径 41.2%）"),
    "x4_v2": (0.453, "X4 错位·平静市门禁版（干净口径 45.3%）"),
    "quote_momentum_v1": (0.799, "A 顺势 t90-120s q≥0.69→买DOWN"),
    "quote_momentum_v2": (0.799, "A 顺势·门禁版（同区间）"),
    "quote_contrarian_v1": (0.240, "B 逆势 t45-60s q0.15-0.25 买低价侧（赔率型）"),
    "quote_contrarian_v2": (0.240, "B 逆势·门禁版（同区间）"),
}
MAX_ORDER = 50.0  # 系统单笔硬上限（MAX_ORDER_AMOUNT_USDT）


class Tee:
    def __init__(self):
        self.f = open(LOG, "w", encoding="utf-8")
        try:
            sys.__stdout__.reconfigure(encoding="utf-8")
        except Exception:
            pass

    def write(self, s):
        try:
            sys.__stdout__.write(s)
        except Exception:
            pass
        self.f.write(s)

    def flush(self):
        try:
            sys.__stdout__.flush()
        except Exception:
            pass
        self.f.flush()


def get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=60) as r:
        return json.loads(r.read().decode())


def binom_p_le(n: int, k: int, p: float) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(0, k + 1))


def kelly(p: float, q_entry: float) -> float:
    """全凯利比例（单注占资金比例）。b = 0.98/q−1；f* = p − (1−p)/b。"""
    b = (1 - FEE) / q_entry - 1
    if b <= 0:
        return 0.0
    return p - (1 - p) / b


def main() -> int:
    sys.stdout = Tee()

    # ================= 拉取 =================
    wallet = get("/api/prediction-wallet")
    trades = get("/api/trades/recent?limit=100")["orders"]
    fb_sigs = get("/api/fake-breakout/signals?limit=200")["signals"]
    mis = {v: get(f"/api/misalignment/signals?limit=200&version={v}")
           for v in BENCH_SHADOW}
    chans = get("/api/misalignment/signals?limit=1")["live_channels"]["channels"]

    pred_bal = wallet.get("prediction_usdt_free")
    spot_bal = wallet.get("spot_usdt_free")
    print("=" * 78)
    print("〇、账户余额与实盘通道状态（实时拉取）")
    print("=" * 78)
    print(f"预测钱包余额（下单可用）: {pred_bal} USDT")
    print(f"现货余额（可划转）    : {round(spot_bal, 2)} USDT")
    bankroll = float(pred_bal or 0)
    print(f"通道状态（enabled / amount / 护栏 / 今日成交）:")
    for c in chans:
        print(f"  {c['channel']:<28} {'ON ' if c['enabled'] else 'off'} "
              f"{c['amount_usdt']:>5}U  max_exec={c['max_exec_price']:<5} "
              f"fired={c['fire_total']:<3} filled_today={c['filled_today']}")

    # 实盘订单盈亏（按通道汇总）
    print()
    print(f"最近 {len(trades)} 笔实盘订单（含人工测试单）：")
    by_ver: dict[str, dict] = {}
    for o in trades:
        v = o["signal_version"] or "?"
        g = by_ver.setdefault(v, {"n": 0, "filled": 0, "win": 0, "pnl": 0.0})
        g["n"] += 1
        if o["status"] in ("FILLED", "SETTLED", "WON", "LOST") or o.get("win") is not None:
            g["filled"] += 1
        if o.get("win") is True:
            g["win"] += 1
        if o.get("pnl") is not None:
            g["pnl"] += o["pnl"]
    for v, g in sorted(by_ver.items()):
        print(f"  {v:<28} 单数={g['n']:<3} 胜={g['win']:<3} 累计pnl={g['pnl']:+.2f} USDT")

    # ================= 场景信号 =================
    print()
    print("=" * 78)
    print("一、场景信号（FakeBreakout，15m 结算口径，线上真实数据）")
    print("=" * 78)
    groups: dict[str, dict] = {}
    for s in fb_sigs:
        if not s["settle_outcome"]:
            continue
        pt = s["pattern_type"] or s.get("pattern") or "legacy"
        target = "UP" if pt.startswith("bear") else "DOWN"
        g = groups.setdefault(pt, {"rows": [], "target": target})
        g["rows"].append(s)
    print(f"{'场景':<26}{'n':>4}{'线上胜率':>9}{'回测':>7}{'偏离':>8}"
          f"{'尾概率':>8}{'均入场价':>9}{'逐笔均EV':>9}{'累计EV':>9}")
    scene_stats: dict[str, dict] = {}
    for pt, g in sorted(groups.items()):
        rows, t = g["rows"], g["target"]
        n, w = len(rows), sum(r["settle_outcome"] == t for r in rows)
        evs, qs = [], []
        for r in rows:
            q = r["entry_up_price_15m"] if t == "UP" else r["entry_down_price_15m"]
            if q is None:
                continue
            qs.append(q)
            evs.append((0.98 / q - 1) if r["settle_outcome"] == t else -1.0)
        wr = w / n
        b = BENCH_SCENE.get(pt, (None, ""))[0]
        qbar = sum(qs) / len(qs) if qs else float("nan")
        ev = sum(evs) / len(evs) if evs else float("nan")
        scene_stats[pt] = {"n": n, "wr": wr, "qbar": qbar, "ev": ev,
                           "target": t, "tail": None, "bench": b}
        if b is not None:
            scene_stats[pt]["tail"] = binom_p_le(n, w, b)
        label = f"{pt}→买{t}"
        tail_s = f"{scene_stats[pt]['tail']:.3f}" if scene_stats[pt]["tail"] is not None else "—"
        b_s = f"{b:.1%}" if b is not None else "—"
        dev_s = f"{wr - b:+.1%}" if b is not None else ""
        print(f"{label:<28}{n:>4}{wr:>9.1%}{b_s:>7}{dev_s:>8}{tail_s:>8}"
              f"{qbar:>9.3f}{ev:>+9.3f}{sum(evs):>+9.2f}")

    # S5 逐笔明细
    print()
    print("-- S5（bull_exhaust_confirm）逐笔明细：入场价 / 结算 / 单位EV")
    import datetime as dt
    for r in groups.get("bull_exhaust_confirm", {}).get("rows", []):
        q = r["entry_down_price_15m"]
        t = dt.datetime.fromtimestamp(r["signal_time"] / 1000, tz=dt.timezone.utc)
        ev = ((0.98 / q - 1) if r["settle_outcome"] == "DOWN" else -1.0) if q else None
        qs = f"q={q:.2f}" if q else "q=—"
        evs = f"{ev:+.3f}" if ev is not None else "—"
        print(f"  #{r['id']:>3} {t:%m-%d %H:%M} {qs:<8} 结算={r['settle_outcome']:<4} "
              f"EV={evs}  近7d事件数={r.get('n_events_last_7d')}")
    # S5 频率
    s5_rows = groups.get("bull_exhaust_confirm", {}).get("rows", [])
    if len(s5_rows) >= 2:
        ts = [r["signal_time"] for r in s5_rows]
        days = (max(ts) - min(ts)) / 86_400_000
        print(f"  S5 信号频率: {len(s5_rows)} 条 / {days:.1f} 天 ≈ {len(s5_rows)/days:.2f} 条/天")

    # ================= 影子信号 =================
    print()
    print("=" * 78)
    print("二、影子检测器实盘（misalignment_signals，线上真实数据）")
    print("=" * 78)
    print(f"{'version':<24}{'n结算':>6}{'胜率':>8}{'回测':>7}{'偏离':>8}{'尾概率':>8}"
          f"{'avgEV':>8}{'实价率':>7}{'条/天':>7}")
    shadow_stats: dict[str, dict] = {}
    for v, meta in BENCH_SHADOW.items():
        bwr, desc = meta
        st = mis[v]["stats"]
        rows = mis[v]["signals"]
        n_set, wr, ev, cov = st["settled"], st["win_rate"], st["avg_ev"], st["real_quote_coverage"]
        n_win = sum(1 for r in rows if r["win"] is True)
        tail = binom_p_le(n_set, n_win, bwr) if n_set else None
        ws = [r["window_start"] for r in rows]
        freq = len(rows) / ((max(ws) - min(ws)) / 86_400_000) if len(ws) >= 2 else 0.0
        shadow_stats[v] = {"n": n_set, "wr": wr, "bench": bwr, "ev": ev,
                           "tail": tail, "freq": freq, "desc": desc}
        print(f"{v:<24}{n_set:>6}{(f'{wr:.1%}' if wr is not None else '—'):>8}"
              f"{bwr:>7.1%}"
              f"{(f'{wr-bwr:+.1%}' if wr is not None else '—'):>8}"
              f"{(f'{tail:.3f}' if tail is not None else '—'):>8}"
              f"{(f'{ev:+.3f}' if ev is not None else '—'):>8}"
              f"{(f'{cov:.0%}' if cov is not None else '—'):>7}{freq:>7.1f}")

    # ================= 凯利仓位分析（贝叶斯收缩 + 分数凯利） =================
    print()
    print("=" * 78)
    print(f"三、下注金额分析（预测钱包余额 {bankroll:.2f} USDT，单笔硬上限 {MAX_ORDER:.0f} USDT）")
    print("=" * 78)
    print("方法：")
    print("  a) 胜率不用线上点估计也不用回测原值，用贝叶斯收缩：")
    print("     先验 = 回测胜率×0.95（迁移损耗折扣，6 个族线上全部低于回测 6~20pp），")
    print("           Beta 先验强度固定 100（等效信任 100 注）；")
    print("     后验 = 先验 + 线上观测（线上 n 越大权重越大）。")
    print("  b) 赔率用线上真实入场价 q̄：b = 0.98/q̄ − 1（含 2% 费）。")
    print("  c) 凯利 f* = p − (1−p)/b；建议金额 = 余额 × f* × 1/4（分数凯利）。")
    print("  d) 输时 −100% 单注本金；S5 类高入场价（q̄≈0.72）单注波动大。\n")
    print(f"{'信号':<26}{'p̂线上':>7}{'p后验':>7}{'q̄':>6}{'b':>6}{'EV̂':>7}"
          f"{'f*':>7}{'1/4凯利':>8}{'1/2凯利':>8}{'全凯利':>7}{'频率':>6}{'日敞口':>8}")

    PRIOR_STRENGTH = 100.0
    MIGRATION_DISCOUNT = 0.95  # 线上普遍低于回测的系统性迁移损耗

    def kelly_row(name: str, n: int, p_hat: float, p_prior: float,
                  q: float, freq_per_day: float):
        # Beta-Beta 收缩：后验均值 = (S·p_prior + n·p_hat) / (S + n)
        p_post = (PRIOR_STRENGTH * p_prior + n * p_hat) / (PRIOR_STRENGTH + n)
        b = 0.98 / q - 1
        ev = p_post * b - (1 - p_post)
        fk = kelly(p_post, q)
        rec_q = bankroll * fk * 0.25
        rec_h = bankroll * fk * 0.5
        rec_full = bankroll * fk
        daily = rec_q * freq_per_day
        if ev <= 0 or rec_q <= 0:
            rec_s = "—(EV≤0 勿开)"
            daily = 0.0
        else:
            rec_s = f"{rec_q:.1f}U"
        print(f"{name:<28}{p_hat:>7.1%}{p_post:>7.1%}{q:>6.3f}{b:>6.2f}{ev:>+7.3f}"
              f"{fk:>7.1%}{rec_q:>8.1f}{rec_h:>8.1f}{rec_full:>7.1f}"
              f"{freq_per_day:>5.1f}/d{daily:>7.1f}U")
        return {"n": n, "p": p_hat, "p_prior": p_prior, "p_post": p_post,
                "q": q, "b": b, "ev": ev, "fk": fk,
                "rec_q": max(0.0, rec_q), "freq": freq_per_day}

    results: dict[str, dict] = {}
    # 场景族（先验 = 回测基准 × 0.95；频率 = 信号条数 / 天数跨度）
    for pt, st in scene_stats.items():
        if st["n"] < 3 or math.isnan(st["qbar"]) or st["qbar"] <= 0:
            continue
        ts_all = [r["signal_time"] for r in groups[pt]["rows"]]
        days = ((max(ts_all) - min(ts_all)) / 86_400_000) if len(ts_all) >= 2 else 1.0
        freq = st["n"] / max(days, 1.0)
        p_prior = (st["bench"] or st["wr"]) * MIGRATION_DISCOUNT
        results[pt] = kelly_row(pt, st["n"], st["wr"], p_prior, st["qbar"], freq)
    # 影子族：入场价从信号行取（方向侧报价）；先验 = 回测基准 × 0.95
    for v, st in shadow_stats.items():
        rows = mis[v]["signals"]
        qs = []
        for r in rows:
            if r.get("win") is None:
                continue
            eq = r.get("entry_down_price") if r["direction"] == "DOWN" else r.get("entry_up_price")
            if eq is not None:
                qs.append(eq)
        if not qs or st["wr"] is None:
            continue
        qbar = sum(qs) / len(qs)
        p_prior = st["bench"] * MIGRATION_DISCOUNT
        results[v] = kelly_row(v, st["n"], st["wr"], p_prior, qbar, st["freq"])

    # ================= 固定注额回撤模拟 =================
    # 系统语义：固定 amount_usdt 每单（非复利）。按后验胜率 + 建议金额
    # （1/4 凯利；EV≤0 的跳过）模拟 30 天（频率×30 注），统计回撤分布。
    print()
    print("-- 固定注额 30 天蒙特卡洛（10,000 次，1/4 凯利建议额，非复利）：")
    print(f"{'信号':<26}{'注数':>5}{'p后验':>7}{'注额':>7}"
          f"{'回撤中位':>9}{'回撤95%':>9}{'最差回撤':>9}{'连败95%':>8}")
    import random
    random.seed(42)
    for name, st in results.items():
        if st["rec_q"] <= 0 or st["ev"] <= 0:
            continue
        n_bets = max(1, round(st["freq"] * 30))
        p, q, stake = st["p_post"], st["q"], st["rec_q"]
        b = 0.98 / q - 1
        drawdowns, max_streaks = [], []
        for _ in range(10_000):
            eqty, peak, streak, worst_dd, worst_streak = bankroll, bankroll, 0, 0.0, 0
            for _ in range(n_bets):
                if random.random() < p:
                    eqty += stake * b
                    streak = 0
                else:
                    eqty -= stake
                    streak += 1
                    worst_streak = max(worst_streak, streak)
                peak = max(peak, eqty)
                worst_dd = max(worst_dd, (peak - eqty) / peak if peak > 0 else 1.0)
            drawdowns.append(worst_dd)
            max_streaks.append(worst_streak)
        drawdowns.sort()
        max_streaks.sort()
        med = drawdowns[len(drawdowns) // 2]
        p95 = drawdowns[int(len(drawdowns) * 0.95)]
        worst = drawdowns[-1]
        streak95 = max_streaks[int(len(max_streaks) * 0.95)]
        print(f"  {name:<26}{n_bets:>4}{p:>7.1%}{stake:>6.1f}U"
              f"{med:>9.0%}{p95:>9.0%}{worst:>9.0%}{streak95:>8}")

    # ================= 汇总建议 =================
    print()
    print("=" * 78)
    print("四、结论要点")
    print("=" * 78)
    print(f"1. 资金规模：预测钱包 {bankroll:.2f} USDT（+现货 {spot_bal:.2f} 可划转，"
          f"合计 {bankroll + spot_bal:.2f}）。")
    print(f"2. 系统硬上限：单笔 {MAX_ORDER:.0f} USDT；当前通道默认 2 USDT/单"
          f"（S5 通道现设 10 USDT/单）。")
    print("3. 建议金额 = 贝叶斯收缩胜率的 1/4 凯利：小样本族自动收缩、")
    print("   大样本族贴近线上实测；EV≤0 的通道保持关闭。")
    print("4. S5 实盘入场价 q̄≈0.72 显著高于回测假设 0.51——真实赔率 b≈0.37，")
    print("   输时 −100% 本金、赢时仅 +37%：当前 10U/单已超回测口径全凯利，建议下调。")
    print("5. 多通道并行时总敞口 = Σ(各通道单注×频率)：建议日敞口 ≤ 余额 30% "
          f"(≈{bankroll * 0.3:.1f} USDT)。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
