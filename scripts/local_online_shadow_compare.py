"""线上场景信号 vs 影子检测器实盘 对比（2026-08-20 拉取）。

数据源：output/online_signals_now_*.json（FakeBreakout）+ output/online_shadow_now_*.json
（misalignment_signals：x4_v1 / quote_momentum_v1 / quote_contrarian_v1）。
回测基准（同源冻结口径）：
    X4 错位        合并 63.5%（IS 65.6/OOS 57.8） EV+0.254 CI(0.038,0.493)
    A 顺势 momentum 79.9% EV+0.097
    B 逆势 contrarian 24.0% EV+0.155（赔率型）
    场景 S1 bull_exhaust 64.4% / S2 bear_exhaust 53.6% / S4 momentum_fade 55.4%
    盈亏平衡胜率 52.0%（费2%+溢0.01，赔率0.9216）
"""
import glob
import json
import math

SIG = sorted(glob.glob("output/online_signals_now_2*.json"))[-1]
SHD = sorted(glob.glob("output/online_shadow_now_2*.json"))[-1]
D = json.load(open(SIG, encoding="utf-8"))
S = json.load(open(SHD, encoding="utf-8"))
sigs, shadow = D["signals"], S["by_version"]

BREAKEVEN = 0.5204


def binom_p_le(n, k, p):
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def days_span(ts_list):
    return max(1.0, (max(ts_list) - min(ts_list)) / 86_400_000)


print("=" * 78)
print("一、线上场景信号（FakeBreakout，15m 结算口径）")
print("=" * 78)
BENCH = D["stats"]["research_win_rates"]
groups = {}
for s in sigs:
    if not s["settle_outcome"]:
        continue
    pt = s["pattern_type"] or s.get("pattern") or "legacy"
    target = "UP" if pt.startswith("bear") else "DOWN"
    g = groups.setdefault(pt, {"n": 0, "w": 0, "target": target, "sigs": []})
    g["n"] += 1
    g["w"] += s["settle_outcome"] == g["target"]
    g["sigs"].append(s)
print(f"{'场景':<24}{'线上':>12}{'回测':>8}{'偏离':>8}{'尾概率P(X<=k)':>14}{'均EV':>9}{'累计EV':>9}{'报价覆盖':>9}")
tot_n = tot_w = 0
tot_ev_sum = tot_ev_n = 0
for pt, g in sorted(groups.items()):
    n, w, t = g["n"], g["w"], g["target"]
    tot_n, tot_w = tot_n + n, tot_w + w
    # 逐笔 EV（与回测同口径：费2%+溢0.01，赢 0.98/q-1，输 -1；
    # bear 族买 UP 用 entry_up_price_15m，其余买 DOWN 用 entry_down_price_15m）
    evs = []
    for s in g["sigs"]:
        q = s["entry_up_price_15m"] if t == "UP" else s["entry_down_price_15m"]
        if q is None or not s["settle_outcome"]:
            continue
        evs.append((0.98 / q - 1) if s["settle_outcome"] == t else -1.0)
    tot_ev_sum += sum(evs)
    tot_ev_n += len(evs)
    wr, b = w / n, BENCH.get(pt)
    label = f"{pt}→买{t}"
    ev_str = f"{sum(evs)/len(evs):+.3f}" if evs else "—"
    cum_str = f"{sum(evs):+.2f}" if evs else "—"
    cov_str = f"{len(evs)}/{n}" if evs else f"0/{n}"
    if b is None:
        print(f"{label:<28}{w}/{n} = {wr:.1%}{'—':>8}{'':>14}{ev_str:>9}{cum_str:>9}{cov_str:>9}")
    else:
        print(f"{label:<28}{w}/{n} = {wr:.1%}{b:>8.1%}{wr-b:>+8.1%}{binom_p_le(n, w, b):>14.3f}"
              f"{ev_str:>9}{cum_str:>9}{cov_str:>9}")
print(f"{'合计(方向结算)':<24}{tot_w}/{tot_n} = {tot_w/tot_n:.1%}{'':>22}"
      f"{tot_ev_sum/tot_ev_n:>+9.3f}{tot_ev_sum:>+9.2f}{tot_ev_n}/{tot_n:>2}")
ts = [s["signal_time"] for s in sigs]
print(f"信号频率: {len(sigs)} 条 / {days_span(ts):.1f} 天 ≈ {len(sigs)/days_span(ts):.1f} 条/天")

print()
print("=" * 78)
print("二、影子检测器实盘（misalignment_signals，逐 version）")
print("=" * 78)
BENCH_SH = {
    "x4_v1": (0.635, 0.254, "收阳&end≤40→次窗DOWN"),
    "quote_momentum_v1": (0.799, 0.097, "A顺势 t90-120s q≥0.69→买DOWN"),
    "quote_contrarian_v1": (0.240, 0.155, "B逆势 t45-60s q0.15-0.25→买低价侧赔率型"),
}
print(f"{'version':<22}{'n(结算)':>8}{'胜率':>10}{'回测':>8}{'偏离':>8}{'尾概率':>8}{'avgEV':>8}{'回测EV':>8}{'实价率':>8}{'条/天':>7}")
for v, meta in BENCH_SH.items():
    bwr, bev, _ = meta
    st = shadow[v]["stats"]
    rows = shadow[v]["signals"]
    n_set, wr, ev, cov = st["settled"], st["win_rate"], st["avg_ev"], st["real_quote_coverage"]
    n_win = sum(1 for r in rows if r["win"] is True)
    # contrarian 是赔率型（胜率 24% 仍正 EV），检验方向取"偏离回测"双侧意义用下尾
    tail = binom_p_le(n_set, n_win, bwr) if n_set else None
    ws = [r["window_start"] for r in rows]
    freq = len(rows) / days_span(ws) if ws else 0.0
    print(f"{v:<22}{n_set:>8}{('—' if wr is None else f'{wr:.1%}'):>10}{bwr:>8.1%}"
          f"{('—' if wr is None else f'{wr-bwr:+.1%}'):>8}"
          f"{('—' if tail is None else f'{tail:.3f}'):>8}"
          f"{('—' if ev is None else f'{ev:+.3f}'):>8}{bev:>+8.3f}"
          f"{('—' if cov is None else f'{cov:.0%}'):>8}{freq:>7.1f}")

print()
print("=" * 78)
print("三、影子逐笔明细（最近结算）")
print("=" * 78)
import datetime as dt
for v in BENCH_SH:
    print(f"-- {v}（{BENCH_SH[v][2]}）")
    for r in shadow[v]["signals"]:
        if r["status"] != "SETTLED":
            continue
        t = dt.datetime.fromtimestamp(r["window_start"] / 1000, tz=dt.timezone.utc).strftime("%m-%d %H:%M")
        eq = r.get("entry_down_price") if r["direction"] == "DOWN" else r.get("entry_up_price")
        eqs = f"q={eq:.2f}" if eq is not None else "q=—"
        evs = f"{r['ev_at_entry']:+.3f}" if r["ev_at_entry"] is not None else "  —"
        print(f"  #{r['id']} {t} end_pct={r['end_pct']:.2f} 押{r['direction']} {eqs} "
              f"结算={r['settle_outcome']} {'赢' if r['win'] else '输'} EV {evs} ({r['entry_quote_kind']})")
    print()

print("=" * 78)
print("四、横向对比小结")
print("=" * 78)
print(f"盈亏平衡胜率 {BREAKEVEN:.1%}（费2%+溢0.01）")
print("线上场景信号 EV 口径：入场快照报价（entry_*_price_15m），无报价的 legacy/旧信号不计入 EV。")
print("影子 EV 口径：服务端落库 ev_at_entry（决策点/首中报价，同回测公式）。")
