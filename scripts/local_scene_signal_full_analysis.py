"""线上场景信号（FakeBreakout）完整解析：胜率曲线 × BTC K 线 × 周期切分（2026-08-21）。

数据源：output/online_signals_now_2*.json（最新快照）+ Binance 公共 K 线。
方向映射（与 local_online_shadow_compare.py 同口径）：
    bear_exhaust → 买 UP（反弹）；bull_exhaust / momentum_fade / bull_exhaust_confirm → 买 DOWN。
EV 口径：入场快照报价（entry_*_price_15m），赢 0.98/(q+0.01)−1 / 输 −1
    （费 2%+溢 0.01，与 backtest/stats.py ev() 回测冻结口径逐字段一致；2026-08-21 审计修正）。
"""
import datetime as dt
import glob
import json
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

D = json.load(open(sorted(glob.glob("output/online_signals_now_2*.json"))[-1], encoding="utf-8"))
sigs = [s for s in D["signals"] if s["settle_outcome"]]
BENCH = D["stats"]["research_win_rates"]
TARGET = {"bear_exhaust": "UP", "bull_exhaust": "DOWN",
          "momentum_fade": "DOWN", "bull_exhaust_confirm": "DOWN"}
LABEL = {"bull_exhaust": "S1 bull_exhaust→DOWN", "bear_exhaust": "S2 bear_exhaust→UP",
         "momentum_fade": "S4 momentum_fade→DOWN",
         "bull_exhaust_confirm": "S5 confirm→DOWN", None: "legacy→DOWN"}

# ---- 逐笔整理 ----
recs = []
for s in sigs:
    pt = s["pattern_type"] or None
    t = TARGET.get(pt, "DOWN")
    q = s["entry_up_price_15m"] if t == "UP" else s["entry_down_price_15m"]
    win = s["settle_outcome"] == t
    recs.append({
        "ts": s["signal_time"], "pt": pt, "target": t, "win": win, "q": q,
        "ev": ((0.98 / min(max(q + 0.01, 0.01), 0.99) - 1) if win else -1.0) if q else None,
        "close_pos": s.get("close_pos"), "vol_ratio": s.get("vol_ratio"),
        "btc": s.get("btc_price"),
    })
recs.sort(key=lambda r: r["ts"])

# ---- 拉 BTC 行情：信号区间 1h 线 + 最近 10 根日线 ----
lo = min(r["ts"] for r in recs) - 3_600_000
hi = max(r["ts"] for r in recs) + 3_600_000


def klines(interval, start, limit=1000):
    url = (f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT"
           f"&interval={interval}&startTime={start}&limit={limit}")
    return json.loads(urllib.request.urlopen(url, timeout=30).read().decode())


h1 = klines("1h", lo)
ht = [dt.datetime.fromtimestamp(b[0] / 1000, tz=dt.timezone.utc) for b in h1]
hc = [float(b[4]) for b in h1]
d0 = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=10)).replace(
    hour=0, minute=0, second=0, microsecond=0)
daily = klines("1d", int(d0.timestamp() * 1000))

print("=" * 84)
print("一、场景全景表（线上 vs 360 天回测）")
print("=" * 84)
print(f"{'场景':<26}{'n':>4}{'胜率':>8}{'回测':>8}{'偏离':>8}{'avgEV':>8}{'累计EV':>9}{'有报价':>7}")
for pt in ("bull_exhaust", "bull_exhaust_confirm", "momentum_fade", "bear_exhaust", None):
    g = [r for r in recs if r["pt"] == pt]
    if not g:
        continue
    n, w = len(g), sum(r["win"] for r in g)
    evs = [r["ev"] for r in g if r["ev"] is not None]
    b = BENCH.get(pt)
    wr = w / n
    print(f"{LABEL[pt]:<28}{n:>4}{wr:>8.1%}"
          f"{(f'{b:.1%}' if b else '—'):>8}{(f'{wr-b:+.1%}' if b else '—'):>8}"
          f"{(f'{sum(evs)/len(evs):+.3f}' if evs else '—'):>8}"
          f"{(f'{sum(evs):+.2f}' if evs else '—'):>9}{f'{len(evs)}/{n}':>7}")

print()
print("=" * 84)
print("二、BTC 日线背景（UTC）")
print("=" * 84)
for b in daily:
    t = dt.datetime.fromtimestamp(b[0] / 1000, tz=dt.timezone.utc).strftime("%m-%d")
    o, h, l, c = (float(b[1]), float(b[2]), float(b[3]), float(b[4]))
    print(f"{t}  {o:.0f}→{c:.0f} {c/o-1:+.2%} {'阳' if c>=o else '阴'} 振幅{(h-l)/o:.2%}")

print()
print("=" * 84)
print("三、UTC 时段切分（4h 桶）")
print("=" * 84)
buckets = {}
for r in recs:
    hh = (r["ts"] // 3_600_000) % 24
    buckets.setdefault(hh // 4 * 4, []).append(r)
for k in sorted(buckets):
    g = buckets[k]
    n, w = len(g), sum(r["win"] for r in g)
    print(f"UTC {k:02d}-{k+4:02d}h: {w}/{n} = {w/n:.1%}")

print()
print("=" * 84)
print("四、逐笔明细（按时间序）")
print("=" * 84)
for r in recs:
    t = dt.datetime.fromtimestamp(r["ts"] / 1000, tz=dt.timezone.utc).strftime("%m-%d %H:%M")
    evs = f"{r['ev']:+.2f}" if r["ev"] is not None else "  —"
    qs = f"q={r['q']:.2f}" if r["q"] else "q=—"
    print(f"{t}  {LABEL[r['pt']]:<28} 押{r['target']:<4} {qs} "
          f"{'赢' if r['win'] else '输'} EV {evs}")

# ---- 图：累计胜率曲线 × BTC 走势 ----
COLORS = {"bull_exhaust": "#1f77b4", "bull_exhaust_confirm": "#9467bd",
          "momentum_fade": "#2ca02c", "bear_exhaust": "#d62728"}
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1]})
ax, ax2 = axes
ax2b = ax2.twinx()
ax2b.plot(ht, hc, color="#999", lw=1.2, label="BTC 1h close")
ax2b.set_ylabel("BTC (USDT)", color="#666")
for pt, col in COLORS.items():
    g = [r for r in recs if r["pt"] == pt]
    if len(g) < 3:
        continue
    xs = [dt.datetime.fromtimestamp(r["ts"] / 1000, tz=dt.timezone.utc) for r in g]
    cum = [sum(r["win"] for r in g[:i + 1]) / (i + 1) for i in range(len(g))]
    ax.plot(xs, cum, color=col, lw=2, marker="o", ms=4,
            label=f"{LABEL[pt]}  n={len(g)} 终值{cum[-1]:.0%}")
    ax.axhline(BENCH[pt], color=col, ls="--", lw=0.8, alpha=0.5)
ax.axhline(0.5204, color="k", ls=":", lw=1, alpha=0.6)
ax.text(xs[0], 0.525, "盈亏平衡 52.0%（q≈0.50）", fontsize=9, alpha=0.7)
ax.set_ylabel("累计胜率（实线）vs 回测基准（虚线）")
ax.set_ylim(0, 1)
ax.legend(loc="lower left", fontsize=9)
ax.set_title("线上场景信号累计胜率曲线（2026-08-15 ~ 08-21）× BTC 1h 走势", fontsize=12)
ax.grid(alpha=0.3)
# 日线柱状背景（涨绿跌红）
for b in daily:
    t0 = dt.datetime.fromtimestamp(b[0] / 1000, tz=dt.timezone.utc)
    o, c = float(b[1]), float(b[4])
    ax2.axvspan(t0, t0 + dt.timedelta(days=1), alpha=0.10,
                color="green" if c >= o else "red")
    ax2.text(t0 + dt.timedelta(hours=6), 0.03, f"{c/o-1:+.1%}", fontsize=8, alpha=0.8)
ax2.set_ylabel("信号落点（赢=●赢色 / 输=×）")
for r in recs:
    if r["pt"] is None:
        continue
    t = dt.datetime.fromtimestamp(r["ts"] / 1000, tz=dt.timezone.utc)
    ax2.plot(t, 0.5, "o" if r["win"] else "x", color=COLORS[r["pt"]], ms=6,
             alpha=0.9 if r["win"] else 0.6)
ax2.set_ylim(0, 1)
ax2.set_yticks([0.5])
ax2.set_yticklabels(["信号"])
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax2.set_xlabel("日期（UTC）；背景色 = 当日 BTC 涨跌")
fig.tight_layout()
out = "output/scene_signals_winrate_curve.png"
fig.savefig(out, dpi=110)
print(f"\n图已保存: {out}")
