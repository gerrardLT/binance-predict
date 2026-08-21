"""线上影子检测器完整解析：胜率曲线 × BTC K 线 × 周期切分（2026-08-21）。

数据源：output/online_shadow_now_2*.json（最新快照）+ Binance 公共 K 线。
回测基准（同源冻结口径，与 local_online_shadow_compare.py 一致）：
    X4 错位         合并 63.5%（IS 65.6/OOS 57.8） EV+0.254 CI(0.038,0.493)
    A 顺势 momentum  79.9% EV+0.097
    B 逆势 contrarian 24.0% EV+0.155（赔率型：胜率低但赔率高）
EV 口径（服务端落库 ev_at_entry，2026-08-21 已逐笔交叉验证零误差；各版本与
其回测冻结口径一致，勿混用）：
    x4_v1               赢 0.98/(q+0.01)−1（截断[0.01,0.99]）/ 输 −1（费2%+溢0.01，
                        与 local_misalignment_scan.py ev_eval 同口径）
    quote_momentum_v1   赢 0.98/q−1 / 输 −1（费2% 无溢价，
                        与 local_quote_bin_winrate.py 同口径）
    quote_contrarian_v1 赢 0.98/q−1 / 输 −1（费2% 无溢价，同上）
盈亏平衡胜率随 q 变化：be = (q+0.01)/0.98（x4）或 q/0.98（其余），按各自实际样本计。
周期切分：08-19 00:00 UTC 为大涨分界（此前 62-64k 震荡，此后三根大阳 +17.5%）。
报告同时落盘 output/shadow_full_analysis_report.txt（规避控制台编码问题）。
"""
import datetime as dt
import glob
import io
import json
import math
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

_report = io.StringIO()


def out(line=""):
    print(line)
    _report.write(line + "\n")


SHD = sorted(glob.glob("output/online_shadow_now_2*.json"))[-1]
S = json.load(open(SHD, encoding="utf-8"))
out(f"数据快照: {SHD}")

BENCH = {  # version -> (回测胜率, 回测EV, 说明)
    "x4_v1": (0.635, 0.254, "错位: 本窗收阳&end≤40 → 次窗 DOWN"),
    "quote_momentum_v1": (0.799, 0.097, "顺势: 深折价方向同窗押注"),
    "quote_contrarian_v1": (0.240, 0.155, "逆势: 赔率型，胜率低赔率高"),
}
LABEL = {"x4_v1": "X4 misalign→DOWN", "quote_momentum_v1": "A momentum",
         "quote_contrarian_v1": "B contrarian"}
COLORS = {"x4_v1": "#1f77b4", "quote_momentum_v1": "#d62728",
          "quote_contrarian_v1": "#2ca02c"}
BREAKEVEN = 0.5204  # 已废弃：仅 q=0.50 含溢价口径成立，改为按版本逐笔计算


def breakeven_of(v: str, q: float) -> float:
    """逐笔盈亏平衡胜率（与各版本 EV 口径一致）：x4 含溢 0.01，其余无溢价。"""
    return (q + 0.01) / 0.98 if v == "x4_v1" else q / 0.98


PUMP_TS = int(dt.datetime(2026, 8, 19, tzinfo=dt.timezone.utc).timestamp() * 1000)


def binom_p_le(n, k, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def days_span(ts_list):
    return max(1.0, (max(ts_list) - min(ts_list)) / 86_400_000)


# ---- 逐笔整理（仅 SETTLED）----
recs = []
for v, d in S["by_version"].items():
    for r in d["signals"]:
        if r["status"] != "SETTLED":
            continue
        q = r["entry_down_price"] if r["direction"] == "DOWN" else r["entry_up_price"]
        recs.append({
            "v": v, "ts": r["window_start"], "dir": r["direction"], "win": r["win"],
            "q": q, "ev": r.get("ev_at_entry"), "end_pct": r.get("end_pct"),
            "phase": "pump" if r["window_start"] >= PUMP_TS else "pre",
        })
recs.sort(key=lambda r: r["ts"])

out()
print("=" * 84)
out("=" * 84)
out("一、影子全景表（线上 vs 回测冻结基准）")
out("=" * 84)
out(f"{'版本':<24}{'n':>4}{'胜率':>8}{'回测':>8}{'偏离':>8}{'尾概率':>8}"
    f"{'盈亏平衡':>8}{'avgEV':>8}{'累计EV':>9}{'频率/日':>8}")
for v in ("x4_v1", "quote_momentum_v1", "quote_contrarian_v1"):
    g = [r for r in recs if r["v"] == v]
    if not g:
        continue
    bwr, bev, _ = BENCH[v]
    n, w = len(g), sum(r["win"] for r in g)
    wr = w / n
    evs = [r["ev"] for r in g if r["ev"] is not None]
    bes = [breakeven_of(v, r["q"]) for r in g if r["q"]]
    be = sum(bes) / len(bes) if bes else None
    tail = binom_p_le(n, w, bwr)
    freq = n / days_span([r["ts"] for r in g])
    out(f"{LABEL[v]:<26}{n:>4}{wr:>8.1%}{bwr:>8.1%}{wr-bwr:>+8.1%}{tail:>8.3f}"
        f"{(f'{be:.1%}' if be is not None else '—'):>8}"
        f"{(sum(evs)/len(evs) if evs else 0):>+8.3f}{(sum(evs) if evs else 0):>+9.2f}{freq:>8.1f}")

out()
out("=" * 84)
out(f"二、周期切分：大涨前（< 08-19 UTC）vs 大涨期（≥ 08-19，三根大阳 +17.5%）")
out("=" * 84)
out(f"{'版本':<24}{'阶段':<6}{'n':>4}{'胜率':>8}{'回测':>8}{'avgEV':>8}{'累计EV':>9}")
for v in ("x4_v1", "quote_momentum_v1", "quote_contrarian_v1"):
    bwr, bev, _ = BENCH[v]
    for ph, phn in (("pre", "震荡期"), ("pump", "大涨期")):
        g = [r for r in recs if r["v"] == v and r["phase"] == ph]
        if not g:
            continue
        n, w = len(g), sum(r["win"] for r in g)
        evs = [r["ev"] for r in g if r["ev"] is not None]
        out(f"{LABEL[v]:<26}{phn:<6}{n:>4}{w/n:>8.1%}{bwr:>8.1%}"
            f"{(sum(evs)/len(evs) if evs else 0):>+8.3f}{(sum(evs) if evs else 0):>+9.2f}")

# ---- 拉 BTC 行情 ----
lo = min(r["ts"] for r in recs) - 3_600_000
hi = max(r["ts"] for r in recs) + 6 * 3_600_000


def klines(interval, start, limit=1000):
    url = (f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT"
           f"&interval={interval}&startTime={start}&limit={limit}")
    return json.loads(urllib.request.urlopen(url, timeout=30).read().decode())


h1 = klines("1h", lo)
ht = [dt.datetime.fromtimestamp(b[0] / 1000, tz=dt.timezone.utc) for b in h1]
hc = [float(b[4]) for b in h1]
d0 = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=11)).replace(
    hour=0, minute=0, second=0, microsecond=0)
daily = klines("1d", int(d0.timestamp() * 1000))

out()
out("=" * 84)
out("三、BTC 日线背景（UTC）")
out("=" * 84)
for b in daily:
    t = dt.datetime.fromtimestamp(b[0] / 1000, tz=dt.timezone.utc).strftime("%m-%d")
    o, h, l, c = (float(b[1]), float(b[2]), float(b[3]), float(b[4]))
    mark = "  <<< 大涨分界" if b[0] == PUMP_TS else ""
    out(f"{t}  {o:.0f}→{c:.0f} {c/o-1:+.2%} {'阳' if c >= o else '阴'} 振幅{(h-l)/o:.2%}{mark}")

out()
out("=" * 84)
out("四、UTC 时段切分（4h 桶，全部影子合并）")
out("=" * 84)
buckets = {}
for r in recs:
    hh = (r["ts"] // 3_600_000) % 24
    buckets.setdefault(hh // 4 * 4, []).append(r)
for k in sorted(buckets):
    g = buckets[k]
    n, w = len(g), sum(r["win"] for r in g)
    out(f"UTC {k:02d}-{k+4:02d}h: {w}/{n} = {w/n:.1%}")

out()
out("=" * 84)
out("五、逐笔明细（按时间序）")
out("=" * 84)
for r in recs:
    t = dt.datetime.fromtimestamp(r["ts"] / 1000, tz=dt.timezone.utc).strftime("%m-%d %H:%M")
    evs = f"{r['ev']:+.2f}" if r["ev"] is not None else "  —"
    qs = f"q={r['q']:.2f}" if r["q"] else "q=—"
    out(f"{t}  {LABEL[r['v']]:<22} 押{r['dir']:<4} end={r['end_pct'] if r['end_pct'] is not None else '—':<5} "
        f"{qs} {'赢' if r['win'] else '输'} EV {evs}")

# ---- 图：累计胜率曲线 × BTC 走势 ----
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1.4]})
ax, ax2 = axes
ax2b = ax2.twinx()
ax2b.plot(ht, hc, color="#999", lw=1.2, label="BTC 1h close")
ax2b.set_ylabel("BTC (USDT)", color="#666")
for v in ("x4_v1", "quote_momentum_v1", "quote_contrarian_v1"):
    g = [r for r in recs if r["v"] == v]
    if len(g) < 2:
        continue
    bwr = BENCH[v][0]
    xs = [dt.datetime.fromtimestamp(r["ts"] / 1000, tz=dt.timezone.utc) for r in g]
    cum = [sum(r["win"] for r in g[:i + 1]) / (i + 1) for i in range(len(g))]
    bes = [breakeven_of(v, r["q"]) for r in g if r["q"]]
    be = sum(bes) / len(bes) if bes else 0.0
    ax.plot(xs, cum, color=COLORS[v], lw=2, marker="o", ms=4,
            label=f"{LABEL[v]}  n={len(g)} 终值{cum[-1]:.0%} 盈亏平衡{be:.0%}")
    ax.axhline(bwr, color=COLORS[v], ls="--", lw=0.8, alpha=0.5)
    ax.axhline(be, color=COLORS[v], ls=":", lw=0.9, alpha=0.45)
ax.axvline(dt.datetime.fromtimestamp(PUMP_TS / 1000, tz=dt.timezone.utc),
           color="orange", lw=1.5, ls="-.", alpha=0.8)
ax.text(dt.datetime.fromtimestamp(PUMP_TS / 1000, tz=dt.timezone.utc), 0.97,
        " 08-19 大涨分界", fontsize=9, color="darkorange", va="top")
ax.set_ylabel("累计胜率（实线）vs 回测基准（虚线 --）vs 盈亏平衡（点线 :，逐版本）")
ax.set_ylim(0, 1)
ax.legend(loc="upper left", fontsize=9)
ax.set_title("线上影子信号累计胜率曲线（misalignment_signals）× BTC 1h 走势 × 周期分界",
             fontsize=12)
ax.grid(alpha=0.3)
# 日线柱状背景（涨绿跌红）
for b in daily:
    t0 = dt.datetime.fromtimestamp(b[0] / 1000, tz=dt.timezone.utc)
    o, c = float(b[1]), float(b[4])
    ax2.axvspan(t0, t0 + dt.timedelta(days=1), alpha=0.10,
                color="green" if c >= o else "red")
    ax2.text(t0 + dt.timedelta(hours=6), 0.02, f"{c/o-1:+.1%}", fontsize=8, alpha=0.8)
# 各版本信号落点错开三行
ROWS = {"quote_momentum_v1": 0.75, "x4_v1": 0.5, "quote_contrarian_v1": 0.25}
for r in recs:
    t = dt.datetime.fromtimestamp(r["ts"] / 1000, tz=dt.timezone.utc)
    ax2.plot(t, ROWS[r["v"]], "o" if r["win"] else "x", color=COLORS[r["v"]],
             ms=5, alpha=0.9 if r["win"] else 0.55)
ax2.set_ylim(0, 1)
ax2.set_yticks([0.25, 0.5, 0.75])
ax2.set_yticklabels(["B contrarian", "X4", "A momentum"])
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax2.set_xlabel("日期（UTC）；背景色 = 当日 BTC 涨跌；橙线 = 08-19 大涨分界")
fig.tight_layout()
out_png = "output/shadow_signals_winrate_curve.png"
fig.savefig(out_png, dpi=110)
out(f"\n图已保存: {out_png}")

with open("output/shadow_full_analysis_report.txt", "w", encoding="utf-8") as f:
    f.write(_report.getvalue())
print("\n报告已保存: output/shadow_full_analysis_report.txt")
