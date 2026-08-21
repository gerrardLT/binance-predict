"""quote_momentum_v1 线上影子 56 笔 regime 归因（2026-08-21）。

背景：线上胜率 62.5% 显著低于回测 79.9%（尾概率 0.002），avgEV -0.140。
本脚本拉 Binance 1m K 线，为每笔窗口计算前置行情特征，按胜负切分找失效 regime。
特征均为窗口开始前的信息（t 时刻特征不用 t+1 信息）。
"""
import glob
import json
import math
import statistics
import urllib.request

SHD = sorted(glob.glob("output/online_shadow_now_2*.json"))[-1]
rows = [r for r in json.load(open(SHD, encoding="utf-8"))["by_version"]
        ["quote_momentum_v1"]["signals"] if r["status"] == "SETTLED"]
rows.sort(key=lambda r: r["window_start"])
print(f"样本: {len(rows)} 笔 settled | 胜 {sum(r['win'] for r in rows)} "
      f"({sum(r['win'] for r in rows)/len(rows):.1%})")

# ---- 拉 1m K 线（最早窗口前 70min ~ 最晚窗口结束）----
lo = min(r["window_start"] for r in rows) - 70 * 60_000
hi = max(r["window_end"] for r in rows) + 5 * 60_000
kl = []
cur = lo
while cur < hi:
    url = ("https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m"
           f"&startTime={cur}&endTime={hi}&limit=1000")
    with urllib.request.urlopen(url, timeout=30) as resp:
        batch = json.loads(resp.read().decode())
    if not batch:
        break
    kl.extend(batch)
    cur = batch[-1][0] + 60_000
    if len(batch) < 1000:
        break
bars = {b[0]: b for b in kl}  # open_time_ms -> [o,h,l,c,...]
print(f"1m K线: {len(kl)} 根 "
      f"覆盖 {len(kl)*60_000/3_600_000:.1f}h")


def price_at(ts_ms):
    """ts 所在 1m bar 的 open（近似 ts 时刻价格，避免用未来信息）"""
    b = bars.get(ts_ms - ts_ms % 60_000)
    return float(b[1]) if b else None


# ---- 逐笔特征 ----
recs = []
for r in rows:
    ws = r["window_start"]
    p_now = price_at(ws)
    p_1h = price_at(ws - 3_600_000)
    p_30 = price_at(ws - 1_800_000)
    p_prev5_open = price_at(ws - 300_000)
    p_entry = price_at(r["entry_quote_ts"])
    # 前 30min 1m 收益率 std（bps）
    rets = []
    for k in range(30):
        t = ws - (30 - k) * 60_000
        b0, b1 = bars.get(t - 60_000), bars.get(t)
        if b0 and b1:
            rets.append(float(b1[4]) / float(b0[4]) - 1)
    vol30 = statistics.stdev(rets) * 10_000 if len(rets) >= 20 else None
    # 前 1h 区间位置
    hh = max(float(b[2]) for t, b in bars.items() if ws - 3_600_000 <= t < ws)
    ll = min(float(b[3]) for t, b in bars.items() if ws - 3_600_000 <= t < ws)
    recs.append({
        "id": r["id"], "win": r["win"], "q": r["entry_down_price"],
        "utc_h": (ws // 3_600_000) % 24,
        "ret1h": (p_now / p_1h - 1) if p_now and p_1h else None,
        "ret30m": (p_now / p_30 - 1) if p_now and p_30 else None,
        "prev5_ret": (p_now / p_prev5_open - 1) if p_now and p_prev5_open else None,
        "vol30": vol30,
        "range_pos": (p_now - ll) / (hh - ll) if hh > ll else None,
        "entry_move": (p_entry / p_now - 1) if p_entry and p_now else None,
    })
recs = [x for x in recs if None not in (x["ret1h"], x["vol30"], x["range_pos"],
                                        x["prev5_ret"], x["entry_move"])]
print(f"特征齐全: {len(recs)} 笔\n")

W = [x for x in recs if x["win"]]
L = [x for x in recs if not x["win"]]


def dist(vals):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return "—"
    return (f"min={vals[0]:+.4f} p25={vals[len(vals)//4]:+.4f} "
            f"med={vals[len(vals)//2]:+.4f} p75={vals[3*len(vals)//4]:+.4f} "
            f"max={vals[-1]:+.4f}")


print("=" * 76)
print("一、连续特征分布（胜 vs 负）")
print("=" * 76)
for name, key, fmt in [
    ("前1h收益", "ret1h", "{:+.3%}"), ("前30m收益", "ret30m", "{:+.3%}"),
    ("前5m收益", "prev5_ret", "{:+.3%}"), ("前30m波动bps", "vol30", "{:.1f}"),
    ("1h区间位置", "range_pos", "{:.2f}"),
    ("入场前90s走势", "entry_move", "{:+.3%}"),
]:
    print(f"\n{name}:")
    print(f"  胜({len(W)}): " + dist([x[key] for x in W]))
    print(f"  负({len(L)}): " + dist([x[key] for x in L]))

print()
print("=" * 76)
print("二、二分切桶胜率（以中位数为界；检验：桶内二项尾概率 vs 总体 62.5%）")
print("=" * 76)
base_wr = len(W) / len(recs)


def bucket_report(name, key, thr=None):
    vals = sorted(x[key] for x in recs)
    thr = thr if thr is not None else vals[len(vals) // 2]
    lo_g = [x for x in recs if x[key] <= thr]
    hi_g = [x for x in recs if x[key] > thr]
    out = []
    for label, g in ((f"低({name}<={thr:.4f})" if isinstance(thr, float) else f"低", lo_g),
                     ("高", hi_g)):
        n, w = len(g), sum(x["win"] for x in g)
        # 桶胜率 vs 总体 base_wr 的双侧近似：取两侧尾概率较小者×2
        p_lo = sum(math.comb(n, i) * base_wr**i * (1-base_wr)**(n-i)
                   for i in range(w + 1))
        p_hi = sum(math.comb(n, i) * base_wr**i * (1-base_wr)**(n-i)
                   for i in range(w, n + 1))
        p = min(1.0, 2 * min(p_lo, p_hi))
        out.append(f"{label}: {w}/{n} = {w/n:.1%} (p={p:.2f})")
    print(f"{name:<12}{' | '.join(out)}")


for name, key in [("前1h收益", "ret1h"), ("前30m收益", "ret30m"),
                  ("前5m收益", "prev5_ret"), ("前30m波动", "vol30"),
                  ("区间位置", "range_pos"), ("入场前90s", "entry_move")]:
    bucket_report(name, key)

# UTC 时段：亚洲早盘 vs 欧美盘
print()
asia = [x for x in recs if 0 <= x["utc_h"] < 8]
west = [x for x in recs if x["utc_h"] >= 8]
for label, g in (("UTC 0-8h", asia), ("UTC 8-24h", west)):
    n, w = len(g), sum(x["win"] for x in g)
    print(f"{label:<12}{w}/{n} = {w/n:.1%}" if n else f"{label}: n=0")

print()
print("=" * 76)
print("三、时间序列：每 6 小时一段（看是否随时间劣化）")
print("=" * 76)
t0 = recs[0]["id"]
seg = {}
for i, x in enumerate(recs):
    seg.setdefault(i // 15, []).append(x)
import datetime as dt
for k in sorted(seg):
    g = seg[k]
    n, w = len(g), sum(x["win"] for x in g)
    ts = dt.datetime.fromtimestamp(rows[0]["window_start"] / 1000, tz=dt.timezone.utc)
    print(f"段{k+1} (#{g[0]['id']}-#{g[-1]['id']}): {w}/{n} = {w/n:.1%}")

print()
print("=" * 76)
print("四、组合切分：波动 × 前1h方向")
print("=" * 76)
vol_med = sorted(x["vol30"] for x in recs)[len(recs) // 2]
for vlabel, vcond in (("低波动", lambda x: x["vol30"] <= vol_med),
                      ("高波动", lambda x: x["vol30"] > vol_med)):
    for tlabel, tcond in (("前1h下跌", lambda x: x["ret1h"] <= 0),
                          ("前1h上涨", lambda x: x["ret1h"] > 0)):
        g = [x for x in recs if vcond(x) and tcond(x)]
        if not g:
            continue
        n, w = len(g), sum(x["win"] for x in g)
        print(f"{vlabel}+{tlabel}: {w}/{n} = {w/n:.1%}")
