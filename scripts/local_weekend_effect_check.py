"""周末效应检验（2026-08-19）：用户假设"BTC 周末波动小"。

A. 720d 全市场：4h K 线按 UTC 星期分组，对比平均振幅/绝对收益（检验假设本身）
B. S1 事件（2294 窗口）：按信号落点星期分组，对比 胜率P(DOWN) / 窗口波动
C. 结论：回测是否该剔除周末 / 周末是否需要单独模式
"""

import json
import statistics
import time
import urllib.request
from collections import defaultdict

WEEK_CN = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

# ============================================================
# A. 720d 4h K 线（Binance 公开 API，5 次请求）
# ============================================================
print("=" * 78)
print("A. 全市场波动：720 天 4h K 线按星期分组（UTC）")
print("=" * 78)

kl4h = []
end_ms = int(time.time() * 1000)
cur = end_ms - 720 * 86_400_000
while cur < end_ms:
    url = (f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=4h"
           f"&startTime={cur}&limit=1000")
    with urllib.request.urlopen(url, timeout=30) as r:
        batch = json.loads(r.read())
    if not batch:
        break
    kl4h.extend(batch)
    cur = batch[-1][6] + 1  # 最后一根的 close_time + 1
    time.sleep(0.15)

amp_by_dow = defaultdict(list)
ret_by_dow = defaultdict(list)
for k in kl4h:
    ts, o, h, l, c = int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4])
    dow = time.gmtime(ts / 1000).tm_wday
    amp_by_dow[dow].append((h - l) / o * 10000)   # 振幅 bp
    ret_by_dow[dow].append(abs(c - o) / o * 10000)

print(f"4h K 总数: {len(kl4h)}（约 {len(kl4h)/6:.0f} 天）")
print(f"{'星期':<6}{'4h根数':>8}{'平均振幅bp':>12}{'平均|收益|bp':>14}")
wk_amp = wd_amp = None
for d in range(7):
    a = statistics.mean(amp_by_dow[d])
    r_ = statistics.mean(ret_by_dow[d])
    print(f"{WEEK_CN[d]:<6}{len(amp_by_dow[d]):>8}{a:>12.1f}{r_:>14.1f}")
wk = amp_by_dow[5] + amp_by_dow[6]
wd = [x for d in range(5) for x in amp_by_dow[d]]
wk_amp, wd_amp = statistics.mean(wk), statistics.mean(wd)
print(f"\n周末平均振幅 {wk_amp:.1f}bp vs 工作日 {wd_amp:.1f}bp → "
      f"周末{'小' if wk_amp < wd_amp else '大'} {abs(wk_amp-wd_amp)/wd_amp*100:.1f}%")

# ============================================================
# B. S1 事件 2294 窗口按星期分组
# ============================================================
print()
print("=" * 78)
print("B. S1 事件（2294 窗口）：按信号次周期起始的星期分组（UTC）")
print("=" * 78)

WIN = json.load(open("output/klines_1m_s1_windows.json", encoding="utf-8"))
ev_by_dow = defaultdict(list)
for key, klines in WIN.items():
    start = int(key)
    dow = time.gmtime(start / 1000).tm_wday
    o = float(klines[0][1])
    c = float(klines[-1][4])
    hi = max(float(k[2]) for k in klines)
    lo = min(float(k[3]) for k in klines)
    ev_by_dow[dow].append({
        "win": c < o,                       # S1 买 DOWN
        "amp": (hi - lo) / o * 10000,       # 16 分钟窗口振幅
    })

print(f"{'星期':<6}{'事件n':>8}{'P(DOWN)':>10}{'窗口振幅bp':>12}")
for d in range(7):
    evs = ev_by_dow[d]
    if not evs:
        continue
    p = sum(1 for e in evs if e["win"]) / len(evs)
    amp = statistics.mean(e["amp"] for e in evs)
    print(f"{WEEK_CN[d]:<6}{len(evs):>8}{p:>10.1%}{amp:>12.1f}")

ev_wk = [e for d in (5, 6) for e in ev_by_dow.get(d, [])]
ev_wd = [e for d in range(5) for e in ev_by_dow.get(d, [])]
p_wk = sum(1 for e in ev_wk if e["win"]) / len(ev_wk)
p_wd = sum(1 for e in ev_wd if e["win"]) / len(ev_wd)
a_wk = statistics.mean(e["amp"] for e in ev_wk)
a_wd = statistics.mean(e["amp"] for e in ev_wd)
print(f"\n周末事件 n={len(ev_wk)}  P(DOWN)={p_wk:.1%}  振幅={a_wk:.1f}bp")
print(f"工作日事件 n={len(ev_wd)}  P(DOWN)={p_wd:.1%}  振幅={a_wd:.1f}bp")
print(f"胜率差: {p_wk-p_wd:+.1%} | 振幅差: {(a_wk-a_wd)/a_wd*100:+.1f}%")

# 北京时间视角（周末感知差 8 小时）
print()
print("--- 北京时间口径（+8h 重算星期）---")
ev_cn = defaultdict(list)
for key, klines in WIN.items():
    start = int(key)
    dow = time.gmtime((start + 8 * 3600_000) / 1000).tm_wday
    o = float(klines[0][1])
    c = float(klines[-1][4])
    ev_cn[dow].append(c < o)
cn_wk = [x for d in (5, 6) for x in ev_cn.get(d, [])]
cn_wd = [x for d in range(5) for x in ev_cn.get(d, [])]
print(f"周末(北京) n={len(cn_wk)} P={sum(cn_wk)/len(cn_wk):.1%} | "
      f"工作日(北京) n={len(cn_wd)} P={sum(cn_wd)/len(cn_wd):.1%}")
