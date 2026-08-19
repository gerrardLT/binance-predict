"""线上实时统计 vs 回测基准对比（2026-08-18）。

线上 24 条信号（8/13 起）按场景对比 research_win_rates（720d 回测基准），
二项检验判断偏离显著性；S5 口径（t=5 回落确认 + q5m 报价入场）单独模拟 EV。
"""

import json
import math

D = json.load(open("output/online_signals_now.json", encoding="utf-8"))
sigs = D["signals"]

# research_win_rates（回测基准，720d，detector 内置）
BENCH = {"bull_exhaust": 0.644, "bear_exhaust": 0.536,
         "momentum_fade": 0.554, "bull_exhaust_confirm": 0.785}


def binom_p_le(n, k, p):
    """P(X <= k | Bin(n, p)) 尾概率（线上偏低方向的检验）。"""
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(0, k + 1))


print("=" * 72)
print("一、按场景：线上实时 vs 回测基准")
print("=" * 72)
print(f"{'场景':<22}{'线上':>10}{'回测':>8}{'偏离':>8}{'尾概率P(X<=k)':>14}")
groups = {}
for s in sigs:
    if s["settle_outcome"] is None:
        continue
    # 分组键与 stats 端点对齐：pattern_type（场景版本）→ pattern（老信号）→ legacy
    pt = s["pattern_type"] or s.get("pattern") or "legacy"
    # 结算方向：bear 族（低价跌破）买 UP 反弹，其余（bull/momentum）买 DOWN 回落
    target = "UP" if pt.startswith("bear") else "DOWN"
    g = groups.setdefault(pt, [0, 0, target])
    g[0] += 1
    g[1] += 1 if s["settle_outcome"] == target else 0
for pt, (n, w, target) in sorted(groups.items()):
    wr = w / n
    b = BENCH.get(pt)
    label = f"{pt}→买{target}"
    if b is None:
        print(f"{label:<26}{w}/{n} = {wr:.1%}{('—'):>8}{'':>8}")
    else:
        tail = binom_p_le(n, w, b)
        print(f"{label:<26}{w}/{n} = {wr:.1%}{b:>8.1%}{wr-b:>+8.1%}{tail:>14.3f}")

# 合并 S1 口径（bull_exhaust 全体 + confirm 子集：高价突破回落族）
s1_all = [s for s in sigs if s["settle_outcome"]
          and (s.get("pattern") == "bull_exhaust" or s["pattern_type"] in ("bull_exhaust", "bull_exhaust_confirm"))]
n, w = len(s1_all), sum(1 for s in s1_all if s["settle_outcome"] == "DOWN")
print(f"{'S1合并(high回落族)':<22}{w}/{n} = {w/n:.1%}{0.644:>8.1%}{w/n-0.644:>+8.1%}{binom_p_le(n, w, 0.644):>14.3f}")

print()
print("=" * 72)
print("二、S5 口径模拟：t=5 报价买入 DOWN 的逐笔 EV（q5m 有值的信号）")
print("=" * 72)
print(f"{'id':>4}{'场景':<22}{'q5m_down':>10}{'结算':>7}{'单位EV':>10}")
ev_rows = []
for s in sigs:
    q = s.get("quote5m_down_15m")
    if q is None or s["settle_outcome"] is None:
        continue
    win = s["settle_outcome"] == "DOWN"
    ev = (0.98 / q - 1) if win else -1.0
    ev_rows.append(ev)
    note = "← 未回落(q0.11市场币DOWN仅11%)，S5严格口径应放弃" if q < 0.2 else ""
    print(f"{s['id']:>4}{(s['pattern_type'] or s.get('pattern') or '?'):<22}{q:>10.2f}{s['settle_outcome']:>7}{ev:>+10.3f} {note}")
print(f"{'合计':>26}{'n=%d' % len(ev_rows):>12}{'mean EV':>10}{sum(ev_rows)/len(ev_rows):>+10.3f}"
      f"  median={sorted(ev_rows)[len(ev_rows)//2]:+.3f}"
      if ev_rows else "无样本")
print()
print("（回测基准：S5 t=5 确认 P=76.5% q=0.686 → EV=+0.104；n=1314, 720d）")

print()
print("=" * 72)
print("三、全体口径")
print("=" * 72)
st = [s for s in sigs if s["settle_outcome"]]
nw = sum(1 for s in st if s["settle_outcome"] == "DOWN")
print(f"全体 DOWN 结算比例: {nw}/{len(st)} = {nw/len(st):.1%}（含 bear_exhaust/legacy 等非 S1 场景，")
print(f"回测可比基准应按场景拆分；全体口径无单一回测对应值）")
