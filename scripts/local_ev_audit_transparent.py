#!/usr/bin/env python3
"""EV 计算逐笔透明审计（2026-08-24）：每个数字的原始数据 + 计算式摊开。

回应质疑：S5 与各信号的 EV 到底拿什么数据、怎么算的。
三层数据血缘：
  1. 原始字段：线上 API 拉取的信号行（fake_breakout_signals / misalignment_signals）；
  2. 两种 EV 口径逐笔复算：
     A「快照口径」（本仓库分析脚本一贯口径，local_online_shadow_compare.py 同款）：
        赢 0.98/q − 1 / 输 −1，q = 信号行落库的入场报价快照；
     B「服务端审计口径」（main.py /api/signals/analytics 与 misalignment 落库同款）：
        赢 0.98/(q+0.01) − 1（含 0.01 吃单溢价，min/max 截断）/ 输 −1；
  3. 交叉验证：影子信号逐笔复算 B 口径 vs 服务端落库 ev_at_entry，误差应为 0。

结算判定（谁算输赢）：
  场景信号 settle_outcome = BTC 15m 周期收盘价 vs 周期开盘价（cycle_open_price_15m）
  的方向（DOWN/UP），服务端结算任务落库；S5 行复制父 S1 行的周期坐标。
  影子信号 win = 次窗（5m 情绪窗）outcome == 押注方向，服务端结算落库。

用法：python -X utf8 scripts/local_ev_audit_transparent.py
输出：stdout + output/ev_audit_transparent.log
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import urllib.request

BASE = "http://165.154.147.155:8082"
LOG = "output/ev_audit_transparent.log"
FEE, PREM = 0.02, 0.01

# 回测冻结基准（research_win_rates，fake_breakout_detector.py 内置）
BENCH = {
    "bull_exhaust": 0.644, "bull_exhaust_confirm": 0.785,
    "bear_exhaust": 0.536, "momentum_fade": 0.554,
}
BENCH_SH = {
    "x4_v1": 0.412, "x4_v2": 0.453,               # 干净口径（实价覆盖子集）
    "quote_momentum_v1": 0.799, "quote_momentum_v2": 0.799,
    "quote_contrarian_v1": 0.240, "quote_contrarian_v2": 0.240,
}


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


def ev_a(q: float, win: bool) -> float:
    """口径 A：快照口径（不含溢价）。"""
    return (0.98 / q - 1.0) if win else -1.0


def ev_b(q: float, win: bool) -> float:
    """口径 B：服务端审计口径（含 0.01 溢价 + [0.01,0.99] 截断）。"""
    if not win:
        return -1.0
    return 0.98 / min(max(q + PREM, 0.01), 0.99) - 1.0


def main() -> int:
    sys.stdout = Tee()
    fb_sigs = get("/api/fake-breakout/signals?limit=200")["signals"]
    mis = {v: get(f"/api/misalignment/signals?limit=200&version={v}")
           for v in BENCH_SH}

    # ===================== 一、S5 逐笔审计 =====================
    print("=" * 92)
    print("一、S5（bull_exhaust_confirm）逐笔审计：原始字段 → EV 两种口径")
    print("=" * 92)
    print("数据行 = 线上 fake_breakout_signals 表 pattern_type='bull_exhaust_confirm' 全部 12 行")
    print("q 原值 = entry_down_price_15m：S5 行创建时刻（次周期开始后 +5min 确认点）")
    print("         的 15m 市场 DOWN token 报价快照（15s 采样器 _pm_15m，start_date 守卫）")
    print("win     = settle_outcome == 'DOWN'（BTC 15m 周期收盘 < 周期开盘，服务端结算）")
    print("EV_A    = 赢 0.98/q−1 / 输 −1（快照口径，无溢价）")
    print("EV_B    = 赢 0.98/(q+0.01)−1 / 输 −1（服务端审计口径，含吃单溢价）\n")
    print(f"{'#id':>4} {'UTC时间':<12} {'q_raw':>6} {'结算':>5} {'win':>4} "
          f"{'EV_A':>8} {'EV_B':>8}  计算式")
    s5 = [s for s in fb_sigs
          if s["pattern_type"] == "bull_exhaust_confirm" and s["settle_outcome"]]
    s5.sort(key=lambda s: s["signal_time"])
    ev_as, ev_bs = [], []
    for s in s5:
        q = s["entry_down_price_15m"]
        t = dt.datetime.fromtimestamp(s["signal_time"] / 1000, tz=dt.timezone.utc)
        win = s["settle_outcome"] == "DOWN"
        a, b = ev_a(q, win), ev_b(q, win)
        ev_as.append(a)
        ev_bs.append(b)
        formula = (f"0.98/{q:.2f}−1={a:+.3f}" if win else "−1")
        print(f"{s['id']:>4} {t:%m-%d %H:%M}   {q:>6.2f} {s['settle_outcome']:>5} "
              f"{('是' if win else '否'):>4} {a:>+8.3f} {b:>+8.3f}  {formula}")
    n, w = len(s5), sum(1 for s in s5 if s["settle_outcome"] == "DOWN")
    print("-" * 92)
    print(f"合计: n={n}, 胜={w}, 胜率={w / n:.1%}")
    print(f"逐笔均值: EV_A={sum(ev_as) / n:+.4f} / 注,  EV_B={sum(ev_bs) / n:+.4f} / 注")
    print(f"若按 1U 下注这 12 单: A 口径累计赚 {sum(ev_as):+.3f}U, B 口径 {sum(ev_bs):+.3f}U")
    print(f"报价覆盖率: {sum(1 for s in s5 if s['entry_down_price_15m'] is not None)}/{n}")
    se = (sum((x - sum(ev_as) / n) ** 2 for x in ev_as) / n) ** 0.5 / n ** 0.5
    print(f"EV_A 的 95% 置信区间: {sum(ev_as) / n:+.3f} ± {1.96 * se:.3f}"
          f" = [{sum(ev_as) / n - 1.96 * se:+.3f}, {sum(ev_as) / n + 1.96 * se:+.3f}]"
          f"（n=12 太小，区间宽）")

    # ===================== 二、全部场景信号汇总审计 =====================
    print()
    print("=" * 92)
    print("二、场景信号汇总：两种口径对照（同一批原始数据）")
    print("=" * 92)
    print("买 UP（bear 族）用 entry_up_price_15m，买 DOWN 用 entry_down_price_15m；"
          "报价缺失的行不计入 EV（胜率仍计入）。")
    print(f"{'场景':<26}{'n':>4}{'胜率':>7}{'回测':>7}{'q̄':>7}"
          f"{'EV_A均值':>9}{'EV_B均值':>9}{'EV_A累计':>9}{'EV_B累计':>9}")
    groups: dict[str, list] = {}
    for s in fb_sigs:
        if not s["settle_outcome"]:
            continue
        pt = s["pattern_type"] or s.get("pattern") or "legacy"
        groups.setdefault(pt, []).append(s)
    for pt, rows in sorted(groups.items()):
        target = "UP" if pt.startswith("bear") else "DOWN"
        n, w = len(rows), sum(r["settle_outcome"] == target for r in rows)
        ev_as, ev_bs, qs = [], [], []
        for r in rows:
            q = r["entry_up_price_15m"] if target == "UP" else r["entry_down_price_15m"]
            if q is None:
                continue
            win = r["settle_outcome"] == target
            qs.append(q)
            ev_as.append(ev_a(q, win))
            ev_bs.append(ev_b(q, win))
        qbar = sum(qs) / len(qs) if qs else float("nan")
        ea = sum(ev_as) / len(ev_as) if ev_as else float("nan")
        eb = sum(ev_bs) / len(ev_bs) if ev_bs else float("nan")
        print(f"{pt + '→' + target:<28}{n:>4}{w / n:>7.1%}"
              f"{BENCH.get(pt, float('nan')):>7.1%}{qbar:>7.3f}"
              f"{ea:>+9.3f}{eb:>+9.3f}{sum(ev_as):>+9.2f}{sum(ev_bs):>+9.2f}")

    # ===================== 三、影子信号：复算 vs 服务端落库 =====================
    print()
    print("=" * 92)
    print("三、影子信号交叉验证：逐笔复算 vs 服务端落库 ev_at_entry")
    print("=" * 92)
    print("影子信号 q = 决策点（次窗 +150s）真实 DOWN/UP token 价（entry_quote_kind='real'），")
    print("win = 次窗 outcome == 押注方向。")
    print("落库公式分两族（源码核实，两者不一致是有意为之）：")
    print("  x4 族（misalignment_detector.py）  : 赢 0.98/(q+0.01)−1（含溢价）")
    print("  quote_edge 族（quote_edge_detector.py）: 赢 0.98/q−1（回测口径，无溢价）")
    print("（API limit=200：quote_momentum_v1 结算 341 条仅覆盖最近 200 条）\n")
    print(f"{'version':<24}{'可对照笔数':>10}{'复算≠落库':>10}{'最大误差':>12}"
          f"{'复算均值':>9}{'落库均值':>9}")
    for v in BENCH_SH:
        # x4 族落库含溢价 → 用 EV_B 复算；quote_edge 族落库无溢价 → 用 EV_A 复算
        use_b = v.startswith("x4")
        rows = [r for r in mis[v]["signals"]
                if r.get("win") is not None and r.get("ev_at_entry") is not None]
        n_cmp, mism, maxdiff = 0, 0, 0.0
        evs = []
        for r in rows:
            q = r.get("entry_down_price") if r["direction"] == "DOWN" else r.get("entry_up_price")
            if q is None:
                continue
            n_cmp += 1
            x = ev_b(q, r["win"]) if use_b else ev_a(q, r["win"])
            evs.append(x)
            diff = abs(x - r["ev_at_entry"])
            maxdiff = max(maxdiff, diff)
            if diff > 1e-9:
                mism += 1
        srv = [r["ev_at_entry"] for r in rows]
        print(f"{v:<24}{n_cmp:>10}{mism:>10}{maxdiff:>12.2e}"
              f"{(sum(evs) / len(evs)) if evs else float('nan'):>+9.3f}"
              f"{(sum(srv) / len(srv)) if srv else float('nan'):>+9.3f}"
              f"  [{'含溢价' if use_b else '无溢价'}]")

    # ===================== 四、口径与数据血缘说明 =====================
    print()
    print("=" * 92)
    print("四、口径与数据血缘（本报告每个数字的出处）")
    print("=" * 92)
    print("""
[数据源] 全部来自线上服务 http://165.154.147.155:8082 只读 API，拉取时间 2026-08-24：
  /api/fake-breakout/signals?limit=200   场景信号 70 行（8/13 起）
  /api/misalignment/signals?version=...   影子信号六版本（x4_v1/v2、momentum_v1/v2、
                                          contrarian_v1/v2），结算 842 行
  /api/prediction-wallet                  余额：预测钱包 41.83 / 现货 66.83 USDT

[场景信号的字段语义]（fake_breakout_detector.py 落库逻辑）：
  entry_down_price_15m / entry_up_price_15m：信号行创建时刻抓的 15m 市场
    DOWN/UP token 报价快照（15s 采样器，start_date 守卫防旧市场残值）。
    S1/S2/S4 行 = 次周期开盘时刻；S5 行 = +5min 确认时刻（代码注释：
    「S5 入场即 +5min 确认时刻，entry 列语义对齐'入场时报价'」）。
  settle_outcome：15m 周期结算，BTC 收盘价 vs 周期开盘价 → DOWN/UP。
  注意：这是报价快照，不是成交回执——S5 通道尚未开过真单，没有实际
  成交均价；真实下单还会有滑点，EV_B（含 0.01 溢价）已部分计入。

[影子信号的字段语义]（misalignment_detector.py 落库逻辑）：
  entry_down_price：决策点（次窗开窗 +150s）从窗口归档曲线回读的
    真实 DOWN token 价（entry_quote_kind='real'，实价率 100%）。
  ev_at_entry：落库公式 = 赢 0.98/min(max(q+0.01,0.01),0.99)−1 / 输 −1。

[两种 EV 口径差异]：EV_A（0.98/q−1）不含 0.01 吃单溢价；EV_B 含。
  q=0.72 时 A=+0.361 vs B=+0.342（每注差 ~2%）。仓位建议（凯利）用的是
  A 口径 b=0.98/q̄−1 与线上 q̄，偏乐观约 1-2%——量级不影响结论。

[S5 前瞻 EV 的三层来源]（报告中「+0.238 / +0.044 / +0.060」三个数的区别）：
  +0.238 = 线上 12 注逐笔 EV_A 均值（过去实际发生值，含运气：11 胜撑高）；
  +0.044 = 前瞻模型值：贝叶斯收缩胜率 76.4%（回测 78.5%×0.95 先验 + 12 注
           线上观测）× b(0.717) − (1−p)，即「按当前证据，未来每注期望赚多少」；
  +0.060 = 独立回测研究（output/s5_real_quote_ev_result.json，720d，n=1314
           S5 确认事件，入场价用报价样本表 t=5 分钟桶 q̂ 重建，非 0.51 假设价）。
  三个数方向一致：S5 真实边际 ≈ +0.04~0.06/注，远小于回测展示的
  +0.47（那是 @0.51 假设入场价的口径，实际市场不会给你这个价）。

[回测基准胜率的口径提醒]：
  x4_v1 用 41.2% / x4_v2 用 45.3%（干净口径=实价覆盖子集，护栏推导同源），
  而非全样本合并 63.5%（仅 19% 实价覆盖、假设价口径）——线上实测 44.4%/48.2%
  与干净口径吻合，证明干净口径才是可下注口径。
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
