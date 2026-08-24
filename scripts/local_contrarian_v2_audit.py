#!/usr/bin/env python3
"""quote_contrarian_v2 胜率/EV 真实性逐笔审计。

问题：contrarian_v2（B 格逆势，5m 窗 t∈[45,60)s DOWN 报价进入 [0.15,0.25) 押 DOWN，
门禁 = 触发时点 BTC 未高于开盘 ≥0.10%）的胜率 24.4% / EV +0.163 是否真实？

审计链（四层）：
  1. 算术层：线上逐笔复算 ev_at_entry vs 落库（应零误差，口径 0.98/q−1）
  2. 规则层：entry_quote_ts 必须落在窗口 [45,60)s；q 必须 ∈[0.15,0.25)；
     entry_up_price ≈ 1−q（双 token 报价一致性）
  3. 结构层：时间分段（防单段运气）、q 分布、盈亏来源（赢的是低 q 高赔率还是
     什么）、Wilson CI、隐含盈亏平衡胜率
  4. 实盘层：该通道 2026-08-22 起以 2U 实盘开启 → 拉真实订单对照：
     实际成交均价 vs 影子快照价（快照可实现性的直接检验）

数据：线上 API（http://165.154.147.155:8082）：
  /api/misalignment/signals?version=quote_contrarian_v2&limit=500（翻页全量）
  /api/trades/recent（实盘订单）
先验对照：quote_edge_detector.py 注释 v1 回测 24.0%/EV+0.155（local_quote_bin_winrate.py，
  真实报价样本 + K 线结算口径）。
"""
from __future__ import annotations

import datetime as dt
import json
import math
import sys
import urllib.request

BASE = "http://165.154.147.155:8082"
LOG = "output/contrarian_v2_audit.log"
PREM = 0.01


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


def wilson(k: int, n: int) -> tuple[float, float]:
    p = k / n
    z = 1.96
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def main() -> int:
    sys.stdout = Tee()

    # ---------- 拉全量 contrarian_v2（单次大 limit；API 不支持 offset 翻页） ----------
    d = get("/api/misalignment/signals?version=quote_contrarian_v2&limit=300")
    rows = d.get("signals", [])
    settled = [r for r in rows if r.get("settle_outcome") in ("UP", "DOWN")]
    settled.sort(key=lambda r: r["window_start"])
    print(f"contrarian_v2 拉取 {len(rows)} 行，其中可判定结算 {len(settled)} 行")
    if not settled:
        print("无数据，退出")
        return 1

    # ---------- 1+2. 算术层 & 规则层逐笔审计 ----------
    print()
    print("=" * 100)
    print("一、逐笔审计（算术层 + 规则层）：q 快照 / 结算 / EV 双口径 / 规则校验")
    print("=" * 100)
    n_err = 0        # ev 复算误差笔数
    n_win = 0
    ev_a_sum = ev_b_sum = 0.0
    q_list: list[float] = []
    rule_ts_bad = rule_q_bad = updown_bad = 0
    win_rows, lose_rows = [], []
    print(f"{'#':>3} {'UTC时间':<11} {'q快照':>6} {'UP价':>5} {'t_rel':>5} {'结算':>4} "
          f"{'win':>3} {'EV_A':>7} {'EV_B':>7}  校验")
    for i, r in enumerate(settled, 1):
        q = r["entry_down_price"]
        up = r.get("entry_up_price")
        ts = int(r["entry_quote_ts"]) if r.get("entry_quote_ts") else None
        ws = int(r["window_start"])
        win = r["settle_outcome"] == "DOWN"
        ev_a = (0.98 / q - 1.0) if win else -1.0
        ev_b = (0.98 / min(max(q + PREM, 0.01), 0.99) - 1.0) if win else -1.0
        # 校验
        ev_db = r.get("ev_at_entry")
        ok_ev = ev_db is not None and abs(ev_db - ev_a) < 1e-9
        t_rel = (ts - ws) / 1000.0 if ts else float("nan")
        ok_ts = 45.0 <= t_rel < 60.0 if ts else False
        ok_q = 0.15 <= q < 0.25
        ok_ud = up is not None and abs(up + q - 1.0) < 0.03
        flags = []
        if not ok_ev:
            flags.append("EV复算不符!")
            n_err += 1
        if not ok_ts:
            flags.append(f"t={t_rel:.0f}s越界!")
            rule_ts_bad += 1
        if not ok_q:
            flags.append(f"q={q:.2f}越界!")
            rule_q_bad += 1
        if not ok_ud:
            flags.append("UP+DOWN≠1!")
            updown_bad += 1
        n_win += win
        ev_a_sum += ev_a
        ev_b_sum += ev_b
        q_list.append(q)
        (win_rows if win else lose_rows).append((q, ev_a))
        if i <= 40 or flags:
            print(f"{i:>3} {dt.datetime.fromtimestamp(ws/1000, tz=dt.timezone.utc):%m-%d %H:%M}"
                  f" {q:>6.2f} {up if up is not None else -1:>5.2f} {t_rel:>5.0f}"
                  f" {r['settle_outcome']:>4} {'是' if win else '否':>3}"
                  f" {ev_a:>+7.3f} {ev_b:>+7.3f}  {' '.join(flags) if flags else 'ok'}")
    if len(settled) > 40:
        print(f"  ……（中间 {len(settled) - 40} 行省略，全量见日志首 40 行 + 异常行）")

    n = len(settled)
    print("-" * 100)
    print(f"合计: n={n} 胜={n_win} 胜率={n_win / n:.1%} | 复算误差 {n_err} 笔"
          f" | t 越界 {rule_ts_bad} | q 越界 {rule_q_bad} | UP+DOWN≠1 {updown_bad} 笔")
    print(f"逐笔均值: EV_A(无溢价,落库口径)={ev_a_sum / n:+.4f} / 注"
          f"  EV_B(含0.01溢价)={ev_b_sum / n:+.4f} / 注")
    qbar = sum(q_list) / n
    p_ = n_win / n
    lo, hi = wilson(n_win, n)
    print(f"q̄ 快照均价 = {qbar:.4f} → 隐含盈亏平衡胜率(费2%无溢价) = 0.51/q̄*… "
          f"即 q̄/{0.98:.2f} = {qbar / 0.98:.1%}")
    print(f"实际胜率 {p_:.1%} vs 盈亏平衡 {qbar / 0.98:.1%} → 边际 {p_ - qbar / 0.98:+.1%}pp")
    print(f"胜率 95%CI（Wilson）: [{lo:.1%}, {hi:.1%}]")

    # ---------- 3. 结构层 ----------
    print()
    print("=" * 100)
    print("二、结构层：时间分段 / q 分位 / 盈亏来源")
    print("=" * 100)
    # 前后半
    mid = settled[len(settled) // 2]["window_start"]
    for name, grp in (("前半", settled[:len(settled) // 2]), ("后半", settled[len(settled) // 2:])):
        w = sum(1 for r in grp if r["settle_outcome"] == "DOWN")
        ev = sum((0.98 / r["entry_down_price"] - 1.0) if r["settle_outcome"] == "DOWN" else -1.0
                 for r in grp) / len(grp)
        qb = sum(r["entry_down_price"] for r in grp) / len(grp)
        print(f"  {name}: n={len(grp):>4} 胜率={w / len(grp):.1%} q̄={qb:.3f} EV_A={ev:+.4f}")
    # 按天
    print("  按天：")
    by_day: dict[str, list] = {}
    for r in settled:
        key = dt.datetime.fromtimestamp(r["window_start"] / 1000, tz=dt.timezone.utc).strftime("%m-%d")
        by_day.setdefault(key, []).append(r)
    for key in sorted(by_day):
        grp = by_day[key]
        w = sum(1 for r in grp if r["settle_outcome"] == "DOWN")
        ev = sum((0.98 / r["entry_down_price"] - 1.0) if r["settle_outcome"] == "DOWN" else -1.0
                 for r in grp) / len(grp)
        print(f"    {key}: n={len(grp):>3} 胜率={w / len(grp):5.1%} EV={ev:+7.3f}"
              f"  {'█' * max(1, int((ev + 1) * 20))}")
    # 盈亏来源：赢的笔 q 分布
    print(f"  赢的 {len(win_rows)} 笔 q 分布：", end="")
    for lo_, hi_, name in ((0.15, 0.18, "0.15-0.18"), (0.18, 0.21, "0.18-0.21"),
                           (0.21, 0.25, "0.21-0.25")):
        cnt = sum(1 for q, _ in win_rows if lo_ <= q < hi_)
        print(f" {name}×{cnt}", end="")
    print()
    if win_rows:
        ev_win = sum(e for _, e in win_rows)
        print(f"  赢侧贡献 EV 总额 {ev_win:+.2f}（均值 {ev_win / n:+.3f}/注）"
              f" | 输侧 {len(lose_rows)} 笔贡献 {-len(lose_rows) / n:+.3f}/注")
        top = sorted(win_rows, key=lambda x: -x[1])[:5]
        print("  最大 5 笔赢单（EV 贡献）: " + ", ".join(f"q={q:.2f}→{e:+.2f}" for q, e in top))

    # ---------- 4. 实盘层 ----------
    print()
    print("=" * 100)
    print("三、实盘层：contrarian_v2 通道真实订单 vs 影子快照")
    print("=" * 100)
    try:
        tr = get("/api/trades/recent?limit=500")
        trades = tr.get("trades", tr.get("orders", []))
        cv2 = [t for t in trades if "contrarian" in str(t.get("channel", ""))]
        print(f"  订单端点返回 {len(trades)} 笔，其中 contrarian 通道 {len(cv2)} 笔")
        for t in cv2[:30]:
            print(f"    {t}")
        if not cv2:
            print("  （无 contrarian 实盘订单记录——若通道已开但未触发或未落单，则为空）")
    except Exception as ex:
        print(f"  （实盘订单拉取失败：{ex}）")

    # ---------- 5. 结论 ----------
    print()
    print("=" * 100)
    print("四、结论要点（对照回测先验 v1 24.0%/EV+0.155，v2=同区间+假冲高门禁）")
    print("=" * 100)
    print(f"  线上观测: n={n} 胜率={p_:.1%} EV_A={ev_a_sum / n:+.4f} EV_B={ev_b_sum / n:+.4f}")
    print(f"  先验(回测): v1 24.0% / +0.155（真实报价+K线结算，quote_edge_detector.py 注释）")
    print(f"  门禁效果: v2 归因依据 = 平盘窗(|chg|<0.05%)贡献 86% 利润、melt≥0.3% 段 wr 0~7%")
    print("  快照可实现性: entry=曲线 15s 采样 API 挂牌价，非成交回执；")
    print("    实际吃单按 EV_B（+0.01 溢价）估算更保守；深度档 q∈[0.15,0.25) 流动性需实盘验证。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
