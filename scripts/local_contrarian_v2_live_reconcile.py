#!/usr/bin/env python3
"""contrarian 实盘订单 vs 影子记录逐笔对账（EV 可实现性的最终检验）。

背景：影子记录（快照价口径）无成交回执；08-23 15:35 起用户开了
quote_contrarian_v1 实盘（11 笔），08-24 起切 quote_contrarian_v2（2 笔）
→ trades 表有真实链上订单（average_price = 实际成交均价，pnl = 实际盈亏）。

重要口径发现（由赢单 pnl 反推）：
  id23 avg=0.15 pnl=+11.333 = 2/0.15−2 → 兑付 1:1 **无 2% 费**；
  而影子 EV_A 用 0.98 系数（保守假设）→ 影子口径低估实际 EV 约 2%/q。

对账内容：
  1. 实盘单逐笔：window_start ↔ 影子行对齐，比 average_price vs 影子 q、
     pnl/unit vs 影子 EV_A（0.98 口径）与 EV_C（1.0 口径）、结算一致性
  2. 通道运行期影子触发但无实盘单的窗口
  3. 汇总：实盘已实现 EV vs 影子同期 EV —— 快照口径的可实现性折扣
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import urllib.request

BASE = "http://165.154.147.155:8082"
LOG = "output/contrarian_v2_live_reconcile.log"


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


def main() -> int:
    sys.stdout = Tee()

    shadow = []
    for ver in ("quote_contrarian_v1", "quote_contrarian_v2"):
        shadow.extend(get(f"/api/misalignment/signals?version={ver}&limit=300")["signals"])
    settled_shadow = {int(r["window_start"]): r for r in shadow
                      if r.get("settle_outcome") in ("UP", "DOWN")}
    tr_resp = get("/api/trades/recent?limit=100")
    trades = tr_resp.get("trades") or tr_resp.get("orders") or []
    cv2_orders = [t for t in trades
                  if t.get("signal_version") in ("quote_contrarian_v1", "quote_contrarian_v2")]
    cv2_orders.sort(key=lambda t: t["window_start"])
    print(f"影子 {len(shadow)} 行（可结算 {len(settled_shadow)}） | "
          f"contrarian 实盘订单 {len(cv2_orders)} 笔（v1 {sum(1 for t in cv2_orders if t['signal_version'].endswith('v1'))}"
          f" / v2 {sum(1 for t in cv2_orders if t['signal_version'].endswith('v2'))}）")

    # ---------- 1. 逐笔对账 ----------
    print()
    print("=" * 108)
    print("一、实盘订单 vs 影子记录逐笔对账")
    print("=" * 108)
    print(f"{'id':>3} {'版本':<4} {'UTC时间':<11} {'成交均价':>7} {'影子q':>5} {'价差':>7} "
          f"{'结算':>4} {'win':>3} {'pnl':>8} {'EV_A×2U':>8} {'偏差A':>7} 状态")
    matched = 0
    diffs = []
    diffs_c = []
    pnl_sum = 0.0
    ev_shadow_sum = 0.0
    amt_sum = 0.0
    price_gaps = []
    for t in cv2_orders:
        ws = int(t["window_start"])
        avg_p = t.get("average_price")
        amt = int(t["amount_in"]) / 1e18 if t.get("amount_in") else 0.0
        pnl = t.get("pnl")
        s = settled_shadow.get(ws)
        ver = "v1" if t["signal_version"].endswith("v1") else "v2"
        tstr = dt.datetime.fromtimestamp(ws / 1000, tz=dt.timezone.utc).strftime("%m-%d %H:%M")
        if s is None or avg_p is None:
            print(f"{t['id']:>3} {ver:<4} {tstr:<11} {avg_p if avg_p is not None else '—':>7} "
                  f"{'—':>5} {'—':>7} {t.get('settle_outcome') or '—':>4} {'—':>3} "
                  f"{pnl if pnl is not None else '—':>8} {'—':>8} {'—':>7} {t['status']}"
                  f"{'（无影子行匹配）' if s is None else ''}")
            continue
        matched += 1
        q = s["entry_down_price"]
        ev_a = s["ev_at_entry"]
        price_gaps.append(avg_p - q)
        # 双向校验：结算方向必须一致
        agree = (t.get("settle_outcome") or "") == s["settle_outcome"]
        pnl_per_unit = (pnl / amt) if (pnl is not None and amt > 0) else None
        ev_c = (1.0 / q - 1.0) if s["settle_outcome"] == "DOWN" else -1.0  # 无费口径
        shadow_ev = ev_a * amt
        dev = (pnl_per_unit - ev_a) if pnl_per_unit is not None else None
        dev_c = (pnl_per_unit - ev_c) if pnl_per_unit is not None else None
        if pnl is not None:
            pnl_sum += pnl
            ev_shadow_sum += shadow_ev
            amt_sum += amt
        if pnl_per_unit is not None:
            diffs.append(pnl_per_unit - ev_a)
            diffs_c.append(dev_c)
        print(f"{t['id']:>3} {ver:<4} {tstr:<11} {avg_p:>7.3f} {q:>5.2f} {avg_p - q:>+7.3f} "
              f"{s['settle_outcome']:>4} {'是' if t.get('win') else '否':>3} "
              f"{pnl:>+8.3f} {shadow_ev:>+8.3f}"
              f" {(dev if dev is not None else float('nan')):>+7.3f} {t['status']}"
              f"{'' if agree else ' ⚠结算不一致!'}")
    print("-" * 108)
    if matched:
        print(f"对上 {matched} 笔；成交均价−影子q 均值 {sum(price_gaps) / len(price_gaps):+.4f}"
              f"（负 = 实盘买得更便宜）")
        print(f"实盘已实现 pnl 合计 {pnl_sum:+.3f}U（本金 {amt_sum:.0f}U）"
              f" vs 影子同期 EV_A(0.98口径)×本金 {ev_shadow_sum:+.3f}U")
        if diffs:
            print(f"逐笔 pnl/unit − EV_A(0.98)：均值 {sum(diffs) / len(diffs):+.4f}"
                  f"，最大 {max(diffs):+.4f}，最小 {min(diffs):+.4f}")
            print(f"逐笔 pnl/unit − EV_C(1.00无费)：均值 {sum(diffs_c) / len(diffs_c):+.4f}"
                  f"，最大 {max(diffs_c):+.4f}，最小 {min(diffs_c):+.4f}")

    # ---------- 2. 通道开启后影子触发但无实盘单 ----------
    print()
    print("=" * 108)
    print("二、通道运行期影子触发 ↔ 实盘单覆盖")
    print("=" * 108)
    if cv2_orders:
        t0 = int(cv2_orders[0]["window_start"])
        after = {ws: s for ws, s in settled_shadow.items() if ws >= t0}
        order_ws = {int(t["window_start"]) for t in cv2_orders}
        missing = sorted(ws for ws in after if ws not in order_ws)
        print(f"实盘首单 {dt.datetime.fromtimestamp(t0/1000, tz=dt.timezone.utc):%m-%d %H:%M}"
              f" 后影子触发 {len(after)} 笔，实盘单 {len(cv2_orders)} 笔，缺口 {len(missing)} 笔")
        for ws in missing:
            s = after[ws]
            print(f"  缺口 {dt.datetime.fromtimestamp(ws/1000, tz=dt.timezone.utc):%m-%d %H:%M}"
                  f" 影子q={s['entry_down_price']:.2f} 结算={s['settle_outcome']}"
                  f" EV_A={s['ev_at_entry']:+.3f}")
        if missing:
            miss_ev = sum(after[ws]["ev_at_entry"] for ws in missing) / len(missing)
            print(f"  （缺口笔若全部实盘，按影子 EV_A 均值 {miss_ev:+.3f}/注）")

    # ---------- 3. FAILED 单归档 ----------
    failed = [t for t in cv2_orders if t.get("status") == "FAILED"]
    print(f"\nFAILED 单 {len(failed)} 笔：")
    for t in failed:
        ws = int(t["window_start"])
        print(f"  id{t['id']} {dt.datetime.fromtimestamp(ws/1000, tz=dt.timezone.utc):%m-%d %H:%M}"
              f" err={str(t.get('error_message'))[:80]}")
    print("\n结果已存日志 output/contrarian_v2_live_reconcile.log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
