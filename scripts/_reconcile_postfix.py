"""临时对账：修复部署（8/23 16:40 UTC）后 影子 vs 实盘 全明细"""
from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone

BASE = "http://165.154.147.155:8082"
FIX_TS = datetime(2026, 8, 23, 16, 40, tzinfo=timezone.utc).timestamp() * 1000


def get(path: str):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.loads(r.read().decode())


def utc_hhmm(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%m-%d %H:%M")


def main() -> None:
    now = datetime.now(timezone.utc)
    print(f"now_utc={now.strftime('%m-%d %H:%M')}")

    data = get("/api/misalignment/signals?version=quote_contrarian_v1&limit=50")
    sigs = [s for s in data["signals"]
            if s["version"] == "quote_contrarian_v1" and s["window_start"] >= FIX_TS]
    orders = get("/api/trades/recent?limit=30").get("orders", [])
    live = {o["window_start"]: o for o in orders
            if o.get("signal_version") == "quote_contrarian_v1"
            and o.get("window_start") and o["window_start"] >= FIX_TS}

    sig_map = {s["window_start"]: s for s in sigs}
    all_ws = sorted(set(sig_map) | set(live))
    print(f"\n修复后窗口: 影子={len(sig_map)} 实盘={len(live)}")
    print(f"{'窗口UTC':<12} {'影子q':>6} {'实盘均价':>8} {'结算':<5} {'影子win':<7} {'实盘pnl':>7}  备注")
    for ws in all_ws:
        s, o = sig_map.get(ws), live.get(ws)
        sq = f"{s['entry_down_price']}" if s else "—"
        oq = f"{o['average_price']:.3f}" if o and o.get("average_price") else ("PENDING" if o else "—")
        settle = (s or {}).get("settle_outcome") or (o or {}).get("settle_outcome") or "—"
        swin = str((s or {}).get("win")) if s else "—"
        pnl = f"{o['pnl']:+.1f}" if o and o.get("pnl") is not None else "—"
        note = ""
        if s and not o:
            note = "影子有/实盘无!"
        elif o and not s:
            note = "实盘有/影子无!"
        elif s and o:
            if o.get("average_price") and abs(o["average_price"] - s["entry_down_price"]) > 0.05:
                note = f"价差{abs(o['average_price'] - s['entry_down_price']):.3f}"
        print(f"{utc_hhmm(ws):<12} {sq:>6} {oq:>8} {settle:<5} {swin:<7} {pnl:>7}  {note}")

    wins = [s for s in sigs if s["status"] == "SETTLED" and s["win"] is True]
    settled = [s for s in sigs if s["status"] == "SETTLED" and s["win"] is not None]
    evs = [s["ev_at_entry"] for s in settled if s["ev_at_entry"] is not None]
    if settled:
        print(f"\n修复后影子统计: settled={len(settled)} wins={len(wins)} "
              f"WR={len(wins)/len(settled)*100:.1f}% "
              f"avgEV={sum(evs)/len(evs):+.3f}")
    live_settled = [o for o in live.values() if o.get("pnl") is not None]
    if live_settled:
        lpnl = sum(o["pnl"] for o in live_settled)
        lw = sum(1 for o in live_settled if o.get("win"))
        print(f"修复后实盘统计: settled={len(live_settled)} wins={lw} 累计pnl={lpnl:+.1f}U")


if __name__ == "__main__":
    main()
