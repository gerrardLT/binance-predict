"""
Daily Live Trading Report Generator.
Usage:
    .venv/Scripts/python.exe scripts/daily_report.py [--date YYYY-MM-DD]
Default date is yesterday in Asia/Shanghai (UTC+8).
"""

import argparse
import json
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


def load_env() -> dict:
    env = {}
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip("\"'")
    return env


def generate_daily_report(target_date_str: str | None = None) -> str:
    env = load_env()
    base = env.get("PROD_API_BASE", "http://165.154.147.155:8082")
    token = env.get("PROD_API_TOKEN", "")

    tz8 = timezone(timedelta(hours=8))
    now8 = datetime.now(tz8)

    if not target_date_str:
        target_date = (now8 - timedelta(days=1)).date()
        target_date_str = target_date.strftime("%Y-%m-%d")
    else:
        target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()

    # 1. Fetch channel definitions
    ch_info = {}
    try:
        req = urllib.request.Request(
            f"{base}/api/misalignment/signals",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for c in data.get("live_channels", {}).get("channels", []):
            ch_info[c["channel"]] = c
    except Exception as e:
        pass

    # 2. Fetch trades
    req = urllib.request.Request(
        f"{base}/api/trades/recent?limit=5000",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        trades_data = json.loads(resp.read().decode("utf-8"))

    orders = trades_data.get("orders", [])

    # Filter target date automated orders
    day_orders = []
    for o in orders:
        v = o.get("signal_version") or ""
        if v.startswith("manual_"):
            continue
        created = o.get("created_at")
        if not created:
            continue
        dt = datetime.fromisoformat(created.replace("Z", "+00:00")).astimezone(tz8)
        if dt.date() == target_date:
            day_orders.append((dt, o))

    day_orders.sort(key=lambda x: x[0])

    tot_req = len(day_orders)
    filled_orders = [o for _, o in day_orders if o.get("status") == "FILLED"]
    abandoned_orders = [o for _, o in day_orders if o.get("status") != "FILLED"]
    tot_fill = len(filled_orders)
    fill_rate = (tot_fill / tot_req * 100) if tot_req > 0 else 0.0

    tot_amt = sum(float(o.get("amount_in") or 0) / 1e18 for o in filled_orders)
    tot_pnl = sum(float(o.get("pnl") or 0) for o in filled_orders if o.get("pnl") is not None)
    roi = (tot_pnl / tot_amt * 100) if tot_amt > 0 else 0.0

    wins = sum(1 for o in filled_orders if o.get("win") is True)
    losses = sum(1 for o in filled_orders if o.get("win") is False)
    settled = wins + losses
    win_rate = (wins / settled * 100) if settled > 0 else 0.0

    # Channel stats
    c_stats = defaultdict(lambda: {
        "req": 0, "fill": 0, "amt": 0.0, "pnl": 0.0, "wins": 0, "losses": 0
    })
    for _, o in day_orders:
        v = o.get("signal_version") or "UNKNOWN"
        st = c_stats[v]
        st["req"] += 1
        if o.get("status") == "FILLED":
            st["fill"] += 1
            st["amt"] += float(o.get("amount_in") or 0) / 1e18
            if o.get("pnl") is not None:
                st["pnl"] += float(o.get("pnl"))
            if o.get("win") is True:
                st["wins"] += 1
            elif o.get("win") is False:
                st["losses"] += 1

    lines = []
    lines.append(f"## 📊 线上实盘交易日报 [{target_date_str}]")
    lines.append(f"> 生成时间: {now8.strftime('%Y-%m-%d %H:%M:%S')} (UTC+8)\n")

    lines.append("### 一、总体表现概览")
    lines.append(f"- **总触发信号**: **{tot_req}** 笔")
    lines.append(f"- **实际成交单**: **{tot_fill}** 笔 (成交率: **{fill_rate:.1f}%**，护栏弃单: **{len(abandoned_orders)}** 笔)")
    lines.append(f"- **日实际投入**: **{tot_amt:.2f} U** (笔均约 {tot_amt/tot_fill:.2f} U)" if tot_fill > 0 else "- **日实际投入**: 0.00 U")
    lines.append(f"- **当日净收益**: **{tot_pnl:+.2f} U**")
    lines.append(f"- **资金回报率 (ROI)**: **{roi:+.2f}%**")
    lines.append(f"- **成交流水胜率**: **{win_rate:.1f}%** ({wins}胜 / {losses}负)")
    lines.append("")

    lines.append("### 二、各通道详细明细")
    lines.append("| 通道中文名 | 通道标识 | 触发/成交 | 投入(U) | 净收益(U) | 胜率 | 状态 |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")

    # Sort channels by pnl desc
    for v, st in sorted(c_stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
        meta = ch_info.get(v, {})
        display_name = meta.get("display_name") or v
        enabled = meta.get("enabled", False)
        status_tag = "🟢开启" if enabled else "⚪已停"
        s_count = st["wins"] + st["losses"]
        wr_str = f"{st['wins']/s_count*100:.1f}%" if s_count > 0 else "N/A"
        lines.append(
            f"| {display_name} | `{v}` | {st['req']}/{st['fill']} | {st['amt']:.1f} | {st['pnl']:+.2f} | {wr_str} | {status_tag} |"
        )
    lines.append("")

    # Highlights
    if c_stats:
        best_ch = max(c_stats.items(), key=lambda x: x[1]["pnl"])
        worst_ch = min(c_stats.items(), key=lambda x: x[1]["pnl"])
        lines.append("### 三、通道归因要点")
        best_name = ch_info.get(best_ch[0], {}).get("display_name") or best_ch[0]
        worst_name = ch_info.get(worst_ch[0], {}).get("display_name") or worst_ch[0]
        if best_ch[1]["pnl"] > 0:
            lines.append(f"- 🏆 **最佳贡献**: **{best_name}** (`{best_ch[0]}`)，净收益 **{best_ch[1]['pnl']:+.2f} U** ({best_ch[1]['wins']}胜/{best_ch[1]['losses']}负)")
        else:
            lines.append("- ⚠️ **无正收益通道**")
        if worst_ch[1]["pnl"] < 0:
            lines.append(f"- 🩸 **最大回撤**: **{worst_name}** (`{worst_ch[0]}`)，净收益 **{worst_ch[1]['pnl']:+.2f} U** ({worst_ch[1]['wins']}胜/{worst_ch[1]['losses']}负)")

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, help="Target date YYYY-MM-DD", default=None)
    args = parser.parse_args()
    print(generate_daily_report(args.date))
