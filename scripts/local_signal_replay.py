"""一次性排查：按线上判定逻辑重放最近 12h，定位新系统 0 信号原因。

对齐 fake_breakout_detector 口径：
- 位势：滚动前 48 根 5m close 的 max/min（线上用 sentiment exit_price≈5m close）
- 破位：5m high > 阻力*1.0005（场景①pending）/ 5m low < 支撑*0.9995（场景②pending）
- 收盘确认：15m 聚合 close_pos=(C-L)/(H-L)≥0.85 且收阳（①）/ vol_ratio≥2.0 且收阴（②）
输出每个 15m 周期的破位与形态情况，覆盖新旧系统时段对照。
"""
from __future__ import annotations

import asyncio
import time

import httpx

EPS = 0.0005
CLOSE_POS_MIN = 0.85
VOL_RATIO_MIN = 2.0


async def main() -> None:
    async with httpx.AsyncClient(timeout=20) as cli:
        r = await cli.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "5m", "limit": 200},
        )
        r.raise_for_status()
        raw = r.json()
    # 只留已收盘根；closes 用 close(4)，high=2 low=3 vol=5
    now_ms = int(time.time() * 1000)
    bars = [k for k in raw if int(k[0]) + 300_000 <= now_ms + 500]
    closes = [float(k[4]) for k in bars]

    # 聚合 15m：open_time 对齐 900_000
    agg: dict[int, dict] = {}
    for k in bars:
        ot = int(k[0])
        c15 = ot // 900_000 * 900_000
        a = agg.setdefault(c15, {"o": float(k[1]), "h": float(k[2]), "l": float(k[3]),
                                 "c": float(k[4]), "v": 0.0, "broke": set()})
        a["h"] = max(a["h"], float(k[2]))
        a["l"] = min(a["l"], float(k[3]))
        a["c"] = float(k[4])
        a["v"] += float(k[5])

    # 逐 15m 周期：位势 = 该周期前 48 根 5m close 极值（不含周期内）；破位看周期内 5m high/low
    cycs = sorted(agg)
    print(f"5m bars={len(bars)} 15m cycles={len(cycs)} "
          f"({time.strftime('%m-%d %H:%M', time.gmtime(cycs[0]/1000))} ~ "
          f"{time.strftime('%m-%d %H:%M', time.gmtime(cycs[-1]/1000))} UTC)\n")

    hist: list[float] = []  # 5m closes 时间升序（截至周期开始前）
    bar_i = 0
    vols_15m: list[float] = []
    hit_cnt = {"1": 0, "2": 0}
    for c15 in cycs:
        # 推进 5m closes 到周期开始前
        while bar_i < len(bars) and int(bars[bar_i][0]) < c15:
            hist.append(float(bars[bar_i][4]))
            bar_i += 1
        a = agg[c15]
        if len(hist) >= 48:
            window = hist[-48:]
            res, sup = max(window), min(window)
            if a["h"] > res * (1 + EPS):
                a["broke"].add("high")
            if a["l"] < sup * (1 - EPS):
                a["broke"].add("low")
        vol_ma = (sum(vols_15m[-20:]) / len(vols_15m[-20:])) if len(vols_15m) >= 10 else None
        vols_15m.append(a["v"])

        rng = a["h"] - a["l"]
        close_pos = (a["c"] - a["l"]) / rng if rng > 0 else None
        vol_ratio = a["v"] / vol_ma if vol_ma else None
        broke = "+".join(sorted(a["broke"])) or "-"
        p1 = p2 = " "
        if "high" in a["broke"] and close_pos is not None and a["c"] > a["o"] and close_pos >= CLOSE_POS_MIN:
            p1 = "①"
            hit_cnt["1"] += 1
        if "low" in a["broke"] and vol_ratio is not None and a["c"] < a["o"] and vol_ratio >= VOL_RATIO_MIN:
            p2 = "②"
            hit_cnt["2"] += 1
        t = time.strftime("%m-%d %H:%M", time.gmtime(c15 / 1000))
        cp = f"{close_pos:.2f}" if close_pos is not None else " n/a"
        vr = f"{vol_ratio:.1f}" if vol_ratio is not None else " n/a"
        red = "阴" if a["c"] < a["o"] else "阳"
        print(f"{t} | 破位[{broke:>5}] {red} cp={cp} vr={vr} | {p1}{p2}")

    print(f"\n命中：场景① {hit_cnt['1']} 条 | 场景② {hit_cnt['2']} 条")


if __name__ == "__main__":
    asyncio.run(main())
