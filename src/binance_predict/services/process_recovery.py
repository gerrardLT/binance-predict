"""过程段低价触达后的 DOWN 双确认；仅使用决策点之前的数据。"""
import math

VERSION = "process_recovery_down_v1"


def extract_process_recovery(start, btc_open, down_curve, btc_curve, max_ts=None):
    def series(raw):
        out = {}
        for p in raw or []:
            try:
                t, v = int(p['t']), float(p['v'])
            except (KeyError, TypeError, ValueError):
                continue
            if not start <= t < start + 300_000 or (max_ts is not None and t > max_ts):
                continue
            if not math.isfinite(v) or v <= 0:
                continue
            if t in out and out[t] != v:
                return None
            out[t] = v
        return sorted(out.items())

    down, btc = series(down_curve), series(btc_curve)
    if not down or not btc:
        return None
    bo = float(btc_open or btc[0][1])
    if not math.isfinite(bo) or bo <= 0:
        return None
    touch = next(((t, q) for t, q in down if start + 30_000 <= t < start + 180_000 and q <= .20), None)
    if touch is None:
        return None
    tt, tq = touch
    for t, q in down:
        if not tt < t < start + 180_000 or q < tq + .02:
            continue
        pre = [(bt, v) for bt, v in btc if bt <= t]
        if not pre or t - pre[-1][0] > 15_000:
            continue
        signed = [-(v / bo - 1) * 10_000 for _, v in pre]
        adverse = min([0.0, *signed])
        recovery = (signed[-1] - adverse) / max(abs(adverse), 1e-9)
        if recovery < .35:
            continue
        # 首个双确认点价格越界即整窗拒绝，不等待更便宜的第二次确认。
        if not .15 < q <= .35:
            return None
        values = [bo, *(v for _, v in pre)]
        hi, low = max(values), min(values)
        span = hi - low
        return dict(q=q, trigger_ts=t, td_sec=(t-start)/1000,
                    chg_bps=-signed[-1], body_r=abs(pre[-1][1]-bo)/span if span else 0.,
                    wick01=float(hi > max(bo, pre[-1][1])), rng_bps=span/bo*10_000,
                    npts=len(pre), dvol=None, dpar=None,
                    touch_ts=tt, touch_q=tq, recovery_frac=recovery)
    return None
