"""首触研究账本纯函数（Phase 1，规范 §4/§12.1/§12.2）。

纪律：
- 特征/门与生产同源：复用 firsthit_shadow_detector 冻结纯函数，禁止重写公式；
- 全窗口审计层：无首触/缺失/未结算的窗也留痕，供 Q 族缺失机制审计；
- 缺失纪律：dvol/dpar 缺失保持 None + 指示位，不填 0（规范 §14.4）；
- 本模块只产出 intention-to-trigger 口径 EV；quote/guard 与 fill 口径属 Phase 2+。
"""
from __future__ import annotations

from binance_predict.services.firsthit_shadow_detector import (
    FEE_RET,
    FIRSTHIT_SPECS,
    MIN_PTS,
    Q_HI,
    Q_LO,
    _ev_at_entry,
    _extract,
    _gate_of,
    _outcome_of,
    _ser,
)

AUDIT_VERSION = "window_audit_v1"
FEATURE_VERSION = "mother_event_v1"
EXPECTED_SAMPLES_PER_WINDOW = 20  # 300s / 15s 采样

# streak 门需要前驱 5m K 线，窗内不可判 → 服务侧标签恒 None（离线审计脚本回填）
_STREAK_VERSIONS = {"g7_streak_v1", "g7_strict_v1"}


def stratum_of(*, g1: bool, g3: bool) -> str:
    """四互斥层（规范 §4.1）：G1×G3 完备划分，加总恒等于 G0。"""
    if g1 and g3:
        return "both"
    if g1:
        return "g1_only"
    if g3:
        return "g3_only"
    return "neither"


def intention_ev(win: bool | None, q: float | None) -> float | None:
    """intention-to-trigger 理论 EV：赢 0.98/q−1 / 输 −1；未知保持 None。"""
    if win is None or q is None or q <= 0:
        return None
    return _ev_at_entry(win, q)


def _firsthit_point(down_curve) -> dict | None:
    """首触点：升序采样中第一个 down_price ∈ (Q_LO, Q_HI]（与生产同式）。"""
    for p in _ser(down_curve):
        if Q_LO < float(p["v"]) <= Q_HI:
            return p
    return None


def _exclusion_reason(window) -> str | None:
    """首触未成母事件的原因分类（规范 §12.1「为何被排除」）。"""
    dn = _ser(getattr(window, "curve_down_price", None))
    btc = _ser(getattr(window, "curve_btc_price", None))
    trig = _firsthit_point(dn)
    if trig is None:
        return "no_firsthit"
    bo = getattr(window, "entry_price", None)
    bo = float(bo) if (bo is not None and float(bo) > 0) else (
        float(btc[0]["v"]) if btc else None)
    if not bo or bo <= 0:
        return "no_btc_open"
    pre = [p for p in btc if int(p["t"]) <= int(trig["t"])]
    if len(pre) < MIN_PTS:
        return "npts_lt_8"
    return None  # extract 成功的其他边缘情形由调用方兜底


def build_window_audit(window) -> dict:
    """全窗口审计行（规范 §12.1）：每个 5m 窗一行，含原始首触、排除原因与数据质量。"""
    dn = _ser(getattr(window, "curve_down_price", None))
    up = _ser(getattr(window, "curve_up_price", None))
    btc = _ser(getattr(window, "curve_btc_price", None))
    sample_count = getattr(window, "sample_count", None)
    actual = int(sample_count) if sample_count else len(dn)
    gaps = [int(b["t"]) - int(a["t"]) for a, b in zip(dn, dn[1:])]
    outcome = _outcome_of(window)  # None = 不可判定结算
    raw_firsthit = _firsthit_point(dn) is not None
    mother = build_mother_event(window)
    reason = None if mother is not None else _exclusion_reason(window)
    return dict(
        window_start=int(window.start_time), window_end=int(window.end_time),
        audit_version=AUDIT_VERSION,
        expected_samples=EXPECTED_SAMPLES_PER_WINDOW, actual_samples=actual,
        down_pts=len(dn), up_pts=len(up), btc_pts=len(btc),
        max_gap_ms=max(gaps) if gaps else None,
        raw_firsthit_detected=raw_firsthit,
        firsthit_detected=mother is not None,
        exclusion_reason=reason,
        outcome=outcome, settled=outcome is not None,
    )


def _labels_of(ext: dict) -> dict:
    """逐版本门标签：直接调用生产 _gate_of，保证零公式漂移。"""
    labels: dict[str, bool | None] = {}
    for version, _ in FIRSTHIT_SPECS:
        if version in _STREAK_VERSIONS:
            labels[version] = None
        else:
            labels[version] = _gate_of(version, ext, None)
    labels["g0"] = True
    return labels


def build_mother_event(window) -> dict | None:
    """唯一 G0 母事件行（规范 §12.2）：每窗至多一行，G1/G3/G4/G7 族为标签。"""
    ext = _extract(window)  # 生产同源：首触/特征/dvol/dpar（严格 ex-ante）
    if ext is None:
        return None
    trigger_ts = int(ext["trigger_ts"])

    # q_prev / Δt：同窗触发前最后一个 down 采样（可能不存在 → None）
    q_prev, dt_prev_ms = None, None
    for p in _ser(getattr(window, "curve_down_price", None)):
        if int(p["t"]) < trigger_ts:
            q_prev, dt_prev_ms = float(p["v"]), trigger_ts - int(p["t"])

    # UP 同步价与和价偏差（UP+DOWN−1；缺曲线保持 None，规范 M6）
    up_at = None
    for p in _ser(getattr(window, "curve_up_price", None)):
        if int(p["t"]) <= trigger_ts:
            up_at = float(p["v"])
    sum_gap = (up_at + float(ext["q"]) - 1.0) if up_at is not None else None

    labels = _labels_of(ext)
    g1 = bool(labels["firsthit_down_body_v1"])
    g3 = bool(labels["firsthit_down_chg_v1"])

    outcome = _outcome_of(window)
    win: bool | None = None if outcome is None else (outcome == "DOWN")
    intention = None if (win is None or ext["q"] is None) else _ev_at_entry(win, float(ext["q"]))

    btc = _ser(getattr(window, "curve_btc_price", None))
    pre = [float(p["v"]) for p in btc if int(p["t"]) <= trigger_ts]
    bo = float(window.entry_price) if (getattr(window, "entry_price", None) is not None
                                       and float(window.entry_price) > 0) else (
        float(btc[0]["v"]) if btc else None)
    pts = [bo] + pre

    return dict(
        window_start=int(window.start_time), window_end=int(window.end_time),
        trigger_ts=trigger_ts, feature_version=FEATURE_VERSION,
        q=float(ext["q"]), q_prev=q_prev, dt_prev_ms=dt_prev_ms,
        up_price_at_trigger=up_at, sum_gap=sum_gap,
        btc_open=bo, btc_trigger=pre[-1], path_hi=max(pts), path_lo=min(pts),
        chg_bps=ext["chg_bps"], body_r=ext["body_r"], wick01=ext["wick01"],
        upper_wick_bps=ext.get("upper_wick_bps"), rng_bps=ext["rng_bps"],
        npts=ext["npts"], td_sec=ext["td_sec"],
        dvol=ext.get("dvol"), dpar=ext.get("dpar"),
        dvol_missing=ext.get("dvol") is None, dpar_missing=ext.get("dpar") is None,
        labels=labels, stratum=stratum_of(g1=g1, g3=g3),
        settle_outcome=outcome, win=win, intention_ev=intention,
    )


def rows_from_window(window) -> tuple[dict, dict | None]:
    """窗口 → (审计行, 母事件行 | None)。回填脚本与后续服务复用的唯一入口。"""
    return build_window_audit(window), build_mother_event(window)
