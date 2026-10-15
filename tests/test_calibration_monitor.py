"""买贵校准监控（2026-10-16）：桶级校准指标/灯判定/报告装配/快照行/连涨序列。

起因：用户指出实盘通道几乎全押 DOWN，怕「5m K 一直涨」的强多头里连环买贵。
2026-10-15 实测 DOWN 单 ROI 随 5m 连涨根数单调恶化（+8%→-22%），据此把
「通道×行情档 实际胜率 vs 入场价保本线」做成常驻体检（signal_health 扩展）。
"""
from __future__ import annotations

from scipy.stats import beta

from binance_predict.services import signal_health as sh
from binance_predict.services import signal_health_service as shs


def _events(spec: list[tuple[float, bool]], regime: dict | None = None) -> list[dict]:
    """spec = [(保本线, 是否赢), ...]；regime 附加到每个事件。"""
    out = []
    for i, (be, win) in enumerate(spec):
        e = {"ts": 1_000_000 + i * 300_000, "win": win, "ev": (1.0 - be) if win else -be,
             "be": be, "q": be * 0.98, "direction": "DOWN"}
        if regime:
            e.update(regime)
        out.append(e)
    return out


# ---------------- up_streak_series：5m 连续收阳根数 ----------------

def test_up_streak_series_counts_and_resets():
    assert shs.up_streak_series([100.0, 101.0, 102.0, 101.5, 102.0, 103.0]) == [0, 1, 2, 0, 1, 2]


def test_up_streak_series_missing_price_breaks_to_none():
    out = shs.up_streak_series([100.0, 101.0, None, 102.0, 103.0])
    assert out == [0, 1, None, 0, 1]


def test_up_streak_series_equal_price_resets():
    # 平盘（收平不算收阳）清零
    assert shs.up_streak_series([100.0, 100.0, 101.0]) == [0, 0, 1]


# ---------------- 桶级标签 ----------------

def test_up_streak_label_buckets():
    assert sh.up_streak_label(None) == "未知"
    assert sh.up_streak_label(0) == "0-1根"
    assert sh.up_streak_label(3) == "2-3根"
    assert sh.up_streak_label(5) == "4-5根"
    assert sh.up_streak_label(7) == "6-7根"
    assert sh.up_streak_label(8) == "≥8根"
    assert sh.up_streak_label(23) == "≥8根"


def test_ret_24h_label_matches_trend_buckets():
    assert sh.ret_24h_label(None) == "未知"
    assert sh.ret_24h_label(-0.03) == "暴跌(<-2%)"
    assert sh.ret_24h_label(-0.01) == "下行(-2%~0)"
    assert sh.ret_24h_label(0.01) == "上行(0~+2%)"
    assert sh.ret_24h_label(0.025) == "过热(>+2%)"


# ---------------- calibration_metrics / classify_calibration ----------------

def test_calibration_metrics_edge_and_posterior():
    # 30 单、胜率 40%（12 胜）、平均保本 0.60 → 明显买贵
    evs = _events([(0.60, i < 12) for i in range(30)])
    m = sh.calibration_metrics(evs)
    assert m["n"] == 30 and m["wins"] == 12
    assert abs(m["win_rate"] - 0.4) < 1e-9
    assert abs(m["mean_breakeven"] - 0.60) < 1e-9
    assert abs(m["edge"] - (0.4 - 0.60)) < 1e-9
    assert abs(m["p_edge_negative"] - float(beta.cdf(0.60, 13, 19))) < 1e-9
    light, reasons = sh.classify_calibration(m)
    assert light == "RED" and reasons


def test_classify_calibration_gray_on_small_n():
    # 样本 < BUCKET_MIN_N 一律灰灯（即使胜率惨淡）
    evs = _events([(0.60, False)] * (sh.BUCKET_MIN_N - 1))
    light, _ = sh.classify_calibration(sh.calibration_metrics(evs))
    assert light == "GRAY"


def test_classify_calibration_red_requires_min_n():
    # 高置信买贵但 n < BUCKET_RED_MIN_N → 只能黄灯
    evs = _events([(0.70, i < 5) for i in range(20)])  # 胜率 25% vs 保本 70%
    m = sh.calibration_metrics(evs)
    light, _ = sh.classify_calibration(m)
    assert m["p_edge_negative"] >= sh.BUCKET_RED_P_NEG
    assert 20 < sh.BUCKET_RED_MIN_N
    assert light == "YELLOW"


def test_classify_calibration_gray_without_breakeven():
    evs = [{"ts": i, "win": True, "ev": None, "be": None, "q": None} for i in range(20)]
    light, reasons = sh.classify_calibration(sh.calibration_metrics(evs))
    assert light == "GRAY" and "保本线" in reasons[0]


def test_classify_calibration_green_when_winning():
    evs = _events([(0.55, True)] * 30)
    light, reasons = sh.classify_calibration(sh.calibration_metrics(evs))
    assert light == "GREEN" and reasons == []


# ---------------- calibration_rows：桶顺序与空桶跳过 ----------------

def test_calibration_rows_order_and_skip_empty():
    evs = (_events([(0.5, True)] * 3, {"up_streak_5m": 0})
           + _events([(0.8, False)] * 2, {"up_streak_5m": 9}))
    rows = sh.calibration_rows(evs, "up_streak_5m",
                               lambda e: sh.up_streak_label(e.get("up_streak_5m")),
                               sh.UP_STREAK_LABELS)
    assert [r["bucket"] for r in rows] == ["0-1根", "≥8根"]
    assert all(r["lens"] == "up_streak_5m" for r in rows)
    assert rows[1]["n"] == 2 and rows[1]["small_sample"]


# ---------------- build_calibration_report：合并层与通道层 ----------------

def _live_meta(enabled: dict[str, bool]) -> dict:
    return {ch: {"enabled": on, "retired": False} for ch, on in enabled.items()}


def test_build_calibration_report_pooled_by_direction_and_channels():
    live_ev = {
        "ch_a": _events([(0.5, True)] * 6, {"ret_24h": 0.01, "up_streak_5m": 0}),
        "ch_b": _events([(0.8, False)] * 6, {"ret_24h": 0.03, "up_streak_5m": 9}),
        # 下线但历史单 ≥30 的注册通道 → 仍保留体检行（enabled=False）
        "ch_c": _events([(0.6, True)] * 30, {"ret_24h": -0.03, "up_streak_5m": 3}),
    }
    # ch_b 事件 direction 改成 UP 模拟押 UP 通道
    for e in live_ev["ch_b"]:
        e["direction"] = "UP"
    meta = _live_meta({"ch_a": True, "ch_b": False, "ch_c": False})
    cal = shs.build_calibration_report(
        live_ev, meta, channel_direction={"ch_a": "DOWN", "ch_b": "UP", "ch_c": "DOWN"},
        channel_display={"ch_a": "通道A", "ch_b": "通道B", "ch_c": "通道C"}, now_ms=42,
    )
    assert cal["generated_at"] == 42
    assert set(cal["lenses"]) == {"ret_24h", "up_streak_5m"}

    ret_rows = cal["lenses"]["ret_24h"]
    pooled_down = [r for r in ret_rows if r["entity"] == "ALL_DOWN"]
    pooled_up = [r for r in ret_rows if r["entity"] == "ALL_UP"]
    assert pooled_down and sum(r["n"] for r in pooled_down) == 36   # ch_a 6 + ch_c 30（跨桶）
    assert pooled_up and sum(r["n"] for r in pooled_up) == 6
    assert pooled_down[0]["entity_kind"] == "pooled"
    assert pooled_down[0]["display_name"] == "全部DOWN单"
    # 合并行排最前
    assert ret_rows[0]["entity_kind"] == "pooled"

    ch_rows = [r for r in ret_rows if r["entity_kind"] == "channel"]
    # ch_a 在线在列；ch_b 下线且样本 <30 → 不单列；ch_c 下线但 ≥30 → 保留
    assert {r["entity"] for r in ch_rows} == {"ch_a", "ch_c"}
    assert next(r for r in ch_rows if r["entity"] == "ch_a")["display_name"] == "通道A"
    assert next(r for r in ch_rows if r["entity"] == "ch_c")["enabled"] is False

    # 5m 连涨透镜：ch_b 的 6 单全落 ≥8根 桶
    streak_rows = cal["lenses"]["up_streak_5m"]
    ge8 = [r for r in streak_rows if r["bucket"] == "≥8根"]
    assert sum(r["n"] for r in ge8) == 6


def test_build_calibration_report_skips_unregistered_and_small_channels():
    live_ev = {
        "g7_legacy": _events([(0.5, True)] * 40),   # 不在 meta（退役通道）→ 不单列
        "ch_small": _events([(0.5, True)] * 5),     # 注册但下线且样本 <30 → 跳过
    }
    cal = shs.build_calibration_report(live_ev, _live_meta({"ch_small": False}))
    entities = {r["entity"] for lens in cal["lenses"].values() for r in lens}
    assert "g7_legacy" not in entities and "ch_small" not in entities
    # 但退役通道的单仍计入合并层
    assert any(r["entity"] == "ALL_DOWN" and r["n"] == 45 for lens in cal["lenses"].values() for r in lens)


# ---------------- 快照行与转红告警 ----------------

def test_calibration_snapshot_rows_shape_and_alert_text():
    live_ev = {"ch_a": _events([(0.70, False)] * 30, {"up_streak_5m": 9})}
    cal = shs.build_calibration_report(
        live_ev, _live_meta({"ch_a": True}),
        channel_direction={"ch_a": "DOWN"}, channel_display={"ch_a": "通道A"},
    )
    snaps = shs.calibration_snapshot_rows(cal)
    assert snaps and all(s["scope"] == "calibration" for s in snaps)
    red = [s for s in snaps if s["light"] == "RED"]
    assert red, "30 连败 vs 保本 0.70 应触发红灯"
    # 合并行也红（ALL_DOWN 同样 30 连败），取通道行验证文案
    s = next(x for x in red if x["metrics"]["entity"] == "ch_a")
    assert s["key"] == f"{s['metrics']['lens']}|{s['metrics']['entity']}|{s['metrics']['bucket']}"
    assert s["metrics"]["display_name"] == "通道A"
    assert s["metrics"]["lens_label"] and s["metrics"]["bucket"]

    # 转红检测（复用通道级机制）+ 告警文案含关键数字
    transitions = sh.new_red_transitions({}, snaps)
    assert not transitions, "首轮（prev 无记录）不告警，防冷启动刷屏"
    prev = {s["key"] and (s["scope"], s["key"]): "GREEN" for s in snaps}
    transitions = sh.new_red_transitions(prev, snaps)
    assert any(t["scope"] == "calibration" for t in transitions)
    text = sh.format_calibration_red_alert(s)
    assert "买贵校准转红" in text and "通道A" in text and "押DOWN" not in text  # 方向字段在标题括号里
    assert "DOWN" in text


def test_up_streak_buckets_detail_view():
    evs = _events([(0.5, True)] * 4, {"up_streak_5m": 2}) + _events([(0.5, False)] * 1, {"up_streak_5m": 10})
    rows = sh.up_streak_buckets(evs)
    assert {r["segment"] for r in rows} == {"2-3根", "≥8根"}
    assert all(set(r) >= {"segment", "n", "win_rate", "win_rate_ci95", "avg_ev", "small_sample"} for r in rows)
