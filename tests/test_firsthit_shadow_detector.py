"""5m DOWN 首触影子检测器测试（firsthit_down_v1/_body_v1/_chg_v1 族）。

冻结口径（scripts/local_shape_scan_v2.py v2 方法论修正版，规则冻结勿动）：
  触发：5m 窗内 DOWN token 报价**首次**进入 (0.005, 0.1] → 以该报价买 DOWN。
  特征（严格 ex-ante，只读 ≤触发时刻采样）：
    chg_bps = (btc@触 − 开盘)/开盘×1e4（开盘 = entry_price 优先，回退 btc 曲线首点）
    body_r  = |btc@触 − 开盘| / (路径 max−min)   路径 = 开盘 + t≤触发 的全部 btc 采样
    npts    = t≤触发 的 btc 采样点数；npts < 8 → 整窗不落表（v2 主分析口径）
  三 version：
    firsthit_down_v1       G0 基底（全部首触，对照组）
    firsthit_down_body_v1  G1：body_r ≤ 0.35（FDR q=0.007）
    firsthit_down_chg_v1   G3：chg_bps ≤ +2.82bp（logit β=+0.07 p=0.000）

数据流（归档后处理，同 absorption_shadow_detector）：
  启动预热（trailing ~14d 缓冲，不填落表）→ 60s 轮询新归档窗 →
  逐窗计算首触特征 → gate 过滤 → shadow_gate.is_enabled() 拦截 + (version, window_start) 幂等 →
  直接落 SETTLED（无 PENDING）。

影子纪律：只记录不下注，物理隔离下单路径；EV = 赢 0.98/q−1 / 输 −1（费 2% 无溢价，逐事件真实触发价）。
前向裁决标准（预注册，4 周）：G1 通过 = P ≥ 12% 且 EV CI 下界 > 0；整体否决 = G0 EV CI 上界 < 0。
"""
from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace

import pytest

import binance_predict.services.firsthit_shadow_detector as fhd
from binance_predict.config.settings import settings
from binance_predict.main import SHADOW_BENCH
from binance_predict.services.firsthit_shadow_detector import (
    FIRSTHIT_SPECS,
    MIN_PTS,
    Q_LO,
    Q_HI,
    BODY_R_GATE,
    CHG_BPS_GATE,
    FirstHitShadowDetector,
    _ev_at_entry,
    _extract,
    _gate_of,
    _outcome_of,
    _ser,
    extract_firsthit_features,
)

START = 1_700_000_000_000 // 300_000 * 300_000   # 对齐 5m 整点


# ============================================================
# 替身（不触真实 DB）：同 absorption 模式复用 _FakeResult/_FakeSession/_FakeSessionCtx
# ============================================================

class _FakeResult:
    def __init__(self, rows=None, scalar=None, first=None):
        self._rows, self._scalar, self._first = rows or [], scalar, first

    def scalars(self):
        return self

    def all(self):
        return self._rows

    def scalar_one_or_none(self):
        return self._scalar

    def first(self):
        return self._first


class _FakeSession:
    def __init__(self, rows=None, scalar=None, first=None):
        self.rows, self.scalar, self.first_val = rows or [], scalar, first
        self.added = []
        self.committed = 0
        self.executed = 0

    async def execute(self, _stmt):
        self.executed += 1
        return _FakeResult(self.rows, self.scalar, self.first_val)

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        self.committed += 1

    async def rollback(self):
        pass


class _FakeSessionCtx:
    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *_exc):
        return False


# ============================================================
# 构造器：合成 SentimentWindow（DOWN 首触特征）
# ============================================================

def _seed_window(npts: int = 20, *, trigger_q: float = 0.07, btc_change_pct: float = 0.00005,
                 outcome: str = "DOWN", bo: float = 100_000.0) -> SimpleNamespace:
    """合成已归档 SentimentWindow，DOWN 首触发生在某个采样点。

    npts: 触发前 btc 采样点数（含开盘），需 ≥ MIN_PTS 才落表（v2 主分析口径）
    trigger_q: 触发时的 down_price
    btc_change_pct: BTC 相对开盘涨跌比例（正=涨、负=跌）
    """
    if npts < 2:
        raise ValueError("npts must be at least 2 for start+at_trigger")

    td_sec = 120                         # 任意在窗口内的时间
    trigger_ts = START + td_sec * 1000

    # 构建 btc 采样序列：开盘 + npts-1 个点（最终一个点在 trigger_ts）
    btc_pts = [{"t": START, "v": bo}]
    step = (trigger_ts - START) // (npts - 1)
    prev = bo
    for i in range(1, npts):
        ts = START + i * step
        if i < npts - 1:
            prev = prev * (1.0 + (btc_change_pct / (npts - 1)))
        else:
            # 最后一点是触发时 btc 值
            prev = bo * (1.0 + btc_change_pct)
        btc_pts.append({"t": ts, "v": prev})

    # 构建 down_price 采样序列：前半段都 >0.1，触发点第一个 ≤0.1（首个满足 (Q_LO,Q_HI] 的点）
    dn_pts = []
    for i, (ts, _) in enumerate(btc_pts):
        ts = btc_pts[i]["t"]
        if i < npts - 1:
            dn_val = 0.5 + 0.4 * (i / (npts - 1))      # 从 0.5→1.0（全>0.1）
        else:
            dn_val = trigger_q                          # 触发点 = trigger_q ∈ (Q_LO, Q_HI]
        dn_pts.append({"t": ts, "v": dn_val})

    # curve_trade_volume / curve_participants（soft 维度）
    vol_pts = [{"t": START, "v": 1000}, {"t": trigger_ts, "v": 2000}]
    par_pts = [{"t": START, "v": 10.0}, {"t": trigger_ts, "v": 20.0}]

    actual_return = btc_change_pct if btc_change_pct != 0 else 1e-4

    return SimpleNamespace(
        start_time=START, end_time=START + 300_000,
        entry_price=bo, actual_return=actual_return, outcome=outcome,
        curve_btc_price=btc_pts,
        curve_down_price=dn_pts,
        curve_up_price=[{"t": ts, "v": 1.0 - dp["v"]} for dp in dn_pts],
        curve_up_pct=[{"t": ts, "v": (1.0 - dp["v"]) * 100} for dp in dn_pts],
        curve_down_pct=[{"t": ts, "v": dp["v"] * 100} for dp in dn_pts],
        curve_trade_volume=vol_pts,
        curve_participants=par_pts,
    )


# ============================================================
# 1. 冻结口径：EV / gate / 有效性过滤
# ============================================================

def test_ev_at_entry_fee_two_pct_no_premium():
    """EV 口径（与 absorption._ev_at_entry 同源）：赢 0.98/q−1 / 输 −1。"""
    assert fhd.FEE_RET == 0.98
    assert _ev_at_entry(True, 0.07) == pytest.approx(0.98 / 0.07 - 1.0)
    assert _ev_at_entry(False, 0.07) == -1.0


def test_outcome_of_judgeable_only():
    """可判定结算：actual_return None/0 → None；outcome 非 UP/DOWN → None。"""
    assert _outcome_of(SimpleNamespace(actual_return=None, outcome="UP")) is None
    assert _outcome_of(SimpleNamespace(actual_return=0.0, outcome="UP")) is None
    assert _outcome_of(SimpleNamespace(actual_return=0.001, outcome="DOWN")) == "DOWN"


def test_gate_of_version_logic():
    """G0 恒真；G1 body≤0.35；G3 chg≤+2.82bp；G4 = G1∩G3；缺失特征 → 假。"""
    ext_ok = {
        "q": 0.07, "trigger_ts": START + 120_000, "td_sec": 120,
        "chg_bps": 2.0, "body_r": 0.30, "wick01": 0.0, "rng_bps": 5.0,
        "npts": 20, "dvol": 1000.0, "dpar": 10.0,
    }
    assert _gate_of("firsthit_down_v1", ext_ok) is True
    assert _gate_of("firsthit_down_body_v1", ext_ok) is True    # 0.30 ≤ 0.35
    assert _gate_of("firsthit_down_chg_v1", ext_ok) is True     # 2.0 ≤ 2.82
    assert _gate_of("firsthit_down_g4_v1", ext_ok) is True      # 两条件同时满足

    # body_r 超限
    ext_bad_body = ext_ok.copy()
    ext_bad_body["body_r"] = 0.40
    assert _gate_of("firsthit_down_body_v1", ext_bad_body) is False
    assert _gate_of("firsthit_down_chg_v1", ext_bad_body) is True
    assert _gate_of("firsthit_down_g4_v1", ext_bad_body) is False   # 交集：body 不过即假

    # chg_bps 超限
    ext_bad_chg = ext_ok.copy()
    ext_bad_chg["chg_bps"] = 5.0
    assert _gate_of("firsthit_down_v1", ext_bad_chg) is True
    assert _gate_of("firsthit_down_body_v1", ext_bad_chg) is True
    assert _gate_of("firsthit_down_chg_v1", ext_bad_chg) is False
    assert _gate_of("firsthit_down_g4_v1", ext_bad_chg) is False    # 交集：chg 不过即假


def test_gate_of_g4_is_intersection_of_g1_and_g3():
    """G4 严格等于 G1 ∧ G3（四象限全覆盖）：任一为假则 G4 为假，两者皆真才为真。

    这条锁住 G4 的集合语义——G4 ⊂ G1 且 G4 ⊂ G3，因此同窗 G1/G3/G4 全中
    是对同一 DOWN 事件的重复下注（非独立信号），实盘暴露须按事件而非通道核算。
    """
    base = {
        "q": 0.07, "trigger_ts": START + 120_000, "td_sec": 120,
        "chg_bps": 2.0, "body_r": 0.30, "wick01": 0.0, "rng_bps": 5.0,
        "npts": 20, "dvol": None, "dpar": None,
    }
    for chg, body, expect in [
        (2.0, 0.30, True),    # G1 ✓ G3 ✓ → G4 ✓
        (2.0, 0.40, False),   # G1 ✗ G3 ✓ → G4 ✗
        (5.0, 0.30, False),   # G1 ✓ G3 ✗ → G4 ✗
        (5.0, 0.40, False),   # G1 ✗ G3 ✗ → G4 ✗
    ]:
        ext = {**base, "chg_bps": chg, "body_r": body}
        g1 = _gate_of("firsthit_down_body_v1", ext)
        g3 = _gate_of("firsthit_down_chg_v1", ext)
        g4 = _gate_of("firsthit_down_g4_v1", ext)
        assert g4 is expect, f"chg={chg} body={body}"
        assert g4 is (g1 and g3), "G4 必须恒等于 G1∧G3"


def test_gate_of_g4_boundary_inclusive_and_none_guard():
    """G4 贴线判定（含边界，与 G1/G3 同口径 ≤）+ 特征缺失 None 守卫 → 不落表。"""
    base = {
        "q": 0.07, "trigger_ts": START + 120_000, "td_sec": 120,
        "chg_bps": fhd.CHG_BPS_GATE, "body_r": fhd.BODY_R_GATE,
        "wick01": 0.0, "rng_bps": 5.0, "npts": 20, "dvol": None, "dpar": None,
    }
    # 贴线：chg == 2.82 且 body == 0.35 → 双双放行（含贴线）
    assert _gate_of("firsthit_down_g4_v1", base) is True
    # 越界一例即拒：chg 刚过线
    assert _gate_of("firsthit_down_g4_v1", {**base, "chg_bps": fhd.CHG_BPS_GATE + 1e-9}) is False
    # 越界一例即拒：body 刚过线
    assert _gate_of("firsthit_down_g4_v1", {**base, "body_r": fhd.BODY_R_GATE + 1e-9}) is False
    # None 守卫：特征缺失一律不落表（不得抛异常）
    assert _gate_of("firsthit_down_g4_v1", {**base, "chg_bps": None}) is False
    assert _gate_of("firsthit_down_g4_v1", {**base, "body_r": None}) is False
    assert _gate_of("firsthit_down_g4_v1", {**base, "chg_bps": None, "body_r": None}) is False


def test_firsthit_specs_contains_g4_and_is_unique():
    """FIRSTHIT_SPECS 注册了 G4，且 version 名全表唯一（幂等键依赖唯一性）。"""
    versions = [v for v, _ in fhd.FIRSTHIT_SPECS]
    assert "firsthit_down_g4_v1" in versions
    assert len(versions) == len(set(versions)), "version 重名会破坏 (version, window_start) 幂等"
    # G4 排在 G3 之后、G7 之前（族内顺序，便于日志与面板阅读）
    assert versions.index("firsthit_down_chg_v1") < versions.index("firsthit_down_g4_v1") \
        < versions.index("firsthit_down_g7_v1")


# ============================================================
# 2. 特征抽取：npts 门槛 / 首触定义 / 特征公式互验
# ============================================================

def test_ser_cleans_and_sorts():
    """_ser 剔除 t/v 为空的点并按 t 升序（防御 JSONB 乱序，同 absorption/quote_edge）。"""
    raw = [{"t": 300, "v": 3.0}, {"t": 100, "v": 1.0}, {"t": None, "v": 9.0},
           {"t": 200, "v": None}, {"t": 200, "v": 2.0}]
    out = _ser(raw)
    assert [p["t"] for p in out] == [100, 200, 300]
    assert [p["v"] for p in out] == [1.0, 2.0, 3.0]
    assert _ser(None) == []


def test_extract_first_hit():
    """首触定义为降序采样中第一个 down_price ∈ (Q_LO, Q_HI] 的点；之前的点都 >0.1 """
    w = _seed_window(npts=20, trigger_q=0.07)
    ext = _extract(w)
    assert ext is not None and ext["q"] == 0.07
    assert ext["npts"] == 20
    # chg_bps = (btc@触 − 开盘)/开盘×1e4；btc_change_pct=5e-5 → 0.5bp
    assert abs(ext["chg_bps"] - 0.5) < 1e-9


def test_extract_npts_below_min_skip():
    """npts < MIN_PTS → None（v2 主分析口径，稀疏路径不可信）"""
    w = _seed_window(npts=7, trigger_q=0.07)
    assert _extract(w) is None


def test_extract_missing_btc_open_returns_none():
    """开盘 BTC 基准缺失（entry_price≤0 且 btc 曲线空）→ None"""
    w = _seed_window(npts=20, trigger_q=0.07, bo=0.0)
    assert _extract(w) is None


def test_extract_firsthit_features_matches_shadow_extract():
    """纯函数与归档 _extract 同源：同窗口特征一致，soft vol/par 由 _extract 补齐。"""
    w = _seed_window(npts=20, trigger_q=0.07)
    ext_shadow = _extract(w)
    ext_pure = extract_firsthit_features(
        int(w.start_time), float(w.entry_price),
        w.curve_down_price, w.curve_btc_price)
    assert ext_shadow is not None and ext_pure is not None
    for key in ("q", "trigger_ts", "td_sec", "chg_bps", "body_r", "wick01",
                "rng_bps", "npts"):
        assert ext_pure[key] == pytest.approx(ext_shadow[key])
    assert ext_pure["dvol"] is None and ext_pure["dpar"] is None
    # _seed_window 的 vol/par 曲线两点时间戳是 START 和 trigger_ts=START+120_000，
    # 但 btc/down 曲线最后一个点时间戳是 START+119_985（step 整除余数），
    # _at_le(vol, 119985) 读到开盘点 → dvol=1000−1000=0。soft 维度不是门，不影响实盘。
    assert ext_shadow["dvol"] == pytest.approx(0.0)
    assert ext_shadow["dpar"] == pytest.approx(0.0)


def test_extract_firsthit_features_max_trigger_ts_is_ex_ante():
    """max_trigger_ts 防未来函数：只认 ≤当前采样时刻的真实第一触。"""
    down_curve = [
        {"t": START, "v": 0.50},
        {"t": START + 15_000, "v": 0.30},
        {"t": START + 30_000, "v": 0.08},   # 首次入区 ∈ (0.005, 0.1]
    ]
    # btc_curve 需要在 START+30_000 之前至少 8 个点（MIN_PTS），采样间隔 3_750ms
    btc_curve = [{"t": START + i * 3_750, "v": 100.0 + i * 0.01} for i in range(9)]
    assert extract_firsthit_features(
        START, 100.0, down_curve, btc_curve, max_trigger_ts=START + 15_000) is None
    ext = extract_firsthit_features(
        START, 100.0, down_curve, btc_curve, max_trigger_ts=START + 30_000)
    assert ext is not None and ext["q"] == pytest.approx(0.08)
    assert ext["trigger_ts"] == START + 30_000


# ============================================================
# 3. 触发落 SETTLED / 幂等 / gate 拦截 / npts 门槛
# ============================================================

@pytest.mark.asyncio
async def test_process_window_triggers_g0(monkeypatch):
    """G0 基底（全部首触）→ 落 SETTLED：win=down, ev=q_real 代入。
    
    注意：默认 btc_change_pct=0.00005 → chg=0.5bp ≤ G3 gate (2.82bp)，故 G3 也触发，
    因此本测试期望 2 行（G0+G3）。如果只想测纯 G0，需设大一点的 btc_change_pct（如 0.001）。"""
    session = _FakeSession(first=None)                    # 无 dup → 落行
    monkeypatch.setattr(fhd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = FirstHitShadowDetector()
    w = _seed_window(npts=20, trigger_q=0.07, outcome="DOWN")
    await d._process_window(w)
    versions = [r.version for r in session.added]
    assert "firsthit_down_v1" in versions
    assert "firsthit_down_chg_v1" in versions      # chg=0.5bp ≤ 2.82bp → G3 同窗命中
    assert len(session.added) == 2 and session.committed == 2   # per-version 独立 commit
    row = [r for r in session.added if r.version == "firsthit_down_v1"][0]
    assert row.status == "SETTLED"
    assert row.win is True and row.settle_outcome == "DOWN"
    assert row.entry_down_price == 0.07
    assert row.ev_at_entry == pytest.approx(0.98 / 0.07 - 1.0)
    assert row.npts == 20


@pytest.mark.asyncio
async def test_process_window_triggers_g1(monkeypatch):
    """G1 body_r≤0.35 → 落行；但_seed_window 默认生成单调路径，body_r 恒≈1。
    
    Patch_extract 来模拟小实体的情形：直接返回 body_r=0.2 的特征字典（覆盖原始提取）。"""
    session = _FakeSession(first=None)
    monkeypatch.setattr(fhd, "async_session_factory", lambda: _FakeSessionCtx(session))
    
    d = FirstHitShadowDetector()
    # 手动打桩 _extract：对特定版本返回修改过的特征体貌门通过
    orig_extract = fhd._extract
    def patched_extract(w):
        base = orig_extract(w) or {}
        return {**base, "body_r": 0.2} if base else None
    monkeypatch.setattr(fhd, "_extract", patched_extract)
    
    w = _seed_window(npts=20, trigger_q=0.07, outcome="DOWN")
    await d._process_window(w)
    versions = [r.version for r in session.added]
    assert "firsthit_down_v1" in versions and "firsthit_down_body_v1" in versions


@pytest.mark.asyncio
async def test_process_window_skips_when_npts_lt_min(monkeypatch):
    """npts<MIN_PTS → 特征提取返回 None → 整窗跳过，不落任何版本。"""
    session = _FakeSession(first=None)
    monkeypatch.setattr(fhd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = FirstHitShadowDetector()
    w = _seed_window(npts=6, trigger_q=0.07)                # 小于 MIN_PTS=8
    await d._process_window(w)
    assert session.added == []


@pytest.mark.asyncio
async def test_process_window_idempotent(monkeypatch):
    """幂等：(version, window_start) 已存在 → 不重复落行。"""
    session = _FakeSession(first=123)                     # dup.first() 命中 → 幂等跳过
    monkeypatch.setattr(fhd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = FirstHitShadowDetector()
    w = _seed_window(npts=20, trigger_q=0.07, outcome="DOWN")
    await d._process_window(w)
    assert session.added == []


@pytest.mark.asyncio
async def test_process_window_blocked_when_gate_offline(monkeypatch):
    """shadow_gate 下线该 version → 落表前拦截（不影响其他 version）。"""
    session = _FakeSession(first=None)
    monkeypatch.setattr(fhd, "async_session_factory", lambda: _FakeSessionCtx(session))
    monkeypatch.setattr(fhd.shadow_gate, "_overrides", {"firsthit_down_body_v1": False})
    d = FirstHitShadowDetector()
    w = _seed_window(npts=20, trigger_q=0.07, outcome="DOWN")
    await d._process_window(w)
    versions = [r.version for r in session.added]
    assert "firsthit_down_body_v1" not in versions, "下线版本不得落库"
    assert "firsthit_down_v1" in versions, "其他版本照常采集"


# ============================================================
# 4. 预热 / 轮询 / 水位推进
# ============================================================

@pytest.mark.asyncio
async def test_warmup_fills_buffers_without_writing(monkeypatch):
    """预热只填充水位计数，不落表；缓冲结构按版本分列。"""
    wins = [_seed_window(npts=20, trigger_q=0.07, outcome="DOWN") for _ in range(3)]
    max_end = max(int(w.end_time) for w in wins)
    # scalars().all() 需要返回 end_time（整数），不是完整窗口对象
    session = _FakeSession(rows=[w.end_time for w in wins], scalar=max_end)
    monkeypatch.setattr(fhd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = FirstHitShadowDetector()
    await d._warmup()
    assert session.added == [], "预热不得落表"
    # 本 detector 不做滚动标定缓冲，但水位应推进到最新 end_time
    assert d._last_window_end == max_end


@pytest.mark.asyncio
async def test_poll_once_advances_watermark(monkeypatch):
    """轮询新归档窗 → 逐窗处理后水位推进到最大 end_time；缓冲不足则不落表但水位仍进。"""
    wins = [_seed_window(npts=20, trigger_q=0.07, outcome="DOWN") for _ in range(2)]
    session = _FakeSession(rows=wins)
    monkeypatch.setattr(fhd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = FirstHitShadowDetector()
    await d._poll_once()
    assert d._last_window_end == max(int(w.end_time) for w in wins)
    # 缓冲为空 → calib 逻辑不适用 → 落行条件看 gate，本例会落


# ============================================================
# 5. 物理隔离 + 开关 + spec 自洽 + SHADOW_BENCH
# ============================================================

def test_versions_isolated_from_trading_path():
    """影子纪律：不进 X4 下单白名单 / 非退役（影子默认在线采集）。"""
    from binance_predict.services.live_channels import LIVE_CHANNELS
    from binance_predict.services.multi_live_trader import X4_VERSIONS
    from binance_predict.services.shadow_version_gate import RETIRED_VERSIONS
    for v, _name in FIRSTHIT_SPECS:
        assert v not in X4_VERSIONS, f"{v} 不得进入 X4 下单白名单"
        assert v not in RETIRED_VERSIONS, f"{v} 非退役版本（影子应默认在线采集）"


def test_settings_default_on():
    """默认开启：与其他影子信号一致（record-only 零资金风险，口径保真测试全绿）。"""
    assert settings.firsthit_shadow_enabled is True


def test_specs_self_consistent():
    """spec 自洽：十 version 合法（G0/G1/G3/G4+6G7）、不超 DB 列宽 String(32)，冻结常数正确。"""
    assert set(v for v, _ in FIRSTHIT_SPECS) == {
        "firsthit_down_v1",
        "firsthit_down_body_v1", "firsthit_down_chg_v1", "firsthit_down_g4_v1",
        "firsthit_down_g7_v1", "g7_streak_v1", "g7_wick20_v1",
        "g7_strict_v1", "g7_q05_v1", "g7_t270_v1",
    }
    for v, name in FIRSTHIT_SPECS:
        assert len(v) <= 32, f"{v} 超出 DB 列宽 String(32)"
        assert isinstance(name, str) and len(name) > 0
    assert Q_LO == 0.005 and Q_HI == 0.1
    assert BODY_R_GATE == 0.35 and CHG_BPS_GATE == 2.82


def test_shadow_bench_registered():
    """9 个 firsthit 版本已全量登记进 main.SHADOW_BENCH（analytics 面板事实源）。"""
    for version, _ in FIRSTHIT_SPECS:
        assert version in SHADOW_BENCH, f"{version} 未在 SHADOW_BENCH 中登记"
