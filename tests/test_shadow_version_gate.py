"""影子版本开关 gate 测试：默认在线语义 + set/refresh + DB 故障保守 + 拦截集成 + API。

不触真实 DB：session 全用替身。gate 是全部影子检测器落库前的统一闸门
（8 处落库点接入），本测试锁定其核心语义：
    1. 无覆盖行 → 默认在线（部署零影响，与既有行为一致）
    2. set_enabled 下线/上线 → upsert 落库 + 本进程缓存立即生效
    3. refresh 全量加载覆盖表；DB 故障保守维持现状（不停采集）
    4. 检测器集成：下线版本 _record_signal 直接拒绝（不查库不落库）
    5. API：白名单校验（SHADOW_BENCH）+ gate 写入 + analytics enabled 下发
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import binance_predict.services.shadow_version_gate as svg
from binance_predict.services.shadow_version_gate import ShadowVersionGate


class _FakeResult:
    def __init__(self, rows: list | None = None, scalar=None) -> None:
        self._rows, self._scalar = rows or [], scalar

    def scalars(self) -> "_FakeResult":
        return self

    def all(self) -> list:
        return self._rows

    def scalar_one_or_none(self):
        return self._scalar


class _FakeSession:
    def __init__(self, rows: list | None = None, scalar=None) -> None:
        self.rows, self.scalar = rows or [], scalar
        self.added: list = []
        self.committed = False

    async def execute(self, _stmt) -> _FakeResult:
        return _FakeResult(self.rows, self.scalar)

    def add(self, obj) -> None:
        self.added.append(obj)

    async def commit(self) -> None:
        self.committed = True


class _FakeSessionCtx:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session

    async def __aenter__(self) -> _FakeSession:
        return self._session

    async def __aexit__(self, *exc) -> bool:
        return False


# ============================================================
# gate 核心语义
# ============================================================

def test_default_enabled_without_override() -> None:
    """无覆盖行 → 默认在线（部署零影响，与既有采集行为完全一致）。"""
    g = ShadowVersionGate()
    assert g.is_enabled("hm_touch_down_v1") is True
    assert g.is_enabled("combo_p1_v1") is True
    assert g.is_enabled("any_unknown_version") is True
    assert g.all_overrides() == {}


@pytest.mark.asyncio
async def test_set_enabled_offline_then_online(monkeypatch) -> None:
    """下线 → is_enabled False + insert 落库；再上线 → update 路径 + True。"""
    session = _FakeSession(scalar=None)  # 无既有行 → insert
    monkeypatch.setattr(svg, "async_session_factory", lambda: _FakeSessionCtx(session))
    g = ShadowVersionGate()
    await g.set_enabled("hm_touch_down_v1", False)
    assert g.is_enabled("hm_touch_down_v1") is False
    assert len(session.added) == 1
    row = session.added[0]
    assert row.version == "hm_touch_down_v1" and row.enabled is False
    assert session.committed
    # 其它版本不受影响（下线是 version 级隔离）
    assert g.is_enabled("combo_p1_v1") is True

    # 再上线：已有行 → update 路径
    session2 = _FakeSession(scalar=row)
    monkeypatch.setattr(svg, "async_session_factory", lambda: _FakeSessionCtx(session2))
    await g.set_enabled("hm_touch_down_v1", True)
    assert g.is_enabled("hm_touch_down_v1") is True
    assert row.enabled is True and session2.added == []  # update 不新增行


@pytest.mark.asyncio
async def test_refresh_loads_overrides(monkeypatch) -> None:
    """refresh 全量加载覆盖表：False 行生效下线，无覆盖行仍默认在线。"""
    session = _FakeSession(rows=[("hm_touch_down_v1", False), ("x4_v1", True)])
    monkeypatch.setattr(svg, "async_session_factory", lambda: _FakeSessionCtx(session))
    g = ShadowVersionGate()
    await g.refresh()
    assert g.is_enabled("hm_touch_down_v1") is False
    assert g.is_enabled("x4_v1") is True
    assert g.is_enabled("combo_p1_v1") is True


@pytest.mark.asyncio
async def test_refresh_db_failure_conservative(monkeypatch) -> None:
    """DB 故障 → refresh 不抛、保守维持现状（开关表不可用不停影子采集）。"""
    def _boom():
        raise RuntimeError("db down")
    monkeypatch.setattr(svg, "async_session_factory", _boom)
    g = ShadowVersionGate()
    await g.refresh()  # 不应抛异常
    assert g.is_enabled("anything") is True  # 首次加载失败 → 全默认在线
    # 已有缓存时故障不清空（维持最后一次成功加载的状态）
    g._overrides = {"hm_touch_down_v1": False}
    await g.refresh()
    assert g.is_enabled("hm_touch_down_v1") is False


# ============================================================
# 检测器集成：下线版本落库被拦截（combo 为代表，8 处落库点同模式）
# ============================================================

@pytest.mark.asyncio
async def test_combo_record_signal_blocked_when_offline(monkeypatch) -> None:
    """combo_p1_v1 下线 → _record_signal 直接拒绝（不查存在性、不落库）。"""
    import binance_predict.services.combo_shadow_detector as csd

    session = _FakeSession(scalar=None)  # 若未被拦截，此桩会导致落行
    monkeypatch.setattr(csd, "async_session_factory", lambda: _FakeSessionCtx(session))
    monkeypatch.setattr(csd.shadow_gate, "_overrides", {"combo_p1_v1": False})
    d = csd.ComboShadowDetector(collector=None, pm_15m_latest={})
    bar = {"open_time": 1_700_000_000_000 // 900_000 * 900_000, "open": 100.0,
           "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1.0}
    spec_p1 = next(s for s in d._specs if s["version"] == "combo_p1_v1")
    added = await d._record_signal(session, spec_p1, bar, {}, 0)
    assert added is False and session.added == [], "下线版本不得落库"
    # 同检测器其它版本不受影响（version 级隔离）
    assert csd.shadow_gate.is_enabled("combo_p2_v1") is True


@pytest.mark.asyncio
async def test_nextbar_record_signal_blocked_when_offline(monkeypatch) -> None:
    """nextbar 15m 冠军下线 → _record_signal 拒绝（K 线族同模式代表之二）。"""
    import binance_predict.services.nextbar_shadow_detector as nsd
    from binance_predict.discovery.features import build_feature_matrix

    session = _FakeSession(scalar=None)
    monkeypatch.setattr(nsd, "async_session_factory", lambda: _FakeSessionCtx(session))
    monkeypatch.setattr(nsd.shadow_gate, "_overrides", {"nb_zschamp_15m_v1": False})
    d = nsd.NextbarShadowDetector(collector=None, pm_15m_latest={}, pm_5m_info={})
    t0 = 1_700_000_000_000 // 900_000 * 900_000
    rows = [{"open_time": t0 + i * 900_000, "open": 100.0, "high": 100.05,
             "low": 99.95, "close": 100.0, "volume": 1.0} for i in range(40)]
    fm = build_feature_matrix(nsd._to_klines(rows, 900_000), 900_000)
    spec15 = next(s for s in d._specs if s["version"] == "nb_zschamp_15m_v1")
    added = await d._record_signal(session, spec15, rows[-1], fm, len(rows) - 1)
    assert added is False and session.added == []


# ============================================================
# API：白名单校验 + gate 写入 + 响应语义
# ============================================================

@pytest.mark.asyncio
async def test_shadow_toggle_api_rejects_unknown_version() -> None:
    """白名单=SHADOW_BENCH：未知版本 422 拒绝（防脏数据写覆盖表）。"""
    import binance_predict.main as m
    from fastapi import HTTPException

    from binance_predict.models.schemas import ToggleShadowRequest
    req = ToggleShadowRequest(version="nonexistent_v9", enabled=False)
    with pytest.raises(HTTPException) as ei:
        await m.shadow_toggle(req, None)
    assert ei.value.status_code == 422


@pytest.mark.asyncio
async def test_shadow_toggle_api_writes_gate(monkeypatch) -> None:
    """合法版本 → gate.set_enabled 被调用 + 响应含中文语义。"""
    import binance_predict.main as m

    from binance_predict.models.schemas import ToggleShadowRequest
    calls: list[tuple[str, bool]] = []

    async def _fake_set(version: str, enabled: bool) -> None:
        calls.append((version, enabled))

    monkeypatch.setattr(m.shadow_gate, "set_enabled", _fake_set)
    req = ToggleShadowRequest(version="hm_touch_down_v1", enabled=False)
    out = await m.shadow_toggle(req, None)
    assert calls == [("hm_touch_down_v1", False)]
    assert out["version"] == "hm_touch_down_v1" and out["enabled"] is False
    assert "下线" in out["message"]

    req2 = ToggleShadowRequest(version="combo_p1_v1", enabled=True)
    out2 = await m.shadow_toggle(req2, None)
    assert out2["enabled"] is True and "上线" in out2["message"]
