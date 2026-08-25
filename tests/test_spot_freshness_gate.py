"""R3 喂价新鲜度闸 + health 喂价停摆告警的单元测试（2026-08-25 风险评审）。

覆盖：
- MarketDataStore.mid_price_age_s / fresh_mid_price 的新鲜度语义
- _safe_cached_mid_price 拒绝回退陈旧缓存价（REST 失败时宁可顺延）
- derive_alerts 的 SPOT_FEED_STALE 告警（None=不评估 / 有限值 / inf=从未更新）
"""

from __future__ import annotations

import time

from binance_predict.config.settings import settings
from binance_predict.services.data_collector import (
    BinanceDataCollector,
    MarketDataStore,
)
from binance_predict.services.health import derive_alerts


# ============================================================
# MarketDataStore 新鲜度能力
# ============================================================

def _store_with_price(age_s: float | None) -> MarketDataStore:
    """构造带报价的 store；age_s=None 表示从未收到更新。"""
    store = MarketDataStore(best_bid=100.0, best_ask=102.0)
    if age_s is not None:
        store.last_ws_spot_update = time.time() - age_s
    return store


def test_mid_price_age_s_none_when_never_updated() -> None:
    assert MarketDataStore().mid_price_age_s() is None


def test_mid_price_age_s_returns_elapsed() -> None:
    store = _store_with_price(age_s=30.0)
    age = store.mid_price_age_s()
    assert age is not None and 29.0 <= age <= 32.0


def test_fresh_mid_price_returns_price_when_fresh() -> None:
    store = _store_with_price(age_s=1.0)
    assert store.fresh_mid_price() == 101.0


def test_fresh_mid_price_rejects_stale() -> None:
    store = _store_with_price(age_s=settings.spot_price_max_age_s + 5)
    assert store.fresh_mid_price() == 0.0
    # 裸 mid_price 仍可见（展示类端点用），闸门只作用于 fresh 版
    assert store.mid_price == 101.0


def test_fresh_mid_price_rejects_never_updated() -> None:
    store = MarketDataStore(best_bid=100.0, best_ask=102.0)  # 有价但无更新时间
    assert store.fresh_mid_price() == 0.0


def test_fresh_mid_price_custom_max_age() -> None:
    store = _store_with_price(age_s=40.0)
    assert store.fresh_mid_price(max_age_s=60.0) == 101.0
    assert store.fresh_mid_price(max_age_s=10.0) == 0.0


# ============================================================
# _safe_cached_mid_price：REST 失败回退也过新鲜度闸
# ============================================================

def test_safe_cached_returns_fresh_price() -> None:
    collector = BinanceDataCollector()
    collector.store.best_bid = 100.0
    collector.store.best_ask = 102.0
    collector.store.last_ws_spot_update = time.time()
    assert collector._safe_cached_mid_price() == 101.0


def test_safe_cached_rejects_stale_price() -> None:
    collector = BinanceDataCollector()
    collector.store.best_bid = 100.0
    collector.store.best_ask = 102.0
    collector.store.last_ws_spot_update = time.time() - 3600  # 1 小时前的陈旧价
    assert collector._safe_cached_mid_price() == 0.0


# ============================================================
# derive_alerts：SPOT_FEED_STALE
# ============================================================

def _alert_kwargs(**overrides) -> dict:
    kwargs = dict(
        window_continuity={"last_window_age_s": 60.0, "gap_count": 0, "recent_count": 10},
        predict_stats={"total": 0, "matched": 0, "active_pattern_count": 0},
        phase_ages={},
        queue_depth=None,
        llm=None,
        consecutive_failures=None,
        has_memory=True,
    )
    kwargs.update(overrides)
    return kwargs


def test_spot_feed_stale_not_evaluated_when_none() -> None:
    alerts = derive_alerts(**_alert_kwargs(spot_feed_age_s=None))
    assert not any(a.code == "SPOT_FEED_STALE" for a in alerts)


def test_spot_feed_stale_fresh_no_alert() -> None:
    alerts = derive_alerts(**_alert_kwargs(spot_feed_age_s=5.0))
    assert not any(a.code == "SPOT_FEED_STALE" for a in alerts)


def test_spot_feed_stale_critical_when_over_limit() -> None:
    alerts = derive_alerts(
        **_alert_kwargs(spot_feed_age_s=settings.spot_price_max_age_s + 10)
    )
    hits = [a for a in alerts if a.code == "SPOT_FEED_STALE"]
    assert len(hits) == 1 and hits[0].level == "CRITICAL"


def test_spot_feed_stale_critical_when_never_updated() -> None:
    alerts = derive_alerts(**_alert_kwargs(spot_feed_age_s=float("inf")))
    hits = [a for a in alerts if a.code == "SPOT_FEED_STALE"]
    assert len(hits) == 1 and hits[0].level == "CRITICAL"
    assert "从未收到更新" in hits[0].message
