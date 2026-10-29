"""动态仓位纯函数单测：% 基准本金 → 单笔金额 + 硬界。"""
from __future__ import annotations

import pytest

from binance_predict.services.position_sizing import (
    DEFAULT_AMOUNT_CAP_USDT,
    MAX_AMOUNT_CAP_USDT,
    MIN_AMOUNT_USDT,
    resolve_amount,
    validate_base_equity,
    validate_pct,
)


def _resolve(**kw):
    base = dict(pct=0.01, base_equity=200.0, requested_amount=2.0,
                base_amount=2.0, cap_usdt=None)
    base.update(kw)
    return resolve_amount(**base)


def test_pct_off_returns_none():
    """pct=None（通道未启用 % 模式）→ 调用方保持固定金额。"""
    assert _resolve(pct=None) is None


def test_missing_base_equity_returns_none():
    assert _resolve(base_equity=None) is None


def test_basic_amount():
    d = _resolve(pct=0.01, base_equity=200.0)   # 200 × 1% = 2.0
    assert d is not None and d.amount == 2.0


def test_preserves_call_site_multiplier():
    """调用点乘数（如高波半仓 0.5）必须保留。"""
    d = _resolve(pct=0.02, base_equity=200.0, requested_amount=1.0, base_amount=2.0)
    assert d is not None
    assert d.ratio == 0.5
    assert d.amount == 2.0          # 200×2%×0.5


def test_cap_truncates():
    d = _resolve(pct=0.05, base_equity=200.0, cap_usdt=3.0)
    assert d is not None and d.amount == 3.0


def test_cap_hard_limit():
    d = _resolve(pct=0.05, base_equity=5000.0, cap_usdt=100.0)  # 250 → 硬帽 50
    assert d is not None and d.amount == MAX_AMOUNT_CAP_USDT and d.cap == MAX_AMOUNT_CAP_USDT


def test_floor_quantization_never_rounds_up():
    d = _resolve(pct=0.0123, base_equity=200.0)   # 2.46 → 2.4（向下）
    assert d is not None and d.amount == 2.4


def test_below_floor_flagged_and_clamped():
    d = _resolve(pct=0.002, base_equity=20.0)     # 0.04 → 抬到 0.1 并标记
    assert d is not None
    assert d.amount == MIN_AMOUNT_USDT
    assert d.below_floor is True


def test_default_cap_applied_when_missing():
    d = _resolve(pct=0.05, base_equity=2000.0)    # 100 → 默认 cap 20
    assert d is not None and d.amount == DEFAULT_AMOUNT_CAP_USDT


@pytest.mark.parametrize("pct,equity", [(0, 200.0), (-0.01, 200.0),
                                        (float("nan"), 200.0),
                                        (0.01, 0)])
def test_invalid_inputs_return_none_or_zero(pct, equity):
    d = _resolve(pct=pct, base_equity=equity)
    assert d is None or d.amount >= MIN_AMOUNT_USDT


def test_validate_pct_bounds():
    assert validate_pct(0.002, 0.002, 0.05) == 0.002
    with pytest.raises(ValueError):
        validate_pct(0.051, 0.002, 0.05)
    with pytest.raises(ValueError):
        validate_pct(0.0, 0.002, 0.05)


def test_validate_base_equity_bounds():
    assert validate_base_equity(200) == 200.0
    with pytest.raises(ValueError):
        validate_base_equity(5)
    with pytest.raises(ValueError):
        validate_base_equity(9999)


def test_tiny_multiplier_never_amplifies():
    """乘数只降不升：requested 远小于 base 时不得因下限被抬回放大金额。"""
    d = _resolve(pct=0.05, base_equity=200.0, requested_amount=0.2, base_amount=10.0)
    assert d is not None
    assert d.ratio == 0.02          # 0.2/10，不抬到 0.05
    assert d.amount == 0.2          # 200×5%×0.02 = 0.2（≤ 请求的 0.2）
    assert d.amount <= 0.2 + 1e-9
