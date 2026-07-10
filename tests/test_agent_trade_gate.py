"""Agent 自动交易总开关的回归测试。"""

from binance_predict.services.agent_logic import should_trade


def test_disabled_auto_trade_blocks_high_confidence_direction() -> None:
    allowed, reason = should_trade(
        "UP", confidence=0.99, threshold=0.6, auto_trade_enabled=False
    )

    assert allowed is False
    assert "总开关" in reason


def test_enabled_auto_trade_keeps_existing_confidence_rule() -> None:
    allowed, _ = should_trade(
        "DOWN", confidence=0.61, threshold=0.6, auto_trade_enabled=True
    )

    assert allowed is True
