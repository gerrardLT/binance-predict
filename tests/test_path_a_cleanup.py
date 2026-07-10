"""路径 A 遗留契约和依赖的清理回归测试。"""

from pathlib import Path


def test_path_a_models_and_dependencies_are_removed() -> None:
    schemas = Path("src/binance_predict/models/schemas.py").read_text(encoding="utf-8")
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    for symbol in (
        "DecisionOutput",
        "ReviewOutput",
        "MarketSnapshot",
        "PredictionRecord",
        "PredictionResult",
        "CustomRule",
        "ReviewMemory",
        "TradeOrderRecord",
    ):
        assert f"class {symbol}" not in schemas

    assert "redis[hiredis]" not in pyproject
    assert "apscheduler" not in pyproject
