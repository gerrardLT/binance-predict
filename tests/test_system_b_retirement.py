"""系统B（情绪 Agent Loop）退役回归测试（M0，2026-08-16 拍板）。

退役方式：settings 开关 gate（类与表保留只读存档，可随时翻回 True 恢复）。
本文件验证退役的关键不变量：
- 开关默认值：agent_loop_enabled=False / pattern_reeval_enabled=False
- main.py 语法完整且 SentimentAgent 实例化位于 agent_loop_enabled 分支内
  （文本级检查：实例化行缩进深于 `if settings.agent_loop_enabled:` 行）
- 迁移文件语法完整
- SceneParamVersion 模型可导入（v1 种子由 alembic 提供；create_all 环境无种子，
  服务层必须容忍无 ACTIVE 行并回退默认常量——由 M2/M3 的 DEFAULT_SCENE_PARAMS 兑现）
- 场景信号系统（系统A）不受退役影响：test_fake_breakout_patterns.py 全量即回归保障
"""

from __future__ import annotations

import ast
from pathlib import Path

from binance_predict.config.settings import settings
from binance_predict.db.models import SceneParamVersion

ROOT = Path(__file__).resolve().parents[1]


def test_retirement_switches_default_off() -> None:
    """退役默认态：预测循环与模式池重回测均关闭。"""
    assert settings.agent_loop_enabled is False
    assert settings.pattern_reeval_enabled is False


def test_main_instantiation_gated_by_switch() -> None:
    """2026-10-19 更新：系统B 代码已整体移除——main.py 不得再引用任何系统B 模块。

    （旧口径「实例化须在 agent_loop_enabled 分支内」随分支一起删除；
    新口径：imports/调用点零残留，sentiment_windows 归档与 momentum 不受影响。）
    """
    src = (ROOT / "src/binance_predict/main.py").read_text(encoding="utf-8")
    ast.parse(src)
    for forbidden in (
        "from .services.sentiment_agent import",
        "from .services.agent_scheduler import",
        "from .services.pattern_reevaluator import",
        "SentimentAgent(",
        "AgentScheduler(",
    ):
        assert forbidden not in src, f"main.py 残留系统B 引用: {forbidden}"
    # 保留项仍在：情绪窗口归档（场景信号位势源）与动量预测端点
    assert "_sentiment_window_archiver" in src
    assert '"/api/sentiment/momentum-predict"' in src


def test_retirement_migration_syntax() -> None:
    """迁移文件语法完整（存档隔离 + scene_param_versions + v1 种子）。"""
    p = ROOT / "alembic/versions/o5f6g7h8i9j0_retire_system_b_and_scene_param_versions.py"
    ast.parse(p.read_text(encoding="utf-8"), str(p))


def test_scene_param_version_model_importable() -> None:
    """SceneParamVersion 表定义完整（LLM 自进化体系载体）。"""
    assert SceneParamVersion.__tablename__ == "scene_param_versions"
    cols = {c.name for c in SceneParamVersion.__table__.columns}
    assert {
        "id", "version", "params", "status", "backtest_report",
        "proposed_by", "reviewed_by", "review_note",
        "created_at", "activated_at", "retired_at",
    } <= cols
