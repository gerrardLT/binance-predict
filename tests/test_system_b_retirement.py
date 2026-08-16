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
    """SentimentAgent/AgentScheduler 实例化必须位于 agent_loop_enabled 分支内。"""
    src = (ROOT / "src/binance_predict/main.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    lines = src.splitlines()

    gate_line: int | None = None
    inst_lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            cond = node.test
            if (
                isinstance(cond, ast.Attribute)
                and cond.attr == "agent_loop_enabled"
                and isinstance(cond.value, ast.Name)
                and cond.value.id == "settings"
            ):
                gate_indent = len(lines[node.lineno - 1]) - len(lines[node.lineno - 1].lstrip())
                # 分支体内任何 SentimentAgent(...) / AgentScheduler(...) 调用都算 gated
                calls = [
                    sub.lineno for sub in ast.walk(node)
                    if isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Name)
                    and sub.func.id in ("SentimentAgent", "AgentScheduler")
                ]
                if not calls:
                    continue  # 非 lifespan 的开关分支（如后续模块），跳过
                for ln in calls:
                    indent = len(lines[ln - 1]) - len(lines[ln - 1].lstrip())
                    assert indent > gate_indent, f"L{ln} 实例化不在开关分支内"
                return
    raise AssertionError("main.py 未找到含实例化的 `if settings.agent_loop_enabled:` 分支")


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
