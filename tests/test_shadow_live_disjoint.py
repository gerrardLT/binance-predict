"""R9 影子/实盘集合不相交断言测试（2026-08-25 风险评审）。

此前影子不变量靠"白名单缺席"隐式保证：v3a/v3b 不在 LIVE_CHANNELS
即为纯影子。谁把 v3a 加进白名单，它立刻变可开火 + 可推邮件通道，
且没有任何测试失败——本文件把该不变量显式钉死：

- SHADOW_ONLY_VERSIONS 与 LIVE_CHANNELS 键集合不相交（双向）
- 影子集合非空且均为已登记检测版本（防止集合腐化成假护栏）
- 各消费侧（修复工具的 v3 集合、信号推送 resolver 语义）与显式集合一致
"""

from __future__ import annotations

from binance_predict.services.live_channels import (
    LIVE_CHANNELS,
    SHADOW_ONLY_VERSIONS,
    scene_pattern_to_channel,
)
from binance_predict.services import signal_notify


def test_shadow_and_live_sets_disjoint() -> None:
    """核心不变量：纯影子版本绝不能出现在实盘白名单（双向不相交）。"""
    overlap = SHADOW_ONLY_VERSIONS & set(LIVE_CHANNELS)
    assert overlap == set(), (
        f"影子版本 {overlap} 出现在 LIVE_CHANNELS 白名单——加入即变可开火"
        f"+可推邮件通道；若确要实盘化，请先走实盘门槛评审并更新"
        f" SHADOW_ONLY_VERSIONS 登记"
    )


def test_shadow_set_registered_and_nonempty() -> None:
    """影子集合非空且是冻结登记（防集合被清空后断言恒真的假护栏）。"""
    assert SHADOW_ONLY_VERSIONS == frozenset({
        "quote_contrarian_v3a", "quote_contrarian_v3b",
    })


def test_shadow_versions_never_resolve_to_live_channel() -> None:
    """场景映射器不会把任何影子版本解析成实盘通道（钩子路径无旁门）。"""
    for v in SHADOW_ONLY_VERSIONS:
        assert v not in LIVE_CHANNELS
        assert scene_pattern_to_channel(v) is None


def test_repair_v3_versions_subset_of_shadow_registry() -> None:
    """污染修复工具的 v3 处理集合与影子登记一致（防两处定义漂移）。"""
    from binance_predict.services.archive_contamination_repair import (
        QE_V3_VERSIONS,
    )
    assert set(QE_V3_VERSIONS) <= SHADOW_ONLY_VERSIONS


def test_live_resolver_default_rejects_shadow_versions() -> None:
    """resolver 未注入（装配失败 fail-safe）时影子版本查询恒 False——
    与白名单缺席双闸语义一致（signal_notify.is_live_enabled 宁少勿多）。"""
    for v in SHADOW_ONLY_VERSIONS:
        assert signal_notify.is_live_enabled(v) is False
