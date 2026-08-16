"""场景参数的唯一定义（detector 与科学回测引擎共用）。

场景①②判定参数自此有版本化载体（db.SceneParamVersion）：
- 180 天发现集→验证集盲验的原始口径 = DEFAULT_SCENE_PARAMS（v1-20260816）
- LLM 研究员提出的新假设 = 对本结构的覆盖子集，经科学回测裁决 + 人工放行后
  以新版本（SHADOW→ACTIVE）生效——线上判定参数永不热改（验证跟随冻结）。
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping


@dataclass(frozen=True)
class SceneParams:
    """场景①②判定参数集（与 fake_breakout_signals 场景口径一一对应）。

    - close_pos_min: 场景①收盘位置 (C-L)/(H-L) 下限（光头程度）
    - vol_ratio_min: 场景②量比下限（本周期量 / 前 vol_ma_window 根均量）
    - vol_ma_window: 均量窗口（根，不含当前周期）
    - eps: 破位阈值（mid 越过位势 ×(1±eps) 判破位）
    - level_lookbacks: 级别 → 回看 5m 窗口 closes 数（位势极值窗口）
    """

    close_pos_min: float = 0.85
    vol_ratio_min: float = 2.0
    vol_ma_window: int = 20
    eps: float = 0.0005
    level_lookbacks: Mapping[str, int] = field(default_factory=lambda: {"4h": 48})

    def to_params_json(self) -> dict:
        return {
            "close_pos_min": self.close_pos_min,
            "vol_ratio_min": self.vol_ratio_min,
            "vol_ma_window": self.vol_ma_window,
            "eps": self.eps,
            "level_lookbacks": dict(self.level_lookbacks),
        }

    @classmethod
    def from_params_json(cls, d: Mapping | None) -> "SceneParams":
        """从 scene_param_versions.params JSON 重建；容忍缺省键（回退默认）。"""
        base = DEFAULT_SCENE_PARAMS
        if not d:
            return base
        return cls(
            close_pos_min=float(d.get("close_pos_min", base.close_pos_min)),
            vol_ratio_min=float(d.get("vol_ratio_min", base.vol_ratio_min)),
            vol_ma_window=int(d.get("vol_ma_window", base.vol_ma_window)),
            eps=float(d.get("eps", base.eps)),
            level_lookbacks=dict(d.get("level_lookbacks", base.level_lookbacks)),
        )

    def with_overrides(self, overrides: Mapping) -> "SceneParams":
        """按 LLM 假设的覆盖子集生成新参数（未覆盖键保持不变）。"""
        return replace(self, **{
            k: v for k, v in dict(overrides).items()
            if k in ("close_pos_min", "vol_ratio_min", "vol_ma_window", "eps", "level_lookbacks")
        })


# 现行线上口径（v1-20260816 种子的字段来源；180 天验证集：场景① 63.6% / ② 57.8%）
DEFAULT_SCENE_PARAMS = SceneParams()
