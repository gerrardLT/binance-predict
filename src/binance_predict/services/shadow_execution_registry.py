"""Registry for every shadow version exposed by ``/api/signals/analytics``."""
from __future__ import annotations

from dataclasses import dataclass

from .live_channels import LIVE_CHANNELS, RETIRED_CHANNEL_SPECS
from .shadow_execution_types import EXECUTION_POLICY_VERSION, SourceType


@dataclass(frozen=True)
class ShadowVersionSpec:
    version: str
    source_type: SourceType
    market_period: str
    target_semantics: str
    direction_mode: str
    live_channel: str | None
    live_retired: bool = False
    policy_version: str = EXECUTION_POLICY_VERSION
    family: str = "legacy"
    role: str = "STRATEGY"
    display_name: str | None = None


# Order is API behavior: keep additions explicit rather than sorting this registry.
_VERSION_ROWS: tuple[tuple[str, SourceType, str, str, str], ...] = (
    ("x4_v1", SourceType.MISALIGNMENT, "5m", "next_window", "row"),
    ("quote_momentum_v1", SourceType.MISALIGNMENT, "5m", "same_window", "row"),
    ("quote_contrarian_v1", SourceType.MISALIGNMENT, "5m", "same_window", "row"),
    ("x4_v2", SourceType.MISALIGNMENT, "5m", "next_window", "row"),
    ("quote_momentum_v2", SourceType.MISALIGNMENT, "5m", "same_window", "row"),
    ("quote_contrarian_v2", SourceType.MISALIGNMENT, "5m", "same_window", "row"),
    ("quote_contrarian_z_high_v1", SourceType.MISALIGNMENT, "5m", "same_window", "row"),
    ("quote_contrarian_z_mid_v1", SourceType.MISALIGNMENT, "5m", "same_window", "row"),
    ("x4_v3", SourceType.MISALIGNMENT, "5m", "next_window", "row"),
    ("quote_contrarian_v3a", SourceType.MISALIGNMENT, "5m", "same_window", "row"),
    ("quote_contrarian_v3b", SourceType.MISALIGNMENT, "5m", "same_window", "row"),
    ("quote_contrarian_v4", SourceType.MISALIGNMENT, "5m", "same_window", "row"),
    ("late_night_contrarian_v1", SourceType.MISALIGNMENT, "5m", "same_window", "row"),
    ("late_night_contrarian_v2", SourceType.MISALIGNMENT, "5m", "same_window", "row"),
    ("krev_a_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("krev_b_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("hm_touch_down_v1", SourceType.PATTERN, "15m", "target_bar", "down"),
    ("hm_touch_down_v2", SourceType.PATTERN, "15m", "target_bar", "down"),
    ("rev_p1_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("rev_p2_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("nb_zschamp_15m_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("nb_smaslope_5m_v1", SourceType.KLINE, "5m", "target_bar", "row"),
    ("combo_p1_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("combo_p2_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("combo_p3_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("combo_p4_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("combo_p5_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("s5_deep_z20_v1", SourceType.PATTERN, "15m", "target_bar", "down"),
    ("s1_dyn_sq_v1", SourceType.PATTERN, "15m", "target_bar", "down"),
    ("quote_momentum_v3", SourceType.MISALIGNMENT, "5m", "same_window", "row"),
    ("absorption_follow_td120_v1", SourceType.ABSORPTION, "5m", "same_window", "dynamic"),
    ("absorption_follow_td150_v1", SourceType.ABSORPTION, "5m", "same_window", "dynamic"),
    ("s2_cond_t4_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("s2_cond_t5d_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("scene_bear_exhaust_opt_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("s4_delay60_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("s4_hiconf_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("s1_early2_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("s1_dip45_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("firsthit_down_chg_early180_v1", SourceType.FIRSTHIT, "5m", "same_window_first_touch", "down"),
    ("firsthit_down_k10_early180_v1", SourceType.FIRSTHIT, "5m", "same_window_first_touch", "down"),
    ("process_recovery_down_v1", SourceType.FIRSTHIT, "5m", "same_window", "down"),
    ("firsthit_down_v1", SourceType.FIRSTHIT, "5m", "same_window_first_touch", "down"),
    ("firsthit_down_body_v1", SourceType.FIRSTHIT, "5m", "same_window_first_touch", "down"),
    ("firsthit_down_chg_v1", SourceType.FIRSTHIT, "5m", "same_window_first_touch", "down"),
    ("firsthit_down_k10_v1", SourceType.FIRSTHIT, "5m", "same_window_first_touch", "down"),
    ("firsthit_down_k10_profit_v1", SourceType.FIRSTHIT, "5m", "same_window_first_touch", "down"),
    ("firsthit_down_g4_v1", SourceType.FIRSTHIT, "5m", "same_window_first_touch", "down"),
    ("firsthit_down_g7_v1", SourceType.FIRSTHIT, "5m", "same_window_first_touch", "down"),
    ("g7_streak_v1", SourceType.FIRSTHIT, "5m", "same_window_first_touch", "down"),
    ("g7_wick20_v1", SourceType.FIRSTHIT, "5m", "same_window_first_touch", "down"),
    ("g7_strict_v1", SourceType.FIRSTHIT, "5m", "same_window_first_touch", "down"),
    ("g7_q05_v1", SourceType.FIRSTHIT, "5m", "same_window_first_touch", "down"),
    ("g7_t270_v1", SourceType.FIRSTHIT, "5m", "same_window_first_touch", "down"),
    # 报价错杀版（2026-10-05）：G3 门 ∩ 浅价 q≥0.07 ∩ 触发前 60 采样点 BTC 未上行；
    # forward 母体 ex-ante 扫描候选，仅影子前向验证。
    ("firsthit_down_btcsoft_v1", SourceType.FIRSTHIT, "5m", "same_window_first_touch", "down"),
    # G3v2 双波峰版（2026-10-06 升级）：双波峰利润带原生门，与实盘严格同口径。
    ("firsthit_down_chg_v2", SourceType.FIRSTHIT, "5m", "same_window_first_touch", "down"),
    # 2026-10-13 分桶组合 D/E（q≥0.07 ∩ chg≤3.5bp ∩ 首触 15s 分桶），同名实盘默认 OFF。
    ("firsthit_down_chg35_d_v1", SourceType.FIRSTHIT, "5m", "same_window_first_touch", "down"),
    ("firsthit_down_chg35_e_v1", SourceType.FIRSTHIT, "5m", "same_window_first_touch", "down"),
    ("hm_inside_15m_v2", SourceType.KLINE, "15m", "target_bar", "row"),
    ("ih_inside_15m_v2", SourceType.KLINE, "15m", "target_bar", "row"),
    ("hm_inside_5m_v2", SourceType.KLINE, "5m", "target_bar", "row"),
    # 2026-10-23 用户拍板注册（盘口 EV 验证后）：5m 收顶阳线+长下影阴线两版。
    ("hm_quality_5m_v1", SourceType.KLINE, "5m", "target_bar", "row"),
    ("hm_moderate_5m_v1", SourceType.KLINE, "5m", "target_bar", "row"),
    # CRV 情绪反转族（2026-10-04 研究 run 20261004T083106Z）：15m/5m 突破族押次根
    # 收阴 DOWN + 三个三因子探索/观察级（dnex/os_st4/hi_z3，仅影子）；direction 按行（row）。
    ("crv_hi_brk20_15m_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("crv_brk20_15m_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("crv_brk50_hi_15m_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("crv_dnex_15m_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("crv_brk20_5m_v1", SourceType.KLINE, "5m", "target_bar", "row"),
    ("crv_os_st4_5m_v1", SourceType.KLINE, "5m", "target_bar", "row"),
    # 2026-10-04 AMENDMENT-2 全枚举新发现（观察级，仅影子）
    ("crv_hi_z3_15m_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    # 2026-10-12 复核研究 run 20261012T084631Z「A 信号下破半」（阴线吞没∧破24低→押UP）
    ("crv_dnbrk24_15m_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("crv_dnbrk24_5m_v1", SourceType.KLINE, "5m", "target_bar", "row"),
    # 2026-10-15 反转研究回顾性验证：放量下破 5m 求值→15m 市场押 UP；大实体突破 5m 押 DOWN
    ("crv_dnvol20_15m_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("crv_bodybrk20_5m_v1", SourceType.KLINE, "5m", "target_bar", "row"),
    # KDC 方向确定性族（2026-10-16 研究 run 20261015T153500Z）：15m 极值反转
    # 四版本（破96高收顶/超24高冲顶/收顶3连阳破20高 押 DOWN；破20低3连阴大阴 押
    # UP）；direction 按行（row）。影子行附执行层优化快照（quote_cross_cost/
    # vol_tier/entry_band/q10_*）。
    ("kdc_s1_96hi_15m_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("kdc_s2_ext_15m_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("kdc_s3_3up_15m_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("kdc_s4_3dn_15m_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    # BRK 假突破回归族（2026-10-05 研究 run 20261005T051500Z）：收盘突破前 8h
    # 高点 → 押次根收阴 DOWN；direction 按行（row）。
    ("brkrv_brk8h_15m_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("brkrv_brk8h_5m_v1", SourceType.KLINE, "5m", "target_bar", "row"),
    # 长影短实体检测器按 timeframe 只落一个物理事件；18 个冻结逻辑版本作为标签，
    # 不在此投影表重复展开，避免同窗主版本/严格子集/组件被统计成多次事件。
    ("candle_hm_bull_5m_event_v1", SourceType.KLINE, "5m", "target_bar", "row"),
    ("candle_hm_bull_15m_event_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    # 报价-人群错位族（2026-10-10 研究 run 20261009T191824Z-reversal-state-certainty
    # 盲测确认 + 用户拍板注册影子+实盘，实盘默认 OFF）：窗后半段 gap≤−0.05 → 买 UP；
    # 同窗入场（same_window），direction 按行（row，恒 UP）。
    ("gap_crowd_5m_v1", SourceType.GAP_CROWD, "5m", "same_window", "row"),
    ("gap_crowd_15m_v1", SourceType.GAP_CROWD, "15m", "same_window", "row"),
    # ------------------------------------------------------------------
    # 2026-10-14 P1 补齐：在 LIVE_CHANNELS 中有实盘通道、但在旧账本白名单
    # 漏登记的 22 个通道。登记后：① 实盘订单对账不再 skipped；② 运行时
    # 拦截（日限/互斥/防重）正确记录 assessment 痕迹；③ 漏斗与分析端点正常汇总。
    # ------------------------------------------------------------------
    # 4 个场景实盘通道（15m K线形态/破位）：
    ("scene_bull_exhaust", SourceType.KLINE, "15m", "target_bar", "row"),
    ("scene_bull_exhaust_confirm", SourceType.KLINE, "15m", "target_bar", "row"),
    ("scene_bear_exhaust", SourceType.KLINE, "15m", "target_bar", "row"),
    ("scene_momentum_fade", SourceType.KLINE, "15m", "target_bar", "row"),
    # 18 个蜡烛反转逻辑实盘通道（长影短实体族，同名实盘通道）：
    ("candle_hm_bull_5m_consensus2_shadow_v1", SourceType.KLINE, "5m", "target_bar", "row"),
    ("candle_hm_bull_5m_consensus3_shadow_v1", SourceType.KLINE, "5m", "target_bar", "row"),
    ("candle_hm_bull_15m_union_shadow_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("candle_hm_bull_15m_consensus3_shadow_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("candle_hm_bull_5m_ret5_majority_component_shadow_v1", SourceType.KLINE, "5m", "target_bar", "row"),
    ("candle_hm_bull_5m_ret3_component_shadow_v1", SourceType.KLINE, "5m", "target_bar", "row"),
    ("candle_hm_bull_15m_ret3_component_shadow_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("candle_hm_bull_15m_ma10_component_shadow_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("candle_hm_bull_5m_volume_mid_high_shadow_hyp_v1", SourceType.KLINE, "5m", "target_bar", "row"),
    ("candle_hm_bull_5m_trend4h_up_shadow_hyp_v1", SourceType.KLINE, "5m", "target_bar", "row"),
    ("candle_hm_bull_5m_asia_shadow_hyp_v1", SourceType.KLINE, "5m", "target_bar", "row"),
    ("candle_hm_bull_15m_q45_55_shadow_hyp_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("candle_hm_bull_15m_trend1h_up_shadow_hyp_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("candle_hm_bull_15m_asia_shadow_hyp_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("candidate_5m_ret5_majority_up_upper_bull_l70b30m10_shadow_hyp_v1", SourceType.KLINE, "5m", "target_bar", "row"),
    ("candidate_5m_ret3_up_lower_bear_l50b50m0_shadow_hyp_v1", SourceType.KLINE, "5m", "target_bar", "row"),
    ("candidate_15m_ret5_majority_up_lower_bull_l50b40m10_shadow_hyp_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("candidate_15m_ret3_down_upper_bull_l50b40m10_shadow_hyp_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    # 2026-10-25 补登记（窗口内接盘 wbid 四版 + 布防接盘 excur）：注册时漏登
    # _VERSION_ROWS，导致 /api/signals/analytics?active_only=true 的
    # visible_versions 不含这些版本——即使结算完成也不返回，新信号在信号
    # 分析页任何视图不可见（叠加结算循环 close_time KeyError，已另行修复）。
    # 同窗结算、K 线源、按行 direction 判定。
    ("wbid_d03_k2_5m_v1", SourceType.KLINE, "5m", "same_window", "row"),
    ("wbid_d03_k3_5m_v1", SourceType.KLINE, "5m", "same_window", "row"),
    ("wbid_d05_k2_5m_v1", SourceType.KLINE, "5m", "same_window", "row"),
    ("wbid_d03_k2_fast_5m_v1", SourceType.KLINE, "5m", "same_window", "row"),
    ("excur_bid30_15m_v1", SourceType.KLINE, "15m", "same_window", "row"),
    ("prebar_dn_bull_bid30_15m_v1", SourceType.KLINE, "15m", "same_window", "row"),
    # 2026-10-27 cic 反向格（注册即登记，吸取 wbid/excur 漏登教训）：15m 同窗
    # 结算（第 8 分钟决策、当窗收盘结算）、K 线源、按行 direction 判定。
    ("cic_shallow_15m_v1", SourceType.KLINE, "15m", "same_window", "row"),
    # 2026-10-30 endgame 尾盘买热门（注册即登记）：15m 同窗结算（最后 45s 触发、
    # 本窗收盘结算）、K 线源、direction 由触发时热门侧决定。
    ("endgame_fav_15m_v1", SourceType.KLINE, "15m", "same_window", "row"),
)


def _live_mapping(version: str) -> tuple[str | None, bool]:
    if version in LIVE_CHANNELS:
        return version, False
    if version in RETIRED_CHANNEL_SPECS:
        return version, True
    return None, False


def _family_of(version: str, source: SourceType) -> str:
    if version.startswith("process_recovery_"):
        return "process_recovery"
    if version.startswith(("candle_hm_bull_", "candidate_")):
        return "candlestick_reversal"
    if version.startswith(("quote_", "late_night_")):
        return "quote_edge"
    if version.startswith("x4_"):
        return "x4"
    if version.startswith(("firsthit_", "g7_")):
        return "firsthit"
    if version.startswith("absorption_"):
        return "absorption"
    if version.startswith("s2_cond_"):
        return "s2_cond"
    if version.startswith("scene_") or version in (
        "s4_delay60_v1", "s4_hiconf_v1", "s1_early2_v1", "s1_dip45_v1"
    ):
        return "scene"
    if version.startswith("combo_"):
        return "combo"
    if version.startswith(("nb_", "hm_inside_", "ih_inside_", "hm_quality_", "hm_moderate_")):
        return "nextbar"
    if version.startswith(("krev_", "rev_", "crv_", "brkrv_", "kdc_")):
        return "kline_reversal"
    if version.startswith(("hm_touch_", "s5_deep_", "s1_dyn_")):
        return "pattern"
    if version.startswith("gap_crowd_"):
        return "gap_crowd"
    return source.value

SHADOW_VERSION_SPECS: dict[str, ShadowVersionSpec] = {}
for _version, _source, _period, _target, _direction in _VERSION_ROWS:
    _live, _retired = _live_mapping(_version)
    SHADOW_VERSION_SPECS[_version] = ShadowVersionSpec(
        _version, _source, _period, _target, _direction, _live, _retired,
        family=_family_of(_version, _source),
        role=("EVENT" if _version.endswith("_event_v1") else "STRATEGY"),
    )
SHADOW_VERSIONS: tuple[str, ...] = tuple(SHADOW_VERSION_SPECS)


def _validate() -> None:
    if len(_VERSION_ROWS) != len(SHADOW_VERSION_SPECS):
        raise RuntimeError("shadow execution registry must contain unique versions")
    if {spec.source_type for spec in SHADOW_VERSION_SPECS.values()} != set(SourceType):
        raise RuntimeError("shadow execution registry must cover every source type")
    valid_targets = {"same_window", "next_window", "target_bar", "same_window_first_touch"}
    valid_directions = {"row", "down", "dynamic"}
    for version, spec in SHADOW_VERSION_SPECS.items():
        if version != spec.version:
            raise RuntimeError(f"registry key/version mismatch: {version}")
        if spec.market_period not in {"5m", "15m"}:
            raise RuntimeError(f"invalid market period for {version}: {spec.market_period}")
        if spec.target_semantics not in valid_targets:
            raise RuntimeError(f"invalid target semantics for {version}: {spec.target_semantics}")
        if spec.direction_mode not in valid_directions:
            raise RuntimeError(f"invalid direction mode for {version}: {spec.direction_mode}")
        if spec.policy_version != EXECUTION_POLICY_VERSION:
            raise RuntimeError(f"invalid execution policy for {version}: {spec.policy_version}")
        expected_live, expected_retired = _live_mapping(version)
        if (spec.live_channel, spec.live_retired) != (expected_live, expected_retired):
            raise RuntimeError(f"invalid live mapping for {version}: {spec.live_channel}")


_validate()
