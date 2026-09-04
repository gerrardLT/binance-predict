"""影子信号版本开关 gate：前端手动下线/上线的运行时闸门（口径单一事实源）。

语义（与 live_channel_overrides 同构的影子版，默认方向相反）：
    - 默认在线：shadow_version_overrides 无覆盖行的版本 enabled=True——部署零影响，
      既有影子采集行为不变；
    - 下线：前端 toggle → 覆盖行 enabled=False → 各检测器落库前被 is_enabled()
      拦截（停止采集该版本新信号）→ analytics 下发 enabled=False 面板置灰；
      历史已落库信号不受影响（下线≠删数据，曲线照常显示已有样本）；
    - 上线：enabled=True（或删行）→ 恢复采集。

实现：内存缓存 + 60s 后台刷新兜底（单写者事件循环，读多写少无锁）；
toggle API 写 DB 成功后同步更新缓存（本进程立即生效；多 worker 部署靠 TTL 收敛，
toggle 频率极低可接受）。检测器热路径用同步 is_enabled()——不 await、零延迟。
DB 故障保守全在线：开关表不可用不应成为停掉全部影子采集的理由。
"""
from __future__ import annotations

import asyncio
import time

from loguru import logger
from sqlalchemy import select as sa_select

from binance_predict.db.engine import async_session_factory
from binance_predict.db.models import ShadowVersionOverride

REFRESH_INTERVAL = 60.0  # 后台刷新间隔（秒）：toggle 跨进程收敛的上限


class ShadowVersionGate:
    """影子版本开关闸门：内存缓存 {version: enabled}，只存覆盖行（默认 True 不占内存）。"""

    def __init__(self) -> None:
        self._overrides: dict[str, bool] = {}
        self._loaded_at = 0.0
        self._running = False
        self._task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # 读路径（检测器热调用，同步零延迟）
    # ------------------------------------------------------------------

    def is_enabled(self, version: str) -> bool:
        """无覆盖行或 enabled=True → 在线（默认全在线，与既有行为一致）。"""
        return self._overrides.get(version, True)

    def all_overrides(self) -> dict[str, bool]:
        """当前覆盖表快照（审计/状态端点用）。"""
        return dict(self._overrides)

    # ------------------------------------------------------------------
    # 写路径（toggle API）
    # ------------------------------------------------------------------

    async def set_enabled(self, version: str, enabled: bool) -> None:
        """upsert 覆盖行 + 立即刷新本进程缓存（运行时即时生效）。"""
        async with async_session_factory() as session:
            row = (await session.execute(
                sa_select(ShadowVersionOverride).where(
                    ShadowVersionOverride.version == version
                )
            )).scalar_one_or_none()
            if row is None:
                session.add(ShadowVersionOverride(version=version, enabled=enabled))
            else:
                row.enabled = enabled
            await session.commit()
        self._overrides[version] = enabled
        self._loaded_at = time.monotonic()
        logger.info("影子版本开关 | {} → {}", version, "在线" if enabled else "下线")

    async def refresh(self) -> None:
        """全量读覆盖表刷新缓存（启动时 / 后台任务 / toggle 兜底）。

        DB 故障保守保持现有缓存（首次加载失败则全默认在线），只告警不抛——
        开关表不可用不应停掉影子采集。
        """
        try:
            async with async_session_factory() as session:
                rows = (await session.execute(
                    sa_select(ShadowVersionOverride.version, ShadowVersionOverride.enabled)
                )).all()
            self._overrides = {str(v): bool(e) for v, e in rows}
            self._loaded_at = time.monotonic()
        except Exception as exc:
            logger.warning("影子版本 gate 刷新失败（保守维持现状/全在线）| {}", exc)

    # ------------------------------------------------------------------
    # 生命周期（lifespan 装配：先于检测器启动，保证首轮判定可用）
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        await self.refresh()
        self._task = asyncio.create_task(self._loop(), name="shadow_version_gate")
        offline = [v for v, e in self._overrides.items() if not e]
        logger.info("影子版本 gate 启动 | 覆盖 {} 行 | 下线: {}",
                    len(self._overrides), offline or "无")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def _loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(REFRESH_INTERVAL)
                await self.refresh()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("影子版本 gate 后台刷新异常 | {}", exc)


# 模块级单例：检测器 import 即用（与 main 全局检测器实例同构的共享服务）
shadow_gate = ShadowVersionGate()
