"""只读核查：线上场景信号的实跑统计（FakeBreakoutSignal + SceneParamVersion）。"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from sqlalchemy import func, select

from binance_predict.db.engine import async_session_factory
from binance_predict.db.models import FakeBreakoutSignal, SceneParamVersion


async def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    async with async_session_factory() as s:
        total = (await s.execute(select(func.count(FakeBreakoutSignal.id)))).scalar()
        print(f"信号总数: {total}")

        rows = (await s.execute(
            select(
                FakeBreakoutSignal.pattern,
                FakeBreakoutSignal.version,
                FakeBreakoutSignal.side,
                FakeBreakoutSignal.status,
                func.count(FakeBreakoutSignal.id),
            ).group_by(
                FakeBreakoutSignal.pattern, FakeBreakoutSignal.version,
                FakeBreakoutSignal.side, FakeBreakoutSignal.status,
            ).order_by(FakeBreakoutSignal.pattern, FakeBreakoutSignal.version)
        )).all()
        print("\n按 场景×版本×方向×状态 分组:")
        for r in rows:
            print(f"  pattern={r[0]} version={r[1]} side={r[2]} status={r[3]}: n={r[4]}")

        # 结算统计（15m 口径）：pattern × version × outcome
        st = (await s.execute(
            select(
                FakeBreakoutSignal.pattern,
                FakeBreakoutSignal.version,
                FakeBreakoutSignal.settle_outcome,
                func.count(FakeBreakoutSignal.id),
            ).where(FakeBreakoutSignal.status == "SETTLED")
            .group_by(FakeBreakoutSignal.pattern, FakeBreakoutSignal.version,
                      FakeBreakoutSignal.settle_outcome)
        )).all()
        print("\n15m 结算方向分布（SETTLED）:")
        for r in st:
            print(f"  pattern={r[0]} version={r[1]} outcome={r[2]}: n={r[3]}")

        # 场景① 命中率（bull_exhaust 预测 DOWN → settle=DOWN 为赢）
        s1 = (await s.execute(
            select(FakeBreakoutSignal.settle_outcome, func.count(FakeBreakoutSignal.id))
            .where(FakeBreakoutSignal.pattern == "bull_exhaust")
            .where(FakeBreakoutSignal.status == "SETTLED")
            .group_by(FakeBreakoutSignal.settle_outcome)
        )).all()
        s1d = dict(s1)
        n1 = sum(s1d.values())
        if n1:
            print(f"\n场景① bull_exhaust（预测DOWN）: 赢(DOWN) {s1d.get('DOWN', 0)}/{n1}"
                  f" = {s1d.get('DOWN', 0) / n1:.1%}")

        s2 = (await s.execute(
            select(FakeBreakoutSignal.settle_outcome, func.count(FakeBreakoutSignal.id))
            .where(FakeBreakoutSignal.pattern == "bear_exhaust")
            .where(FakeBreakoutSignal.status == "SETTLED")
            .group_by(FakeBreakoutSignal.settle_outcome)
        )).all()
        s2d = dict(s2)
        n2 = sum(s2d.values())
        if n2:
            print(f"场景② bear_exhaust（预测UP）: 赢(UP) {s2d.get('UP', 0)}/{n2}"
                  f" = {s2d.get('UP', 0) / n2:.1%}")

        # 逐条明细（最近 12 条）
        items = (await s.execute(
            select(FakeBreakoutSignal).order_by(FakeBreakoutSignal.signal_time.desc()).limit(12)
        )).scalars().all()
        import datetime as dt
        print(f"\n最近 {len(items)} 条明细:")
        for x in items:
            t = dt.datetime.fromtimestamp(x.signal_time / 1000, tz=dt.timezone.utc).strftime("%m-%d %H:%M")
            print(f"  #{x.id} {t} {x.pattern}·{x.version} side={x.side} close_pos={x.close_pos} "
                  f"vol_ratio={x.vol_ratio} | fire报价 D{x.down_price_15m}/U{x.up_price_15m} "
                  f"| 入场快照 D{x.entry_down_price_15m}/U{x.entry_up_price_15m} "
                  f"| 结算 {x.settle_outcome}({x.status})")

        # 场景参数版本
        vs = (await s.execute(select(SceneParamVersion))).scalars().all()
        print(f"\n场景参数版本 {len(vs)} 个:")
        for v in vs:
            print(f"  {v.version} status={v.status} params={dict(v.params)} activated={v.activated_at}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
