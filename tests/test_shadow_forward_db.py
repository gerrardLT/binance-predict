"""Real PostgreSQL test, explicitly isolated from application/production DB."""
import os
import pytest
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy import select
from binance_predict.db.models import ShadowExecutionAssessment
from binance_predict.services.shadow_forward_evidence import freeze_event, settle_firsthit


@pytest.mark.asyncio
async def test_postgres_freeze_restart_and_settlement():
    url = os.environ.get("SHADOW_TEST_DATABASE_URL")
    if not url:
        pytest.skip("SHADOW_TEST_DATABASE_URL required; never use application DB")
    from sqlalchemy.engine import make_url
    parsed = make_url(url)
    assert parsed.host in {"127.0.0.1", "localhost"} and parsed.port == 55439
    engine = create_async_engine(url)
    table = ShadowExecutionAssessment.__table__
    try:
        async with engine.begin() as conn:
            await conn.run_sync(lambda c: table.create(c, checkfirst=True))
        factory = async_sessionmaker(engine, expire_on_commit=False)
        snapshot = {"window_start": 123000, "window_end": 423000, "trigger_ts": 124000, "side": "DOWN"}
        version = "absorption_follow_td120_v1"
        async with factory() as session:
            await freeze_event(session, snapshot, "LIVE", {}, version, True)
            await session.commit()
        async with factory() as session:
            await freeze_event(session, {**snapshot, "trigger_ts": 125000}, "RESTORED", {}, version, False)
            await settle_firsthit(session, 123000, "DOWN", "5m", version)
            await session.commit()
        async with factory() as session:
            row = (await session.execute(select(ShadowExecutionAssessment))).scalar_one()
            assert row.trigger_ts == 124000
            assert row.input_snapshot["capture_mode"] == "LIVE"
            assert row.strategy_eligible is True
            assert row.theoretical_win is True
            assert row.quote_snapshot["fill"] == "NOT_OBSERVED"
            assert row.decision_snapshot["shadow_capacity"]["count_before"] == 0
        # New sessions simulate restart; duplicate did not consume another slot.
        for i in range(1, 22):
            async with factory() as session:
                event = {**snapshot, "window_start": 123000+i*300000}
                await freeze_event(session, event, "LIVE", {}, version, True)
                await session.commit()
        async with factory() as session:
            rows = (await session.execute(select(ShadowExecutionAssessment).order_by(
                ShadowExecutionAssessment.target_window_start))).scalars().all()
            assert rows[1].decision_snapshot["shadow_capacity"]["count_before"] == 1
            assert rows[20].decision_snapshot["shadow_capacity"]["reason"] == "DAILY_CAPACITY_REACHED"
            for family, period in (("firsthit_control", "5m"), ("krev_a_v1", "15m")):
                await freeze_event(session, snapshot, "LIVE", {}, family, True, period)
            tomorrow = {**snapshot, "window_start": 86400000+123000}
            await freeze_event(session, tomorrow, "LIVE", {}, version, True)
            await session.commit()
        async with factory() as session:
            rows = (await session.execute(select(ShadowExecutionAssessment))).scalars().all()
            fresh = [r for r in rows if r.target_window_start >= 86400000 or r.signal_version != version]
            assert all(r.decision_snapshot["shadow_capacity"]["count_before"] == 0 for r in fresh)
            assert all(r.decision_snapshot["shadow_capacity"]["real_slot_reserved"] is False for r in rows)
    finally:
        async with engine.begin() as conn:
            await conn.run_sync(lambda c: table.drop(c, checkfirst=True))
        await engine.dispose()
