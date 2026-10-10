"""复利下单链：纯函数 + 内存替身状态机（不碰 DB / 币安）。"""
import asyncio

from binance_predict.services.compound_chain import (
    STEP_MS, CompoundChainRunner, decide, early_trigger, next_stake, validate_plan)

NOW = 1_800_000_000_000 // STEP_MS * STEP_MS + 10_000  # 当前窗开盘后 10s
W = [NOW // STEP_MS * STEP_MS + STEP_MS * i for i in range(1, 4)]  # 3 个未来窗


def test_validate_plan():
    ok = lambda **k: validate_plan(**{"base": 1, "direction": "UP", "windows": W,
                                      "guard": 0.65, "now_ms": NOW, **k})
    assert ok() is None
    assert ok(base=0.05) and ok(base=10.1) and ok(direction="X")   # 首腿 0.1~10U
    assert ok(guard=1.0) and ok(guard=0.0)                          # 开区间
    assert ok(windows=W[:1]) and ok(windows=W[::-1])                # 腿数 / 升序
    assert ok(windows=[W[0] + 1, W[1]])                             # 网格
    cur = NOW // STEP_MS * STEP_MS
    assert ok(windows=[cur, W[0]]) is None                          # 当前窗剩余 >20s 可作首腿
    assert ok(windows=[cur - STEP_MS, W[0]])                        # 早于当前窗
    assert ok(now_ms=cur + STEP_MS - 10_000, windows=[cur, W[0]])   # 当前窗临收盘


def test_decide_and_next_stake():
    assert decide("UP", 100, 101) == "WIN" and decide("UP", 100, 99) == "LOSS"
    assert decide("DOWN", 100, 99) == "WIN" and decide("UP", 100, 100) == "NOISE"
    assert next_stake(1.99999) == 1.9999 and next_stake(63.0) == 63.0  # 不封顶


def test_early_trigger_rules():
    # 条件①：剩余 >60s 且领先一侧报价 ≥0.95
    assert early_trigger("UP", 0.96, 0.04, 100.0, 100.0, 120_000) == "quote"
    assert early_trigger("UP", 0.94, 0.06, 100.0, 100.0, 120_000) is None  # 报价不够
    assert early_trigger("UP", 0.96, 0.04, 100.0, 100.0, 50_000) is None   # 剩余不足 60s
    # 条件②：剩余 ≤15s 且 |现价−开盘| ≥3bp 且方向一致
    assert early_trigger("UP", 0.60, 0.40, 100.04, 100.0, 10_000) == "lead"
    assert early_trigger("DOWN", 0.40, 0.60, 99.96, 100.0, 10_000) == "lead"
    assert early_trigger("UP", 0.60, 0.40, 100.02, 100.0, 10_000) is None  # 幅度仅 2bp
    assert early_trigger("UP", 0.60, 0.40, 99.96, 100.0, 10_000) is None   # 方向相反
    # 市场领先方向与下注相反：两条都不下
    assert early_trigger("UP", 0.40, 0.60, 100.04, 100.0, 10_000) is None


class Fake(CompoundChainRunner):
    def __init__(self, trader, closes, redeem_ok=True, quote=(0.50, 0.50), mid=None, windows=None):
        super().__init__(trader, self, None)
        self.row = {"id": 1, "status": "RUNNING", "direction": "UP", "base_amount": 1.0,
                    "max_exec_price": 0.65, "windows": list(windows or W), "legs": [],
                    "stop_reason": None}
        self.closes, self.redeem_ok, self.t = closes, redeem_ok, NOW
        self.quote, self.mid = quote, mid
        self.orders, self.redeems, self.placed = {}, 0, []
        trader.redeem_tokens = self._fake_redeem
        trader.last_api_error = "x"
        trader.execute_signal_trade = self._exec
        trader._5m_up_price, trader._5m_down_price = quote

    async def _exec(self, **kw):
        self.t = max(self.t, kw["window_start"])  # 下单发生在该腿窗口内
        self.placed.append((kw["window_start"], kw["amount_usdt"]))
        shares = kw["amount_usdt"] / 0.5
        o = {"id": len(self.orders) + 1, "status": "FILLED", "token_id": f"t{kw['window_start']}",
             "amount_in": str(int(kw["amount_usdt"] * 1e18)),
             "quote_json": {"filledShareQty": shares, "averagePrice": 0.5}}
        self.orders[kw["window_start"]] = o
        return o

    async def _fake_redeem(self, toks):
        self.redeems += 1
        return {"ok": 1} if self.redeem_ok else None

    # 替身：时间 / 持久化 / 行情
    def _now_ms(self): return self.t
    async def _sleep(self, sec): self.t += int(sec * 1000)
    async def _sleep_until(self, ms):
        while self.t < ms:
            self.t += 2000
    async def _load(self, cid): return dict(self.row, legs=[dict(l) for l in self.row["legs"]])
    async def _save(self, cid, **v): self.row.update(v)
    async def _existing_order(self, ver, ws): return self.orders.get(ws)
    async def _mark_redeemed(self, oid): pass
    def _notify(self, t): pass
    def _mid_price(self): return self.mid
    async def _window_open(self, ws): return 100.0
    async def fetch_kline_open(self, i, ws): return 100.0
    async def fetch_kline_close(self, i, ws): return self.closes[W.index(ws)]

    def _redeem_bg(self, leg):  # 后台赎回用独立重试循环，不推进链的共用时钟
        async def bg():
            for _ in range(5):
                if await self._trader.redeem_tokens([leg["token_id"]]) is not None:
                    return
                await asyncio.sleep(0)
        t = asyncio.create_task(bg())
        self._bg.add(t)
        t.add_done_callback(self._bg.discard)


def run(f, chain_id=1):
    async def go():
        await f._run(chain_id)
        await asyncio.gather(*f._bg)
    asyncio.run(go())
    return f.row


def test_win_win_loss_stops_and_compounds():
    from types import SimpleNamespace
    f = Fake(SimpleNamespace(), closes=[101, 101, 99])
    r = run(f)
    assert r["status"] == "DONE" and [l["result"] for l in r["legs"]] == ["WIN", "WIN", "LOSS"]
    assert [l["stake"] for l in r["legs"]] == [1.0, 2.0, 4.0]   # 1U@0.5 → 2 股 → 4 股
    assert f.redeems == 2 and len(f.orders) == 3


def test_early_trigger_places_next_leg_before_close():
    from types import SimpleNamespace
    f = Fake(SimpleNamespace(), closes=[101, 101, 101], quote=(0.97, 0.03))
    r = run(f)
    assert r["status"] == "DONE" and len(r["legs"]) == 3
    assert all(l.get("early") == "quote" for l in r["legs"][:2])  # 前两腿提前下单
    assert f.orders[W[1]]["amount_in"] == str(int(2.0 * 1e18))   # 提前下按预估到手股数（1U@0.5→2 股）


def test_loss_after_early_place_leaves_orphan_leg():
    from types import SimpleNamespace
    f = Fake(SimpleNamespace(), closes=[99, 101, 101], quote=(0.97, 0.03))
    r = run(f)
    assert r["status"] == "DONE" and r["legs"][0]["result"] == "LOSS"
    assert len(r["legs"]) == 2 and r["legs"][0].get("next_placed") is True  # 孤儿腿已押上
    assert "提前" in (r["stop_reason"] or "")


def test_loss_on_first_leg_places_no_more_orders():
    from types import SimpleNamespace
    f = Fake(SimpleNamespace(), closes=[99, 101, 101])
    r = run(f)
    assert r["status"] == "DONE" and len(r["legs"]) == 1 and f.redeems == 0


def test_redeem_failure_does_not_block_next_leg():
    from types import SimpleNamespace
    f = Fake(SimpleNamespace(), closes=[101, 101, 101], redeem_ok=False)
    r = run(f)
    assert r["status"] == "DONE" and len(r["legs"]) == 3 and len(f.orders) == 3
    assert f.redeems > 3  # 后台持续重试


def test_restart_claims_existing_order_instead_of_replacing():
    from types import SimpleNamespace
    f = Fake(SimpleNamespace(), closes=[99, 99, 99])
    asyncio.run(f._exec(amount_usdt=1.0, window_start=W[0]))   # 崩溃前已下单但 legs 未落库
    calls = []
    orig = f._exec

    async def spy(**kw):
        calls.append(kw)
        return await orig(**kw)
    f._trader.execute_signal_trade = spy
    r = run(f)
    assert calls == [] and r["legs"][0]["order_id"] == 1
