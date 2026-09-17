# S2 条件单交易结算修复方案

## 问题概述

**现象：** FILLED 订单被错误标记为 EXPIRED + pnl=0
**影响：** `s2_cond_t4_v1`, `s2_cond_t5d_v1`, `hm_inside_15m_v2`, `ih_inside_15m_v2`
**根因：** 15m 市场订单缺少 scene_signal_id → trade_settler 直接出清为 EXPIRED

---

## 修改文件

**文件：** `src/binance_predict/services/trade_settler.py`

---

## 修改位置

**方法：** `_settle_row(self, row: TradeOrderModel) -> bool` （第 140 行开始）

---

## 修改前代码（原逻辑）

```python
async def _settle_row(self, row: TradeOrderModel) -> bool:
    # 口径分流：15m 场景订单走 FakeBreakoutSignal 结算（防错配，见模块 docstring）
    if row.market_period == "15m" or row.scene_signal_id is not None:
        return await self._settle_scene_row(row)  # ← 问题在这里

    # 原 5m 市场路径...
```

---

## 修改后代码（新逻辑）

```python
async def _settle_row(self, row: TradeOrderModel) -> bool:
    # 口径分流：15m 场景订单走 FakeBreakoutSignal 结算（防错配，见模块 docstring）
    
    # === S2 条件单专用路径（新增）===
    if row.signal_version in ("s2_cond_t4_v1", "s2_cond_t5d_v1"):
        return await self._settle_s2_cond_row(row)
    
    # === 其他 15m 场景（scene/hm/ih）===
    if row.market_period == "15m":
        if row.scene_signal_id is not None:
            # scene 族有 scene_signal_id → FakeBreakoutSignal
            return await self._settle_scene_row(row)
        else:
            # hm_inside_15m_v2, ih_inside_15m_v2 无关联 → EXPIRED（已知限制）
            logger.error(
                "15m 订单 %s 无结算源 | signal=%s | window=%s",
                row.id, row.signal_version, row.window_start
            )
            return await self._expire_row(row, datetime.now(timezone.utc))
    
    # === 原 5m 市场路径（不变）===
    # ... rest of original code ...
```

---

## 新增方法：_settle_s2_cond_row()

```python
async def _settle_s2_cond_row(self, row: TradeOrderModel) -> bool:
    """S2 条件单结算：回读 KlineShadowSignal 判输赢。
    
    判向规则（direction=UP 恒定的条件下）：
    - 次 15m 收阳 (close > open) → outcome=UP → win=True
    - 次 15m 收阴 (close < open) → outcome=DOWN → win=False  
    - 次 15m 十字星 (close == open) → outcome=NOISE → win=None
    
    盈亏计算与 trade_settler 主逻辑同口径：
    - 赢 + 有 filledShareQty：pnl = shares - amount
    - 赢 + 无 filledShareQty：pnl = amount / avg_price - amount
    - 输：pnl = -amount
    
    结算源：KlineShadowSignal.target_bar_start == window_start
    """
    from binance_predict.db.models import KlineShadowSignal
    
    now_dt = datetime.now(timezone.utc)
    
    # 1. 查找影子信号行
    async with async_session_factory() as session:
        sig_stmt = sa_select(KlineShadowSignal).where(
            KlineShadowSignal.version == row.signal_version,
            KlineShadowSignal.target_bar_start == row.window_start,
        )
        sig = (await session.execute(sig_stmt)).scalar_one_or_none()
    
    if sig is None:
        # 影子信号不存在 → EXPIRED
        logger.critical(
            "订单结算 | id=%s | S2 影子信号缺失 (version=%s, target_bar=%s)"
            "→ EXPIRED", row.id, row.signal_version, row.window_start)
        return await self._expire_row(row, now_dt)
    
    # 2. 判断方向（用 settle_open/settle_close）
    o = float(sig.settle_open or 0)
    c = float(sig.settle_close or 0)
    
    if o == 0 or c == 0:
        # 归档缺失 → EXPIRED
        logger.warning(
            "订单结算 | id=%s | S2 影子价格缺失 (open=%f, close=%f) → EXPIRED",
            row.id, o, c)
        return await self._expire_row(row, now_dt)
    
    if c > o:
        outcome = "UP"
        win = True
    elif c < o:
        outcome = "DOWN"
        win = False
    else:
        outcome = "NOISE"
        win = None
    
    # 3. 计算 pnl（复用已有辅助函数）
    amount = self._amount_usdt(row)
    avg_price = self._avg_price(row)
    shares = self._shares(row)
    
    if win and amount is not None and shares is not None:
        pnl = shares - amount
    elif win and amount is not None and avg_price is not None:
        pnl = amount / avg_price - amount
    elif not win and amount is not None:
        pnl = -amount
    else:
        pnl = None
    
    settle_price = round(c, 8)
    
    # 4. 幂等更新（WHERE settled_at IS NULL）
    async with async_session_factory() as session:
        stmt = (
            sa_update(TradeOrderModel)
            .where(
                TradeOrderModel.id == row.id,
                TradeOrderModel.settled_at.is_(None),
            )
            .values(
                settle_outcome=outcome,
                win=win,
                settle_price=settle_price,
                pnl=pnl,
                settled_at=now_dt,
            )
        )
        result = await session.execute(stmt)
        await session.commit()
    
    if result.rowcount == 0:
        return False  # 已被并发结算（幂等守卫生效）
    self._settled_count += 1
    
    logger.info(
        "订单结算 | id=%s | S2 条件单 | version=%s | window=%s | direction=%s → %s"
        " | win=%s | pnl=%s | outcome=%s",
        row.id, row.signal_version, row.window_start, row.direction, outcome,
        f"{win}" if win is not None else "N/A",
        f"{pnl:+.4f}" if pnl is not None else "N/A",
        outcome)
    
    try:
        await wechat_notifier.notify_order_settled(
            channel=row.signal_version or "s2_cond",
            window_start=int(row.window_start),
            direction=str(row.direction),
            outcome=str(outcome),
            win=win,
            pnl=pnl,
            amount_usdt=self._amount_usdt(row),
            settle_price=settle_price,
        )
    except Exception as exc:
        logger.warning("S2 结算通知异常：%s", exc)
    
    return True
```

---

## 部署步骤

1. **备份当前状态**（在生产执行）
   ```sql
   SELECT COUNT(*) FROM trade_orders 
   WHERE status='FILLED' 
     AND signal_version IN ('s2_cond_t4_v1','s2_cond_t5d_v1')
     AND settled_at IS NULL;
   ```

2. **应用修改**
   - 编辑 `src/binance_predict/services/trade_settler.py`
   - 添加 `from binance_predict.db.models import KlineShadowSignal`
   - 修改 `_settle_row()` 路由逻辑
   - 添加 `_settle_s2_cond_row()` 方法

3. **重启服务**
   ```bash
   # VPS 上
   docker compose -f docker/docker-compose.prod.yml restart backend
   ```

4. **验证修复**
   ```bash
   # 手动触发一次结算扫描
   curl -X POST "https://PRODUCTION_API/api/trades/settle-scan" \
     -H "Authorization: Bearer btc-predict-7f3a9c2e1d"
   
   # 检查是否有更多订单完成结算
   ```

5. **清理历史错误数据**（可选，需评估损失金额后决策）

---

## 风险评估

| 风险点 | 概率 | 影响 | 缓解措施 |
|--------|------|------|---------|
| 新增方法 Bug | 低 | 中 | 幂等守卫 WHERE settled_at IS NULL |
| 与原结算器竞争 | 无 | 无 | 同一入口，不会双写 |
| KlineShadowSignal 缺失 | 中 | 低 | 降级为 EXPIRED（保留原行为） |
| 通知邮件刷屏 | 低 | 低 | 已有重试/限流机制 |

---

## 后续建议

1. **hm_inside_15m_v2 / ih_inside_15m_v2** 也需要类似修复
   - 要么增加 `scene_signal_id` 字段指向对应的影子信号
   - 要么在 `_settle_row()` 中增加 `nextbar` 族的专用分支

2. **统一 15m 订单结算范式**
   - 考虑所有 15m 检测器都落地到同一个影子表（FakeBreakoutSignal 或 KlineShadowSignal）
   - 避免现在这种分散、多表的状态混乱

3. **数据库迁移**
   - 给 `trade_orders` 表增加索引 `(signal_version, window_start)` 加速查询
   - 考虑增加外键约束（未来版本）

---

## 参考链接

- [AGENTS.md](../agents.md) — 生产运维规范
- [trade_settler.py](../src/binance_predict/services/trade_settler.py)
- [live_channels.py](../src/binance_predict/services/live_channels.py) — S2 通道配置
- [s2_cond_shadow_detector.py](../src/binance_predict/services/s2_cond_shadow_detector.py) — S2 影子检测器
