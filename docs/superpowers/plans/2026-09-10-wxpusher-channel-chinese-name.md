# WxPusher / 企业微信策略通道增加中文名称 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 WxPusher / 企业微信消息推送（雷达预警、挂单成交、护栏弃单、结算复盘）中为策略通道展示中文策略名与英文代码格式 `**{display_name}** (`{channel}`)`。

**Architecture:** 
在 `src/binance_predict/services/live_channels.py` 中导出纯函数工具 `format_channel_title(channel: str) -> str`，利用现有的 `LIVE_CHANNELS` 与 `RETIRED_CHANNEL_SPECS` 字典解析通道的 `display_name`；然后在 `src/binance_predict/services/wechat_notifier.py` 中的四个核心通知函数中将原有的 `> **策略通道**：`{channel}`` 替换为 `> **策略通道**：{format_channel_title(channel)}`。

**Tech Stack:** Python 3.11, pytest, pytest-asyncio, Git

---

### Task 1: 在 `live_channels.py` 中实现通道名称格式化函数并在测试中验证

**Files:**
- Modify: `src/binance_predict/services/live_channels.py`
- Modify: `tests/test_wechat_notifier.py`

- [ ] **Step 1: 在 `tests/test_wechat_notifier.py` 中编写针对 `format_channel_title` 的测试用例**

```python
def test_format_channel_title():
    from binance_predict.services.live_channels import format_channel_title

    # 1. 正常活跃通道
    res_active = format_channel_title("hm_inside_15m_v2")
    assert res_active == "**15m孕线上吊线反转（押DOWN）** (`hm_inside_15m_v2`)"

    # 2. 退役通道兜底
    res_retired = format_channel_title("quote_momentum_v1")
    assert res_retired == "**报价动量（A格顺势）** (`quote_momentum_v1`)"

    # 3. 未知通道安全回退
    res_unknown = format_channel_title("unknown_channel_xyz")
    assert res_unknown == "`unknown_channel_xyz`"

    # 4. 空字符串或 None 安全兜底
    assert format_channel_title("") == "``"
```

- [ ] **Step 2: 运行测试以确认失败**

运行：
```bash
.venv/Scripts/python.exe -m pytest tests/test_wechat_notifier.py -k test_format_channel_title -q
```
预期：FAIL（ImportError: cannot import name 'format_channel_title' from 'binance_predict.services.live_channels'）

- [ ] **Step 3: 在 `src/binance_predict/services/live_channels.py` 中实现 `format_channel_title`**

```python
def format_channel_title(channel: str | None) -> str:
    """将策略通道英文标识转换为中文+英文展示格式。
    
    例如: 'hm_inside_15m_v2' -> '**15m孕线上吊线反转（押DOWN）** (`hm_inside_15m_v2`)'
    若未匹配到中文名或为空，则安全回退为 '`channel`'。
    """
    if not channel:
        return f"`{channel or ''}`"
    spec = LIVE_CHANNELS.get(channel) or RETIRED_CHANNEL_SPECS.get(channel)
    if spec and spec.display_name:
        return f"**{spec.display_name}** (`{channel}`)"
    return f"`{channel}`"
```

- [ ] **Step 4: 重新运行测试以确认通过**

运行：
```bash
.venv/Scripts/python.exe -m pytest tests/test_wechat_notifier.py -k test_format_channel_title -q
```
预期：PASS

---

### Task 2: 更新 `wechat_notifier.py` 通知模板并增加格式断言测试

**Files:**
- Modify: `src/binance_predict/services/wechat_notifier.py:200-310`
- Modify: `tests/test_wechat_notifier.py`

- [ ] **Step 1: 在 `tests/test_wechat_notifier.py` 中编写针对通知模板中文策略通道展示的测试**

```python
def test_wechat_notifier_channel_formatting(notifier):
    with patch.object(settings, "wechat_radar_enabled", True), \
         patch.object(notifier, "push_markdown") as mock_push:
        
        # 1. 提前预警雷达
        notifier.notify_pre_trade_radar(
            radar_type="15m 经典孕线反转",
            channel="hm_inside_15m_v2",
            direction="DOWN",
            target_time_ms=1725900000000,
            lead_seconds=90,
            max_exec_price=0.30,
            features_desc="前根大实体",
            dedup_key="radar_test_fmt",
        )
        assert mock_push.called
        content = mock_push.call_args[0][0]
        assert "> **策略通道**：**15m孕线上吊线反转（押DOWN）** (`hm_inside_15m_v2`)" in content

    with patch.object(notifier, "push_markdown") as mock_push:
        # 2. 实盘订单成交
        notifier.notify_order_filled(
            channel="quote_contrarian_v2",
            direction="DOWN",
            window_start=1725900000000,
            avg_price=0.25,
            amount_usdt=5.0,
            shares=20.0,
            order_id=123,
        )
        content = mock_push.call_args[0][0]
        assert "> **策略通道**：**报价反向·门禁版** (`quote_contrarian_v2`)" in content

    with patch.object(notifier, "push_markdown") as mock_push:
        # 3. 护栏弃单
        notifier.notify_order_abandoned(
            channel="x4_v2",
            direction="DOWN",
            window_start=1725900000000,
            quote_price=0.55,
            guard_price=0.50,
            reason="报价超出护栏",
        )
        content = mock_push.call_args[0][0]
        assert "> **策略通道**：**情绪错位·平静市门禁版** (`x4_v2`)" in content

    with patch.object(notifier, "push_markdown") as mock_push:
        # 4. 结算复盘
        notifier.notify_order_settled(
            channel="s5_deep_z20_v1",
            window_start=1725900000000,
            direction="DOWN",
            outcome="DOWN",
            win=True,
            pnl=3.5,
            amount_usdt=5.0,
            settle_price=65000.0,
        )
        content = mock_push.call_args[0][0]
        assert "> **策略通道**：**S5深档·深回落门禁版** (`s5_deep_z20_v1`)" in content
```

- [ ] **Step 2: 运行测试以确认失败**

运行：
```bash
.venv/Scripts/python.exe -m pytest tests/test_wechat_notifier.py -k test_wechat_notifier_channel_formatting -q
```
预期：FAIL（断言包含 `**15m孕线上吊线反转（押DOWN）**...` 失败，当前还是纯 `` `hm_inside_15m_v2` ``）

- [ ] **Step 3: 修改 `src/binance_predict/services/wechat_notifier.py`**

1. 导入 `format_channel_title`:
```python
from .live_channels import format_channel_title
```
2. 更新 `notify_pre_trade_radar` 中的模板：
```python
            f"### ⚡【潜在限价单提前预警】<{dir_color}>{radar_type}</{dir_color}>\n"
            f"> **策略通道**：{format_channel_title(channel)}\n"
            f"> **预测方向**：**{dir_emoji}**\n"
```
3. 更新 `notify_order_filled` 中的模板：
```python
            f"### 🎯【实盘订单已成交】\n"
            f"> **策略通道**：{format_channel_title(channel)}\n"
            f"> **标的周期**：`{fmt_bjt(window_start)}`\n"
```
4. 更新 `notify_order_abandoned` 中的模板：
```python
            f"### 🛡️【护栏安全弃单保护】\n"
            f"> **策略通道**：{format_channel_title(channel)}\n"
            f"> **标的周期**：`{fmt_bjt(window_start)}`\n"
```
5. 更新 `notify_order_settled` 中的模板：
```python
            f"### 🏁【实盘订单结算复盘】<{res_color}>{res_tag}</{res_color}>\n"
            f"> **策略通道**：{format_channel_title(channel)}\n"
            f"> **所属窗口**：`{fmt_bjt(window_start)}`\n"
```

- [ ] **Step 4: 运行测试以确认通过**

运行：
```bash
.venv/Scripts/python.exe -m pytest tests/test_wechat_notifier.py -q
```
预期：PASS（所有 test_wechat_notifier 测试通过）

- [ ] **Step 5: 运行全量测试套件确认无回归问题**

运行：
```bash
.venv/Scripts/python.exe -m pytest tests/test_multi_live_trader.py -q
```
以及全量：
```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```
预期：全量测试均 PASS。
