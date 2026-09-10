# 策略通道推送增加中文名称设计文档

## 1. 目标
在 WxPusher 与企业微信的 Markdown 通知卡片中，对所有策略通道展示其友好的中文策略名称，并保留英文代码（格式：`**{display_name}** ({channel})`），方便日常移动端看盘及回溯代码。

## 2. 方案设计

### 2.1 中文名称来源与解析
- **数据源**：`src/binance_predict/services/live_channels.py` 中的 `LIVE_CHANNELS` 和 `RETIRED_CHANNEL_SPECS`。
- **纯函数工具**：在 `live_channels.py` 中导出 `format_channel_title(channel: str) -> str`：
  ```python
  def format_channel_title(channel: str) -> str:
      spec = LIVE_CHANNELS.get(channel) or RETIRED_CHANNEL_SPECS.get(channel)
      if spec and spec.display_name:
          return f"**{spec.display_name}** (`{channel}`)"
      return f"`{channel}`"
  ```

### 2.2 涉及通知函数
修改 `src/binance_predict/services/wechat_notifier.py`：
1. `notify_pre_trade_radar`:
   ```markdown
   > **策略通道**：{format_channel_title(channel)}
   ```
2. `notify_order_filled`:
   ```markdown
   > **策略通道**：{format_channel_title(channel)}
   ```
3. `notify_order_abandoned`:
   ```markdown
   > **策略通道**：{format_channel_title(channel)}
   ```
4. `notify_order_settled`:
   ```markdown
   > **策略通道**：{format_channel_title(channel)}
   ```

### 2.3 测试验证
- 在 `tests/test_wechat_notifier.py` 中增加对 `format_channel_title` 以及通知消息内容的断言测试。
- 确保已知通道、退役通道、未登记通道均正常格式化，不出现 KeyError 或未捕获异常。
