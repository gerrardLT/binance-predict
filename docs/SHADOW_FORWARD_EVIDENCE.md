# 影子前向证据实施与验收

## 安全边界

仅冻结事件、读取市场/钱包地址/金额报价、写影子评估账本。不得调用下单、占位、划转、余额扣减。独立报价客户端不复用实盘交易锁。部署只走 CI；本轮经用户明确授权审阅、测试后部署。

## 数据语义

forward-v2 cohort 由版本、运行时金额与规则组成的实际冻结 profile SHA-256 派生；配置变更隔离新 cohort。触发采样同步复制配置与运营输入，再异步落账与报价。独立 policy_version 隔离旧数据；首次冻结加 PostgreSQL 行锁，恢复事件不能覆盖实时事件。报价不是成交，fill 保持 NOT_OBSERVED。费用 2% 是未验证假设，仅持有到结算。

实时入口包括首触母事件、absorption TD120/150、KREV、CRV、Rev2Inside、nextbar、candlestick、S2 条件/优化、S1 变体。能力依赖采集器开启、实时新鲜信号与所需标定可用；未接等价入口的版本保留归档且阻塞。当前覆盖矩阵依观测实时行标识，零样本不能证明入口健康。

报价预算最多 8 个等待/执行请求，等待 2 秒，统一 ladder 4 秒；1/5/10U 加运行时配置金额。缺少市场、报价、运营状态不得解释为通过。

运营约束判定金额、余额、日容量和互斥；真实未知返回 unknown。余额只读已有钱包缓存，超过30秒未知；互斥只读已成交窗口缓存，不能证明完整状态时未知。日容量复用 MultiLiveTrader._count_filled_today 的 FILLED 订单计数，异步15秒缓存，首次、过期、失败、UTC日切均返回未知；不计 fired，不写真实槽。

首触、吸收、K线统一走 freeze_event。影子容量按 cohort/版本/周期隔离，默认每日100U/20次；在独立账本持久记录 accepted/count_before/balance_before，版本级 advisory lock 防竞争。重复冻结不重复消耗，日切重置，重启直接从持久账本还原。假设不代表真实余额或成交；无运行时金额时使用明确1U容量假设，真实金额仍未知阻塞。

实际首触策略目前仅适配 firsthit_down_chg_v2（G3）；control/recovery 是对照，不是现役策略。其他 firsthit 版本不由 G3 代表，coverage 明确归档阻塞。K线入口接入不等于全族实时等价认证。

共享报价按窗口/周期/方向/token/金额缓存500ms，保留原始 requested_at/received_at；报价早于新冻结 trigger 的事件被收益层拒绝，不能改写时间戳冒充新报价。实盘锁或在途任务活跃时 shadow 背压，不新增请求；不抢实盘锁，不取消正在执行的实盘任务。已经在途的shadow HTTP不支持抢占。首触生产入口复用同一共享报价设施。

## 统计局限

收益仅为报价结算估计，含 1%/3% 价格压力。独立窗口/UTC 日计数，日聚类压力收益使用 99% 正态近似下界；少于 14 日不输出下界。该近似不是可上线保证，也不解决跨策略依赖或多重比较。

## 二次 review 修订

G3 保留特征真实首触时间，仅与当前采样同刻才为 LIVE；旧触发 RESTORED 不报价。吸收判定复用实盘 TD+20秒宽限，失败不占成功机会，报价使用自身版本和冻结金额。

容量仅为冻结当前金额的假设状态；其余1/5/10U是反事实报价，capacity_passed_for_amount=false，不能声称金额容量准入。amountOut仅校验正整数字符值，尚未核实币安单位/费用与兑付映射，收益仍基于averagePrice且始终阻塞。

顶部跨cohort仅观测总数，不能作为准入窗日；series按版本/policy/金额分组。K线落账任务最多32，超限与捕获异常累加进程级overflow_counts；该计数不持久化，重启会归零，不冒充完整逐笔证据。所有observe_kline同步异常隔离，不阻断live hook。

专项回归覆盖旧G3时间、吸收宽限重试、配置cohort变化、反事实容量、任务满/捕获异常、amountOut合法性；尚未适配版本保持阻塞。

## 验证

全量测试、build、lint 输出保存 `.pytest_tmp/shadow-final-{tests,build,lint}.log`。
真实 PostgreSQL 冻结集成测试 `tests/test_shadow_forward_db.py` 使用专用环境变量 SHADOW_TEST_DATABASE_URL；强制 localhost:55439，绝不读取应用 DB URL。测试验证跨 session 首次冻结不覆写与结算持久化。隔离实例位于 `.pytest_tmp/shadow_pg`。

## 未满足项

未等价接入的家族仍归档阻塞；首次或过期真实运营缓存仍未知。费用、成交率、跨策略依赖和多重比较不能由报价证据解决。API 不自动启用任何实盘通道。容量专项覆盖拒绝/未知、重复、独立版本、日切、跨session重启；真实DB测试只操作隔离实例。
