# 决策台架定期重跑 Runbook（decision_bench）

> 目的：每 2~4 周用新累积的数据重跑一次经济账，检验两个悬而未决的假设，
> 并守住「没有统计上站得住的正 EV 就不开自动下注」的底线。
>
> 背景结论（2026-07-17，3522 窗定论版）：方向预测力真实存在（安慰剂已验证），
> 但市场定价基本有效——所有策略费前 EV≈0，计费后为负。详见
> `output/decision_bench_v2.json`。

## 一、服务器导出（部署机 /www/wwwroot/binance-predict/）

```bash
DB=$(docker ps --format '{{.Names}}' | grep -i db | head -1)
docker exec -i "$DB" psql -U postgres -d binance_predict -t -A -c \
  "SELECT COALESCE(json_agg(t), '[]') FROM (
     SELECT id, start_time, end_time, curve_up_pct, curve_down_pct,
            curve_up_price, curve_down_price,
            outcome, actual_return, sample_count
       FROM sentiment_windows
      WHERE outcome IS NOT NULL AND curve_up_pct IS NOT NULL
        AND curve_down_pct IS NOT NULL
      ORDER BY start_time ASC) t" > windows_with_price.json
```

注：`curve_up_price/curve_down_price` 自迁移 f6a7b8c9d0e1 部署后开始归档时
永久化（此前的历史窗口该字段为 NULL，因采样表仅保留 1 小时已不可恢复）。

## 二、本地重跑（项目根目录）

```powershell
# 1. 主对照台架：三决策点 × 策略 × 经济账（含 bootstrap 95% CI）
uv run python scripts/decision_bench.py --from-file windows_with_price.json --out output/decision_bench_latest.json

# 2. 仅新时段重跑（把上一轮的窗口数替换 N_PREV，做样本外复现检验）
uv run python -c "import json; ws=json.load(open('windows_with_price.json',encoding='utf-8')); ws.sort(key=lambda w: w['start_time']); json.dump(ws[N_PREV:], open('output/windows_new_period.json','w',encoding='utf-8'))"
uv run python scripts/decision_bench.py --from-file output/windows_new_period.json --out output/decision_bench_newonly.json

# 3. 台架回归测试
uv run pytest tests/test_feature_bench.py -q
```

## 三、判据（事前登记，防事后合理化）

重点盯两个上轮留下的悬案：

| 假设 | 格子 | 上轮点估计 | 判活标准 | 判死标准 |
|---|---|---|---|---|
| fade 共识（逆向买便宜票） | fade10 × 三决策点 | 6/6 格为正（+1%~+9%，CI 全含 0） | bootstrap CI 下界 > 0 且新旧时段同向 | 点估计转负，或新时段方向翻转 |
| t90 k-NN | t90 × knn3 | 新时段 +6.6% [−1.7, +14.8] | 同上 | 同上 |

**开自动下注的门槛（一票否决制）**：
1. 某策略在「费率 2%（已实测确认，2026-07-17）+ 溢价 0.01」情形下 EV 的 bootstrap 95% CI 下界 > 0；
2. 出手数 ≥ 500；
3. 新时段单独重跑同向为正。
三条全满足才允许把 `AGENT_AUTO_TRADE` 置 true，否则保持 false。
当前状态（2026-07-17）：费率 2% 下无任何策略 CI 下界 > 0，不开。

## 四、上轮基线存档

| 文件 | 内容 |
|---|---|
| `output/decision_bench_v2.json` | 3522 窗全量定论版（holdout 1053） |
| `output/decision_bench_newonly.json` | 新时段 1774 窗单独验证 |
| `output/curve_lockin_scan.json` | 答案锁定度逐时间点分析 |
| `output/curve_shape_scan.json` | 形态描述量分布 + 分离度 |

价格近似校准（12 窗 236 配对点实测）：price − chance 中位 +0.005、p90 +0.015，
`up_price + down_price ≈ 1.0`。价格数据积累 ≥ 500 窗后，应把 decision_bench
的定价从 chance 近似切换为真实 `curve_up_price`。
