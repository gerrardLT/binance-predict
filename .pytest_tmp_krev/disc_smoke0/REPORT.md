# 5m 720d K 线科学发现报告（v2）

- 数据：5760 根， ~ （gap=?）
- 切分：0.6/0.2/0.2 时序三段；漏斗预算：{"discovery_frac": 0.6, "validation_frac": 0.2, "holdout_frac": 0.2, "quantiles": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9], "min_samples": 80, "min_sample_frac": 0.01, "fdr_alpha": 1.0, "min_lift_pp": 0.0, "min_validation_lift_pp": -1000000000.0, "min_interaction_gain_pp": -1000000000.0, "max_l1": 15, "max_l2": 20, "max_l3": 20, "max_l3_tests": 50000, "n_min_l2": 60, "n_min_l3": 40, "n_min_holdout": 20, "shortlist_per_target": 8, "final_holdout_rule": "⚠️ Final Holdout 前，shortlist 必须完全冻结（只触碰一次）"}
- 总检验数：4012；holdout 只触碰一次（）

## Top 10 发现（按 score = holdout_lift × √n × min(retention,1)）

### 1. [WEAK] reversal_1 @ L2（score=70.4801）
- 触发条件：`failed_breakout_high_50 == True AND failed_breakout_high_100 == True`
- 机制：（数据驱动，无预注册机制说明）
- 发现段：n=98，胜率 0.438776（lift -5.52911pp）
- holdout：n=46，胜率 0.608696 [Wilson 0.464568~0.736068]，lift 10.3917pp，retention 1.87946
- 盈亏比（MFE/MAE, ATR 口径）：1.10903；费后 EV@0.50：0.16965；Kelly：0.184089
- walk-forward 逐折胜率：F1:0.428571 → F2:0.45 → F3:0.677419 → F4:0.25 → F5:0.25 → F6:0.642857 → F7:0.428571 → F8:0.6875
- 一致性：月 0.5 / 波动regime 0；run 块自助 CI [0.44241, 0.59196]

### 2. [WEAK] continuation_1 @ L2（score=43.3254）
- 触发条件：`regime_ranging == True AND prior_range_atr_5 >= 3.01099005`
- 机制：（数据驱动，无预注册机制说明）
- 发现段：n=68，胜率 0.5（lift -0.593343pp）
- holdout：n=13，胜率 0.615385 [Wilson 0.355229~0.822903]，lift 12.0163pp，retention 20.2519
- 盈亏比（MFE/MAE, ATR 口径）：1.56723；费后 EV@0.50：0.182504；Kelly：0.198036
- walk-forward 逐折胜率：F1:0.75 → F2:0.5 → F3:0.7 → F4:0.48 → F5:0.25 → F6:0.285714 → F7:0.5 → F8:0.583333
- 一致性：月 1 / 波动regime 0；run 块自助 CI [0.379736, 0.589052]

### 3. [WEAK] reversal_2 @ L2（score=39.913）
- 触发条件：`open_loc >= 46796.5837 AND dist_prior_low_atr_50 <= 0.482441883`
- 机制：（数据驱动，无预注册机制说明）
- 发现段：n=78，胜率 0.333333（lift -16.4496pp）
- holdout：n=17，胜率 0.647059 [Wilson 0.413004~0.826903]，lift 12.6189pp，retention 0.767127
- 盈亏比（MFE/MAE, ATR 口径）：1.01553；费后 EV@0.50：0.243368；Kelly：0.26408
- walk-forward 逐折胜率：F1:0.357143 → F2:0.307692 → F3:0.230769 → F4:0.266667 → F5:0.692308 → F6:0.357143 → F7:0.384615 → F8:0.666667
- 一致性：月 1 / 波动regime 0；run 块自助 CI [0.316667, 0.479339]

### 4. [WEAK] continuation_2 @ L3（score=32.4244）
- 触发条件：`dist_prior_low_atr_50 <= 0.482441883 AND reject_down_depth_atr_50 <= 0.482441883 AND atr_pct_200 >= 0.905`
- 机制：（数据驱动，无预注册机制说明）
- 发现段：n=41，胜率 0.536585（lift 3.44146pp）
- holdout：n=18，胜率 0.555556 [Wilson 0.337164~0.754405]，lift 7.64251pp，retention 2.22072
- 盈亏比（MFE/MAE, ATR 口径）：1.05853；费后 EV@0.50：0.0675381；Kelly：0.0732861
- walk-forward 逐折胜率：F1:0.466667 → F2:0.5 → F3:0.4 → F4:0.666667 → F5:0.666667 → F6:0.166667 → F7:0.6 → F8:0.461538
- 一致性：月 0 / 波动regime 0；run 块自助 CI [0.394737, 0.613333]

### 5. [PROMISING] reversal_1 @ L2（score=32.0729）
- 触发条件：`failed_breakout_low_50 == True AND failed_breakout_low_100 == True`
- 机制：（数据驱动，无预注册机制说明）
- 发现段：n=133，胜率 0.593985（lift 9.99184pp）
- holdout：n=31，胜率 0.580645 [Wilson 0.407663~0.735845]，lift 7.58667pp，retention 0.759287
- 盈亏比（MFE/MAE, ATR 口径）：0.666197；费后 EV@0.50：0.11575；Kelly：0.125601
- walk-forward 逐折胜率：F1:0.56 → F2:0.466667 → F3:0.73913 → F4:0.583333 → F5:0.6875 → F6:0.444444 → F7:0.576923 → F8:0.666667
- 一致性：月 1 / 波动regime 0；run 块自助 CI [0.522517, 0.654641]

### 6. [WEAK] continuation_1 @ L2（score=30.1006）
- 触发条件：`rv_20 <= 0.0010267951 AND body_atr <= 0.0611627302`
- 机制：（数据驱动，无预注册机制说明）
- 发现段：n=84，胜率 0.392857（lift -11.3076pp）
- holdout：n=26，胜率 0.576923 [Wilson 0.389486~0.744556]，lift 8.17015pp，retention 0.722535
- 盈亏比（MFE/MAE, ATR 口径）：0.481307；费后 EV@0.50：0.108597；Kelly：0.11784
- walk-forward 逐折胜率：F1:0.423077 → F2:0.333333 → F3:0.375 → F4:0.428571 → F5:0.444444 → F6:0.6 → F7:0.52381 → F8:0.666667
- 一致性：月 0.5 / 波动regime 0；run 块自助 CI [0.381679, 0.546154]

### 7. [PROMISING] continuation_2 @ L2（score=26.3107）
- 触发条件：`dist_prior_low_atr_50 <= 0.482441883 AND new_24h_lo == True`
- 机制：（数据驱动，无预注册机制说明）
- 发现段：n=112，胜率 0.598214（lift 9.60435pp）
- holdout：n=33，胜率 0.545455 [Wilson 0.379859~0.701571]，lift 6.63241pp，retention 0.690563
- 盈亏比（MFE/MAE, ATR 口径）：1.16198；费后 EV@0.50：0.0481283；Kelly：0.0522244
- walk-forward 逐折胜率：F1:0.62963 → F2:0.555556 → F3:0.611111 → F4:0.333333 → F5:0.517241 → F6:0.5 → F7:0.409091 → F8:0.625
- 一致性：月 1 / 波动regime 0；run 块自助 CI [0.471189, 0.604064]

### 8. [WEAK] continuation_2 @ L2（score=25.9256）
- 触发条件：`bullish_engulfing == True AND absret_3 <= 0.000273203232`
- 机制：（数据驱动，无预注册机制说明）
- 发现段：n=63，胜率 0.365079（lift -13.7091pp）
- holdout：n=23，胜率 0.565217 [Wilson 0.368114~0.743654]，lift 8.6087pp，retention 0.627953
- 盈亏比（MFE/MAE, ATR 口径）：0.486479；费后 EV@0.50：0.086104；Kelly：0.093432
- walk-forward 逐折胜率：F1:0.416667 → F2:0.5 → F3:0.285714 → F4:0.230769 → F5:0.4375 → F6:0.6 → F7:0.4375 → F8:0.666667
- 一致性：月 0.5 / 波动regime 0；run 块自助 CI [0.342857, 0.533333]

### 9. [WEAK] reversal_2 @ R3（score=16.6157）
- 触发条件：`doji == True AND inside_bar == True`
- 机制：test
- 发现段：n=104，胜率 0.490385（lift -0.744462pp）
- holdout：n=29，胜率 0.551724 [Wilson 0.37548~0.715868]，lift 3.08546pp，retention 4.14455
- 盈亏比（MFE/MAE, ATR 口径）：1.19545；费后 EV@0.50：0.0601758；Kelly：0.0652971
- walk-forward 逐折胜率：F1:0.444444 → F2:0.6 → F3:0.529412 → F4:0.555556 → F5:0.458333 → F6:0.578947 → F7:0.666667 → F8:0.5
- 一致性：月 0 / 波动regime 0；run 块自助 CI [0.447205, 0.603659]

### 10. [WEAK] continuation_1 @ L2（score=11.48）
- 触发条件：`rv_20 <= 0.0010267951 AND atr_ratio_14_50 <= 0.844642608`
- 机制：（数据驱动，无预注册机制说明）
- 发现段：n=139，胜率 0.388489（lift -11.7444pp）
- holdout：n=52，胜率 0.538462 [Wilson 0.405036~0.666595]，lift 4.324pp，retention 0.368175
- 盈亏比（MFE/MAE, ATR 口径）：1.32082；费后 EV@0.50：0.0346908；Kelly：0.0376432
- walk-forward 逐折胜率：F1:0.395349 → F2:0.333333 → F3:0.25 → F4:0.44 → F5:0.571429 → F6:0.526316 → F7:0.461538 → F8:0.580645
- 一致性：月 0.5 / 波动regime 0；run 块自助 CI [0.390244, 0.528991]

## 与既有最强发现对照（R1 复刻可比性）

- 旧产物基准：`breakout_high_50 == True AND dist_prior_high_atr_10 >= -0.0148 AND efficiency_8 >= 0.692` → reversal holdout 61.5% (n=558, lift +10.3pp)
- 本轮 shortlist 未包含该字面条件（由 `--replay-legacy` 单独对照）。

## 负结果（波普尔式保留）

- **continuation_1**：7 条组合进入 shortlist，但 holdout 无一通过（最强 regime_ranging == True AND prior_range_atr_5 >= 3.01099005 胜率 0.615385，Wilson 下界 0.355229，打平线 0.520408）——费后无正期望。
- **reversal_2**：8 条组合进入 shortlist，但 holdout 无一通过（最强 open_loc >= 46796.5837 AND dist_prior_low_atr_50 <= 0.482441883 胜率 0.647059，Wilson 下界 0.413004，打平线 0.520408）——费后无正期望。

_所有裁决仅以冻结 holdout 为准；名义 lift 与费后 EV 相关性低（r≈0.16），每条结论均附经济账。_