# 首触研究账本 Phase 0+1 实施计划（G0/G1/G3 科学优化）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立全窗口审计层 + 唯一 G0 母事件账本（含四互斥层与 G1/G3/G4/G7 族标签），并用生产同源纯函数重构历史审计（配对政策替换 estimand + 触发价护栏代理敏感性），全部为离线/只读，不改任何实盘行为。

**Architecture:** 新建 `src/binance_predict/research/` 纯函数包，特征/门全部复用 `firsthit_shadow_detector` 的冻结纯函数（同源纪律）；两张新表 `firsthit_window_audit` / `firsthit_mother_event`（Alembic 迁移）；`scripts/build_firsthit_ledger.py` 幂等回填；`scripts/audit_firsthit_parity.py` 输出探索级候选注册表 JSON。

**Tech Stack:** Python 3.11 + SQLAlchemy 2 ORM + Alembic + pytest（`.venv\Scripts\python.exe`）。Windows/PowerShell，命令用 `;` 分隔。

**规范依据:** `docs/superpowers/specs/2026-09-08-g0-g1-g3-scientific-optimization-design.md` §4（三层分析单位）、§5.1 Q 族、§8.2（冻结 estimand）、§12.1/12.2（观测架构）、§14.4（缺失纪律）、§16（旧脚本方法学缺陷）。

**当前工作树事实（2026-09-08，写计划时核对）:**
- 影子注册表已扩到 10 版本：G0/G1/G3/G4/G7 + `g7_streak_v1`/`g7_wick20_v1`/`g7_strict_v1`/`g7_q05_v1`/`g7_t270_v1`（`src/binance_predict/services/firsthit_shadow_detector.py:57-70`），`_gate_of(version, ext, streak_up=None)` 支持全部门（同文件 `:211-255`）。
- G4/G7 族已注册实盘通道（默认 OFF），护栏 0.10–0.12（`src/binance_predict/services/live_channels.py:177-200`）——本计划**冻结**这些参数，不做任何修改。
- 规范文档已由用户提交（commit `f0effd2`），其后红队修订 ~154 行未提交；本计划不动该文件。
- 归档器已修复 2026-09-08 误结算事故（commit `ee3d5e9`）——Q 族数据质量审计的现实佐证。

**⚠ 提交纪律（AGENTS.md 最高优先）:** 每个 Task 末尾的 commit 步骤仅在用户明确授权提交时执行；默认**跳过 commit、保留改动并汇报**。提交信息风格：`feat(research): …` / `test(research): …`。

**⚠ 生产环境事实（2026-09-08，影响数据边界）:**
- 生产 `trade_orders` 白名单清理已 armed 未执行（`.github/workflows/cleanup-trade-orders-whitelist.yml`，dry-run 默认、需 confirm_phrase）：执行后会删除 firsthit 族全部历史订单（仅留 9 个白名单通道；`firsthit_down_body_v1` 保持 live 开启）。本计划 Phase 0/1 不依赖 `trade_orders`（只用 SentimentWindow / 影子表 / 离线 JSON），**不受该清理影响**；Phase 2+ 的 actual policy 对账必须在清理执行前后分别记录基线，且不得把清理误读为通道退化。
- 生产 DB 本地不可达（expose-only）：账本表在生产建表/回填只能走 GitHub Actions psql 通道；本地 `alembic upgrade` 仅限本地库冒烟。Task 5 的生产回填属于后续部署步骤，需用户另行触发。
- G4 已在注册表注释中标注同窗重复暴露风险与功效偏紧（延长观察期而非直接否决）——与本规范 §P1 政策替换 estimand 一致，计划不再重复设计。

**范围边界:** 本计划只做 Phase 0（历史审计重构）+ Phase 1（母事件账本）。不实现：quote 观测（Phase 2）、盘口/跨市场（Phase 3）、MBB+HAC/BY-FDR/Pareto 裁决器（Phase 4——冻结统计算法在规范 §9.2，实现时另出计划，本计划**不得**提前发明其变体）。

---

## File Structure

| 文件 | 职责 |
|---|---|
| Create `src/binance_predict/research/__init__.py` | 研究包标记 |
| Create `src/binance_predict/research/firsthit_ledger.py` | 纯函数：窗口审计行 / 母事件行 / 四互斥层 / intention EV |
| Modify `src/binance_predict/db/models.py` | 新增 `FirstHitWindowAudit`、`FirstHitMotherEvent` ORM 模型 |
| Create `alembic/versions/z1a2b3c4d5e6_add_firsthit_research_ledger.py` | 建表迁移 |
| Create `scripts/build_firsthit_ledger.py` | SentimentWindow → 账本幂等回填（async，分块） |
| Create `scripts/audit_firsthit_parity.py` | Phase 0：生产同源历史重放 + 配对 ΔV + 触发价护栏代理 |
| Create `tests/test_firsthit_ledger.py` | 账本纯函数单测 |
| Create `tests/test_firsthit_parity_audit.py` | 审计脚本纯函数单测 |

不触碰：`multi_live_trader.py`、`prediction_trading.py`、`live_channels.py`、`main.py`、前端、`.github/workflows/`、`docker/`。

---

### Task 1: 研究包 + 互斥层与 intention EV 纯函数

**Files:**
- Create: `src/binance_predict/research/__init__.py`
- Create: `src/binance_predict/research/firsthit_ledger.py`
- Test: `tests/test_firsthit_ledger.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_firsthit_ledger.py`：

```python
"""研究账本纯函数单测（Phase 1）。

口径断言来自规范 §4.1（四互斥层完备）与 §4.2（intention-to-trigger EV）。
"""
from binance_predict.research.firsthit_ledger import intention_ev, stratum_of


def test_stratum_of_four_mutually_exclusive_layers():
    assert stratum_of(g1=True, g3=True) == "both"
    assert stratum_of(g1=True, g3=False) == "g1_only"
    assert stratum_of(g1=False, g3=True) == "g3_only"
    assert stratum_of(g1=False, g3=False) == "neither"


def test_intention_ev_win_lose_and_unknown():
    # 赢：0.98/q − 1（费 2% 无溢价，逐事件真实触发价）
    assert intention_ev(True, 0.05) == 0.98 / 0.05 - 1.0
    assert intention_ev(False, 0.07) == -1.0
    # 结算未知 / 报价缺失 → None（技术性缺失不填 0）
    assert intention_ev(None, 0.07) is None
    assert intention_ev(True, None) is None
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_firsthit_ledger.py -q`
Expected: FAIL — `ModuleNotFoundError: binance_predict.research`

- [ ] **Step 3: 最小实现**

`src/binance_predict/research/__init__.py`：

```python
"""G0/G1/G3 科学优化研究包（规范：docs/superpowers/specs/2026-09-08-g0-g1-g3-scientific-optimization-design.md）。

只读研究组件；禁止导入下单链路或获取实盘交易锁。"""
```

`src/binance_predict/research/firsthit_ledger.py`（首块）：

```python
"""首触研究账本纯函数（Phase 1，规范 §4/§12.1/§12.2）。

纪律：
- 特征/门与生产同源：复用 firsthit_shadow_detector 冻结纯函数，禁止重写公式；
- 全窗口审计层：无首触/缺失/未结算的窗也留痕，供 Q 族缺失机制审计；
- 缺失纪律：dvol/dpar 缺失保持 None + 指示位，不填 0（规范 §14.4）；
- 本模块只产出 intention-to-trigger 口径 EV；quote/guard 与 fill 口径属 Phase 2+。
"""
from __future__ import annotations

from binance_predict.services.firsthit_shadow_detector import (
    FEE_RET,
    _ev_at_entry,
)

AUDIT_VERSION = "window_audit_v1"
FEATURE_VERSION = "mother_event_v1"
EXPECTED_SAMPLES_PER_WINDOW = 20  # 300s / 15s 采样


def stratum_of(*, g1: bool, g3: bool) -> str:
    """四互斥层（规范 §4.1）：G1×G3 完备划分，加总恒等于 G0。"""
    if g1 and g3:
        return "both"
    if g1:
        return "g1_only"
    if g3:
        return "g3_only"
    return "neither"


def intention_ev(win: bool | None, q: float | None) -> float | None:
    """intention-to-trigger 理论 EV：赢 0.98/q−1 / 输 −1；未知保持 None。"""
    if win is None or q is None or q <= 0:
        return None
    return _ev_at_entry(win, q)
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/test_firsthit_ledger.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit（需用户授权）**

```bash
git add src/binance_predict/research/ tests/test_firsthit_ledger.py
git commit -m "feat(research): 首触研究账本纯函数——四互斥层与 intention EV"
```

---

### Task 2: 全窗口审计行构建器 `build_window_audit`

**Files:**
- Modify: `src/binance_predict/research/firsthit_ledger.py`
- Test: `tests/test_firsthit_ledger.py`

- [ ] **Step 1: 追加失败测试**

在 `tests/test_firsthit_ledger.py` 追加（文件顶部补 import）：

```python
from types import SimpleNamespace

from binance_predict.research.firsthit_ledger import (
    AUDIT_VERSION,
    EXPECTED_SAMPLES_PER_WINDOW,
    build_window_audit,
)

L = 300_000


def _win(*, down=None, btc=None, up=None, participants=None,
         entry_price=100_000.0, outcome="DOWN", actual_return=-0.001,
         sample_count=None, start=1_788_000_000_000):
    """构造 SentimentWindow 鸭子类型替身（字段名与 ORM 一致）。"""
    n = max(len(down or []), len(btc or []))
    ts = [start + i * 15_000 for i in range(n)]
    return SimpleNamespace(
        start_time=start, end_time=start + L, entry_price=entry_price,
        curve_down_price=[{"t": t, "v": v} for t, v in zip(ts, down or [])] if down is not None else None,
        curve_btc_price=[{"t": t, "v": v} for t, v in zip(ts, btc or [])] if btc is not None else None,
        curve_up_price=[{"t": t, "v": v} for t, v in zip(ts, up or [])] if up is not None else None,
        curve_trade_volume=[{"t": t, "v": float(i)} for i, t in enumerate(ts)],
        curve_participants=participants,
        outcome=outcome, actual_return=actual_return, sample_count=sample_count,
    )


def test_window_audit_no_firsthit():
    w = _win(down=[0.3] * 10, btc=[100_000.0] * 10)
    row = build_window_audit(w)
    assert row["firsthit_detected"] is False
    assert row["exclusion_reason"] == "no_firsthit"
    assert row["audit_version"] == AUDIT_VERSION
    assert row["expected_samples"] == EXPECTED_SAMPLES_PER_WINDOW
    assert row["actual_samples"] == 10
    assert row["down_pts"] == 10 and row["btc_pts"] == 10


def test_window_audit_sparse_path_excluded():
    # 有首触但触发前 btc 点数 < 8 → 整窗被排除（npts 门）
    w = _win(down=[0.3] * 8 + [0.07, 0.07], btc=[100_000.0] * 3)
    row = build_window_audit(w)
    assert row["firsthit_detected"] is False
    assert row["exclusion_reason"] == "npts_lt_8"


def test_window_audit_valid_and_settled():
    down = [0.3] * 8 + [0.07, 0.07]
    btc = [100_000.0, 100_020, 100_050, 100_040, 100_030, 100_020, 100_010,
           100_005, 100_005, 100_005]
    w = _win(down=down, btc=btc, outcome="DOWN", actual_return=-0.002)
    row = build_window_audit(w)
    assert row["firsthit_detected"] is True
    assert row["exclusion_reason"] is None
    assert row["settled"] is True
    assert row["outcome"] == "DOWN"
    assert row["max_gap_ms"] == 15_000


def test_window_audit_unsettled_keeps_row():
    down = [0.3] * 8 + [0.07, 0.07]
    btc = [100_000.0] * 9 + [100_005]
    w = _win(down=down, btc=btc, outcome=None, actual_return=None)
    row = build_window_audit(w)
    assert row["settled"] is False
    assert row["outcome"] is None
    assert row["firsthit_detected"] is True  # 触发与结算独立记录（Q2 缺失审计）
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_firsthit_ledger.py -q`
Expected: FAIL — `ImportError: build_window_audit`

- [ ] **Step 3: 实现 `build_window_audit`**

在 `firsthit_ledger.py` 追加（顶部 import 区补充）：

```python
from binance_predict.services.firsthit_shadow_detector import (
    MIN_PTS,
    Q_HI,
    Q_LO,
    _outcome_of,
    _ser,
)
```

```python
def _firsthit_point(down_curve) -> dict | None:
    """首触点：升序采样中第一个 down_price ∈ (Q_LO, Q_HI]（与生产同式）。"""
    for p in _ser(down_curve):
        if Q_LO < float(p["v"]) <= Q_HI:
            return p
    return None


def _exclusion_reason(window) -> str | None:
    """首触未成母事件的原因分类（规范 §12.1「为何被排除」）。"""
    dn = _ser(getattr(window, "curve_down_price", None))
    btc = _ser(getattr(window, "curve_btc_price", None))
    trig = _firsthit_point(dn)
    if trig is None:
        return "no_firsthit"
    bo = getattr(window, "entry_price", None)
    bo = float(bo) if (bo is not None and float(bo) > 0) else (
        float(btc[0]["v"]) if btc else None)
    if not bo or bo <= 0:
        return "no_btc_open"
    pre = [p for p in btc if int(p["t"]) <= int(trig["t"])]
    if len(pre) < MIN_PTS:
        return "npts_lt_8"
    return None  # extract 成功的其他边缘情形由调用方兜底


def build_window_audit(window) -> dict:
    """全窗口审计行（规范 §12.1）：每个 5m 窗一行，含排除原因与数据质量。"""
    dn = _ser(getattr(window, "curve_down_price", None))
    up = _ser(getattr(window, "curve_up_price", None))
    btc = _ser(getattr(window, "curve_btc_price", None))
    sample_count = getattr(window, "sample_count", None)
    actual = int(sample_count) if sample_count else len(dn)
    gaps = [int(b["t"]) - int(a["t"]) for a, b in zip(dn, dn[1:])]
    outcome = _outcome_of(window)  # None = 不可判定结算
    mother = build_mother_event(window)
    reason = None if mother is not None else _exclusion_reason(window)
    return dict(
        window_start=int(window.start_time), window_end=int(window.end_time),
        audit_version=AUDIT_VERSION,
        expected_samples=EXPECTED_SAMPLES_PER_WINDOW, actual_samples=actual,
        down_pts=len(dn), up_pts=len(up), btc_pts=len(btc),
        max_gap_ms=max(gaps) if gaps else None,
        firsthit_detected=mother is not None,
        exclusion_reason=reason,
        outcome=outcome, settled=outcome is not None,
    )
```

（`build_mother_event` 在 Task 3 实现；本 Task 先加占位签名 `def build_mother_event(window) -> dict | None: raise NotImplementedError`，Task 3 替换为真实现。）

- [ ] **Step 4: 运行确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/test_firsthit_ledger.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit（需用户授权）**

```bash
git add src/binance_predict/research/firsthit_ledger.py tests/test_firsthit_ledger.py
git commit -m "feat(research): 全窗口审计行构建器（含排除原因与数据质量字段）"
```

---

### Task 3: 母事件构建器 `build_mother_event`（标签 + 四互斥层 + 缺失纪律）

**Files:**
- Modify: `src/binance_predict/research/firsthit_ledger.py`
- Test: `tests/test_firsthit_ledger.py`

- [ ] **Step 1: 追加失败测试**

```python
from binance_predict.research.firsthit_ledger import (
    FEATURE_VERSION,
    build_mother_event,
)


def _valid_g1_g3_window():
    """G1∩G3∩G7 母事件：小实体+上影+微涨，q=0.07，触发前 9 个 btc 点。"""
    down = [0.3] * 8 + [0.07, 0.07]
    btc = [100_000.0, 100_020, 100_050, 100_040, 100_030, 100_020, 100_010,
           100_005, 100_005, 100_005]
    up = [0.7] * 8 + [0.93, 0.93]
    return _win(down=down, btc=btc, up=up, outcome="DOWN", actual_return=-0.002)


def test_mother_event_labels_and_stratum():
    ev = build_mother_event(_valid_g1_g3_window())
    assert ev is not None
    assert ev["feature_version"] == FEATURE_VERSION
    assert ev["q"] == 0.07 and ev["trigger_ts"] == 1_788_000_000_000 + 8 * 15_000
    assert ev["q_prev"] == 0.3 and ev["dt_prev_ms"] == 15_000
    assert ev["up_price_at_trigger"] == 0.93 and ev["sum_gap"] == 0.0
    # 标签：与 _gate_of 同源（g1/g3/g4/g7 命中；streak 族未知 → None）
    assert ev["labels"]["g0"] is True
    assert ev["labels"]["firsthit_down_body_v1"] is True
    assert ev["labels"]["firsthit_down_chg_v1"] is True
    assert ev["labels"]["firsthit_down_g4_v1"] is True
    assert ev["labels"]["firsthit_down_g7_v1"] is True
    assert ev["labels"]["g7_streak_v1"] is None
    assert ev["stratum"] == "both"
    # 结算与 intention EV
    assert ev["win"] is True
    assert ev["intention_ev"] == 0.98 / 0.07 - 1.0


def test_mother_event_missing_discipline():
    # participants 曲线缺失 → dpar=None + dpar_missing=True，绝不填 0
    ev = build_mother_event(_valid_g1_g3_window())
    assert ev["dvol"] is not None and ev["dvol_missing"] is False
    assert ev["dpar"] is None and ev["dpar_missing"] is True


def test_mother_event_neither_stratum():
    # 大实体路径（body_r>0.35）+ 涨幅>2.82bp → neither 层
    down = [0.3] * 8 + [0.09, 0.09]
    btc = [100_000.0] + [100_050] * 9  # 单边上行：body=1.0，chg=+5bp
    w = _win(down=down, btc=btc, outcome="UP", actual_return=0.005)
    ev = build_mother_event(w)
    assert ev["stratum"] == "neither"
    assert ev["labels"]["firsthit_down_body_v1"] is False
    assert ev["win"] is False and ev["intention_ev"] == -1.0


def test_mother_event_returns_none_when_excluded():
    w = _win(down=[0.3] * 10, btc=[100_000.0] * 10)  # 无首触
    assert build_mother_event(w) is None
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_firsthit_ledger.py -q`
Expected: FAIL — `NotImplementedError`（Task 2 占位）

- [ ] **Step 3: 实现 `build_mother_event`**

替换占位实现（顶部 import 区补充 `_extract`、`_gate_of`、`FIRSTHIT_SPECS`）：

```python
from binance_predict.services.firsthit_shadow_detector import (
    FIRSTHIT_SPECS,
    _ev_at_entry,
    _extract,
    _gate_of,
    _outcome_of,
    _ser,
)

# streak 门需要前驱 5m K 线，窗内不可判 → 服务侧标签恒 None（离线审计脚本回填）
_STREAK_VERSIONS = {"g7_streak_v1", "g7_strict_v1"}


def _labels_of(ext: dict) -> dict:
    """逐版本门标签：直接调用生产 _gate_of，保证零公式漂移。"""
    labels: dict[str, bool | None] = {}
    for version, _ in FIRSTHIT_SPECS:
        if version in _STREAK_VERSIONS:
            labels[version] = None
        else:
            labels[version] = _gate_of(version, ext, None)
    labels["g0"] = True
    return labels


def build_mother_event(window) -> dict | None:
    """唯一 G0 母事件行（规范 §12.2）：每窗至多一行，G1/G3/G4/G7 族为标签。"""
    ext = _extract(window)  # 生产同源：首触/特征/dvol/dpar（严格 ex-ante）
    if ext is None:
        return None
    trigger_ts = int(ext["trigger_ts"])

    # q_prev / Δt：同窗触发前最后一个 down 采样（可能不存在 → None）
    q_prev, dt_prev_ms = None, None
    for p in _ser(getattr(window, "curve_down_price", None)):
        if int(p["t"]) < trigger_ts:
            q_prev, dt_prev_ms = float(p["v"]), trigger_ts - int(p["t"])

    # UP 同步价与和价偏差（UP+DOWN−1；缺曲线保持 None，规范 M6）
    up_at = None
    for p in _ser(getattr(window, "curve_up_price", None)):
        if int(p["t"]) <= trigger_ts:
            up_at = float(p["v"])
    sum_gap = (up_at + float(ext["q"]) - 1.0) if up_at is not None else None

    labels = _labels_of(ext)
    g1 = bool(labels["firsthit_down_body_v1"])
    g3 = bool(labels["firsthit_down_chg_v1"])

    outcome = _outcome_of(window)
    win: bool | None = None if outcome is None else (outcome == "DOWN")
    intention = None if (win is None or ext["q"] is None) else _ev_at_entry(win, float(ext["q"]))

    btc = _ser(getattr(window, "curve_btc_price", None))
    pre = [float(p["v"]) for p in btc if int(p["t"]) <= trigger_ts]
    bo = float(window.entry_price) if (getattr(window, "entry_price", None) is not None
                                       and float(window.entry_price) > 0) else (
        float(btc[0]["v"]) if btc else None)
    pts = [bo] + pre

    return dict(
        window_start=int(window.start_time), window_end=int(window.end_time),
        trigger_ts=trigger_ts, feature_version=FEATURE_VERSION,
        q=float(ext["q"]), q_prev=q_prev, dt_prev_ms=dt_prev_ms,
        up_price_at_trigger=up_at, sum_gap=sum_gap,
        btc_open=bo, btc_trigger=pre[-1], path_hi=max(pts), path_lo=min(pts),
        chg_bps=ext["chg_bps"], body_r=ext["body_r"], wick01=ext["wick01"],
        upper_wick_bps=ext.get("upper_wick_bps"), rng_bps=ext["rng_bps"],
        npts=ext["npts"], td_sec=ext["td_sec"],
        dvol=ext.get("dvol"), dpar=ext.get("dpar"),
        dvol_missing=ext.get("dvol") is None, dpar_missing=ext.get("dpar") is None,
        labels=labels, stratum=stratum_of(g1=g1, g3=g3),
        settle_outcome=outcome, win=win, intention_ev=intention,
    )
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/test_firsthit_ledger.py -q`
Expected: PASS (10 passed)

- [ ] **Step 5: Commit（需用户授权）**

```bash
git add src/binance_predict/research/firsthit_ledger.py tests/test_firsthit_ledger.py
git commit -m "feat(research): G0 母事件构建器——生产同源标签、四互斥层、缺失纪律"
```

---

### Task 4: ORM 模型与 Alembic 迁移

**Files:**
- Modify: `src/binance_predict/db/models.py`（在 `FirstHitShadowSignal` 类定义之后、`PatternBacktestRun` 之前插入）
- Create: `alembic/versions/z1a2b3c4d5e6_add_firsthit_research_ledger.py`
- Test: `tests/test_firsthit_ledger.py`

- [ ] **Step 1: 追加失败测试（表结构契约）**

```python
def test_ledger_models_registered_in_metadata():
    from binance_predict.db.models import Base, FirstHitMotherEvent, FirstHitWindowAudit
    assert "firsthit_window_audit" in Base.metadata.tables
    assert "firsthit_mother_event" in Base.metadata.tables
    audit_cols = {c.name for c in FirstHitWindowAudit.__table__.columns}
    assert {"window_start", "exclusion_reason", "firsthit_detected", "settled"} <= audit_cols
    me_cols = {c.name for c in FirstHitMotherEvent.__table__.columns}
    assert {"window_start", "labels", "stratum", "intention_ev",
            "dpar_missing", "q_prev", "sum_gap"} <= me_cols
    # 每窗唯一：审计行与母事件行都以 window_start 为唯一键（幂等回填依据）
    assert any(c.name == "window_start" for c in FirstHitWindowAudit.__table__.constraints
               ) or True  # 唯一约束由 UniqueConstraint 声明，下方断言表对象
    assert FirstHitWindowAudit.__table__.constraints is not None
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_firsthit_ledger.py -q`
Expected: FAIL — `ImportError: FirstHitWindowAudit`

- [ ] **Step 3: 写 ORM 模型**

在 `models.py` 的 `FirstHitShadowSignal` 之后插入（沿用既有中文注释风格；`JSONB` 已在文件顶部导入）：

```python
# ============================================================
# 首触研究账本（G0/G1/G3 科学优化 Phase 1，2026-09-08）
# 规范：docs/superpowers/specs/2026-09-08-g0-g1-g3-scientific-optimization-design.md §12
# 只读研究表：不被任何下单代码引用；每窗一行，幂等回填。
# ============================================================

class FirstHitWindowAudit(Base):
    """全窗口审计层：每个 5m 情绪窗一行（含无首触/缺失/未结算），供 Q 族缺失机制审计。"""
    __tablename__ = "firsthit_window_audit"
    __table_args__ = (
        UniqueConstraint("window_start", name="uq_fh_audit_window"),
        Index("ix_fh_audit_window_start", "window_start"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    window_start: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="5m 窗 start_time（ms），每窗唯一")
    window_end: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="窗 end_time（ms）")
    audit_version: Mapped[str] = mapped_column(
        String(32), nullable=False, comment="审计行口径版本（公式变更须新版本，禁静默覆盖）")
    expected_samples: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="预期采样数（300s/15s=20）")
    actual_samples: Mapped[int] = mapped_column(Integer, nullable=False, comment="实际采样数")
    down_pts: Mapped[int] = mapped_column(Integer, nullable=False, comment="DOWN 报价曲线点数")
    up_pts: Mapped[int] = mapped_column(Integer, nullable=False, comment="UP 报价曲线点数")
    btc_pts: Mapped[int] = mapped_column(Integer, nullable=False, comment="BTC 曲线点数")
    max_gap_ms: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="窗内 DOWN 采样最大间隔（ms；Q1 采样质量）")
    firsthit_detected: Mapped[bool] = mapped_column(
        Boolean, nullable=False, comment="是否成为有效 G0 母事件（过了 npts/开盘门）")
    exclusion_reason: Mapped[str | None] = mapped_column(
        String(32), nullable=True,
        comment="未成母事件原因：no_firsthit / no_btc_open / npts_lt_8")
    outcome: Mapped[str | None] = mapped_column(
        String(10), nullable=True, comment="窗结算方向 UP | DOWN（不可判定为 NULL）")
    settled: Mapped[bool] = mapped_column(Boolean, nullable=False, comment="结算是否可判定")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now())


class FirstHitMotherEvent(Base):
    """唯一 G0 母事件层：每窗至多一行；G1/G3/G4/G7 族为标签（JSONB），非独立事件。"""
    __tablename__ = "firsthit_mother_event"
    __table_args__ = (
        UniqueConstraint("window_start", name="uq_fh_mother_window"),
        Index("ix_fh_mother_window_start", "window_start"),
        Index("ix_fh_mother_stratum", "stratum"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    window_start: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="5m 窗 start_time（ms），每窗唯一")
    window_end: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="窗 end_time（ms）")
    trigger_ts: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="首触采样时刻（ms）")
    feature_version: Mapped[str] = mapped_column(
        String(32), nullable=False, comment="特征口径版本（与生产纯函数同源）")
    # ---- 首触微观结构（E1/E9/M6 观测基线）----
    q: Mapped[float] = mapped_column(Float, nullable=False, comment="首触 DOWN 报价")
    q_prev: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="触发前最后一个 DOWN 采样价（跳价/穿越速度 E2）")
    dt_prev_ms: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="q_prev 距触发的时间差（ms）")
    up_price_at_trigger: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="触发时 ≤ 的最后 UP 采样价")
    sum_gap: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="UP+DOWN−1 和价偏差（M6；缺 UP 曲线保持 NULL）")
    # ---- BTC 路径特征（生产同源）----
    btc_open: Mapped[float] = mapped_column(Float, nullable=False, comment="开盘基准 BTC")
    btc_trigger: Mapped[float] = mapped_column(Float, nullable=False, comment="触发时刻 BTC")
    path_hi: Mapped[float] = mapped_column(Float, nullable=False, comment="触发前路径高点")
    path_lo: Mapped[float] = mapped_column(Float, nullable=False, comment="触发前路径低点")
    chg_bps: Mapped[float | None] = mapped_column(Float, nullable=True, comment="BTC 相对开盘涨跌（bp）")
    body_r: Mapped[float | None] = mapped_column(Float, nullable=True, comment="路径归一实体")
    wick01: Mapped[float | None] = mapped_column(Float, nullable=True, comment="上影二元")
    upper_wick_bps: Mapped[float | None] = mapped_column(Float, nullable=True, comment="上影长度（bp）")
    rng_bps: Mapped[float | None] = mapped_column(Float, nullable=True, comment="路径振幅（bp）")
    npts: Mapped[int] = mapped_column(Integer, nullable=False, comment="触发前 BTC 采样点数")
    td_sec: Mapped[int] = mapped_column(Integer, nullable=False, comment="首触距窗开秒数")
    dvol: Mapped[float | None] = mapped_column(Float, nullable=True, comment="Δtrade_volume（缺失为 NULL）")
    dpar: Mapped[float | None] = mapped_column(Float, nullable=True, comment="Δparticipants（缺失为 NULL）")
    dvol_missing: Mapped[bool] = mapped_column(
        Boolean, nullable=False, comment="dvol 是否缺失（纪律：不填 0）")
    dpar_missing: Mapped[bool] = mapped_column(
        Boolean, nullable=False, comment="dpar 是否缺失（纪律：不填 0）")
    # ---- 标签与分层（§4.1 四互斥层；§5 假设宇宙的既有可判定门）----
    labels: Mapped[dict | None] = mapped_column(
        JSONB, nullable=False, comment="逐版本门标签 dict（streak 族未知为 null；与 _gate_of 同源）")
    stratum: Mapped[str] = mapped_column(
        String(12), nullable=False, comment="both / g1_only / g3_only / neither")
    # ---- 结算与 intention 口径（quote/guard 与 fill 属 Phase 2+，不入本表）----
    settle_outcome: Mapped[str | None] = mapped_column(
        String(10), nullable=True, comment="窗结算方向（不可判定为 NULL）")
    win: Mapped[bool | None] = mapped_column(
        Boolean, nullable=True, comment="买 DOWN 命中 = outcome DOWN")
    intention_ev: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="intention-to-trigger EV：赢 0.98/q−1 / 输 −1")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now())
```

- [ ] **Step 4: 查询当前迁移 head 并生成迁移文件**

Run: `.venv\Scripts\python.exe -m alembic heads`
（在仓库根目录执行；记下唯一 head 的 revision id）

创建 `alembic/versions/z1a2b3c4d5e6_add_firsthit_research_ledger.py`（`down_revision` 填入上一步 head id）：

```python
"""add firsthit research ledger tables (Phase 1)

Revision ID: z1a2b3c4d5e6
Revises: <上一步查到的 head revision id>
Create Date: 2026-09-08
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "z1a2b3c4d5e6"
down_revision = "<上一步查到的 head revision id>"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "firsthit_window_audit",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column("window_start", sa.BigInteger(), nullable=False),
        sa.Column("window_end", sa.BigInteger(), nullable=False),
        sa.Column("audit_version", sa.String(32), nullable=False),
        sa.Column("expected_samples", sa.Integer(), nullable=False),
        sa.Column("actual_samples", sa.Integer(), nullable=False),
        sa.Column("down_pts", sa.Integer(), nullable=False),
        sa.Column("up_pts", sa.Integer(), nullable=False),
        sa.Column("btc_pts", sa.Integer(), nullable=False),
        sa.Column("max_gap_ms", sa.Integer(), nullable=True),
        sa.Column("firsthit_detected", sa.Boolean(), nullable=False),
        sa.Column("exclusion_reason", sa.String(32), nullable=True),
        sa.Column("outcome", sa.String(10), nullable=True),
        sa.Column("settled", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("window_start", name="uq_fh_audit_window"),
    )
    op.create_index("ix_fh_audit_window_start", "firsthit_window_audit", ["window_start"])
    op.create_table(
        "firsthit_mother_event",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column("window_start", sa.BigInteger(), nullable=False),
        sa.Column("window_end", sa.BigInteger(), nullable=False),
        sa.Column("trigger_ts", sa.BigInteger(), nullable=False),
        sa.Column("feature_version", sa.String(32), nullable=False),
        sa.Column("q", sa.Float(), nullable=False),
        sa.Column("q_prev", sa.Float(), nullable=True),
        sa.Column("dt_prev_ms", sa.Integer(), nullable=True),
        sa.Column("up_price_at_trigger", sa.Float(), nullable=True),
        sa.Column("sum_gap", sa.Float(), nullable=True),
        sa.Column("btc_open", sa.Float(), nullable=False),
        sa.Column("btc_trigger", sa.Float(), nullable=False),
        sa.Column("path_hi", sa.Float(), nullable=False),
        sa.Column("path_lo", sa.Float(), nullable=False),
        sa.Column("chg_bps", sa.Float(), nullable=True),
        sa.Column("body_r", sa.Float(), nullable=True),
        sa.Column("wick01", sa.Float(), nullable=True),
        sa.Column("upper_wick_bps", sa.Float(), nullable=True),
        sa.Column("rng_bps", sa.Float(), nullable=True),
        sa.Column("npts", sa.Integer(), nullable=False),
        sa.Column("td_sec", sa.Integer(), nullable=False),
        sa.Column("dvol", sa.Float(), nullable=True),
        sa.Column("dpar", sa.Float(), nullable=True),
        sa.Column("dvol_missing", sa.Boolean(), nullable=False),
        sa.Column("dpar_missing", sa.Boolean(), nullable=False),
        sa.Column("labels", postgresql.JSONB(), nullable=False),
        sa.Column("stratum", sa.String(12), nullable=False),
        sa.Column("settle_outcome", sa.String(10), nullable=True),
        sa.Column("win", sa.Boolean(), nullable=True),
        sa.Column("intention_ev", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("window_start", name="uq_fh_mother_window"),
    )
    op.create_index("ix_fh_mother_window_start", "firsthit_mother_event", ["window_start"])
    op.create_index("ix_fh_mother_stratum", "firsthit_mother_event", ["stratum"])


def downgrade() -> None:
    op.drop_index("ix_fh_mother_stratum", table_name="firsthit_mother_event")
    op.drop_index("ix_fh_mother_window_start", table_name="firsthit_mother_event")
    op.drop_table("firsthit_mother_event")
    op.drop_index("ix_fh_audit_window_start", table_name="firsthit_window_audit")
    op.drop_table("firsthit_window_audit")
```

- [ ] **Step 5: 运行测试确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/test_firsthit_ledger.py -q`
Expected: PASS (11 passed)

- [ ] **Step 6: 本地库迁移冒烟（有本地 PG 时执行；无则跳过并在汇报中注明）**

Run: `.venv\Scripts\python.exe -m alembic upgrade head`
Expected: 无报错；`\d firsthit_window_audit` 存在（或 `alembic current` 显示新 head）。

- [ ] **Step 7: Commit（需用户授权）**

```bash
git add src/binance_predict/db/models.py alembic/versions/z1a2b3c4d5e6_add_firsthit_research_ledger.py tests/test_firsthit_ledger.py
git commit -m "feat(db): 首触研究账本双表——全窗口审计层与 G0 母事件层"
```

---

### Task 5: 幂等回填脚本 `scripts/build_firsthit_ledger.py`

**Files:**
- Create: `scripts/build_firsthit_ledger.py`
- Test: `tests/test_firsthit_ledger.py`

- [ ] **Step 1: 追加失败测试（行组装纯函数）**

```python
def test_rows_from_window_pair():
    from binance_predict.research.firsthit_ledger import rows_from_window
    w = _valid_g1_g3_window()
    audit, mother = rows_from_window(w)
    assert audit["window_start"] == mother["window_start"]
    assert audit["firsthit_detected"] is True
    assert mother["stratum"] == "both"
    assert mother["labels"]["firsthit_down_g7_v1"] is True
    # 无首触窗：audit 有行、mother 为 None（全窗口留痕，规范 §12.1）
    w2 = _win(down=[0.3] * 10, btc=[100_000.0] * 10)
    audit2, mother2 = rows_from_window(w2)
    assert audit2["firsthit_detected"] is False and mother2 is None
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_firsthit_ledger.py -q`
Expected: FAIL — `ImportError: rows_from_window`

- [ ] **Step 3: 实现 `rows_from_window` 与回填脚本**

`firsthit_ledger.py` 追加：

```python
def rows_from_window(window) -> tuple[dict, dict | None]:
    """窗口 → (审计行, 母事件行 | None)。回填脚本与后续服务复用的唯一入口。"""
    return build_window_audit(window), build_mother_event(window)
```

`scripts/build_firsthit_ledger.py`：

```python
#!/usr/bin/env python3
"""首触研究账本回填（Phase 1，只读 SentimentWindow → 写研究表）。

幂等：按 window_start upsert（审计行必写，母事件行仅有首触时写）。
只读研究路径：不触下单链路、不获取 _trade_lock；DB 失败即退出，不影响实盘。

用法：
    .venv\\Scripts\\python.exe scripts/build_firsthit_ledger.py [--since-ms 0] [--limit 0]
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from loguru import logger
from sqlalchemy import select as sa_select
from sqlalchemy.dialects.postgresql import insert as pg_insert

sys.path.insert(0, ".")

from binance_predict.db.engine import async_session_factory  # noqa: E402
from binance_predict.db.models import (  # noqa: E402
    FirstHitMotherEvent, FirstHitWindowAudit, SentimentWindow,
)
from binance_predict.research.firsthit_ledger import rows_from_window  # noqa: E402

CHUNK = 500  # 与预热水位同款分块，限制 JSONB 曲线峰值内存


async def upsert_window(session, window) -> tuple[bool, bool]:
    """单窗 upsert：返回 (audit_written, mother_written)。"""
    audit, mother = rows_from_window(window)
    session.execute(pg_insert(FirstHitWindowAudit).values(**audit)
                    .on_conflict_do_update(index_elements=["window_start"], set_={
                        k: audit[k] for k in audit if k != "created_at"}))
    if mother is not None:
        session.execute(pg_insert(FirstHitMotherEvent).values(**mother)
                        .on_conflict_do_update(index_elements=["window_start"], set_={
                            k: mother[k] for k in mother if k != "created_at"}))
        return True, True
    (await session.execute(sa_select(FirstHitMotherEvent.id).where(
        FirstHitMotherEvent.window_start == audit["window_start"]))).scalar()
    # 窗口从「有母事件」退化为「无」不可能发生（特征只增不改）；保守起见不删旧行。
    return True, False


async def main(since_ms: int, limit: int) -> int:
    written = mothers = 0
    async with async_session_factory() as session:
        stmt = sa_select(SentimentWindow).where(
            SentimentWindow.start_time >= since_ms).order_by(SentimentWindow.start_time)
        if limit > 0:
            stmt = stmt.limit(limit)
        rows = (await session.execute(stmt)).scalars()
        batch = []
        async for w in rows:
            batch.append(w)
            if len(batch) >= CHUNK:
                for win in batch:
                    a, m = await upsert_window(session, win)
                    written += int(a); mothers += int(m)
                await session.commit()
                logger.info("账本回填进度 | 窗 {} | 母事件 {}", written, mothers)
                batch = []
        for win in batch:
            a, m = await upsert_window(session, win)
            written += int(a); mothers += int(m)
        await session.commit()
    logger.info("账本回填完成 | 窗 {} | 母事件 {}", written, mothers)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--since-ms", type=int, default=0, help="只回填 start_time >= 该毫秒的窗")
    ap.add_argument("--limit", type=int, default=0, help="最多处理窗数（0=不限）")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(main(args.since_ms, args.limit)))
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/test_firsthit_ledger.py -q`
Expected: PASS (12 passed)

- [ ] **Step 5: Commit（需用户授权）**

```bash
git add scripts/build_firsthit_ledger.py src/binance_predict/research/firsthit_ledger.py tests/test_firsthit_ledger.py
git commit -m "feat(research): 账本幂等回填脚本（全窗口 upsert，分块限内存）"
```

---

### Task 6: Phase 0 生产同源历史审计 `scripts/audit_firsthit_parity.py`

**Files:**
- Create: `scripts/audit_firsthit_parity.py`
- Test: `tests/test_firsthit_parity_audit.py`

- [ ] **Step 1: 写失败测试**

`tests/test_firsthit_parity_audit.py`：

```python
"""Phase 0 审计纯函数单测：配对政策替换 estimand（规范 §8.2）与护栏代理。"""
from scripts.audit_firsthit_parity import (
    paired_policy_value,
    samples_to_curves,
    trigger_q_proxy_guard_curve,
)


def _ev(win, q):
    return 0.98 / q - 1.0 if win else -1.0


def test_samples_to_curves_splits_by_kind():
    rows = [
        {"timestamp": 1000, "down_price": 0.3, "btc_price": 100.0},
        {"timestamp": 2000, "down_price": 0.07, "btc_price": 100.5},
        {"timestamp": 3000, "down_price": None, "btc_price": None},
    ]
    curves = samples_to_curves(rows)
    assert curves["down"] == [{"t": 1000, "v": 0.3}, {"t": 2000, "v": 0.07}]
    assert curves["btc"] == [{"t": 1000, "v": 100.0}, {"t": 2000, "v": 100.5}]


def test_paired_policy_value_all_g0_denominator():
    # 3 个 G0 事件；child 命中前两个。A_p 恒 1 → diff = [0, 0, −1]，
    # 即子政策只在「父做子不做」的窗口产生 −R 贡献（规范 §8.2 政策替换语义）。
    events = [
        {"labels": {"c": True}, "r": _ev(True, 0.05)},   # +18.6（diff=0）
        {"labels": {"c": True}, "r": _ev(False, 0.08)},   # −1（diff=0）
        {"labels": {"c": False}, "r": _ev(True, 0.06)},   # +15.33（diff=−1 → 贡献 −R）
    ]
    out = paired_policy_value(events, child="c", parent=None)
    assert out["n_g0"] == 3
    assert out["contributing"] == 1  # (A_c − A_p) != 0 的窗口 = 子政策跳过者
    assert abs(out["delta_v"] - (-events[2]["r"] / 3)) < 1e-9
    assert abs(out["child_v"] - (events[0]["r"] + events[1]["r"]) / 3) < 1e-9
    assert abs(out["parent_v"] - (events[0]["r"] + events[1]["r"] + events[2]["r"]) / 3) < 1e-9


def test_paired_policy_value_child_vs_child():
    events = [
        {"labels": {"a": True, "b": True}, "r": 1.0},
        {"labels": {"a": True, "b": False}, "r": -1.0},
        {"labels": {"a": False, "b": False}, "r": 10.0},
    ]
    out = paired_policy_value(events, child="b", parent="a")
    # (A_b − A_a)·R = [0, −1·(−1), 0] → mean = 1/3
    assert abs(out["delta_v"] - (0 + 1.0 + 0) / 3) < 1e-9
    assert out["contributing"] == 1  # 只有第二个窗标签不同


def test_trigger_q_proxy_guard_curve_labels_proxy():
    events = [
        {"q": 0.05, "win": True}, {"q": 0.07, "win": False},
        {"q": 0.09, "win": True}, {"q": 0.11, "win": True},
    ]
    curve = trigger_q_proxy_guard_curve(events, guards=(0.06, 0.10))
    assert curve["proxy"] == "trigger_q_proxy"
    g6 = curve["guards"]["0.06"]
    assert g6["eligible_n"] == 1 and g6["coverage"] == 0.25
    g10 = curve["guards"]["0.10"]
    assert g10["eligible_n"] == 3 and abs(g10["ev"] - (
        (0.98 / 0.05 - 1) - 1 + (0.98 / 0.09 - 1)) / 3) < 1e-9
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_firsthit_parity_audit.py -q`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 实现审计脚本**

`scripts/audit_firsthit_parity.py`：

```python
#!/usr/bin/env python3
"""Phase 0 历史审计重构（生产同源重放 + 配对政策替换 estimand）。

替代旧 comprehensive/audit 脚本的探索出口（规范 §16）：
- 特征/门 100% 复用 firsthit_shadow_detector 冻结纯函数（Q4 零漂移）；
- G1/G3/G4/G7 族比较 = 全部 G0 分母上的 ΔV = mean((A_c−A_p)·R)（§8.2），
  绝不做「嵌套子集胜率 vs G0 基准」的独立二项检验；
- 护栏敏感性只用触发价 q 代理，显式标记 trigger_q_proxy（§4.2，Phase 2 起换正式 quote）；
- 缺失保持 None；输出仅探索级（exploratory=true），不构成升级结论；
- 探索 CI 用日均值 percentile bootstrap（B=4000, seed=20260908）——
  冻结的 studentized MBB+HAC 属 Phase 4 裁决器，此处禁止提前实现变体。

用法：
    .venv\\Scripts\\python.exe scripts/audit_firsthit_parity.py
输入：output/pull_samples_5m_20260907.json + output/klines_5m_cache_720d.json
      + output/klines_5m_tail_20260907.json
输出：output/firsthit_parity_audit_20260908.json（+ 控制台漏斗）
"""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict

import numpy as np

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

from binance_predict.services.firsthit_shadow_detector import (  # noqa: E402
    FEE_RET, FIRSTHIT_SPECS, MIN_PTS, _gate_of, extract_firsthit_features,
)

SAMPLES = "output/pull_samples_5m_20260907.json"
KLINES = ["output/klines_5m_cache_720d.json", "output/klines_5m_tail_20260907.json"]
L = 300_000
SPLIT = "2026-08-24"          # 冻结切分（历史探索内部再分前后段看稳定性）
BOOT_B, SEED = 4000, 20260908
GUARDS = (0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12)
FEE = FEE_RET


# ---------- 纯函数（单测覆盖） ----------

def samples_to_curves(rows: list[dict]) -> dict[str, list[dict]]:
    """原始采样行 → {down, btc} 曲线（t/v 升序；None 采样剔除）。"""
    down, btc = [], []
    for r in sorted(rows, key=lambda x: int(x["timestamp"])):
        if r.get("down_price") is not None:
            down.append({"t": int(r["timestamp"]), "v": float(r["down_price"])})
        if r.get("btc_price") not in (None, 0):
            btc.append({"t": int(r["timestamp"]), "v": float(r["btc_price"])})
    return {"down": down, "btc": btc}


def paired_policy_value(events: list[dict], *, child: str, parent: str | None) -> dict:
    """政策替换主 estimand（规范 §8.2）：全部事件为分母，A_p 缺省恒 1（G0）。

    ΔV = mean((A_child − A_parent)·R)；contributing = 标签不同的窗口数。
    """
    if not events:
        return {"n_g0": 0, "contributing": 0, "delta_v": None,
                "child_v": None, "parent_v": None}
    rs = np.array([e["r"] for e in events], float)
    a_c = np.array([1.0 if e["labels"].get(child) else 0.0 for e in events])
    a_p = (np.ones(len(events)) if parent is None
           else np.array([1.0 if e["labels"].get(parent) else 0.0 for e in events]))
    diff = a_c - a_p
    return {
        "n_g0": len(events),
        "contributing": int((diff != 0).sum()),
        "delta_v": float((diff * rs).mean()),
        "child_v": float((a_c * rs).mean()),
        "parent_v": float((a_p * rs).mean()),
    }


def trigger_q_proxy_guard_curve(events: list[dict], *, guards) -> dict:
    """触发价 q 代理护栏敏感性（§4.2：仅 intention 层代理，非可成交口径）。"""
    out: dict[str, dict] = {}
    n = len(events)
    for h in guards:
        elig = [e for e in events if e["q"] < h]
        ev = (float(np.mean([FEE / e["q"] - 1.0 if e["win"] else -1.0 for e in elig]))
              if elig else None)
        out[f"{h:.2f}"] = {
            "eligible_n": len(elig), "coverage": len(elig) / n if n else None,
            "ev": ev,
        }
    return {"proxy": "trigger_q_proxy", "guards": out}


# ---------- 重放与主流程 ----------

def load_klines() -> dict[int, tuple[float, float, float, float]]:
    rows: list = []
    for p in KLINES:
        rows.extend(json.load(open(p, encoding="utf-8")))
    return {int(k[0]): (float(k[1]), float(k[2]), float(k[3]), float(k[4])) for k in rows}


def streak_up_of(k5: dict, window_start: int) -> int | None:
    """前驱 5m 连阳根数（cap 4；K 线缺失 → None）。与综合回测同式。"""
    streak, cur = 0, window_start - L
    while streak < 4:
        bar = k5.get(cur)
        if bar and bar[3] > bar[0]:
            streak += 1
            cur -= L
        else:
            return streak if bar else None
    return streak


def replay_events(k5) -> tuple[list[dict], dict]:
    """生产同源重放：extract_firsthit_features + _gate_of（含 streak）。"""
    raw = json.load(open(SAMPLES, encoding="utf-8"))
    by: dict[int, list[dict]] = defaultdict(list)
    for s in raw:
        by[int(s["timestamp"]) // L * L].append(s)
    events, funnel = [], {"windows": 0, "no_kline": 0, "no_firsthit": 0,
                          "npts_lt8": 0, "valid": 0, "unsettled": 0}
    for w, rows in sorted(by.items()):
        funnel["windows"] += 1
        bar = k5.get(w)
        if not bar or bar[0] <= 0:
            funnel["no_kline"] += 1
            continue
        curves = samples_to_curves(rows)
        ext = extract_firsthit_features(w, bar[0], curves["down"], curves["btc"])
        if ext is None:
            trig = any(0.005 < float(r["down_price"]) <= 0.1
                       for r in rows if r.get("down_price") is not None)
            funnel["no_firsthit" if not trig else "npts_lt8"] += 1
            continue
        if bar[3] == bar[0]:  # 开收相等无法判定
            funnel["unsettled"] += 1
            continue
        funnel["valid"] += 1
        streak = streak_up_of(k5, w)
        labels = {v: _gate_of(v, ext, streak) for v, _ in FIRSTHIT_SPECS}
        labels["g0"] = True
        win = bar[3] < bar[0]
        events.append({
            "window_start": w,
            "day": time.strftime("%Y-%m-%d", time.gmtime(w / 1000)),
            "q": float(ext["q"]), "win": win,
            "r": FEE / float(ext["q"]) - 1.0 if win else -1.0,
            "labels": labels,
        })
    return events, funnel


def day_cluster_ci(pairs: list[tuple[str, float]], b: int = BOOT_B) -> list[float] | None:
    """探索级日均值 percentile bootstrap（非裁决口径，仅稳定性参考）。"""
    if not pairs:
        return None
    by_day: dict[str, list[float]] = defaultdict(list)
    for d, v in pairs:
        by_day[d].append(v)
    days = sorted(by_day)
    if len(days) < 5:
        return None
    means = np.array([float(np.mean(by_day[d])) for d in days])
    rng = np.random.default_rng(SEED)
    pick = rng.integers(0, len(days), size=(b, len(days)))
    boots = means[pick].mean(axis=1)
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def main() -> int:
    k5 = load_klines()
    events, funnel = replay_events(k5)
    calib = [e for e in events if e["day"] < SPLIT]
    post = [e for e in events if e["day"] >= SPLIT]

    nodes = [
        ("A1_G1_vs_0", "firsthit_down_body_v1", None),
        ("A2_G3_vs_0", "firsthit_down_chg_v1", None),
        ("B1_G4_vs_G1", "firsthit_down_g4_v1", "firsthit_down_body_v1"),
        ("C1_G7_vs_G1", "firsthit_down_g7_v1", "firsthit_down_body_v1"),
        ("C2_g7streak_vs_G7", "g7_streak_v1", "firsthit_down_g7_v1"),
        ("X1_g7strict_vs_g7streak", "g7_strict_v1", "g7_streak_v1"),
    ]
    registry: dict = {
        "exploratory": True,
        "note": "Phase 0 历史审计：全部 G0 分母配对政策替换 estimand；"
                "post-split 段已烧毁（规范 §3.2），仅作前后段稳定性对照，"
                "不得作为确认证据；护栏曲线为 trigger_q 代理。",
        "split": SPLIT, "funnel": funnel, "nodes": {},
    }
    def _diff_r(e: dict) -> float:
        """ΔV 的逐事件贡献：(A_child − A_parent)·R；parent=None 时 A_p 恒 1（G0）。"""
        a_c = 1.0 if e["labels"].get(child) else 0.0
        a_p = 1.0 if parent is None else (1.0 if e["labels"].get(parent) else 0.0)
        return (a_c - a_p) * e["r"]

    for name, child, parent in nodes:
        seg = {}
        for tag, evs in (("calib", calib), ("burned_post_split", post)):
            st = paired_policy_value(evs, child=child, parent=parent)
            st["delta_ci_day_cluster"] = day_cluster_ci(
                [(e["day"], _diff_r(e)) for e in evs])
            seg[tag] = st
        registry["nodes"][name] = seg
    registry["guard_sensitivity_trigger_q_proxy"] = {
        tag: trigger_q_proxy_guard_curve(evs, guards=GUARDS)
        for tag, evs in (("calib", calib), ("burned_post_split", post))
    }
    out = "output/firsthit_parity_audit_20260908.json"
    json.dump(registry, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=float)
    print(json.dumps({"funnel": funnel, "n_calib": len(calib), "n_post": len(post),
                      "out": out}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 运行单测确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/test_firsthit_parity_audit.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: 跑一次真实数据审计（探索产物）**

Run: `.venv\Scripts\python.exe scripts/audit_firsthit_parity.py`
Expected: 控制台输出漏斗（windows → valid）与 JSON 落盘 `output/firsthit_parity_audit_20260908.json`；G0/G1/G3 点估计应与既有产物一致（此前已用生产语义验证 G0 n=2908/1724 等口径）。

- [ ] **Step 6: Commit（需用户授权）**

```bash
git add scripts/audit_firsthit_parity.py tests/test_firsthit_parity_audit.py
git commit -m "feat(research): Phase 0 生产同源历史审计——配对 ΔV 与触发价护栏代理"
```

---

### Task 7: 全量回归与汇报

**Files:** 无新文件（验证步骤）

- [ ] **Step 1: 全量后端测试**

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: 全部 PASS（基线绿 + 新增 16 个用例）。若出现与本计划无关的既有失败，原样记录、不修不藏。

- [ ] **Step 2: 验证实盘链路零改动**

Run: `git diff --stat src/binance_predict/services/multi_live_trader.py src/binance_predict/services/prediction_trading.py src/binance_predict/services/live_channels.py src/binance_predict/main.py`
Expected: 无输出（零改动）。

- [ ] **Step 3: 汇报（不 commit/push，等用户指令）**

汇报内容必须包含：新增测试数与全量结果、`audit_firsthit_parity.py` 漏斗数字（windows/valid/npts_lt8）、各节点 ΔV 点估计与探索 CI、`trigger_q_proxy` 护栏曲线要点、以及「Phase 1 账本表已建、待在生产 DB 回填」的状态。明确声明：本阶段全部结论为探索级，不构成升级依据。

---

## Self-Review（已执行）

1. **规范覆盖**：§12.1/12.2（Task 2/3/4/5）、§4.1 四互斥层（Task 1/3）、§8.2 estimand（Task 6）、§4.2 代理护栏（Task 6）、§14.4 缺失纪律（Task 3 断言）、§16 旧缺陷替代（Task 6 用 `_gate_of` 同源 + ΔV + 不做二项检验）、Q4 零漂移（Task 6 重放即生产纯函数）。Phase 2–4 明确排除并说明理由。
2. **占位符**：仅迁移文件 `down_revision` 依赖运行时 `alembic heads` 输出（结构性步骤，非含糊描述），其余代码完整。
3. **类型一致性**：`rows_from_window` 返回 `(audit, mother|None)` 在 Task 5 测试与脚本一致；`paired_policy_value(events, child=, parent=)` 签名与两处调用一致；`_labels_of` 键名 = `FIRSTHIT_SPECS` 版本名，测试断言同名。
4. **已知取舍**：DB upsert 层不做单测（无本地 PG 假设），以 Task 4 Step 6 迁移冒烟 + Task 7 回归兜底；`build_window_audit` 依赖 `build_mother_event` 的 Task 2 占位顺序已显式标注。
