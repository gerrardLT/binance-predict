"""时段胜率分析数据拉取（只读）：线上影子全版本 + 场景信号 + analytics 聚合。

GitHub Actions 全量导出（export-signals-readonly.yml）完成前的降级路径，
也是之后的快速刷新通道。鉴权：.env 的 LOGIN_PASSWORD（全局中间件要求
所有 /api/* 携带 Bearer）。已知限制：列表接口 limit≤200 硬上限
（main.py:1393/1321），quote_momentum_v1 等 511 条版本会截到最近 200 条
——按 window_start 倒序截断，不引入时段偏差；快照头部记录
stats.settled（全量数）与列表长度，供分析脚本标注截断。
用法：.venv\\Scripts\\python.exe -X utf8 scripts/_pull_timeslice_data.py
"""
import json
import os
import time
import urllib.error
import urllib.request

BASE = "http://165.154.147.155:8082"

# 影子版本全集（对齐 main.py /api/signals/analytics 的 VERSIONS 口径）
SHADOW_VERSIONS = [
    "x4_v1", "x4_v2",
    "quote_momentum_v1", "quote_momentum_v2",
    "quote_contrarian_v1", "quote_contrarian_v2",
    "quote_contrarian_v3a", "quote_contrarian_v3b",
    "late_night_contrarian_v1", "late_night_contrarian_v2",
]


def read_login_password() -> str | None:
    """从本地 .env 读 LOGIN_PASSWORD（仅本行，不加载整个文件）。"""
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.exists(env_path):
        return None
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("LOGIN_PASSWORD="):
                return line.split("=", 1)[1].strip()
    return None


TOKEN = read_login_password()


def get(path: str) -> dict:
    req = urllib.request.Request(f"{BASE}{path}")
    if TOKEN:
        req.add_header("Authorization", f"Bearer {TOKEN}")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


health = get("/api/health")
print("health:", json.dumps(health, ensure_ascii=False)[:200])

# ---- 影子信号（misalignment_signals，逐版本；stats 为全量累计不随 limit 截断）----
shadow: dict[str, dict] = {}
for v in SHADOW_VERSIONS:
    d = get(f"/api/misalignment/signals?limit=200&version={v}")
    # stats 为该版本全量累计（带 version 查询时不随 limit 截断）
    shadow[v] = {"stats": d["stats"], "signals": d["signals"]}
    print(f"shadow {v}: stats.settled={d['stats']['settled']} list={len(d['signals'])}"
          + ("  [截断]" if d["stats"]["settled"] and d["stats"]["settled"] > len(d["signals"]) else ""))

# ---- 场景信号（fake_breakout_signals，<200 条可全量）----
fb = get("/api/fake-breakout/signals?limit=200")
print(f"fake-breakout signals: {fb['total']}")

# ---- analytics 全量聚合（对账基准，无截断）----
try:
    analytics = get("/api/signals/analytics")
    print("analytics: ok")
except urllib.error.HTTPError as exc:
    analytics = {"error": f"HTTP {exc.code}"}
    print("analytics failed:", analytics["error"])

stamp = time.strftime("%Y%m%d_%H%M")
payload = {
    "fetched_at": stamp,
    "by_version": shadow,
    "fake_breakout": {"signals": fb["signals"], "total": fb["total"]},
    "analytics": analytics,
}
out_path = f"output/timeslice_snapshot_{stamp}.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False)
print("saved:", out_path)
