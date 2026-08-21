"""只读拉取：线上场景信号（FakeBreakout）+ 三个影子检测器实盘数据，落盘 output/。"""
import json
import time
import urllib.request

BASE = "http://165.154.147.155:8082"


def get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=60) as r:
        return json.loads(r.read().decode())


health = get("/api/health")
print("health:", json.dumps(health, ensure_ascii=False)[:300])

# ---- 线上场景信号 ----
fb_stats = get("/api/fake-breakout/stats")
fb_sigs = get("/api/fake-breakout/signals?limit=200")
print(f"fake-breakout signals: {fb_sigs['total']}")

# ---- 影子检测器（misalignment_signals 表，version 区分）----
shadow = {}
for v in ("x4_v1", "quote_momentum_v1", "quote_contrarian_v1"):
    d = get(f"/api/misalignment/signals?limit=200&version={v}")
    shadow[v] = {"stats": d["stats"], "signals": d["signals"]}
    print(f"shadow {v}: settled={d['stats']['settled']} list={len(d['signals'])}")

# 全版本列表（不带 version 过滤，看总量与最近明细）
shadow_all = get("/api/misalignment/signals?limit=200")
print(f"shadow all-version list: {shadow_all['total']}")

stamp = time.strftime("%Y%m%d_%H%M")
with open(f"output/online_signals_now_{stamp}.json", "w", encoding="utf-8") as f:
    json.dump({"fetched_at": stamp, "stats": fb_stats, "signals": fb_sigs["signals"]},
              f, ensure_ascii=False, indent=1)
with open(f"output/online_shadow_now_{stamp}.json", "w", encoding="utf-8") as f:
    json.dump({"fetched_at": stamp, "by_version": shadow,
               "all_version_list": shadow_all["signals"],
               "detector": shadow_all.get("detector"),
               "quote_edge_detector": shadow_all.get("quote_edge_detector")},
              f, ensure_ascii=False, indent=1)
# 兼容旧脚本文件名
with open("output/online_signals_now.json", "w", encoding="utf-8") as f:
    json.dump({"fetched_at": stamp, "stats": fb_stats, "signals": fb_sigs["signals"]},
              f, ensure_ascii=False, indent=1)
print("saved:", stamp)
