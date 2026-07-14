#!/usr/bin/env python3
"""
Agent 运行健康诊断 CLI

两种取数方式：
- 默认（HTTP）：调用运行中 Web 进程的 GET /api/agent/health，可拿到完整 5 类
  指标（含调度器心跳/LLM 错误率等内存态）。适合线上进程存活时使用。
- --db-only（兜底）：直连数据库自行聚合，仅产出 DB 层指标（窗口连续性/匹配率/
  校准），不含内存态。适合 Web 进程不可用、或在无 HTTP 环境下快速体检。

用法示例：
    python scripts/agent_health.py                 # HTTP，默认 localhost:8000
    python scripts/agent_health.py --json          # 输出原始 JSON
    python scripts/agent_health.py --url http://1.2.3.4:8000 --token XXX
    python scripts/agent_health.py --db-only        # 直连 DB 兜底

退出码：OK=0，WARN=1，CRITICAL=2，执行异常=3（便于接入外部巡检/告警）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

# 允许以 `python scripts/agent_health.py` 直接运行（补齐 src 到 import 路径）
import os

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

_EXIT_BY_STATUS = {"OK": 0, "WARN": 1, "CRITICAL": 2}


async def _fetch_http(url: str, token: str, timeout: float) -> dict:
    """调用运行中进程的 /api/agent/health 端点。"""
    import httpx

    endpoint = url.rstrip("/") + "/api/agent/health"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(endpoint, headers=headers)
        resp.raise_for_status()
        return resp.json()


async def _build_db_only() -> dict:
    """直连数据库自行聚合（不含内存态指标）。"""
    from binance_predict.db.engine import async_session_factory
    from binance_predict.services.health import health_service

    async with async_session_factory() as db:
        report = await health_service.build_report(db)  # 三个内存态入参留空
        return report.model_dump(mode="json")


def _print_human(report: dict) -> None:
    """人类可读输出。"""
    status = report.get("overall_status", "?")
    mark = {"OK": "✅", "WARN": "⚠️", "CRITICAL": "❌"}.get(status, "❓")
    print(f"{mark}  总体状态: {status}")
    print(f"生成时间: {report.get('generated_at')}")
    print(f"\n诊断: {report.get('summary', '')}")

    alerts = report.get("alerts", [])
    if alerts:
        print(f"\n告警（{len(alerts)}）:")
        for a in alerts:
            print(f"  [{a.get('level')}] {a.get('code')}: {a.get('message')}")
    else:
        print("\n告警: 无")

    wc = report.get("window_continuity", {})
    ps = report.get("predict_stats", {})
    print("\n窗口连续性:", json.dumps(wc, ensure_ascii=False))
    print("predict 统计:", json.dumps(ps, ensure_ascii=False))

    calib = report.get("calibration", [])
    if calib:
        print("置信度校准:")
        for b in calib:
            print(
                f"  {b.get('range')} | n={b.get('count')} | "
                f"avg_conf={b.get('avg_confidence')} | hit={b.get('hit_rate')} | gap={b.get('gap')}"
            )
    else:
        print("置信度校准: 样本不足，暂无分桶")

    sched = report.get("scheduler", {})
    llm = report.get("llm", {})
    if sched.get("phase_ages_s") or sched.get("queue_depth") is not None:
        print("调度器:", json.dumps(sched, ensure_ascii=False))
    if llm:
        print("LLM:", json.dumps(
            {k: llm.get(k) for k in ("call_count", "total_cost", "consecutive_failures") if k in llm},
            ensure_ascii=False,
        ))


async def _run(args: argparse.Namespace) -> int:
    try:
        if args.db_only:
            report = await _build_db_only()
        else:
            from binance_predict.config.settings import settings

            url = args.url or f"http://localhost:{settings.api_port}"
            token = args.token if args.token is not None else settings.api_auth_token
            report = await _fetch_http(url, token, args.timeout)
    except Exception as exc:  # noqa: BLE001 —— CLI 顶层统一兜底
        print(f"[ERROR] 获取健康报告失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        if not args.db_only:
            print("        可尝试 --db-only 直连数据库兜底。", file=sys.stderr)
        return 3

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_human(report)

    return _EXIT_BY_STATUS.get(report.get("overall_status"), 3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent 运行健康诊断 CLI")
    parser.add_argument("--url", default=None, help="Web 进程基址，默认 http://localhost:<api_port>")
    parser.add_argument("--token", default=None, help="Bearer Token，默认取 settings.api_auth_token")
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP 超时秒数（默认 10）")
    parser.add_argument("--db-only", action="store_true", help="跳过 HTTP，直连数据库兜底聚合")
    parser.add_argument("--json", action="store_true", help="输出原始 JSON")
    args = parser.parse_args()

    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
