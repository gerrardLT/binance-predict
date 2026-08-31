"""Part2：Predexon BTC UP/DOWN 历史订单簿回填（可断点续传）。

前置：output/predexon_orderbooks/markets_map_{period}.json（先跑 .pytest_tmp/predexon_map.py）。
对每个市场拉取窗口 [epoch, epoch+period) 内订单簿快照（分页 limit=200），
写入按月分片 output/predexon_orderbooks/{period}_YYYYMM.jsonl.gz，
每行紧凑格式 {"slug","market_id","start_time","n_snap","snapshots":[[ts,bid,ask],...]}
（--full 时每快照追加完整 bids/asks 两元素）。
断点：_done.jsonl（追加式已完成 market_id）+ _progress.json（状态快照）。
失败：_failed.jsonl（追加，重跑自动重试）。
清单：每次运行结束对涉及分片写 _manifest.jsonl 一行（整文件 md5 + 本次行数增量）。
纪律：仅公开 GET、每 key 1 rps（1.05s 间隔，key 池线程池并发，IP 不参与限流）、
429/5xx 指数退避重试≤3、空快照率>30% 警告。

用法（PowerShell，cwd=仓库根）：
  .venv\Scripts\python.exe -X utf8 scripts/predexon_backfill.py --period 5m `
      --start 2026-07-01 --end 2026-09-01     # 优先批：与线上信号运行期重叠
  .venv\Scripts\python.exe -X utf8 scripts/predexon_backfill.py --period 5m `
      --start 2026-03-11 --end 2026-07-01     # 第二批：向前补
"""
import argparse
import datetime as dtm
import gzip
import hashlib
import json
import os
import subprocess
import threading
import time
import urllib.parse

BASE = "https://api.predexon.com"
OUT_DIR = os.path.join("output", "predexon_orderbooks")
N_REQ = [0]
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

env = {}
for line in open(".env", encoding="utf-8"):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
KEYS = [k.strip() for k in env.get("PREDEXON_API_KEY", "").split(",") if k.strip()]
if not KEYS:
    raise SystemExit("缺少 PREDEXON_API_KEY（.env，多 key 逗号分隔）")
# 归因实测（2026-08-31）：限流按 key 计（同 key 1s 内 2 发→429；不同 key 各自
# 连发→200），IP 不参与 → 多 key 单 IP 直连并发，无需代理。


def get(path, params, key=None, timeout=40):
    """公开 GET，curl 执行（urllib 被 Cloudflare 1010 拦），所属 key 1 rps，退避重试≤3。"""
    key = key or KEYS[0]
    url = BASE + path + ("?" + urllib.parse.urlencode(params) if params else "")
    for attempt in range(3):
        p = subprocess.run(
            ["curl.exe", "-s", "-A", UA, "-H", f"x-api-key: {key}",
             "-w", "\n%{http_code}", "--max-time", str(timeout), url],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        parts = p.stdout.rsplit("\n", 1)
        code = int(parts[-1].strip() or 0) if len(parts) > 1 else 0
        body = parts[0] if len(parts) > 1 else ""
        time.sleep(1.05)  # 免费档 1 rps
        N_REQ[0] += 1
        if code == 200:
            return json.loads(body)
        if code == 0 or code in (429, 500, 502, 503):
            # 500=上游冷查失败（重试必中缓存，短退避）；429=限速（指数退避）
            time.sleep(0.5 if code == 500 else 2 ** (attempt + 1))
            continue
        raise RuntimeError(f"HTTP {code}: {body[:200]}")
    raise RuntimeError(f"GET {path} 重试耗尽 last_code={code}")


def fetch_snapshots(market_id, start_ms, end_ms, full=False, key=None):
    """拉取单市场时间范围内全部快照（分页），返回紧凑快照列表。"""
    snaps, cursor = [], None
    while True:
        params = {"market_id": market_id, "start_time": start_ms,
                  "end_time": end_ms, "limit": 200}
        if cursor:
            params["pagination_key"] = cursor
        data = get("/v2/predictfun/orderbooks", params, key=key)
        for s in (data.get("snapshots") or []):
            ts = s.get("timestamp")
            if ts is None:
                continue
            row = [int(ts), s.get("best_bid"), s.get("best_ask")]
            if full:
                row += [s.get("bids"), s.get("asks")]
            snaps.append(row)
        pg = data.get("pagination", {})
        cursor = pg.get("pagination_key")
        if not pg.get("has_more") or not cursor or len(snaps) > 50_000:
            break
    return snaps


def slug_epoch(slug):
    try:
        return int(slug.rsplit("-", 1)[-1])
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="5m", choices=("5m", "15m"))
    ap.add_argument("--start", default="2026-07-01", help="批次起点（含）")
    ap.add_argument("--end", default=None, help="批次终点（不含），默认=现在")
    ap.add_argument("--full", action="store_true", help="保留完整 bids/asks（体积大）")
    ap.add_argument("--t-start", type=int, default=38,
                    help="快照起点（相对窗口开始秒）。默认 38 覆盖 L1 触发窗 [45,60)s")
    ap.add_argument("--t-end", type=int, default=68,
                    help="快照终点（相对窗口开始秒）。默认 68；全窗用 300")
    ap.add_argument("--limit-market", type=int, default=0, help="调试：只处理前 N 个")
    ap.add_argument("--run-tag", default=time.strftime("%m%d%H%M"),
                    help="本轮分片后缀 tag（防 gzip 追加损坏）")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    map_path = os.path.join(OUT_DIR, f"markets_map_{args.period}.json")
    if not os.path.exists(map_path):
        raise SystemExit(f"缺少 {map_path}，先跑 predexon_map.py")
    hits = json.load(open(map_path, encoding="utf-8")).get("hits", {})
    if not hits:
        raise SystemExit("映射 hits 为空")

    s0 = int(dtm.datetime.strptime(args.start, "%Y-%m-%d")
             .replace(tzinfo=dtm.timezone.utc).timestamp())
    e1 = (int(dtm.datetime.strptime(args.end, "%Y-%m-%d")
              .replace(tzinfo=dtm.timezone.utc).timestamp())
          if args.end else int(time.time()))

    done = set()
    done_path = os.path.join(OUT_DIR, "_done.jsonl")
    if os.path.exists(done_path):
        with open(done_path, encoding="utf-8") as fh:
            for ln in fh:
                try:
                    done.add(json.loads(ln)["market_id"])
                except Exception:
                    continue

    todo = []
    for slug, info in hits.items():
        ep = slug_epoch(slug)
        mid = info.get("market_id")
        if ep is None or not (s0 <= ep < e1) or not mid or mid in done:
            continue
        todo.append((ep, slug, mid))
    todo.sort(reverse=True)  # 新→旧：优先覆盖与线上运行期重叠段
    if args.limit_market:
        todo = todo[:args.limit_market]
    if not todo:
        print("本批次无可回填市场（全部已完成或范围外）")
        return
    n_workers = max(1, min(len(KEYS), 16))
    est_min = len(todo) * 1.05 / 60 / n_workers
    print(f"待回填 {len(todo)} 市场（{args.start} ~ {args.end or 'now'}），"
          f"已完成 {len(done)}，{n_workers} key 并发，预计 ~{est_min:.0f} 分钟")

    failed_path = os.path.join(OUT_DIR, "_failed.jsonl")
    empty_path = os.path.join(OUT_DIR, "_empty.jsonl")
    prog_path = os.path.join(OUT_DIR, "_progress.json")
    handles, rows_per_ym, t0 = {}, {}, time.time()
    shared = {"n_empty": 0, "done": 0}
    io_lock = threading.Lock()

    def write_progress(processed):
        prog = {"period": args.period, "batch": [args.start, args.end or "now"],
                "todo": len(todo), "processed": processed,
                "rows_total": sum(rows_per_ym.values()), "rows_per_ym": rows_per_ym,
                "n_empty": shared["n_empty"], "n_requests": N_REQ[0],
                "elapsed_s": round(time.time() - t0, 1),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(prog_path, "wt", encoding="utf-8") as pf:
            json.dump(prog, pf, ensure_ascii=False, indent=1)

    def process_one(epoch, slug, mid, key):
        """单市场拉取+落盘；共享句柄与计数全部在 io_lock 内操作。"""
        # 只拉触发窗附近：密度实测 ~5 快照/s，42s ≈ 210 条 ≈ 1-2 页
        start_ms = epoch * 1000 + args.t_start * 1000
        end_ms = epoch * 1000 + args.t_end * 1000
        ym = dtm.datetime.fromtimestamp(epoch, dtm.timezone.utc).strftime("%Y%m")
        try:
            snaps = fetch_snapshots(mid, start_ms, end_ms, full=args.full, key=key)
        except Exception as ex:
            with io_lock:
                with open(failed_path, "at", encoding="utf-8") as ff:
                    ff.write(json.dumps({"market_id": mid, "slug": slug,
                                         "error": str(ex)}) + "\n")
            print(f"  {slug} 失败: {ex}")
            return
        with io_lock:
            if ym not in handles:
                # 每 run 独立分片（防进程被杀后 gzip 追加损坏）；Part3 读取时按 market_id 去重
                handles[ym] = gzip.open(
                    os.path.join(OUT_DIR,
                                 f"{args.period}_{ym}.r{args.run_tag}.jsonl.gz"),
                    "wt", encoding="utf-8")
            handles[ym].write(json.dumps(
                {"slug": slug, "market_id": mid, "start_time": epoch * 1000,
                 "t_range": [args.t_start, args.t_end],
                 "n_snap": len(snaps), "snapshots": snaps},
                ensure_ascii=False, separators=(",", ":")) + "\n")
            handles[ym].flush()
            rows_per_ym[ym] = rows_per_ym.get(ym, 0) + 1
            if not snaps:
                shared["n_empty"] += 1
            # 空快照不记 done：未来窗/归档延迟(<~1h)会在后续重跑中补齐
            if snaps:
                with open(done_path, "at", encoding="utf-8") as df:
                    df.write(json.dumps({"market_id": mid, "slug": slug}) + "\n")
            else:
                with open(empty_path, "at", encoding="utf-8") as ef:
                    ef.write(json.dumps({"market_id": mid, "slug": slug}) + "\n")
            shared["done"] += 1
            i = shared["done"]
            if i % 50 == 0 or i == len(todo):
                rate = (time.time() - t0) / i
                eta = rate * (len(todo) - i)
                print(f"  [{i}/{len(todo)}] 空 {shared['n_empty']} 请求 {N_REQ[0]} "
                      f"ETA {eta / 60:.0f}min（{time.strftime('%H:%M:%S')}）")
                write_progress(i)
                if shared["n_empty"] > max(30, i * 0.3):
                    print("⚠ 空快照率过高，先检查 market_id/时间参数是否正确再继续")

    def worker(ep_list, key):
        for epoch, slug, mid in ep_list:
            process_one(epoch, slug, mid, key)

    try:
        if n_workers == 1:
            worker(todo, KEYS[0])
        else:
            threads = [threading.Thread(
                            target=worker,
                            args=(todo[w::n_workers], KEYS[w % len(KEYS)]),
                            daemon=True)
                       for w in range(n_workers)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
    finally:
        with io_lock:
            for fh in handles.values():
                fh.close()
            write_progress(shared["done"])

    # 分片清单：整文件 md5 + 本次行数增量
    man_path = os.path.join(OUT_DIR, "_manifest.jsonl")
    with open(man_path, "at", encoding="utf-8") as mf:
        for ym in rows_per_ym:
            path = os.path.join(OUT_DIR,
                                f"{args.period}_{ym}.r{args.run_tag}.jsonl.gz")
            h = hashlib.md5()
            with open(path, "rb") as bf:
                for blk in iter(lambda: bf.read(1 << 20), b""):
                    h.update(blk)
            mf.write(json.dumps(
                {"file": os.path.basename(path), "rows_this_run": rows_per_ym[ym],
                 "md5": h.hexdigest(),
                 "run": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "batch": [args.start, args.end or "now"]}) + "\n")
    print(f"\n完成：{sum(rows_per_ym.values())} 行，失败 {len(todo) - sum(rows_per_ym.values())}，"
          f"空快照 {shared['n_empty']}，分片 {sorted(rows_per_ym)}，清单已写 {man_path}")


if __name__ == "__main__":
    main()
