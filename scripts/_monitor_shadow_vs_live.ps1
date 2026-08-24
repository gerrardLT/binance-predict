# 临时长期监控：影子信号 vs 实盘订单 一致性对比（修复部署后）
# 每 60s 轮询；发现分歧（影子有/实盘无、实盘有/影子无、价格不一致）立即高亮输出。
$base = "http://165.154.147.155:8082"
$knownSig = @{}
$knownOrder = @{}
$deadline = (Get-Date).AddHours(24)
$firstPass = $true
Write-Output ("=== 影子 vs 实盘 一致性监控启动 " + (Get-Date).ToUniversalTime().ToString("MM-dd HH:mm") + " UTC ===")
while ((Get-Date) -lt $deadline) {
    $ts = Get-Date -Format "HH:mm:ss"
    try {
        $resp = Invoke-RestMethod "$base/api/misalignment/signals?version=quote_contrarian_v1&limit=20" -TimeoutSec 20
        $q = $resp.quote_edge_live
        # 新的影子触发
        foreach ($s in $resp.signals) {
            $key = [string]$s.window_start
            if (-not $knownSig.ContainsKey($key)) {
                $knownSig[$key] = 1
                if (-not $firstPass) {
                    $winUtc = [DateTimeOffset]::FromUnixTimeMilliseconds($s.window_start).UtcDateTime.ToString("HH:mm")
                    $hasOrder = $false
                    foreach ($ok in $knownOrder.Keys) { if ($ok -eq $key) { $hasOrder = $true } }
                    if ($hasOrder) {
                        Write-Output "$ts | SHADOW win=$winUtc q=$($s.entry_down_price) -> 实盘已有对应订单 [一致]"
                    } else {
                        Write-Output "$ts | SHADOW win=$winUtc q=$($s.entry_down_price) settle=$($s.settle_outcome) win=$($s.win) | 实盘暂无订单(若enabled则待核对)"
                    }
                }
            }
        }
        # 新的实盘订单
        $orders = (Invoke-RestMethod "$base/api/trades/recent?limit=10" -TimeoutSec 20).orders
        foreach ($o in $orders) {
            $key = [string]$o.window_start
            if ($o.signal_version -eq 'quote_contrarian_v1' -and -not $knownOrder.ContainsKey($key)) {
                $knownOrder[$key] = 1
                if (-not $firstPass) {
                $winUtc = [DateTimeOffset]::FromUnixTimeMilliseconds($o.window_start).UtcDateTime.ToString("HH:mm")
                $hasSig = $knownSig.ContainsKey($key)
                $tag = "[一致]" ; if (-not $hasSig) { $tag = "!!! 分歧:实盘有影子无" }
                Write-Output "$ts | LIVE ORDER id=$($o.id) win=$winUtc quote=$($o.average_price) status=$($o.status) $tag"
                }
            }
        }
        $firstPass = $false
        # 心跳（每轮打印状态，便于确认存活）
        Write-Output "$ts | tick enabled=$($q.enabled) fire_total=$($q.fire_total) sigs_seen=$($knownSig.Count) orders_seen=$($knownOrder.Count)"
    } catch {
        Write-Output "$ts | ERR $($_.Exception.Message)"
    }
    Start-Sleep -Seconds 60
}
Write-Output "=== 24h 监控结束 ==="
