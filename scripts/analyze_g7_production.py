import os
import json
import requests

token = None
with open('.env', encoding='utf-8') as f:
    for line in f:
        if line.startswith('LOGIN_PASSWORD='):
            token = line.strip().split('=', 1)[1].strip('\"\'')
            break

headers = {'Authorization': f'Bearer {token}'}
base_url = 'http://165.154.147.155:8082'

# 1. Fetch recent trades
trades_resp = requests.get(f'{base_url}/api/trades/recent?limit=5000', headers=headers)
orders = trades_resp.json().get('orders', [])

# 2. Filter G7 family
g7_versions = {
    'firsthit_down_g7_v1',
    'g7_streak_v1',
    'g7_wick20_v1',
    'g7_strict_v1',
    'g7_q05_v1',
    'g7_t270_v1',
}
g7_orders = [o for o in orders if o.get('signal_version') in g7_versions]
print(f"Total G7 orders: {len(g7_orders)}")

# 3. For each unique window_start, let's query the market outcome and klines
windows = sorted(list(set(o['window_start'] for o in g7_orders)))
print(f"Unique windows: {len(windows)}")

# Query sentiment windows or btc klines
# Let's inspect each window's final settlement outcome
results = []
for o in sorted(g7_orders, key=lambda x: x['id']):
    ws = o['window_start']
    # Query klines for this 5m window: [ws, ws + 300_000]
    kline_resp = requests.get(f'{base_url}/api/chart/btc-klines?start_ms={ws - 60000}&end_ms={ws + 360000}&interval=5m', headers=headers)
    klines = kline_resp.json() if kline_resp.status_code == 200 else []
    
    # Try to find the exact 5m kline
    k_match = None
    if isinstance(klines, list):
        for k in klines:
            if abs(k.get('open_time', 0) - ws) < 30000:
                k_match = k
                break
    
    results.append({
        'id': o['id'],
        'created_at': o['created_at'],
        'signal': o['signal_version'],
        'window_start': ws,
        'status': o['status'],
        'price_kind': o.get('price_kind'),
        'avg_price': o.get('average_price'),
        'win': o.get('win'),
        'pnl': o.get('pnl'),
        'settle_outcome': o.get('settle_outcome'),
        'error_message': o.get('error_message'),
        'kline': k_match,
        'quote_json': o.get('quote_json'),
    })

with open('.pytest_tmp/g7_deep_analysis.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Saved deep analysis data to .pytest_tmp/g7_deep_analysis.json")
