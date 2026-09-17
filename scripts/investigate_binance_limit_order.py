import urllib.request
import urllib.parse
import json
import hmac
import hashlib
import time

api_key = None
api_secret = None

with open('.env', 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if line.startswith('BINANCE_API_KEY='):
            api_key = line.strip().split('=', 1)[1].strip('\"\'')
        elif line.startswith('BINANCE_API_SECRET='):
            api_secret = line.strip().split('=', 1)[1].strip('\"\'')

def sign_query(params: dict) -> str:
    params['timestamp'] = int(time.time() * 1000)
    query_str = urllib.parse.urlencode(params)
    signature = hmac.new(
        api_secret.encode('utf-8'),
        query_str.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return f"{query_str}&signature={signature}"

# 1. Fetch wallet info
query = sign_query({})
url = f"https://api.binance.com/sapi/v1/w3w/wallet/prediction/wallet/list?{query}"
req = urllib.request.Request(url, headers={"X-MBX-APIKEY": api_key})
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        wallets_data = json.loads(resp.read().decode())
        print("Wallets data:", wallets_data)
        wallet = wallets_data['wallets'][0]
        wallet_address = wallet['walletAddress']
        wallet_id = wallet['walletId']
except Exception as e:
    print("Error fetching wallet:", e)
    wallet_address, wallet_id = None, None

# 2. Test get-quote with different orderType!
# Can orderType be "LIMIT"? Does get-quote or place-order-bundle accept "LIMIT"?
print("\nTesting get-quote with orderType=LIMIT...")
if wallet_address:
    # Get a valid market first to find a token
    q_market = sign_query({"limit": 5, "offset": 0})
    u_market = f"https://api.binance.com/sapi/v1/w3w/wallet/prediction/market/list?{q_market}"
    req_m = urllib.request.Request(u_market, headers={"X-MBX-APIKEY": api_key})
    with urllib.request.urlopen(req_m, timeout=10) as resp:
        markets = json.loads(resp.read().decode())
        print(f"Found {len(markets.get('list', []))} markets")
        first_m = markets['list'][0]
        print(f"First market: title={first_m.get('title')}, symbol={first_m.get('symbol')}")
        tokens = first_m.get('tokens', [])
        up_token = tokens[0]['tokenId'] if tokens else None
        print("Sample token ID:", up_token)

    # Test get-quote with orderType="LIMIT"
    if up_token:
        for ot in ["LIMIT", "MARKET"]:
            params = {
                "walletAddress": wallet_address,
                "tokenId": up_token,
                "side": "BUY",
                "amountIn": str(int(2.0 * 10**18)),
                "orderType": ot,
                "slippageBps": 1200
            }
            if ot == "LIMIT":
                params["price"] = "0.30"
                params["limitPrice"] = "0.30"
            q_str = sign_query(params)
            u_quote = f"https://api.binance.com/sapi/v1/w3w/wallet/prediction/trade/get-quote?{q_str}"
            req_q = urllib.request.Request(u_quote, data=b"", headers={"X-MBX-APIKEY": api_key})
            try:
                with urllib.request.urlopen(req_q, timeout=10) as resp:
                    print(f"get-quote with orderType={ot} SUCCESS:", resp.read().decode()[:200])
            except urllib.error.HTTPError as e:
                print(f"get-quote with orderType={ot} FAILED HTTP {e.code}:", e.read().decode()[:300])
            except Exception as e:
                print(f"get-quote with orderType={ot} ERROR:", e)
