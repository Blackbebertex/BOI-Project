import json, urllib.request, sys
payload = {
    "account_id": "test123",
    "features": {
        "F115": 0.1, "F321": 0.2, "F527": 0.3, "F531": 0.4, "F670": 0.5,
        "F1692": 0.6, "F2082": 0.7, "F2122": 0.8, "F2582": 0.9, "F2678": 1.0,
        "F2737": 1.1, "F2956": 1.2, "F3043": 1.3, "F3836": 1.4, "F3887": 1.5,
        "F3889": 1.6, "F3891": 1.7, "F3894": 1.8
    }
}
url = "http://127.0.0.1:8000/score"
req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req) as resp:
    sys.stdout.write(resp.read().decode('utf-8'))
