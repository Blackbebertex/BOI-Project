import requests


REQUIRED = [
    "F115",
    "F321",
    "F527",
    "F531",
    "F670",
    "F1692",
    "F2082",
    "F2122",
    "F2582",
    "F2678",
    "F2737",
    "F2956",
    "F3043",
    "F3836",
    "F3887",
    "F3889",
    "F3891",
    "F3894",
]


def main() -> None:
    accounts = []
    for i in range(150):
        features = {f: round(0.1 * (i % 10 + 1), 2) for f in REQUIRED}
        accounts.append({"account_id": f"ACC-TEST-{i}", "features": features})

    resp = requests.post(
        "http://127.0.0.1:8000/score/batch",
        json={"accounts": accounts},
        headers={"X-API-Key": "demo-investigator-key"},
        timeout=30,
    )
    data = resp.json()
    print("Status     :", resp.status_code)
    print("Total      :", data.get("total"))
    print("Blocked    :", data.get("blocked"))
    print("Approved   :", data.get("approved"))
    print("Latency ms :", data.get("batch_latency_ms"))

    if resp.status_code == 200 and data.get("total") == 150:
        print("\nBATCH VECTORIZED PATH (150 rows): PASS")
    else:
        print("\nFAIL:", data)


if __name__ == "__main__":
    main()
